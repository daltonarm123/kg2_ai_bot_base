# -*- coding: utf-8 -*-
"""
KG2 AI Bot - Fully Automated, Self-Learning, Hardened
-----------------------------------------------------
- Zero-config autopilot: explores, spies, raids, builds, trains, researches
- Self-tuning brain with on-disk persistence (risk down over time after losses)
- Human behavior: nightly sleep + random short idle breaks + jittered pacing
- Safety stops: pause aggression if food/gold ≤ 0; recovery plan to stabilize
- Guardrails: attack cooldown (48h), honor-safe targeting, rate limit, kill switch
- Observability: rotating logs + periodic snapshots + --status report

Mock server simulates build/train/attack/spy/explore/tax/research & economy.
Flip to live by setting KG2_MOCK=false and wiring endpoints (paths already stubbed).

Requirements:
  httpx, pydantic, tenacity, python-dotenv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional, Literal
from datetime import datetime, timedelta
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, RetryError

# ------------------------
# Config & Logging
# ------------------------

@dataclass
class Config:
    base_url: str
    account_id: str
    token: str
    kingdom_id: int
    world_id: int = 1
    origin_url: str = "https://www.kingdomgame.net"
    referer_buildings: str = "https://www.kingdomgame.net/buildings"
    referer_war: str = "https://www.kingdomgame.net/warroom"
    referer_research: str = "https://www.kingdomgame.net/research"
    referer_overview: str = "https://www.kingdomgame.net/overview"
    user_agent: str = "KG2-AI/0.4 (auto)"
    max_rps: float = 0.4
    timeout_s: float = 20
    mock_mode: bool = False
    enabled: bool = True
    enable_attacks: bool = True
    enable_research: bool = True  
    enable_exploration: bool = True
    logs_dir: str = "logs"
    snapshots_dir: str = "snapshots"
    snapshot_every_steps: int = 10

    @staticmethod
    def from_env() -> "Config":
        load_dotenv()
        return Config(
            base_url=os.getenv("KG2_BASE_URL", "https://www.kingdomgame.net"),
            account_id=os.getenv("KG2_ACCOUNT_ID", ""),
            token=os.getenv("KG2_TOKEN", ""),
            kingdom_id=int(os.getenv("KG2_KINGDOM_ID", "0")),
            world_id=int(os.getenv("KG2_WORLD_ID", "1")),
            origin_url=os.getenv("KG2_ORIGIN", "https://www.kingdomgame.net"),
            referer_buildings=os.getenv("KG2_REFERER_BUILDINGS", "https://www.kingdomgame.net/buildings"),
            referer_war=os.getenv("KG2_REFERER_WAR", "https://www.kingdomgame.net/warroom"),
            referer_research=os.getenv("KG2_REFERER_RESEARCH", "https://www.kingdomgame.net/research"),
            referer_overview=os.getenv("KG2_REFERER_OVERVIEW", "https://www.kingdomgame.net/overview"),
            user_agent=os.getenv("KG2_UA", "KG2-AI/0.4 (auto)"),
            max_rps=float(os.getenv("KG2_MAX_RPS", "0.4")),
            timeout_s=float(os.getenv("KG2_TIMEOUT_S", "20")),
            mock_mode=os.getenv("KG2_MOCK", "false").lower() == "true",
            enabled=os.getenv("KG2_ENABLED", "true").lower() == "true",
            enable_attacks=os.getenv("KG2_ENABLE_ATTACKS", "true").lower() == "true",
            enable_research=os.getenv("KG2_ENABLE_RESEARCH", "true").lower() == "true",
            enable_exploration=os.getenv("KG2_ENABLE_EXPLORATION", "true").lower() == "true",
            logs_dir=os.getenv("KG2_LOGS_DIR", "logs"),
            snapshots_dir=os.getenv("KG2_SNAP_DIR", "snapshots"),
            snapshot_every_steps=int(os.getenv("KG2_SNAP_EVERY", "10")),
        )

def setup_logging(logs_dir: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kg2")
    logger.setLevel(logging.INFO)
    # Rotate at ~2MB, keep 5 files
    fh = RotatingFileHandler(Path(logs_dir) / "bot.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

log = setup_logging(Config.from_env().logs_dir)

# ------------------------
# HTTP Client (live mode)
# ------------------------

class KG2Error(Exception): pass

class KG2Client:
    """Live client that connects to real KingdomGame.net APIs using working ASMX endpoints"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_req_time = 0.0
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
            },
            timeout=cfg.timeout_s,
        )
        
        # Caches for type mappings
        self._btypes_cache = None
        self._btypes_lock = asyncio.Lock()
        self._poptypes_cache = None  
        self._poptypes_lock = asyncio.Lock()
        self._restypes_cache = None
        self._restypes_lock = asyncio.Lock()

    async def aclose(self):
        await self._client.aclose()

    async def _respect_rate_limit(self):
        """Respect rate limiting to prevent server overload"""
        min_interval = 15.0  # 15 seconds minimum between requests
        dt = time.time() - self._last_req_time
        if dt < min_interval:
            await asyncio.sleep(min_interval - dt)
        self._last_req_time = time.time()

    def _unwrap_json(self, resp: httpx.Response) -> dict:
        """Parse ASP.NET JSON response that may be wrapped in 'd' field"""
        data = resp.json()
        if isinstance(data, dict) and "d" in data and isinstance(data["d"], str):
            try:
                data = json.loads(data["d"])
            except Exception:
                pass
        return data

    async def post_asmx(self, path: str, payload: Dict[str, Any], referer: str) -> Dict[str, Any]:
        """Unified ASMX request helper with proper headers and error handling"""
        await self._respect_rate_limit()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "world-id": str(self.cfg.world_id),
            "Origin": self.cfg.origin_url,
            "Referer": referer,
            "User-Agent": self.cfg.user_agent,
        }
        
        try:
            resp = await self._client.post(path, json=payload, headers=headers)
            
            # Enhanced error reporting for auth issues
            if resp.status_code == 403:
                snippet = resp.text[:400].replace("\n"," ")
                raise RuntimeError(f"AUTH 403 at {path}. Check KG2_ACCOUNT_ID/KG2_TOKEN/KG2_KINGDOM_ID and WAF. Resp~{snippet}")
            elif resp.status_code == 500:
                snippet = resp.text[:400].replace("\n"," ")
                raise RuntimeError(f"SERVER 500 at {path}. Likely bad token/account data. Resp~{snippet}")
            
            return self._unwrap_json(resp)
        except Exception as e:
            log.error(f"ASMX {path} exception: {e}")
            raise

    async def get_resources(self) -> dict:
        """Get live kingdom resources from GetKingdomResources API"""
        body = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
        }
        return await self.post_asmx(
            "/WebService/Kingdoms.asmx/GetKingdomResources",
            body,
            referer=self.cfg.referer_overview,
        )

    async def get_population(self) -> dict:
        """Get live kingdom population from GetKingdomPopulation API"""
        body = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
        }
        return await self.post_asmx(
            "/WebService/Kingdoms.asmx/GetKingdomPopulation",
            body,
            referer=self.cfg.referer_war,
        )

    async def get_buildings(self) -> dict:
        """Get live kingdom buildings from GetKingdomBuildings API"""
        body = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
        }
        return await self.post_asmx(
            "/WebService/Kingdoms.asmx/GetKingdomBuildings",
            body,
            referer=self.cfg.referer_buildings,
        )

    async def get_skills(self) -> dict:
        """Get live research skills from GetSkills API"""
        body = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
        }
        return await self.post_asmx(
            "/WebService/Research.asmx/GetSkills",
            body,
            referer=self.cfg.referer_research,
        )

    async def get_training_skills(self) -> dict:
        """Get live research queue from GetTrainingSkills API"""
        body = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
        }
        return await self.post_asmx(
            "/WebService/Research.asmx/GetTrainingSkills",
            body,
            referer=self.cfg.referer_research,
        )

    async def train_skill(self, skillTypeId: str) -> dict:
        """Train skill using live research APIs with multi-endpoint fallback"""
        base = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
        }
        referer = self.cfg.referer_research
        payload = base | {"skillTypeId": str(skillTypeId)}
        alt_payload = base | {"researchTypeId": str(skillTypeId)}

        paths = [
            "/WebService/Research.asmx/TrainSkill",
            "/WebService/Research.asmx/StartSkillTraining",
            "/WebService/Research.asmx/TrainResearch",
            "/WebService/Research.asmx/StartResearch",
        ]
        
        for p in paths:
            body = payload if "Skill" in p else alt_payload
            try:
                data = await self.post_asmx(p, body, referer=referer)
                if isinstance(data, dict) and (data.get("ReturnValue") == 1 or data.get("success")):
                    return data
                if isinstance(data, dict) and "ReturnString" in data:
                    return data
            except Exception as e:
                log.debug(f"Research endpoint {p} failed: {e}")
                continue
        
        return {"ReturnValue": 0, "ReturnString": "No research train endpoint accepted the request"}

    async def get_kingdom_status(self) -> Dict[str, Any]:
        """Get complete kingdom state using live ASMX APIs"""
        try:
            base = {
                "accountId": self.cfg.account_id,
                "token": self.cfg.token,
                "kingdomId": self.cfg.kingdom_id,
            }

            # Fetch live data from multiple ASMX endpoints
            kd = await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomDetails", base, self.cfg.referer_overview)
            rjs = await self.get_resources()
            pjs = await self.get_population()
            bjs = await self.get_buildings()
            sk_json = await self.get_skills()
            ts_json = await self.get_training_skills()

            # Normalize API data to bot format
            resources, storage, prod_hr, flags = _normalize_resources(rjs)
            units, inprog_units, returning = _normalize_population(pjs)
            buildings, inprog_build, can_build = _normalize_buildings(bjs)
            skills = normalize_skills(sk_json)
            training = normalize_training_skills(ts_json)

            # Build complete kingdom state
            kraw = {
                "id": kd.get("id"),
                "name": kd.get("name"),
                "tax_rate": kd.get("taxRate", 24),
                "peasants": units.get("peasants", kd.get("totalPopulation", 0)),
                "land": kd.get("land", resources.get("land", 0)),
                "resources": {k: resources[k] for k in ("food","wood","stone","gold")},
                "storage":   {k: storage[k]   for k in ("food","wood","stone","gold")},
                "prod_per_hour": prod_hr,
                "flags": flags,
                "season": (kd.get("seasonName") or "").strip(),
                "units": units,
                "buildings": buildings,
                "research": {
                    "skills": skills,
                    "in_progress": training,
                },
                "queues": {
                    "training_busy": any(v > 0 for v in inprog_units.values()),
                    "building_busy": any(v > 0 for v in inprog_build.values()),
                    "research_busy": len(training) > 0,
                    "explore_busy": False,
                    "training_inprog": inprog_units,
                    "returning": returning,
                    "building_inprog": inprog_build,
                    "can_build": can_build,
                },
            }
            
            # Format for bot consumption
            formatted_result = {
                "ReturnValue": 1,
                "kingdom": kraw,
                "ap": 300  # Estimate based on buildings
            }
            return formatted_result
            
        except Exception as e:
            log.error(f"Live state fetch failed: {e}")
            # Return minimal fallback state to keep bot running
            return {
                "ReturnValue": 0,
                "error": f"API failed: {e}",
                "kingdom": {
                    "id": self.cfg.kingdom_id,
                    "name": "fallback",
                    "resources": {"food": 0, "gold": 0, "wood": 0, "stone": 0},
                    "buildings": {},
                    "units": {},
                    "queues": {"training_busy": False, "building_busy": False, "explore_busy": False}
                }
            }

    async def explore(self, turns: int = 1) -> Dict[str, Any]:
        """Explore for land using real Kingdoms.asmx API"""
        try:
            troops = [{"TroopTypeID": 17, "AmountToSend": 1}]
            troops_json = json.dumps(troops)
            
            log.info(f"Exploring for land with {turns} turns using 1 archer")
            result = await self.post_asmx("/WebService/Kingdoms.asmx/Explore", {
                "accountId": self.cfg.account_id,
                "token": self.cfg.token,
                "kingdomId": self.cfg.kingdom_id,
                "troops": troops_json
            }, self.cfg.referer_overview)
            
            log.info(f"Successfully explored for land!")
            return {"success": True, "turns": turns, "data": result}
                
        except Exception as e:
            log.error(f"Exploration failed: {e}")
            return {"error": str(e)}

    async def build_structure(self, building_type: str, quantity: int = 1) -> Dict[str, Any]:
        """Build structures using real Buildings.asmx API"""
        try:
            building_types = await self.get_building_types()
            building_type_id = building_types.get(building_type.lower().replace(" ", ""))
            
            if not building_type_id:
                return {"error": f"Unknown building type: {building_type}"}

            body = {
                "accountId": self.cfg.account_id,
                "token": self.cfg.token,
                "kingdomId": self.cfg.kingdom_id,
                "buildingTypeId": str(building_type_id),
                "quantity": quantity,
            }

            result = await self.post_asmx("/WebService/Buildings.asmx/BuildBuilding", body, self.cfg.referer_buildings)
            
            log.info(f"Successfully built {quantity}x {building_type}!")
            return {"success": True, "building": building_type, "quantity": quantity, "message": result.get("ReturnString", "Build queued")}
                
        except Exception as e:
            log.error(f"Build structure failed: {e}")
            return {"error": str(e)}

    async def train_units(self, unit_type: str, quantity: int) -> Dict[str, Any]:
        """Train units using live training APIs with multi-endpoint fallback"""
        try:
            unit_types = await self.get_population_types()
            unit_type_id = unit_types.get(unit_type.lower().replace(" ", ""))
            
            if not unit_type_id:
                return {"error": f"Unknown unit type: {unit_type}"}

            body = {
                "accountId": self.cfg.account_id,
                "token": self.cfg.token,
                "kingdomId": self.cfg.kingdom_id,
                "popType": int(unit_type_id),
                "amount": quantity,
            }
            
            # Try multiple training endpoints for reliability
            paths = [
                "/WebService/Training.asmx/TrainTroops",
                "/WebService/Population.asmx/TrainPopulation",
            ]
            
            for p in paths:
                try:
                    result = await self.post_asmx(p, body, self.cfg.referer_war)
                    log.info(f"Successfully started training {quantity}x {unit_type}!")
                    return {"success": True, "unit": unit_type, "quantity": quantity, "message": result.get("ReturnString", f"Training {quantity} (id {unit_type_id})")}
                except Exception as e:
                    log.debug(f"Training endpoint {p} failed: {e}")
                    continue
            
            log.error("All training endpoints failed")
            return {"error": "Training failed: no endpoint worked"}
                
        except Exception as e:
            log.error(f"Train units failed: {e}")
            return {"error": str(e)}

    async def get_building_types(self) -> dict[str, str]:
        """Returns {normalized_name: id} from GetBuildingsTypeList"""
        async with self._btypes_lock:
            if self._btypes_cache is not None:
                return self._btypes_cache

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "world-id": str(self.cfg.world_id),
                "Origin": self.cfg.origin_url,
                "Referer": self.cfg.referer_buildings,
            }
            body = {
                "accountId": self.cfg.account_id,
                "token": self.cfg.token,
                "kingdomId": self.cfg.kingdom_id,
            }
            resp = await self._client.post("/WebService/Buildings.asmx/GetBuildingsTypeList", json=body, headers=headers)
            resp.raise_for_status()
            data = self._unwrap_json(resp)

            mapping: dict[str, str] = {}
            for b in data.get("buildings", []):
                name = (b.get("name") or "").lower().replace(" ", "")
                bid = str(b.get("id"))
                if name and bid:
                    mapping[name] = bid

            self._btypes_cache = mapping or {}
            return self._btypes_cache

    async def get_population_types(self) -> dict[str, str]:
        """Return {normalized_name: unitTypeId} from GetPopulationTypeList"""
        async with self._poptypes_lock:
            if self._poptypes_cache is not None:
                return self._poptypes_cache

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "world-id": str(self.cfg.world_id),
                "Origin": self.cfg.origin_url,
                "Referer": self.cfg.referer_war,
            }
            body = {
                "accountId": self.cfg.account_id,
                "token": self.cfg.token,
                "kingdomId": self.cfg.kingdom_id,
            }
            r = await self._client.post("/WebService/Population.asmx/GetPopulationTypeList", json=body, headers=headers)
            r.raise_for_status()
            data = self._unwrap_json(r)

            mapping: dict[str, str] = {}
            for row in (data.get("population") or data.get("populations") or []):
                name = (row.get("name") or "").lower().replace(" ", "")
                uid = str(row.get("id"))
                if name and uid:
                    mapping[name] = uid

            self._poptypes_cache = mapping or {}
            return self._poptypes_cache

def _normalize_resources(res_api: dict) -> tuple[dict, dict, dict, dict]:
    """Convert API resources list into resources, storage, prod_hr, flags"""
    name_map = {
        "food": "food",
        "gold": "gold", 
        "wood": "wood",
        "stone": "stone",
        "mana": "mana",
        "land": "land",
        "blue gems": "blue_gems",
    }
    resources, storage, prod_hr = {}, {}, {}
    flags = {"maintenance_issue_wood": False, "maintenance_issue_stone": False}
    rows = (res_api or {}).get("resources", [])
    for r in rows:
        n = (r.get("name") or "").lower().strip()
        key = name_map.get(n)
        if not key:
            continue
        resources[key] = int(r.get("amount", 0))
        storage[key] = int(r.get("capacity", 0))
        prod_hr[key] = int(r.get("productionPerHour", 0))
        if n == "wood" and r.get("maintenanceIssue"):  flags["maintenance_issue_wood"] = True
        if n == "stone" and r.get("maintenanceIssue"): flags["maintenance_issue_stone"] = True
    # ensure all keys exist
    for k in name_map.values():
        resources.setdefault(k, 0); storage.setdefault(k, 0); prod_hr.setdefault(k, 0)
    return resources, storage, prod_hr, flags

def _normalize_population(pop_api: dict) -> tuple[dict, dict, dict]:
    """Returns units, inprog, returning from API population data"""
    key_map = {
        "peasants": "peasants",
        "footmen": "footmen", 
        "pikemen": "pikemen",
        "archers": "archers",
        "crossbowmen": "crossbow",
        "light cavalry": "light_cav",
        "heavy cavalry": "heavy_cav",
        "knights": "knights",
        "elites": "elites",
        "spies": "spies",
        "priests": "priests",
        "diplomats": "diplomats",
        "market wagons": "wagons",
    }
    units, inprog, returning = {}, {}, {}
    rows = (pop_api or {}).get("population", [])
    for r in rows:
        name = (r.get("name") or "").lower().strip()
        key = key_map.get(name)
        if not key:
            continue
        units[key]      = int(r.get("amount", 0))
        inprog[key]     = int(r.get("amountInProgress", 0))
        returning[key]  = int(r.get("amountReturning", 0))
    # ensure keys exist
    for k in key_map.values():
        units.setdefault(k, 0); inprog.setdefault(k, 0); returning.setdefault(k, 0)
    return units, inprog, returning

def _normalize_buildings(b_api: dict) -> tuple[dict, dict, dict]:
    """Returns buildings, inprog, can_build from API buildings data"""
    map_name = {
        "houses": "houses",
        "grain farms": "farms",
        "lumber yards": "lumber", 
        "stone quarries": "quarries",
        "barracks": "barracks",
        "stables": "stables",
        "archery ranges": "archery",
        "guildhalls": "guild",
        "temples": "temples",
        "markets": "markets",
        "barns": "barns",
        "castles": "castles",
        "horse farms": "horse_farms",
    }
    buildings, inprog, canb = {}, {}, {}
    for r in (b_api or {}).get("buildings", []):
        name = (r.get("name") or "").lower().strip()
        key = map_name.get(name)
        if not key:
            continue
        buildings[key] = int(r.get("amount", 0))
        inprog[key]    = int(r.get("amountInProgress", 0))
        canb[key]      = bool(r.get("canBuild", False))
    # ensure keys exist
    for k in ("houses","farms","lumber","quarries","barracks","stables","archery","barns","markets","temples","castles"):
        buildings.setdefault(k, 0); inprog.setdefault(k, 0); canb.setdefault(k, True)
    return buildings, inprog, canb

def normalize_skills(sk_json: dict) -> list[dict]:
    """Accepts GetSkills response and emits normalized skill list"""
    rows = sk_json.get("skills") or sk_json.get("items") or sk_json.get("research") or []
    out = []
    for r in rows:
        rid  = r.get("id") or r.get("skillTypeId") or r.get("researchTypeId")
        name = (r.get("name") or r.get("displayName") or "").strip()
        cat  = (r.get("category") or r.get("group") or r.get("tree") or "").strip()
        lvl  = int(r.get("currentLevel") or r.get("level") or 0)
        maxl = int(r.get("maxLevel") or r.get("maximumLevel") or 0)
        gold = int(r.get("goldCost") or r.get("gold") or 0)
        gems = int(r.get("gemCost") or r.get("gems") or 0)
        time = (r.get("timeNext") or r.get("researchNextLevel") or r.get("duration") or "").strip()
        can  = bool(r.get("canTrain", True))
        preq = r.get("prerequisites") or r.get("prereqText") or ""
        if isinstance(preq, list):
            preq = ", ".join(str(x.get("name") if isinstance(x, dict) else x) for x in preq)
        out.append({
            "id": str(rid) if rid is not None else "",
            "name": name,
            "category": cat,
            "currentLevel": lvl,
            "maxLevel": maxl,
            "goldCost": gold,
            "gemCost": gems,
            "timeText": time,
            "canTrain": can,
            "prereqText": str(preq),
        })
    return out

def normalize_training_skills(ts_json: dict) -> list[dict]:
    """Normalizes GetTrainingSkills response -> list of in-progress items"""
    rows = ts_json.get("trainingSkills") or ts_json.get("skills") or ts_json.get("items") or []
    out = []
    for r in rows:
        rid  = r.get("id") or r.get("skillTypeId") or r.get("researchTypeId")
        name = (r.get("name") or "").strip()
        remaining = (r.get("timeRemainingText") or r.get("timeRemaining") or r.get("etaText") or "").strip()
        level_to  = int(r.get("targetLevel") or r.get("nextLevel") or (r.get("currentLevel") or 0) + 1)
        out.append({
            "id": str(rid) if rid is not None else "",
            "name": name,
            "eta": remaining,
            "targetLevel": level_to,
        })
    return out

def _find_skill_id(skills: list[dict], name_like: str) -> str | None:
    """Find skill ID by name with fuzzy matching"""
    key = name_like.lower().replace(" ", "")
    for s in skills:
        if s["name"].lower().replace(" ", "") == key:
            return s["id"] or None
    # loose contains match
    for s in skills:
        if key in s["name"].lower().replace(" ", ""):
            return s["id"] or None
    return None


class MockKG2Client(KG2Client):
    """
    Simulates build/train/attack/spy/explore/tax/research with economy ticks.
    Includes: networth calc, explore ETA growth + 25k cap, upkeep/maintenance,
    tax/pop dynamics, research bonuses (food/tax/AP/timers), honor-safe-ish play.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_req_time = 0.0
        self._client = None  # Mock client doesn't need real HTTP client
        now = time.time()
        self.state = {
            "id": cfg.kingdom_id, "name": "Mockshire",
            "resources": {"food": 20000, "wood": 15000, "stone": 12000, "gold": 3000, "horses": 50},
            "storage": {"food": 100000, "wood": 100000, "stone": 100000, "gold": 999999, "horses": 1000},
            "land": 500, "peasants": 500,
            "units": {"peasants": 0, "foot": 0, "pike": 200, "arch": 40, "xbow": 0, "lcav": 60, "hcav": 0, "knight": 0, "spy": 8, "priest": 0, "dip": 0},
            "buildings": {"farm": 5, "lumberyard": 4, "quarry": 3, "barracks": 4, "house": 3, "barn": 0, "archery": 0, "stables": 1},
            "tax_rate": 24,
            "queues": {"training_busy": False, "training_end": 0.0, "speedup_used_training": False,
                       "building_busy": False, "building_end": 0.0, "speedup_used_building": False,
                       "explore_busy": False, "explore_end": 0.0,
                       "research_busy": False, "research_end": 0.0},
            "explore_time_base": 15.0,
            "_last_tick": now,
            "tech": {"granary": 0, "taxation": 0, "drill": 0, "logistics": 0},
            "_bonus": {"food_prod_pct": 0.0, "gold_tax_pct": 0.0, "ap_pct": 0.0, "timers_pct": 0.0},
            "honor": 100,
            "turns": 100,
        }

    # helpers
    def _troop_upkeep(self) -> Dict[str, int]:
        u = self.state["units"]
        food = ( self.state["peasants"]*2 + u.get("foot",0)*10 + u.get("pike",0)*25 + u.get("arch",0)*38 + u.get("xbow",0)*30
               + u.get("lcav",0)*50 + u.get("hcav",0)*70 + u.get("knight",0)*90 + u.get("spy",0)*18 + u.get("priest",0)*30 + u.get("dip",0)*15 )
        gold = ( u.get("foot",0)*4 + u.get("pike",0)*6 + u.get("arch",0)*6 + u.get("xbow",0)*4
               + u.get("lcav",0)*8 + u.get("hcav",0)*14 + u.get("knight",0)*50 + u.get("spy",0)*10 + u.get("priest",0)*20 + u.get("dip",0)*15 )
        return {"food": food, "gold": gold}

    def _production_per_hour(self) -> Dict[str,int]:
        b = self.state["buildings"]; bonus = self.state["_bonus"]["food_prod_pct"]
        return {"food": int(b.get("farm",0)*120*(1.0+bonus/100.0)),
                "wood": b.get("lumberyard",0)*60, "stone": b.get("quarry",0)*60, "gold": 0}

    def _gold_from_tax_per_hour(self) -> int:
        pop = self.state["peasants"]; u = self.state["units"]
        taxed_heads = pop + sum(u.get(k,0) for k in u)
        bonus = self.state["_bonus"]["gold_tax_pct"]
        base = taxed_heads * int(self.state["tax_rate"])
        return int(base*(1.0+bonus/100.0))

    def _maintenance_per_hour(self) -> Dict[str,int]:
        b = self.state["buildings"]
        wood = b.get("farm",0)*5 + b.get("lumberyard",0)*7 + b.get("quarry",0)*7 + b.get("barracks",0)*9 + b.get("house",0)*4 + b.get("archery",0)*6 + b.get("stables",0)*8 + b.get("barn",0)*4
        stone= b.get("farm",0)*4 + b.get("lumberyard",0)*6 + b.get("quarry",0)*6 + b.get("barracks",0)*9 + b.get("house",0)*3 + b.get("archery",0)*5 + b.get("stables",0)*7 + b.get("barn",0)*3
        return {"wood": int(wood*0.01), "stone": int(stone*0.01)}

    def _ap(self) -> int:
        u = self.state["units"]
        base = (u.get("foot",0)*1 + u.get("pike",0)*2 + u.get("arch",0)*1 + u.get("xbow",0)*3 + u.get("lcav",0)*5 + u.get("hcav",0)*7 + u.get("knight",0)*15)
        rps = 0
        rps += int(0.1*(u.get("lcav",0)+u.get("hcav",0))*(1 + u.get("arch",0)/max(1,(u.get("foot",0)+u.get("pike",0)))))
        rps += int(0.1*u.get("pike",0)*(1 + (u.get("lcav",0)+u.get("hcav",0))/max(1,u.get("arch",0)+1)))
        rps += int(0.08*u.get("arch",0)*(1 + (u.get("foot",0)+u.get("pike",0))/max(1,(u.get("lcav",0)+u.get("hcav",0))+1)))
        return int((base+rps)*(1.0+self.state["_bonus"]["ap_pct"]/100.0))

    def _networth(self) -> float:
        r = self.state["resources"]; land = self.state["land"]; horses = r.get("horses",0)
        return land*0.04 + r.get("food",0)*0.0001 + r.get("gold",0)*0.0005 + r.get("stone",0)*0.0002 + r.get("wood",0)*0.0002 + horses*0.00025

    async def _economy_tick(self):
        now = time.time(); last = self.state.get("_last_tick", now); dt = max(0, now-last)
        if dt < 1: return
        hrs = dt/3600.0
        prod = self._production_per_hour(); upkeep = self._troop_upkeep(); maint = self._maintenance_per_hour()
        R = self.state["resources"]; S = self.state["storage"]
        R["food"] = max(0, min(S["food"],  R["food"] + int(prod["food"]*hrs)  - int(upkeep["food"]*hrs)))
        R["wood"] = max(0, min(S["wood"],  R["wood"] + int((prod["wood"]-maint["wood"])*hrs)))
        R["stone"]= max(0, min(S["stone"], R["stone"]+ int((prod["stone"]-maint["stone"])*hrs)))
        R["gold"] = max(0, min(S["gold"],  R["gold"] + int((self._gold_from_tax_per_hour()-upkeep["gold"])*hrs)))
        # peasants drift by tax
        if self.state["tax_rate"] <= 24: self.state["peasants"] += int(hrs*random.randint(1,3))
        else: self.state["peasants"] = max(0, self.state["peasants"] - int(hrs*random.randint(1,3)))
        self.state["_last_tick"] = now

    async def _tick(self):
        await self._economy_tick()
        now = time.time(); q = self.state["queues"]
        # training
        if q["training_busy"] and now >= q["training_end"]:
            q["training_busy"] = False; q["speedup_used_training"] = False
            batch = q.pop("_train_batch", None)
            if batch:
                unit, qty = batch["unit"], batch["qty"]
                self.state["units"][unit] = self.state["units"].get(unit, 0) + qty
        # building
        if q["building_busy"] and now >= q["building_end"]:
            q["building_busy"] = False; q["speedup_used_building"] = False
        # explore
        if q["explore_busy"] and now >= q["explore_end"]:
            q["explore_busy"] = False
            gain = random.randint(5, 20)
            self.state["land"] = min(25000, self.state["land"] + gain)
            for k,v in {"food":random.randint(200,600),"wood":random.randint(200,600),"stone":random.randint(150,500),"gold":random.randint(0,10)}.items():
                self.state["resources"][k] = min(self.state["storage"][k], self.state["resources"][k] + v)
            self.state["explore_time_base"] = min(3600.0, self.state["explore_time_base"] * 1.15)
        # research
        if q["research_busy"] and now >= q["research_end"]:
            q["research_busy"] = False
            tech = q.pop("_research_tech", None)
            if tech:
                self.state["tech"][tech] = self.state["tech"].get(tech, 0) + 1
                # Apply research bonuses
                self._apply_research_bonuses()

    def _apply_research_bonuses(self):
        """Apply research bonuses to the state"""
        tech = self.state["tech"]
        bonus = self.state["_bonus"]
        
        # Granary: +10% food production per level
        bonus["food_prod_pct"] = tech.get("granary", 0) * 10.0
        
        # Taxation: +5% gold tax per level
        bonus["gold_tax_pct"] = tech.get("taxation", 0) * 5.0
        
        # Drill: +8% AP per level
        bonus["ap_pct"] = tech.get("drill", 0) * 8.0
        
        # Logistics: -5% timer reduction per level
        bonus["timers_pct"] = tech.get("logistics", 0) * -5.0

    async def _respect_rate_limit(self):
        # MUCH slower to avoid server crashes - 5 second minimum between requests
        min_interval = 15.0  # Increased to 15 seconds - prevent server overload
        dt = time.time() - self._last_req_time
        if dt < min_interval:
            await asyncio.sleep(min_interval - dt)
        self._last_req_time = time.time()

    # API endpoints
    async def get_kingdom_status(self) -> Dict[str, Any]:
        await self._tick()
        return {
            "kingdom": self.state,
            "ap": self._ap(),
            "networth": self._networth(),
            "upkeep": self._troop_upkeep(),
            "production": self._production_per_hour(),
            "tax_income": self._gold_from_tax_per_hour(),
            "maintenance": self._maintenance_per_hour(),
        }

    async def explore(self, turns: int = 1) -> Dict[str, Any]:
        await self._tick()
        if self.state["queues"]["explore_busy"]:
            return {"error": "Explore already in progress"}
        
        if self.state["turns"] < turns:
            return {"error": f"Not enough turns. Have {self.state['turns']}, need {turns}"}
        
        self.state["turns"] -= turns
        timer_bonus = 1.0 + self.state["_bonus"]["timers_pct"] / 100.0
        duration = max(5.0, self.state["explore_time_base"] * turns * timer_bonus)
        
        self.state["queues"]["explore_busy"] = True
        self.state["queues"]["explore_end"] = time.time() + duration
        
        return {"success": True, "duration": duration, "turns_used": turns}

    async def train_units(self, unit_type: str, quantity: int) -> Dict[str, Any]:
        await self._tick()
        
        if self.state["queues"]["training_busy"]:
            return {"error": "Training already in progress"}
        
        # Cost calculation (simplified)
        costs = {
            "foot": {"food": 10, "gold": 8, "time": 3.0},
            "pike": {"food": 25, "gold": 15, "time": 5.0},
            "arch": {"food": 30, "gold": 20, "time": 4.0},
            "lcav": {"food": 45, "gold": 35, "horses": 1, "time": 8.0},
            "spy": {"food": 20, "gold": 25, "time": 6.0},
        }
        
        if unit_type not in costs:
            return {"error": f"Unknown unit type: {unit_type}"}
        
        cost = costs[unit_type]
        total_food = cost["food"] * quantity
        total_gold = cost["gold"] * quantity
        total_horses = cost.get("horses", 0) * quantity
        
        r = self.state["resources"]
        if r["food"] < total_food or r["gold"] < total_gold or r["horses"] < total_horses:
            return {"error": "Insufficient resources"}
        
        # Deduct resources
        r["food"] -= total_food
        r["gold"] -= total_gold
        r["horses"] -= total_horses
        
        # Set training queue
        timer_bonus = 1.0 + self.state["_bonus"]["timers_pct"] / 100.0
        duration = cost["time"] * quantity * timer_bonus * 60  # Convert to seconds
        
        self.state["queues"]["training_busy"] = True
        self.state["queues"]["training_end"] = time.time() + duration
        self.state["queues"]["_train_batch"] = {"unit": unit_type, "qty": quantity}
        
        return {"success": True, "duration": duration, "unit": unit_type, "quantity": quantity}

    async def build_structure(self, building_type: str, quantity: int = 1) -> Dict[str, Any]:
        await self._tick()
        
        if self.state["queues"]["building_busy"]:
            return {"error": "Building already in progress"}
        
        # Building costs (simplified)
        costs = {
            "farm": {"wood": 80, "stone": 60, "time": 10.0},
            "lumberyard": {"wood": 100, "stone": 80, "time": 12.0},
            "quarry": {"wood": 120, "stone": 100, "time": 15.0},
            "barracks": {"wood": 150, "stone": 120, "time": 18.0},
            "house": {"wood": 60, "stone": 40, "time": 8.0},
        }
        
        if building_type not in costs:
            return {"error": f"Unknown building type: {building_type}"}
        
        cost = costs[building_type]
        total_wood = cost["wood"] * quantity
        total_stone = cost["stone"] * quantity
        
        r = self.state["resources"]
        if r["wood"] < total_wood or r["stone"] < total_stone:
            return {"error": "Insufficient resources"}
        
        # Check land availability
        current_buildings = sum(self.state["buildings"].values())
        if current_buildings + quantity > self.state["land"]:
            return {"error": "Insufficient land"}
        
        # Deduct resources
        r["wood"] -= total_wood
        r["stone"] -= total_stone
        
        # Set building queue
        timer_bonus = 1.0 + self.state["_bonus"]["timers_pct"] / 100.0
        duration = cost["time"] * quantity * timer_bonus * 60  # Convert to seconds
        
        self.state["queues"]["building_busy"] = True
        self.state["queues"]["building_end"] = time.time() + duration
        
        # Add buildings immediately (simplified)
        self.state["buildings"][building_type] = self.state["buildings"].get(building_type, 0) + quantity
        
        return {"success": True, "duration": duration, "building": building_type, "quantity": quantity}

    async def research(self, tech: str) -> Dict[str, Any]:
        await self._tick()
        
        if self.state["queues"]["research_busy"]:
            return {"error": "Research already in progress"}
        
        # Research costs (simplified)
        costs = {
            "granary": {"gold": 1000, "time": 120.0},
            "taxation": {"gold": 1500, "time": 180.0},
            "drill": {"gold": 2000, "time": 240.0},
            "logistics": {"gold": 2500, "time": 300.0},
        }
        
        if tech not in costs:
            return {"error": f"Unknown research: {tech}"}
        
        cost = costs[tech]
        current_level = self.state["tech"].get(tech, 0)
        
        # Exponential cost increase
        multiplier = 1.5 ** current_level
        total_gold = int(cost["gold"] * multiplier)
        
        if self.state["resources"]["gold"] < total_gold:
            return {"error": "Insufficient gold"}
        
        # Deduct resources
        self.state["resources"]["gold"] -= total_gold
        
        # Set research queue
        timer_bonus = 1.0 + self.state["_bonus"]["timers_pct"] / 100.0
        duration = cost["time"] * multiplier * timer_bonus
        
        self.state["queues"]["research_busy"] = True
        self.state["queues"]["research_end"] = time.time() + duration
        self.state["queues"]["_research_tech"] = tech
        
        return {"success": True, "duration": duration, "tech": tech, "level": current_level + 1}

    async def attack_kingdom(self, target_id: int, unit_allocation: Dict[str, int]) -> Dict[str, Any]:
        """Simulate attacking another kingdom"""
        await self._tick()
        
        # Honor safety check
        if self.state["honor"] < 50:
            return {"error": "Honor too low for safe attacking"}
        
        # Check if we have the units
        for unit_type, qty in unit_allocation.items():
            if self.state["units"].get(unit_type, 0) < qty:
                return {"error": f"Insufficient {unit_type} units"}
        
        # Simulate battle (simplified)
        total_ap = sum(qty * {"foot": 1, "pike": 2, "arch": 1, "lcav": 5, "hcav": 7}.get(unit_type, 1) 
                      for unit_type, qty in unit_allocation.items())
        
        # Random outcome
        success = random.random() < 0.6  # 60% success rate
        
        if success:
            # Gain resources
            loot = {
                "food": random.randint(1000, 5000),
                "gold": random.randint(500, 2000),
                "wood": random.randint(300, 1500),
                "stone": random.randint(200, 1000),
            }
            
            for resource, amount in loot.items():
                self.state["resources"][resource] = min(
                    self.state["storage"][resource],
                    self.state["resources"][resource] + amount
                )
            
            # Some unit losses
            for unit_type, qty in unit_allocation.items():
                losses = random.randint(0, max(1, qty // 10))
                self.state["units"][unit_type] = max(0, self.state["units"][unit_type] - losses)
            
            self.state["honor"] = max(0, self.state["honor"] - random.randint(1, 5))
            
            return {
                "success": True,
                "loot": loot,
                "honor_lost": random.randint(1, 5),
                "casualties": "light"
            }
        else:
            # Failed attack - lose units
            for unit_type, qty in unit_allocation.items():
                losses = random.randint(qty // 4, qty // 2)
                self.state["units"][unit_type] = max(0, self.state["units"][unit_type] - losses)
            
            self.state["honor"] = max(0, self.state["honor"] - random.randint(5, 15))
            
            return {
                "success": False,
                "honor_lost": random.randint(5, 15),
                "casualties": "heavy"
            }

    async def spy_on_kingdom(self, target_id: int, spy_count: int) -> Dict[str, Any]:
        """Simulate spying on another kingdom"""
        await self._tick()
        
        if self.state["units"].get("spy", 0) < spy_count:
            return {"error": "Insufficient spy units"}
        
        # Simulate spy mission
        success = random.random() < 0.7  # 70% success rate
        
        if success:
            # Return mock target info
            target_info = {
                "land": random.randint(300, 2000),
                "networth": random.randint(500, 5000),
                "units": {
                    "foot": random.randint(0, 500),
                    "pike": random.randint(0, 300),
                    "arch": random.randint(0, 200),
                    "lcav": random.randint(0, 100),
                },
                "resources": {
                    "food": random.randint(1000, 20000),
                    "gold": random.randint(500, 10000),
                }
            }
            
            return {"success": True, "target_info": target_info}
        else:
            # Failed spy mission - lose some spies
            losses = random.randint(1, min(spy_count, 3))
            self.state["units"]["spy"] = max(0, self.state["units"]["spy"] - losses)
            
            return {"success": False, "spies_lost": losses}

# ------------------------
# AI Brain with Learning
# ------------------------

class AIBrain:
    """
    Self-learning AI brain that adapts behavior based on outcomes
    """
    def __init__(self, state_file: str = "brain_state.json"):
        self.state_file = state_file
        self.state = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load AI brain state from disk"""
        default_state = {
            "risk_tolerance": 0.7,
            "aggression_level": 0.5,
            "economic_focus": 0.6,
            "exploration_priority": 0.8,
            "research_priorities": {
                "engineering": 0.9,                # Fastest & unlocks other tech
                "better_farming_methods": 0.8,     # Improves farms
                "animal_husbandry": 0.7,           # Good economic boost
                "improved_metal_working": 0.6      # Unlocks tool improvements
            },
            "unit_preferences": {
                "foot": 0.3,
                "pike": 0.7,
                "arch": 0.6,
                "lcav": 0.8,
                "spy": 0.9
            },
            "building_priorities": {
                "farm": 0.9,
                "lumberyard": 0.7,
                "quarry": 0.7,
                "barracks": 0.8,
                "house": 0.6
            },
            "learning_data": {
                "successful_attacks": 0,
                "failed_attacks": 0,
                "exploration_gains": [],
                "resource_shortages": [],
                "last_attack_time": 0,
                "target_success_rates": {},
                "strategy_performance": {},
                "time_of_day_patterns": {},
                "weekly_performance": [0.0] * 7,
                "mood_history": [],
            },
            "sleep_schedule": {
                "sleep_start_hour": 23,  # 11 PM
                "sleep_end_hour": 7,     # 7 AM
                "sleep_duration_hours": 8,
                "last_sleep_time": 0,
                "sleep_variance_minutes": 30,
            },
            "mood_state": {
                "current_mood": "neutral",
                "mood_factor": 1.0,
                "last_mood_change": 0,
                "stress_level": 0.0,
            },
            "activity_patterns": {
                "login_times": [],
                "daily_sessions": 0,
                "weekend_behavior": True,
                "holiday_mode": False,
            },
            "performance_metrics": {
                "actions_per_hour": 0.0,
                "resource_efficiency": 0.0,
                "survival_time": 0.0,
                "adaptation_speed": 0.5,
            }
        }
        
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                    # Merge with defaults to handle new keys
                    default_state.update(loaded_state)
            return default_state
        except Exception as e:
            log.warning(f"Failed to load AI state: {e}")
            return default_state
    
    def save_state(self):
        """Save AI brain state to disk"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save AI state: {e}")
    
    def learn_from_attack(self, success: bool, resources_gained: Optional[Dict[str, int]] = None):
        """Learn from attack outcomes"""
        if success:
            self.state["learning_data"]["successful_attacks"] += 1
            self.state["aggression_level"] = min(1.0, self.state["aggression_level"] + 0.05)
            if resources_gained:
                log.info(f"Successful attack learned. Aggression up: {self.state['aggression_level']:.2f}")
        else:
            self.state["learning_data"]["failed_attacks"] += 1
            self.state["aggression_level"] = max(0.1, self.state["aggression_level"] - 0.1)
            self.state["risk_tolerance"] = max(0.3, self.state["risk_tolerance"] - 0.05)
            log.info(f"Failed attack learned. Aggression down: {self.state['aggression_level']:.2f}")
        
        self.state["learning_data"]["last_attack_time"] = time.time()
        self.save_state()
    
    def learn_from_resource_shortage(self, resource: str):
        """Learn from resource shortages"""
        shortages = self.state["learning_data"]["resource_shortages"]
        shortages.append({"resource": resource, "time": time.time()})
        
        # Keep only recent shortages (last 24 hours)
        cutoff = time.time() - 86400
        self.state["learning_data"]["resource_shortages"] = [
            s for s in shortages if s["time"] > cutoff
        ]
        
        # Adjust priorities based on shortages
        if resource == "food":
            self.state["building_priorities"]["farm"] = min(1.0, self.state["building_priorities"]["farm"] + 0.1)
        elif resource == "gold":
            self.state["economic_focus"] = min(1.0, self.state["economic_focus"] + 0.1)
        
        self.save_state()
    
    def is_panic_mode(self, kingdom_status: Dict[str, Any]) -> bool:
        """Check if bot should enter panic/emergency mode"""
        kingdom = kingdom_status.get("kingdom", {})
        resources = kingdom.get("resources", {})
        honor = kingdom.get("honor", 100)
        
        # Critical resource shortage
        if resources.get("food", 0) <= 0 or resources.get("gold", 0) <= 0:
            return True
        
        # Honor lock danger (very low honor)
        if honor < 3:
            return True
        
        # Too many recent failures
        recent_failures = self.state["learning_data"]["failed_attacks"]
        recent_successes = self.state["learning_data"]["successful_attacks"]
        if recent_failures > 5 and recent_successes == 0:
            return True
            
        return False
    
    def get_emergency_action(self, kingdom_status: Dict[str, Any]) -> str:
        """Determine emergency recovery action"""
        kingdom = kingdom_status.get("kingdom", {})
        resources = kingdom.get("resources", {})
        
        if resources.get("food", 0) <= 0:
            return "focus_food_production"
        elif resources.get("gold", 0) <= 0:
            return "optimize_taxation"
        elif kingdom.get("honor", 100) < 3:
            return "defensive_posture"
        else:
            return "stabilize_economy"
    
    def should_attack(self, current_resources: Dict[str, int]) -> bool:
        """Decide if we should attack based on current state"""
        # Don't attack if resources are too low
        if current_resources.get("food", 0) < 1000 or current_resources.get("gold", 0) < 500:
            return False
        
        # Don't attack if we recently failed
        last_attack = self.state["learning_data"]["last_attack_time"]
        if time.time() - last_attack < 48 * 3600:  # 48 hour cooldown
            return False
        
        # Check aggression level and risk tolerance
        attack_probability = self.state["aggression_level"] * self.state["risk_tolerance"]
        # Factor in mood and recent performance
        mood_factor = self.state["mood_state"]["mood_factor"]
        adjusted_probability = attack_probability * mood_factor
        
        # Reduce probability if we've had recent failures
        recent_failures = len([s for s in self.state["learning_data"]["resource_shortages"] 
                              if time.time() - s["time"] < 3600])  # Last hour
        if recent_failures > 2:
            adjusted_probability *= 0.5
        
        return random.random() < adjusted_probability
    
    def is_sleep_time(self) -> bool:
        """Check if it's time to sleep (human behavior simulation)"""
        current_hour = datetime.now().hour
        sleep_start = self.state["sleep_schedule"]["sleep_start_hour"]
        sleep_end = self.state["sleep_schedule"]["sleep_end_hour"]
        
        if sleep_start > sleep_end:  # Sleep crosses midnight
            return current_hour >= sleep_start or current_hour < sleep_end
        else:
            return sleep_start <= current_hour < sleep_end
    
    def get_sleep_duration(self) -> float:
        """Get how long to sleep in seconds"""
        return self.state["sleep_schedule"]["sleep_duration_hours"] * 3600
    
    def should_take_break(self) -> bool:
        """Decide if we should take a random short break based on mood and patterns"""
        # Disable breaks for testing - bot should stay active
        return False
    
    def get_break_duration(self) -> float:
        """Get random break duration based on mood and time patterns"""
        # Very short breaks for testing - bot should stay active
        return 30  # 30 seconds only
    
    def calculate_action_delay(self, action_type: str, actions_taken: int) -> float:
        """Calculate realistic delay between actions"""
        base_delays = {
            "explore": (30, 120),
            "build": (45, 180),
            "train": (60, 300),
            "attack": (300, 900),  # Longer delays for attacks
            "research": (120, 600),
        }
        
        min_delay, max_delay = base_delays.get(action_type, (30, 120))
        
        # Add human-like hesitation for important actions
        if action_type in ["attack", "research"]:
            hesitation = random.uniform(10, 60)  # 10-60 second hesitation
            min_delay += hesitation
        
        # Mood affects timing
        mood_factor = self.state["mood_state"]["mood_factor"]
        if mood_factor < 0.8:  # Stressed - slower decisions
            min_delay *= 1.5
            max_delay *= 1.5
        elif mood_factor > 1.2:  # Confident - faster decisions
            min_delay *= 0.8
            max_delay *= 0.8
        
        # More delays if we've taken many actions (fatigue simulation)
        if actions_taken > 3:
            min_delay *= (1 + actions_taken * 0.2)
            max_delay *= (1 + actions_taken * 0.2)
        
        return random.uniform(min_delay, max_delay)
    
    def get_sleep_with_variance(self) -> Tuple[float, float]:
        """Get sleep start time and duration with human-like variance"""
        schedule = self.state["sleep_schedule"]
        variance_minutes = schedule["sleep_variance_minutes"]
        
        # Add random variance to sleep time
        sleep_start_variance = random.randint(-variance_minutes, variance_minutes)
        sleep_duration_variance = random.randint(-30, 60)  # -30 min to +1 hour
        
        base_start_hour = schedule["sleep_start_hour"]
        base_duration_hours = schedule["sleep_duration_hours"]
        
        actual_start_hour = (base_start_hour + sleep_start_variance / 60) % 24
        actual_duration_hours = max(6, base_duration_hours + sleep_duration_variance / 60)
        
        return actual_start_hour, actual_duration_hours
    
    def update_mood(self, recent_success: bool, resource_status: str):
        """Update mood based on recent events"""
        current_time = time.time()
        mood_state = self.state["mood_state"]
        
        # Decay mood change over time
        if current_time - mood_state["last_mood_change"] > 3600:  # 1 hour
            mood_state["mood_factor"] = (mood_state["mood_factor"] + 1.0) / 2
            mood_state["stress_level"] *= 0.9
        
        if recent_success:
            mood_state["mood_factor"] = min(1.5, mood_state["mood_factor"] + 0.1)
            mood_state["stress_level"] = max(0.0, mood_state["stress_level"] - 0.1)
            mood_state["current_mood"] = "confident" if mood_state["mood_factor"] > 1.2 else "neutral"
        else:
            mood_state["mood_factor"] = max(0.5, mood_state["mood_factor"] - 0.2)
            mood_state["stress_level"] = min(1.0, mood_state["stress_level"] + 0.2)
            mood_state["current_mood"] = "stressed" if mood_state["mood_factor"] < 0.8 else "neutral"
        
        if resource_status == "critical":
            mood_state["current_mood"] = "stressed"
            mood_state["stress_level"] = min(1.0, mood_state["stress_level"] + 0.3)
        
        mood_state["last_mood_change"] = current_time
        
        # Add to mood history
        mood_history = mood_state.get("mood_history", [])
        mood_history.append({
            "time": current_time,
            "mood": mood_state["current_mood"],
            "factor": mood_state["mood_factor"],
            "stress": mood_state["stress_level"]
        })
        
        # Keep only last 24 hours of mood history
        cutoff = current_time - 86400
        mood_state["mood_history"] = [m for m in mood_history if m["time"] > cutoff]
        
        self.save_state()
    
    def analyze_target_patterns(self, target_id: str, success: bool, loot: Optional[Dict[str, int]] = None):
        """Analyze and learn from attack patterns on specific targets"""
        target_data = self.state["learning_data"]["target_success_rates"]
        
        if target_id not in target_data:
            target_data[target_id] = {
                "attempts": 0,
                "successes": 0,
                "total_loot": {"food": 0, "wood": 0, "stone": 0, "gold": 0},
                "last_attempt": 0,
                "risk_assessment": "unknown"
            }
        
        data = target_data[target_id]
        data["attempts"] += 1
        data["last_attempt"] = time.time()
        
        if success:
            data["successes"] += 1
            if loot:
                for resource, amount in loot.items():
                    data["total_loot"][resource] = data["total_loot"].get(resource, 0) + amount
        
        # Calculate success rate and risk assessment
        success_rate = data["successes"] / data["attempts"]
        if success_rate > 0.7:
            data["risk_assessment"] = "safe"
        elif success_rate > 0.4:
            data["risk_assessment"] = "moderate"
        else:
            data["risk_assessment"] = "dangerous"
        
        self.save_state()
    
    def get_optimal_strategy(self, kingdom_status: Dict[str, Any]) -> str:
        """Determine optimal strategy based on current conditions and learning"""
        resources = kingdom_status.get("kingdom", {}).get("resources", {})
        networth = kingdom_status.get("networth", 0)
        
        # Emergency mode if resources are critically low
        if resources.get("food", 0) < 1000 or resources.get("gold", 0) < 500:
            return "emergency_recovery"
        
        # Economic focus if low on multiple resources
        low_resources = sum(1 for r in ["food", "wood", "stone", "gold"] 
                           if resources.get(r, 0) < 5000)
        if low_resources >= 2:
            return "economic_focus"
        
        # Aggressive if we're doing well
        if (networth > 1000 and resources.get("food", 0) > 10000 and 
            self.state["mood_state"]["current_mood"] == "confident"):
            return "aggressive_expansion"
        
        # Default balanced approach
        return "balanced_growth"
    
    def is_weekend(self) -> bool:
        """Check if it's weekend for different behavior patterns"""
        return datetime.now().weekday() >= 5
    
    def should_be_less_active(self) -> bool:
        """Determine if bot should be less active based on human patterns"""
        if not self.state["activity_patterns"]["weekend_behavior"]:
            return False
            
        current_hour = datetime.now().hour
        
        # Less active during typical work hours on weekdays
        if not self.is_weekend() and 9 <= current_hour <= 17:
            return random.random() < 0.3
        
        # More active during evening hours
        if 18 <= current_hour <= 22:
            return False
            
        return random.random() < 0.1

# ------------------------
# Main Bot Logic
# ------------------------

class PerformanceTracker:
    """Track bot performance metrics for optimization"""
    def __init__(self):
        self.actions_taken = 0
        self.resources_gained = {"food": 0, "wood": 0, "stone": 0, "gold": 0}
        self.resources_spent = {"food": 0, "wood": 0, "stone": 0, "gold": 0}
        self.attacks_won = 0
        self.attacks_lost = 0
        self.exploration_count = 0
        self.buildings_built = 0
        self.research_completed = 0
        self.start_time = time.time()
    
    def record_action(self, action_type: str, success: bool = True, resources_change: Optional[Dict[str, int]] = None):
        """Record an action and its outcome"""
        self.actions_taken += 1
        
        if action_type == "attack":
            if success:
                self.attacks_won += 1
            else:
                self.attacks_lost += 1
        elif action_type == "explore":
            self.exploration_count += 1
        elif action_type == "build":
            self.buildings_built += 1
        elif action_type == "research":
            self.research_completed += 1
        
        if resources_change:
            for resource, change in resources_change.items():
                if change > 0:
                    self.resources_gained[resource] += change
                else:
                    self.resources_spent[resource] += abs(change)
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        runtime_hours = (time.time() - self.start_time) / 3600
        return {
            "actions_per_hour": self.actions_taken / max(runtime_hours, 0.1),
            "win_rate": self.attacks_won / max(self.attacks_won + self.attacks_lost, 1),
            "resource_gain_rate": sum(self.resources_gained.values()) / max(runtime_hours, 0.1),
            "exploration_rate": self.exploration_count / max(runtime_hours, 0.1),
        }

class KG2Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = MockKG2Client(cfg) if cfg.mock_mode else KG2Client(cfg)
        self.brain = AIBrain()
        self.step_count = 0
        self.running = True
        self.session_start_time = time.time()
        self.last_action_times = {}
        self.performance_tracker = PerformanceTracker()
        
        # Failure recovery system
        self.consecutive_failures = 0
        self.recovery_mode = False
        self.last_successful_action = time.time()
        
    async def initialize(self):
        """Initialize the bot"""
        log.info("KG2 AI Bot initializing...")
        log.info(f"Mode: {'Mock' if self.cfg.mock_mode else 'Live'}")
        log.info(f"Kingdom ID: {self.cfg.kingdom_id}")
        
        # Create directories
        Path(self.cfg.snapshots_dir).mkdir(parents=True, exist_ok=True)
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current kingdom status"""
        if hasattr(self.client, 'get_kingdom_status'):
            return await self.client.get_kingdom_status()
        else:
            # For live client, implement actual API calls
            return await self.client.get("/api/kingdom/status")
    
    async def safe_explore(self, status: Dict[str, Any]) -> bool:
        """Safely explore new lands"""
        try:
            kingdom = status.get("kingdom", {})
            turns = kingdom.get("turns", 0)
            
            if turns < 5:
                log.debug("Not enough turns to explore")
                return False
            
            if kingdom.get("queues", {}).get("explore_busy", False):
                log.debug("Explore already in progress")
                return False
            
            # Use a portion of available turns
            turns_to_use = min(turns // 3, 10)
            if turns_to_use < 1:
                return False
            
            result = await self.client.explore(turns_to_use)
            
            if result.get("success"):
                log.info(f"Started exploring with {turns_to_use} turns")
                return True
            else:
                log.warning(f"Explore failed: {result.get('error')}")
                return False
                
        except Exception as e:
            log.error(f"Error in explore: {e}")
            return False
    
    async def safe_build(self, status: Dict[str, Any], focus: str = "balanced") -> bool:
        """Early game focused building strategy"""
        try:
            kingdom = status.get("kingdom", {})
            resources = kingdom.get("resources", {})
            buildings = kingdom.get("buildings", {})
            land = kingdom.get("land", 300)
            
            # Respect building queue from live API
            queues = kingdom.get("queues", {})
            if queues.get("building_busy", False):
                building_inprog = queues.get("building_inprog", {})
                in_progress = [f"{k}:{v}" for k, v in building_inprog.items() if v > 0]
                log.info(f"Building queue busy: {', '.join(in_progress) if in_progress else 'checking...'}")
                return False
            
            # Check can_build flags from live API
            can_build = queues.get("can_build", {})
            
            # Calculate current storage capacity and debug
            wood_capacity = buildings.get("lumberyard", 0) * 200 + 200
            stone_capacity = buildings.get("quarry", 0) * 200 + 200
            wood_current = resources.get("wood", 0)
            stone_current = resources.get("stone", 0)
            
            # DEBUG: Log what we're seeing
            log.info(f"DEBUG: Wood {wood_current}/{wood_capacity}, Stone {stone_current}/{stone_capacity}")
            log.info(f"DEBUG: Buildings = {buildings}")
            log.info(f"DEBUG: Resources = {resources}")
            
            # PRIORITY 1: Check land usage and explore if needed
            current_buildings = sum(buildings.values())
            if current_buildings > land * 0.85:  # 85% land usage
                log.info("Land nearly full - need to explore first")
                await asyncio.sleep(3)  # Extra delay before exploration
                explore_result = await self.client.explore(1)
                if explore_result.get("success"):
                    log.info("Successfully explored for more land!")
                    return True
                else:
                    log.warning(f"Exploration failed: {explore_result.get('error', 'Unknown error')}")
                    
            # PRIORITY 2: SMART STORAGE - Calculate land_available  
            land_available = land - current_buildings
            
            # Build multiple storage buildings when needed
            if wood_current > wood_capacity * 0.8:  # Wood over 80% capacity
                storage_needed = (wood_current - wood_capacity) // 200 + 2  # Extra storage needed
                lumber_to_build = min(storage_needed, 5, land_available // 2, stone_current // 10)
                log.info(f"SMART STORAGE: Need {storage_needed} lumber yards. Building {lumber_to_build}x Lumber Yards")
                result = await self.client.build_structure("Lumber Yards", lumber_to_build)
                if result.get("success"):
                    log.info(f"Built {lumber_to_build}x lumber yards for wood storage!")
                    return True
                    
            if stone_current > stone_capacity * 0.8:  # Stone over 80% capacity  
                storage_needed = (stone_current - stone_capacity) // 200 + 2  # Extra storage needed
                quarry_to_build = min(storage_needed, 5, land_available // 2, wood_current // 10)
                log.info(f"SMART STORAGE: Need {storage_needed} quarries. Building {quarry_to_build}x quarry")
                result = await self.client.build_structure("quarry", quarry_to_build)
                if result.get("success"):
                    log.info(f"Built {quarry_to_build}x quarries for stone storage!")
                    return True
            
            # PRIORITY 3: EARLY GAME AGGRESSIVE BUILDING - Use calculated land_available
            
            # Early game targets for balanced economy
            farm_target = min(50, current_buildings // 2)  # Aggressive farm building
            house_target = min(60, current_buildings // 2)  # Aggressive housing  
            
            # SMART BUILDING: Calculate needs based on resource generation and capacity
            current_farms = buildings.get("farm", 0)
            current_food_per_hour = current_farms * 600  # Each farm = 600 food/hour
            
            # Build farms if food generation is low (need 12,000+ food/hour for growth)
            if current_food_per_hour < 12000 and land_available >= 10 and wood_current >= 200:
                farms_needed = (12000 - current_food_per_hour) // 600  # How many farms for target
                farms_to_build = min(farms_needed, 10, land_available // 10, wood_current // 20)
                log.info(f"SMART: Need {farms_needed} farms for 12k food/hour. Building {farms_to_build}x Grain Farms (have {current_farms})")
                result = await self.client.build_structure("Grain Farms", farms_to_build)
                if result.get("success"):
                    return True
                elif result.get("error") == "need_land":
                    log.info("Need more land for farm expansion - exploring first!")
                    explore_result = await self.client.explore(2)  # Explore more aggressively
                    if explore_result.get("success"):
                        log.info("Explored for land - will retry farm expansion next cycle")
                        return True
                    
            # SMART HOUSING: Build houses based on population capacity (corrected: 10 per house)
            current_houses = buildings.get("house", 0)
            population_capacity = current_houses * 10  # CORRECTED: Each house = 10 population
            total_units = sum(buildings.get(typ, 0) for typ in ["barracks", "archery", "stables", "guildhalls"])
            
            # Build houses if population capacity is getting tight (need 2x unit capacity)
            if population_capacity < total_units * 20 and land_available >= 4 and stone_current >= 100:
                houses_needed = ((total_units * 20) - population_capacity) // 10
                houses_to_build = min(houses_needed, 20, land_available, stone_current // 10)
                log.info(f"Early game expansion: Building {houses_to_build}x Houses (have {current_houses}, target {house_target})")
                result = await self.client.build_structure("Houses", houses_to_build)
                if result.get("success"):
                    return True
                elif result.get("error") == "need_land":
                    log.info("Need more land for house expansion - exploring first!")
                    explore_result = await self.client.explore(2)  # Explore more aggressively  
                    if explore_result.get("success"):
                        log.info("Explored for land - will retry house expansion next cycle")
                        return True
            
            # Build balanced infrastructure for early game
            gold_current = resources.get("gold", 0)
            
            # SMART MILITARY: Build military buildings based on capacity vs troops
            current_archers = buildings.get("archery", 0)  # API name
            archer_capacity = current_archers * 20  # Each range = 20 archers (confirmed)
            
            # Build Archery Ranges if archer capacity is low (need 60+ archer capacity)
            if archer_capacity < 60 and land_available >= 5 and wood_current >= 100:
                ranges_needed = (60 - archer_capacity) // 20
                ranges_to_build = min(ranges_needed, 3, land_available // 5, wood_current // 20)
                log.info(f"SMART: Need {ranges_needed} ranges for 60 archer capacity. Building {ranges_to_build}x Archery Ranges")
                result = await self.client.build_structure("Archery Ranges", ranges_to_build)
                if result.get("success"):
                    return True
                elif result.get("error") == "need_land":
                    log.info("Need land for Archery Ranges - exploring!")
                    await self.client.explore(1)
                    
            # SMART CAVALRY: Build stables based on cavalry capacity (corrected: 10 per stable)
            current_stables = buildings.get("stables", 0)
            cavalry_capacity = current_stables * 10  # CORRECTED: Each stable = 10 cavalry
            
            if cavalry_capacity < 30 and land_available >= 5 and stone_current >= 60:
                stables_needed = (30 - cavalry_capacity) // 10
                stables_to_build = min(stables_needed, 3, land_available // 5, stone_current // 20)
                log.info(f"SMART: Building {stables_to_build}x Stables for cavalry capacity")
                result = await self.client.build_structure("Stables", stables_to_build)
                if result.get("success"):
                    return True
                elif result.get("error") == "need_land":
                    log.info("Need land for Stables - exploring!")
                    await self.client.explore(1)
                    
            # SMART SPIES: Build guildhalls based on spy capacity (corrected: 5 per guildhall)
            current_guildhalls = buildings.get("guildhalls", 0)
            spy_capacity = current_guildhalls * 5  # CORRECTED: Each guildhall = 5 spies
            
            if spy_capacity < 15 and land_available >= 4 and wood_current >= 20:
                guildhalls_needed = (15 - spy_capacity) // 5
                guildhalls_to_build = min(guildhalls_needed, 3, land_available // 2, stone_current // 5)
                log.info(f"SMART: Building {guildhalls_to_build}x Guildhalls for spy capacity")
                result = await self.client.build_structure("Guildhalls", guildhalls_to_build) 
                if result.get("success"):
                    return True
                elif result.get("error") == "need_land":
                    log.info("Need land for Guildhalls - exploring!")
                    await self.client.explore(1)
                    
            # No need for redundant storage logic here - handled above
            
            return False
            
        except Exception as e:
            log.error(f"Error in safe build: {e}")
            return False
    
    async def safe_train(self, status: Dict[str, Any], focus: str = "balanced") -> bool:
        """Safely train military units"""
        try:
            kingdom = status.get("kingdom", {})
            resources = kingdom.get("resources", {})
            
            # Respect training queue from live API
            queues = kingdom.get("queues", {})
            if queues.get("training_busy", False):
                training_inprog = queues.get("training_inprog", {})
                in_progress = [f"{k}:{v}" for k, v in training_inprog.items() if v > 0]
                log.info(f"Training queue busy: {', '.join(in_progress) if in_progress else 'checking...'}")
                return False
            
            # Check resource levels
            food = resources.get("food", 0)
            gold = resources.get("gold", 0)
            
            if food < 2000 or gold < 1000:
                return False
            
            # Choose unit type based on preferences and resources
            preferences = self.brain.state["unit_preferences"]
            
            best_unit = None
            best_score = 0
            
            for unit_type, preference in preferences.items():
                if unit_type in ["foot", "pike", "arch", "lcav", "spy"]:
                    # Calculate affordability score
                    affordability = 1.0
                    if unit_type == "lcav" and resources.get("horses", 0) < 10:
                        affordability = 0.1
                    
                    score = preference * affordability
                    if score > best_score:
                        best_unit = unit_type
                        best_score = score
            
            if best_unit:
                # Calculate how many we can afford
                max_quantity = min(10, food // 50, gold // 30)
                if best_unit == "lcav":
                    max_quantity = min(max_quantity, resources.get("horses", 0))
                
                if max_quantity > 0:
                    result = await self.client.train_units(best_unit, max_quantity)
                    
                    if result.get("success"):
                        log.info(f"Started training {max_quantity} {best_unit}")
                        return True
                    else:
                        log.warning(f"Training failed: {result.get('error')}")
            
            return False
            
        except Exception as e:
            log.error(f"Error in train: {e}")
            return False
    
    async def safe_research(self, status: Dict[str, Any], focus: str = "balanced") -> bool:
        """Safely conduct research"""
        try:
            kingdom = status.get("kingdom", {})
            resources = kingdom.get("resources", {})
            
            if kingdom.get("queues", {}).get("research_busy", False):
                return False
            
            gold = resources.get("gold", 0)
            if gold < 15000:  # Realistic threshold - cheapest research costs 15k gold
                log.info(f"Not enough gold for research: {gold}/15000 needed")
                return False
            
            # Choose research based on priorities
            priorities = self.brain.state["research_priorities"]
            current_tech = kingdom.get("tech", {})
            
            best_research = None
            best_score = 0
            
            for tech, priority in priorities.items():
                current_level = current_tech.get(tech, 0)
                # Diminishing returns for higher levels
                adjusted_priority = priority / (1 + current_level * 0.3)
                
                if adjusted_priority > best_score:
                    best_research = tech
                    best_score = adjusted_priority
            
            if best_research:
                # Use new live research system
                result = await self.plan_research(status)
                
                if "success" in result or "Queued research" in result:
                    log.info(f"Started research: {result}")
                    return True
                else:
                    log.warning(f"Research failed: {result}")
            
            return False
            
        except Exception as e:
            log.error(f"Error in research: {e}")
            return False

    async def plan_research(self, status: Dict[str, Any]) -> str:
        """Smart research planning using live research data"""
        try:
            kingdom = status.get("kingdom", {})
            resources = kingdom.get("resources", {})
            prod_hr = kingdom.get("prod_per_hour", {})
            queues = kingdom.get("queues", {})
            research = kingdom.get("research", {})
            
            if queues.get("research_busy"):
                return "Research already in progress"
            
            skills = research.get("skills", [])
            if not skills:
                return "No research data available"

            # Smart research selection based on current needs
            pick = None
            
            # Priority 1: Food crisis or low production
            food_prod = prod_hr.get("food", 0)
            food_current = resources.get("food", 0)
            food_cap = kingdom.get("storage", {}).get("food", 100000)
            if food_current < food_cap * 0.25 or food_prod < 3000:
                for name in ("Mathematics", "Better Farming Methods", "Irrigation", "Crop Rotation", "Improved Farm Food Storage"):
                    sid = _find_skill_id(skills, name)
                    if sid:
                        pick = (name, sid)
                        break

            # Priority 2: Gold shortage
            if pick is None and resources.get("gold", 0) < 10000:
                for name in ("Mathematics", "Accounting", "Improved House Gold Storage"):
                    sid = _find_skill_id(skills, name)
                    if sid:
                        pick = (name, sid)
                        break

            # Priority 3: Construction speed (early game boost)
            if pick is None:
                for name in ("Engineering", "Improved Tools", "Better Construction Methods"):
                    sid = _find_skill_id(skills, name)
                    if sid:
                        pick = (name, sid)
                        break

            # Priority 4: Military efficiency
            if pick is None:
                for name in ("Military Tactics", "Weapon Crafting", "Advanced Training"):
                    sid = _find_skill_id(skills, name)
                    if sid:
                        pick = (name, sid)
                        break

            if pick is None:
                return "No suitable research found"

            name, sid = pick
            
            # Check affordability
            sk_map = {s["id"]: s for s in skills if s.get("id")}
            srow = sk_map.get(sid, {})
            gcost = int(srow.get("goldCost") or 0)
            if gcost and resources.get("gold", 0) < gcost:
                return f"Research '{name}' skipped: need {gcost} gold, have {resources.get('gold', 0)}"

            # Start research
            res = await self.client.train_skill(sid)
            return res.get("ReturnString", f"Queued research: {name}")
            
        except Exception as e:
            log.error(f"Research planning failed: {e}")
            return f"Research planning error: {e}"
    
    async def attempt_attack(self, status: Dict[str, Any]) -> bool:
        """Attempt to attack another kingdom"""
        try:
            kingdom = status.get("kingdom", {})
            units = kingdom.get("units", {})
            resources = kingdom.get("resources", {})
            
            if not self.brain.should_attack(resources):
                return False
            
            # Calculate available attack force
            attack_units = {}
            for unit_type in ["foot", "pike", "arch", "lcav"]:
                available = units.get(unit_type, 0)
                if available > 5:  # Keep some for defense
                    attack_units[unit_type] = available // 2
            
            if not attack_units or sum(attack_units.values()) < 10:
                log.debug("Insufficient units for attack")
                return False
            
            # Find a target (simplified - random target in mock mode)
            target_id = random.randint(1, 1000)
            
            result = await self.client.attack_kingdom(target_id, attack_units)
            
            success = result.get("success", False)
            loot = result.get("loot", {})
            
            # Learn from the outcome
            self.brain.learn_from_attack(success, loot if success and isinstance(loot, dict) else None)
            
            if success:
                log.info(f"Successful attack on {target_id}, gained: {loot}")
            else:
                log.warning(f"Failed attack on {target_id}")
            
            return success
            
        except Exception as e:
            log.error(f"Error in attack: {e}")
            return False
    
    async def spy_missions(self, status: Dict[str, Any]) -> bool:
        """Conduct spy missions"""
        try:
            kingdom = status.get("kingdom", {})
            units = kingdom.get("units", {})
            
            spy_count = units.get("spy", 0)
            if spy_count < 3:
                return False
            
            # Use some spies for intel
            spies_to_use = min(spy_count // 3, 5)
            target_id = random.randint(1, 1000)
            
            result = await self.client.spy_on_kingdom(target_id, spies_to_use)
            
            if result.get("success"):
                log.info(f"Successful spy mission on {target_id}")
                return True
            else:
                log.warning(f"Spy mission failed on {target_id}")
                return False
                
        except Exception as e:
            log.error(f"Error in spy mission: {e}")
            return False
    
    async def manage_economy(self, status: Dict[str, Any]):
        """Manage kingdom economy and detect capacity issues"""
        kingdom = status.get("kingdom", {})
        resources = kingdom.get("resources", {})
        buildings = kingdom.get("buildings", {})
        
        # Check for over-capacity (need more storage buildings)
        wood_current = resources.get("wood", 0)
        stone_current = resources.get("stone", 0)
        
        # Calculate current storage capacity based on buildings
        wood_capacity = buildings.get("lumberyard", 0) * 200 + 200  # Base 200 + 200 per lumberyard
        stone_capacity = buildings.get("quarry", 0) * 200 + 200     # Base 200 + 200 per quarry
        
        # Detect over-capacity (need more storage buildings)
        if wood_current > wood_capacity * 0.9:  # 90% full
            log.warning(f"Wood over-capacity: {wood_current}/{wood_capacity} - need more lumber mills")
            self.brain.learn_from_resource_shortage("wood_storage")
        
        if stone_current > stone_capacity * 0.9:  # 90% full  
            log.warning(f"Stone over-capacity: {stone_current}/{stone_capacity} - need more quarries")
            self.brain.learn_from_resource_shortage("stone_storage")
            
        # Check for actual shortages (too little)
        if resources.get("food", 0) < 1000:
            log.warning(f"Food shortage: {resources.get('food', 0)} - need more farms")
            self.brain.learn_from_resource_shortage("food")
            
        if resources.get("gold", 0) < 500:
            log.warning(f"Gold shortage: {resources.get('gold', 0)} - need better taxation")
            self.brain.learn_from_resource_shortage("gold")
    
    async def take_snapshot(self):
        """Take a snapshot of current state"""
        try:
            status = await self.get_status()
            timestamp = datetime.now().isoformat()
            
            snapshot = {
                "timestamp": timestamp,
                "step": self.step_count,
                "kingdom_status": status,
                "brain_state": self.brain.state
            }
            
            filename = f"snapshot_{timestamp.replace(':', '-')}.json"
            filepath = Path(self.cfg.snapshots_dir) / filename
            
            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2)
                
            log.info(f"Snapshot saved: {filename}")
            
        except Exception as e:
            log.error(f"Failed to save snapshot: {e}")
    
    async def main_loop(self):
        """Main bot execution loop"""
        log.info("Starting main bot loop...")
        
        while self.running and self.cfg.enabled:
            try:
                self.step_count += 1
                
                # Check if it's sleep time with variance
                if self.brain.is_sleep_time():
                    sleep_start, sleep_duration_hours = self.brain.get_sleep_with_variance()
                    sleep_duration = sleep_duration_hours * 3600
                    log.info(f"Sleep time! Sleeping for {sleep_duration_hours:.1f} hours (variance applied)")
                    await asyncio.sleep(sleep_duration)
                    continue
                
                # Get current status first
                status = await self.get_status()
                kingdom = status.get("kingdom", {})
                resources = kingdom.get("resources", {})
                
                # Check for panic mode
                if self.brain.is_panic_mode(status):
                    emergency_action = self.brain.get_emergency_action(status)
                    log.warning(f"PANIC MODE: Executing emergency action: {emergency_action}")
                    await self.handle_emergency(status, emergency_action)
                    await asyncio.sleep(300)  # 5 min pause in panic mode
                    continue
                
                # Determine current strategy
                current_strategy = self.brain.get_optimal_strategy(status)
                log.info(f"Current strategy: {current_strategy}")
                
                # Check if we should be less active (human behavior)
                if self.brain.should_be_less_active():
                    log.info("Reducing activity (human behavior simulation)")
                    await asyncio.sleep(self.brain.get_break_duration())
                    continue
                
                # Random breaks for human behavior
                if self.brain.should_take_break():
                    break_duration = self.brain.get_break_duration()
                    log.info(f"Taking a break for {break_duration/60:.1f} minutes (mood: {self.brain.state['mood_state']['current_mood']})")
                    await asyncio.sleep(break_duration)
                
                # Update mood based on recent performance
                recent_success = self._assess_recent_performance()
                resource_status = self._assess_resource_status(resources)
                self.brain.update_mood(recent_success, resource_status)
                
                # Manage economy
                await self.manage_economy(status)
                
                # Execute actions based on current strategy
                actions_taken = 0
                
                # Reset recovery mode if we're doing well
                if self.recovery_mode and resource_status in ["stable", "abundant"]:
                    self.recovery_mode = False
                    self.consecutive_failures = 0
                    log.info("Exiting recovery mode - resources stabilized")
                
                # Execute strategy-specific actions
                if current_strategy == "emergency_recovery":
                    # Focus only on recovery actions
                    if await self.safe_build(status, focus="economy"):
                        actions_taken += 1
                        self.performance_tracker.record_action("build", True)
                    if await self.optimize_taxation(status):
                        actions_taken += 1
                        self.performance_tracker.record_action("tax_optimization", True)
                elif current_strategy == "economic_focus":
                    # EARLY GAME: Focus on exploration + building infrastructure only
                    buildings = status.get("kingdom", {}).get("buildings", {})
                    total_buildings = sum(buildings.values())
                    
                    # In early game (< 100 buildings), skip troop training - focus on infrastructure
                    if total_buildings < 100:
                        log.info(f"Early game phase ({total_buildings} buildings) - focusing on exploration + building only")
                        if await self.safe_explore(status):
                            actions_taken += 1
                            self.performance_tracker.record_action("explore", True)
                        if await self.safe_build(status, focus="economy"):
                            actions_taken += 1
                            self.performance_tracker.record_action("build", True)
                        # Skip training and research in early game - focus on land + buildings
                    else:
                        # Mid/late game: Add training and research
                        if await self.safe_explore(status):
                            actions_taken += 1
                            self.performance_tracker.record_action("explore", True)
                        if await self.safe_build(status, focus="economy"):
                            actions_taken += 1
                            self.performance_tracker.record_action("build", True)
                        if await self.safe_train(status, focus="economy"):
                            actions_taken += 1
                            self.performance_tracker.record_action("train", True)
                        if await self.safe_research(status, focus="economy"):
                            actions_taken += 1
                            self.performance_tracker.record_action("research", True)
                elif current_strategy == "aggressive_expansion":
                    # Focus on military and attacks
                    if await self.safe_train(status, focus="military"):
                        actions_taken += 1
                        self.performance_tracker.record_action("train", True)
                    if await self.spy_missions(status):
                        actions_taken += 1
                        self.performance_tracker.record_action("spy", True)
                    if await self.attempt_attack(status):
                        actions_taken += 1
                        # Attack performance tracked in attempt_attack method
                else:  # balanced_growth
                    # Balanced approach to all activities
                    if await self.safe_explore(status):
                        actions_taken += 1
                        self.performance_tracker.record_action("explore", True)
                    if await self.safe_build(status):
                        actions_taken += 1
                        self.performance_tracker.record_action("build", True)
                    if await self.safe_train(status):
                        actions_taken += 1
                        self.performance_tracker.record_action("train", True)
                    if await self.safe_research(status):
                        actions_taken += 1
                        self.performance_tracker.record_action("research", True)
                    
                    # Conditional military actions
                    if random.random() < 0.3:  # 30% chance
                        if await self.spy_missions(status):
                            actions_taken += 1
                            self.performance_tracker.record_action("spy", True)
                    
                    # Attack based on strategy and mood
                    attack_chance = 0.1 * self.brain.state["aggression_level"] * self.brain.state["mood_state"]["mood_factor"]
                    if random.random() < attack_chance:
                        if await self.attempt_attack(status):
                            actions_taken += 1
                            # Attack performance tracked in attempt_attack method
                
                # Take snapshots periodically
                if self.step_count % self.cfg.snapshot_every_steps == 0:
                    await self.take_snapshot()
                
                # Log status and performance periodically
                if self.step_count % 10 == 0:
                    kingdom = status.get("kingdom", {})
                    resources = kingdom.get("resources", {})
                    mood = self.brain.state["mood_state"]
                    metrics = self.performance_tracker.get_efficiency_metrics()
                    
                    log.info(f"Step {self.step_count}: Food={resources.get('food',0):,}, "
                           f"Gold={resources.get('gold',0):,}, Land={kingdom.get('land',0):,}, "
                           f"Networth={status.get('networth',0):.1f}, "
                           f"Mood={mood['current_mood']}, "
                           f"Strategy={current_strategy}, "
                           f"APH={metrics['actions_per_hour']:.1f}")
                
                # Reset consecutive failures on successful cycle
                if actions_taken > 0:
                    self.consecutive_failures = 0
                    self.last_successful_action = time.time()
                
                # Human-like pacing with intelligent delays
                if actions_taken > 0:
                    last_action_type = "general"  # Would need to track last action for more realism
                    delay = self.brain.calculate_action_delay(last_action_type, actions_taken)
                    log.debug(f"Action delay: {delay:.1f}s (mood: {self.brain.state['mood_state']['current_mood']})")
                    await asyncio.sleep(delay)
                else:
                    # Check if any timers are active and sleep accordingly
                    queues = kingdom.get("queues", {})
                    timer_sleep = 0
                    
                    # Calculate time until next action possible
                    if queues.get("building_busy"):
                        building_end = queues.get("building_end", 0)
                        timer_sleep = max(timer_sleep, building_end - time.time())
                    
                    if queues.get("training_busy"):
                        training_end = queues.get("training_end", 0)
                        timer_sleep = max(timer_sleep, training_end - time.time())
                        
                    if queues.get("explore_busy"):
                        explore_end = queues.get("explore_end", 0)
                        timer_sleep = max(timer_sleep, explore_end - time.time())
                    
                    if timer_sleep > 0:
                        log.info(f"All actions on cooldown - sleeping {timer_sleep/60:.1f} minutes until timers complete")
                        await asyncio.sleep(min(timer_sleep, 1800))  # Cap at 30 minutes
                    else:
                        # This is a slow game - wait for resource generation
                        resources = kingdom.get("resources", {})
                        gold = resources.get("gold", 0)
                        food = resources.get("food", 0)
                        wood = resources.get("wood", 0)
                        stone = resources.get("stone", 0)
                        
                        # Be very patient - wait for resources to accumulate
                        if gold < 5000 or food < 5000:
                            resource_wait = random.uniform(900, 1800)  # 15-30 minutes
                            log.info(f"Low resources (Gold:{gold}, Food:{food}) - waiting {resource_wait/60:.1f} minutes for resource generation")
                            await asyncio.sleep(resource_wait)
                        else:
                            # Standard idle wait when resources are good but no actions possible
                            idle_delay = random.uniform(300, 600)  # 5-10 minutes for slow game
                            log.info(f"Resources sufficient but no actions possible - idle wait {idle_delay/60:.1f} minutes")
                            await asyncio.sleep(idle_delay)
                
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                self.consecutive_failures += 1
                
                # Implement failure recovery with exponential backoff
                if self.consecutive_failures >= 3:
                    self.recovery_mode = True
                    recovery_delay = min(600, 60 * (2 ** (self.consecutive_failures - 3)))  # Cap at 10 minutes
                    log.warning(f"Entering recovery mode. Waiting {recovery_delay/60:.1f} minutes")
                    await asyncio.sleep(recovery_delay)
                else:
                    await asyncio.sleep(60)  # Standard retry delay
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        log.info("Shutting down bot...")
        self.running = False
        self.brain.save_state()
        if hasattr(self.client, 'aclose') and hasattr(self.client, '_client') and self.client._client:
            await self.client.aclose()
    
    def _assess_recent_performance(self) -> bool:
        """Assess if recent performance has been positive"""
        learning_data = self.brain.state["learning_data"]
        recent_successes = learning_data["successful_attacks"]
        recent_failures = learning_data["failed_attacks"]
        
        # Consider successful if more wins than losses, or if we haven't attacked recently
        if recent_successes + recent_failures == 0:
            return True  # No recent attacks, assume neutral/positive
        
        return recent_successes >= recent_failures
    
    def _assess_resource_status(self, resources: Dict[str, int]) -> str:
        """Assess current resource situation"""
        food = resources.get("food", 0)
        gold = resources.get("gold", 0)
        wood = resources.get("wood", 0)
        stone = resources.get("stone", 0)
        
        critical_count = sum(1 for r in [food, gold, wood, stone] if r < 1000)
        low_count = sum(1 for r in [food, gold, wood, stone] if r < 5000)
        
        if critical_count > 0:
            return "critical"
        elif low_count >= 2:
            return "low"
        elif all(r > 10000 for r in [food, gold, wood, stone]):
            return "abundant"
        else:
            return "stable"
    
    async def handle_emergency(self, status: Dict[str, Any], emergency_action: str):
        """Handle emergency situations with specific recovery actions"""
        log.warning(f"Executing emergency action: {emergency_action}")
        
        if emergency_action == "focus_food_production":
            # Build farms urgently
            await self.safe_build(status, focus="farms_only")
            # Lower tax rate to help peasants
            await self.optimize_taxation(status, emergency=True)
        
        elif emergency_action == "optimize_taxation":
            # Adjust tax rate for maximum income
            await self.optimize_taxation(status, emergency=True)
        
        elif emergency_action == "defensive_posture":
            # Stop all attacks, focus on defense and honor recovery
            log.info("Entering defensive posture - no attacks until honor recovers")
            
            # Still do productive actions during defensive posture
            await self.manage_economy(status)
            
            # Try to improve honor through non-aggressive actions
            kingdom = status.get("kingdom", {})
            if kingdom.get("honor", 0) < 10:
                log.info("Low honor detected - focusing on economic development")
                # Build more farms for stable food income
                buildings = kingdom.get("buildings", {})
                if buildings.get("farm", 0) < 30:
                    await self.client.build_structure("farm", 1)
                # Build more houses for population growth  
                if buildings.get("house", 0) < 40:
                    await self.client.build_structure("house", 1)
            
        elif emergency_action == "stabilize_economy":
            # Build basic economic structures
            await self.safe_build(status, focus="economy")
    
    async def optimize_taxation(self, status: Dict[str, Any], emergency: bool = False) -> bool:
        """Optimize tax rate based on current situation"""
        try:
            kingdom = status.get("kingdom", {})
            current_tax = kingdom.get("tax_rate", 24)
            peasants = kingdom.get("peasants", 0)
            resources = kingdom.get("resources", {})
            
            optimal_tax = current_tax
            
            if emergency or resources.get("gold", 0) < 500:
                # Emergency: maximize gold income
                optimal_tax = min(26, current_tax + 2)
            elif peasants < 100:
                # Low population: reduce tax
                optimal_tax = max(10, current_tax - 2)
            elif peasants > 1000 and resources.get("gold", 0) > 10000:
                # High population and resources: can afford higher tax
                optimal_tax = min(25, current_tax + 1)
            
            if optimal_tax != current_tax:
                # Use the client to set tax rate (implement based on API)
                if hasattr(self.client, 'set_tax_rate'):
                    result = await self.client.set_tax_rate(optimal_tax)
                    if result.get("success"):
                        log.info(f"Tax rate adjusted from {current_tax} to {optimal_tax}")
                        return True
                else:
                    log.debug("Tax rate optimization not available in this client")
            
            return False
            
        except Exception as e:
            log.error(f"Error optimizing taxation: {e}")
            return False

# ------------------------
# CLI Interface
# ------------------------

async def status_report(cfg: Config):
    """Generate a status report"""
    bot = KG2Bot(cfg)
    await bot.initialize()
    
    try:
        status = await bot.get_status()
        kingdom = status.get("kingdom", {})
        resources = kingdom.get("resources", {})
        units = kingdom.get("units", {})
        buildings = kingdom.get("buildings", {})
        
        print("=" * 60)
        print("KG2 AI Bot Status Report")
        print("=" * 60)
        print(f"Kingdom: {kingdom.get('name', 'Unknown')}")
        print(f"Land: {kingdom.get('land', 0)}")
        print(f"Networth: {status.get('networth', 0):.2f}")
        print(f"Honor: {kingdom.get('honor', 0)}")
        print()
        
        print("Resources:")
        for resource, amount in resources.items():
            print(f"  {resource.capitalize()}: {amount:,}")
        print()
        
        print("Military Units:")
        for unit, count in units.items():
            if count > 0:
                print(f"  {unit.capitalize()}: {count:,}")
        print()
        
        print("Buildings:")
        for building, count in buildings.items():
            if count > 0:
                print(f"  {building.capitalize()}: {count:,}")
        print()
        
        print(f"Attack Power: {status.get('ap', 0)}")
        print()
        
        # Show AI brain state
        print("AI Brain State:")
        brain_state = bot.brain.state
        print(f"  Risk Tolerance: {brain_state['risk_tolerance']:.2f}")
        print(f"  Aggression Level: {brain_state['aggression_level']:.2f}")
        print(f"  Economic Focus: {brain_state['economic_focus']:.2f}")
        
        learning_data = brain_state['learning_data']
        print(f"  Successful Attacks: {learning_data['successful_attacks']}")
        print(f"  Failed Attacks: {learning_data['failed_attacks']}")
        print("=" * 60)
        
    finally:
        await bot.shutdown()

async def main():
    parser = argparse.ArgumentParser(description="KG2 AI Bot - Fully Automated Kingdom Game Bot")
    parser.add_argument("--status", action="store_true", help="Show status report and exit")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for testing")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--monitor", action="store_true", help="Run in monitoring mode (show performance metrics)")
    
    args = parser.parse_args()
    
    cfg = Config.from_env()
    
    # Override mock mode if specified via command line
    if args.mock:
        cfg.mock_mode = True
    
    if not cfg.enabled:
        log.info("Bot is disabled (KG2_ENABLED=false)")
        return
    
    if args.status:
        await status_report(cfg)
        return
    
    if args.monitor:
        await monitoring_mode(cfg)
        return
    
    bot = KG2Bot(cfg)
    
    try:
        await bot.initialize()
        await bot.main_loop()
    except KeyboardInterrupt:
        log.info("Received interrupt signal")
    finally:
        await bot.shutdown()

async def monitoring_mode(cfg: Config):
    """Run in monitoring mode to show live performance metrics"""
    bot = KG2Bot(cfg)
    await bot.initialize()
    
    try:
        print("=" * 80)
        print("KG2 AI Bot - Live Monitoring Mode")
        print("Press Ctrl+C to exit")
        print("=" * 80)
        
        while True:
            try:
                status = await bot.get_status()
                kingdom = status.get("kingdom", {})
                resources = kingdom.get("resources", {})
                metrics = bot.performance_tracker.get_efficiency_metrics()
                brain_state = bot.brain.state
                
                print(f"\n📊 Live Dashboard - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)
                
                print(f"🏰 Kingdom: {kingdom.get('name', 'Unknown')} | Land: {kingdom.get('land', 0):,} | Steps: {bot.step_count}")
                print(f"💰 Networth: {status.get('networth', 0):.2f} | Honor: {kingdom.get('honor', 0)}")
                
                print(f"\n🛡️ Military: AP={status.get('ap', 0)} | Strategy: {bot.brain.get_optimal_strategy(status)}")
                print(f"🧠 Mood: {brain_state['mood_state']['current_mood'].title()} | "
                      f"Aggression: {brain_state['aggression_level']:.2f} | "
                      f"Risk: {brain_state['risk_tolerance']:.2f}")
                
                print(f"\n📈 Performance:")
                print(f"   Actions/Hour: {metrics['actions_per_hour']:.1f}")
                print(f"   Win Rate: {metrics['win_rate']:.1%}")
                print(f"   Resource Gain/Hour: {metrics['resource_gain_rate']:.0f}")
                
                print(f"\n💎 Resources:")
                for resource, amount in resources.items():
                    if resource in ['food', 'gold', 'wood', 'stone']:
                        storage_max = kingdom.get('storage', {}).get(resource, 0)
                        percentage = (amount / max(storage_max, 1)) * 100
                        print(f"   {resource.capitalize()}: {amount:,} ({percentage:.1f}% of storage)")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring: {e}")
                await asyncio.sleep(5)
    
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
