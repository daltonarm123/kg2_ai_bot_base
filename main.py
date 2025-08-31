# -*- coding: utf-8 -*-
"""
KG2 AI Bot - single file main.py
- Live + mock modes
- Render-ready (env secrets)
- Early-game: EXPLORE HARD using /WebService/Kingdoms.asmx/Explore
"""

import os, sys, json, asyncio, logging, math, random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import httpx

# -----------------------
# Config / Environment
# -----------------------

def env_bool(key: str, default: bool=False) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

@dataclass
class Config:
    base_url: str = os.getenv("KG2_BASE_URL", "https://www.kingdomgame.net")
    world_id: str = os.getenv("KG2_WORLD_ID", "1")
    account_id: str = os.getenv("KG2_ACCOUNT_ID", "")
    token: str = os.getenv("KG2_TOKEN", "")
    kingdom_id: int = int(os.getenv("KG2_KINGDOM_ID", "0"))

    referer_overview: str = os.getenv("KG2_REFERER_OVERVIEW", "https://www.kingdomgame.net/overview")
    referer_buildings: str = os.getenv("KG2_REFERER_BUILDINGS", "https://www.kingdomgame.net/buildings")
    referer_war: str = os.getenv("KG2_REFERER_WAR", "https://www.kingdomgame.net/warroom")
    referer_research: str = os.getenv("KG2_REFERER_RESEARCH", "https://www.kingdomgame.net/research")

    mock: bool = env_bool("KG2_MOCK", False)
    max_rps: float = float(os.getenv("KG2_MAX_RPS", "0.4"))
    timeout_s: int = int(os.getenv("KG2_TIMEOUT_S", "20"))

    origin_url: str = "https://www.kingdomgame.net"

# -----------------------
# Logging
# -----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("kg2bot")

# -----------------------
# HTTP Client
# -----------------------

class KG2Client:
    TROOP_ID = {
        "peasants": 1,
        "footmen": 17,
        "pikemen": 18,
        "archers": 20,
        "crossbow": 22,
        "light_cav": 23,
        "heavy_cav": 24,
        "knights": 25,
        "spies": 5,
        "priests": 6,
        "diplomats": 30,
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            follow_redirects=True,
            timeout=self.cfg.timeout_s,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/122 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
            }
        )

    async def close(self):
        await self._client.aclose()

    async def warmup(self):
        # get a page to establish cookies/session
        r = await self._client.get(
            "/overview",
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://www.kingdomgame.net/",
                "Origin": self.cfg.origin_url,
            }
        )
        # do not raise; some hosts mask with 500 on warmup — not fatal here
        logger.info("Warmup GET /overview -> %s", r.status_code)

    async def post_asmx(self, path: str, body: dict, referer: str) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "world-id": str(self.cfg.world_id),
            "Origin": self.cfg.origin_url,
            "Referer": referer,
            "User-Agent": self._client.headers.get("User-Agent"),
        }
        resp = await self._client.post(path, json=body, headers=headers)
        if resp.status_code in (401, 403):
            raise RuntimeError(f"AUTH_FAIL {resp.status_code} at {path}. Check token/ids/referer/world-id.")
        if resp.status_code == 500 and resp.headers.get("Content-Type", "").startswith("text/html"):
            snippet = resp.text[:300].replace("\n", " ")
            raise RuntimeError(f"AUTH_FAIL 500-HTML at {path}. Likely bad/expired token or IP blocked. Snip: {snippet}")
        resp.raise_for_status()
        data = resp.json()
        # unwrap ASP.NET JSON with {"d": "...json..."} if present
        if isinstance(data, dict) and "d" in data and isinstance(data["d"], str):
            try:
                return json.loads(data["d"])
            except Exception:
                return data
        return data

    # ---- Kingdoms / Overview ----
    async def get_details(self) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomDetails", body, self.cfg.referer_overview)

    async def get_resources(self) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomResources", body, self.cfg.referer_overview)

    async def get_population(self) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomPopulation", body, self.cfg.referer_war)

    async def get_buildings(self) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomBuildings", body, self.cfg.referer_buildings)

    # ---- Buildings (build) ----
    async def build(self, buildingTypeId: str, quantity: int) -> dict:
        body = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
            "buildingTypeId": str(buildingTypeId),
            "quantity": int(quantity),
        }
        return await self.post_asmx("/WebService/Buildings.asmx/BuildBuilding", body, self.cfg.referer_buildings)

    # ---- Research ----
    async def get_skills(self) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Research.asmx/GetSkills", body, self.cfg.referer_research)

    async def get_training_skills(self) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Research.asmx/GetTrainingSkills", body, self.cfg.referer_research)

    async def train_skill(self, skillTypeId: str) -> dict:
        base = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        referer = self.cfg.referer_research
        attempts = [
            ("/WebService/Research.asmx/TrainSkill", base | {"skillTypeId": str(skillTypeId)}),
            ("/WebService/Research.asmx/StartSkillTraining", base | {"skillTypeId": str(skillTypeId)}),
            ("/WebService/Research.asmx/TrainResearch", base | {"researchTypeId": str(skillTypeId)}),
            ("/WebService/Research.asmx/StartResearch", base | {"researchTypeId": str(skillTypeId)}),
        ]
        for path, body in attempts:
            try:
                d = await self.post_asmx(path, body, referer)
                if d.get("ReturnValue") == 1 or d.get("success"):
                    return d
                if "ReturnString" in d:
                    return d
            except Exception:
                continue
        return {"ReturnValue": 0, "ReturnString": "No research train endpoint accepted the request"}

    # ---- Explore (exact shape with troops JSON-string) ----
    async def explore(self, send: Dict[str, int]) -> dict:
        """
        POST /WebService/Kingdoms.asmx/Explore
        Body: { accountId, token, kingdomId, troops: "[{TroopTypeID,AmountToSend}, ...]" }
        Referer: /warroom
        """
        arr = []
        for k, n in send.items():
            if not n: continue
            tid = self.TROOP_ID.get(k)
            if not tid: continue
            arr.append({"TroopTypeID": tid, "AmountToSend": int(n)})

        body = {
            "accountId": self.cfg.account_id,
            "token": self.cfg.token,
            "kingdomId": self.cfg.kingdom_id,
            "troops": json.dumps(arr),  # IMPORTANT: stringified array
        }
        return await self.post_asmx("/WebService/Kingdoms.asmx/Explore", body, self.cfg.referer_war)

# -----------------------
# State & Normalizers
# -----------------------

@dataclass
class KingdomState:
    id: int = 0
    name: str = ""
    tax_rate: int = 24
    land: int = 0

    resources: Dict[str, int] = field(default_factory=dict)
    storage: Dict[str, int] = field(default_factory=dict)
    prod_per_hour: Dict[str, int] = field(default_factory=dict)

    units: Dict[str, int] = field(default_factory=dict)
    buildings: Dict[str, int] = field(default_factory=dict)
    research: Dict[str, Any] = field(default_factory=dict)
    queues: Dict[str, Any] = field(default_factory=dict)
    season: str = ""

def _normalize_resources(res_api: dict) -> Tuple[dict, dict, dict, dict]:
    name_map = {
        "food": "food", "gold": "gold", "wood": "wood", "stone": "stone",
        "mana": "mana", "land": "land", "blue gems": "blue_gems",
    }
    resources, storage, prod = {}, {}, {}
    flags = {"maintenance_issue_wood": False, "maintenance_issue_stone": False}
    for r in (res_api or {}).get("resources", []):
        n = (r.get("name") or "").lower().strip()
        key = name_map.get(n)
        if not key: continue
        resources[key] = int(r.get("amount", 0))
        storage[key] = int(r.get("capacity", 0))
        prod[key] = int(r.get("productionPerHour", 0))
        if n == "wood" and r.get("maintenanceIssue"): flags["maintenance_issue_wood"] = True
        if n == "stone" and r.get("maintenanceIssue"): flags["maintenance_issue_stone"] = True
    for k in name_map.values():
        resources.setdefault(k, 0); storage.setdefault(k, 0); prod.setdefault(k, 0)
    return resources, storage, prod, flags

def _normalize_population(pop_api: dict) -> Tuple[dict, dict, dict]:
    key_map = {
        "peasants": "peasants", "footmen": "footmen", "pikemen": "pikemen",
        "archers": "archers", "crossbowmen": "crossbow",
        "light cavalry": "light_cav", "heavy cavalry": "heavy_cav",
        "knights": "knights", "elites": "elites", "spies": "spies",
        "priests": "priests", "diplomats": "diplomats", "market wagons": "wagons",
    }
    units, inprog, returning = {}, {}, {}
    for r in (pop_api or {}).get("population", []):
        name = (r.get("name") or "").lower().strip()
        key = key_map.get(name)
        if not key: continue
        units[key] = int(r.get("amount", 0))
        inprog[key] = int(r.get("amountInProgress", 0))
        returning[key] = int(r.get("amountReturning", 0))
    for k in key_map.values():
        units.setdefault(k, 0); inprog.setdefault(k, 0); returning.setdefault(k, 0)
    return units, inprog, returning

def _normalize_buildings(b_api: dict) -> Tuple[dict, dict, dict]:
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
        if not key: continue
        buildings[key] = int(r.get("amount", 0))
        inprog[key] = int(r.get("amountInProgress", 0))
        canb[key] = bool(r.get("canBuild", False))
    for k in ("houses","farms","lumber","quarries","barracks","stables","archery","barns","markets","temples","castles"):
        buildings.setdefault(k, 0); inprog.setdefault(k, 0); canb.setdefault(k, True)
    return buildings, inprog, canb

def normalize_skills(sk_json: dict) -> List[dict]:
    rows = sk_json.get("skills") or sk_json.get("items") or sk_json.get("research") or []
    out = []
    for r in rows:
        rid = r.get("id") or r.get("skillTypeId") or r.get("researchTypeId")
        name = (r.get("name") or r.get("displayName") or "").strip()
        cat = (r.get("category") or r.get("group") or r.get("tree") or "").strip()
        lvl = int(r.get("currentLevel") or r.get("level") or 0)
        maxl = int(r.get("maxLevel") or r.get("maximumLevel") or 0)
        gold = int(r.get("goldCost") or r.get("gold") or 0)
        gems = int(r.get("gemCost") or r.get("gems") or 0)
        time = (r.get("timeNext") or r.get("researchNextLevel") or r.get("duration") or "").strip()
        can = bool(r.get("canTrain", True))
        preq = r.get("prerequisites") or r.get("prereqText") or ""
        if isinstance(preq, list):
            preq = ", ".join(str(x.get("name") if isinstance(x, dict) else x) for x in preq)
        out.append({
            "id": str(rid) if rid is not None else "",
            "name": name, "category": cat,
            "currentLevel": lvl, "maxLevel": maxl,
            "goldCost": gold, "gemCost": gems,
            "timeText": time, "canTrain": can, "prereqText": str(preq),
        })
    return out

def normalize_training_skills(ts_json: dict) -> List[dict]:
    rows = ts_json.get("trainingSkills") or ts_json.get("skills") or ts_json.get("items") or []
    out = []
    for r in rows:
        rid = r.get("id") or r.get("skillTypeId") or r.get("researchTypeId")
        name = (r.get("name") or "").strip()
        remaining = (r.get("timeRemainingText") or r.get("timeRemaining") or r.get("etaText") or "").strip()
        level_to = int(r.get("targetLevel") or r.get("nextLevel") or (r.get("currentLevel") or 0) + 1)
        out.append({"id": str(rid) if rid is not None else "", "name": name, "eta": remaining, "targetLevel": level_to})
    return out

# -----------------------
# Bot
# -----------------------

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = KG2Client(cfg)
        self.logger = logger

    async def fetch_state(self) -> KingdomState:
        kd = await self.client.get_details()
        rjs = await self.client.get_resources()
        pjs = await self.client.get_population()
        bjs = await self.client.get_buildings()
        sk = await self.client.get_skills()
        ts = await self.client.get_training_skills()

        resources, storage, prod, flags = _normalize_resources(rjs)
        units, inprog_units, returning = _normalize_population(pjs)
        buildings, inprog_build, can_build = _normalize_buildings(bjs)
        skills = normalize_skills(sk)
        training = normalize_training_skills(ts)

        st = KingdomState(
            id=int(kd.get("id", 0)),
            name=kd.get("name", ""),
            tax_rate=int(kd.get("taxRate", 24)),
            land=int(kd.get("land", resources.get("land", 0))),
            resources={"food": resources["food"], "wood": resources["wood"], "stone": resources["stone"], "gold": resources["gold"]},
            storage={"food": storage["food"], "wood": storage["wood"], "stone": storage["stone"], "gold": storage["gold"]},
            prod_per_hour=prod,
            units=units,
            buildings=buildings,
            research={"skills": skills, "in_progress": training},
            queues={
                "training_busy": any(v > 0 for v in inprog_units.values()),
                "building_busy": any(v > 0 for v in inprog_build.values()),
                "explore_busy": False,  # set true locally when we send
                "training_inprog": inprog_units,
                "returning": returning,
                "building_inprog": inprog_build,
                "can_build": can_build,
                "maintenance_flags": flags,
            },
            season=(kd.get("seasonName") or "").strip(),
        )
        return st

    # ------------ Strategy helpers ------------

    def is_early_game(self, st: KingdomState) -> bool:
        return st.land < 5000 or sum(st.buildings.values()) < 200

    def idle_units(self, st: KingdomState) -> Dict[str, int]:
        u = st.units or {}
        return {
            "peasants":  u.get("peasants", 0),
            "footmen":   u.get("footmen", 0),
            "pikemen":   u.get("pikemen", 0),
            "archers":   u.get("archers", 0),
            "light_cav": u.get("light_cav", 0),
            "heavy_cav": u.get("heavy_cav", 0),
            "knights":   u.get("knights", 0),
        }

    async def plan_exploration(self, st: KingdomState) -> str:
        if st.queues.get("explore_busy"):
            return "Explore: busy"
        keep = {"peasants": 20, "footmen": 10, "pikemen": 10, "archers": 10, "light_cav": 0, "heavy_cav": 0, "knights": 0}
        idl = self.idle_units(st)
        pct = 0.85 if self.is_early_game(st) else 0.4
        to_send = {}
        for k in ("peasants","footmen","pikemen","archers","light_cav","heavy_cav","knights"):
            avail = max(0, idl.get(k, 0) - keep.get(k, 0))
            to_send[k] = int(avail * pct)
        if sum(to_send.values()) < 25 and idl.get("peasants", 0) > keep["peasants"]:
            to_send["peasants"] = max(to_send["peasants"], min(50, idl["peasants"] - keep["peasants"]))
        if st.resources.get("gold", 0) < 50:
            return "Explore: waiting for gold ≥ 50"
        res = await self.client.explore(to_send)
        if str(res.get("ReturnValue")) == "1":
            st.queues["explore_busy"] = True
            return f"Explore sent: {to_send}"
        return f"Explore failed: {res.get('ReturnString','unknown')}"

    async def maybe_fix_storage(self, st: KingdomState) -> Optional[str]:
        msgs = []
        # If over capacity, prioritize storage producers next
        if st.resources["wood"] >= st.storage["wood"]:
            if st.queues["can_build"].get("lumber", True) and st.queues["building_inprog"].get("lumber", 0) == 0:
                d = await self.client.build(buildingTypeId="9", quantity=1)  # Lumber Yards
                msgs.append(f"Build Lumber Yard -> {d.get('ReturnString','OK')}")
        if st.resources["stone"] >= st.storage["stone"]:
            if st.queues["can_build"].get("quarries", True) and st.queues["building_inprog"].get("quarries", 0) == 0:
                d = await self.client.build(buildingTypeId="6", quantity=1)  # Stone Quarries
                msgs.append(f"Build Stone Quarry -> {d.get('ReturnString','OK')}")
        if msgs:
            return " | ".join(msgs)
        return None

    def _find_skill_id(self, skills: List[dict], name_like: str) -> Optional[str]:
        key = name_like.lower().replace(" ", "")
        for s in skills:
            if s["name"].lower().replace(" ", "") == key:
                return s["id"] or None
        for s in skills:
            if key in s["name"].lower().replace(" ", ""):
                return s["id"] or None
        return None

    async def maybe_research(self, st: KingdomState) -> str:
        if (st.research.get("in_progress") or []):
            return "Research in progress"
        skills = st.research.get("skills") or []
        pick = None
        # economy first
        if st.resources["food"] < max(10000, st.storage["food"] * 0.25) or st.prod_per_hour.get("food", 0) < 3000:
            for n in ("Better Farming Methods","Irrigation","Mathematics"):
                sid = self._find_skill_id(skills, n)
                if sid: pick = (n, sid); break
        if pick is None and st.resources["gold"] < max(10000, st.storage["gold"] * 0.2):
            for n in ("Mathematics","Accounting"):
                sid = self._find_skill_id(skills, n)
                if sid: pick = (n, sid); break
        if pick is None:
            for n in ("Engineering","Improved Tools","Better Construction Methods"):
                sid = self._find_skill_id(skills, n)
                if sid: pick = (n, sid); break
        if pick is None:
            return "No suitable research"
        name, sid = pick
        row = next((s for s in skills if s.get("id") == sid), {})
        gcost = int(row.get("goldCost") or 0)
        if gcost and st.resources["gold"] < gcost:
            return f"Research '{name}': need {gcost} gold"
        res = await self.client.train_skill(sid)
        return res.get("ReturnString", f"Research queued: {name}")

    async def step(self) -> None:
        st = await self.fetch_state()

        self.logger.info("Current strategy: %s", "economic_focus")
        # capacity warnings
        if st.resources["wood"] >= st.storage["wood"]:
            self.logger.warning("Wood over-capacity: %s/%s - need more lumber mills", st.resources["wood"], st.storage["wood"])
        if st.resources["stone"] >= st.storage["stone"]:
            self.logger.warning("Stone over-capacity: %s/%s - need more quarries", st.resources["stone"], st.storage["stone"])

        # Early-game explore-first
        if self.is_early_game(st):
            self.logger.info("EARLY EXPLORE | land=%s", st.land)
            msg = await self.plan_exploration(st)
            self.logger.info(msg)

        # Storage fixes
        if not st.queues.get("building_busy"):
            m = await self.maybe_fix_storage(st)
            if m:
                self.logger.info(m)

        # Research
        rmsg = await self.maybe_research(st)
        self.logger.info(rmsg)

        # Idle wait based on weakest resource
        gold_hr = st.prod_per_hour.get("gold", 0)
        food_hr = st.prod_per_hour.get("food", 0)
        wait_min = 5.0
        if gold_hr > 0:
            # try to wait until we have >= 10k gold or a minimum of 5 minutes
            deficit = max(0, 10000 - st.resources["gold"])
            if deficit > 0:
                wait_min = max(5.0, min(30.0, deficit / max(1.0, gold_hr) * 60.0 / 60.0))
        self.logger.info("Waiting %.1f minutes for resources...", wait_min)
        await asyncio.sleep(wait_min * 60)

    async def run(self):
        self.logger.info("KG2 AI Bot initializing...")
        self.logger.info("Mode: %s", "Mock" if self.cfg.mock else "Live")
        self.logger.info("Kingdom ID: %s", self.cfg.kingdom_id)
        if not self.cfg.mock:
            await self.client.warmup()
        self.logger.info("Starting main bot loop...")
        try:
            while True:
                try:
                    await self.step()
                except Exception as e:
                    self.logger.error("Step error: %s", e)
                    # backoff a bit on errors
                    await asyncio.sleep(30)
        finally:
            await self.client.close()

# -----------------------
# Entrypoint
# -----------------------

if __name__ == "__main__":
    cfg = Config()
    # quick validation
    if not cfg.mock and (not cfg.account_id or not cfg.token or not cfg.kingdom_id):
        logger.error("Missing live auth env (KG2_ACCOUNT_ID / KG2_TOKEN / KG2_KINGDOM_ID). Set KG2_MOCK=true to run sim.")
        sys.exit(1)
    asyncio.run(Bot(cfg).run())
