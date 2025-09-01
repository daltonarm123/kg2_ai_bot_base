# -*- coding: utf-8 -*-
"""
KG2 AI Bot - main.py
Auth-hardening version:
- Warmup GET for each ASMX call with the exact Referer page
- Adds X-Requested-With and browser-like headers
- Logs richer diagnostics on 403/500
Other features:
- SQLite "brain" (events + learning)
- Adaptive exploration with early-game peasants (toggle)
- Tick-aware sleep + human-like windowing
- Research + storage fixes
- Messages inbox handling (tutorial/system)
"""

import os, sys, json, asyncio, logging, random, sqlite3, re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
try:
    import zoneinfo
except Exception:
    zoneinfo = None
import httpx
from urllib.parse import urlsplit

# -----------------------
# Time helpers
# -----------------------

def local_now(tz_name: str = None) -> datetime:
    tz = None
    if tz_name and zoneinfo:
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            tz = None
    return datetime.now(tz or timezone.utc)

def in_human_sleep_window(tz_name: str = "America/Chicago") -> bool:
    now = local_now(tz_name); h = now.hour
    return 1 <= h < 6

def minutes_until(minute_past_hour: int, tz_name: str) -> float:
    now = local_now(tz_name)
    target = now.replace(minute=minute_past_hour, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(hours=1)
    return (target - now).total_seconds() / 60.0

# -----------------------
# Config / Environment
# -----------------------

def env_bool(key: str, default: bool=False) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

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
    referer_messages: str = os.getenv("KG2_REFERER_MESSAGES", "https://www.kingdomgame.net/messages")

    mock: bool = env_bool("KG2_MOCK", False)
    max_rps: float = float(os.getenv("KG2_MAX_RPS", "0.4"))
    timeout_s: int = int(os.getenv("KG2_TIMEOUT_S", "25"))
    origin_url: str = "https://www.kingdomgame.net"
    tz_name: str = os.getenv("KG2_TZ", "America/Chicago")

    allow_peasant_explore: bool = env_bool("KG2_ALLOW_PEASANT_EXPLORE", False)
    tick_minute: int = int(os.getenv("KG2_TICK_MINUTE", "2"))
    threat_window_hrs: int = int(os.getenv("KG2_THREAT_WINDOW_HRS", "48"))

    messages_enabled: bool = env_bool("KG2_MESSAGES_ENABLED", True)

# -----------------------
# Logging
# -----------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kg2bot")

# -----------------------
# SQLite Memory ("brain")
# -----------------------

class Brain:
    def __init__(self, path="bot_memory.sqlite", tz="America/Chicago"):
        self.tz = tz
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._migrate()

    def _migrate(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS events_explore(
            id INTEGER PRIMARY KEY, ts TEXT, sent_json TEXT, result TEXT, land_before INT, land_after INT
        );
        CREATE TABLE IF NOT EXISTS events_attack_in(
            id INTEGER PRIMARY KEY, ts TEXT, from_kingdom INT, ap_est INT, comp_json TEXT, result TEXT
        );
        CREATE TABLE IF NOT EXISTS events_attack_out(
            id INTEGER PRIMARY KEY, ts TEXT, to_kingdom INT, our_force_json TEXT, result TEXT, loot_json TEXT
        );
        CREATE TABLE IF NOT EXISTS spy_reports(
            id INTEGER PRIMARY KEY, ts TEXT, target INT, payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS gem_usage(
            id INTEGER PRIMARY KEY, ts TEXT, context TEXT, amount INT, note TEXT
        );
        CREATE TABLE IF NOT EXISTS tick_log(
            id INTEGER PRIMARY KEY, ts TEXT, minute INT
        );
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY, ts TEXT, message_id INT, subject TEXT, sender TEXT, body TEXT, acted INTEGER DEFAULT 0
        );
        """)

    def _nowstr(self):
        return local_now(self.tz).isoformat(timespec="seconds")

    def log_explore(self, sent, result, land_before=None, land_after=None):
        self.conn.execute(
            "INSERT INTO events_explore(ts,sent_json,result,land_before,land_after) VALUES(?,?,?,?,?)",
            (self._nowstr(), json.dumps(sent), str(result), land_before, land_after))
    def log_attack_in(self, from_k, ap_est=None, comp=None, result=None):
        self.conn.execute(
            "INSERT INTO events_attack_in(ts,from_kingdom,ap_est,comp_json,result) VALUES(?,?,?,?,?)",
            (self._nowstr(), from_k, ap_est, json.dumps(comp or {}), str(result or "")))
    def log_attack_out(self, to_k, our_force=None, result=None, loot=None):
        self.conn.execute(
            "INSERT INTO events_attack_out(ts,to_kingdom,our_force_json,result,loot_json) VALUES(?,?,?,?,?)",
            (self._nowstr(), to_k, json.dumps(our_force or {}), str(result or ""), json.dumps(loot or {})))
    def log_spy(self, target, payload):
        self.conn.execute(
            "INSERT INTO spy_reports(ts,target,payload_json) VALUES(?,?,?)",
            (self._nowstr(), target, json.dumps(payload or {})))
    def log_gem(self, context, amount, note=""):
        self.conn.execute(
            "INSERT INTO gem_usage(ts,context,amount,note) VALUES(?,?,?,?)",
            (self._nowstr(), str(context), int(amount or 0), str(note or "")))
    def log_tick(self, minute):
        self.conn.execute("INSERT INTO tick_log(ts,minute) VALUES(?,?)", (self._nowstr(), int(minute)))
    def log_message(self, message_id: int, subject: str, sender: str, body: str):
        self.conn.execute(
            "INSERT INTO messages(ts,message_id,subject,sender,body,acted) VALUES(?,?,?,?,?,0)",
            (self._nowstr(), int(message_id), subject or "", sender or "", body or ""))
    def mark_message_acted(self, message_id: int):
        self.conn.execute("UPDATE messages SET acted=1 WHERE message_id=?", (int(message_id),))
    def recent_threat_score(self, hours=48) -> float:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events_attack_in WHERE ts >= datetime('now', ?)",
                    (f'-{int(hours)} hours',))
        hits = cur.fetchone()[0] or 0
        return min(10.0, (hits / 24.0) * 10.0)
    def recent_spy_targets(self, hours=24) -> List[int]:
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT target FROM spy_reports WHERE ts >= datetime('now', ?)",
                    (f'-{int(hours)} hours',))
        return [row[0] for row in cur.fetchall()]

# -----------------------
# HTTP Client
# -----------------------

class KG2Client:
    TROOP_ID = {
        "footmen":   17, "pikemen": 18, "archers": 20, "crossbow": 22,
        "light_cav": 23, "heavy_cav": 24, "knights": 25,
        "peasants": 1,
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            follow_redirects=True,
            timeout=self.cfg.timeout_s,
            headers={
                # Browser-ish defaults
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

    async def close(self): await self._client.aclose()

    async def _warmup_for(self, referer_url: str):
        """GET the referer page before we POST to its ASMX. Sends world-id header too."""
        try:
            path = urlsplit(referer_url).path or "/"
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": self.cfg.origin_url + "/",  # site root
                "Origin": self.cfg.origin_url,
                "world-id": str(self.cfg.world_id),
            }
            r = await self._client.get(path, headers=headers)
            logging.info("Warmup GET %s -> %s", path, r.status_code)
        except Exception as e:
            logging.warning("Warmup failed for %s: %s", referer_url, e)

    async def post_asmx(self, path: str, body: dict, referer: str) -> dict:
        # Always warmup with the exact referer page first
        await self._warmup_for(referer)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "world-id": str(self.cfg.world_id),
            "Origin": self.cfg.origin_url,
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": self._client.headers.get("User-Agent"),
        }
        resp = await self._client.post(path, json=body, headers=headers)
        if resp.status_code in (401,403):
            tok = str(self.cfg.token)
            tok_masked = tok[:6] + "..." + tok[-4:] if len(tok) > 10 else "***"
            raise RuntimeError(
                f"AUTH_FAIL {resp.status_code} at {path} | referer={referer} | world={self.cfg.world_id} "
                f"| acct={self.cfg.account_id} | king={self.cfg.kingdom_id} | token={tok_masked}"
            )
        if resp.status_code == 500 and resp.headers.get("Content-Type","").startswith("text/html"):
            snippet = resp.text[:300].replace("\n"," ")
            raise RuntimeError(f"AUTH_FAIL 500-HTML at {path}. Likely bad/expired token or IP blocked. Snip: {snippet}")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "d" in data and isinstance(data["d"], str):
            try: return json.loads(data["d"])
            except Exception: return data
        return data

    # Kingdoms
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

    # Buildings
    async def build(self, buildingTypeId: str, quantity: int) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id,
                "buildingTypeId": str(buildingTypeId), "quantity": int(quantity)}
        return await self.post_asmx("/WebService/Buildings.asmx/BuildBuilding", body, self.cfg.referer_buildings)

    # Research
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
                if "ReturnString" in d: return d
            except Exception:
                continue
        return {"ReturnValue":0,"ReturnString":"No research train endpoint accepted the request"}

    # Explore
    async def explore(self, troops_array: List[dict]) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id,
                "troops": json.dumps(troops_array)}
        return await self.post_asmx("/WebService/Kingdoms.asmx/Explore", body, self.cfg.referer_war)

    # Messages
    async def get_messages_list(self) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Messages.asmx/GetMessages", body, self.cfg.referer_messages)

    async def get_message(self, message_id: int) -> dict:
        body = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id,
                "messageId": int(message_id)}
        per_msg_referer = f"https://www.kingdomgame.net/messages/message/{int(message_id)}"
        return await self.post_asmx("/WebService/Messages.asmx/GetMessage", body, per_msg_referer)

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
    name_map = {"food":"food","gold":"gold","wood":"wood","stone":"stone","mana":"mana","land":"land","blue gems":"blue_gems"}
    resources, storage, prod, flags = {}, {}, {}, {"maintenance_issue_wood": False, "maintenance_issue_stone": False}
    for r in (res_api or {}).get("resources", []):
        n = (r.get("name") or "").lower().strip(); key = name_map.get(n); 
        if not key: continue
        resources[key] = int(r.get("amount", 0)); storage[key] = int(r.get("capacity", 0)); prod[key] = int(r.get("productionPerHour", 0))
        if n=="wood" and r.get("maintenanceIssue"): flags["maintenance_issue_wood"]=True
        if n=="stone" and r.get("maintenanceIssue"): flags["maintenance_issue_stone"]=True
    for k in name_map.values(): resources.setdefault(k,0); storage.setdefault(k,0); prod.setdefault(k,0)
    return resources, storage, prod, flags

def _normalize_population(pop_api: dict) -> Tuple[dict, dict, dict]:
    key_map = {"peasants":"peasants","footmen":"footmen","pikemen":"pikemen","archers":"archers","crossbowmen":"crossbow",
               "light cavalry":"light_cav","heavy cavalry":"heavy_cav","knights":"knights","elites":"elites",
               "spies":"spies","priests":"priests","diplomats":"diplomats","market wagons":"wagons"}
    units, inprog, returning = {}, {}, {}
    for r in (pop_api or {}).get("population", []):
        name = (r.get("name") or "").lower().strip(); key = key_map.get(name); 
        if not key: continue
        units[key] = int(r.get("amount", 0)); inprog[key]=int(r.get("amountInProgress",0)); returning[key]=int(r.get("amountReturning",0))
    for k in key_map.values(): units.setdefault(k,0); inprog.setdefault(k,0); returning.setdefault(k,0)
    return units, inprog, returning

def _normalize_buildings(b_api: dict) -> Tuple[dict, dict, dict]:
    map_name = {"houses":"houses","grain farms":"farms","lumber yards":"lumber","stone quarries":"quarries","barracks":"barracks",
                "stables":"stables","archery ranges":"archery","guildhalls":"guild","temples":"temples","markets":"markets",
                "barns":"barns","castles":"castles","horse farms":"horse_farms"}
    buildings, inprog, canb = {}, {}, {}
    for r in (b_api or {}).get("buildings", []):
        name = (r.get("name") or "").lower().strip(); key = map_name.get(name); 
        if not key: continue
        buildings[key]=int(r.get("amount",0)); inprog[key]=int(r.get("amountInProgress",0)); canb[key]=bool(r.get("canBuild",False))
    for k in ("houses","farms","lumber","quarries","barracks","stables","archery","barns","markets","temples","castles"):
        buildings.setdefault(k,0); inprog.setdefault(k,0); canb.setdefault(k,True)
    return buildings, inprog, canb

def normalize_skills(sk_json: dict) -> List[dict]:
    rows = sk_json.get("skills") or sk_json.get("items") or sk_json.get("research") or []
    out=[]
    for r in rows:
        rid = r.get("id") or r.get("skillTypeId") or r.get("researchTypeId")
        name = (r.get("name") or r.get("displayName") or "").strip()
        cat = (r.get("category") or r.get("group") or r.get("tree") or "").strip()
        lvl = int(r.get("currentLevel") or r.get("level") or 0)
        maxl= int(r.get("maxLevel") or r.get("maximumLevel") or 0)
        gold= int(r.get("goldCost") or r.get("gold") or 0)
        gems= int(r.get("gemCost") or r.get("gems") or 0)
        time= (r.get("timeNext") or r.get("researchNextLevel") or r.get("duration") or "").strip()
        can = bool(r.get("canTrain", True))
        preq= r.get("prerequisites") or r.get("prereqText") or ""
        if isinstance(preq,list): preq=", ".join(str(x.get("name") if isinstance(x,dict) else x) for x in preq)
        out.append({"id":str(rid) if rid is not None else "","name":name,"category":cat,"currentLevel":lvl,"maxLevel":maxl,
                    "goldCost":gold,"gemCost":gems,"timeText":time,"canTrain":can,"prereqText":str(preq)})
    return out

def normalize_training_skills(ts_json: dict) -> List[dict]:
    rows = ts_json.get("trainingSkills") or ts_json.get("skills") or ts_json.get("items") or []
    out=[]
    for r in rows:
        rid = r.get("id") or r.get("skillTypeId") or r.get("researchTypeId")
        name = (r.get("name") or "").strip()
        remaining = (r.get("timeRemainingText") or r.get("timeRemaining") or r.get("etaText") or "").strip()
        level_to = int(r.get("targetLevel") or r.get("nextLevel") or (r.get("currentLevel") or 0) + 1)
        out.append({"id":str(rid) if rid is not None else "","name":name,"eta":remaining,"targetLevel":level_to})
    return out

# -----------------------
# Bot
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

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = KG2Client(cfg)
        self.logger = logger
        self.brain = Brain(tz=cfg.tz_name)
        self._explore_back_at: Optional[datetime] = None

    async def fetch_state(self) -> KingdomState:
        kd = await self.client.get_details()
        rjs = await self.client.get_resources()
        pjs = await self.client.get_population()
        bjs = await self.client.get_buildings()
        sk  = await self.client.get_skills()
        ts  = await self.client.get_training_skills()

        resources, storage, prod, flags = _normalize_resources(rjs)
        units, inprog_units, returning = _normalize_population(pjs)
        buildings, inprog_build, can_build = _normalize_buildings(bjs)
        skills = normalize_skills(sk)
        training = normalize_training_skills(ts)

        now = local_now(self.cfg.tz_name)
        explore_busy_local = bool(self._explore_back_at and now < self._explore_back_at)
        if self._explore_back_at and now >= self._explore_back_at:
            self._explore_back_at = None

        st = KingdomState(
            id=int(kd.get("id", 0)), name=kd.get("name",""),
            tax_rate=int(kd.get("taxRate",24)), land=int(kd.get("land", resources.get("land",0))),
            resources={"food":resources["food"],"wood":resources["wood"],"stone":resources["stone"],"gold":resources["gold"]},
            storage={"food":storage["food"],"wood":storage["wood"],"stone":storage["stone"],"gold":storage["gold"]},
            prod_per_hour=prod, units=units, buildings=buildings,
            research={"skills":skills, "in_progress":training},
            queues={"training_busy": any(v>0 for v in inprog_units.values()),
                    "building_busy": any(v>0 for v in inprog_build.values()),
                    "explore_busy": explore_busy_local,
                    "training_inprog": inprog_units, "returning": returning,
                    "building_inprog": inprog_build, "can_build": can_build,
                    "maintenance_flags": flags},
            season=(kd.get("seasonName") or "").strip(),
        )
        return st

    def is_early_game(self, st: KingdomState) -> bool:
        return st.land < 5000 or sum(st.buildings.values()) < 200

    def threat_level(self) -> float:
        return self.brain.recent_threat_score(hours=self.cfg.threat_window_hrs)

    def idle_units(self, st: KingdomState) -> Dict[str,int]:
        u = st.units or {}
        return {"peasants":u.get("peasants",0),"footmen":u.get("footmen",0),"pikemen":u.get("pikemen",0),
                "archers":u.get("archers",0),"light_cav":u.get("light_cav",0),
                "heavy_cav":u.get("heavy_cav",0),"knights":u.get("knights",0)}

    def _plan_explore_payload(self, st: KingdomState, danger: float) -> List[dict]:
        keep = {"peasants": 30 if danger>=3 else 20, "footmen": max(10,int(5+danger*2)), "pikemen":10, "archers":10,
                "light_cav": 0 if danger<5 else 5, "heavy_cav": 0 if danger<7 else 5, "knights": 0 if danger<8 else 2}
        idl = self.idle_units(st)
        pct = 0.85 if self.is_early_game(st) and danger < 3 else (0.4 if danger < 5 else 0.2)

        plan = {}
        for k in ("peasants","footmen","pikemen","archers","light_cav","heavy_cav","knights"):
            avail = max(0, idl.get(k,0) - keep.get(k,0))
            plan[k] = int(avail * pct)

        if sum(plan.values()) < 15 and idl.get("peasants",0) > keep["peasants"]:
            plan["peasants"] = max(plan["peasants"], min(40, idl["peasants"] - keep["peasants"]))

        arr = []
        if self.cfg.allow_peasant_explore and plan.get("peasants",0) > 0:
            arr.append({"TroopTypeID": self.client.TROOP_ID["peasants"], "AmountToSend": int(plan["peasants"])})
        for k in ("footmen","pikemen","archers","light_cav","heavy_cav","knights"):
            n = int(plan.get(k,0))
            if n <= 0: continue
            if danger >= 5:
                n = min(n, 3 if k in ("light_cav","heavy_cav","knights") else 5)
            arr.append({"TroopTypeID": self.client.TROOP_ID[k], "AmountToSend": n})
        return arr

    async def plan_exploration(self, st: KingdomState) -> str:
        if st.queues.get("explore_busy"): return "Explore: busy"
        danger = self.threat_level()
        payload = self._plan_explore_payload(st, danger)
        if not payload:
            return "Explore: no eligible units or danger too high"
        res = await self.client.explore(payload)
        if str(res.get("ReturnValue")) == "1":
            minutes = random.uniform(30, 45) if self.is_early_game(st) else random.uniform(60, 90)
            self._explore_back_at = local_now(self.cfg.tz_name) + timedelta(minutes=minutes)
            self.brain.log_explore(sent=payload, result="OK", land_before=st.land, land_after=None)
            return f"Explore sent (danger={danger:.1f}, ETA ~{minutes:.0f}m)"
        payload_no_peas = [t for t in payload if t.get("TroopTypeID") != self.client.TROOP_ID["peasants"]]
        if len(payload_no_peas) != len(payload) and payload_no_peas:
            res2 = await self.client.explore(payload_no_peas)
            if str(res2.get("ReturnValue")) == "1":
                minutes = random.uniform(30, 45) if self.is_early_game(st) else random.uniform(60, 90)
                self._explore_back_at = local_now(self.cfg.tz_name) + timedelta(minutes=minutes)
                self.brain.log_explore(sent=payload_no_peas, result="OK(fallback)", land_before=st.land, land_after=None)
                return f"Explore sent (fallback no-peas, danger={danger:.1f}, ETA ~{minutes:.0f}m)"
        self.brain.log_explore(sent=payload, result=res.get("ReturnString","fail"), land_before=st.land, land_after=None)
        return f"Explore failed: {res.get('ReturnString','unknown')}"

    async def maybe_fix_storage(self, st: KingdomState) -> Optional[str]:
        msgs=[]
        if st.resources["wood"] >= st.storage["wood"]:
            if st.queues["can_build"].get("lumber", True) and st.queues["building_inprog"].get("lumber",0)==0:
                d = await self.client.build(buildingTypeId="9", quantity=1); msgs.append(f"Build Lumber Yard -> {d.get('ReturnString','OK')}")
        if st.resources["stone"] >= st.storage["stone"]:
            if st.queues["can_build"].get("quarries", True) and st.queues["building_inprog"].get("quarries",0)==0:
                d = await self.client.build(buildingTypeId="6", quantity=1); msgs.append(f"Build Stone Quarry -> {d.get('ReturnString','OK')}")
        return " | ".join(msgs) if msgs else None

    def _find_skill_id(self, skills: List[dict], name_like: str) -> Optional[str]:
        key = name_like.lower().replace(" ","")
        for s in skills:
            if s["name"].lower().replace(" ","")==key: return s["id"] or None
        for s in skills:
            if key in s["name"].lower().replace(" ",""): return s["id"] or None
        return None

    async def maybe_research(self, st: KingdomState) -> str:
        if (st.research.get("in_progress") or []): return "Research in progress"
        skills = st.research.get("skills") or []
        pick=None
        if st.resources["food"] < max(10000, st.storage["food"]*0.25) or st.prod_per_hour.get("food",0) < 3000:
            for n in ("Better Farming Methods","Irrigation","Mathematics"):
                sid = self._find_skill_id(skills,n)
                if sid: pick=(n,sid); break
        if pick is None and st.resources["gold"] < max(10000, st.storage["gold"]*0.2):
            for n in ("Mathematics","Accounting"):
                sid = self._find_skill_id(skills,n)
                if sid: pick=(n,sid); break
        if pick is None:
            for n in ("Engineering","Improved Tools","Better Construction Methods"):
                sid = self._find_skill_id(skills,n)
                if sid: pick=(n,sid); break
        if pick is None: return "No suitable research"
        name,sid = pick
        row = next((s for s in skills if s.get("id")==sid), {})
        gcost = int(row.get("goldCost") or 0)
        if gcost and st.resources["gold"] < gcost:
            return f"Research '{name}': need {gcost} gold"
        res = await self.client.train_skill(sid)
        return res.get("ReturnString", f"Research queued: {name}")

    def _parse_message_actions(self, subject: str, body: str) -> List[Tuple[str, Any]]:
        s = (subject or "").lower()
        b = (body or "").lower()
        actions: List[Tuple[str, Any]] = []

        if "tutorial 1" in s and "explore" in (s+b):
            actions.append(("explore_hint", None))
        if "tutorial 2" in s and "gems" in s:
            actions.append(("note", "tutorial_gems_info"))
        if "completed quest" in s:
            actions.append(("note", "quest_completed"))
        if "spy report" in s:
            actions.append(("note", "log_spy_report"))
        if "attack report" in s:
            actions.append(("note", "log_attack_report"))

        m = re.search(r"set\s+tax\s+to\s+(\d+)", b)
        if m:
            rate = max(0, min(26, int(m.group(1))))
            actions.append(("set_tax", rate))

        if any(k in b for k in ("build farm", "grain farm", "more food", "run out of food")):
            actions.append(("build", {"typeId":"2","qty":1}))
        if any(k in b for k in ("build barracks", "train footmen")):
            actions.append(("build", {"typeId":"16","qty":1}))
        if any(k in b for k in ("build lumber", "wood storage", "wood cap")):
            actions.append(("build", {"typeId":"9","qty":1}))
        if any(k in b for k in ("build quarry", "stone storage", "stone cap")):
            actions.append(("build", {"typeId":"6","qty":1}))
        if "build barn" in b:
            actions.append(("build", {"typeId":"22","qty":1}))

        if "send explorers" in b or "explore for land" in b:
            actions.append(("explore_hint", None))

        if "research" in b:
            if "farming" in b or "food" in b:
                actions.append(("research_hint", "Better Farming Methods"))
            elif "engineering" in b or "build faster" in b:
                actions.append(("research_hint", "Engineering"))
            elif "mathematics" in b or "tax" in b:
                actions.append(("research_hint", "Mathematics"))

        if "train footmen" in b:
            actions.append(("train_hint", "footmen"))
        if "train spies" in b or "build spies" in b:
            actions.append(("train_hint", "spies"))

        return actions

    async def process_messages(self, st: KingdomState):
        if not self.cfg.messages_enabled:
            return

        inbox = await self.client.get_messages_list()
        rows = inbox.get("messages") or inbox.get("items") or inbox.get("inbox") or []
        msgs = []
        for r in rows:
            mid = r.get("id") or r.get("messageId") or r.get("Id")
            if not mid: continue
            subject = r.get("subject") or r.get("Subject") or ""
            sender  = r.get("from") or r.get("sender") or r.get("From") or "System"
            is_read = bool(r.get("isRead") or r.get("read") or r.get("IsRead") or False)
            msgs.append((int(mid), subject, sender, is_read))

        pri = []
        for mid, sub, snd, read in msgs:
            score = (0 if not read else 10)
            if (sub or "").lower().startswith(("tutorial","completed quest")): score -= 5
            if (snd or "").lower() == "system": score -= 1
            pri.append((score, mid, sub, snd, read))
        pri.sort()
        to_open = [(mid, sub, snd) for _, mid, sub, snd, _ in pri[:5]]

        acted_any = False
        for mid, sub, snd in to_open:
            try:
                m = await self.client.get_message(int(mid))
            except Exception as e:
                self.logger.warning("GetMessage failed for %s: %s", mid, e)
                continue

            body = m.get("body") or m.get("Body") or m.get("messageBody") or m.get("MessageBody") or ""
            subject = m.get("subject") or m.get("Subject") or sub or ""
            sender  = m.get("from") or m.get("sender") or m.get("From") or snd or ""
            self.brain.log_message(message_id=int(mid), subject=subject, sender=sender, body=body)
            self.logger.info("MESSAGE [%s] %s | from=%s", mid, subject, sender)

            actions = self._parse_message_actions(subject, body)
            for kind, payload in actions:
                if kind == "set_tax":
                    self.logger.info("MSG ACTION: would set tax to %s (endpoint TBD)", payload)
                elif kind == "build":
                    type_id = payload.get("typeId"); qty = int(payload.get("qty",1))
                    try:
                        res = await self.client.build(type_id, qty)
                        self.logger.info("MSG ACTION: build %s x%s -> %s", type_id, qty, res.get("ReturnString","OK"))
                        acted_any = True
                    except Exception as e:
                        self.logger.info("MSG ACTION: build %s failed: %s", type_id, e)
                elif kind == "explore_hint":
                    msg = await self.plan_exploration(st)
                    self.logger.info("MSG ACTION: %s", msg); acted_any = True
                elif kind == "research_hint":
                    skills = st.research.get("skills") or []
                    sid = self._find_skill_id(skills, payload)
                    if sid:
                        res = await self.client.train_skill(sid)
                        self.logger.info("MSG ACTION: research %s -> %s", payload, res.get("ReturnString","OK"))
                        acted_any = True
                    else:
                        self.logger.info("MSG ACTION: research hint %s not available", payload)
                elif kind == "train_hint":
                    self.logger.info("MSG ACTION: would train %s (endpoint TBD)", payload)
                elif kind == "note":
                    self.logger.info("MSG NOTE: %s", payload)
            if actions:
                self.brain.mark_message_acted(int(mid))

        if acted_any:
            self.logger.info("Finished processing messages with actions.")
        elif to_open:
            self.logger.info("Processed messages; no actionable tutorial keywords found.")

    def compute_idle_wait_minutes(self, st: KingdomState) -> float:
        tz = self.cfg.tz_name
        mins_to_tick = minutes_until(self.cfg.tick_minute, tz)
        if 2 <= mins_to_tick <= 7:
            return max(2.0, mins_to_tick + random.uniform(-1, 1))

        if st.queues.get("building_busy") or st.queues.get("training_busy") or st.queues.get("explore_busy"):
            base = random.uniform(3, 12)
            if in_human_sleep_window(tz): base = random.uniform(8, 18)
            return base

        near_cap_wood  = st.storage["wood"]  > 0 and st.resources["wood"]  >= st.storage["wood"]  * 0.95
        near_cap_stone = st.storage["stone"] > 0 and st.resources["stone"] >= st.storage["stone"] * 0.95
        if near_cap_wood or near_cap_stone: return random.uniform(4, 8)

        gold_hr = max(0, st.prod_per_hour.get("gold",0))
        target_gold = max(6000, min(20000, st.resources["gold"]+8000))
        gold_deficit = max(0, target_gold - st.resources["gold"])
        wait_for_gold = (gold_deficit / gold_hr * 60) if gold_hr>0 else 15.0

        if in_human_sleep_window(tz):
            return max(12.0, min(45.0, wait_for_gold + random.uniform(-3,6)))
        else:
            return max(5.0,  min(20.0, wait_for_gold + random.uniform(-2,4)))

    async def step(self) -> None:
        st = await self.fetch_state()
        self.logger.info("Strategy: economic + adaptive | land=%s", st.land)

        if st.resources["wood"] >= st.storage["wood"]:
            self.logger.warning("Wood over-capacity: %s/%s - need more lumber", st.resources["wood"], st.storage["wood"])
        if st.resources["stone"] >= st.storage["stone"]:
            self.logger.warning("Stone over-capacity: %s/%s - need more quarries", st.resources["stone"], st.storage["stone"])

        self.brain.log_tick(self.cfg.tick_minute)

        try:
            await self.process_messages(st)
        except Exception as e:
            self.logger.warning("Message processing error: %s", e)

        if self.is_early_game(st):
            idl = self.idle_units(st)
            self.logger.info("EARLY EXPLORE | idle P/F/Pi/A=%s/%s/%s/%s | explore_busy=%s | allow_peas=%s",
                             idl["peasants"], idl["footmen"], idl["pikemen"], idl["archers"],
                             st.queues.get("explore_busy"), self.cfg.allow_peasant_explore)
            msg = await self.plan_exploration(st)
            self.logger.info(msg)

        if not st.queues.get("building_busy"):
            m = await self.maybe_fix_storage(st)
            if m: self.logger.info(m)

        rmsg = await self.maybe_research(st); self.logger.info(rmsg)

        wait_min = self.compute_idle_wait_minutes(st)
        self.logger.info("Idle sleep for %.1f minutes (queues build=%s train=%s explore=%s, gold/hr=%s)",
                         wait_min, st.queues.get("building_busy"), st.queues.get("training_busy"),
                         st.queues.get("explore_busy"), st.prod_per_hour.get("gold",0))
        await asyncio.sleep(wait_min * 60)

    async def run(self):
        self.logger.info("KG2 AI Bot initializing...")
        self.logger.info("Mode: %s", "Mock" if self.cfg.mock else "Live")
        self.logger.info("Kingdom ID: %s", self.cfg.kingdom_id)
        if not self.cfg.mock:
            # global warmup
            try:
                await self.client._warmup_for(self.cfg.referer_overview)
            except Exception:
                pass
        self.logger.info("Starting main bot loop...")
        try:
            while True:
                try:
                    await self.step()
                except Exception as e:
                    self.logger.error("Step error: %s", e)
                    await asyncio.sleep(30)
        finally:
            await self.client.close()

# -----------------------
# Entrypoint
# -----------------------

if __name__ == "__main__":
    cfg = Config()
    if not cfg.mock and (not cfg.account_id or not cfg.token or not cfg.kingdom_id):
        logger.error("Missing live auth env (KG2_ACCOUNT_ID / KG2_TOKEN / KG2_KINGDOM_ID). Set KG2_MOCK=true to run sim.")
        sys.exit(1)
    asyncio.run(Bot(cfg).run())
