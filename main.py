# -*- coding: utf-8 -*-
"""
KG2 AI Bot - resilient auth (anti-forgery discovery + cookie fallback) + ASMX client
- Tries multiple anti-forgery token placements (hidden input, Input.__..., meta, JS)
- Falls back to antiforgery cookie (.AspNetCore.Antiforgery*) as RequestVerificationToken header
- Tries multiple login endpoints + field-name shapes
- Supports pasting session cookies via KG2_SESSION_COOKIE / KG2_SESSION_COOKIE_FILE
- Injects Cookie header on ALL requests (plus httpx cookie jar) for stricter WAFs
- Keeps gameplay logic: explore, research, messages, adaptive idle sleep, sqlite "brain"
"""
import os, sys, re, json, asyncio, logging, random, sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
try:
    import zoneinfo
except Exception:
    zoneinfo = None
import httpx
from httpx import Cookies
from urllib.parse import urlsplit

# ---------------- time helpers ----------------
def local_now(tz_name: str = None) -> datetime:
    tz = None
    if tz_name and zoneinfo:
        try: tz = zoneinfo.ZoneInfo(tz_name)
        except Exception: tz = None
    return datetime.now(tz or timezone.utc)

def in_human_sleep_window(tz_name: str = "America/Chicago") -> bool:
    h = local_now(tz_name).hour
    return 1 <= h < 6

def minutes_until(minute_past_hour: int, tz_name: str) -> float:
    now = local_now(tz_name)
    tgt = now.replace(minute=minute_past_hour, second=0, microsecond=0)
    if tgt <= now: tgt = tgt + timedelta(hours=1)
    return (tgt - now).total_seconds() / 60.0

# ---------------- config ----------------
def env_bool(key: str, default: bool=False) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

@dataclass
class Config:
    base_url: str = os.getenv("KG2_BASE_URL", "https://www.kingdomgame.net")
    world_id: str = os.getenv("KG2_WORLD_ID", "1")
    account_id: str = os.getenv("KG2_ACCOUNT_ID", "")
    kingdom_id: int = int(os.getenv("KG2_KINGDOM_ID", "0"))
    token: str = os.getenv("KG2_TOKEN", "")
    username: str = os.getenv("KG2_USERNAME", "")
    password: str = os.getenv("KG2_PASSWORD", "")

    referer_overview: str  = os.getenv("KG2_REFERER_OVERVIEW",  "https://www.kingdomgame.net/overview")
    referer_buildings: str = os.getenv("KG2_REFERER_BUILDINGS", "https://www.kingdomgame.net/buildings")
    referer_war: str       = os.getenv("KG2_REFERER_WAR",       "https://www.kingdomgame.net/warroom")
    referer_research: str  = os.getenv("KG2_REFERER_RESEARCH",  "https://www.kingdomgame.net/research")
    referer_messages: str  = os.getenv("KG2_REFERER_MESSAGES",  "https://www.kingdomgame.net/messages")

    origin_url: str = "https://www.kingdomgame.net"
    tz_name: str = os.getenv("KG2_TZ", "America/Chicago")

    allow_peasant_explore: bool = env_bool("KG2_ALLOW_PEASANT_EXPLORE", True)
    tick_minute: int = int(os.getenv("KG2_TICK_MINUTE", "2"))
    threat_window_hrs: int = int(os.getenv("KG2_THREAT_WINDOW_HRS", "48"))
    messages_enabled: bool = env_bool("KG2_MESSAGES_ENABLED", True)
    auto_token: bool = env_bool("KG2_AUTO_TOKEN", True)
    use_browser_headers: bool = env_bool("KG2_USE_BROWSER_HEADERS", True)
    mock: bool = env_bool("KG2_MOCK", False)

    # Session cookie support
    bootstrap_cookie: str = os.getenv("KG2_SESSION_COOKIE", "")
    bootstrap_cookie_file: str = os.getenv("KG2_SESSION_COOKIE_FILE", "")
    timeout_s: int = int(os.getenv("KG2_TIMEOUT_S", "25"))

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kg2bot")

# ---------------- sqlite brain ----------------
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
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY, ts TEXT, message_id INT, subject TEXT, sender TEXT, body TEXT, acted INTEGER DEFAULT 0
        );
        """)
    def _nowstr(self): return local_now(self.tz).isoformat(timespec="seconds")
    def log_explore(self, sent, result, land_before=None, land_after=None):
        self.conn.execute(
            "INSERT INTO events_explore(ts,sent_json,result,land_before,land_after) VALUES(?,?,?,?,?)",
            (self._nowstr(), json.dumps(sent), str(result), land_before, land_after))
    def log_message(self, message_id: int, subject: str, sender: str, body: str):
        self.conn.execute(
            "INSERT INTO messages(ts,message_id,subject,sender,body,acted) VALUES(?,?,?,?,?,0)",
            (self._nowstr(), int(message_id), subject or "", sender or "", body or ""))

# ---------------- HTTP client ----------------
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "sec-ch-ua": '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"',
    "sec-ch-ua-platform": '"Windows"',
    "sec-ch-ua-mobile": "?0",
    "X-Requested-With": "XMLHttpRequest",
}
HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

def _load_cookie_header(cfg: Config) -> str:
    # prefer env, else file
    raw = (cfg.bootstrap_cookie or "").strip()
    if not raw and cfg.bootstrap_cookie_file and os.path.exists(cfg.bootstrap_cookie_file):
        try:
            with open(cfg.bootstrap_cookie_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except Exception as e:
            logger.warning("Could not read KG2_SESSION_COOKIE_FILE: %s", e)
    return raw

def _apply_cookie_jar_and_header(raw_cookie: str, base_url: str) -> Tuple[Cookies, str]:
    cookies = Cookies()
    header = ""
    host = urlsplit(base_url).hostname
    if raw_cookie:
        parts = [p.strip() for p in raw_cookie.split(";") if p.strip() and "=" in p]
        for part in parts:
            k, v = part.split("=", 1)
            cookies.set(k.strip(), v.strip(), domain=host)
        header = "; ".join([p.strip() for p in raw_cookie.split(";") if "=" in p])
    return cookies, header

class KG2Client:
    TROOP_ID = {
        "peasants": 1, "footmen": 17, "pikemen": 18, "archers": 20, "crossbow": 22,
        "light_cav": 23, "heavy_cav": 24, "knights": 25
    }
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Load cookie header (env or file), also fill cookie jar
        self._cookie_header_str = _load_cookie_header(cfg)
        cookies, header = _apply_cookie_jar_and_header(self._cookie_header_str, cfg.base_url)
        self._cookie_header_str = header or self._cookie_header_str  # normalize spacing/order
        if self._cookie_header_str:
            logger.info("Loaded session cookies (%d pairs).", self._cookie_header_str.count("="))

        self._client = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            follow_redirects=True,
            timeout=self.cfg.timeout_s,
            headers=BROWSER_HEADERS if cfg.use_browser_headers else {"User-Agent": BROWSER_HEADERS["User-Agent"]},
            cookies=cookies,
        )

    async def close(self): await self._client.aclose()

    async def _warmup_for(self, referer_url: str):
        try:
            path = urlsplit(referer_url).path or "/"
            hdrs = {
                "Accept": HTML_ACCEPT,
                "Referer": self.cfg.origin_url + "/",
                "Origin": self.cfg.origin_url,
                "World-Id": str(self.cfg.world_id),
            }
            if self._cookie_header_str:
                hdrs["Cookie"] = self._cookie_header_str
            r = await self._client.get(path, headers=hdrs)
            logging.info("Warmup GET %s -> %s", path, r.status_code)
        except Exception as e:
            logging.warning("Warmup failed for %s: %s", referer_url, e)

    # --- Anti-forgery discovery ---
    def _find_anti_token(self, html: str) -> Optional[str]:
        rx_list = [
            r'name="__RequestVerificationToken"\s+type="hidden"\s+value="([^"]+)"',
            r'name="Input\.__RequestVerificationToken"\s+type="hidden"\s+value="([^"]+)"',
            r'meta\s+name="__RequestVerificationToken"\s+content="([^"]+)"',
            r'__RequestVerificationToken"\s*:\s*"([^"]+)"',
        ]
        for rx in rx_list:
            m = re.search(rx, html, re.I)
            if m: return m.group(1)
        return None

    def _get_antiforgery_cookie(self) -> Optional[str]:
        for c in self._client.cookies.jar:
            if c.name.lower().startswith(".aspnetcore.antiforgery"):
                return c.value
        return None

    async def _login_form(self) -> bool:
        if not (self.cfg.username and self.cfg.password):
            return False
        lg_get = await self._client.get("/Account/Login", headers={
            "Accept": HTML_ACCEPT, "World-Id": str(self.cfg.world_id),
            "Referer": self.cfg.origin_url + "/", "Origin": self.cfg.origin_url,
            **({"Cookie": self._cookie_header_str} if self._cookie_header_str else {})
        })
        if lg_get.status_code != 200:
            logging.info("Login GET unexpected: %s", lg_get.status_code); return False

        anti = self._find_anti_token(lg_get.text) or self._get_antiforgery_cookie()
        if not anti:
            logging.info("Anti-forgery token not found on login page.")
        endpoints = ["/Account/Login", "/Account/Login?handler=Login", "/Identity/Account/Login"]
        forms = [
            {"Email": self.cfg.username, "Password": self.cfg.password, "RememberMe": "true"},
            {"Input.Email": self.cfg.username, "Input.Password": self.cfg.password, "RememberMe": "true"},
        ]
        headers_base = {
            "Content-Type":"application/x-www-form-urlencoded",
            "Origin": self.cfg.origin_url,
            "Referer": self.cfg.origin_url + "/Account/Login",
            "World-Id": str(self.cfg.world_id),
        }
        if anti:
            headers_base["RequestVerificationToken"] = anti
        if self._cookie_header_str:
            headers_base["Cookie"] = self._cookie_header_str

        for ep in endpoints:
            for form in forms:
                if "__RequestVerificationToken" not in form and anti:
                    form = form.copy()
                    form["__RequestVerificationToken"] = anti
                try:
                    r = await self._client.post(ep, data=form, headers=headers_base)
                    if r.status_code in (200,302):
                        await self._warmup_for(self.cfg.referer_overview)
                        logging.info("Form login established via %s", ep)
                        return True
                except Exception as e:
                    logging.info("Login try %s failed: %s", ep, e)
        return False

    # --- token sniff ---
    def _extract_token(self, data: Any) -> Optional[str]:
        if isinstance(data, dict):
            for k in ("token","authToken","Token"):
                v = data.get(k)
                if isinstance(v, str) and len(v) >= 16:
                    return v
            for v in data.values():
                t = self._extract_token(v)
                if t: return t
        elif isinstance(data, list):
            for it in data:
                t = self._extract_token(it)
                if t: return t
        elif isinstance(data, str):
            try:
                j = json.loads(data)
                return self._extract_token(j)
            except Exception:
                pass
        return None

    # --- low-level ASMX ---
    def _asmx_headers(self, referer: str) -> Dict[str,str]:
        h = {
            "Content-Type":"application/json",
            "Accept":"application/json, text/plain, */*",
            "World-Id": str(self.cfg.world_id),
            "Origin": self.cfg.origin_url,
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
        }
        if self._cookie_header_str:
            h["Cookie"] = self._cookie_header_str
        return h

    async def _post_raw(self, path: str, body: dict, referer: str) -> httpx.Response:
        await self._warmup_for(referer)
        return await self._client.post(path, json=body, headers=self._asmx_headers(referer))

    async def post_asmx(self, path: str, body: dict, referer: str) -> dict:
        resp = await self._post_raw(path, body, referer)
        html_500 = (resp.status_code == 500 and resp.headers.get("Content-Type","").startswith("text/html"))
        if resp.status_code in (401,403) or html_500:
            logging.error("Auth-ish failure %s on %s; attempting refresh...", resp.status_code, path)
            if await self._login_form():
                if "token" in body: body["token"] = self.cfg.token
                resp = await self._post_raw(path, body, referer)
        if resp.status_code in (401,403):
            raise RuntimeError(f"AUTH_FAIL {resp.status_code} at {path} | referer={referer} | world={self.cfg.world_id} | acct={self.cfg.account_id} | king={self.cfg.kingdom_id} | token=***")
        if resp.status_code == 500 and resp.headers.get("Content-Type","").startswith("text/html"):
            raise RuntimeError(f"AUTH_FAIL 500-HTML at {path}")
        resp.raise_for_status()
        data = resp.json()
        # unwrap .d
        if isinstance(data, dict) and "d" in data and isinstance(data["d"], str):
            try: parsed = json.loads(data["d"])
            except Exception: parsed = data["d"]
            tok = self._extract_token(parsed)
            if tok and tok != self.cfg.token:
                self.cfg.token = tok
                logging.info("Observed token rotation (%s). Stored new token.", path)
            return parsed
        tok2 = self._extract_token(data)
        if tok2 and tok2 != self.cfg.token:
            self.cfg.token = tok2
            logging.info("Observed token rotation (%s). Stored new token.", path)
        return data

    # --- API wrappers ---
    async def get_details(self) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomDetails", b, self.cfg.referer_overview)
    async def get_resources(self) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomResources", b, self.cfg.referer_overview)
    async def get_population(self) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomPopulation", b, self.cfg.referer_war)
    async def get_buildings(self) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Kingdoms.asmx/GetKingdomBuildings", b, self.cfg.referer_buildings)
    async def build(self, buildingTypeId: str, quantity: int) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id,
             "buildingTypeId": str(buildingTypeId), "quantity": int(quantity)}
        return await self.post_asmx("/WebService/Buildings.asmx/BuildBuilding", b, self.cfg.referer_buildings)
    async def get_skills(self) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Research.asmx/GetSkills", b, self.cfg.referer_research)
    async def get_training_skills(self) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Research.asmx/GetTrainingSkills", b, self.cfg.referer_research)
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
                if d.get("ReturnValue") == 1 or d.get("success"): return d
                if "ReturnString" in d: return d
            except Exception:
                continue
        return {"ReturnValue":0,"ReturnString":"No research endpoint accepted the request"}
    async def explore(self, troops_array: List[dict]) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id,
             "troops": json.dumps(troops_array)}
        return await self.post_asmx("/WebService/Kingdoms.asmx/Explore", b, self.cfg.referer_war)
    async def get_messages_list(self) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id}
        return await self.post_asmx("/WebService/Messages.asmx/GetMessages", b, self.cfg.referer_messages)
    async def get_message(self, message_id: int) -> dict:
        b = {"accountId": self.cfg.account_id, "token": self.cfg.token, "kingdomId": self.cfg.kingdom_id,
             "messageId": int(message_id)}
        per_msg_referer = f"https://www.kingdomgame.net/messages/message/{int(message_id)}"
        return await self.post_asmx("/WebService/Messages.asmx/GetMessage", b, per_msg_referer)

# ---------------- state normalization ----------------
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

def _normalize_resources(rjs: dict):
    name_map = {"land":"land","food":"food","gold":"gold","stone":"stone","wood":"wood","blue gems":"blue_gems","mana":"mana"}
    resources, storage, prod, flags = {}, {}, {}, {"maintenance_issue_wood": False, "maintenance_issue_stone": False}
    for r in (rjs or {}).get("resources", []):
        n = (r.get("name") or "").lower().strip()
        key = name_map.get(n)
        if not key: continue
        resources[key] = int(r.get("amount", 0))
        storage[key]   = int(r.get("capacity", 0))
        prod[key]      = int(r.get("productionPerHour", 0))
        if n=="wood" and r.get("maintenanceIssue"): flags["maintenance_issue_wood"]=True
        if n=="stone" and r.get("maintenanceIssue"): flags["maintenance_issue_stone"]=True
    for k in name_map.values(): resources.setdefault(k,0); storage.setdefault(k,0); prod.setdefault(k,0)
    return resources, storage, prod, flags

def _normalize_population(pjs: dict):
    key_map = {"peasants":"peasants","footmen":"footmen","pikemen":"pikemen","archers":"archers","crossbowmen":"crossbow",
               "light cavalry":"light_cav","heavy cavalry":"heavy_cav","knights":"knights","elites":"elites",
               "spies":"spies","priests":"priests","diplomats":"diplomats","market wagons":"wagons"}
    units, inprog, returning = {}, {}, {}
    for r in (pjs or {}).get("population", []):
        name = (r.get("name") or "").lower().strip(); key = key_map.get(name)
        if not key: continue
        units[key] = int(r.get("amount", 0))
        inprog[key]=int(r.get("amountInProgress",0))
        returning[key]=int(r.get("amountReturning",0))
    for k in key_map.values(): units.setdefault(k,0); inprog.setdefault(k,0); returning.setdefault(k,0)
    return units, inprog, returning

def _normalize_buildings(bjs: dict):
    map_name = {"houses":"houses","grain farms":"farms","lumber yards":"lumber","stone quarries":"quarries","barracks":"barracks",
                "stables":"stables","archery ranges":"archery","guildhalls":"guild","temples":"temples","markets":"markets",
                "barns":"barns","castles":"castles","horse farms":"horse_farms"}
    buildings, inprog, canb = {}, {}, {}
    for r in (bjs or {}).get("buildings", []):
        name = (r.get("name") or "").lower().strip(); key = map_name.get(name)
        if not key: continue
        buildings[key]=int(r.get("amount",0))
        inprog[key]=int(r.get("amountInProgress",0))
        canb[key]=bool(r.get("canBuild",False))
    for k in ("houses","farms","lumber","quarries","barracks","stables","archery","barns","markets","temples","castles"):
        buildings.setdefault(k,0); inprog.setdefault(k,0); canb.setdefault(k,True)
    return buildings, inprog, canb

def normalize_skills(sk_json: dict) -> List[dict]:
    rows = sk_json.get("skills") or sk_json.get("items") or sk_json.get("research") or []
    out=[]
    for r in rows:
        rid = r.get("id") or r.get("skillTypeId") or r.get("researchTypeId")
        name = (r.get("name") or r.get("displayName") or "").strip()
        lvl = int(r.get("currentLevel") or r.get("level") or 0)
        gold= int(r.get("goldCost") or r.get("gold") or 0)
        gems= int(r.get("gemCost") or r.get("gems") or 0)
        time= (r.get("timeNext") or r.get("researchNextLevel") or r.get("duration") or "").strip()
        out.append({"id":str(rid) if rid is not None else "","name":name,"currentLevel":lvl,"goldCost":gold,"gemCost":gems,"timeText":time})
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

# ---------------- bot core ----------------
class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = KG2Client(cfg)
        self.logger = logger
        self.brain = Brain(tz=cfg.tz_name)
        self._explore_back_at: Optional[datetime] = None

    async def fetch_state(self) -> "KingdomState":
        kd = await self.client.get_details()
        rjs = await self.client.get_resources()
        pjs = await self.client.get_population()
        bjs = await self.client.get_buildings()
        sk  = await self.client.get_skills()
        ts  = await self.client.get_training_skills()
        resources, storage, prod, flags = _normalize_resources(rjs)
        units, inprog_units, returning = _normalize_population(pjs)
        buildings, inprog_build, can_build = _normalize_buildings(bjs)
        skills = normalize_skills(sk); training = normalize_training_skills(ts)
        now = local_now(self.cfg.tz_name)
        explore_busy_local = bool(self._explore_back_at and now < self._explore_back_at)
        if self._explore_back_at and now >= self._explore_back_at: self._explore_back_at = None
        return KingdomState(
            id=int(kd.get("id", 0)), name=kd.get("name",""),
            tax_rate=int(kd.get("taxRate",24)),
            land=int(kd.get("land", resources.get("land",0))),
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

    def is_early_game(self, st: "KingdomState") -> bool:
        return st.land < 5000 or sum(st.buildings.values()) < 200

    def idle_units(self, st: "KingdomState") -> Dict[str,int]:
        u = st.units or {}
        return {"peasants":u.get("peasants",0),"footmen":u.get("footmen",0),"pikemen":u.get("pikemen",0),
                "archers":u.get("archers",0),"light_cav":u.get("light_cav",0),
                "heavy_cav":u.get("heavy_cav",0),"knights":u.get("knights",0)}

    def _plan_explore_payload(self, st: "KingdomState") -> List[dict]:
        keep = {"peasants": 20, "footmen": 8, "pikemen":10, "archers":10, "light_cav":0, "heavy_cav":0, "knights":0}
        idl = self.idle_units(st)
        plan = {}
        for k in ("peasants","footmen","pikemen","archers","light_cav","heavy_cav","knights"):
            avail = max(0, idl.get(k,0) - keep.get(k,0))
            plan[k] = int(avail * (0.85 if self.is_early_game(st) else 0.4))
        if sum(plan.values()) < 15 and idl.get("peasants",0) > keep["peasants"]:
            plan["peasants"] = max(plan["peasants"], min(40, idl["peasants"] - keep["peasants"]))
        arr=[]
        if self.cfg.allow_peasant_explore and plan.get("peasants",0) > 0:
            arr.append({"TroopTypeID": self.client.TROOP_ID["peasants"], "AmountToSend": int(plan["peasants"])})
        for k in ("footmen","pikemen","archers","light_cav","heavy_cav","knights"):
            n = int(plan.get(k,0))
            if n>0:
                arr.append({"TroopTypeID": self.client.TROOP_ID[k], "AmountToSend": n})
        return arr

    async def plan_exploration(self, st: "KingdomState") -> str:
        if st.queues.get("explore_busy"): return "Explore: busy"
        payload = self._plan_explore_payload(st)
        if not payload: return "Explore: no eligible units"
        res = await self.client.explore(payload)
        if str(res.get("ReturnValue")) == "1":
            minutes = random.uniform(30, 45) if self.is_early_game(st) else random.uniform(60, 90)
            self._explore_back_at = local_now(self.cfg.tz_name) + timedelta(minutes=minutes)
            self.brain.log_explore(sent=payload, result="OK", land_before=st.land, land_after=None)
            return f"Explore sent, ETA ~{minutes:.0f}m"
        # retry without peasants if rejected
        payload2 = [t for t in payload if t["TroopTypeID"] != self.client.TROOP_ID["peasants"]]
        if payload2:
            res2 = await self.client.explore(payload2)
            if str(res2.get("ReturnValue")) == "1":
                minutes = random.uniform(30, 45) if self.is_early_game(st) else random.uniform(60, 90)
                self._explore_back_at = local_now(self.cfg.tz_name) + timedelta(minutes=minutes)
                self.brain.log_explore(sent=payload2, result="OK(no-peasants)", land_before=st.land, land_after=None)
                return f"Explore sent (no peasants), ETA ~{minutes:.0f}m"
        return f"Explore failed: {res.get('ReturnString','unknown')}"

    async def maybe_research(self, st: "KingdomState") -> str:
        if (st.research.get("in_progress") or []): return "Research in progress"
        skills = st.research.get("skills") or []
        def find(name_like: str) -> Optional[str]:
            k = name_like.lower().replace(" ","")
            for s in skills:
                if (s["name"] or "").lower().replace(" ","")==k: return s["id"] or None
            for s in skills:
                if k in (s["name"] or "").lower().replace(" ",""): return s["id"] or None
            return None
        pick=None
        if st.prod_per_hour.get("food",0) < 3000:    pick=("Better Farming Methods", find("Better Farming Methods"))
        if not pick and st.prod_per_hour.get("gold",0) < 500: pick=("Mathematics", find("Mathematics"))
        if not pick: pick=("Engineering", find("Engineering"))
        if not pick or not pick[1]: return "No suitable research"
        name,sid = pick
        res = await self.client.train_skill(sid)
        return res.get("ReturnString", f"Research queued: {name}")

    async def process_messages(self, st: "KingdomState"):
        if not self.cfg.messages_enabled: return
        inbox = await self.client.get_messages_list()
        rows = inbox.get("messages") or inbox.get("items") or inbox.get("inbox") or []
        mids=[]
        for r in rows[:6]:
            mid = r.get("id") or r.get("messageId") or r.get("Id")
            if mid: mids.append(int(mid))
        for mid in mids:
            try:
                m = await self.client.get_message(mid)
            except Exception as e:
                self.logger.info("GetMessage %s failed: %s", mid, e); continue
            subject = m.get("subject") or m.get("Subject") or ""
            sender  = m.get("from") or m.get("From") or "System"
            body    = m.get("body") or m.get("Body") or ""
            self.logger.info("MESSAGE [%s] %s | from=%s", mid, subject, sender)
            self.brain.log_message(mid, subject, sender, body)

    def compute_idle_wait_minutes(self, st: "KingdomState") -> float:
        tz = self.cfg.tz_name
        mins_to_tick = minutes_until(self.cfg.tick_minute, tz)
        if 2 <= mins_to_tick <= 7:
            return max(2.0, mins_to_tick + random.uniform(-1, 1))
        if st.queues.get("building_busy") or st.queues.get("training_busy") or st.queues.get("explore_busy"):
            base = random.uniform(3, 12)
            if in_human_sleep_window(tz): base = random.uniform(8, 18)
            return base
        gold_hr = max(0, st.prod_per_hour.get("gold",0))
        target_gold = max(6000, min(20000, st.resources["gold"]+8000))
        gold_deficit = max(0, target_gold - st.resources["gold"])
        wait_for_gold = (gold_deficit / gold_hr * 60) if gold_hr>0 else 15.0
        if in_human_sleep_window(tz):
            return max(12.0, min(45.0, wait_for_gold + random.uniform(-3,6)))
        else:
            return max(5.0,  min(20.0, wait_for_gold + random.uniform(-2,4)))

    async def fetch_state_safe(self) -> "KingdomState":
        return await self.fetch_state()

    async def step(self) -> None:
        st = await self.fetch_state_safe()
        self.logger.info("Strategy: economic+adaptive | land=%s", st.land)
        try: await self.process_messages(st)
        except Exception as e: self.logger.warning("Message processing error: %s", e)
        if self.is_early_game(st):
            self.logger.info("EARLY EXPLORE | land=%s | idle P/F/Pi/A=%s/%s/%s/%s | explore_busy=%s",
                             st.land, st.units.get("peasants",0), st.units.get("footmen",0),
                             st.units.get("pikemen",0), st.units.get("archers",0),
                             st.queues.get("explore_busy"))
            msg = await self.plan_exploration(st); self.logger.info(msg)
        rmsg = await self.maybe_research(st); self.logger.info(rmsg)
        wait_min = self.compute_idle_wait_minutes(st)
        self.logger.info("Idle sleep for %.1f minutes (buildBusy=%s trainBusy=%s exploreBusy=%s)",
                         wait_min, st.queues.get("building_busy"), st.queues.get("training_busy"),
                         st.queues.get("explore_busy"))
        await asyncio.sleep(wait_min * 60)

    async def run(self):
        self.logger.info("KG2 AI Bot initializing...")
        self.logger.info("Mode: %s", "Mock" if self.cfg.mock else "Live")
        self.logger.info("Kingdom ID: %s", self.cfg.kingdom_id)
        if not self.cfg.mock:
            try: await self.client._warmup_for(self.cfg.referer_overview)
            except Exception: pass
        self.logger.info("Starting main bot loop...")
        try:
            while True:
                try:
                    await self.step()
                except RuntimeError as e:
                    if "AUTH_FAIL" in str(e):
                        self.logger.error("Auth error: %s", e)
                        await asyncio.sleep(20)
                    else:
                        self.logger.error("Step error: %s", e); await asyncio.sleep(20)
                except Exception as e:
                    self.logger.error("Step error: %s", e); await asyncio.sleep(20)
        finally:
            await self.client.close()

# --------------- entry ---------------
if __name__ == "__main__":
    cfg = Config()
    if not cfg.mock and (not cfg.account_id or not cfg.kingdom_id):
        logger.error("Missing KG2_ACCOUNT_ID / KG2_KINGDOM_ID. Set KG2_MOCK=true to run sim.")
        sys.exit(1)
    # Either user/pass or pasted session cookie
    if not cfg.mock and not (cfg.username and cfg.password) and not (cfg.bootstrap_cookie or cfg.bootstrap_cookie_file):
        logger.error("Provide KG2_USERNAME/KG2_PASSWORD (preferred) or KG2_SESSION_COOKIE / KG2_SESSION_COOKIE_FILE.")
        sys.exit(1)
    asyncio.run(Bot(cfg).run())
