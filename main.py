"""
main.py — KG2 AI Bot Base (async, concurrent)

Features
- .env-driven config: BASE_URL, ACCOUNT_ID, TOKEN, KINGDOM_ID
- Async HTTP via httpx + retries (tenacity) + structured logging
- TrainPopulation & BuildBuilding helpers (quantity per call)
- Concurrency fan-out with RSS-aware stop (stops after repeated "not enough resources")
- Speedup/spend-style generic action
- Raw POST escape hatch
- Pydantic v2-compatible validators
- Helpful CLI help & examples when no subcommand is provided

Install:
    pip install "httpx[http2]" pydantic tenacity python-dotenv

Examples:
    python main.py train --troop foot --qty 1
    python main.py train --troop archer --qty 25 --per 1 --concurrent 5
    python main.py build --building barracks --count 3 --concurrent 3
    python main.py speedup --type-id 134207 --amount 50
"""

from __future__ import annotations

import argparse
import shlex
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Callable, Tuple

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kg2.ai")

# Optional HTTP/2 support. If the h2 package isn't installed, fall back to HTTP/1.1.
try:
    import h2  # noqa: F401
    HTTP2_ENABLED = True
except ModuleNotFoundError:
    HTTP2_ENABLED = False
    log.warning(
        "h2 package not installed; HTTP/2 disabled. Install httpx[http2] to enable."
    )

# ---------- Config ----------
load_dotenv()

def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val or ""

BASE_URL   = env("BASE_URL", required=True).rstrip("/")
ACCOUNT_ID = env("ACCOUNT_ID", required=True)
TOKEN      = env("TOKEN", required=True)
KINGDOM_ID = env("KINGDOM_ID", required=True)

# Optional referer URLs for requests that require them
REFERER_OVERVIEW  = env("REFERER_OVERVIEW",  f"{BASE_URL}/overview")
REFERER_BUILDINGS = env("REFERER_BUILDINGS", f"{BASE_URL}/buildings")
REFERER_WAR       = env("REFERER_WAR",       f"{BASE_URL}/warroom")
REFERER_RESEARCH  = env("REFERER_RESEARCH",  f"{BASE_URL}/research")

# Origin and user-agent headers (some endpoints reject missing or non-browser values)
ORIGIN_URL = env("ORIGIN_URL", BASE_URL)
USER_AGENT = env(
    "USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
)

# Endpoints (adjust if your API uses a sub-path)
TRAIN_POPULATION_ENDPOINT = f"{BASE_URL}/TrainPopulation"
BUILD_ENDPOINT            = f"{BASE_URL}/BuildBuilding"
GENERIC_ACTION_ENDPOINT   = f"{BASE_URL}/Action"

HTTP_TIMEOUT = 20.0
RETRIES = 4
DEFAULT_CONCURRENCY = 5
CALL_SPACING_SEC = 0.05   # small delay between enqueued tasks

# ---------- Models ----------
class TrainPopulationReq(BaseModel):
    accountId: str
    token: str
    kingdomId: int
    popTypeId: int
    quantity: int

class BuildReq(BaseModel):
    accountId: str
    token: str
    kingdomId: int
    buildingTypeId: int
    quantity: int = 1

class InnerReturn(BaseModel):
    ReturnValue: int = Field(..., description="Server-defined (0/1 ok, ≥2 error)")
    ReturnString: str = ""

class GenericActionReq(BaseModel):
    accountId: str
    token: str
    kingdomId: int
    type: str
    typeId: int
    amount: int

    @field_validator("amount")
    @classmethod
    def nonzero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("amount cannot be 0")
        return v

class ApiError(RuntimeError):
    pass

class AlreadyUsedError(ApiError):
    pass

# ---------- HTTP Client ----------
@dataclass
class ApiClient:
    client: httpx.AsyncClient

    @retry(
        stop=stop_after_attempt(RETRIES),
        wait=wait_exponential_jitter(initial=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, ApiError)),
        reraise=True,
    )
    async def post_json(self, url: str, payload: Dict[str, Any], referer: Optional[str] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {}
        if referer:
            headers["Referer"] = referer
        r = await self.client.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Some endpoints return {"d":"{\"ReturnValue\":2,...}"} — unwrap if present
        if isinstance(data, dict) and "d" in data and isinstance(data["d"], str):
            try:
                inner = json.loads(data["d"])
                if isinstance(inner, dict) and "ReturnValue" in inner:
                    return {"ReturnValue": inner.get("ReturnValue", -1), "ReturnString": inner.get("ReturnString", "")}
                return inner
            except json.JSONDecodeError:
                return data
        return data

    # ---- Domain helpers ----
    async def train_population(self, pop_type_id: int, qty: int) -> InnerReturn:
        req = TrainPopulationReq(
            accountId=ACCOUNT_ID,
            token=TOKEN,
            kingdomId=int(KINGDOM_ID),
            popTypeId=pop_type_id,
            quantity=qty,
        ).model_dump()
        resp = await self.post_json(TRAIN_POPULATION_ENDPOINT, req, referer=REFERER_WAR)
        ir = self._normalize(resp)
        self._raise_if_error(ir)
        return ir

    async def build_building(self, building_type_id: int, qty: int = 1) -> InnerReturn:
        req = BuildReq(
            accountId=ACCOUNT_ID,
            token=TOKEN,
            kingdomId=int(KINGDOM_ID),
            buildingTypeId=building_type_id,
            quantity=qty,
        ).model_dump()
        resp = await self.post_json(BUILD_ENDPOINT, req, referer=REFERER_BUILDINGS)
        ir = self._normalize(resp)
        self._raise_if_error(ir)
        return ir

    async def generic_action(self, type_code: str, type_id: int, amount: int) -> InnerReturn:
        req = GenericActionReq(
            accountId=ACCOUNT_ID,
            token=TOKEN,
            kingdomId=int(KINGDOM_ID),
            type=type_code,
            typeId=type_id,
            amount=amount,
        ).model_dump()
        resp = await self.post_json(GENERIC_ACTION_ENDPOINT, req, referer=REFERER_RESEARCH)
        ir = self._normalize(resp)
        if "already used a speedup" in (ir.ReturnString or "").lower():
            raise AlreadyUsedError(ir.ReturnString)
        self._raise_if_error(ir)
        return ir

    @staticmethod
    def _normalize(resp: Dict[str, Any]) -> InnerReturn:
        if "ReturnValue" in resp:
            return InnerReturn(**resp)
        msg = resp.get("message") or resp.get("error") or str(resp)
        return InnerReturn(ReturnValue=0 if resp.get("ok") else 1, ReturnString=msg)

    @staticmethod
    def _raise_if_error(ir: InnerReturn) -> None:
        if ir.ReturnValue >= 2:
            raise ApiError(f"Server error {ir.ReturnValue}: {ir.ReturnString or '<no message>'}")

# ---------- ID Maps ----------
# Troops confirmed so far (add more as you confirm):
TROOP_TO_POPTYPE: Dict[str, int] = {
    "foot": 17,        # Footmen
    "archer": 20,      # Archers
    "pike": 21,        # Pikemen
    "lightcav": 22,    # Light Cavalry
    "peasant": 24,     # Peasants / workers
}

# Buildings from your BuildBuilding payloads:
BUILDING_TO_TYPEID: Dict[str, int] = {
    "house": 1,            # Houses
    "grainfarm": 2,        # Grain Farms
    "horsefarm": 3,        # Horse Farms
    "quarry": 6,           # Stone Quarries
    "lumberyard": 9,       # Lumber Yards
    "barracks": 16,        # Barracks
    "archeryrange": 17,    # Archery Ranges
    "stable": 18,          # Stables
    "castle": 19,          # Castles
    "temple": 20,          # Temples
    "guildhall": 21,       # Guildhalls
    "barn": 22,            # Barns
    "embassy": 23,         # Embassies
    "market": 24,          # Markets
}

# ---------- Concurrency helpers ----------
def looks_like_rss_error(msg: str) -> bool:
    m = (msg or "").lower()
    return any(s in m for s in [
        "not enough resources",
        "insufficient",
        "lack of resources",
        "not enough wood",
        "not enough stone",
        "not enough food",
        "not enough gold",
        "resource limit",
    ])

async def fan_out(
    total: int,
    per_request: int,
    concurrency: int,
    worker: Callable[[int], "asyncio.Future[InnerReturn]"],
    label: str,
) -> None:
    if per_request <= 0:
        raise ValueError("--per must be >= 1")
    tasks_needed = (total + per_request - 1) // per_request
    sem = asyncio.Semaphore(max(1, concurrency))
    rss_fail_streak = 0
    MAX_RSS_FAIL_STREAK = 3

    async def run_one(i: int) -> Tuple[int, Optional[str]]:
        nonlocal rss_fail_streak
        async with sem:
            try:
                ir = await worker(per_request)
                msg = ir.ReturnString or "<ok>"
                log.info("%s [%d/%d] ok → %s", label, i + 1, tasks_needed, msg)
                rss_fail_streak = 0
                return (i, None)
            except ApiError as e:
                msg = str(e)
                log.warning("%s [%d/%d] server error → %s", label, i + 1, tasks_needed, msg)
                if looks_like_rss_error(msg):
                    rss_fail_streak += 1
                    if rss_fail_streak >= MAX_RSS_FAIL_STREAK:
                        return (i, "rss-stop")
                return (i, None)
            except httpx.HTTPError as e:
                log.warning("%s [%d/%d] network error → %s", label, i + 1, tasks_needed, e)
                return (i, None)

    running: List[asyncio.Task] = []
    for i in range(tasks_needed):
        running.append(asyncio.create_task(run_one(i)))
        await asyncio.sleep(CALL_SPACING_SEC)

    for t in asyncio.as_completed(running):
        _, stop_reason = await t
        if stop_reason == "rss-stop":
            log.warning("Stopping remainder due to repeated resource failures.")
            for other in running:
                if not other.done():
                    other.cancel()
            break

# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kg2-ai",
        description="KG2 AI Bot Base (concurrency)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    # train
    p_train = sub.add_parser("train", help="Train population units (concurrent)")
    p_train.add_argument("--troop", required=True, choices=sorted(set(list(TROOP_TO_POPTYPE.keys()) + ["custom"])))
    p_train.add_argument("--qty", type=int, default=1, help="Total units to train")
    p_train.add_argument("--per", type=int, default=1, help="Units per request (payload 'quantity')")
    p_train.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent requests")
    p_train.add_argument("--pop-type-id", type=int, help="Required if --troop custom")

    # build
    p_build = sub.add_parser("build", help="Construct buildings (concurrent)")
    p_build.add_argument("--building", required=True, choices=sorted(set(list(BUILDING_TO_TYPEID.keys()) + ["custom"])))
    p_build.add_argument("--count", type=int, default=1, help="Total buildings to place")
    p_build.add_argument("--per", type=int, default=1, help="Buildings per request (payload 'quantity')")
    p_build.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent requests")
    p_build.add_argument("--building-type-id", type=int, help="Required if --building custom")

    # speedup / spend-like action
    p_sp = sub.add_parser("speedup", help="Attempt single speedup/spend-style action")
    p_sp.add_argument("--type-id", type=int, required=True)
    p_sp.add_argument("--amount", type=int, required=True)

    # raw post
    p_raw = sub.add_parser("raw", help="POST arbitrary JSON to a URL")
    p_raw.add_argument("--url", required=True)
    p_raw.add_argument("--json", required=True, help="JSON string payload (single quotes OK)")

    return p

def print_examples() -> None:
    print(
        "\nExamples:\n"
        "  python main.py train --troop foot --qty 1\n"
        "  python main.py train --troop archer --qty 25 --per 1 --concurrent 5\n"
        "  python main.py build --building barracks --count 3 --concurrent 3\n"
        "  python main.py speedup --type-id 134207 --amount 50\n"
        "  python main.py raw --url https://example/TrainPopulation "
        "--json '{\"accountId\":\"32\",\"token\":\"...\",\"kingdomId\":41,\"popTypeId\":20,\"quantity\":1}'\n"
    )

# ---------- Runners ----------
async def run_train(api: ApiClient, troop: str, qty: int, per: int, concurrent: int, pop_type_id: Optional[int]) -> None:
    if troop == "custom":
        if not pop_type_id:
            raise SystemExit("--pop-type-id is required when --troop custom")
        ptid = pop_type_id
        troop_name = f"custom({ptid})"
    else:
        if troop not in TROOP_TO_POPTYPE:
            raise SystemExit(f"Unknown troop: {troop}. Known: {', '.join(TROOP_TO_POPTYPE)}")
        ptid = TROOP_TO_POPTYPE[troop]
        troop_name = troop

    async def worker(chunk_qty: int) -> InnerReturn:
        return await api.train_population(ptid, chunk_qty)

    await fan_out(
        total=qty,
        per_request=per,
        concurrency=concurrent,
        worker=worker,
        label=f"train {troop_name}",
    )

async def run_build(api: ApiClient, building: str, count: int, per: int, concurrent: int, building_type_id: Optional[int]) -> None:
    if building == "custom":
        if not building_type_id:
            raise SystemExit("--building-type-id is required when --building custom")
        btid = building_type_id
        bname = f"custom({btid})"
    else:
        if building not in BUILDING_TO_TYPEID:
            raise SystemExit(f"Unknown building: {building}. Known: {', '.join(BUILDING_TO_TYPEID)}")
        btid = BUILDING_TO_TYPEID[building]
        bname = building

    async def worker(chunk_count: int) -> InnerReturn:
        return await api.build_building(btid, chunk_count)

    await fan_out(
        total=count,
        per_request=per,
        concurrency=concurrent,
        worker=worker,
        label=f"build {bname}",
    )

async def run_speedup(api: ApiClient, type_id: int, amount: int) -> None:
    try:
        ir = await api.generic_action(type_code="S", type_id=type_id, amount=amount)
        log.info("Speedup ok → ReturnValue=%s msg=%s", ir.ReturnValue, ir.ReturnString or "<ok>")
    except AlreadyUsedError as e:
        log.warning("Speedup blocked: %s", e)
    except ApiError as e:
        log.error("Server error: %s", e)

async def run_raw(api: ApiClient, url: str, payload_str: str) -> None:
    payload_str = payload_str.strip().replace("'", '"')
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON: {e}")
    data = await api.post_json(url, payload)
    print(json.dumps(data, indent=2))

# ---------- Entry ----------
async def main_async() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    # If no subcommand, try START_CMD from env; otherwise either idle or print help.
    if not args.cmd:
        start_cmd = os.getenv("START_CMD", "").strip()
        if start_cmd:
            log.info("No CLI args provided. Using START_CMD from .env: %s", start_cmd)
            # Re-parse as if the user typed: python main.py <START_CMD>
            args = parser.parse_args(shlex.split(start_cmd))
        else:
            if os.getenv("KEEP_ALIVE", "0") == "1":
                parser.print_help()
                print_examples()
                log.info("KEEP_ALIVE=1 set. Idling because no command was provided…")
                # Sleep forever without exiting so your host doesn't restart the process.
                await asyncio.Event().wait()
                return
            else:
                # Quiet exit when no command is supplied to avoid repeated help spam.
                log.info("No command provided. Exiting.")
                return

    async with httpx.AsyncClient(
        headers={
            "User-Agent": USER_AGENT,
            "Origin": ORIGIN_URL,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        },
        http2=HTTP2_ENABLED,
    ) as client:
        api = ApiClient(client=client)

        if args.cmd == "train":
            await run_train(
                api,
                troop=args.troop.lower(),
                qty=args.qty,
                per=args.per,
                concurrent=args.concurrent,
                pop_type_id=args.pop_type_id,
            )
        elif args.cmd == "build":
            await run_build(
                api,
                building=args.building.lower(),
                count=args.count,
                per=args.per,
                concurrent=args.concurrent,
                building_type_id=args.building_type_id,
            )
        elif args.cmd == "speedup":
            await run_speedup(api, type_id=args.type_id, amount=args.amount)
        elif args.cmd == "raw":
            await run_raw(api, url=args.url, payload_str=args.json)
        else:
            parser.print_help()
            print_examples()
            return


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log.info("Exiting…")
    except Exception as e:
        log.exception("Fatal: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
