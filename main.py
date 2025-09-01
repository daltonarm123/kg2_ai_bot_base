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
    log.info("HTTP/2 support enabled")
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

# Use environment variables (required for production)
BASE_URL   = env("BASE_URL", required=True).rstrip("/")
ACCOUNT_ID = env("ACCOUNT_ID", required=True)
TOKEN      = env("TOKEN", required=True)
try:
    KINGDOM_ID = int(env("KINGDOM_ID", required=True))
except ValueError:
    raise RuntimeError("KINGDOM_ID must be a valid integer")

# Login credentials for the game
USERNAME = env("USERNAME", "osrs7214@gmail.com")  # Your actual game email
PASSWORD = env("PASSWORD", "Armstrong7397!")  # Your actual game password

# Optional referer URLs for requests that require them
REFERER_OVERVIEW  = env("REFERER_OVERVIEW",  f"{BASE_URL}/overview")
REFERER_BUILDINGS = env("REFERER_BUILDINGS", f"{BASE_URL}/buildings")
REFERER_WAR       = env("REFERER_WAR",       f"{BASE_URL}/warroom")
REFERER_RESEARCH  = env("REFERER_RESEARCH",  f"{BASE_URL}/research")

# Endpoints (adjust if your API uses a sub-path)
# Try different endpoint patterns - the API might use a different structure
TRAIN_POPULATION_ENDPOINT = f"{BASE_URL}/api/TrainPopulation"  # Try /api/ prefix
BUILD_ENDPOINT            = f"{BASE_URL}/api/BuildBuilding"     # Try /api/ prefix
GENERIC_ACTION_ENDPOINT   = f"{BASE_URL}/api/Action"           # Try /api/ prefix

# Fallback endpoints if the above don't work
FALLBACK_TRAIN_ENDPOINT = f"{BASE_URL}/TrainPopulation"
FALLBACK_BUILD_ENDPOINT = f"{BASE_URL}/BuildBuilding"
FALLBACK_ACTION_ENDPOINT = f"{BASE_URL}/Action"

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
    _endpoint_cache: Dict[str, str] = None
    _session_valid: bool = True
    _last_successful_request: float = 0

    def __post_init__(self):
        if self._endpoint_cache is None:
            self._endpoint_cache = {}
    
    async def login_to_game(self) -> bool:
        """Attempt to login to the game using account credentials"""
        try:
            log.info("Attempting to login to the game...")
            
            # First, get the login page to extract any CSRF tokens or session cookies
            login_page_url = f"{BASE_URL}/login"
            r = await self.client.get(login_page_url, timeout=HTTP_TIMEOUT)
            
            if r.status_code != 200:
                log.warning(f"Failed to access login page: {r.status_code}")
                return False
            
            # Try to find login form and submit credentials
            # Use actual username/password if provided, otherwise fall back to account ID/token
            username = USERNAME if USERNAME else ACCOUNT_ID
            password = PASSWORD if PASSWORD else TOKEN
            
            login_data = {
                "username": username,
                "password": password,
                "email": username,
                "accountId": ACCOUNT_ID,
                "token": TOKEN,
                "login": username,
                "user": username,
                "account": ACCOUNT_ID,
            }
            
            # Try different login endpoints
            login_endpoints = [
                f"{BASE_URL}/login",
                f"{BASE_URL}/api/login", 
                f"{BASE_URL}/auth/login",
                f"{BASE_URL}/user/login",
                f"{BASE_URL}/account/login"
            ]
            
            for endpoint in login_endpoints:
                try:
                    log.info(f"Trying login endpoint: {endpoint}")
                    
                    # Try POST first
                    r = await self.client.post(endpoint, data=login_data, timeout=HTTP_TIMEOUT)
                    if r.status_code == 200:
                        log.info(f"Login successful via POST to {endpoint}")
                        return True
                    
                    # Try GET with parameters
                    r = await self.client.get(endpoint, params=login_data, timeout=HTTP_TIMEOUT)
                    if r.status_code == 200:
                        log.info(f"Login successful via GET to {endpoint}")
                        return True
                        
                except Exception as e:
                    log.debug(f"Login attempt failed for {endpoint}: {e}")
                    continue
            
            # If direct login doesn't work, try using the existing credentials in a different way
            log.info("Direct login failed, trying credential validation...")
            return await self.validate_credentials()
            
        except Exception as e:
            log.warning(f"Login attempt failed: {e}")
            return False

    async def validate_credentials(self) -> bool:
        """Validate that our credentials work by trying a simple API call"""
        try:
            # Try a simple API call that might work with just credentials
            test_url = f"{BASE_URL}/api/account"
            test_data = {
                "accountId": ACCOUNT_ID,
                "token": TOKEN
            }
            
            # Try different approaches
            for method in ["GET", "POST"]:
                try:
                    if method == "GET":
                        r = await self.client.get(test_url, params=test_data, timeout=HTTP_TIMEOUT)
                    else:
                        r = await self.client.post(test_url, json=test_data, timeout=HTTP_TIMEOUT)
                    
                    if r.status_code == 200:
                        response_text = r.text
                        if not (response_text.strip().startswith('<!DOCTYPE html>') or '<html>' in response_text):
                            log.info(f"Credentials validated successfully via {method}")
                            return True
                            
                except Exception as e:
                    log.debug(f"Credential validation failed via {method}: {e}")
                    continue
            
            log.warning("Credential validation failed - credentials may be invalid")
            return False
            
        except Exception as e:
            log.warning(f"Credential validation failed: {e}")
            return False

    async def validate_session(self) -> bool:
        """Check if the current session is still valid by making a simple request"""
        try:
            # Try a simple request to see if we get HTML (session expired) or JSON (session valid)
            test_url = f"{BASE_URL}/api/TrainPopulation"
            test_payload = {
                "accountId": ACCOUNT_ID,
                "token": TOKEN,
                "kingdomId": KINGDOM_ID,
                "popTypeId": 17,
                "quantity": 0  # Try with 0 quantity to avoid actually training
            }
            
            r = await self.client.get(test_url, params=test_payload, timeout=HTTP_TIMEOUT)
            response_text = r.text
            
            if response_text.strip().startswith('<!DOCTYPE html>') or '<html>' in response_text:
                log.warning("Session validation failed - received HTML response")
                self._session_valid = False
                return False
            else:
                log.debug("Session validation successful")
                self._session_valid = True
                self._last_successful_request = asyncio.get_event_loop().time()
                return True
                
        except Exception as e:
            log.warning(f"Session validation failed: {e}")
            self._session_valid = False
            return False

    @retry(
        stop=stop_after_attempt(RETRIES),
        wait=wait_exponential_jitter(initial=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, ApiError)),
        reraise=True,
    )
    async def post_json(self, url: str, payload: Dict[str, Any], referer: Optional[str] = None, 
                       fallback_url: Optional[str] = None) -> Dict[str, Any]:
        headers = {"Referer": referer} if referer else None
        
        try:
            # Try POST first
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
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 405:
                # Try GET with query parameters as fallback
                log.info(f"POST failed with 405, trying GET with query parameters: {url}")
                params = {k: str(v) for k, v in payload.items()}
                r = await self.client.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                
                # Try to parse as JSON first
                try:
                    data = r.json()
                    if isinstance(data, dict) and "d" in data and isinstance(data["d"], str):
                        try:
                            inner = json.loads(data["d"])
                            if isinstance(inner, dict) and "ReturnValue" in inner:
                                return {"ReturnValue": inner.get("ReturnValue", -1), "ReturnString": inner.get("ReturnString", "")}
                            return inner
                        except json.JSONDecodeError:
                            return data
                    return data
                except:
                    # If not JSON, check if it's HTML (game loading page)
                    response_text = r.text
                    if response_text.strip().startswith('<!DOCTYPE html>') or '<html>' in response_text:
                        log.warning("Received HTML response - game may be loading or session expired")
                        return {"ReturnValue": 1, "ReturnString": "Game session expired or loading"}
                    # Return the text response for other non-JSON responses
                    return {"ReturnValue": 0, "ReturnString": response_text}
            elif fallback_url and fallback_url != url:
                log.info(f"Endpoint {url} failed, trying fallback: {fallback_url}")
                # Try the fallback URL
                r = await self.client.post(fallback_url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                # Cache the working endpoint
                self._endpoint_cache[url] = fallback_url
                # Unwrap if needed
                if isinstance(data, dict) and "d" in data and isinstance(data["d"], str):
                    try:
                        inner = json.loads(data["d"])
                        if isinstance(inner, dict) and "ReturnValue" in inner:
                            return {"ReturnValue": inner.get("ReturnValue", -1), "ReturnString": inner.get("ReturnString", "")}
                        return inner
                    except json.JSONDecodeError:
                        return data
                return data
            else:
                raise

    # ---- Domain helpers ----
    async def train_population(self, pop_type_id: int, qty: int) -> InnerReturn:
        req = TrainPopulationReq(
            accountId=ACCOUNT_ID,
            token=TOKEN,
            kingdomId=KINGDOM_ID,
            popTypeId=pop_type_id,
            quantity=qty,
        ).model_dump()
        resp = await self.post_json(TRAIN_POPULATION_ENDPOINT, req, referer=REFERER_WAR, 
                                  fallback_url=FALLBACK_TRAIN_ENDPOINT)
        ir = self._normalize(resp)
        self._raise_if_error(ir)
        return ir

    async def build_building(self, building_type_id: int, qty: int = 1) -> InnerReturn:
        req = BuildReq(
            accountId=ACCOUNT_ID,
            token=TOKEN,
            kingdomId=KINGDOM_ID,
            buildingTypeId=building_type_id,
            quantity=qty,
        ).model_dump()
        resp = await self.post_json(BUILD_ENDPOINT, req, referer=REFERER_BUILDINGS,
                                  fallback_url=FALLBACK_BUILD_ENDPOINT)
        ir = self._normalize(resp)
        self._raise_if_error(ir)
        return ir

    async def generic_action(self, type_code: str, type_id: int, amount: int) -> InnerReturn:
        req = GenericActionReq(
            accountId=ACCOUNT_ID,
            token=TOKEN,
            kingdomId=KINGDOM_ID,
            type=type_code,
            typeId=type_id,
            amount=amount,
        ).model_dump()
        resp = await self.post_json(GENERIC_ACTION_ENDPOINT, req, referer=REFERER_RESEARCH,
                                  fallback_url=FALLBACK_ACTION_ENDPOINT)
        ir = self._normalize(resp)
        if "already used a speedup" in (ir.ReturnString or "").lower():
            raise AlreadyUsedError(ir.ReturnString)
        self._raise_if_error(ir)
        return ir

    @staticmethod
    def _normalize(resp: Dict[str, Any]) -> InnerReturn:
        try:
            if "ReturnValue" in resp:
                return InnerReturn(**resp)
            msg = resp.get("message") or resp.get("error") or str(resp)
            return InnerReturn(ReturnValue=0 if resp.get("ok") else 1, ReturnString=msg)
        except Exception as e:
            # Fallback for malformed responses
            return InnerReturn(ReturnValue=1, ReturnString=f"Failed to parse response: {e}")

    @staticmethod
    def _raise_if_error(ir: InnerReturn) -> None:
        if ir.ReturnValue >= 2:
            raise ApiError(f"Server error {ir.ReturnValue}: {ir.ReturnString or '<no message>'}")
    
    async def discover_endpoints(self) -> None:
        """Test different endpoint patterns to find working ones"""
        log.info("Discovering working API endpoints...")
        
        # Test patterns - try various common API endpoint structures
        test_patterns = [
            # Standard API patterns
            f"{BASE_URL}/api/TrainPopulation",
            f"{BASE_URL}/api/train",
            f"{BASE_URL}/api/population/train",
            
            # Direct endpoints
            f"{BASE_URL}/TrainPopulation", 
            f"{BASE_URL}/train",
            f"{BASE_URL}/population/train",
            
            # Game-specific patterns
            f"{BASE_URL}/game/TrainPopulation",
            f"{BASE_URL}/game/train",
            f"{BASE_URL}/game/population/train",
            f"{BASE_URL}/game/ajax/TrainPopulation",
            
            # AJAX patterns
            f"{BASE_URL}/ajax/TrainPopulation",
            f"{BASE_URL}/ajax/train",
            
            # Kingdom-specific patterns
            f"{BASE_URL}/kingdom/TrainPopulation",
            f"{BASE_URL}/kingdom/train",
            
            # Alternative patterns
            f"{BASE_URL}/action/TrainPopulation",
            f"{BASE_URL}/action/train",
            f"{BASE_URL}/cmd/TrainPopulation",
            f"{BASE_URL}/cmd/train",
            
            # Try with different HTTP methods
            f"{BASE_URL}/TrainPopulation",
            f"{BASE_URL}/train",
            f"{BASE_URL}/population/train"
        ]
        
        test_payload = {
            "accountId": ACCOUNT_ID,
            "token": TOKEN,
            "kingdomId": KINGDOM_ID,
            "popTypeId": 17,  # foot
            "quantity": 1
        }
        
        # Test both POST and GET methods
        http_methods = ["POST", "GET"]
        
        for method in http_methods:
            log.info(f"Testing {method} method...")
            for pattern in test_patterns:
                try:
                    log.info(f"Testing {method} {pattern}")
                    
                    if method == "POST":
                        r = await self.client.post(pattern, json=test_payload, timeout=5.0)
                    else:  # GET
                        # For GET, try with query parameters
                        params = {k: str(v) for k, v in test_payload.items()}
                        r = await self.client.get(pattern, params=params, timeout=5.0)
                    
                    if r.status_code not in [405, 404, 501, 500]:
                        log.info(f"✅ Working endpoint found: {method} {pattern} (status: {r.status_code})")
                        
                        # Try to parse the response to see if it's valid
                        try:
                            data = r.json()
                            log.info(f"Response data: {data}")
                        except:
                            log.info(f"Response text: {r.text[:200]}...")
                        
                        # Update the working endpoint
                        if "TrainPopulation" in pattern or "train" in pattern.lower():
                            global TRAIN_POPULATION_ENDPOINT
                            TRAIN_POPULATION_ENDPOINT = pattern
                            log.info(f"Updated TRAIN_POPULATION_ENDPOINT to: {pattern}")
                        
                        return  # Found a working endpoint
                    else:
                        log.info(f"❌ {method} {pattern} returned {r.status_code}")
                        
                except Exception as e:
                    log.info(f"❌ {method} {pattern} failed: {type(e).__name__}")
                    continue
        
        log.warning("No working endpoints found. You may need to manually discover the correct API structure.")
        log.info("Try visiting the game website and inspecting network requests to find the correct endpoints.")
        
        # Try to examine the website structure
        try:
            log.info("Attempting to examine website structure...")
            r = await self.client.get(f"{BASE_URL}/", timeout=10.0)
            if r.status_code == 200:
                html = r.text
                log.info(f"Website loaded successfully. Looking for API endpoints...")
                
                # Look for common patterns in the HTML
                import re
                
                # Look for AJAX endpoints
                ajax_patterns = re.findall(r'["\']([^"\']*ajax[^"\']*)["\']', html, re.IGNORECASE)
                if ajax_patterns:
                    log.info(f"Found potential AJAX endpoints in HTML: {ajax_patterns[:5]}")
                
                # Look for form actions
                form_patterns = re.findall(r'action=["\']([^"\']*)["\']', html, re.IGNORECASE)
                if form_patterns:
                    log.info(f"Found form actions: {form_patterns[:5]}")
                
                # Look for JavaScript API calls
                js_patterns = re.findall(r'["\']([^"\']*api[^"\']*)["\']', html, re.IGNORECASE)
                if js_patterns:
                    log.info(f"Found potential API endpoints in JavaScript: {js_patterns[:5]}")
                    
        except Exception as e:
            log.info(f"Could not examine website structure: {e}")

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
                elif "session expired" in msg.lower() or "loading" in msg.lower():
                    log.error("Game session expired or loading - stopping all operations")
                    return (i, "session-stop")
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
            # Cancel remaining tasks and wait for them to finish
            for other in running:
                if not other.done():
                    other.cancel()
            # Wait for all tasks to complete (including canceled ones)
            await asyncio.gather(*running, return_exceptions=True)
            break
        elif stop_reason == "session-stop":
            log.error("Stopping all operations due to session expiration.")
            # Cancel remaining tasks and wait for them to finish
            for other in running:
                if not other.done():
                    other.cancel()
            # Wait for all tasks to complete (including canceled ones)
            await asyncio.gather(*running, return_exceptions=True)
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

    # test endpoints
    p_test = sub.add_parser("test-endpoints", help="Test different API endpoint patterns")

    return p

def print_examples() -> None:
    print(
        "\nExamples:\n"
        "  python main.py train --troop foot --qty 1\n"
        "  python main.py train --troop archer --qty 25 --per 1 --concurrent 5\n"
        "  python main.py build --building barracks --count 3 --concurrent 3\n"
        "  python main.py speedup --type-id 134207 --amount 50\n"
        "  python main.py test-endpoints\n"
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

async def run_test_endpoints(api: ApiClient) -> None:
    """Test different endpoint patterns to find working ones"""
    await api.discover_endpoints()

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

    # Create HTTP client with proper HTTP/2 handling
    client_kwargs = {
        "headers": {"User-Agent": "kg2-ai/1.3"},
        "timeout": HTTP_TIMEOUT,
        "limits": httpx.Limits(max_connections=20, max_keepalive_connections=10)
    }
    
    if HTTP2_ENABLED:
        client_kwargs["http2"] = True
        log.info("Creating HTTP/2 client")
    else:
        log.info("Creating HTTP/1.1 client")

    async with httpx.AsyncClient(**client_kwargs) as client:
        api = ApiClient(client=client)
        
        # Auto-discover working endpoints if this is the first run
        if not hasattr(api, '_endpoints_discovered'):
            try:
                await api.discover_endpoints()
                api._endpoints_discovered = True
            except Exception as e:
                log.warning(f"Endpoint discovery failed: {e}")

        # Check if we should keep alive and loop
        keep_alive = os.getenv("KEEP_ALIVE", "0") == "1"
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while True:
            try:
                # Try to login/validate session before attempting operations
                if not await api.validate_session():
                    log.warning("Session invalid, attempting to login...")
                    if not await api.login_to_game():
                        log.error("Failed to login to game")
                        if keep_alive:
                            log.warning("Waiting 300 seconds before retry...")
                            await asyncio.sleep(300)
                            continue
                        else:
                            break
                
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
                elif args.cmd == "test-endpoints":
                    await run_test_endpoints(api)
                else:
                    parser.print_help()
                    print_examples()
                    return
                
                # If keep_alive is disabled, break after one run
                if not keep_alive:
                    break
                    
                # If keep_alive is enabled, wait before next run
                log.info("KEEP_ALIVE=1 enabled. Waiting 60 seconds before next run...")
                await asyncio.sleep(60)
                
            except Exception as e:
                consecutive_failures += 1
                error_msg = str(e).lower()
                
                if "session expired" in error_msg or "loading" in error_msg or "invalid" in error_msg:
                    if keep_alive:
                        if consecutive_failures >= max_consecutive_failures:
                            log.error(f"Too many consecutive failures ({consecutive_failures}). Waiting 600 seconds (10 minutes) before retry...")
                            await asyncio.sleep(600)  # Wait 10 minutes for persistent failures
                            consecutive_failures = 0  # Reset counter after long wait
                        else:
                            log.warning(f"Session issue (failure #{consecutive_failures}). Waiting 300 seconds (5 minutes) before retry...")
                            await asyncio.sleep(300)  # Wait 5 minutes for session issues
                        continue
                    else:
                        log.error("Session expired and KEEP_ALIVE=0. Exiting.")
                        break
                else:
                    # For other errors, wait shorter time and retry
                    if keep_alive:
                        log.warning(f"Operation failed (failure #{consecutive_failures}): {e}")
                        if consecutive_failures >= max_consecutive_failures:
                            log.error(f"Too many consecutive failures ({consecutive_failures}). Waiting 600 seconds (10 minutes) before retry...")
                            await asyncio.sleep(600)
                            consecutive_failures = 0
                        else:
                            await asyncio.sleep(60)  # Wait 1 minute for other errors
                        continue
                    else:
                        # Re-raise other exceptions if not keeping alive
                        raise


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
