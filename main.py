"""
main.py — KG2 AI Bot with Fixed API Integration

Features:
- Fixed GET-based API calls for kingdomgame.net
- Proper session management and authentication
- Endpoint discovery to find working API paths
- Browser-based login fallback with Playwright
- Comprehensive error handling and retry logic
- Async concurrent operations with resource checking
- Environment-driven configuration

Install:
    pip install "httpx[http2]" pydantic tenacity python-dotenv playwright

Setup:
    playwright install chromium

Examples:
    python main.py train --troop foot --qty 1
    python main.py train --troop archer --qty 25 --per 1 --concurrent 5
    python main.py build --building barracks --count 3 --concurrent 3
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
import re
import threading
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable, Tuple
from urllib.parse import urlencode, parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Try to import playwright for browser automation
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    async_playwright = None
    PLAYWRIGHT_AVAILABLE = False

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

# Optional HTTP/2 support
try:
    import h2  # noqa: F401
    HTTP2_ENABLED = True
    log.info("HTTP/2 support enabled")
except ModuleNotFoundError:
    HTTP2_ENABLED = False
    log.warning("h2 package not installed; HTTP/2 disabled. Install httpx[http2] to enable.")

# ---------- Config ----------
load_dotenv()

def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val or ""

# Environment variables (using KG2_ prefix as in user's .env)
BASE_URL = env("KG2_BASE_URL", required=True).rstrip("/")
ACCOUNT_ID = env("KG2_ACCOUNT_ID", required=True)
TOKEN = env("KG2_TOKEN", required=True)
try:
    KINGDOM_ID = int(env("KG2_KINGDOM_ID", required=True))
except ValueError:
    raise RuntimeError("KG2_KINGDOM_ID must be a valid integer")

# Login credentials
USERNAME = env("KG2_USERNAME", "osrs7214@gmail.com")
PASSWORD = env("KG2_PASSWORD", "Armstrong7397!")

# Optional settings from user's .env
WORLD_ID = env("KG2_WORLD_ID", "1")
MAX_RPS = float(env("KG2_MAX_RPS", "0.4"))
MOCK_MODE = env("KG2_MOCK", "false").lower() == "true"
USE_BROWSER_HEADERS = env("KG2_USE_BROWSER_HEADERS", "true").lower() == "true"

# Configuration
# Configuration from .env or defaults
HTTP_TIMEOUT = float(env("KG2_TIMEOUT_S", "20.0"))
RETRIES = 4
DEFAULT_CONCURRENCY = 5
CALL_SPACING_SEC = 1.0 / MAX_RPS if MAX_RPS > 0 else 0.05  # Respect user's rate limit
SESSION_CHECK_INTERVAL = 300  # Check session validity every 5 minutes

# AI Strategy Configuration
MIN_RANDOM_DELAY = int(env("MIN_RANDOM_DELAY", "30"))  # 30 seconds
MAX_RANDOM_DELAY = int(env("MAX_RANDOM_DELAY", "300")) # 5 minutes
AGGRESSIVENESS_LEVEL = float(env("AGGRESSIVENESS", "0.7"))  # 0.0-1.0
RESOURCE_THRESHOLD = float(env("RESOURCE_THRESHOLD", "0.8"))  # Use 80% of resources
SPY_FREQUENCY = int(env("SPY_FREQUENCY", "3"))  # Spy every 3 actions
EXPLORE_FREQUENCY = int(env("EXPLORE_FREQUENCY", "5"))  # Explore every 5 actions

# AI Action Tracking
completed_actions = set()  # Track completed actions to avoid repeating
action_history = []  # Track recent actions for variety

# Troop type mappings
TROOP_TYPES = {
    "foot": 17,
    "archer": 18,
    "cavalry": 19,
    "siege": 20,
}

# Spy action types
SPY_TYPES = {
    "infiltrate": 1,
    "sabotage": 2,
    "gather_intel": 3,
    "steal_resources": 4,
    "assassinate": 5,
}

# Exploration types
EXPLORE_TYPES = {
    "scout": 1,
    "expand": 2,
    "search_resources": 3,
    "find_enemies": 4,
}

# Attack types
ATTACK_TYPES = {
    "raid": 1,
    "siege": 2,
    "pillage": 3,
    "conquer": 4,
}

# Building type mappings  
BUILDING_TYPES = {
    "Barracks": 1,
    "Archery Ranges": 2,
    "Stables": 3,
    "Guildhalls": 4,
    "Castles": 5,
    "Houses": 6,
    "Grain Farms": 7,
    "Barns": 8,
    "Markets": 9,
    "Embassies": 10,
    "Horse Farms": 11,
    "Temples": 12,
    "Lumber Yards": 13,
    "Stone Quarries": 14,
}

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

class SpyReq(BaseModel):
    accountId: str
    token: str
    kingdomId: int
    targetKingdomId: int
    spyTypeId: int
    spyCount: int = 1

class ExploreReq(BaseModel):
    accountId: str
    token: str
    kingdomId: int
    exploreTypeId: int
    direction: str = "north"  # north, south, east, west
    distance: int = 1

class AttackReq(BaseModel):
    accountId: str
    token: str
    kingdomId: int
    targetKingdomId: int
    attackTypeId: int
    troopCount: int
    
    @field_validator("troopCount")
    @classmethod
    def nonzero_troops(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("troopCount must be positive")
        return v

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

class SessionError(ApiError):
    pass


# ---------- Health Check Server ----------
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        # Silence the default logging
        return

def run_health_check_server():
    """Run a simple HTTP server for health checks."""
    try:
        server_address = ('', 8080)
        httpd = HTTPServer(server_address, HealthCheckHandler)
        log.info("Health check server running on port 8080")
        httpd.serve_forever()
    except Exception as e:
        log.error(f"Health check server failed: {e}")


# ---------- HTTP Client ----------
@dataclass
class ApiClient:
    client: httpx.AsyncClient
    _endpoint_cache: Dict[str, str] = field(default_factory=dict)
    _session_valid: bool = True
    _last_successful_request: float = 0
    _working_endpoints: Dict[str, str] = field(default_factory=dict)

    async def discover_endpoints(self) -> None:
        """Discover working API endpoints by testing different patterns"""
        log.info("Discovering working API endpoints...")
        
        # Based on user's API information, these are the correct endpoints
        self._working_endpoints = {
            # Updated from user list
            "TrainPopulation": f"{BASE_URL}/Population.asmx/TrainPopulation",
            "BuildBuilding": f"{BASE_URL}/Buildings.asmx/BuildBuilding",
            "Explore": f"{BASE_URL}/Kingdoms.asmx/Explore",
            "Attack": f"{BASE_URL}/Kingdoms.asmx/AttackKingdom",
            "Spy": f"{BASE_URL}/Kingdoms.asmx/SpyOnKingdom",

            # Kept from old list
            "Action": f"{BASE_URL}/api/Action",
            "SpyAction": f"{BASE_URL}/api/SpyAction",
            "ExploreTerritory": f"{BASE_URL}/api/ExploreTerritory",
            "Raid": f"{BASE_URL}/api/Raid",
            "Battle": f"{BASE_URL}/api/Battle",
            "GetKingdomInfo": f"{BASE_URL}/api/GetKingdomInfo",
            "GetPlayerList": f"{BASE_URL}/api/GetPlayerList",
            "GetResources": f"{BASE_URL}/api/GetResources",
            "Login": f"{BASE_URL}/login",
            "ApiLogin": f"{BASE_URL}/api/login",
            "Auth": f"{BASE_URL}/api/auth",
            "Account": f"{BASE_URL}/api/account",
            "User": f"{BASE_URL}/api/user",
            "Kingdom": f"{BASE_URL}/api/kingdom",
            "Game": f"{BASE_URL}/api/game",

            # --- Added Endpoints from user list ---

            # Authentication (User.asmx)
            "UserAppLogin": f"{BASE_URL}/User.asmx/AppLogin",
            "UserAppRegister": f"{BASE_URL}/User.asmx/AppRegister",
            "UserCheckToken": f"{BASE_URL}/User.asmx/CheckToken",
            "UserGetKingdomIdForAccount": f"{BASE_URL}/User.asmx/GetKingdomIdForAccount",
            "UserRequestAccountTransfer": f"{BASE_URL}/User.asmx/RequestAccountTransfer",
            "UserTransferAccount": f"{BASE_URL}/User.asmx/TransferAccount",
            "UserGetSubscriptionDetails": f"{BASE_URL}/User.asmx/GetSubscriptionDetails",

            # Worlds (Worlds.asmx)
            "WorldsGetWorlds": f"{BASE_URL}/Worlds.asmx/GetWorlds",
            "WorldsGetAlluviaStatus": f"{BASE_URL}/Worlds.asmx/GetAlluviaStatus",
            "WorldsGetZethraStatus": f"{BASE_URL}/Worlds.asmx/GetZethraStatus",

            # Payments & Subscriptions (Payments.asmx)
            "PaymentsFulfillmentHandler": f"{BASE_URL}/Payments.asmx/FulfillmentHandler",
            "PaymentsValidateSubscription": f"{BASE_URL}/Payments.asmx/ValidateSubscription",

            # Push Notifications (PushNotifications.asmx)
            "PushNotificationsCreateOrUpdateInstallation": f"{BASE_URL}/PushNotifications.asmx/CreateOrUpdateInstallation",

            # Kingdoms (Kingdoms.asmx)
            "KingdomsGetKingdomDetails": f"{BASE_URL}/Kingdoms.asmx/GetKingdomDetails",
            "KingdomsGetKingdomResources": f"{BASE_URL}/Kingdoms.asmx/GetKingdomResources",
            "KingdomsGetKingdomPopulation": f"{BASE_URL}/Kingdoms.asmx/GetKingdomPopulation",
            "KingdomsGetKingdomBuildings": f"{BASE_URL}/Kingdoms.asmx/GetKingdomBuildings",
            "KingdomsGetInProgressItems": f"{BASE_URL}/Kingdoms.asmx/GetInProgressItems",
            "KingdomsGetReturningForces": f"{BASE_URL}/Kingdoms.asmx/GetReturningForces",
            "KingdomsSearchByName": f"{BASE_URL}/Kingdoms.asmx/SearchByName",
            "KingdomsAttackKingdom": f"{BASE_URL}/Kingdoms.asmx/AttackKingdom",
            "KingdomsSpyOnKingdom": f"{BASE_URL}/Kingdoms.asmx/SpyOnKingdom",

            # Population (Population.asmx)
            "PopulationGetPopulationTypeList": f"{BASE_URL}/Population.asmx/GetPopulationTypeList",
            "PopulationGetTrainingCost": f"{BASE_URL}/Population.asmx/GetTrainingCost",

            # Buildings (Buildings.asmx)
            "BuildingsGetBuildingsTypeList": f"{BASE_URL}/Buildings.asmx/GetBuildingsTypeList",
            "BuildingsGetBuildingCost": f"{BASE_URL}/Buildings.asmx/GetBuildingCost",
            "BuildingsCancelBuilding": f"{BASE_URL}/Buildings.asmx/CancelBuilding",

            # Messages (Messages.asmx)
            "MessagesGetMessages": f"{BASE_URL}/Messages.asmx/GetMessages",
            "MessagesGetMessage": f"{BASE_URL}/Messages.asmx/GetMessage",
            "MessagesMarkMessageReadUnread": f"{BASE_URL}/Messages.asmx/MarkMessageReadUnread",
        }
        
        # Test endpoints only if we have an active session
        if await self.validate_session():
            log.info("✅ Endpoints configured successfully with valid session")
        else:
            log.info("📝 Endpoints configured, session validation needed")

    async def validate_session(self) -> bool:
        """Validate if current session is still valid by testing game page access"""
        try:
            # Test session by accessing a game page that requires authentication
            test_params = {
                "accountId": ACCOUNT_ID,
                "token": TOKEN,
                "kingdomId": KINGDOM_ID
            }
            
            # Test with overview page first
            overview_url = f"{BASE_URL}/overview"
            headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
            
            response = await self.client.get(overview_url, params=test_params, headers=headers, timeout=HTTP_TIMEOUT)
            
            if response.status_code == 200:
                response_text = response.text.lower()
                
                # Check if we got actual game content (not login page)
                if ("overview" in response_text and 
                    "logout" in response_text and
                    "login" not in response_text):
                    log.info("Session validation successful - authenticated game page")
                    self._last_successful_request = time.time()
                    return True
                elif "login" in response_text:
                    log.warning("Session validation failed - redirected to login")
                    return False
                else:
                    # If we got some content, assume session is valid
                    log.info("Session validation - got game content")
                    self._last_successful_request = time.time()
                    return True
                    
            return False
            
        except Exception as e:
            log.warning(f"Session validation error: {e}")
            return False

    async def browser_login(self) -> bool:
        """Attempt to login using Playwright browser automation"""
        if not PLAYWRIGHT_AVAILABLE or async_playwright is None:
            log.warning("Playwright not available - cannot use browser login")
            return False
        
        try:
            log.info("Attempting browser-based login...")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Navigate to login page
                await page.goto(f"{BASE_URL}/")
                
                # Look for login form elements
                try:
                    # Try common login form selectors
                    email_selectors = [
                        'input[name="email"]',
                        'input[type="email"]',
                        'input[id="email"]',
                        'input[name="username"]',
                        '#email',
                        '#username'
                    ]
                    
                    password_selectors = [
                        'input[name="password"]',
                        'input[type="password"]',
                        'input[id="password"]',
                        '#password'
                    ]
                    
                    email_field = None
                    password_field = None
                    
                    for selector in email_selectors:
                        try:
                            email_field = await page.wait_for_selector(selector, timeout=2000)
                            if email_field:
                                break
                        except:
                            continue
                    
                    for selector in password_selectors:
                        try:
                            password_field = await page.wait_for_selector(selector, timeout=2000)
                            if password_field:
                                break
                        except:
                            continue
                    
                    if email_field and password_field:
                        # Fill login form
                        await email_field.fill(USERNAME)
                        await password_field.fill(PASSWORD)
                        
                        # Submit form
                        submit_selectors = [
                            'input[type="submit"]',
                            'button[type="submit"]',
                            'button:has-text("Login")',
                            'button:has-text("Sign In")',
                            '#login-button'
                        ]
                        
                        for selector in submit_selectors:
                            try:
                                submit_btn = await page.query_selector(selector)
                                if submit_btn:
                                    await submit_btn.click()
                                    break
                            except:
                                continue
                        
                        # Wait for navigation after login
                        await page.wait_for_timeout(3000)
                        
                        # Extract cookies for requests
                        cookies = await context.cookies()
                        for cookie in cookies:
                            domain = cookie.get('domain')
                            if domain:
                                self.client.cookies.set(cookie['name'], cookie['value'], domain=domain)
                            else:
                                self.client.cookies.set(cookie['name'], cookie['value'])
                        
                        log.info("Browser login completed, cookies extracted")
                        await browser.close()
                        return True
                        
                except Exception as e:
                    log.warning(f"Browser login form interaction failed: {e}")
                
                await browser.close()
                return False
                
        except Exception as e:
            log.warning(f"Browser login failed: {e}")
            return False
    
    async def browser_action(self, action_type: str, **params) -> bool:
        """Perform actual game actions using browser automation"""
        if not PLAYWRIGHT_AVAILABLE:
            log.error("❌ Playwright required for real game actions!")
            return False
        
        try:
            log.info(f"🎮 Performing REAL game action: {action_type}")
            
            if not async_playwright:
                log.error("❌ Playwright not available - cannot perform browser actions")
                return False
            
            async with async_playwright() as p:
                # Detect if running locally vs in cloud environment
                is_local = not os.getenv('REPLIT_ENVIRONMENT') and not os.getenv('CODESPACE_NAME')
                
                if is_local:
                    # Local execution: Use visible browser for debugging
                    log.info("🖥️ Running locally - using visible browser for debugging")
                    browser = await p.chromium.launch(
                        headless=False,  # Visible browser for local debugging
                        slow_mo=500,     # Slow down actions to see what's happening
                        args=['--start-maximized']
                    )
                else:
                    # Cloud environment: Use headless mode
                    log.info("☁️ Running in cloud - using headless browser")
                    browser = await p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox', 
                            '--disable-dev-shm-usage',
                            '--disable-gpu',
                            '--window-size=1920x1080'
                        ]
                    )
                context = await browser.new_context()
                page = await context.new_page()
                
                # Go directly to login page first
                login_url = f"{BASE_URL}/login"
                log.info(f"🔐 Starting login process at {login_url}")
                await page.goto(login_url)
                await page.wait_for_timeout(3000)
                
                # Perform login
                login_success = await self._browser_login_process(page)
                if not login_success:
                    log.error("❌ Browser login failed")
                    return False
                
                # Now navigate to the specific page needed for each action
                if action_type == "train":
                    # Training is at warroom (no action parameter)
                    page_url = f"{BASE_URL}/warroom"
                    log.info(f"🎮 Navigating to warroom for training: {page_url}")
                elif action_type == "explore":
                    # Exploring is at warroom with action=explore
                    page_url = f"{BASE_URL}/warroom?action=explore"
                    log.info(f"🎮 Navigating to warroom for exploring: {page_url}")
                elif action_type == "build":
                    # Building is at /buildings
                    page_url = f"{BASE_URL}/buildings"
                    log.info(f"🎮 Navigating to buildings page: {page_url}")
                elif action_type == "spy":
                    # Spying likely in guildhall or embassy
                    page_url = f"{BASE_URL}/embassy"
                    log.info(f"🎮 Navigating to embassy for spying: {page_url}")
                else:
                    # Default to overview for unknown actions
                    page_url = f"{BASE_URL}/overview"
                    log.info(f"🎮 Navigating to overview page: {page_url}")
                
                await page.goto(page_url)
                
                await page.wait_for_timeout(5000)  # Extra time for game to load
                
                # Instead of specific actions, start the autonomous AI loop
                return await self._autonomous_game_ai(page)

                await browser.close()
                return False
                
        except Exception as e:
            log.error(f"❌ Browser action {action_type} failed: {e}")
            return False
    
    async def _browser_train_troops(self, page, params: dict):
        """
        Train troops using a robust, multi-strategy browser automation approach.
        Handles standard <select> dropdowns and custom <div>-based dropdowns.
        """
        try:
            troop_type = params.get('troop_type', 'foot')
            quantity = params.get('quantity', 1)
            pop_type_id = params.get('pop_type_id')

            log.info(f"🎮 Browser training started for {quantity} {troop_type}(s).")
            await page.wait_for_timeout(3000) # Wait for page to settle

            # --- Strategy 1: Standard <select> dropdown ---
            log.info("🧠 Trying strategy 1: Standard <select> dropdown...")
            try:
                # Find all select elements and check their options
                select_elements = await page.query_selector_all('select')
                for select_element in select_elements:
                    # Try selecting by value (pop_type_id) first
                    if pop_type_id:
                        try:
                            await select_element.select_option(value=str(pop_type_id), timeout=1000)
                            log.info(f"✅ Selected troop by value '{pop_type_id}' in a <select> element.")
                            # Now fill quantity and submit
                            await self._fill_and_submit_training(page, quantity)
                            return True
                        except Exception:
                            pass # Value not found, try by label

                    # Try selecting by label (troop_type)
                    try:
                        await select_element.select_option(label=re.compile(troop_type, re.IGNORECASE), timeout=1000)
                        log.info(f"✅ Selected troop by label '{troop_type}' in a <select> element.")
                        await self._fill_and_submit_training(page, quantity)
                        return True
                    except Exception:
                        continue # Option not in this select, try next
            except Exception as e:
                log.warning(f"⚠️ Strategy 1 failed: {e}")

            # --- Strategy 2: Custom clickable elements (div, button, a) ---
            log.info("🧠 Trying strategy 2: Custom clickable elements...")
            try:
                # Find a clickable element that opens the troop selection
                # This could be a button with text "Select Troop" or the current troop type
                possible_triggers = await page.query_selector_all(
                    'button:has-text("Select"),'
                    'button:has-text("Troop"),'
                    f'button:has-text("{troop_type}")'
                )
                if not possible_triggers:
                    # A more generic search for any clickable element
                    possible_triggers = await page.query_selector_all('div[role="button"], button')

                for trigger in possible_triggers:
                    try:
                        await trigger.click(timeout=1000)
                        await page.wait_for_timeout(1000) # wait for dropdown to open
                        log.info("🖱️ Clicked a potential dropdown trigger.")

                        # Now find the troop option in the opened dropdown
                        # Options might be buttons, divs, or list items
                        troop_option_selector = f'text=/{troop_type}/i'
                        troop_option = await page.wait_for_selector(troop_option_selector, timeout=2000)

                        if troop_option:
                            log.info(f"✅ Found custom dropdown option for '{troop_type}'.")
                            await troop_option.click()
                            await self._fill_and_submit_training(page, quantity)
                            return True
                    except Exception:
                        continue # This trigger didn't work, try the next
            except Exception as e:
                log.warning(f"⚠️ Strategy 2 failed: {e}")

            log.error(f"❌ All strategies to select troop '{troop_type}' failed.")
            await page.screenshot(path=f"debug_train_troops_failed_{troop_type}.png")
            return False

        except Exception as e:
            log.error(f"❌ Browser train failed: {e}")
            await page.screenshot(path="debug_train_troops_exception.png")
            return False

    async def _fill_and_submit_training(self, page, quantity: int):
        """Helper to fill quantity and submit the training form."""
        log.info(f"📝 Filling quantity '{quantity}' and submitting.")

        # Fill quantity
        quantity_input = await page.query_selector('input[name*="quantity"], input[type="number"]')
        if quantity_input:
            await quantity_input.fill(str(quantity))
            log.info("✅ Filled quantity.")
        else:
            log.warning("⚠️ Could not find quantity input field.")

        # Click submit
        submit_button = await page.query_selector('button[type="submit"], input[type="submit"], button:has-text("Train")')
        if submit_button:
            await submit_button.click()
            log.info("✅ Clicked train/submit button.")
            await page.wait_for_timeout(3000) # Wait for action to complete
        else:
            log.warning("⚠️ Could not find submit button.")
    
    
    async def _browser_login_process(self, page):
        """Handle login process in browser"""
        try:
            log.info("🔐 Starting browser login process...")
            
            # Look for email/username field
            email_selectors = [
                'input[name="email"]',
                'input[type="email"]',
                'input[id="email"]',
                'input[name="username"]',
                'input[placeholder*="email"]',
                'input[placeholder*="Email"]'
            ]
            
            email_field = None
            for selector in email_selectors:
                try:
                    email_field = await page.wait_for_selector(selector, timeout=2000)
                    if email_field:
                        log.info(f"✅ Found email field: {selector}")
                        break
                except:
                    continue
            
            if not email_field:
                log.error("❌ Could not find email field")
                return False
            
            # Look for password field
            password_selectors = [
                'input[name="password"]',
                'input[type="password"]',
                'input[id="password"]',
                'input[placeholder*="password"]',
                'input[placeholder*="Password"]'
            ]
            
            password_field = None
            for selector in password_selectors:
                try:
                    password_field = await page.wait_for_selector(selector, timeout=2000)
                    if password_field:
                        log.info(f"✅ Found password field: {selector}")
                        break
                except:
                    continue
            
            if not password_field:
                log.error("❌ Could not find password field")
                return False
            
            # Fill in credentials
            log.info("🔐 Filling login credentials...")
            await email_field.fill(USERNAME)
            await password_field.fill(PASSWORD)
            await page.wait_for_timeout(1000)
            
            # Look for login button
            login_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Login")',
                'button:has-text("Sign In")',
                '.login-button',
                '[class*="login"]'
            ]
            
            login_button = None
            for selector in login_selectors:
                try:
                    login_button = await page.wait_for_selector(selector, timeout=2000)
                    if login_button:
                        log.info(f"✅ Found login button: {selector}")
                        break
                except:
                    continue
            
            if not login_button:
                log.error("❌ Could not find login button")
                return False
            
            # Click login and wait for navigation
            log.info("🔐 Clicking login button...")
            await login_button.click()
            
            # Wait for navigation away from login page
            try:
                await page.wait_for_url(lambda url: "/login" not in url, timeout=10000)
                log.info("✅ Login successful - navigated away from login page")
                return True
            except:
                log.error("❌ Login failed - still on login page after 10 seconds")
                return False
                
        except Exception as e:
            log.error(f"❌ Browser login process failed: {e}")
            return False
    
    async def _browser_explore(self, page, params):
        """Explore using browser clicks on warroom?action=explore page"""
        try:
            log.info(f"🎮 Browser exploration started on explore page...")

            # Look for exploration elements
            explore_selectors = [
                'button:has-text("Explore")',
                'input[type="submit"][value*="Explore"]',
                'button:has-text("Search")',
                'button:has-text("Scout")',
                'button[onclick*="explore"]',
                '.explore-button',
                'form[action*="explore"] button',
                'form[action*="explore"] input[type="submit"]'
            ]

            for selector in explore_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=3000)
                    if element:
                        await element.click()
                        log.info(f"✅ Clicked explore element: {selector}")
                        await page.wait_for_timeout(2000)
                        return True
                except:
                    continue

            log.error("❌ Could not find exploration elements")
            return False

        except Exception as e:
            log.error(f"❌ Browser exploration failed: {e}")
            return False

    async def _browser_build(self, page, params):
        """Build using browser clicks on buildings page"""
        try:
            log.info(f"🎮 Browser building started on buildings page...")

            # Look for building elements
            build_selectors = [
                'button:has-text("Build")',
                'input[type="submit"][value*="Build"]',
                'button:has-text("Houses")',
                'button:has-text("Farms")',
                'button:has-text("Markets")',
                'button[onclick*="build"]',
                '.build-button',
                'form[action*="build"] button',
                'form[action*="build"] input[type="submit"]'
            ]

            for selector in build_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=3000)
                    if element:
                        await element.click()
                        log.info(f"✅ Clicked build element: {selector}")
                        await page.wait_for_timeout(2000)
                        return True
                except:
                    continue

            log.error("❌ Could not find building elements")
            return False

        except Exception as e:
            log.error(f"❌ Browser building failed: {e}")
            return False

    async def _browser_spy(self, page, params):
        """Spy using browser clicks on embassy page"""
        try:
            log.info(f"🎮 Browser spying started on embassy page...")
            
            # Look for spy elements
            spy_selectors = [
                'button:has-text("Spy")',
                'input[type="submit"][value*="Spy"]',
                'button:has-text("Infiltrate")',
                'button:has-text("Scout")',
                'button[onclick*="spy"]',
                '.spy-button',
                'form[action*="spy"] button',
                'form[action*="spy"] input[type="submit"]'
            ]
            
            for selector in spy_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=3000)
                    if element:
                        await element.click()
                        log.info(f"✅ Clicked spy element: {selector}")
                        await page.wait_for_timeout(2000)
                        return True
                except:
                    continue
                    
            log.error("❌ Could not find spy elements")
            return False
            
        except Exception as e:
            log.error(f"❌ Browser spying failed: {e}")
            return False
    
    async def _autonomous_game_ai(self, page):
        """Fully autonomous AI that explores all game pages and makes strategic decisions"""
        try:
            log.info("🧠 AUTONOMOUS AI MODE ACTIVATED - Bot will explore and make its own decisions")

            # Define all available game pages from user's map
            game_pages = {
                "overview": f"{BASE_URL}/overview",
                "warroom": f"{BASE_URL}/warroom",
                "warroom_explore": f"{BASE_URL}/warroom?action=explore",
                "buildings": f"{BASE_URL}/buildings",
                "guildhall": f"{BASE_URL}/guildhall",
                "holycircle": f"{BASE_URL}/holycircle",
                "embassy": f"{BASE_URL}/embassy",
                "settlements": f"{BASE_URL}/settlements",
                "rankings": f"{BASE_URL}/rankings",
                "research": f"{BASE_URL}/research",
                "messages": f"{BASE_URL}/messages"
            }

            # Strategic priorities for the AI
            ai_priorities = [
                "build_economy",      # Focus on resource generation
                "train_military",     # Build army strength
                "expand_territory",   # Gain more land
                "research_tech",      # Unlock new capabilities
                "diplomacy",          # Manage relationships
                "intelligence"        # Gather information
            ]

            # Main autonomous loop
            for cycle in range(10):  # Run for 10 decision cycles
                log.info(f"🧠 === AI DECISION CYCLE {cycle + 1} ===")

                # Phase 1: Gather intelligence from all pages
                game_state = await self._analyze_all_pages(page, game_pages)

                # CRITICAL: Store game state for smart calculations during form filling
                self.latest_game_state = game_state
                log.info(f"💾 Stored game state with {len(game_state.get('resources', {}))} resources, {len(game_state.get('military', {}).get('troops', {}))} troop types")

                # Phase 2: Make strategic decision based on game state
                best_action = await self._make_strategic_decision(game_state, ai_priorities)

                # Phase 3: Execute the chosen action
                success = await self._execute_autonomous_action(page, best_action, game_pages)

                if success:
                    log.info(f"✅ Autonomous action completed: {best_action['action']}")
                else:
                    log.warning(f"⚠️ Autonomous action failed: {best_action['action']}")

                # Wait before next decision cycle
                wait_time = random.randint(30, 120)  # 30-120 seconds between decisions
                log.info(f"🧠 AI thinking... waiting {wait_time} seconds before next decision")
                await page.wait_for_timeout(wait_time * 1000)

            log.info("🧠 Autonomous AI session completed!")
            return True

        except Exception as e:
            log.error(f"❌ Autonomous AI failed: {e}")
            return False

    async def _analyze_all_pages(self, page, game_pages):
        """Visit all game pages and analyze what's available"""
        game_state = {
            "resources": {},
            "military": {},
            "buildings": {},
            "opportunities": [],
            "current_status": {},
            "threats": []
        }
        
        log.info("🔍 AI analyzing all game pages for opportunities...")
        
        for page_name, page_url in game_pages.items():
            try:
                log.info(f"🔍 AI visiting {page_name}: {page_url}")
                await page.goto(page_url)
                await page.wait_for_timeout(3000)

                # Take screenshot for analysis
                screenshot_path = f"ai_analysis_{page_name}.png"
                await page.screenshot(path=screenshot_path)

                # Get page content
                page_text = await page.content()
                page_title = await page.title()

                # Analyze page for opportunities
                opportunities = await self._find_page_opportunities(page, page_name, page_text)
                game_state["opportunities"].extend(opportunities)

                # Extract key information based on page type
                if page_name == "overview":
                    await self._extract_overview_data(page_text, game_state)
                elif "warroom" in page_name:
                    await self._extract_military_data(page_text, game_state)
                elif page_name == "buildings":
                    await self._extract_building_data(page_text, game_state)
                elif page_name == "research":
                    await self._extract_research_data(page_text, game_state)

                log.info(f"✅ AI analyzed {page_name} - found {len(opportunities)} opportunities")

            except Exception as e:
                log.warning(f"⚠️ AI failed to analyze {page_name}: {e}")
                continue
        
        log.info(f"🧠 AI analysis complete - found {len(game_state['opportunities'])} total opportunities")
        return game_state

    async def _find_page_opportunities(self, page, page_name, page_text):
        """Find actionable opportunities on a page"""
        opportunities = []
        
        # Get all visible, clickable elements only
        clickable_elements = await page.query_selector_all('button:visible, a[href]:visible, input[type="submit"]:visible, [onclick]:visible')

        for element in clickable_elements:
            try:
                # Double-check visibility
                is_visible = await element.is_visible()
                if not is_visible:
                    continue

                text = await element.inner_text()
                if text and len(text.strip()) > 0:
                    # Classify opportunity type
                    opportunity_type = self._classify_opportunity(text.strip(), page_name)
                    if opportunity_type:
                        opportunities.append({
                            "page": page_name,
                            "type": opportunity_type,
                            "text": text.strip(),
                            # Remove element reference - we'll find it fresh when executing
                            "priority": self._calculate_opportunity_priority(opportunity_type, text)
                        })
            except:
                continue

        return opportunities

    def _classify_opportunity(self, text, page_name):
        """Classify what type of opportunity an action represents"""
        text_lower = text.lower()

        # Skip navigation links - focus on actual game actions
        navigation_words = ['buildings', 'warroom', 'guildhall', 'embassy', 'overview', 'rankings', 'messages', 'settings', 'logout', 'research', 'settlements', 'holycircle']
        if any(nav_word in text_lower for nav_word in navigation_words):
            return None  # Ignore navigation links

        # Skip common UI elements that aren't game actions
        ui_elements = ['home', 'menu', 'back', 'cancel', 'close', 'help', 'info', 'about']
        if any(ui_word in text_lower for ui_word in ui_elements):
            return None

        # Economic opportunities - look for specific action words
        if any(word in text_lower for word in ['build', 'construct', 'upgrade', 'buy', 'purchase', 'spend', 'invest']):
            return "economic"

        # Military opportunities - look for training/military actions
        if any(word in text_lower for word in ['train', 'recruit', 'attack', 'defend', 'hire', 'enlist']):
            return "military"

        # Expansion opportunities - look for exploration actions
        if any(word in text_lower for word in ['explore', 'expand', 'patrol', 'venture', 'search']):
            return "expansion"

        # Research opportunities - look for research actions
        if any(word in text_lower for word in ['study', 'learn', 'advance', 'develop', 'discover']):
            return "research"

        # Diplomatic opportunities - look for diplomatic actions
        if any(word in text_lower for word in ['alliance', 'treaty', 'negotiate', 'diplomacy', 'send message', 'offer']):
            return "diplomatic"

        # Intelligence opportunities - look for spy actions
        if any(word in text_lower for word in ['spy', 'infiltrate', 'recon', 'intelligence', 'gather info']):
            return "intelligence"

        return None

    def _calculate_opportunity_priority(self, opportunity_type, text):
        """Calculate priority score for an opportunity"""
        base_priorities = {
            "economic": 8,      # High priority - need resources
            "military": 7,      # High priority - need defense
            "expansion": 6,     # Medium-high - need territory
            "research": 5,      # Medium - long term benefits
            "intelligence": 4,  # Medium-low - information gathering
            "diplomatic": 3     # Low - nice to have
        }

        priority = base_priorities.get(opportunity_type, 1)

        # Boost priority for certain keywords
        text_lower = text.lower()
        if any(word in text_lower for word in ['free', 'bonus', 'reward', 'special']):
            priority += 3
        if any(word in text_lower for word in ['urgent', 'limited', 'expire']):
            priority += 2

        return priority

    async def _make_strategic_decision(self, game_state, ai_priorities):
        """AI makes strategic decision about what to do next"""
        global completed_actions, action_history

        log.info("🧠 AI making strategic decision...")

        # STRATEGIC PRIORITIZATION - Early game focus on exploration for fast land gain
        resources = game_state.get("resources", {})
        military = game_state.get("military", {})
        current_land = resources.get("land", 50)
        total_troops = sum(military.get("troops", {}).values()) if military.get("troops") else 0

        log.info(f"🎯 Strategic analysis: {current_land} land, {total_troops} troops")

        # Boost exploration priority in early game
        for opp in game_state["opportunities"]:
            if opp["type"] == "expansion" and current_land < 300:
                opp["priority"] += 50  # Major boost for early exploration
                log.info(f"🚀 BOOSTED exploration priority for early game (land: {current_land})")
            elif opp["type"] == "economic" and 300 <= current_land < 800:
                opp["priority"] += 30  # Boost building in mid game
                log.info(f"🏗️ BOOSTED building priority for mid game (land: {current_land})")
            elif opp["type"] == "military" and total_troops < 100:
                opp["priority"] += 25  # Need troops for exploration
                log.info(f"🗡️ BOOSTED training priority (low troops: {total_troops})")

        # Sort opportunities by priority (including strategic boosts)
        opportunities = sorted(game_state["opportunities"], key=lambda x: x["priority"], reverse=True)

        if not opportunities:
            log.warning("⚠️ AI found no opportunities - will explore randomly")
            return {"action": "random_exploration", "page": "overview"}

        # Filter out recently completed actions for variety
        available_opportunities = []
        for opp in opportunities:
            action_key = f"{opp['page']}_{opp['text']}"
            if action_key not in completed_actions:
                available_opportunities.append(opp)

        if not available_opportunities:
            log.info("🔄 All opportunities completed, clearing history for new cycle")
            completed_actions.clear()  # Reset for new cycle
            action_history.clear()
            available_opportunities = opportunities

        # Add some variety - don't always pick the highest priority
        if len(available_opportunities) > 1 and random.random() < 0.3:  # 30% chance
            # Pick from top 3 options for variety
            top_options = available_opportunities[:min(3, len(available_opportunities))]
            best_opportunity = random.choice(top_options)
            log.info(f"🎲 AI chose variety option instead of highest priority")
        else:
            best_opportunity = available_opportunities[0]

        log.info(f"🎯 AI decided on: {best_opportunity['type']} action - '{best_opportunity['text']}' on {best_opportunity['page']} page (priority: {best_opportunity['priority']})")

        # Track this action
        action_key = f"{best_opportunity['page']}_{best_opportunity['text']}"
        completed_actions.add(action_key)
        action_history.append(action_key)

        # Keep only recent 20 actions in history
        if len(action_history) > 20:
            action_history.pop(0)

        return {
            "action": best_opportunity["type"],
            "page": best_opportunity["page"],
            "text": best_opportunity["text"],
            # Remove stale element reference
            "opportunity": best_opportunity
        }

    async def _execute_autonomous_action(self, page, action, game_pages):
        """Execute the AI's chosen action"""
        try:
            log.info(f"🎯 AI executing {action['action']} on {action['page']} page")

            # Navigate to the correct page
            page_url = game_pages.get(action['page'])
            if page_url:
                await page.goto(page_url)
                await page.wait_for_timeout(3000)

            # Always find element fresh by its text (no stale elements!)
            action_text = action.get('text', '')
            if action_text:
                log.info(f"🔍 AI re-finding element with text: '{action_text}'")

                # Try multiple strategies to find the element
                found_element = None

                # Strategy 1: Find by exact text match (only visible elements)
                try:
                    elements = await page.query_selector_all('button:visible, a[href]:visible, input[type="submit"]:visible, [onclick]:visible')
                    for element in elements:
                        try:
                            # Double check visibility
                            is_visible = await element.is_visible()
                            if not is_visible:
                                continue

                            text = await element.inner_text()
                            if text and text.strip() == action_text:
                                found_element = element
                                log.info(f"✅ Found visible element by exact text match")
                                break
                        except:
                            continue
                except:
                    pass

                # Strategy 2: Find by partial text match (only visible elements)
                if not found_element:
                    try:
                        elements = await page.query_selector_all('button:visible, a[href]:visible, input[type="submit"]:visible, [onclick]:visible')
                        for element in elements:
                            try:
                                # Double check visibility
                                is_visible = await element.is_visible()
                                if not is_visible:
                                    continue

                                text = await element.inner_text()
                                if text and action_text.lower() in text.lower():
                                    found_element = element
                                    log.info(f"✅ Found visible element by partial text match")
                                    break
                            except:
                                continue
                    except:
                        pass

                # Strategy 3: Use Playwright's text selector
                if not found_element:
                    try:
                        # Try various text-based selectors
                        text_selectors = [
                            f'text="{action_text}"',
                            f'text={action_text}',
                            f':has-text("{action_text}")',
                            f'button:has-text("{action_text}")',
                            f'a:has-text("{action_text}")'
                        ]

                        for selector in text_selectors:
                            try:
                                found_element = await page.wait_for_selector(selector, timeout=2000)
                                if found_element:
                                    log.info(f"✅ Found element using selector: {selector}")
                                    break
                            except:
                                continue
                    except:
                        pass

                if found_element:
                    # Comprehensive element interaction with multiple strategies
                    try:
                        # Strategy 1: Scroll element into view
                        log.info(f"🔄 Scrolling element into view: '{action_text}'")
                        await found_element.scroll_into_view_if_needed()
                        await page.wait_for_timeout(1000)  # Wait for scroll animation

                        # Strategy 2: Check if element is really visible and enabled
                        is_visible = await found_element.is_visible()
                        is_enabled = await found_element.is_enabled()

                        if not is_visible:
                            log.warning(f"⚠️ Element not visible after scroll: '{action_text}'")
                            # Try to make it visible by clicking parent or removing overlays
                            try:
                                # Remove common overlay elements
                                await page.evaluate("""
                                    // Remove common modal/overlay elements
                                    document.querySelectorAll('.modal, .overlay, .popup, [style*="z-index"]').forEach(el => {
                                        if (el.style.zIndex > 100) el.remove();
                                    });
                                """)
                                await page.wait_for_timeout(500)
                                is_visible = await found_element.is_visible()
                            except:
                                pass

                        if not is_enabled:
                            log.warning(f"⚠️ Element not enabled: '{action_text}'")

                        # Strategy 3: Multiple click attempts
                        click_success = False

                        if is_visible and is_enabled:
                            # Attempt 1: Normal click
                            try:
                                log.info(f"🖱️ Attempting normal click on: '{action_text}'")
                                await found_element.click(timeout=5000)
                                click_success = True
                                log.info(f"✅ Normal click successful: '{action_text}'")
                            except Exception as e:
                                log.warning(f"⚠️ Normal click failed: {e}")

                        if not click_success:
                            # Attempt 2: Force click with coordinates
                            try:
                                log.info(f"🎯 Attempting coordinate click on: '{action_text}'")
                                box = await found_element.bounding_box()
                                if box:
                                    x = box['x'] + box['width'] / 2
                                    y = box['y'] + box['height'] / 2
                                    await page.mouse.click(x, y)
                                    click_success = True
                                    log.info(f"✅ Coordinate click successful: '{action_text}'")
                            except Exception as e:
                                log.warning(f"⚠️ Coordinate click failed: {e}")

                        if not click_success:
                            # Attempt 3: JavaScript click
                            try:
                                log.info(f"⚡ Attempting JavaScript click on: '{action_text}'")
                                await page.evaluate("(element) => element.click()", found_element)
                                click_success = True
                                log.info(f"✅ JavaScript click successful: '{action_text}'")
                            except Exception as e:
                                log.warning(f"⚠️ JavaScript click failed: {e}")

                        if not click_success:
                            # Attempt 4: Dispatch click event
                            try:
                                log.info(f"📡 Attempting event dispatch on: '{action_text}'")
                                await found_element.dispatch_event('click')
                                click_success = True
                                log.info(f"✅ Event dispatch successful: '{action_text}'")
                            except Exception as e:
                                log.warning(f"⚠️ Event dispatch failed: {e}")

                        if click_success:
                            # Wait and take screenshot of result
                            await page.wait_for_timeout(3000)
                            await page.screenshot(path=f"ai_action_result_{action['action']}.png")

                            # Look for confirmation or next steps
                            await self._handle_action_followup(page, action)

                            return True
                        else:
                            log.error(f"❌ All click strategies failed for: '{action_text}'")
                            await page.screenshot(path=f"ai_all_click_strategies_failed_{action['action']}.png")
                            return False

                    except Exception as interaction_error:
                        log.error(f"❌ Element interaction failed for '{action_text}': {interaction_error}")
                        await page.screenshot(path=f"ai_interaction_error_{action['action']}.png")
                        return False
                else:
                    log.error(f"❌ Could not re-find element with text: '{action_text}'")
                    await page.screenshot(path=f"ai_failed_to_find_{action['action']}.png")
                    return False
            else:
                log.warning(f"⚠️ No text available to find element for action: {action['action']}")
                return False

        except Exception as e:
            log.error(f"❌ AI failed to execute action {action['action']}: {e}")
            return False

    async def _handle_action_followup(self, page, action):
        """Handle any follow-up actions needed after clicking - HANDLES MISSING QUANTITY FIELDS"""
        try:
            await page.wait_for_timeout(2000)  # Wait for page changes
            await page.screenshot(path=f"ai_followup_start_{action['action']}.png")

            # Find ALL input types - text, number, select, hidden, etc.
            all_inputs = await page.query_selector_all('input, select, textarea')

            if not all_inputs:
                log.info("🔍 No input fields found - action may be complete")
                return

            log.info(f"🎯 AI found {len(all_inputs)} input fields, filling intelligently...")

            # DETAILED FIELD ANALYSIS
            log.info(f"🎯 DETAILED FIELD ANALYSIS:")
            for i, input_field in enumerate(all_inputs):
                try:
                    name = await input_field.get_attribute('name') or f'field_{i}'
                    field_type = await input_field.get_attribute('type') or 'unknown'
                    tag = await input_field.evaluate('el => el.tagName.toLowerCase()')
                    placeholder = await input_field.get_attribute('placeholder') or ''
                    value = await input_field.get_attribute('value') or ''
                    is_visible = await input_field.is_visible()

                    log.info(f"  {i+1}. {tag}[{field_type}] name='{name}' placeholder='{placeholder}' value='{value}' visible={is_visible}")
                except Exception as e:
                    log.info(f"  {i+1}. [error reading field: {e}]")

            filled_count = 0
            for i, input_field in enumerate(all_inputs):
                try:
                    input_name = await input_field.get_attribute('name') or ""
                    input_type = await input_field.get_attribute('type') or ""
                    input_id = await input_field.get_attribute('id') or ""
                    tag_name = await input_field.evaluate('el => el.tagName.toLowerCase()')

                    # Skip disabled or readonly fields
                    is_disabled = await input_field.evaluate('el => el.disabled')
                    is_readonly = await input_field.evaluate('el => el.readOnly')
                    if is_disabled or is_readonly:
                        continue

                    field_identifier = f"{input_name or input_id or f'field_{i}'}"
                    log.info(f"🔍 Processing {tag_name} field: {field_identifier} (type: {input_type})")

                    # Handle different field types
                    if tag_name == 'select':
                        filled = await self._handle_select_field(input_field, field_identifier, action)
                    elif input_type in ['number', 'text', '']:
                        filled = await self._handle_number_field(input_field, field_identifier, action, input_name, input_id)
                    elif input_type == 'hidden':
                        filled = await self._handle_hidden_field(input_field, field_identifier, action)
                    elif tag_name == 'textarea':
                        filled = await self._handle_textarea_field(input_field, field_identifier, action)
                    else:
                        log.info(f"⚠️ Skipping unsupported field type: {input_type}")
                        continue

                    if filled:
                        filled_count += 1
                        await page.wait_for_timeout(500)  # Small delay between fills

                except Exception as field_error:
                    log.warning(f"⚠️ Failed to fill field {i}: {field_error}")
                    continue

            log.info(f"✅ AI filled {filled_count} fields successfully")
            await page.screenshot(path=f"ai_followup_filled_{action['action']}.png")

            # CRITICAL: If no quantity field was found, try to find it after form changes
            if filled_count == 0 or (filled_count == 1 and any('select' in str(field) for field in all_inputs)):
                log.info("🎯 No quantity field found - checking for dynamic fields after selection...")
                await page.wait_for_timeout(2000)  # Wait for any dynamic content

                # Look for new fields that might have appeared
                new_inputs = await page.query_selector_all('input, select, textarea')
                if len(new_inputs) > len(all_inputs):
                    log.info(f"🎯 Found {len(new_inputs) - len(all_inputs)} new fields after selection!")

                    # Process the new fields
                    for i, input_field in enumerate(new_inputs[len(all_inputs):]):
                        try:
                            input_type = await input_field.get_attribute('type')
                            if input_type == 'number':
                                # This is likely the quantity field!
                                if 'military' in action['action']:
                                    value = random.randint(200, 1000)
                                elif 'economic' in action['action']:
                                    value = random.randint(50, 500)
                                else:
                                    value = random.randint(100, 300)

                                await input_field.fill(str(value))
                                log.info(f"🎯 FOUND DYNAMIC QUANTITY FIELD: {value}")
                                filled_count += 1
                                break
                        except:
                            continue

            # Force fill ANY number field as quantity if we still haven't filled any
            if filled_count == 0:
                log.info("🎯 No fields filled yet - looking for ANY number field to use as quantity")
                for input_field in all_inputs:
                    try:
                        field_type = await input_field.get_attribute('type')
                        if field_type == 'number':
                            # Force fill this as quantity
                            if 'military' in action['action']:
                                value = random.randint(200, 1000)
                            elif 'economic' in action['action']:
                                value = random.randint(50, 500)
                            else:
                                value = random.randint(100, 300)

                            await input_field.fill(str(value))
                            log.info(f"🎯 FORCE FILLED number field as quantity: {value}")
                            filled_count += 1
                            break
                    except:
                        continue

            # If STILL no quantity field, try clicking multiple times to trigger quantity field
            if filled_count == 0:
                log.info("🎯 Still no quantity field - trying to trigger it with multiple clicks...")
                try:
                    # Look for any buttons that might trigger quantity fields
                    trigger_buttons = await page.query_selector_all('button, a, [onclick]')
                    for button in trigger_buttons:
                        try:
                            text = await button.inner_text()
                            if any(word in text.lower() for word in ['quantity', 'amount', 'how many', 'number']):
                                await button.click()
                                await page.wait_for_timeout(1000)
                                log.info(f"🎯 Clicked potential quantity trigger: {text}")
                                break
                        except:
                            continue
                except:
                    pass

            # Find and click submit buttons with comprehensive selectors
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Confirm")',
                'button:has-text("Build")',
                'button:has-text("Train")',
                'button:has-text("Attack")',
                'button:has-text("Explore")',
                'button:has-text("Research")',
                'button:has-text("Send")',
                'button:has-text("Buy")',
                'button:has-text("Purchase")',
                'button:has-text("Execute")',
                'button:has-text("Go")',
                'button:has-text("Start")',
                'button:has-text("Create")',
                'button[onclick*="submit"]',
                'input[onclick*="submit"]',
                '.submit-btn',
                '.btn-primary',
                '.btn-submit'
            ]

            submit_button = None
            for selector in submit_selectors:
                try:
                    buttons = await page.query_selector_all(selector)
                    for button in buttons:
                        is_visible = await button.is_visible()
                        if is_visible:
                            submit_button = button
                            log.info(f"✅ Found submit button with selector: {selector}")
                            break
                    if submit_button:
                        break
                except:
                    continue

            if submit_button:
                try:
                    # Multiple click strategies for submit button
                    await submit_button.scroll_into_view_if_needed()
                    await page.wait_for_timeout(1000)

                    # Try normal click first
                    try:
                        await submit_button.click(timeout=5000)
                        log.info("✅ AI clicked submit button (normal click)")
                    except:
                        # Try coordinate click
                        try:
                            box = await submit_button.bounding_box()
                            if box:
                                await page.mouse.click(box['x'] + box['width']/2, box['y'] + box['height']/2)
                                log.info("✅ AI clicked submit button (coordinate click)")
                        except:
                            # Try JavaScript click
                            await page.evaluate('(button) => button.click()', submit_button)
                            log.info("✅ AI clicked submit button (JavaScript click)")

                    await page.wait_for_timeout(3000)  # Wait for submission
                    await page.screenshot(path=f"ai_followup_submitted_{action['action']}.png")

                except Exception as click_error:
                    log.error(f"❌ Failed to click submit button: {click_error}")
            else:
                log.warning("⚠️ No submit button found - form may auto-submit or action may be complete")

        except Exception as e:
            log.error(f"❌ Follow-up handling failed: {e}")
            await page.screenshot(path=f"ai_followup_error_{action['action']}.png")

    async def _handle_select_field(self, select_field, field_name, action):
        """Handle dropdown/select fields"""
        try:
            options = await select_field.query_selector_all('option')
            if len(options) <= 1:
                return False

            # Skip first option (usually empty/placeholder)
            selected_option = None

            if 'military' in action['action'] or 'train' in field_name.lower():
                # For military actions, prefer troops
                for option in options[1:]:
                    text = await option.inner_text()
                    if any(word in text.lower() for word in ['foot', 'archer', 'cavalry', 'soldier', 'troop']):
                        selected_option = option
                        break
            elif 'economic' in action['action'] or 'build' in field_name.lower():
                # For economic actions, prefer buildings
                for option in options[1:]:
                    text = await option.inner_text()
                    if any(word in text.lower() for word in ['house', 'farm', 'market', 'mine', 'building']):
                        selected_option = option
                        break

            # Default to random option if no specific match
            if not selected_option and len(options) > 1:
                selected_option = random.choice(options[1:])

            if selected_option:
                option_value = await selected_option.get_attribute('value')
                option_text = await selected_option.inner_text()
                await select_field.select_option(value=option_value)
                log.info(f"🎯 AI selected '{option_text}' in {field_name}")
                return True

        except Exception as e:
            log.warning(f"⚠️ Failed to handle select field {field_name}: {e}")
        return False

    async def _handle_number_field(self, input_field, field_name, action, input_name, input_id):
        """Handle number and text input fields - ULTRA-CONSERVATIVE RESOURCE AWARE"""
        try:
            field_lower = f"{field_name} {input_name} {input_id}".lower()
            # ...




            # Check if this is ANY kind of quantity field
            is_quantity_field = any(word in field_lower for word in [
                'quantity', 'amount', 'count', 'num', 'qty', 'number', 'how', 'many'
            ])

            # Also check if it's a number input with no specific name (often quantity fields)
            is_number_input = await input_field.get_attribute('type') == 'number'
            is_text_input = await input_field.get_attribute('type') in ['text', '']

            # If it's a quantity field, get CURRENT resources from the page RIGHT NOW
            if is_quantity_field or (is_number_input and not input_name) or (is_text_input and not input_name):
                # CRITICAL: Get current page to extract available resources
                page = input_field.page if hasattr(input_field, 'page') else None
                available_resources = {}

                if page:
                    available_resources = await self._extract_available_resources(page)
                    log.info(f"🎯 EXTRACTED REAL-TIME RESOURCES: {available_resources}")
                else:
                    log.warning("⚠️ No page available, using ULTRA-CONSERVATIVE fallbacks")
                    available_resources = {'gold': 100, 'wood': 50, 'stone': 50, 'food': 50, 'land': 1}

                if 'military' in action['action'] or 'train' in field_lower:
                    # Calculate ULTRA-CONSERVATIVE troop training based on CURRENT available gold
                    available_gold = available_resources.get('gold', 100)
                    troop_cost = 75  # Average cost per troop
                    max_affordable = max(1, available_gold // troop_cost)

                    # Train 1-3 troops maximum, never more
                    value = max(1, min(3, max_affordable // 4))
                    log.info(f"🗡️ ULTRA-CONSERVATIVE TRAINING: {value} troops (can afford {max_affordable}, have {available_gold} gold)")

                elif 'economic' in action['action'] or 'build' in field_lower:
                    # Calculate ULTRA-CONSERVATIVE building based on CURRENT available resources
                    available_gold = available_resources.get('gold', 100)
                    available_land = available_resources.get('land', 1)
                    available_wood = available_resources.get('wood', 50)
                    available_stone = available_resources.get('stone', 50)

                    building_cost = 150  # Average cost per building

                    # CRITICAL: Check all resource constraints
                    max_by_land = available_land
                    max_by_gold = max(1, available_gold // building_cost)
                    max_by_wood = max(1, available_wood // 10)  # Assume 10 wood per building
                    max_by_stone = max(1, available_stone // 10)  # Assume 10 stone per building

                    # Take the SMALLEST constraint (most limiting resource)
                    value = min(max_by_land, max_by_gold, max_by_wood, max_by_stone)
                    value = max(1, min(1, value))  # Build 1 building maximum

                    log.info(f"🏗️ ULTRA-CONSERVATIVE BUILDING: {value} buildings (limited by land:{available_land}, gold:{available_gold}, wood:{available_wood}, stone:{available_stone})")

                elif 'expansion' in action['action']:
                    # Conservative exploration - we don't have troop count from this page
                    value = 1  # Send only 1 troop for exploration
                    log.info(f"🗺️ ULTRA-CONSERVATIVE EXPLORATION: {value} troop")

                else:
                    # Default very conservative amount for unknown actions
                    value = 1
                    log.info(f"�� ULTRA-CONSERVATIVE: {value} (unknown action type)")

                await input_field.fill(str(value))
                log.info(f"✅ ULTRA-CONSERVATIVE FILLED: {field_name} = {value}")
                return True



            # Price/Cost fields
            elif any(word in field_lower for word in ['price', 'cost', 'gold', 'money']):
                price = 50  # Conservative prices
                await input_field.fill(str(price))
                log.info(f"�� Filled {field_name} price: {price}")
                return True

            # Generic text fields
            else:
                placeholder = await input_field.get_attribute('placeholder') or ""
                if placeholder:
                    await input_field.fill(placeholder[:10])
                    log.info(f"�� Filled {field_name} with placeholder text")
                    return True
                elif is_number_input or is_text_input:
                    # Ultra-conservative default for unknown fields
                    value = 1
                    await input_field.fill(str(value))
                    log.info(f"�� ULTRA-CONSERVATIVE FILL: {field_name} = {value}")
                    return True

        except Exception as e:
            log.warning(f"⚠️ Failed to handle number field {field_name}: {e}")
        return False

            # Kingdom ID fields for attacking - SAFE DEFAULTS
            elif any(word in field_lower for word in ['kingdom', 'target', 'enemy', 'player', 'id']):
                # Use nearby kingdoms but be conservative
                nearby_kingdoms = [6046, 6047, 6048, 6049]  # Close to our kingdom 6045
                kingdom_id = random.choice(nearby_kingdoms)
                await input_field.fill(str(kingdom_id))
                log.info(f"🎯 Filled {field_name} kingdom ID: {kingdom_id}")
                return True

            # Distance/Direction fields for exploration
            elif any(word in field_lower for word in ['distance', 'range', 'miles', 'steps']):
                distance = random.randint(5, 15)  # Conservative exploration distance
                await input_field.fill(str(distance))
                log.info(f"🎯 Filled {field_name} distance: {distance}")
                return True

            # Price/Cost fields
            elif any(word in field_lower for word in ['price', 'cost', 'gold', 'money']):
                price = random.randint(50, 500)  # Conservative prices
                await input_field.fill(str(price))
                log.info(f"🎯 Filled {field_name} price: {price}")
                return True

            # Generic text fields
            else:
                placeholder = await input_field.get_attribute('placeholder') or ""
                if placeholder:
                    await input_field.fill(placeholder[:10])
                    log.info(f"🎯 Filled {field_name} with placeholder text")
                    return True
                elif is_number_input or is_text_input:
                    # Ultra-conservative default for unknown fields
                    value = random.randint(1, 2)
                    await input_field.fill(str(value))
                    log.info(f"🎯 ULTRA-CONSERVATIVE FILL: {field_name} = {value}")
                    return True

        except Exception as e:
            log.warning(f"⚠️ Failed to handle number field {field_name}: {e}")
        return False

    async def _handle_hidden_field(self, input_field, field_name, action):
        """Handle hidden fields - usually contain required IDs/tokens"""
        try:
            current_value = await input_field.get_attribute('value') or ""
            if not current_value:
                # Fill with our known values if field names match
                field_lower = field_name.lower()
                if 'kingdom' in field_lower and 'id' in field_lower:
                    await input_field.fill(str(KINGDOM_ID))
                    log.info(f"🎯 AI filled hidden {field_name} with kingdom ID")
                    return True
                elif 'account' in field_lower and 'id' in field_lower:
                    await input_field.fill(str(ACCOUNT_ID))
                    log.info(f"🎯 AI filled hidden {field_name} with account ID")
                    return True
                elif 'token' in field_lower:
                    await input_field.fill(TOKEN)
                    log.info(f"🎯 AI filled hidden {field_name} with token")
                    return True
        except Exception as e:
            log.warning(f"⚠️ Failed to handle hidden field {field_name}: {e}")
        return False

    async def _handle_textarea_field(self, textarea_field, field_name, action):
        """Handle textarea fields - usually for messages"""
        try:
            if 'diplomatic' in action['action'] or 'message' in field_name.lower():
                messages = [
                    "Greetings from our kingdom!",
                    "Let us discuss trade opportunities.",
                    "We propose a mutually beneficial alliance.",
                    "Your kingdom seems strong - we respect that.",
                    "Perhaps we can avoid conflict through diplomacy."
                ]
                message = random.choice(messages)
                await textarea_field.fill(message)
                log.info(f"🎯 AI filled {field_name} with diplomatic message")
                return True
        except Exception as e:
            log.warning(f"⚠️ Failed to handle textarea field {field_name}: {e}")
        return False

        async def _extract_available_resources(self, page):
        """Extract available resources and land from the current page"""
        try:
            # Get page content to parse resources
            page_content = await page.content()

            # Look for resource indicators in the HTML - CRITICAL PATTERNS FROM GAME UI
            resource_patterns = {
                'gold': [
                    r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?gold',
                    r'gold[^\d]*(\d+(?:,\d+)*)',
                    r'money[^\d]*(\d+(?:,\d+)*)',
                    r'(\d+)\s*gold',  # Simple pattern
                    r'gold:\s*(\d+)',  # Colon pattern
                    r'gold\s*=\s*(\d+)'  # Equals pattern
                ],
                'food': [
                    r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?food',
                    r'food[^\d]*(\d+(?:,\d+)*)',
                    r'grain[^\d]*(\d+(?:,\d+)*)',
                    r'(\d+)\s*food',
                    r'food:\s*(\d+)',
                    r'food\s*=\s*(\d+)'
                ],
                'wood': [
                    r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?wood',
                    r'wood[^\d]*(\d+(?:,\d+)*)',
                    r'lumber[^\d]*(\d+(?:,\d+)*)',
                    r'(\d+)\s*wood',
                    r'wood:\s*(\d+)',
                    r'wood\s*=\s*(\d+)'
                ],
                'stone': [
                    r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?stone',
                    r'stone[^\d]*(\d+(?:,\d+)*)',
                    r'rock[^\d]*(\d+(?:,\d+)*)',
                    r'(\d+)\s*stone',
                    r'stone:\s*(\d+)',
                    r'stone\s*=\s*(\d+)'
                ],
                'land': [
                    r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?land',
                    r'land[^\d]*(\d+(?:,\d+)*)',
                    r'acres[^\d]*(\d+(?:,\d+)*)',
                    r'territory[^\d]*(\d+(?:,\d+)*)',
                    r'(\d+)\s*acres',
                    r'(\d+)\s*land',
                    r'land:\s*(\d+)',
                    r'territory:\s*(\d+)'
                ]
            }

            available_resources = {}

            for resource_type, patterns in resource_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, page_content, re.IGNORECASE)
                    if matches:
                        # If it's a cost/available pattern (2 numbers), take the second (available)
                        for match in matches:
                            try:
                                if isinstance(match, tuple) and len(match) == 2:
                                    # Pattern like "2,800 / 600" - take available amount (second number)
                                    value = int(match[1].replace(',', ''))
                                else:
                                    # Single number pattern
                                    value = int(str(match).replace(',', ''))

                                if 0 <= value <= 1000000:  # Reasonable range
                                    available_resources[resource_type] = value
                                    log.info(f"💰 EXTRACTED {resource_type}: {value}")
                                    break
                            except:
                                continue
                        if resource_type in available_resources:
                            break

            # FALLBACK: Set minimum safe defaults if no resources found
            fallback_resources = {
                'gold': 100,    # Very conservative
                'wood': 50,     # Very conservative
                'stone': 50,    # Very conservative
                'food': 50,     # Very conservative
                'land': 1       # Very conservative land estimate
            }

            for resource, fallback_value in fallback_resources.items():
                if resource not in available_resources:
                    available_resources[resource] = fallback_value
                    log.warning(f"⚠️ No {resource} found, using fallback: {fallback_value}")

            log.info(f"�� CURRENT RESOURCES: {available_resources}")
            return available_resources

        except Exception as e:
            log.warning(f"⚠️ Failed to extract resources: {e}")
            return {'gold': 100, 'wood': 50, 'stone': 50, 'food': 50, 'land': 1}

    async def _calculate_smart_training_quantity(self, action):
        """Calculate resource-aware training quantities"""
        try:
            # Get latest game state if available
            if hasattr(self, 'latest_game_state') and self.latest_game_state:
                resources = self.latest_game_state.get("resources", {})
                troops = self.latest_game_state.get("military", {}).get("troops", {})

                available_gold = resources.get("gold", 1000)
                total_troops = sum(troops.values()) if troops else 0

                log.info(f"💰 Available gold: {available_gold}, Current troops: {total_troops}")

                # Early game: train more conservatively
                if total_troops < 100:
                    # Use 30% of gold for training when we have few troops
                    max_gold_for_training = int(available_gold * 0.3)
                    quantity = max(10, min(100, max_gold_for_training // 50))  # Assume 50 gold per troop
                elif total_troops < 500:
                    # Mid game: more aggressive
                    max_gold_for_training = int(available_gold * 0.5)
                    quantity = max(20, min(300, max_gold_for_training // 50))
                else:
                    # Late game: very aggressive
                    max_gold_for_training = int(available_gold * 0.7)
                    quantity = max(50, min(1000, max_gold_for_training // 50))

                log.info(f"🎯 Smart training: {quantity} troops (based on {available_gold} gold, {total_troops} current troops)")
                return quantity
            else:
                # Fallback if no game state available
                return random.randint(10, 100)

        except Exception as e:
            log.warning(f"⚠️ Smart training calculation failed: {e}")
            return random.randint(10, 50)

    async def _calculate_smart_building_quantity(self, action):
        """Calculate resource-aware building quantities"""
        try:
            # Get latest game state if available
            if hasattr(self, 'latest_game_state') and self.latest_game_state:
                resources = self.latest_game_state.get("resources", {})

                available_gold = resources.get("gold", 1000)
                available_land = resources.get("land", 50)  # From cost displays

                log.info(f"🏗️ Available: {available_gold} gold, {available_land} land")

                # CRITICAL: Don't build more than we can afford in land
                # Each building typically costs 1 land
                max_by_land = available_land

                # Don't use all gold for buildings - save some for troops
                max_gold_for_buildings = int(available_gold * 0.4)
                max_by_gold = max_gold_for_buildings // 150  # Assume 150 gold per building average

                # Take the smaller limit (land or gold constraint)
                quantity = min(max_by_land, max_by_gold)

                # Minimum 1, maximum based on actual resources
                quantity = max(1, min(quantity, 50))  # Cap at 50 to avoid huge numbers

                log.info(f"🏗️ Smart building: {quantity} buildings (limited by {available_land} land, {available_gold} gold)")
                return quantity
            else:
                # Fallback if no game state available
                return random.randint(1, 20)

        except Exception as e:
            log.warning(f"⚠️ Smart building calculation failed: {e}")
            return random.randint(1, 10)

    async def _calculate_smart_exploration_quantity(self, action):
        """Calculate strategic exploration troop quantities"""
        try:
            # Get latest game state if available
            if hasattr(self, 'latest_game_state') and self.latest_game_state:
                troops = self.latest_game_state.get("military", {}).get("troops", {})
                total_troops = sum(troops.values()) if troops else 0

                log.info(f"🗡️ Available troops for exploration: {total_troops}")

                if total_troops < 50:
                    # Very few troops - send small exploration parties
                    quantity = max(1, total_troops // 4)  # Send 25% of troops
                elif total_troops < 200:
                    # Some troops - moderate exploration
                    quantity = max(10, total_troops // 3)  # Send 33% of troops
                else:
                    # Lots of troops - aggressive exploration
                    quantity = max(20, total_troops // 2)  # Send 50% of troops

                # Cap at reasonable exploration limits
                quantity = min(quantity, 200)

                log.info(f"🗺️ Smart exploration: {quantity} troops (from {total_troops} available)")
                return quantity
            else:
                # Fallback if no game state available
                return random.randint(10, 50)

        except Exception as e:
            log.warning(f"⚠️ Smart exploration calculation failed: {e}")
            return random.randint(5, 30)

    async def _get_valid_attack_target(self):
        """Get a valid kingdom ID for attacking - BE VERY CAREFUL"""
        try:
            # Get latest game state if available
            if hasattr(self, 'latest_game_state') and self.latest_game_state:
                troops = self.latest_game_state.get("military", {}).get("troops", {})
                total_troops = sum(troops.values()) if troops else 0

                # CRITICAL: Only attack if we have substantial troops
                if total_troops < 500:
                    log.warning(f"⚠️ Not enough troops for attack ({total_troops}). Need 500+. Avoiding attack.")
                    return "000"  # Invalid kingdom to avoid actual attack

                # For now, use nearby kingdom IDs (real players are usually 6000-7000 range)
                # But be conservative - only target kingdoms close to our ID
                nearby_targets = [
                    6046, 6047, 6048, 6049, 6050,  # Close to our kingdom ID 6045
                    6041, 6042, 6043, 6044,        # Also close
                    6051, 6052, 6053               # Slightly further
                ]

                target = random.choice(nearby_targets)
                log.info(f"🎯 Selected attack target: {target} (we have {total_troops} troops)")
                return str(target)
            else:
                # Fallback - avoid attacking by using invalid ID
                log.warning("⚠️ No game state available. Avoiding attack.")
                return "000"

        except Exception as e:
            log.warning(f"⚠️ Target selection failed: {e}")
            return "000"  # Invalid kingdom to avoid attack

    async def _calculate_smart_exploration_distance(self):
        """Calculate strategic exploration distances"""
        try:
            # Get latest game state if available
            if hasattr(self, 'latest_game_state') and self.latest_game_state:
                resources = self.latest_game_state.get("resources", {})
                current_land = resources.get("land", 50)

                # Early game: explore close (safer)
                if current_land < 200:
                    distance = random.randint(5, 15)
                # Mid game: explore further
                elif current_land < 500:
                    distance = random.randint(10, 30)
                # Late game: explore far (more land available)
                else:
                    distance = random.randint(20, 50)

                log.info(f"🗺️ Smart exploration distance: {distance} (based on {current_land} current land)")
                return distance
            else:
                # Fallback if no game state available
                return random.randint(10, 25)

        except Exception as e:
            log.warning(f"⚠️ Distance calculation failed: {e}")
            return random.randint(10, 20)

    async def _extract_overview_data(self, page_text, game_state):
        """Extract key data from overview page"""
        try:
            # Extract basic resources from overview
            gold_match = re.search(r'gold.*?(\d+)', page_text, re.IGNORECASE)
            if gold_match:
                game_state["resources"]["gold"] = int(gold_match.group(1))

            food_match = re.search(r'food.*?(\d+)', page_text, re.IGNORECASE)
            if food_match:
                game_state["resources"]["food"] = int(food_match.group(1))

            # Extract territory/land info
            land_patterns = [
                r'(\d+)\s*acres',
                r'territory[^\d]*(\d+)',
                r'land[^\d]*(\d+)',
                r'(\d+)\s*/\s*\d+.*land'  # Pattern like "192 / 500 land"
            ]

            for pattern in land_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match and 1 <= int(match.group(1)) <= 25000:
                    game_state["resources"]["land"] = int(match.group(1))
                    log.info(f"📊 Extracted land: {game_state['resources']['land']} acres")
                    break

        except Exception as e:
            log.debug(f"Failed to extract overview data: {e}")

    async def _extract_military_data(self, page_text, game_state):
        """Extract military information - TROOPS, RESOURCES"""
        try:
            # Extract troop counts - look for patterns like "20 footmen", "3 archers"
            troop_types = [
                ('archers', r'(\d+).*?archers?'),
                ('footmen', r'(\d+).*?footmen'),
                ('foot', r'(\d+).*?foot'),
                ('peasants', r'(\d+).*?peasants?'),
                ('pikemen', r'(\d+).*?pikemen'),
                ('cavalry', r'(\d+).*?cavalry'),
                ('knights', r'(\d+).*?knights?'),
                ('crossbowmen', r'(\d+).*?crossbowmen'),
                ('elites', r'(\d+).*?elites?'),
                ('heavy_cavalry', r'(\d+).*?heavy.*cavalry')
            ]

            total_troops = 0
            game_state["military"]["troops"] = {}

            for troop_name, pattern in troop_types:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    # Take the largest number found (most likely the current troop count)
                    troop_count = max([int(match) for match in matches])
                    if troop_count > 0:
                        game_state["military"]["troops"][troop_name] = troop_count
                        total_troops += troop_count
                        log.info(f"🗡️ Found {troop_count} {troop_name}")

            game_state["military"]["total_troops"] = total_troops
            log.info(f"🗡️ Total military force: {total_troops} troops")

            # Extract resources from warroom if visible (cost displays)
            resource_patterns = [
                (r'(\d+)\s*/\s*(\d+).*?wood', 'wood'),
                (r'(\d+)\s*/\s*(\d+).*?stone', 'stone'),
                (r'(\d+)\s*/\s*(\d+).*?gold', 'gold'),
                (r'(\d+)\s*/\s*(\d+).*?land', 'land'),
                (r'(\d+)\s*/\s*(\d+).*?food', 'food')
            ]

            for pattern, resource_name in resource_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    available = int(match.group(2))  # Second number is what we have
                    game_state["resources"][resource_name] = available
                    log.info(f"💰 Available {resource_name}: {available}")

        except Exception as e:
            log.debug(f"Failed to extract military data: {e}")

    async def _extract_building_data(self, page_text, game_state):
        """Extract building information - RESOURCES, COSTS"""
        try:
            # Extract current building counts
            building_types = [
                ('houses', r'(\d+).*?houses?'),
                ('farms', r'(\d+).*?farms?'),
                ('barracks', r'(\d+).*?barracks?'),
                ('markets', r'(\d+).*?markets?'),
                ('castles', r'(\d+).*?castles?'),
                ('temples', r'(\d+).*?temples?'),
                ('stables', r'(\d+).*?stables?'),
                ('barns', r'(\d+).*?barns?'),
                ('archery_ranges', r'(\d+).*?archery.*ranges?')
            ]

            game_state["buildings"]["current"] = {}

            for building_name, pattern in building_types:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    building_count = max([int(match) for match in matches if int(match) < 10000])  # Avoid cost numbers
                    if building_count > 0:
                        game_state["buildings"]["current"][building_name] = building_count
                        log.info(f"🏠 Current {building_name}: {building_count}")

            # Extract CRITICAL resource info from building cost displays
            # Look for patterns like "2,800 / 192" (needed/available)
            resource_cost_patterns = [
                (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?land', 'land'),
                (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?wood', 'wood'),
                (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?stone', 'stone'),
                (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?gold', 'gold'),
                (r'(\d+(?:,\d+)*)\s*/\s*(\d+).*?food', 'food')
            ]

            for pattern, resource_name in resource_cost_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    needed_str = match.group(1).replace(',', '')
                    available_str = match.group(2).replace(',', '')
                    available = int(available_str)
                    game_state["resources"][resource_name] = available
                    log.info(f"💰 CRITICAL: Available {resource_name}: {available} (from cost display)")

        except Exception as e:
            log.debug(f"Failed to extract building data: {e}")

    async def _extract_research_data(self, page_text, game_state):
        """Extract research information"""
        try:
            # Look for research in progress, available tech
            research_patterns = [
                r'research.*?(\d+).*?progress',
                r'technology.*?(\d+)',
                r'science.*?(\d+)'
            ]

            for pattern in research_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    game_state["research"] = {"progress": int(match.group(1))}
                    break

        except Exception as e:
            log.debug(f"Failed to extract research data: {e}")

    def parse_game_data(self, html_content: str) -> Dict[str, Any]:
        """Extract REAL game data from HTML responses AND detect resource failures"""
        game_data = {}

        # CRITICAL: Check for red text / error messages indicating insufficient resources
        red_text_patterns = [
            r'<[^>]*style[^>]*color[^>]*red[^>]*>([^<]+)</[^>]*>',  # HTML red text
            r'<[^>]*class[^>]*error[^>]*>([^<]+)</[^>]*>',          # Error class
            r'insufficient|not enough|cannot afford|lack|need more|do not have',  # Text patterns
            r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*)',               # "2,800 / 600" patterns
        ]

        has_resource_error = False
        for pattern in red_text_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                has_resource_error = True
                log.warning(f"🚨 DETECTED RESOURCE ERROR: {matches}")
                break

        if has_resource_error:
            game_data['resource_error'] = True
            game_data['action_failed'] = True

        # Extract current resources from cost displays (like "2,800 / 600 gold")
        resource_cost_patterns = [
            (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?gold', 'gold'),
            (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?wood', 'wood'),
            (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?stone', 'stone'),
            (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?food', 'food'),
            (r'(\d+(?:,\d+)*)\s*/\s*(\d+(?:,\d+)*).*?land', 'land'),
        ]

        for pattern, resource_name in resource_cost_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                for needed_str, available_str in matches:
                    try:
                        available = int(available_str.replace(',', ''))
                        needed = int(needed_str.replace(',', ''))

                        # Store both available and needed amounts
                        game_data[f'{resource_name}_available'] = available
                        game_data[f'{resource_name}_needed'] = needed

                        # If needed > available, we have insufficient resources
                        if needed > available:
                            has_resource_error = True
                            game_data['resource_error'] = True
                            log.warning(f"🚨 INSUFFICIENT {resource_name.upper()}: need {needed}, have {available}")

                        log.info(f"💰 EXTRACTED {resource_name}: have {available}, need {needed}")
                        break
                    except:
                        continue

        # Extract networth using various patterns
        nw_patterns = [
            r'networth[^\d]*(\d+)',
            r'net\s*worth[^\d]*(\d+)',
            r'nw[^\d]*(\d+)',
        ]

        for pattern in nw_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match and 100 <= int(match.group(1)) <= 100000:  # Reasonable networth range
                game_data['networth'] = int(match.group(1))
                break

        # Extract territory/acres
        acres_patterns = [
            r'(\d+)\s*acres',
            r'territory[^\d]*(\d+)',
            r'land[^\d]*(\d+)'
        ]

        for pattern in acres_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match and 1 <= int(match.group(1)) <= 25000:  # Valid territory range
                game_data['territory'] = int(match.group(1))
                break

        return game_data

    async def ensure_session(self) -> bool:
        """Ensure we have a valid session, attempt login if needed"""
        # Check if session is still valid
        if self._session_valid and (time.time() - self._last_successful_request) < SESSION_CHECK_INTERVAL:
            return True

        # Validate current session
        if await self.validate_session():
            self._session_valid = True
            return True

        log.warning("Session invalid, attempting to login...")
        self._session_valid = False

        # Try form-based login first (most likely to work)
        if await self.form_login():
            self._session_valid = await self.validate_session()
            if self._session_valid:
                return True

        # Try browser login as fallback
        if await self.browser_login():
            self._session_valid = await self.validate_session()
            if self._session_valid:
                return True
        
        # Try direct API access with credentials as last resort
        log.info("Other login methods failed, trying direct API access...")
        return await self.try_direct_api_access()

    async def form_login(self) -> bool:
        """Attempt session-based authentication using GET requests with credentials"""
        try:
            log.info("Attempting session-based authentication...")
            
            # The game seems to use GET requests with credentials for authentication
            # Let's try to establish a session by accessing game pages with credentials
            auth_params = {
                "accountId": ACCOUNT_ID,
                "token": TOKEN,
                "kingdomId": KINGDOM_ID
            }

            # Try accessing the overview page to establish session
            overview_url = f"{BASE_URL}/overview"
            headers = {
                "Referer": f"{BASE_URL}/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }

            log.info(f"Accessing overview page with credentials: {overview_url}")
            r = await self.client.get(overview_url, params=auth_params, headers=headers, timeout=HTTP_TIMEOUT)

            if r.status_code == 200:
                response_text = r.text
                
                # Check if we got the actual game page (not a login page)
                if ("overview" in response_text.lower() and
                    "logout" in response_text.lower() and
                    not "login" in response_text.lower()):
                    log.info("Session established via overview page")
                    return True
                elif "login" in response_text.lower():
                    log.debug("Still on login page, credentials may be invalid")
                else:
                    log.debug("Got game content, session likely established")
                    return True
            
            # Try buildings page as alternative
            buildings_url = f"{BASE_URL}/buildings"
            log.info(f"Accessing buildings page with credentials: {buildings_url}")
            r = await self.client.get(buildings_url, params=auth_params, headers=headers, timeout=HTTP_TIMEOUT)
            
            if r.status_code == 200:
                response_text = r.text
                if ("buildings" in response_text.lower() and
                    "logout" in response_text.lower()):
                    log.info("Session established via buildings page")
                    return True

            return False

        except Exception as e:
            log.warning(f"Session-based authentication failed: {e}")
            return False

    async def try_direct_api_access(self) -> bool:
        """Try to access the API directly with credentials"""
        try:
            # Try different API endpoints to establish session
            api_endpoints = [
                f"{BASE_URL}/api/account",
                f"{BASE_URL}/api/user",
                f"{BASE_URL}/api/kingdom",
                f"{BASE_URL}/api/game"
            ]
            
            credential_sets = [
                {"accountId": ACCOUNT_ID, "token": TOKEN},
                {"email": USERNAME, "password": PASSWORD},
                {"username": USERNAME, "token": TOKEN},
                {"accountId": ACCOUNT_ID, "email": USERNAME, "token": TOKEN}
            ]
            
            for endpoint in api_endpoints:
                for creds in credential_sets:
                    try:
                        response = await self.client.get(endpoint, params=creds, timeout=HTTP_TIMEOUT)
                        if response.status_code == 200:
                            response_text = response.text
                            if not (response_text.strip().startswith('<!DOCTYPE html>') or '<html>' in response_text):
                                log.info(f"Direct API access successful: {endpoint}")
                                self._session_valid = True
                                self._last_successful_request = time.time()
                                return True
                    except Exception as e:
                        log.debug(f"Direct API access failed for {endpoint}: {e}")
                        continue

            log.warning("All login and API access attempts failed")
            return False

        except Exception as e:
            log.warning(f"Direct API access failed: {e}")
            return False

    @retry(
        stop=stop_after_attempt(RETRIES),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def api_call(self, endpoint: str, params: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
        """Make an API call with automatic retry and session management"""
        # Ensure we have a valid session
        if not await self.ensure_session():
            raise SessionError("Failed to establish valid session")

        try:
            # Add proper headers for game requests
            headers = {
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json" if method.upper() == "POST" else "text/html",
                "Referer": f"{BASE_URL}/overview",
                "X-Requested-With": "XMLHttpRequest",  # Important for AJAX requests
            }

            # Make the request based on method
            if method.upper() == "POST":
                # For game actions, send data as JSON in request body
                response = await self.client.post(endpoint, json=params, headers=headers, timeout=HTTP_TIMEOUT)
            else:
                # For queries, use GET with query parameters
                response = await self.client.get(endpoint, params=params, headers=headers, timeout=HTTP_TIMEOUT)
                
            response.raise_for_status()

            # Handle response content
            response_text = response.text
            log.debug(f"API response from {endpoint}: {response_text[:500]}...")

            # Check if response is HTML - but some game APIs return HTML with embedded data
            if response_text.strip().startswith('<!DOCTYPE html>') or '<html>' in response_text:
                log.info("Received HTML response, analyzing for game data...")
                
                # Look for success indicators in HTML
                response_lower = response_text.lower()
                
                # Check for common success/error indicators in HTML
                if any(indicator in response_lower for indicator in [
                    'success', 'trained', 'built', 'completed',
                    'population increased', 'building constructed'
                ]):
                    log.info("HTML response contains success indicators")
                    self._last_successful_request = time.time()
                    return {"response": "success", "content": response_text[:200], "status_code": response.status_code}
                
                # Check for error indicators (more comprehensive)
                elif any(indicator in response_lower for indicator in [
                    'error', 'failed', 'insufficient', 'not enough', 'cannot',
                    'invalid', 'forbidden', 'unauthorized', 'no troops', 'no resources',
                    'not available', 'must have', 'require', 'missing'
                ]):
                    log.warning("HTML response contains error indicators")
                    return {"response": "error", "content": response_text[:200], "status_code": response.status_code}
                
                # Check if it's a login page
                elif any(indicator in response_lower for indicator in ['login', 'sign in', 'password']):
                    log.warning("Received login page, session expired")
                    self._session_valid = False

                    # Try re-authentication
                    if await self.ensure_session():
                        log.info("Re-authentication successful, retrying request")
                        if method.upper() == "POST":
                            response = await self.client.post(endpoint, json=params, timeout=HTTP_TIMEOUT)
                        else:
                            response = await self.client.get(endpoint, params=params, timeout=HTTP_TIMEOUT)
                        response.raise_for_status()
                        response_text = response.text
                        
                        # Check the retried response
                        if not (response_text.strip().startswith('<!DOCTYPE html>') or '<html>' in response_text):
                            # Got non-HTML response after re-auth, process normally
                            pass
                        else:
                            # Still HTML, but might contain game data
                            log.info("Still HTML after re-auth, treating as game response")
                            return {"response": "html_content", "content": response_text[:500], "status_code": response.status_code}
                    else:
                        raise SessionError("Re-authentication failed")
                else:
                    # Parse REAL game data from HTML instead of guessing
                    log.info("Parsing real game data from HTML response")
                    game_data = self.parse_game_data(response_text)
                    self._last_successful_request = time.time()
                    return {"response": "success", "content": response_text[:200], "status_code": response.status_code, "game_data": game_data}

            # Try to parse response
            try:
                # Try JSON first
                result = response.json()
                self._last_successful_request = time.time()
                log.info(f"Successful JSON API call to {endpoint}")
                return result
            except json.JSONDecodeError:
                # Handle non-JSON responses
                if response.status_code == 200:
                    # Some APIs might return plain text success messages
                    result = {"response": response_text.strip(), "status_code": response.status_code}
                    self._last_successful_request = time.time()
                    log.info(f"Successful text API call to {endpoint}: {response_text[:100]}...")
                    return result
                else:
                    raise ApiError(f"Non-JSON response with status {response.status_code}: {response_text[:200]}...")
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                log.warning("Authentication error, invalidating session")
                self._session_valid = False
                raise SessionError("Authentication failed")
            elif e.response.status_code == 405:
                log.warning(f"Method {method} not allowed for {endpoint}")
                # Try the opposite method if POST/GET failed
                if method.upper() == "POST":
                    log.info(f"POST failed, trying GET for {endpoint}")
                    return await self.api_call(endpoint, params, method="GET")
                else:
                    raise ApiError(f"Method {method} not allowed for {endpoint}")
            else:
                raise
        except Exception as e:
            log.error(f"API call failed: {e}")
            raise

# ---------- Actions ----------
async def create_client() -> ApiClient:
    """Create and configure HTTP client with proper session cookies"""
    log.info("Creating HTTP/2 client")
    
    # Set up session cookies from browser to maintain proper authentication
    cookies = {
        "_gid": "GA1.2.1719491477.1756692416",
        "__gads": "ID=ed96232dc70d1feb:T=1756692416:RT=1756692416:S=ALNI_MYJE1gHSDvxYD4oUQq_qO0OoVZdbg",
        "__gpi": "UID=0000111093cca876:T=1756692416:RT=1756692416:S=ALNI_MaMijdeunbjYhXdfQU7vTN2Mt4Wrg",
        "__eoi": "ID=d2b26359208dc4b3:T=1756692416:RT=1756692416:S=AA-AfjbDQQ57X74wd-ONLoZzOwNS",
        "_ga_D9Q30H7QXH": "GS2.1.s1756692416$o1$g1$t1756692483$j60$l0$h0",
        "_ga": "GA1.2.1059080654.1756692416",
        "_gat_gtag_UA_111624307_1": "1"
    }

    client = httpx.AsyncClient(
        http2=HTTP2_ENABLED,
        timeout=HTTP_TIMEOUT,
        cookies=cookies,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": f"{BASE_URL}/overview",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }
    )

    log.info("✅ Session cookies configured for proper game authentication")

    api_client = ApiClient(client=client)

    # Discover working endpoints
    await api_client.discover_endpoints()

    return api_client

async def train_population(
    api_client: ApiClient,
    troop_type: str,
    quantity: int
) -> Dict[str, Any]:
    """Train population using POST requests to the TrainPopulation API"""
    if troop_type not in TROOP_TYPES:
        raise ValueError(f"Unknown troop type: {troop_type}. Available: {list(TROOP_TYPES.keys())}")

    pop_type_id = TROOP_TYPES[troop_type]
    endpoint = api_client._working_endpoints.get("TrainPopulation", f"{BASE_URL}/api/TrainPopulation")

    params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "popTypeId": pop_type_id,
        "quantity": quantity
    }

    log.info(f"🎮 ATTEMPTING REAL BROWSER TRAINING: {quantity} {troop_type}")

    # Try browser automation first (the real solution!)
    try:
        success = await api_client.browser_action("train",
                                                troop_type=troop_type,
                                                quantity=quantity,
                                                pop_type_id=pop_type_id)
        if success:
            log.info(f"✅ BROWSER TRAINING SUCCESS: {quantity} {troop_type}")
            return {"success": True, "message": f"Browser trained {quantity} {troop_type}"}
    except Exception as e:
        log.error(f"❌ Browser training failed: {e}")

    # Fallback to old API method (will still fail but keeps bot running)
    log.warning(f"🔄 Browser failed, trying API fallback for {quantity} {troop_type}")
    log.info(f"Training {quantity} {troop_type} (popTypeId: {pop_type_id})")
    return await api_client.api_call(endpoint, params, method="POST")

async def build_building(
    api_client: ApiClient,
    building_type: str,
    quantity: int = 1
) -> Dict[str, Any]:
    """Build buildings using POST requests to the BuildBuilding API"""
    if building_type not in BUILDING_TYPES:
        raise ValueError(f"Unknown building type: {building_type}. Available: {list(BUILDING_TYPES.keys())}")

    building_type_id = BUILDING_TYPES[building_type]
    endpoint = api_client._working_endpoints.get("BuildBuilding", f"{BASE_URL}/api/BuildBuilding")

    params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "buildingTypeId": building_type_id,
        "quantity": quantity
    }

    log.info(f"Building {quantity} {building_type} (buildingTypeId: {building_type_id})")
    return await api_client.api_call(endpoint, params, method="POST")

async def spy_on_kingdom(
    api_client: ApiClient,
    target_kingdom_id: int,
    spy_type: str,
    spy_count: int = 1
) -> Dict[str, Any]:
    """Spy on another kingdom"""
    if spy_type not in SPY_TYPES:
        raise ValueError(f"Unknown spy type: {spy_type}. Available: {list(SPY_TYPES.keys())}")

    spy_type_id = SPY_TYPES[spy_type]

    # Try multiple possible spy endpoints
    endpoints_to_try = [
        api_client._working_endpoints.get("Spy", f"{BASE_URL}/api/Spy"),
        api_client._working_endpoints.get("SpyAction", f"{BASE_URL}/api/SpyAction"),
        api_client._working_endpoints.get("Action", f"{BASE_URL}/api/Action")
    ]

    params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "targetKingdomId": target_kingdom_id,
        "spyTypeId": spy_type_id,
        "spyCount": spy_count
    }

    log.info(f"Spying on kingdom {target_kingdom_id} with {spy_count} {spy_type} spies")

    # Try each endpoint until one works
    for endpoint in endpoints_to_try:
        try:
            return await api_client.api_call(endpoint, params, method="POST")
        except Exception as e:
            log.debug(f"Spy endpoint {endpoint} failed: {e}")
            continue

    # If all spy endpoints fail, try as generic action
    generic_params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "type": "spy",
        "typeId": spy_type_id,
        "amount": spy_count,
        "targetId": target_kingdom_id
    }

    return await api_client.api_call(endpoints_to_try[0], generic_params, method="POST")

async def explore_territory(
    api_client: ApiClient,
    explore_type: str,
    direction: str = "north",
    troop_count: int = 10
) -> Dict[str, Any]:
    """Explore territory with optimal troop count for maximum land gain"""
    if explore_type not in EXPLORE_TYPES:
        raise ValueError(f"Unknown explore type: {explore_type}. Available: {list(EXPLORE_TYPES.keys())}")

    explore_type_id = EXPLORE_TYPES[explore_type]

    endpoints_to_try = [
        api_client._working_endpoints.get("Explore", f"{BASE_URL}/api/Explore"),
        api_client._working_endpoints.get("ExploreTerritory", f"{BASE_URL}/api/ExploreTerritory"),
        api_client._working_endpoints.get("Action", f"{BASE_URL}/api/Action")
    ]

    params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "exploreTypeId": explore_type_id,
        "direction": direction,
        "distance": troop_count  # Use troop count for maximum land gain
    }

    log.info(f"Exploring {direction} with {troop_count} troops for {explore_type} mission")

    for endpoint in endpoints_to_try:
        try:
            return await api_client.api_call(endpoint, params, method="POST")
        except Exception as e:
            log.debug(f"Explore endpoint {endpoint} failed: {e}")
            continue

    # Try as generic action
    generic_params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "type": "explore",
        "typeId": explore_type_id,
        "amount": troop_count,
        "direction": direction
    }

    return await api_client.api_call(endpoints_to_try[0], generic_params, method="POST")

async def attack_kingdom(
    api_client: ApiClient,
    target_kingdom_id: int,
    attack_type: str,
    troop_count: int
) -> Dict[str, Any]:
    """Attack another kingdom"""
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(ATTACK_TYPES.keys())}")

    attack_type_id = ATTACK_TYPES[attack_type]

    endpoints_to_try = [
        api_client._working_endpoints.get("Attack", f"{BASE_URL}/api/Attack"),
        api_client._working_endpoints.get("Raid", f"{BASE_URL}/api/Raid"),
        api_client._working_endpoints.get("Battle", f"{BASE_URL}/api/Battle"),
        api_client._working_endpoints.get("Action", f"{BASE_URL}/api/Action")
    ]

    params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "targetKingdomId": target_kingdom_id,
        "attackTypeId": attack_type_id,
        "troopCount": troop_count
    }

    log.info(f"Attacking kingdom {target_kingdom_id} with {troop_count} troops ({attack_type})")

    for endpoint in endpoints_to_try:
        try:
            return await api_client.api_call(endpoint, params, method="POST")
        except Exception as e:
            log.debug(f"Attack endpoint {endpoint} failed: {e}")
            continue

    # Try as generic action
    generic_params = {
        "accountId": ACCOUNT_ID,
        "token": TOKEN,
        "kingdomId": KINGDOM_ID,
        "type": "attack",
        "typeId": attack_type_id,
        "amount": troop_count,
        "targetId": target_kingdom_id
    }

    return await api_client.api_call(endpoints_to_try[0], generic_params, method="POST")

# ---------- AI Intelligence System ----------
@dataclass
class GameState:
    """Track current game state for intelligent decisions"""
    resources: Dict[str, int] = field(default_factory=dict)
    troops: Dict[str, int] = field(default_factory=dict)
    buildings: Dict[str, int] = field(default_factory=dict)
    known_enemies: List[int] = field(default_factory=list)
    last_spy_time: float = 0
    last_explore_time: float = 0
    last_attack_time: float = 0
    action_count: int = 0
    recent_failures: List[str] = field(default_factory=list)
    spy_failures: int = 0  # Track consecutive spy failures
    estimated_resources: Dict[str, int] = field(default_factory=lambda: {
        'gold': 10000,
        'food': 5000,
        'wood': 3000,
        'stone': 2000,
        'population': 100
    })  # Estimated resources for smart spending
    territory_size: int = 1  # Estimated land/territory owned (acres)
    exploration_attempts: int = 0  # Track exploration tries
    exploration_successes: int = 0  # Track successful explorations
    last_exploration_time: float = 0  # Last exploration duration
    game_phase: str = 'early'  # early, mid, late game phases
    target_analysis: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # Track enemy strength
    recently_attacked: List[int] = field(default_factory=list)  # Track recently attacked kingdoms
    estimated_networth: int = 100  # Estimated kingdom networth
    honor_level: int = 5  # Current honor level (starts at 5)

class AdvancedAI:
    """Intelligent decision-making system"""

    def __init__(self):
        self.state = GameState()
        self.random_seed = random.randint(1, 10000)
        random.seed(self.random_seed)

    def get_random_delay(self) -> int:
        """Generate human-like random delay - FASTER for active gameplay"""
        # Much shorter delays for active gameplay
        base_delay = random.uniform(15, 45)  # 15-45 seconds instead of long delays

        # Add some variance based on time of day (simulate human sleep patterns)
        hour = time.localtime().tm_hour
        if 22 <= hour or hour <= 6:  # Night time - moderate delays
            base_delay *= random.uniform(2.0, 4.0)  # 30-180 seconds
        elif 9 <= hour <= 17:  # Work hours - very short delays
            base_delay *= random.uniform(0.8, 1.5)  # 12-67 seconds
        else:  # Evening - normal delays
            base_delay *= random.uniform(1.2, 2.0)  # 18-90 seconds

        return int(base_delay)

    def should_spy(self) -> bool:
        """Decide if we should spy now - LOWER PRIORITY IN EARLY GAME"""
        time_since_last_spy = time.time() - self.state.last_spy_time

        # Don't spy if we just spied recently (avoid spam)
        if time_since_last_spy < 900:  # Wait at least 15 minutes between spies
            return False

        # Early game: Very low spy priority, focus on building kingdom
        if self.state.territory_size < 1000:
            return (self.state.action_count % 15 == 0 and  # Only every 15 actions
                    time_since_last_spy > 7200)  # Only every 2 hours

        # Mid game: Moderate spy activity
        elif self.state.territory_size < 5000:
            return (self.state.action_count % 8 == 0 or
                    time_since_last_spy > 3600)  # Every hour

        # Late game: More aggressive spying
        else:
            spy_urgency = max(1, self.state.spy_failures * 2)
            return (self.state.action_count % max(3, SPY_FREQUENCY - spy_urgency) == 0 or
                    time_since_last_spy > 1800)  # Every 30 min

    def should_explore(self) -> bool:
        """Decide if we should explore now - STRATEGIC DECISION WITH GAME RULES"""
        time_since_last_explore = time.time() - self.state.last_explore_time

        # CRITICAL: Stop exploring at 25,000 acres (game rule limit)
        if self.state.territory_size >= 25000:
            log.info(f"🚫 Reached exploration limit (25,000 acres). Switching to conquest-only mode!")
            return False

        # Calculate exploration efficiency
        if self.state.exploration_attempts > 0:
            success_rate = self.state.exploration_successes / self.state.exploration_attempts
        else:
            success_rate = 0.5  # Assume 50% until we have data

        # Early game (0-1000 acres) - prioritize exploration heavily
        if self.state.territory_size < 1000:
            self.state.game_phase = 'early'
            return (self.state.action_count % max(1, EXPLORE_FREQUENCY // 2) == 0 or
                    time_since_last_explore > 900)  # Explore every 15 min early game

        # Mid game (1000-10000 acres) - balance exploration and attacks
        elif self.state.territory_size < 10000:
            self.state.game_phase = 'mid'
            # Only explore if success rate is good and we have low troop count
            total_troops = sum(self.state.troops.values())
            if success_rate > 0.3 and total_troops < 100:
                return (self.state.action_count % EXPLORE_FREQUENCY == 0 and
                        time_since_last_explore > 1800)
            else:
                return False  # Stop exploring, focus on attacking

        # Late game (10000-25000 acres) - exploration becoming inefficient
        elif self.state.territory_size < 20000:
            self.state.game_phase = 'late'
            # Only explore if we're really weak and success rate is very high
            total_troops = sum(self.state.troops.values())
            return (success_rate > 0.8 and total_troops < 50 and
                    time_since_last_explore > 3600)  # Very rare exploration

        # End game (20000+ acres) - almost no exploration
        else:
            # Exploration is very slow now, focus on conquest
            return (success_rate > 0.9 and
                    time_since_last_explore > 7200)  # Only if nearly guaranteed success

    def should_attack(self) -> bool:
        """Decide if we should attack someone - STRATEGIC DECISION"""
        if not self.state.known_enemies:
            return False

        total_troops = sum(self.state.troops.values())
        time_since_last_attack = time.time() - self.state.last_attack_time

        # Early game - avoid attacking unless very strong
        if self.state.game_phase == 'early':
            return (total_troops > 100 and
                    random.random() < AGGRESSIVENESS_LEVEL * 0.3 and
                    time_since_last_attack > 2400)  # Wait 40 min, be cautious

        # Mid game - moderate attacking when exploration becomes inefficient
        elif self.state.game_phase == 'mid':
            # Attack more often if exploration is failing
            exploration_efficiency = 0.5
            if self.state.exploration_attempts > 0:
                exploration_efficiency = self.state.exploration_successes / self.state.exploration_attempts
            
            attack_bonus = 1.0
            if exploration_efficiency < 0.3:  # Poor exploration success
                attack_bonus = 2.0  # Double attack frequency
            
            return (total_troops > 75 and
                    random.random() < AGGRESSIVENESS_LEVEL * attack_bonus and
                    time_since_last_attack > max(600, 1200 / attack_bonus))  # 10-20 min based on exploration

        # Late game - aggressive attacking is primary expansion method
        else:
            return (total_troops > 50 and
                    random.random() < AGGRESSIVENESS_LEVEL * 1.5 and
                    time_since_last_attack > 300)  # Attack every 5 min if strong enough

    def choose_target_kingdom(self) -> int:
        """Choose a target kingdom to spy on or attack - GAME RULES OPTIMIZED"""
        base_id = KINGDOM_ID

        # Filter out recently attacked kingdoms to avoid diminishing returns
        available_enemies = [k for k in self.state.known_enemies if k not in self.state.recently_attacked[-10:]]

        if available_enemies:
            total_troops = sum(self.state.troops.values())
            estimated_our_nw = self.state.estimated_networth
            
            # HONOR SYSTEM: Prefer attacking larger/equal kingdoms for honor bonus
            # Target kingdoms with higher IDs (potentially equal/larger networth)
            honor_targets = [k for k in available_enemies if k >= base_id - 100]
            
            # Early game - be cautious, pick smaller targets in honor range
            if self.state.game_phase == 'early' and total_troops < 100:
                safe_honor_targets = [k for k in honor_targets if base_id <= k <= base_id + 300]
                if safe_honor_targets:
                    return random.choice(safe_honor_targets)

            # Mid/late game - prefer larger targets for honor and better loot
            elif total_troops > 150:
                # Strong enough to attack larger kingdoms (honor bonus + more land)
                large_targets = [k for k in honor_targets if k >= base_id]
                if large_targets:
                    return random.choice(large_targets)

            # Fallback to any honor-safe target
            if honor_targets:
                return random.choice(honor_targets)
                
        # Generate new strategic targets based on game rules
        if self.state.game_phase == 'early':
            # Target similar-sized newer kingdoms (honor-neutral to positive)
            return random.randint(base_id, base_id + 400)
        elif self.state.game_phase == 'mid':
            # Target mix around our level, prefer slightly larger for honor
            return random.randint(base_id - 25, base_id + 300)
        else:
            # Late game - target anyone equal or larger (honor system)
            return random.randint(base_id - 50, base_id + 200)

    def get_optimal_troop_type(self) -> str:
        """Choose best troop type based on GAME RULES (RPS system)"""
        current_troops = self.state.troops

        # Prioritize spies if we have spy failures
        if self.state.spy_failures > 0:
            return 'foot'  # Footmen can be trained for spy missions

        # GAME RULES: Rock-Paper-Scissors balance
        # Pikeman > Cavalry > Archers > Infantry

        total_troops = sum(current_troops.values())
        if total_troops == 0:
            return 'foot'  # Start with basic infantry

        # Calculate current force composition percentages
        infantry_count = current_troops.get('foot', 0)  # Footmen = Infantry
        archer_count = current_troops.get('archer', 0)   # Archers
        cavalry_count = current_troops.get('cavalry', 0) # Cavalry
        pike_count = current_troops.get('siege', 0)      # Using siege as pikemen

        infantry_pct = infantry_count / total_troops
        archer_pct = archer_count / total_troops
        cavalry_pct = cavalry_count / total_troops
        pike_pct = pike_count / total_troops

        # BALANCED FORCE STRATEGY (based on game rules)
        # Target: 25% each type for maximum effectiveness

        # Early game focus: Build basic infantry first
        if total_troops < 50:
            if infantry_pct < 0.4:  # Need basic troops first
                return 'foot'
            elif archer_pct < 0.3:  # Then archers for defense
                return 'archer'
            else:
                return 'cavalry'  # Then cavalry for attack power

        # Mid/late game: Maintain RPS balance
        if pike_pct < 0.2:  # Need more pikemen (counter cavalry)
            return 'siege'  # Pikemen equivalent
        elif cavalry_pct < 0.25:  # Need cavalry (counter archers)
            return 'cavalry'
        elif archer_pct < 0.25:  # Need archers (counter infantry)
            return 'archer'
        elif infantry_pct < 0.3:  # Need infantry backbone
            return 'foot'
        else:
            # Balanced force - pick based on strategic need
            return random.choice(['foot', 'archer', 'cavalry', 'siege'])

    def calculate_training_quantity(self, troop_type: str) -> int:
        """Smarter calculation for troop training quantity based on resources and game phase."""
        base_cost = {'foot': 50, 'archer': 75, 'cavalry': 150, 'siege': 300}
        unit_cost = base_cost.get(troop_type, 50)
        available_gold = self.state.estimated_resources.get('gold', 0)


        if 'train' in self.state.recent_failures:
            log.warning("🚨 Recent training failure. Training a minimal quantity.")
            return 1

        # Determine the percentage of gold to use based on game phase
        if self.state.game_phase == 'early':
            gold_percent_to_use = 0.20  # Use 20% of gold in early game
        elif self.state.game_phase == 'mid':
            gold_percent_to_use = 0.40  # Use 40% in mid game
        else: # late game
            gold_percent_to_use = 0.60  # Use 60% in late game

        # Calculate how many units can be afforded with the allocated gold
        gold_for_training = available_gold * gold_percent_to_use
        if unit_cost > 0:
            quantity = int(gold_for_training // unit_cost)
        else:
            quantity = 0 # Avoid division by zero

        # Ensure at least 1 troop is trained if affordable
        if quantity == 0 and available_gold >= unit_cost:
            quantity = 1

        log.info(f"TRAINING CALC: Phase: {self.state.game_phase}, Gold: {available_gold}, "
                 f"Unit: {troop_type}, Cost: {unit_cost}, "
                 f"Qty: {quantity}")

        return quantity

    def calculate_building_quantity(self, building_type: str) -> int:
        """Calculate how many buildings to build, respecting land constraints."""

        # Correctly calculate free land
        total_buildings = sum(self.state.buildings.values())
        free_land = self.state.territory_size - total_buildings

        if free_land <= 0:
            log.warning(f"No free land to build. Buildings: {total_buildings}, Land: {self.state.territory_size}")
            return 0

        base_cost = {
            'Houses': 100, 'Grain Farms': 150, 'Barracks': 300,
            'Archery Ranges': 250, 'Stables': 400, 'Markets': 150,
            'Barns': 200, 'Castles': 1000, 'Temples': 600
        }

        building_cost = base_cost.get(building_type, 100)
        available_gold = self.state.estimated_resources.get('gold', 0)

        max_affordable_by_gold = (available_gold // building_cost) if building_cost > 0 else free_land

        # The number of buildings to build is limited by free land and what we can afford.
        quantity = min(free_land, max_affordable_by_gold)

        # Apply game phase strategy to the potential quantity
        if self.state.game_phase == 'early':
            quantity = min(quantity, 1)
        elif self.state.game_phase == 'mid':
            quantity = min(quantity, max(1, int(free_land * 0.5)))
        else: # late game
            pass # Be aggressive

        # Final check to ensure quantity is not negative
        quantity = max(0, quantity)

        log.info(f"Final Build Decision: {quantity} {building_type}(s). Free Land: {free_land}, Gold Affordable: {max_affordable_by_gold}")

        return quantity

    def get_optimal_building_type(self) -> str:
        """Choose a building to construct based on a simple priority list."""
        # Priority: Houses for population, Farms for food, then economic, then military

        # Check if we need more population capacity
        # This is a simplified check; a real implementation would need to know current population vs max.
        if "Houses" in BUILDING_TYPES and random.random() < 0.4:
             return "Houses"

        # Check if we need more food production
        if "Grain Farms" in BUILDING_TYPES and random.random() < 0.3:
            return "Grain Farms"

        # Prioritize economic buildings
        economic_buildings = ["Markets", "Lumber Yards", "Stone Quarries", "Barns"]
        # Filter for buildings that are actually in the game
        available_economic = [b for b in economic_buildings if b in BUILDING_TYPES]
        if available_economic and random.random() < 0.2:
            return random.choice(available_economic)

        # Fallback to any random building
        return random.choice(list(BUILDING_TYPES.keys()))

    def decide_next_action(self) -> Dict[str, Any]:
        """Decide what action to take next using AI logic - EXPLORATION FIRST STRATEGY"""
        self.state.action_count += 1

        # CRITICAL: Check current land/territory size from real-time resources
        current_land = self.state.territory_size  # This should be updated from page scraping

        # NEW STRATEGY: Only explore until we have 600+ spare land
        if current_land < 600:
            log.info(f"🗺️ EXPLORATION PHASE: {current_land} land < 600 - FOCUS ON EXPLORATION ONLY")

            # Check if we have troops to explore
            total_troops = sum(self.state.troops.values())
            if total_troops == 0:
                log.info("🏗️ No troops available - training 1 troop first")
                return {
                    'action': 'train',
                    'troop_type': 'foot',
                    'quantity': 1
                }
            else:
                log.info(f"🗺️ Exploring with {total_troops} available troops")
                return {
                    'action': 'explore',
                    'explore_type': 'scout',
                    'direction': random.choice(['north', 'south', 'east', 'west']),
                    'troop_count': 1  # Send only 1 troop for exploration
                }
        else:
            log.info(f"BUILDING PHASE: {current_land} land >= 600 - START BUILDING")

            # Now we can start building infrastructure
            action_roll = random.random()

            if action_roll < 0.6:  # 60% chance to build
                building_type = self.get_optimal_building_type()
                quantity = 1  # Build only 1 at a time
                return {
                    'action': 'build',
                    'building_type': building_type,
                    'quantity': quantity
                }
            elif action_roll < 0.8:  # 20% chance to train
                troop_type = self.get_optimal_troop_type()
                quantity = 1  # Train only 1 at a time
                return {
                    'action': 'train',
                    'troop_type': troop_type,
                    'quantity': quantity
                }
            else:  # 20% chance to explore
                total_troops = sum(self.state.troops.values())
                if total_troops > 0:
                    return {
                        'action': 'explore',
                        'explore_type': 'scout',
                        'direction': random.choice(['north', 'south', 'east', 'west']),
                        'troop_count': 1
                    }
                else:
                    # Fallback to training if no troops
                    return {
                        'action': 'train',
                        'troop_type': 'foot',
                        'quantity': 1
                    }
    
    async def execute_intelligent_action(self, api_client: ApiClient) -> Dict[str, Any]:
        """Execute an intelligently chosen action"""
        action_plan = self.decide_next_action()

        try:
            if action_plan['action'] == 'train':
                result = await train_population(
                    api_client,
                    action_plan['troop_type'],
                    action_plan['quantity']
                )
                # Update troop count
                troop_type = action_plan['troop_type']
                self.state.troops[troop_type] = self.state.troops.get(troop_type, 0) + action_plan['quantity']
                
            elif action_plan['action'] == 'build':
                result = await build_building(
                    api_client,
                    action_plan['building_type'],
                    action_plan['quantity']
                )
                # Update building count
                building_type = action_plan['building_type']
                self.state.buildings[building_type] = self.state.buildings.get(building_type, 0) + action_plan['quantity']
                
            elif action_plan['action'] == 'spy':
                result = await spy_on_kingdom(
                    api_client,
                    action_plan['target'],
                    action_plan['spy_type']
                )
                # Add target to known enemies if not already there
                if action_plan['target'] not in self.state.known_enemies:
                    self.state.known_enemies.append(action_plan['target'])
                
            elif action_plan['action'] == 'explore':
                # Track exploration attempt
                self.state.exploration_attempts += 1
                exploration_start_time = time.time()
                
                troop_count = action_plan.get('troop_count', 10)
                result = await explore_territory(
                    api_client,
                    action_plan['explore_type'],
                    action_plan['direction'],
                    troop_count
                )
                
                # Track exploration timing
                self.state.last_exploration_time = time.time() - exploration_start_time

                # Use REAL game data instead of fake estimates!
                if result.get('game_data'):
                    game_data = result['game_data']

                    # CRITICAL: Check for resource errors and adjust strategy
                    if game_data.get('resource_error'):
                        log.error(f"🚨 RESOURCE ERROR detected! Action failed due to insufficient resources")
                        self.state.recent_failures.append(action_plan['action'])
                        # Make quantities more conservative after resource errors
                        for resource in ['gold', 'wood', 'stone', 'food', 'land']:
                            if f'{resource}_available' in game_data:
                                actual_available = game_data[f'{resource}_available']
                                self.state.estimated_resources[resource] = actual_available
                                log.info(f"📊 UPDATED {resource.upper()}: {actual_available}")

                    # Store old values to calculate actual gains
                    old_territory = self.state.territory_size
                    old_networth = self.state.estimated_networth

                    # Update with REAL data from game HTML
                    if 'territory' in game_data:
                        self.state.territory_size = game_data['territory']
                    if 'networth' in game_data:
                        self.state.estimated_networth = game_data['networth']
                    if 'gold' in game_data:
                        self.state.estimated_resources['gold'] = game_data['gold']

                    # UPDATE GAME PHASE based on REAL data
                    current_land = self.state.territory_size
                    current_networth = self.state.estimated_networth

                    if current_land < 100 or current_networth < 2000:
                        self.state.game_phase = 'early'
                        log.info(f"🌱 EARLY GAME: {current_land} land, {current_networth} networth - FOCUS ON EXPLORATION")
                    elif current_land < 500 or current_networth < 10000:
                        self.state.game_phase = 'mid'
                        log.info(f"🏗️ MID GAME: {current_land} land, {current_networth} networth - FOCUS ON BUILDING")
                    else:
                        self.state.game_phase = 'late'
                        log.info(f"⚔️ LATE GAME: {current_land} land, {current_networth} networth - FOCUS ON MILITARY")

                    # Calculate REAL gains (not fake ones)
                    land_gained = self.state.territory_size - old_territory
                    nw_change = self.state.estimated_networth - old_networth

                    if land_gained > 0:
                        self.state.exploration_successes += 1
                        log.info(f"🎯 REAL exploration success! Sent {troop_count} troops, gained {land_gained} REAL acres. Territory: {self.state.territory_size} acres, Networth: {self.state.estimated_networth}")
                    else:
                        log.warning(f"⚠️ Exploration sent {troop_count} troops but gained 0 land - action may have failed! Territory: {self.state.territory_size}, Networth: {self.state.estimated_networth}")
                        self.state.recent_failures.append('explore')
                else:
                    # No game data = action probably failed
                    log.error(f"❌ No real game data found - exploration with {troop_count} troops likely FAILED!")
                
            elif action_plan['action'] == 'attack':
                result = await attack_kingdom(
                    api_client,
                    action_plan['target'],
                    action_plan['attack_type'],
                    action_plan['troop_count']
                )
                
                # Track attacked kingdom (avoid diminishing returns)
                target_id = action_plan['target']
                if target_id not in self.state.recently_attacked:
                    self.state.recently_attacked.append(target_id)
                
                # Keep only last 20 attacked kingdoms
                if len(self.state.recently_attacked) > 20:
                    self.state.recently_attacked.pop(0)
                
                # Estimate honor change based on game rules
                if target_id >= KINGDOM_ID:  # Attacking equal/larger kingdoms increases honor
                    self.state.honor_level = min(100, self.state.honor_level + 1)
                    log.info(f"⭐ Honor increased for attacking larger kingdom! Honor: {self.state.honor_level}")

                # Reduce troop count after attack
                total_troops = sum(self.state.troops.values())
                if total_troops > 0:
                    reduction_ratio = action_plan['troop_count'] / total_troops
                    for troop_type in self.state.troops:
                        self.state.troops[troop_type] = int(self.state.troops[troop_type] * (1 - reduction_ratio))
            else:
                # Unknown action type - fallback to training
                result = await train_population(api_client, 'foot', 1)
            
            # Check if this was a spy mission and update spy failure tracking
            if action_plan['action'] == 'spy':
                # Successful spy mission - reset failure count
                self.state.spy_failures = 0
                log.info(f"🕵️ Spy mission successful! Reset failure count.")
            
            log.info(f"✅ Executed {action_plan['action']} successfully")
            return result
            
        except Exception as e:
            log.warning(f"❌ Action {action_plan['action']} failed: {e}")
            
            # Track spy failures specifically
            if action_plan['action'] == 'spy':
                self.state.spy_failures += 1
                log.warning(f"🚨 Spy mission failed! Total failures: {self.state.spy_failures}. Need more spies!")

                # Update estimated spy count (we probably don't have enough)
                current_spies = self.state.troops.get('foot', 0)  # Assume foot soldiers can spy
                if current_spies < 10:
                    log.info(f"🎭 Low spy count detected ({current_spies}). Prioritizing troop training.")
            
            # Track exploration failures specifically
            elif action_plan['action'] == 'explore':
                # Undo the success we counted earlier
                if self.state.exploration_successes > 0:
                    self.state.exploration_successes -= 1

                success_rate = 0
                if self.state.exploration_attempts > 0:
                    success_rate = self.state.exploration_successes / self.state.exploration_attempts

                log.warning(f"🗺️ Exploration failed! Success rate: {success_rate:.1%}. May switch to attacking soon.")

                # If exploration is consistently failing, escalate to more aggressive stance
                if success_rate < 0.2 and self.state.exploration_attempts > 5:
                    log.info(f"🔥 Exploration failing badly ({success_rate:.1%}). Switching to aggressive conquest mode!")
            
            self.state.recent_failures.append(f"{action_plan['action']}: {str(e)[:100]}")
            # Keep only last 10 failures
            if len(self.state.recent_failures) > 10:
                self.state.recent_failures.pop(0)
            raise

async def concurrent_action(
    action_func: Callable,
    total_quantity: int,
    per_call: int = 1,
    concurrency: int = DEFAULT_CONCURRENCY,
    **kwargs
) -> List[Dict[str, Any]]:
    """Execute actions concurrently with resource checking"""
    if total_quantity <= 0:
        return []
    
    num_calls = (total_quantity + per_call - 1) // per_call
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_action(quantity: int) -> Dict[str, Any]:
        async with semaphore:
            await asyncio.sleep(CALL_SPACING_SEC)  # Rate limiting
            return await action_func(quantity=quantity, **kwargs)
    
    # Create tasks
    tasks = []
    remaining = total_quantity
    
    for i in range(num_calls):
        quantity_this_call = min(per_call, remaining)
        remaining -= quantity_this_call
        
        if quantity_this_call > 0:
            tasks.append(bounded_action(quantity_this_call))
    
    # Execute tasks
    log.info(f"Executing {len(tasks)} concurrent calls with concurrency {concurrency}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_results = []
    errors = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"Call {i+1}: {result}")
        else:
            successful_results.append(result)
    
    if errors:
        log.warning(f"Some calls failed: {errors}")
    
    log.info(f"Completed {len(successful_results)}/{len(tasks)} calls successfully")
    return successful_results

# ---------- CLI ----------
def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="KG2 AI Bot - Advanced Automated Kingdom Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py auto                                    # Full AI mode
  python main.py train --troop foot --qty 10
  python main.py build --building barracks --count 2
  python main.py spy --target 1234 --type infiltrate
  python main.py explore --type scout --direction north
  python main.py attack --target 1234 --type raid --troops 50
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Auto AI mode
    auto_parser = subparsers.add_parser("auto", help="Full AI automation mode")
    auto_parser.add_argument("--aggressiveness", type=float, default=AGGRESSIVENESS_LEVEL,
                            help=f"AI aggressiveness level 0.0-1.0 (default: {AGGRESSIVENESS_LEVEL})")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train troops")
    train_parser.add_argument("--troop", required=True, choices=list(TROOP_TYPES.keys()),
                             help="Type of troop to train")
    train_parser.add_argument("--qty", type=int, required=True,
                             help="Total quantity to train")
    train_parser.add_argument("--per", type=int, default=1,
                             help="Quantity per API call (default: 1)")
    train_parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENCY,
                             help=f"Number of concurrent calls (default: {DEFAULT_CONCURRENCY})")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build buildings")
    build_parser.add_argument("--building", required=True, choices=list(BUILDING_TYPES.keys()),
                             help="Type of building to build")
    build_parser.add_argument("--count", type=int, default=1,
                             help="Number of buildings to build (default: 1)")
    build_parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENCY,
                             help=f"Number of concurrent calls (default: {DEFAULT_CONCURRENCY})")
    
    # Spy command
    spy_parser = subparsers.add_parser("spy", help="Spy on other kingdoms")
    spy_parser.add_argument("--target", type=int, required=True,
                           help="Target kingdom ID")
    spy_parser.add_argument("--type", required=True, choices=list(SPY_TYPES.keys()),
                           help="Type of spy mission")
    spy_parser.add_argument("--count", type=int, default=1,
                           help="Number of spies to send (default: 1)")
    
    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Explore territories")
    explore_parser.add_argument("--type", required=True, choices=list(EXPLORE_TYPES.keys()),
                                help="Type of exploration")
    explore_parser.add_argument("--direction", choices=["north", "south", "east", "west"],
                                default="north", help="Direction to explore (default: north)")
    explore_parser.add_argument("--distance", type=int, default=1,
                                help="Distance to explore (default: 1)")
    
    # Attack command
    attack_parser = subparsers.add_parser("attack", help="Attack other kingdoms")
    attack_parser.add_argument("--target", type=int, required=True,
                               help="Target kingdom ID")
    attack_parser.add_argument("--type", required=True, choices=list(ATTACK_TYPES.keys()),
                               help="Type of attack")
    attack_parser.add_argument("--troops", type=int, required=True,
                               help="Number of troops to send")
    
    return parser

async def main():
    """Main application entry point"""
    # Start health check server in a background thread
    health_thread = threading.Thread(target=run_health_check_server, daemon=True)
    health_thread.start()

    # Check for START_CMD in environment
    start_cmd = env("START_CMD", "auto")  # Default to AI mode
    if start_cmd and len(sys.argv) == 1:
        log.info(f"No CLI args provided. Using START_CMD from .env: {start_cmd}")
        # Remove quotes if present and split properly
        start_cmd = start_cmd.strip('"').strip("'")
        sys.argv.extend(shlex.split(start_cmd))
    
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check for continuous running mode
    keep_alive = env("KEEP_ALIVE", "false").lower() in ["true", "1", "yes"]
    
    # Create API client and AI system
    api_client = await create_client()
    ai_system = AdvancedAI() if args.command == "auto" else None
    
    # Override aggressiveness if specified
    if args.command == "auto" and hasattr(args, 'aggressiveness'):
        global AGGRESSIVENESS_LEVEL
        AGGRESSIVENESS_LEVEL = args.aggressiveness
        log.info(f"🤖 AI aggressiveness set to {AGGRESSIVENESS_LEVEL}")
    
    try:
        while True:
            try:
                if args.command == "auto":
                    # Full AI automation mode
                    if ai_system is not None:
                        log.info("🤖 AI taking intelligent action...")
                        result = await ai_system.execute_intelligent_action(api_client)
                        
                        # Add human-like random delay
                        delay = ai_system.get_random_delay()
                        log.info(f"💤 Waiting {delay} seconds (human-like behavior)...")
                        await asyncio.sleep(delay)
                    else:
                        log.error("AI system not initialized for auto mode")
                        break
                    
                elif args.command == "train":
                    results = await concurrent_action(
                        action_func=lambda **kwargs: train_population(api_client, args.troop, **kwargs),
                        total_quantity=args.qty,
                        per_call=args.per,
                        concurrency=args.concurrent
                    )
                    log.info(f"Training completed. {len(results)} successful calls.")
                    
                elif args.command == "build":
                    results = await concurrent_action(
                        action_func=lambda **kwargs: build_building(api_client, args.building, **kwargs),
                        total_quantity=args.count,
                        per_call=1,  # Build one at a time
                        concurrency=args.concurrent
                    )
                    log.info(f"Building completed. {len(results)} successful calls.")
                    
                elif args.command == "spy":
                    result = await spy_on_kingdom(
                        api_client, 
                        args.target, 
                        args.type, 
                        args.count
                    )
                    log.info(f"Spy mission completed on kingdom {args.target}")
                    
                elif args.command == "explore":
                    result = await explore_territory(
                        api_client,
                        args.type,
                        args.direction,
                        args.distance
                    )
                    log.info(f"Exploration completed: {args.type} {args.direction}")
                    
                elif args.command == "attack":
                    result = await attack_kingdom(
                        api_client,
                        args.target,
                        args.type,
                        args.troops
                    )
                    log.info(f"Attack completed on kingdom {args.target} with {args.troops} troops")
                
                # If keep_alive is disabled, run once and exit
                if not keep_alive:
                    log.info("Single execution completed.")
                    break
                    
                # For AI mode, use random delays; for manual commands, use fixed delay
                if args.command == "auto":
                    # AI mode already handled its own delay above
                    pass
                else:
                    # Regular commands - wait before next execution
                    wait_time = int(env("LOOP_DELAY_MINUTES", "5")) * 60
                    log.info(f"Waiting {wait_time//60} minutes before next execution...")
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                log.error(f"Command failed: {e}")
                
                if not keep_alive:
                    sys.exit(1)
                else:
                    # In continuous mode, wait and retry
                    retry_delay = int(env("ERROR_RETRY_DELAY_MINUTES", "2")) * 60
                    log.info(f"Retrying in {retry_delay//60} minutes...")
                    await asyncio.sleep(retry_delay)
                    continue
                    
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
        
    finally:
        await api_client.client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
