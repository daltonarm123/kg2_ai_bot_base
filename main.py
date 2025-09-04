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
            "TrainPopulation": f"{BASE_URL}/api/TrainPopulation",
            "BuildBuilding": f"{BASE_URL}/api/BuildBuilding", 
            "Action": f"{BASE_URL}/api/Action",
            "Spy": f"{BASE_URL}/api/Spy",
            "SpyAction": f"{BASE_URL}/api/SpyAction",
            "Explore": f"{BASE_URL}/api/Explore",
            "ExploreTerritory": f"{BASE_URL}/api/ExploreTerritory",
            "Attack": f"{BASE_URL}/api/Attack",
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
            "Game": f"{BASE_URL}/api/game"
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
                
                # Dispatch to the correct browser helper method
                success = False
                if action_type == "train":
                    success = await self._browser_train_troops(page, params)
                elif action_type == "explore":
                    success = await self._browser_explore(page, params)
                elif action_type == "build":
                    success = await self._browser_build(page, params)
                elif action_type == "spy":
                    success = await self._browser_spy(page, params)
                else:
                    log.error(f"❌ Unknown browser action type: {action_type}")
                    success = False

                await browser.close()
                return success
                
        except Exception as e:
            log.error(f"❌ Browser action {action_type} failed: {e}")
            return False
    
    async def _browser_train_troops(self, page, params):
        """Train troops using intelligent screen analysis"""
        try:
            log.info(f"🎮 Intelligent browser training started...")
            
            # Wait for page to load and take screenshot for analysis
            await page.wait_for_timeout(5000)
            await page.screenshot(path="warroom_analysis.png")
            log.info(f"📸 Warroom screenshot saved for analysis")
            
            # Get page content for analysis
            page_text = await page.content()
            log.info(f"🔍 Page title: {await page.title()}")
            log.info(f"🔍 Page URL: {page.url}")
            
            # CHECK FOR LOGIN PAGE
            if "/login" in page.url:
                log.error("❌ Still on login page - login process failed earlier")
                return False
            
            # Intelligent analysis: Look for training-related elements
            log.info("🧠 Analyzing screen for training options...")
            
            # First, find all clickable elements and analyze them
            all_clickable = await page.query_selector_all('button, a, [onclick], input[type="submit"], [class*="click"]')
            log.info(f"🔍 Found {len(all_clickable)} clickable elements on page")
            
            # Analyze each clickable element for training-related text
            train_candidates = []
            for i, element in enumerate(all_clickable):
                try:
                    text = await element.inner_text()
                    if text and any(keyword in text.lower() for keyword in ['train', 'troops', 'military', 'recruit']):
                        train_candidates.append((element, text.strip()))
                        log.info(f"🎯 Found training candidate {i}: '{text.strip()}'")
                except:
                    pass
            
            if not train_candidates:
                log.warning("⚠️ No obvious training buttons found, looking for plus signs or action buttons...")
                # Look for "+ TRAIN" style buttons or action buttons
                for i, element in enumerate(all_clickable):
                    try:
                        text = await element.inner_text()
                        if text and (text.startswith('+') or 'action' in text.lower()):
                            train_candidates.append((element, text.strip()))
                            log.info(f"🎯 Found action candidate {i}: '{text.strip()}'")
                    except:
                        pass
            
            # Try to click the most promising training candidate
            if not train_candidates:
                log.error("❌ No training options found - taking debug screenshot")
                await page.screenshot(path="debug_no_training_found.png")
                return False
            
            # Click the most promising training option
            best_candidate = train_candidates[0]  # Take the first/best match
            element, text = best_candidate
            
            log.info(f"🎯 Clicking best training candidate: '{text}'")
            await element.click()
            await page.wait_for_timeout(3000)  # Wait for dropdown/menu to appear
            
            # After clicking, take another screenshot to see what opened
            await page.screenshot(path="after_train_click.png")
            log.info(f"📸 Screenshot after clicking training option")
            
            # Now look for troop type options (Foot, Archer, Cavalry, etc.)
            log.info("🧠 Looking for troop type options...")
            
            # Get troop type from params 
            troop_type = params.get('pop_type', 'foot')
            quantity = params.get('quantity', 1)
            
            log.info(f"🎯 Trying to train {quantity} {troop_type} troops")
            
            # Look for troop type buttons/options
            troop_candidates = []
            all_elements = await page.query_selector_all('button, a, [onclick], option, [class*="troop"]')
            
            for element in all_elements:
                try:
                    text = await element.inner_text()
                    if text and troop_type.lower() in text.lower():
                        troop_candidates.append((element, text.strip()))
                        log.info(f"🎯 Found troop type candidate: '{text.strip()}'")
                except:
                    pass
            
            # If we found troop type options, click the right one
            if troop_candidates:
                troop_element, troop_text = troop_candidates[0]
                log.info(f"🎯 Clicking troop type: '{troop_text}'")
                await troop_element.click()
                await page.wait_for_timeout(2000)
            
            # Look for quantity input field
            quantity_inputs = await page.query_selector_all('input[type="number"], input[name*="quantity"], input[name*="amount"]')
            if quantity_inputs:
                quantity_input = quantity_inputs[0]
                log.info(f"🎯 Found quantity input, setting to {quantity}")
                await quantity_input.fill(str(quantity))
                await page.wait_for_timeout(1000)
            
            # Look for final submit/train button
            submit_buttons = await page.query_selector_all('button[type="submit"], input[type="submit"], button:has-text("Train"), button:has-text("Submit")')
            if submit_buttons:
                submit_button = submit_buttons[0]
                submit_text = await submit_button.inner_text()
                log.info(f"🎯 Clicking submit button: '{submit_text}'")
                await submit_button.click()
                await page.wait_for_timeout(2000)
                
                log.info(f"✅ INTELLIGENT BROWSER TRAINING SUCCESS: {quantity} {troop_type}")
                return True
            else:
                log.warning("⚠️ No submit button found after setting up training")
                return False
            
            # Look for troop type (foot soldiers)
            troop_selectors = [
                'button:has-text("Foot")',
                'button:has-text("Infantry")', 
                '.troop-foot',
                'input[name*="foot"]',
                'select option:has-text("Foot")'
            ]
            
            for selector in troop_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element:
                        await element.click()
                        await page.wait_for_timeout(500)
                        log.info(f"✅ Selected foot troops: {selector}")
                        break
                except:
                    continue
            
            # Enter quantity
            quantity = params.get('quantity', 10)
            quantity_selectors = [
                'input[type="number"]',
                'input[name*="quantity"]',
                'input[name*="amount"]',
                '.quantity-input'
            ]
            
            for selector in quantity_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element:
                        await element.fill(str(quantity))
                        await page.wait_for_timeout(500)
                        log.info(f"✅ Entered quantity {quantity}: {selector}")
                        break
                except:
                    continue
            
            # Click train button
            submit_selectors = [
                'button:has-text("Train")',
                'input[type="submit"]',
                'button[type="submit"]',
                '.train-submit',
                '.submit-button'
            ]
            
            for selector in submit_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element:
                        await element.click()
                        await page.wait_for_timeout(2000)
                        log.info(f"✅ Clicked train button: {selector}")
                        return True
                except:
                    continue
                    
            return False
            
        except Exception as e:
            log.error(f"❌ Browser train failed: {e}")
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

    def parse_game_data(self, html_content: str) -> Dict[str, Any]:
        """Parse game data from HTML content using regex"""
        data = {}
        # Simple regex for key-value pairs like "Gold: 1,234"
        patterns = {
            'gold': r"Gold:\s*([\d,]+)",
            'food': r"Food:\s*([\d,]+)",
            'wood': r"Wood:\s*([\d,]+)",
            'stone': r"Stone:\s*([\d,]+)",
            'land': r"Land:\s*([\d,]+)",
            'territory': r"Territory:\s*([\d,]+)\s*Acres",
            'networth': r"Networth:\s*([\d,]+)",
            'population': r"Population:\s*([\d,]+)",
            'foot': r"Footmen:\s*([\d,]+)",
            'archer': r"Archers:\s*([\d,]+)",
            'cavalry': r"Cavalry:\s*([\d,]+)",
            'siege': r"Siege Units:\s*([\d,]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(",", "")
                try:
                    data[key] = int(value_str)
                except ValueError:
                    log.warning(f"Could not parse value for {key}: {value_str}")

        # Check for error messages
        if re.search(r"not enough|insufficient|cannot build", html_content, re.IGNORECASE):
            data['resource_error'] = True

        return data

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

async def update_ai_state(api_client: ApiClient, ai_system: AdvancedAI):
    """Fetch latest game state and update the AI's knowledge base."""
    log.info("📊 Syncing AI state with the latest game data...")
    try:
        # Access a page that shows current resources, like overview
        endpoint = f"{BASE_URL}/overview"
        params = {"accountId": ACCOUNT_ID, "token": TOKEN, "kingdomId": KINGDOM_ID}

        # We expect HTML content from this call. Use the raw client to avoid api_call logic which might parse JSON.
        response = await api_client.client.get(endpoint, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        html_content = response.text
        game_data = api_client.parse_game_data(html_content)

        if not game_data:
            log.warning("Could not parse any game data from the overview page.")
            return

        # Update the AI's state
        state = ai_system.state
        state.resources['gold'] = game_data.get('gold', state.resources.get('gold', 0))
        state.resources['food'] = game_data.get('food', state.resources.get('food', 0))
        state.resources['wood'] = game_data.get('wood', state.resources.get('wood', 0))
        state.resources['stone'] = game_data.get('stone', state.resources.get('stone', 0))

        state.troops['foot'] = game_data.get('foot', state.troops.get('foot', 0))
        state.troops['archer'] = game_data.get('archer', state.troops.get('archer', 0))
        state.troops['cavalry'] = game_data.get('cavalry', state.troops.get('cavalry', 0))
        state.troops['siege'] = game_data.get('siege', state.troops.get('siege', 0))

        state.territory_size = game_data.get('territory', game_data.get('land', state.territory_size))
        state.estimated_networth = game_data.get('networth', state.estimated_networth)

        # Also update the estimated resources used for calculations, which might be different
        state.estimated_resources['gold'] = game_data.get('gold', state.estimated_resources.get('gold', 100))
        state.estimated_resources['wood'] = game_data.get('wood', state.estimated_resources.get('wood', 50))
        state.estimated_resources['stone'] = game_data.get('stone', state.estimated_resources.get('stone', 50))
        # This is the important one for building: free land
        state.estimated_resources['land'] = state.territory_size - sum(state.buildings.values())


        log.info(f"✅ AI state synchronized. Land: {state.territory_size}, Gold: {state.resources['gold']}, Troops: {sum(state.troops.values())}")

    except Exception as e:
        log.error(f"Failed to update AI state: {e}")

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
        """Calculate how many troops to train - ULTRA CONSERVATIVE FOR EARLY GAME"""
        base_cost = {
            'foot': 50,
            'archer': 75,
            'cavalry': 150,
            'siege': 300
        }

        # CRITICAL: Check if we had recent resource failures
        if 'train' in self.state.recent_failures:
            log.warning("🚨 Recent training failures - using MINIMAL quantities")
            return 1  # Only train 1 at a time if we've had failures

        # Use realistic resource estimates based on game phase
        available_gold = self.state.estimated_resources.get('gold', 100)  # Changed from 1000 to 100
        unit_cost = base_cost.get(troop_type, 50)

        # EARLY GAME: Ultra conservative
        if self.state.game_phase == 'early' or self.state.territory_size < 100:
            # Use only 5% of gold for training in early game
            max_affordable = max(1, (available_gold * 0.05) // unit_cost)
            quantity = max(1, min(1, max_affordable))  # NEVER more than 1 in early game
            log.info(f"�� ULTRA-CONSERVATIVE TRAINING: {quantity} {troop_type} troops (using {quantity * unit_cost} gold from {available_gold} available)")

        # MID GAME: Still conservative
        elif self.state.game_phase == 'mid':
            max_affordable = max(1, (available_gold * 0.15) // unit_cost)  # 15% of gold
            quantity = max(1, min(2, max_affordable))  # Changed from 10 to 2
            log.info(f"��️ MID GAME TRAINING: {quantity} {troop_type} troops")

        # LATE GAME: More aggressive
        else:
            max_affordable = max(1, (available_gold * 0.25) // unit_cost)  # 25% of gold
            quantity = max(1, min(5, max_affordable))  # Changed from 20 to 5
            log.info(f"⚔️ LATE GAME TRAINING: {quantity} {troop_type} troops")

        return quantity

    def calculate_exploration_troop_count(self) -> int:
        """Calculate optimal troop count for exploration - but check if we have troops first!"""
        total_troops = sum(self.state.troops.values())
        available_gold = self.state.estimated_resources.get('gold', 1000)

        # CRITICAL: If we have no troops, we can't explore!
        if total_troops == 0:
            log.warning("🚫 Cannot explore - no troops available! Need to train troops first.")
            return 0

        # Early game: Send significant portion for max land gain
        if self.state.territory_size < 500:
            # Send 25-50% of army or at least 5-10 troops
            exploration_force = max(5, int(total_troops * 0.25))
            # But also consider affordability
            if available_gold > 500:
                exploration_force = max(exploration_force, 10)
            return min(exploration_force, min(50, total_troops))  # Don't send more than we have

        # Mid game: Moderate exploration force
        elif self.state.territory_size < 5000:
            exploration_force = max(5, int(total_troops * 0.15))
            return min(exploration_force, min(30, total_troops))

        # Late game: Minimal exploration (should be attacking instead)
        else:
            return max(1, min(5, int(total_troops * 0.05)))

    def get_optimal_building_type(self) -> str:
        """Choose best building type based on GAME RULES & MAINTENANCE"""
        current_buildings = self.state.buildings
        territory = self.state.territory_size

        # Prioritize spy infrastructure if we're having spy failures
        if self.state.spy_failures > 0 and current_buildings.get('spy_den', 0) < 3:
            return 'spy_den'

        # GAME RULES: Buildings need 1% maintenance per hour
        # Prioritize resource production to handle maintenance

        # Early game: Focus on population and basic resources
        if territory < 500:
            if current_buildings.get('Houses', 0) < territory // 20:  # Houses for population
                return 'Houses'
            elif current_buildings.get('Grain Farms', 0) < territory // 30:  # Farms for food
                return 'Grain Farms'
            elif current_buildings.get('Barns', 0) < territory // 50:  # Resource storage
                return 'Barns'
            elif current_buildings.get('Markets', 0) < territory // 50:  # Trade and economy
                return 'Markets'

        # Mid game: Balanced infrastructure
        elif territory < 5000:
            # Resource buildings (maintenance critical)
            if current_buildings.get('Barns', 0) < territory // 40:
                return 'Barns'
            elif current_buildings.get('Markets', 0) < territory // 40:
                return 'Markets'
            elif current_buildings.get('Grain Farms', 0) < territory // 25:
                return 'Grain Farms'
            
            # Population and military
            elif current_buildings.get('Houses', 0) < territory // 15:
                return 'Houses'
            elif current_buildings.get('Barracks', 0) < territory // 100:  # Infantry training
                return 'Barracks'
            elif current_buildings.get('Archery Ranges', 0) < territory // 150:  # Archer training
                return 'Archery Ranges'
            elif current_buildings.get('Stables', 0) < territory // 200:  # Cavalry training
                return 'Stables'

        # Late game: Advanced infrastructure
        else:
            # Ensure massive resource production for large kingdom
            if current_buildings.get('Barns', 0) < territory // 30:
                return 'Barns'
            elif current_buildings.get('Markets', 0) < territory // 30:
                return 'Markets'
            elif current_buildings.get('Grain Farms', 0) < territory // 20:
                return 'Grain Farms'
            
            # Advanced military buildings
            elif current_buildings.get('Castles', 0) < territory // 500:  # Knights & population
                return 'Castles'
            elif current_buildings.get('Embassies', 0) < territory // 1000:  # Spy infrastructure
                return 'Embassies'
            elif current_buildings.get('Temples', 0) < territory // 1000:  # Priests for mana
                return 'Temples'

        # Fallback to basic needs
        return random.choice(['Houses', 'Grain Farms', 'Barns', 'Markets'])

    def calculate_building_quantity(self, building_type: str) -> int:
        """Calculate how many buildings to build, considering available land."""
        base_cost = {
            'Houses': 100, 'Grain Farms': 150, 'Barracks': 300,
            'Archery Ranges': 250, 'Stables': 400, 'Markets': 150,
            'Barns': 200, 'Castles': 1000, 'Temples': 600
        }
        building_cost = base_cost.get(building_type, 100)

        if 'build' in self.state.recent_failures:
            log.warning("🚨 Recent building failures - using MINIMAL quantities")
            return 1

        # Correctly calculate available land
        total_land = self.state.territory_size
        occupied_land = sum(self.state.buildings.values())
        free_land = total_land - occupied_land

        if free_land <= 0:
            log.warning(f"🚫 No free land available to build {building_type}. Total: {total_land}, Occupied: {occupied_land}")
            return 0

        available_gold = self.state.estimated_resources.get('gold', 100)

        # Determine quantity based on game phase
        if self.state.game_phase == 'early' or self.state.territory_size < 100:
            max_by_gold = (available_gold * 0.1) // building_cost
            quantity = min(free_land, max_by_gold)
            quantity = max(0, min(1, quantity)) # Build at most 1
            log.info(f"🌱 EARLY GAME BUILDING: {int(quantity)} {building_type} (Free Land: {free_land})")
        elif self.state.game_phase == 'mid':
            max_by_gold = (available_gold * 0.2) // building_cost
            quantity = min(free_land, max_by_gold)
            quantity = max(0, min(2, quantity)) # Build at most 2
            log.info(f"🏗️ MID GAME BUILDING: {int(quantity)} {building_type} (Free Land: {free_land})")
        else: # Late game
            max_by_gold = (available_gold * 0.3) // building_cost
            quantity = min(free_land, max_by_gold)
            quantity = max(0, min(5, quantity)) # Build at most 5
            log.info(f"⚔️ LATE GAME BUILDING: {int(quantity)} {building_type} (Free Land: {free_land})")
            
        return int(quantity)

    d    def decide_next_action(self) -> Dict[str, Any]:
        """Decide what action to take next using AI logic - EXPLORATION FIRST STRATEGY"""
        self.state.action_count += 1

        # CRITICAL: Check current land/territory size from real-time resources
        current_land = self.state.territory_size  # This should be updated from page scraping

        # NEW STRATEGY: Only explore until we have 600+ spare land
        if current_land < 600:
            log.info(f"�� EXPLORATION PHASE: {current_land} land < 600 - FOCUS ON EXPLORATION ONLY")
            
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
            log.info(f"��️ BUILDING PHASE: {current_land} land >= 600 - START BUILDING")
            
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

        # 2. Check if we should spy (LOW priority in early game)
        if self.should_spy():
            self.state.last_spy_time = time.time()
            spy_type = 'gather_intel'  # Default safe spy type
            if self.state.spy_failures > 0:
                # If we've been failing, try easier spy missions first
                spy_type = random.choice(['infiltrate', 'gather_intel'])
            else:
                # If spying is working, try more aggressive missions
                spy_type = random.choice(list(SPY_TYPES.keys()))

            return {
                'action': 'spy',
                'target': self.choose_target_kingdom(),
                'spy_type': spy_type
            }

        # 3. If we have many spy failures, train troops to fix it
        if self.state.spy_failures > 5:  # Higher threshold
            troop_type = self.get_optimal_troop_type()
            quantity = self.calculate_training_quantity(troop_type)
            return {
                'action': 'train',
                'troop_type': troop_type,
                'quantity': quantity
            }

        if self.should_explore():
            self.state.last_explore_time = time.time()
            return {
                'action': 'explore',
                'explore_type': random.choice(list(EXPLORE_TYPES.keys())),
                'direction': random.choice(['north', 'south', 'east', 'west'])
            }

        if self.should_attack():
            self.state.last_attack_time = time.time()
            total_troops = sum(self.state.troops.values())
            attack_force = int(total_troops * random.uniform(0.3, 0.7))  # Use 30-70% of troops
            return {
                'action': 'attack',
                'target': self.choose_target_kingdom(),
                'attack_type': random.choice(list(ATTACK_TYPES.keys())),
                'troop_count': max(1, attack_force)
            }

        # 4. MAIN ACTIONS: Building kingdom infrastructure
        action_roll = random.random()

        # Adjust priorities based on game phase - FIXED FOR EARLY GAME
        if self.state.game_phase == 'early' or self.state.territory_size < 100:  # TRUE EARLY GAME
            log.info(f"🌱 EARLY GAME STRATEGY: {self.state.territory_size} land - EXPLORATION FOCUSED")

            # EARLY GAME: 80% exploration, 15% troops, 5% buildings
            if action_roll < 0.8:  # 80% chance to explore for land
                if self.should_explore():
                    self.state.last_explore_time = time.time()
                    return {
                        'action': 'explore',
                        'explore_type': 'scout',  # Focus on land gaining
                        'direction': random.choice(['north', 'south', 'east', 'west']),
                        'troop_count': min(10, sum(self.state.troops.values()) // 2)  # Send half troops
                    }
            elif action_roll < 0.95:  # 15% chance to train troops (for exploration)
                troop_type = self.get_optimal_troop_type()
                quantity = max(1, min(5, self.calculate_training_quantity(troop_type)))  # Very small quantities
                return {
                    'action': 'train',
                    'troop_type': troop_type,
                    'quantity': quantity
                }
            else:  # 5% chance to build essential buildings only
                # Only build essential early game buildings
                essential_buildings = ['Houses', 'Grain Farms', 'Barracks']
                building_type = random.choice(essential_buildings)
                quantity = max(1, min(2, self.calculate_building_quantity(building_type)))  # Ultra small quantities
                return {
                    'action': 'build',
                    'building_type': building_type,
                    'quantity': quantity
                }
                
        elif self.state.game_phase == 'mid' or self.state.territory_size < 500:  # MID GAME
            log.info(f"🏗️ MID GAME STRATEGY: {self.state.territory_size} land - BUILDING FOCUSED")

            # MID GAME: 40% building, 30% exploration, 30% troops
            if action_roll < 0.4:  # 40% chance to build infrastructure
                building_type = self.get_optimal_building_type()
                quantity = max(1, min(10, self.calculate_building_quantity(building_type)))  # Conservative quantities
                return {
                    'action': 'build',
                    'building_type': building_type,
                    'quantity': quantity
                }
            elif action_roll < 0.7:  # 30% chance to explore
                if self.should_explore():
                    self.state.last_explore_time = time.time()
                    return {
                        'action': 'explore',
                        'explore_type': random.choice(list(EXPLORE_TYPES.keys())),
                        'direction': random.choice(['north', 'south', 'east', 'west'])
                    }
            else:  # 30% chance to train troops
                troop_type = self.get_optimal_troop_type()
                quantity = max(1, min(20, self.calculate_training_quantity(troop_type)))
                return {
                    'action': 'train',
                    'troop_type': troop_type,
                    'quantity': quantity
                }

        else:  # Mid/late game - more aggressive
            if action_roll < 0.4:  # 40% chance to train troops
                troop_type = self.get_optimal_troop_type()
                quantity = self.calculate_training_quantity(troop_type)
                return {
                    'action': 'train',
                    'troop_type': troop_type,
                    'quantity': quantity
                }
            elif action_roll < 0.7:  # 30% chance to build
                building_type = self.get_optimal_building_type()
                quantity = self.calculate_building_quantity(building_type)
                return {
                    'action': 'build',
                    'building_type': building_type,
                    'quantity': quantity
                }
            else:  # 30% chance to attack
                if self.should_attack():
                    self.state.last_attack_time = time.time()
                    total_troops = sum(self.state.troops.values())
                    attack_force = int(total_troops * random.uniform(0.3, 0.7))
                    return {
                        'action': 'attack',
                        'target': self.choose_target_kingdom(),
                        'attack_type': random.choice(list(ATTACK_TYPES.keys())),
                        'troop_count': max(1, attack_force)
                    }
                else:
                    # Fallback to training
                    troop_type = self.get_optimal_troop_type()
                    quantity = self.calculate_training_quantity(troop_type)
                    return {
                        'action': 'train',
                        'troop_type': troop_type,
                        'quantity': quantity
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
                        # 1. Sync state with the server before making a decision
                        await update_ai_state(api_client, ai_system)

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

# Health check server for deployment
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for health checks"""
        if self.path == "/" or self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "bot": "KG2 AI Bot",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "uptime": time.time() - start_time if 'start_time' in globals() else 0
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress HTTP server logs to keep bot logs clean
        pass

def start_health_server():
    """Start health check server in background thread"""
    port = int(os.environ.get('PORT', 5000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    log.info(f"Health check server starting on port {port}")
    server.serve_forever()

if __name__ == "__main__":
    global start_time
    start_time = time.time()
    
    # Start health check server in background thread for deployment
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    log.info("Health check server started for deployment")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)
