!pip install --quiet google-adk firebase-admin gradio pillow requests


import os
import json
import math
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Google ADK (Agent Development Kit)
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.planners import BuiltInPlanner

# Google GenAI Types
from google.genai import types

# Kaggle Secrets
from kaggle_secrets import UserSecretsClient

# Image Processing
from PIL import Image

print("âœ… All libraries imported successfully")


# Load API Keys from Kaggle Secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key loaded")
except Exception as e:
    print(f"â�Œ GOOGLE_API_KEY error: {e}")

try:
    FIREBASE_CRED_STR = UserSecretsClient().get_secret("FIREBASE_CRED_STR")
    os.environ["FIREBASE_CRED_STR"] = FIREBASE_CRED_STR
    print("âœ… Firebase credentials loaded")
except Exception as e:
    print(f"â�Œ FIREBASE_CRED_STR error: {e}")

# Initialize Firebase/Firestore
if not firebase_admin._apps:
    if FIREBASE_CRED_STR:
        cred_dict = json.loads(FIREBASE_CRED_STR)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("âœ… Firestore initialized")
else:
    db = firestore.client()
    print("âœ… Firestore already initialized")

print("\n" + "="*50)
print("ğŸ�‰ All services ready!")
print("="*50)


# Retry Configuration for Gemini API (handles rate limits)
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Street View Collection Parameters
STREETVIEW_CONFIG = {
    "vantage_offset_meters": 30,      # Distance for offset vantage points (reduced for accuracy)
    "headings": [0, 90, 180, 270],    # Cardinal directions (for outer vantage points)
    "pitches": [0, -10],              # Level and slight downward angle (for steps/ramps)
    "fovs": [90, 50],                 # Wide (context) and medium zoom (detail)
    "image_size": "640x480",          # Street View image dimensions (larger for better AI analysis)
    "max_images_per_location": 40,    # Cap to control API costs
    "use_smart_heading": True,        # Point camera towards building center
}

# ADA Compliance Weights (for Judge Agent scoring)
ADA_WEIGHTS = {
    "ramp_present": 25,               # Major positive if ramp exists
    "steps_without_ramp": -30,        # Major negative
    "handrails_present": 10,          # Positive for safety
    "handrails_missing_on_ramp": -15, # Negative if ramp lacks handrails
    "level_entrance": 20,             # Best case scenario
    "curb_cuts_present": 10,          # Positive for approach
    "accessible_parking": 10,         # Positive if visible
    "obstacles_blocking": -20,        # Negative for obstructions
    "narrow_pathway": -15,            # Negative for width issues
    "uneven_surface": -10,            # Negative for surface condition
}

# Agent App Configuration
APP_NAME = "accessibility-checker"
USER_ID = "user-1"
SESSION_ID = "session-1"

print("âœ… Configuration loaded")
print(f"   ğŸ“¸ Max images per location: {STREETVIEW_CONFIG['max_images_per_location']}")
print(f"   ğŸ�¯ Vantage point offset: {STREETVIEW_CONFIG['vantage_offset_meters']}m")
print(f"   ğŸ§­ Smart heading (point at building): {STREETVIEW_CONFIG['use_smart_heading']}")


# ============================================================
# ğŸ“� TOOL 1: Comprehensive Street View Location Finder (Cloud Run)
# ============================================================
import requests

def find_location_comprehensive(query: str) -> dict:
    """
    Finds a location and gathers comprehensive outdoor Street View imagery
    by calling the deployed Cloud Run service.
    
    Args:
        query: The address or name of the place.
        
    Returns:
        dict: Contains location info and comprehensive street_view_images list.
    """
    print(f"\n{'='*60}")
    print(f"ğŸ”� [SCOUT] Comprehensive Location Search (via Cloud Run)")
    print(f"{'='*60}")
    print(f"ğŸ“� Query: '{query}'")
    
    # Cloud Run Service URL
    SERVICE_URL = "https://streetview-location-finder-cloudrun-169593351486.europe-west1.run.app/find_location"
    
    try:
        print(f"\nğŸ“Œ Connecting to Location Service...")
        response = requests.post(SERVICE_URL, json={"query": query}, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract key info for logging
            name = data.get('name', query)
            address = data.get('address', 'Unknown')
            total_images = data.get('total_images', 0)
            unique_panos = data.get('unique_panoramas', 0)
            lat = data.get('lat')
            lng = data.get('lng')
            
            print(f"   âœ… Location found: {name}")
            print(f"   ğŸ“� Address: {address}")
            print(f"   ğŸ“� Coordinates: ({lat}, {lng})")
            print(f"   ğŸ“¸ Images gathered: {total_images}")
            print(f"   ğŸ�¯ Unique panoramas: {unique_panos}")
            
            print(f"\n{'='*60}")
            return data
        else:
            error_msg = f"Service returned status {response.status_code}: {response.text}"
            print(f"   â�Œ Error: {error_msg}")
            return {"error": error_msg}
            
    except Exception as e:
        print(f"   â�Œ Connection Error: {str(e)}")
        return {"error": str(e)}

print("âœ… find_location_comprehensive() defined (Cloud Run Version)")
print("   â””â”€ Calls external microservice for location & imagery")


# ============================================================
# ğŸ’¾ SECTION 5.2: Save Accessibility Report to Firestore
# ============================================================
# This tool saves the comprehensive accessibility report including:
# - Location info, verdict, score, confidence
# - Detected features and barriers
# - ADA rule check results
# - Image URLs used for analysis
# - Flag for manual review if confidence is low

def save_accessibility_report(
    location_name: str,
    address: str,
    verdict: str,
    score: int,
    confidence: int,
    reason: str,
    features_found: str,
    improvements_needed: str,
    recommended_approach: str,
    ada_rule_checks: str,
    image_urls: str,
    needs_manual_review: bool = False
) -> str:
    """
    Saves a comprehensive accessibility report to Firestore.
    
    Args:
        location_name: Name of the location
        address: Full address
        verdict: ACCESSIBLE, PARTIALLY_ACCESSIBLE, NOT_ACCESSIBLE, NEEDS_VERIFICATION
        score: Accessibility score 0-100
        confidence: Confidence level 0-100
        reason: Detailed explanation of the verdict
        features_found: JSON string of detected accessibility features
        improvements_needed: JSON string of suggested improvements
        recommended_approach: Best entrance/direction for accessibility
        ada_rule_checks: JSON string of ADA compliance checks
        image_urls: JSON string of Street View URLs used for analysis
        needs_manual_review: Flag if confidence is low
        
    Returns:
        str: Confirmation message with Document ID
    """
    print(f"\n{'='*60}")
    print(f"ğŸ’¾ [REPORTER] Saving Accessibility Report")
    print(f"{'='*60}")
    print(f"ğŸ“� Location: {location_name}")
    print(f"ğŸ“Š Score: {score}/100 (Confidence: {confidence}%)")
    print(f"âœ… Verdict: {verdict}")
    
    try:
        doc_ref = db.collection('accessibility_reports').document()
        
        report = {
            # Location Info
            "location_name": location_name,
            "address": address,
            
            # Assessment Results
            "verdict": verdict,
            "score": score,
            "confidence": confidence,
            "reason": reason,
            
            # Detailed Findings
            "features_found": features_found,
            "improvements_needed": improvements_needed,
            "recommended_approach": recommended_approach,
            "ada_rule_checks": ada_rule_checks,
            
            # Evidence
            "image_urls": image_urls,
            
            # Flags
            "needs_manual_review": needs_manual_review,
            
            # Metadata
            "timestamp": firestore.SERVER_TIMESTAMP,
            "version": "2.0"  # Track report version
        }
        
        doc_ref.set(report)
        print(f"   âœ… Report saved! Document ID: {doc_ref.id}")
        print(f"   ğŸš© Manual review needed: {needs_manual_review}")
        print(f"{'='*60}")
        
        return f"Report saved successfully. ID: {doc_ref.id}"
        
    except Exception as e:
        print(f"   â�Œ Error: {str(e)}")
        return f"Error saving report: {str(e)}"


print("âœ… save_accessibility_report() tool defined")


# ============================================================
# ğŸ¤– SECTION 6.0: Agent Setup & Imports
# ============================================================
# Import the planner for AI "thinking" capability

print("="*60)
print("ğŸ¤– INITIALIZING AI AGENTS")
print("="*60)
print("âœ… Planner imports ready (BuiltInPlanner, ThinkingConfig)")


print("\nğŸ“� [1/4] Initializing Scout Agent...")

scout_agent = LlmAgent(
    name="Scout",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=RETRY_CONFIG
    ),
    tools=[find_location_comprehensive],
    instruction="""
    You are the Scout Agent responsible for gathering location intelligence.
    
    Your task:
    1. Use the `find_location_comprehensive` tool with the user's location query
    2. The tool will return comprehensive Street View images from multiple vantage points
    3. Pass ALL returned data to the next agent, especially:
       - 'street_view_images' list with all URLs and their descriptions
       - 'total_images' count
       - 'name' and 'address'
    
    Do NOT summarize or filter the images - pass everything to Vision Agent.
    """
)

print("   âœ… Scout Agent ready")
print("   â””â”€ Model: gemini-2.5-flash-lite")
print("   â””â”€ Tools: find_location_comprehensive")
print("   â””â”€ Thinking: Disabled (simple task)")


print("\nğŸ‘�ï¸� [2/4] Initializing Vision Agent with Thinking...")

vision_agent = LlmAgent(
    name="Vision",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=RETRY_CONFIG
    ),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=8192
        )
    ),
    instruction="""
    You are an Expert Accessibility Analyst with ADA certification.
    
    You will receive Street View images from MULTIPLE vantage points (different positions around the location).
    Each vantage point may have images at different headings, pitches, and zoom levels.
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    COMPREHENSIVE ANALYSIS CHECKLIST - Analyze EVERY image for:
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    ğŸš¶ ENTRANCE ANALYSIS:
    - Count exact number of steps (0, 1, 2, 3+)
    - Identify if there's a ramp (location, estimated slope, surface material)
    - Check for level/flush entrances
    - Note door type (manual, automatic, revolving)
    - Estimate door width (wheelchair accessible requires 32"+ clear)
    
    ğŸ›¡ï¸� SAFETY FEATURES:
    - Handrails present? (one side, both sides, none)
    - Handrail continuity (full length or partial)
    - Tactile warning strips
    - Adequate lighting (visible fixtures)
    - Anti-slip surfaces
    
    ğŸ…¿ï¸� PARKING & APPROACH:
    - Accessible parking signs visible (blue wheelchair symbol)
    - Curb cuts present and properly aligned
    - Pathway width (36"+ required for ADA)
    - Surface condition (cracks, uneven areas, gravel)
    
    ğŸš§ OBSTACLES:
    - Bollards or posts blocking path
    - Temporary barriers (construction, signs)
    - Outdoor furniture blocking access
    - Steep slopes in approach path
    - Gaps or grates in pathway
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    OUTPUT FORMAT (JSON):
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    {
        "step_count": <integer: exact count, 0 if none>,
        "has_ramp": <boolean>,
        "ramp_details": {
            "location": "description of where ramp is",
            "estimated_slope": "gentle/moderate/steep or percentage",
            "has_handrails": <boolean>,
            "handrail_sides": "none/one/both",
            "surface": "concrete/metal/rubber"
        },
        "has_level_entrance": <boolean>,
        "entrance_door": {
            "type": "manual/automatic/revolving/none_visible",
            "width_adequate": <boolean or "uncertain">,
            "notes": "any relevant observations"
        },
        "handrails": {
            "present": <boolean>,
            "location": "steps/ramp/both/none",
            "both_sides": <boolean>,
            "continuous": <boolean>
        },
        "parking": {
            "accessible_spaces_visible": <boolean>,
            "signage_visible": <boolean>,
            "curb_cuts_present": <boolean>,
            "drop_off_area": <boolean>
        },
        "pathway": {
            "width_adequate": <boolean>,
            "surface_condition": "good/fair/poor",
            "obstacles": ["list of obstacles found"],
            "slope": "level/gentle/moderate/steep"
        },
        "confidence_scores": {
            "step_count": <0-100>,
            "ramp_detection": <0-100>,
            "entrance_analysis": <0-100>,
            "parking_analysis": <0-100>,
            "overall": <0-100>
        },
        "analysis_by_vantage": {
            "<vantage_name>": {
                "best_features": ["what this view shows well"],
                "limitations": ["what couldn't be assessed from this angle"],
                "key_observations": "summary"
            }
        },
        "recommended_approach": {
            "direction": "which vantage point shows best access",
            "reason": "why this is the best approach",
            "alternate": "backup approach if primary blocked"
        },
        "image_urls_analyzed": ["list of URLs that were analyzed"],
        "notes": "any additional observations or concerns"
    }
    
    IMPORTANT: 
    - Be specific with counts (not just "has steps" but "3 steps")
    - Confidence scores should reflect image quality and visibility
    - If something is unclear, note it and give lower confidence
    - Consider ALL vantage points before making conclusions
    """
)

print("   âœ… Vision Agent ready")
print("   â””â”€ Model: gemini-2.5-flash")
print("   â””â”€ Thinking: Enabled (budget=8192)")
print("   â””â”€ Analysis: Step count, ramps, handrails, parking, pathways")


print("\nâš–ï¸� [3/4] Initializing Judge Agent with ADA Rules...")

judge_agent = LlmAgent(
    name="Judge",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=RETRY_CONFIG
    ),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=6144
        )
    ),
    instruction="""
    You are a Certified ADA Compliance Officer evaluating accessibility.
    
    Review the comprehensive Vision analysis and apply ADA compliance rules.
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    WEIGHTED ADA COMPLIANCE RULES (Deductions from 100):
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    ğŸš¨ CRITICAL DEDUCTIONS (Severe impact):
    - Steps present AND no ramp available: -30 points
    - Steps > 3 with no ramp: -40 points (additional -10)
    - No accessible entrance visible from ANY angle: -50 points
    
    âš ï¸� MAJOR DEDUCTIONS (Significant impact):
    - Ramp present but no handrails: -15 points
    - Ramp appears too steep (>8.33% / 1:12): -20 points
    - Pathway width appears inadequate: -15 points
    - Significant obstacles blocking path: -20 points
    
    ğŸ“‹ MINOR DEDUCTIONS (Quality of experience):
    - Handrails on only one side: -5 points
    - Door width uncertain/potentially narrow: -10 points
    - Poor pathway surface condition: -10 points
    - No visible accessible parking: -5 points
    - No curb cuts visible: -10 points
    
    âœ… POSITIVE MODIFIERS (Add points):
    - Level entrance available: +10 points (max 100)
    - Automatic doors visible: +5 points (max 100)
    - Clear accessible parking visible: +5 points (max 100)
    - Multiple accessible approaches: +5 points (max 100)
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    CONFIDENCE ADJUSTMENT RULES:
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    - If ANY confidence score < 70%: Flag for manual review
    - If overall confidence < 60%: Verdict should be "NEEDS_VERIFICATION"
    - If only 1-2 images analyzed: Lower confidence by 20%
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    VERDICT THRESHOLDS:
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    - Score 80-100: "ACCESSIBLE" - Meets ADA requirements
    - Score 60-79: "PARTIALLY_ACCESSIBLE" - Usable with limitations
    - Score 40-59: "MODERATELY_ACCESSIBLE" - Significant challenges
    - Score 0-39: "NOT_ACCESSIBLE" - Major barriers present
    - Low confidence: "NEEDS_VERIFICATION" - Recommend on-site check
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    OUTPUT FORMAT (JSON):
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    {
        "location_name": "from context",
        "address": "from context",
        "verdict": "ACCESSIBLE|PARTIALLY_ACCESSIBLE|MODERATELY_ACCESSIBLE|NOT_ACCESSIBLE|NEEDS_VERIFICATION",
        "score": <0-100>,
        "confidence": <0-100>,
        "ada_rule_checks": {
            "critical_issues": [
                {"rule": "description", "deduction": -X, "reason": "why applied"}
            ],
            "major_issues": [...],
            "minor_issues": [...],
            "positive_factors": [
                {"factor": "description", "bonus": +X}
            ],
            "base_score": 100,
            "final_score": <calculated>,
            "deduction_summary": "total points deducted and why"
        },
        "features_found": {
            "accessible": ["list of positive accessibility features"],
            "barriers": ["list of barriers identified"],
            "uncertain": ["items needing verification"]
        },
        "recommended_approach": {
            "primary": "best entrance/direction",
            "reason": "why recommended",
            "alternative": "backup option if available"
        },
        "improvements_needed": [
            {"improvement": "description", "priority": "high/medium/low", "impact": "how it would help"}
        ],
        "needs_manual_review": <boolean>,
        "review_reason": "why manual review is needed (if applicable)",
        "image_evidence": ["URLs of images supporting this verdict"]
    }
    """
)

print("   âœ… Judge Agent ready")
print("   â””â”€ Model: gemini-2.5-flash")
print("   â””â”€ Thinking: Enabled (budget=6144)")
print("   â””â”€ Rules: 15+ weighted ADA compliance checks")


print("\nğŸ“� [4/4] Initializing Reporter Agent...")

reporter_agent = LlmAgent(
    name="Reporter",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=RETRY_CONFIG
    ),
    tools=[save_accessibility_report],
    instruction="""
    You are the Accessibility Report Writer creating the final user-facing report.
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    YOUR RESPONSIBILITIES:
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    1. SAVE THE REPORT using `save_accessibility_report` tool with ALL fields:
       - location_name, address
       - verdict, score, confidence
       - reason (detailed explanation)
       - features_found (as JSON string)
       - improvements_needed (as JSON string)
       - recommended_approach
       - ada_rule_checks (as JSON string)
       - image_urls (as JSON string)
       - needs_manual_review (boolean)
    
    2. CREATE DISABILITY-SPECIFIC RECOMMENDATIONS:
    
    For WHEELCHAIR USERS:
       - Focus on: ramps, door width, pathway width, surface conditions
       - Note: step count, slope percentages, turning space
       - Recommend: best entrance, parking location
    
    For MOBILITY AID USERS (walkers, canes, crutches):
       - Focus on: handrails, surface stability, step height
       - Note: lighting, obstacles at ground level
       - Recommend: safest path, rest areas if visible
    
    For VISUAL IMPAIRMENTS:
       - Focus on: tactile indicators, contrast, signage
       - Note: obstacles, sudden level changes
       - Recommend: approach with best landmarks
    
    3. FORMAT YOUR RESPONSE:
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    ğŸ“� ACCESSIBILITY REPORT: [Location Name]
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    ğŸ“Š SCORE: [XX]/100 ([Verdict])
    ğŸ�¯ CONFIDENCE: [XX]%
    
    ğŸ“‹ SUMMARY:
    [2-3 sentence overview]
    
    âœ… ACCESSIBLE FEATURES:
    - [Feature 1]
    - [Feature 2]
    
    âš ï¸� ACCESSIBILITY BARRIERS:
    - [Barrier 1]
    - [Barrier 2]
    
    ğŸš¶ RECOMMENDED APPROACH:
    [Direction and why]
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    ğŸ‘¥ DISABILITY-SPECIFIC GUIDANCE:
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    ğŸ¦½ Wheelchair Users:
    [Specific guidance]
    
    ğŸ¦¯ Mobility Aid Users:
    [Specific guidance]
    
    ğŸ‘�ï¸� Visual Impairments:
    [Specific guidance]
    
    ğŸ“� IMPROVEMENTS SUGGESTED:
    1. [High priority] - [Improvement]
    2. [Medium priority] - [Improvement]
    
    âš ï¸� NOTE: [If needs manual review, explain why]
    
    Report saved to database âœ…
    """
)

print("   âœ… Reporter Agent ready")
print("   â””â”€ Model: gemini-2.5-flash-lite")
print("   â””â”€ Tools: save_accessibility_report")
print("   â””â”€ Output: Disability-specific recommendations")

print("\n" + "="*60)
print("âœ… ALL 4 AGENTS INITIALIZED SUCCESSFULLY")
print("="*60)


print("ğŸ”— Creating Sequential Agent Pipeline...")
print("   Pipeline: Scout â†’ Vision â†’ Judge â†’ Reporter")

root_agent = SequentialAgent(
    sub_agents=[scout_agent, vision_agent, judge_agent, reporter_agent],
    name="AccessibilityPipelineAgent"
)

print("\nâœ… Pipeline created: AccessibilityPipelineAgent")
print("   â””â”€ Agents connected in sequence for accessibility analysis")


# ============================================================
# ğŸ�ƒ SECTION 8: EXECUTION FUNCTION (Direct Testing)
# ============================================================

import asyncio

APP_NAME = "accessibility-checker"
USER_ID = "user-1"
SESSION_ID = "sess-1"

async def run_accessibility_check(place_query: str):
    """
    Execute the full accessibility check pipeline.
    
    Args:
        place_query: Location name or address to analyze
        
    Returns:
        str: Final accessibility report from the Reporter agent
    """
    print(f"\n{'='*70}")
    print(f"ğŸš€ COMPREHENSIVE ACCESSIBILITY CHECK")
    print(f"{'='*70}")
    print(f"ğŸ“� Target: {place_query}")
    print(f"ğŸ”§ Features: Multi-vantage Street View, AI Thinking, ADA Rules")
    print(f"{'='*70}\n")

    print("ğŸ�ƒ Initializing agent pipeline...")
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

    # Create/ensure session exists
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    print("âœ… Session created")
    print("\nğŸ‘‰ Running analysis (this may take 1-2 minutes for comprehensive check)...\n")

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Check accessibility for: {place_query}")]
    )

    final_response = None
    current_agent = None

    # Stream events from the pipeline
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        # Track which agent is currently working
        if hasattr(event, 'author') and event.author != current_agent:
            current_agent = event.author
            if current_agent:
                agent_emoji = {
                    "Scout": "ğŸ“�",
                    "Vision": "ğŸ‘�ï¸�", 
                    "Judge": "âš–ï¸�",
                    "Reporter": "ğŸ“�"
                }.get(current_agent, "ğŸ¤–")
                print(f"\n{agent_emoji} [{current_agent}] Processing...")
        
        # Capture final response
        if event.is_final_response():
            if event.content and event.content.parts:
                part = event.content.parts[0]
                if hasattr(part, 'text') and part.text:
                    final_response = part.text

    print(f"\n{'='*70}")
    print("âœ… ACCESSIBILITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    if final_response:
        print("\nğŸ“‹ FINAL REPORT:")
        print("-" * 70)
        print(final_response)
        print("-" * 70)
    else:
        print("âš ï¸� No final response captured")
    
    return final_response


# Execute to get the accessibility mapping
await run_accessibility_check("Lombard Street, San Francisco")


# ============================================================
# ğŸ“¦ SECTION 9.1: Gradio Imports & Dependencies
# ============================================================

import gradio as gr
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json

print("âœ… Gradio imports ready")
print("   â””â”€ gradio, requests, PIL, ThreadPoolExecutor, asyncio")


# Helper: Fetch History from Firestore
def get_accessibility_history(limit=10):
    """Fetch recent accessibility reports from Firestore with enhanced display."""
    try:
        reports = db.collection('accessibility_reports').order_by(
            'timestamp', direction=firestore.Query.DESCENDING
        ).limit(limit).stream()
        
        history = []
        for doc in reports:
            data = doc.to_dict()
            score = data.get('score', 'N/A')
            confidence = data.get('confidence', 'N/A')
            verdict = data.get('verdict', 'Unknown')
            location = data.get('location_name', data.get('location', 'Unknown'))
            needs_review = data.get('needs_manual_review', False)
            
            # Verdict emoji
            emoji = {
                "ACCESSIBLE": "âœ…",
                "PARTIALLY_ACCESSIBLE": "âš ï¸�",
                "MODERATELY_ACCESSIBLE": "ğŸ”¶",
                "NOT_ACCESSIBLE": "â�Œ",
                "NEEDS_VERIFICATION": "â�“"
            }.get(verdict, "â�“")
            
            review_flag = " ğŸ”�" if needs_review else ""
            
            history.append(
                f"{emoji} **{location}**{review_flag}\n"
                f"   Score: {score}/100 | Confidence: {confidence}%\n"
                f"   Verdict: {verdict}"
            )
        
        return "\n\n".join(history) if history else "No reports yet. Run an accessibility check!"
    except Exception as e:
        return f"Error fetching history: {str(e)}"


# Helper: Download Street View Images
def download_street_view_image(url):
    """Download Street View image and return as PIL Image."""
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def download_multiple_images(street_view_images):
    """Download all Street View images and return as list of (image, label) tuples."""
    images = []
    for sv in street_view_images:
        img = download_street_view_image(sv["url"])
        if img:
            # Use correct field names from find_location_comprehensive
            vantage = sv.get('vantage_point', sv.get('name', 'Unknown'))
            direction = sv.get('direction', 'N/A')
            heading = sv.get('heading', 'N/A')
            pitch = sv.get('pitch', 0)
            pano_date = sv.get('pano_date', sv.get('date', 'N/A'))
            
            label = f"{vantage} â†’ {direction} | H:{heading}Â° P:{pitch}Â° | {pano_date}"
            images.append((img, label))
    return images


print("âœ… Helper functions defined")
print("   â””â”€ get_accessibility_history() - Fetch Firestore reports")
print("   â””â”€ download_street_view_image() - Download single image")
print("   â””â”€ download_multiple_images() - Download gallery images")


# Thread Pool Executor for Async Handling
executor = ThreadPoolExecutor(max_workers=1)

def run_agent_in_thread(place_query: str):
    """
    Run the comprehensive agent pipeline in a fresh thread with its own event loop.
    Creates agents with full instructions for comprehensive analysis.
    """
    def _run():
        # Create a brand new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create FRESH agents with FULL instructions
            
            # Scout Agent
            thread_scout = LlmAgent(
                name="Scout",
                model=Gemini(model="gemini-2.5-flash-lite", retry_options=RETRY_CONFIG),
                tools=[find_location_comprehensive],
                instruction="""
                You are the Scout Agent. Find the location and gather Street View imagery.
                Use the find_location_comprehensive tool and pass ALL data including street_view_images list to Vision Agent.
                """
            )
            
            # Vision Agent with comprehensive analysis
            thread_vision = LlmAgent(
                name="Vision",
                model=Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG),
                planner=BuiltInPlanner(
                    thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=8192)
                ),
                instruction="""
                You are an Expert Accessibility Analyst analyzing Street View images.
                
                For EACH image, analyze:
                - COUNT exact number of steps (0, 1, 2, 3+)
                - Ramps: presence, slope estimate, handrails
                - Door: type, width adequacy
                - Handrails: present, one/both sides, continuous
                - Parking: accessible spots, signage, curb cuts
                - Pathway: width, surface condition, obstacles
                
                Output JSON with:
                {
                    "step_count": <int>,
                    "has_ramp": <bool>,
                    "ramp_details": {...},
                    "has_level_entrance": <bool>,
                    "handrails": {"present": <bool>, "both_sides": <bool>},
                    "parking": {"accessible_spaces_visible": <bool>, "signage_visible": <bool>},
                    "pathway": {"width_adequate": <bool>, "surface_condition": "good/fair/poor"},
                    "confidence_scores": {"step_count": 0-100, "overall": 0-100},
                    "analysis_by_vantage": {...},
                    "recommended_approach": {...},
                    "image_urls_analyzed": [...]
                }
                """
            )
            
            # Judge Agent with ADA rules
            thread_judge = LlmAgent(
                name="Judge",
                model=Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG),
                planner=BuiltInPlanner(
                    thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=6144)
                ),
                instruction="""
                You are a Certified ADA Compliance Officer.
                
                Apply weighted ADA rules (deductions from 100):
                CRITICAL: Steps + no ramp = -30; Steps>3 + no ramp = -40
                MAJOR: Ramp no handrails = -15; Steep ramp = -20
                MINOR: One-side handrails = -5; No accessible parking = -5
                POSITIVE: Level entrance = +10; Automatic doors = +5
                
                If confidence < 70%, flag for manual review.
                
                Output JSON:
                {
                    "location_name": "...",
                    "address": "...",
                    "verdict": "ACCESSIBLE|PARTIALLY_ACCESSIBLE|...",
                    "score": 0-100,
                    "confidence": 0-100,
                    "ada_rule_checks": {...},
                    "features_found": {...},
                    "improvements_needed": [...],
                    "recommended_approach": {...},
                    "needs_manual_review": <bool>,
                    "image_evidence": [...]
                }
                """
            )
            
            # Reporter Agent with comprehensive output
            thread_reporter = LlmAgent(
                name="Reporter",
                model=Gemini(model="gemini-2.5-flash-lite", retry_options=RETRY_CONFIG),
                tools=[save_accessibility_report],
                instruction="""
                You are the Accessibility Report Writer.
                
                1. SAVE the report using save_accessibility_report with ALL fields:
                   - location_name, address, verdict, score, confidence, reason
                   - features_found, improvements_needed, recommended_approach
                   - ada_rule_checks, image_urls, needs_manual_review
                
                2. Generate DISABILITY-SPECIFIC recommendations:
                   ğŸ¦½ Wheelchair Users: ramps, door width, pathway, surfaces
                   ğŸ¦¯ Mobility Aid Users: handrails, stability, step height
                   ğŸ‘�ï¸� Visual Impairments: landmarks, obstacles, tactile features
                
                Format report clearly with score, verdict, features, barriers, and guidance.
                """
            )
            
            # Create pipeline
            thread_root_agent = SequentialAgent(
                sub_agents=[thread_scout, thread_vision, thread_judge, thread_reporter],
                name="AccessibilityPipelineAgent"
            )
            
            # Run the pipeline
            async def _run_pipeline():
                runner = InMemoryRunner(agent=thread_root_agent, app_name="acc-gradio")
                try:
                    await runner.session_service.create_session(
                        app_name="acc-gradio", 
                        user_id="gradio-user", 
                        session_id="gradio-sess"
                    )
                    
                    user_message = types.Content(
                        role="user",
                        parts=[types.Part(text=f"Check accessibility for: {place_query}")]
                    )
                    
                    final_response = None
                    async for event in runner.run_async(
                        user_id="gradio-user",
                        session_id="gradio-sess",
                        new_message=user_message,
                    ):
                        if event.is_final_response():
                            if event.content and event.content.parts:
                                part = event.content.parts[0]
                                if hasattr(part, 'text') and part.text:
                                    final_response = part.text
                    
                    return final_response
                finally:
                    # Explicitly close the runner to clean up aiohttp sessions
                    if hasattr(runner, 'close'):
                        await runner.close()
                    elif hasattr(runner, 'session_service') and hasattr(runner.session_service, 'close'):
                         await runner.session_service.close()
            
            return loop.run_until_complete(_run_pipeline())
        finally:
            loop.close()
    
    future = executor.submit(_run)
    return future.result(timeout=300)  # 5 minute timeout

print("âœ… Agent runner defined")
print("   â””â”€ run_agent_in_thread() - Executes pipeline in separate thread")
print("   â””â”€ Timeout: 300 seconds (5 minutes)")


def check_accessibility_for_ui(place_query):
    """
    Run the comprehensive accessibility check with STREAMING logs.
    
    This is a generator function that yields progress updates in real-time.
    Each yield updates: (gallery_images, final_response, logs, address, history)
    """
    
    if not place_query or not place_query.strip():
        yield [], "âš ï¸� Please enter a location to analyze", "", "", get_accessibility_history()
        return
    
    logs = []
    gallery_images = []
    address = ""
    
    # ========== STEP 1: Location Search ==========
    logs.append("â•�" * 50)
    logs.append("\nğŸ”� **SCOUT AGENT**: Starting location search...")
    logs.append("\nâ•�" * 50)
    logs.append(f"\nğŸ“� Query: '{place_query}'")
    logs.append("")
    logs.append("\nğŸ”„ Trying Places API (for landmarks/POIs)...")
    yield gallery_images, "ğŸ”„ Searching for location...", "\n".join(logs), address, ""
    
    # Actually find the location
    location_info = find_location_comprehensive(place_query)
    
    if "error" in location_info and location_info["error"]:
        logs.append(f"â�Œ Error: {location_info['error']}")
        yield [], f"â�Œ {location_info['error']}", "\n".join(logs), "", get_accessibility_history()
        return
    
    address = location_info.get('address', place_query)
    name = location_info.get('name', place_query)
    lat = location_info.get('lat')
    lng = location_info.get('lng')
    street_view_images = location_info.get('street_view_images', [])
    total_images = location_info.get('total_images', 0)
    unique_panos = location_info.get('unique_panoramas', 0)
    
    logs.append(f"\nâœ… **Location Found!**")
    logs.append(f"\n   ğŸ“� Name: {name}")
    logs.append(f"\n   ğŸ“� Address: {address}")
    logs.append(f"\n   ğŸ“� Coordinates: ({lat:.6f}, {lng:.6f})")
    logs.append(f"\n   ğŸ“¸ Images gathered: {total_images}")
    logs.append(f"\n   ğŸ�¯ Unique panoramas: {unique_panos}")
    yield gallery_images, "âœ… Location found! Loading images...", "\n".join(logs), address, ""
    
    # ========== STEP 2: Download Images ==========
    logs.append("")
    logs.append("â•�" * 50)
    logs.append("\nğŸ–¼ï¸� **DOWNLOADING STREET VIEW IMAGES**")
    logs.append("\nâ•�" * 50)
    logs.append("\nğŸ”„ Loading images into gallery...")
    yield gallery_images, "ğŸ–¼ï¸� Downloading Street View images...", "\n".join(logs), address, ""
    
    gallery_images = download_multiple_images(street_view_images)
    
    logs.append(f"\nâœ… Loaded {len(gallery_images)} images into gallery")
    yield gallery_images, "âœ… Images loaded! Starting AI analysis...", "\n".join(logs), address, ""
    
    # ========== STEP 3: AI Analysis Pipeline ==========
    logs.append("")
    logs.append("â•�" * 50)
    logs.append("\nğŸ¤– **AI ANALYSIS PIPELINE**")
    logs.append("\nâ•�" * 50)
    yield gallery_images, "ğŸ¤– Running AI analysis pipeline...", "\n".join(logs), address, ""
    
    # Vision Agent
    logs.append("")
    logs.append("\nğŸ‘�ï¸� **VISION AGENT** (AI Thinking: 8192 tokens)")
    logs.append("\n   ğŸ”„ Analyzing all Street View images...")
    logs.append("\n   â€¢ Counting steps and measuring ramp slopes")
    logs.append("\n   â€¢ Detecting handrails and tactile paving")
    logs.append("\n   â€¢ Evaluating parking and pathway conditions")
    logs.append("\n   â€¢ Computing confidence scores per feature")
    yield gallery_images, "ğŸ‘�ï¸� Vision Agent analyzing images...", "\n".join(logs), address, ""
    
    # Judge Agent
    logs.append("")
    logs.append("\nâš–ï¸� **JUDGE AGENT** (AI Thinking: 6144 tokens)")
    logs.append("\n   ğŸ”„ Applying accessibility compliance rules...")
    logs.append("\n   â€¢ Evaluating 15+ weighted compliance rules")
    logs.append("\n   â€¢ Calculating score deductions/bonuses")
    logs.append("\n   â€¢ Determining overall verdict")
    yield gallery_images, "âš–ï¸� Judge Agent evaluating compliance...", "\n".join(logs), address, ""
    
    # Reporter Agent
    logs.append("")
    logs.append("\nğŸ“� **REPORTER AGENT**")
    logs.append("\n   ğŸ”„ Generating comprehensive report...")
    logs.append("\n   â€¢ Saving analysis to Firestore database")
    logs.append("\n   â€¢ Creating disability-specific recommendations")
    yield gallery_images, "ğŸ“� Reporter Agent generating report...", "\n".join(logs), address, ""
    
    # Actually run the agent pipeline
    try:
        logs.append("")
        logs.append("\nâ�³ Processing... (this may take 30-60 seconds)")
        yield gallery_images, "â�³ AI agents working... please wait...", "\n".join(logs), address, ""
        
        final_response = run_agent_in_thread(place_query)
        
        logs.append("")
        logs.append("â•�" * 50)
        logs.append("\nâœ… **ANALYSIS COMPLETE!**")
        logs.append("\nâ•�" * 50)
        
    except Exception as e:
        final_response = f"â�Œ Error during analysis: {str(e)}"
        logs.append("")
        logs.append(f"\nâ�Œ **ERROR**: {str(e)}")
    
    # Final yield with complete results
    history = get_accessibility_history()
    yield gallery_images, final_response or "No response received", "\n".join(logs), address, history

print("âœ… Main UI handler defined (with streaming logs)")
print("   â””â”€ check_accessibility_for_ui() - Generator function for real-time updates")


# ============================================================
# ğŸš€ SECTION 9.5: Launch Gradio UI
# ============================================================

# Custom CSS for scrollable gallery and better styling
custom_css = """
/* Scrollable gallery */
.gallery-container {
    max-height: 500px;
    overflow-y: auto !important;
}
.gradio-gallery {
    overflow-y: auto !important;
}

/* Result box styling */
#result-box {
    max-height: 600px;
    overflow-y: auto;
    padding: 15px;
}
"""

with gr.Blocks(
    title="â™¿ Access-All-Areas: Comprehensive Accessibility Checker",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css=custom_css
) as demo:
    
    # Header
    gr.Markdown("""
    # ğŸ¦½ Access-All-Areas: Agentic Accessibility Mapping
    
    **AI-powered accessibility analysis using Google Street View & Gemini AI with Thinking Capabilities**
    
    ---
    """)
    
    with gr.Row():
        # Left Column - Input & Images
        with gr.Column(scale=1):
            location_input = gr.Textbox(
                label="ğŸ“� Enter Location",
                placeholder="e.g., Empire State Building, New York",
                info="Enter any address, place name, or landmark",
                lines=1
            )
            
            with gr.Row():
                check_btn = gr.Button("ğŸ”� Analyze Accessibility", variant="primary", size="md")
                clear_btn = gr.Button("ğŸ—‘ï¸� Clear", variant="secondary", size="md")
            
            gr.Markdown("### ğŸ—ºï¸� Street View Gallery")
            gr.Markdown("*Images from multiple outdoor vantage points*")
            
            street_view_gallery = gr.Gallery(
                label="Street View Images",
                columns=3,
                rows=1,
                height="auto",
                object_fit="contain"
            )
            address_output = gr.Textbox(label="ğŸ“Œ Verified Address", interactive=False)
        
        # Right Column - Results
        with gr.Column(scale=1):
            gr.Markdown("### ğŸ“‹ Accessibility Report")
            result_output = gr.Markdown(
                value="*Enter a location and click 'Analyze Accessibility' to begin...*",
                elem_id="result-box",
                container=True,
                show_label=False
            )
            
            with gr.Accordion("ğŸ”§ Agent Progress Log", open=False):
                logs_output = gr.Markdown(value="*Analysis logs will appear here...*")
    
    gr.Markdown("---")
    
    # History Section
    with gr.Accordion("ğŸ“š Recent Reports (Firestore Database)", open=True):
        history_output = gr.Markdown(value=get_accessibility_history())
        refresh_btn = gr.Button("ğŸ”„ Refresh History", size="sm")
    
    # Footer
    gr.Markdown("""
    ---
    ### ğŸ“– How It Works:
    1. **Scout Agent** â†’ Finds GPS coordinates & gathers Street View images
    2. **Vision Agent** â†’ Analyzes images with AI thinking, counts steps, detects ramps
    3. **Judge Agent** â†’ Applies 15+ ADA compliance rules with weighted scoring
    4. **Reporter Agent** â†’ Saves to Firestore & generates disability-specific guidance
    """)
    
    # ============================================================
    # Event Handlers
    # ============================================================
    
    check_btn.click(
        fn=check_accessibility_for_ui,
        inputs=[location_input],
        outputs=[street_view_gallery, result_output, logs_output, address_output, history_output]
    )
    
    location_input.submit(
        fn=check_accessibility_for_ui,
        inputs=[location_input],
        outputs=[street_view_gallery, result_output, logs_output, address_output, history_output]
    )
    
    refresh_btn.click(
        fn=get_accessibility_history,
        inputs=[],
        outputs=[history_output]
    )
    
    def clear_inputs():
        return "", [], "*Enter a location and click 'Analyze Accessibility' to begin...*", "*Analysis logs will appear here...*", ""
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[location_input, street_view_gallery, result_output, logs_output, address_output]
    )

# ============================================================
# Launch
# ============================================================
print("=" * 60)
print("ğŸš€ LAUNCHING ACCESS-ALL-AREAS")
print("=" * 60)
print("Features enabled:")
print("   âœ… Multi-vantage outdoor Street View")
print("   âœ… AI Thinking (8192 token budget)")
print("   âœ… ADA compliance weighted rules")
print("   âœ… Confidence scoring")
print("   âœ… Disability-specific recommendations")
print("   âœ… Firestore persistence with image URLs")
print("=" * 60)


# Check if running in Interactive Mode (Editor)
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
    print("ğŸš€ Running in Interactive Mode. Launching Gradio...")
    demo.launch(share=True, debug=True) 
else:
    print("âš ï¸� Running in Commit Mode. Gradio launch skipped to allow commit to finish.")
    # Do NOT launch. This allows the script to finish so you can see your saved notebook.

