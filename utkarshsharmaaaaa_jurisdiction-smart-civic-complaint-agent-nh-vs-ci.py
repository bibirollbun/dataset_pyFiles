!pip install -q --upgrade google-generativeai


import os
import json
from pathlib import Path
from textwrap import dedent
import google.generativeai as genai

print("google-generativeai version:", genai.__version__)


# ============================
# CONFIGURE GEMINI API KEY
# ============================

# Option 1: Paste your key here for private testing
api_key = "AIzaSyAu2602Krj5AHKzzQBrZvOa48yDPwoTW40"  # <-- REPLACE THIS BEFORE RUNNING

# Option 2 (recommended for public notebooks):
# api_key = os.environ.get("GEMINI_API_KEY")

if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    print("âš ï¸� Please insert your Gemini API key above.")
else:
    genai.configure(api_key=api_key)
    print("âœ… Gemini configured successfully.")

model = genai.GenerativeModel("gemini-pro-vision")


# ğŸŸ¦ **CELL 6 â€” CODE (Authority Database)**

# ======================================
# SIMPLE AUTHORITY KNOWLEDGE BASE
# ======================================

AUTHORITIES = {
    "NATIONAL_HIGHWAY_RAIPUR": {
        "name": "Regional Officer, National Highways Authority of India, Raipur",
        "email_primary": "roraipur@nhai.org",
        "email_secondary": "roraipurcg@gmail.com",
        "address": "Regional Office, NHAI, House No. F-1, Anupam Nagar, Near TV Tower, Raipur - 492006"
    },
    "RAIPUR_MUNICIPAL_CORPORATION": {
        "name": "Commissioner, Raipur Municipal Corporation",
        "email_primary": "complaints@raipurmc.example",  # Placeholder email
        "email_secondary": None,
        "address": "Raipur Municipal Corporation, Raipur, Chhattisgarh"
    }
}

print("âœ… Authority knowledge base loaded.")


# ======================================
# IMAGE LOADING HELPER FOR GEMINI
# ======================================

def load_image_for_gemini(image_path: str):
    image_path = Path(image_path)
    ext = image_path.suffix.lower()

    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"

    with open(image_path, "rb") as f:
        return {"mime_type": mime, "data": f.read()}

print("âœ… Image loader ready.")


# ======================================
# ANALYSIS AGENT
# ======================================

def analyze_issue(image_path, location_hint, user_description):
    system_prompt = dedent("""
    You analyze civic issue images for Indian citizens.

    Your job:
    1. Identify the issue (pothole, garbage, broken streetlight, other)
    2. Infer road type:
        - national_highway
        - city_road
        - unsure
    3. Decide jurisdiction:
        - NHAI (National Highways Authority of India)
        - Municipal_Corporation
        - Unknown

    Rules:
    - NH-53 or similar â†’ NHAI
    - Inside colonies/streets â†’ Municipal Corporation

    Return STRICT JSON:

    {
      "issue_type": "",
      "road_type": "",
      "highway_name": "",
      "jurisdiction": "",
      "chosen_authority_key": "",
      "reasoning": ""
    }
    """)

    img_input = load_image_for_gemini(image_path)

    response = model.generate_content(
        [
            system_prompt,
            f"Location hint: {location_hint}",
            f"User description: {user_description}",
            img_input
        ],
        generation_config={"temperature": 0.3}
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") : text.rfind("}")+1]

    return json.loads(text)

print("âœ… Analysis agent ready.")


# ======================================
# MAP ANALYSIS â†’ AUTHORITY
# ======================================

def get_authority_from_result(result):
    key = result.get("chosen_authority_key", "UNKNOWN")

    if key == "NATIONAL_HIGHWAY_RAIPUR":
        return AUTHORITIES["NATIONAL_HIGHWAY_RAIPUR"]
    elif key == "RAIPUR_MUNICIPAL_CORPORATION":
        return AUTHORITIES["RAIPUR_MUNICIPAL_CORPORATION"]
    else:
        return AUTHORITIES["RAIPUR_MUNICIPAL_CORPORATION"]

print("âœ… Authority mapping ready.")


# ======================================
# EMAIL DRAFTING AGENT
# ======================================

def draft_complaint_email(user_name, user_address, user_phone,
                          analysis_result, authority):

    prompt = dedent(f"""
    Write a formal civic complaint email in Indian English.

    Receiver:
    {authority['name']}
    To: {authority['email_primary']}
    CC: {authority['email_secondary']}

    Issue Details:
    {json.dumps(analysis_result, indent=2)}

    Requirements:
    - Write a clear subject line
    - Mention issue + location clearly
    - Polite tone
    - 200â€“250 words
    - End with user's name & phone

    Return STRICT JSON ONLY:

    {{
      "to": "{authority['email_primary']}",
      "cc": "{authority['email_secondary']}",
      "subject": "",
      "body": ""
    }}
    """)

    response = model.generate_content(prompt, generation_config={"temperature": 0.3})

    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") : text.rfind("}")+1]

    return json.loads(text)

print("âœ… Email agent ready.")


import os

print(os.listdir("/kaggle/input"))
print(os.listdir("/kaggle/input/issue-png"))


# USER INFO
user_name = "Rahul Verma"
user_address = "Shankar Nagar, Raipur, Chhattisgarh"
user_phone = "9876543210"

# ISSUE INFO
IMAGE_PATH = "/kaggle/input/issue-png/issue.png"   # <-- UPDATE EXACT NAME
location_hint = "Large pothhole on NH-53 near TV Tower, Raipur"
user_description = "Deep pothole in the middle lane causing vehicles to brake suddenly."

from pathlib import Path

if not Path(IMAGE_PATH).exists():
    print("Error: IMAGE_PATH is incorrect. Please upload the image and update the path.")
else:
    print("ğŸ”� Running analysis...\n")
    analysis = analyze_issue(IMAGE_PATH, location_hint, user_description)
    print(json.dumps(analysis, indent=2))

    authority = get_authority_from_result(analysis)
    print("\n=== Authority Chosen ===")
    print(json.dumps(authority, indent=2))

    email = draft_complaint_email(
        user_name,
        user_address,
        user_phone,
        analysis,
        authority
    )

    print("\n=== Final Email Draft ===")
    print("To:", email["to"])
    print("CC:", email["cc"])
    print("Subject:", email["subject"])
    print("\nBody:\n", email["body"])




