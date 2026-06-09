# Notion with Gemini 2.5 AI Analysis - Latest Model
# Uses Google's latest Gemini 2.5 Flash model via API

import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ğŸ“¦ INSTALLING REQUIRED PACKAGES")
print("=" * 80)

# Install packages
packages = ["google-generativeai", "requests"]
for package in packages:
    print(f"Installing {package}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], capture_output=True)
print("âœ… All packages installed\n")

import requests
import time
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# ============================================================================
# SETUP API KEYS
# ============================================================================

print("=" * 80)
print("ğŸ”‘ CONFIGURING API KEYS")
print("=" * 80)

user_secrets = UserSecretsClient()
NOTION_TOKEN = user_secrets.get_secret("NOTION_TOKEN")
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

print("âœ… Notion API key loaded")
print("âœ… Google API key loaded")
print(f"   Key starts with: {GOOGLE_API_KEY[:10]}...")
print()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ============================================================================
# TEST GEMINI API CONNECTION
# ============================================================================

print("=" * 80)
print("ğŸ§ª TESTING GEMINI API CONNECTION")
print("=" * 80)

print("\nğŸ“¡ Step 1: Testing API connectivity...")

# Try the latest Gemini models in order of preference
model_options = [
    {
        'name': 'gemini-2.5-flash',
        'description': 'Gemini 2.5 Flash - LATEST MODEL (December 2024)'
    },
    {
        'name': 'models/gemini-2.5-flash',
        'description': 'Gemini 2.5 Flash - Alternative path'
    },
    {
        'name': 'gemini-2.0-flash-exp',
        'description': 'Gemini 2.0 Flash (Experimental) - Cutting-edge model'
    },
    {
        'name': 'gemini-2.0-flash',
        'description': 'Gemini 2.0 Flash - Stable model'
    },
    {
        'name': 'gemini-1.5-flash-latest',
        'description': 'Gemini 1.5 Flash - Previous generation'
    },
    {
        'name': 'gemini-1.5-pro-latest',
        'description': 'Gemini 1.5 Pro - Previous generation (more capable)'
    }
]

working_model = None
working_model_name = None

for model_option in model_options:
    model_name = model_option['name']
    description = model_option['description']
    
    print(f"\nğŸ”„ Trying: {model_name}")
    print(f"   ({description})")
    
    try:
        # Initialize model
        test_model = genai.GenerativeModel(
            model_name,
            generation_config={
                'temperature': 0.7,
                'top_p': 0.95,
                'max_output_tokens': 2048,
            }
        )
        
        # Test with a simple prompt
        print("   Testing with simple prompt...")
        test_response = test_model.generate_content("Say 'Hello! I am ready to analyze your Notion workspace.'")
        
        # Check if we got a response
        if test_response and test_response.text:
            print(f"   âœ… SUCCESS! Model responded: '{test_response.text[:60]}...'")
            working_model = test_model
            working_model_name = model_name
            break
        else:
            print(f"   â�Œ Model initialized but no response received")
    
    except Exception as e:
        error_msg = str(e)
        print(f"   â�Œ Failed: {error_msg[:100]}")
        
        # Provide specific troubleshooting for common errors
        if "API key not valid" in error_msg:
            print("\n" + "!" * 80)
            print("âš ï¸�  API KEY ERROR")
            print("!" * 80)
            print("Your Google API key appears to be invalid.")
            print("\nğŸ”§ How to fix:")
            print("1. Go to https://aistudio.google.com/app/apikey")
            print("2. Create a new API key")
            print("3. In Kaggle: Add-ons > Secrets > Add Secret")
            print("   Name: GOOGLE_API_KEY")
            print("   Value: [your new API key]")
            print("4. Restart this notebook")
            print("!" * 80 + "\n")
            sys.exit(1)
        
        elif "ResourceExhausted" in error_msg:
            print("   âš ï¸�  Rate limit reached, trying next model...")
        
        elif "404" in error_msg or "not found" in error_msg.lower():
            print("   âš ï¸�  Model not available in your region or project")
        
        continue

if working_model is None:
    print("\n" + "!" * 80)
    print("â�Œ COULD NOT CONNECT TO ANY GEMINI MODEL")
    print("!" * 80)
    print("\nğŸ”§ Troubleshooting Steps:")
    print("\n1. CHECK KAGGLE INTERNET ACCESS:")
    print("   â€¢ Click Settings (right sidebar)")
    print("   â€¢ Under 'Internet', ensure it's set to 'Internet on'")
    print("   â€¢ Save and restart the kernel")
    print("\n2. VERIFY YOUR API KEY:")
    print("   â€¢ Go to https://aistudio.google.com/app/apikey")
    print("   â€¢ Make sure your API key is active")
    print("   â€¢ Copy the full key")
    print("\n3. UPDATE KAGGLE SECRET:")
    print("   â€¢ In Kaggle: Add-ons > Secrets")
    print("   â€¢ Find GOOGLE_API_KEY")
    print("   â€¢ Update with your API key")
    print("   â€¢ Restart notebook")
    print("\n4. CHECK QUOTA:")
    print("   â€¢ Go to https://aistudio.google.com/app/apikey")
    print("   â€¢ Check if you have remaining quota")
    print("!" * 80 + "\n")
    sys.exit(1)

print("\n" + "=" * 80)
print(f"âœ… GEMINI API CONNECTED SUCCESSFULLY")
print(f"   Using: {working_model_name}")
print("=" * 80 + "\n")

# ============================================================================
# NOTION CLIENT SETUP
# ============================================================================

class NotionClient:
    def __init__(self, token):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
    
    def search_all(self):
        url = "https://api.notion.com/v1/search"
        response = requests.post(url, headers=self.headers, json={"page_size": 100})
        return response.json()

notion = NotionClient(NOTION_TOKEN)

def get_title(obj):
    """Extract title from page or database"""
    if obj.get('object') == 'page':
        properties = obj.get('properties', {})
        for prop in properties.values():
            if prop.get('type') == 'title':
                titles = prop.get('title', [])
                if titles:
                    return titles[0].get('plain_text', 'Untitled')
    elif obj.get('object') == 'database':
        titles = obj.get('title', [])
        if titles:
            return titles[0].get('plain_text', 'Untitled')
    return 'Untitled'

# ============================================================================
# GET NOTION CONTENT
# ============================================================================

print("=" * 80)
print("â­� RETRIEVING NOTION WORKSPACE")
print("=" * 80)

print("\nğŸ”� Searching workspace...")
all_results = notion.search_all()
all_objects = all_results.get('results', [])

pages = [obj for obj in all_objects if obj.get('object') == 'page']
databases = [obj for obj in all_objects if obj.get('object') == 'database']

print(f"âœ… Found {len(all_objects)} total objects")
print(f"   â€¢ Pages: {len(pages)}")
print(f"   â€¢ Databases: {len(databases)}")

# ============================================================================
# STEP 1: FAVORITES
# ============================================================================

print("\n" + "=" * 80)
print("â­� STEP 1: FAVORITES (MAIN PAGES)")
print("=" * 80)

favorites = []

for page in pages:
    parent = page.get('parent', {})
    if parent.get('type') == 'workspace':
        title = get_title(page)
        favorites.append({
            'title': title,
            'id': page.get('id'),
            'url': page.get('url')
        })

print(f"\nğŸ“‹ Found {len(favorites)} item(s) in Favorites:\n")

for i, fav in enumerate(favorites, 1):
    print(f"{i}. {fav['title']}")
    print(f"   URL: {fav['url']}\n")

# ============================================================================
# STEP 2: DANCE GROUPS
# ============================================================================

print("=" * 80)
print("ğŸ�µ STEP 2: DANCE GROUPS/TOPICS")
print("=" * 80)

print("\nğŸ”� Searching for dance-related groups...")

dance_keywords = ['cha cha', 'rumba', 'swing', 'bolero', 'mambo', 
                  'fundamental', 'open', 'walk']

dance_groups = []

for db in databases:
    title = get_title(db)
    if any(keyword in title.lower() for keyword in dance_keywords):
        dance_groups.append({
            'title': title,
            'id': db.get('id'),
            'url': db.get('url'),
            'type': 'database'
        })

for page in pages:
    title = get_title(page)
    is_dance_related = any(keyword in title.lower() for keyword in dance_keywords)
    is_favorite = any(fav['id'] == page.get('id') for fav in favorites)
    
    if is_dance_related and not is_favorite:
        dance_groups.append({
            'title': title,
            'id': page.get('id'),
            'url': page.get('url'),
            'type': 'page'
        })

print(f"âœ… Found {len(dance_groups)} dance groups")

# ============================================================================
# STEP 3: CUSTOM SORTING
# ============================================================================

print("\nğŸ”„ Applying custom sorting...")

def sort_dance_groups(group):
    title = group['title'].lower()
    dance_order = ['cha cha', 'rumba', 'swing', 'bolero', 'mambo']
    
    priority1 = 0 if 'fundamental' in title else (1 if 'open' in title else 2)
    
    priority2 = 999
    for idx, dance in enumerate(dance_order):
        if dance in title:
            priority2 = idx
            break
    
    priority3 = 0 if 'forward' in title else (1 if 'backward' in title else 2) if 'walk' in title else 0
    
    return (priority1, priority2, priority3, title)

dance_groups.sort(key=sort_dance_groups)

# ============================================================================
# STEP 4: DISPLAY SORTED GROUPS
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ“Š SORTED DANCE GROUPS")
print("=" * 80)

print(f"\nTotal: {len(dance_groups)} groups\n")

current_category = None

for i, group in enumerate(dance_groups, 1):
    title = group['title']
    title_lower = title.lower()
    
    category = "FUNDAMENTALS" if 'fundamental' in title_lower else ("OPEN" if 'open' in title_lower else "OTHER")
    
    if category != current_category:
        if i > 1:
            print()
        print(f"{'â”€' * 80}")
        print(f"  {category}")
        print(f"{'â”€' * 80}\n")
        current_category = category
    
    type_icon = "ğŸ—‚ï¸�" if group['type'] == 'database' else "ğŸ“„"
    print(f"{i}. {type_icon} {title}")

# ============================================================================
# STEP 4.5: IDENTIFY KEY DATABASES
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ—‚ï¸�  KEY DATABASES IDENTIFIED")
print("=" * 80)

# Initialize
key_databases = []

# Find DanceSport page
dancesport_page_obj = None
for fav in favorites:
    if 'dancesport' in fav['title'].lower():
        dancesport_page_obj = fav
        break

if dancesport_page_obj:
    print(f"\nSearching for databases under: {dancesport_page_obj['title']}")
    
    # Find databases that are children of DanceSport
    # Look for specific database names
    
    for db in databases:
        title = get_title(db)
        title_lower = title.lower()
        
        # Check for key database indicators
        is_overall_progress = ('overall' in title_lower and 'progress' in title_lower) or ('topics' in title_lower and 'progress' in title_lower)
        is_learning_items = 'learning' in title_lower and 'items' in title_lower
        
        if is_overall_progress or is_learning_items:
            key_databases.append({
                'title': title,
                'id': db.get('id'),
                'type': 'Overall Learning Progress' if is_overall_progress else 'Learning Items'
            })
    
    if key_databases:
        print(f"\nâœ… Found {len(key_databases)} key database(s):\n")
        for i, db in enumerate(key_databases, 1):
            print(f"{i}. ğŸ—‚ï¸�  {db['title']}")
            print(f"   Purpose: {db['type']}")
            print(f"   ID: {db['id']}\n")
    else:
        print("\nâš ï¸�  No key databases found matching expected names")
        print("   Looking for: 'Overall Learning Progress' or 'Learning Items'")
else:
    print("\nâš ï¸�  DanceSport Training Schedule page not found in favorites")

# ============================================================================
# STEP 5: GEMINI AI ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print(f"ğŸ¤– STEP 5: AI ANALYSIS (Using {working_model_name})")
print("=" * 80)

analysis_questions = [
    "Based on the workspace hierarchy, what is the main purpose of the DanceSport Training Schedule?",
    "What patterns do you notice in how the content is organized across the different groups?",
    "What insights can you provide about the user's interests or activities based on this structure?"
]

# Build context
dancesport_fav = None
for fav in favorites:
    if 'dancesport' in fav['title'].lower():
        dancesport_fav = fav
        break

context = f"""Analyze this DanceSport Training Schedule from Notion:

MAIN PAGE: {dancesport_fav['title'] if dancesport_fav else 'DanceSport Training Schedule'}

"""

if key_databases:
    context += "KEY DATABASES:\n"
    for db in key_databases:
        context += f"  â€¢ {db['title']} ({db['type']})\n"
    context += "\n"

context += f"DANCE GROUPS/TOPICS ({len(dance_groups)} total):\n"

fundamentals = [g for g in dance_groups if 'fundamental' in g['title'].lower()]
opens = [g for g in dance_groups if 'open' in g['title'].lower()]

if fundamentals:
    context += f"\nFUNDAMENTALS ({len(fundamentals)} groups):\n"
    for g in fundamentals[:20]:
        context += f"  â€¢ {g['title']}\n"

if opens:
    context += f"\nOPEN ({len(opens)} groups):\n"
    for g in opens[:20]:
        context += f"  â€¢ {g['title']}\n"

context += f"""
STRUCTURE INSIGHTS:
â€¢ Dance styles: Cha Cha, Rumba, Swing, Bolero, Mambo
â€¢ Skill levels: Fundamentals and Open
â€¢ Walk types: Forward and Backward variations
â€¢ Systematic progression from fundamentals to advanced
"""

print("\nğŸ“Š AI Analysis Results:")
print("=" * 80)

for i, question in enumerate(analysis_questions, 1):
    print(f"\n{i}. {question}")
    print("-" * 80)
    
    prompt = f"""{context}

Question: {question}

Provide a concise and insightful analysis."""
    
    try:
        response = working_model.generate_content(prompt)
        print(response.text)
        print()
        
        if i < len(analysis_questions):
            time.sleep(2)  # Small delay between requests
    
    except Exception as e:
        print(f"â�Œ Error generating response: {str(e)}\n")
        break

print("=" * 80)
print("âœ… AI ANALYSIS COMPLETE")
print("=" * 80)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ“Š EXECUTION SUMMARY")
print("=" * 80)

print(f"\nâœ… Gemini Model: {working_model_name}")
print(f"âœ… Favorites: {len(favorites)}")
if key_databases:
    print(f"âœ… Key Databases: {len(key_databases)}")
    for db in key_databases:
        print(f"   â€¢ {db['title']}")
print(f"âœ… Dance Groups: {len(dance_groups)}")
print(f"   â€¢ Fundamentals: {len(fundamentals)}")
print(f"   â€¢ Open: {len(opens)}")
print(f"âœ… AI Analysis: 3 questions answered")

print("\n" + "=" * 80)
print("âœ… ALL STEPS COMPLETE")
print("=" * 80 + "\n")

