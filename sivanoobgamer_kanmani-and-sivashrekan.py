# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

# Load from .env for local dev, from Secrets for Kaggle
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except:
    # Load from environment variable for local development
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("âœ… Gemini API Key configured" if GOOGLE_API_KEY else "â�Œ Please configure GOOGLE_API_KEY")


from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.genai import types

print("âœ… ADK components imported successfully")


# Application constants
APP_NAME = "SkillGardener"
USER_ID = "default_user"

# Retry configuration for API rate limiting
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… Constants configured")


async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """
    Helper function: Run a conversation session
    
    Args:
        runner_instance: Runner instance
        user_queries: User input (single or multiple)
        session_id: Session ID
    """
    print(f"\n### Session: {session_id}")

    # Create or get session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            # Handle tool call events
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # Display tool invocation
                    if hasattr(part, 'function_call') and part.function_call:
                        func_name = part.function_call.name
                        print(f"  ğŸ”§ Tool called: {func_name}")
                    
                    # Display tool response
                    if hasattr(part, 'function_response') and part.function_response:
                        func_name = part.function_response.name
                        print(f"  âœ… Tool returned: {func_name}")
            
            # Display final response
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text and part.text != "None":
                        print(f"\nAgent > {part.text}")


print("âœ… Helper functions defined")


session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

print("âœ… Session and Memory services initialized")


import os
from pathlib import Path

# Determine skills directory based on environment
# On Kaggle: use /kaggle/working/skills (writable directory)
# Locally: use ./skills (relative to notebook)
if os.path.exists("/kaggle"):
    # Kaggle environment - use working directory (writable)
    SKILLS_DIR = Path("/kaggle/working/skills")
else:
    # Local environment
    SKILLS_DIR = Path("./skills")

print(f"ğŸ“� Skills directory: {SKILLS_DIR}")

# Pre-defined sample skill content (English version for Kaggle submission)
INITIAL_SKILLS = {
    "ecommerce-crawler": {
        "content": """---
name: ecommerce-crawler
description: Scrape product listings and price information from e-commerce websites. Use this skill when you need to collect product data, pricing, or inventory status. Supports JSON or CSV output.
usage_count: 0
last_used_at: null
---

# E-commerce Product Crawler

A standardized workflow for scraping product data from e-commerce websites.

## Use Cases

- Collect product data from e-commerce sites
- Gather pricing information
- Monitor inventory or price changes
- Batch export product listings

## Workflow

1. **Confirm Target**
   - Ask user for target website URL
   - Confirm fields to scrape (name, price, stock, images, etc.)
   - Confirm output format (JSON / CSV)

2. **Analyze Page Structure**
   - Visit target page, analyze HTML structure
   - Identify product list container and individual product elements
   - Extract key selectors

3. **Write Crawler Script**
   - Use requests + BeautifulSoup or Playwright
   - Handle pagination logic
   - Add request headers and delays to avoid blocking

4. **Data Processing**
   - Clean and standardize data
   - Output in user-requested format

## Output Format

```json
{
  "source": "Website Name",
  "crawl_time": "2024-01-01T12:00:00Z",
  "total_items": 100,
  "items": [
    {
      "name": "Product Name",
      "price": 99.00,
      "original_price": 199.00,
      "url": "Product URL",
      "image": "Image URL",
      "stock": "In Stock"
    }
  ]
}
```

## Notes

- Respect robots.txt rules
- Add appropriate request intervals (1-3 seconds)
- Set reasonable User-Agent
- Handle anti-scraping mechanisms (CAPTCHA, IP limits)
- Data for personal learning use only
"""
    },
    "monthly-report": {
        "content": """---
name: monthly-report
description: Generate monthly sales reports by extracting sales data and outputting to Excel format. Use this skill for sales statistics, data summaries, or business reports.
usage_count: 0
last_used_at: null
---

# Monthly Sales Report Generation

A standardized workflow for generating structured monthly sales reports from data sources.

## Use Cases

- Generate monthly sales reports
- Export data to Excel
- Perform sales data analysis
- Create business summary reports

## Workflow

1. **Confirm Report Requirements**
   - Ask for time range (month)
   - Confirm data source (database/API/file)
   - Confirm required metrics and dimensions

2. **Data Extraction**
   ```sql
   SELECT 
       DATE_FORMAT(order_date, '%Y-%m') as month,
       product_category,
       COUNT(*) as order_count,
       SUM(amount) as total_sales,
       AVG(amount) as avg_order_value
   FROM orders
   WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31'
   GROUP BY month, product_category
   ORDER BY total_sales DESC
   ```

3. **Data Processing**
   - Calculate month-over-month / year-over-year growth
   - Generate summary statistics
   - Prepare visualization data

4. **Generate Report**
   ```python
   import pandas as pd
   
   with pd.ExcelWriter('monthly_report.xlsx') as writer:
       summary_df.to_excel(writer, sheet_name='Summary')
       detail_df.to_excel(writer, sheet_name='Details')
       chart_df.to_excel(writer, sheet_name='Chart Data')
   ```

## Report Structure

```
Monthly_Sales_Report.xlsx
â”œâ”€â”€ Summary
â”‚   â”œâ”€â”€ Total Sales
â”‚   â”œâ”€â”€ Order Count
â”‚   â”œâ”€â”€ Average Order Value
â”‚   â””â”€â”€ Month-over-Month Growth
â”œâ”€â”€ By Category
â”‚   â””â”€â”€ Sales by Category
â”œâ”€â”€ Trend Analysis
â”‚   â””â”€â”€ Daily/Weekly Sales Trend
â””â”€â”€ Details
    â””â”€â”€ Raw Order Data
```

## Key Metrics

| Metric | Calculation |
|--------|-------------|
| Total Sales | SUM(order_amount) |
| Order Count | COUNT(orders) |
| AOV | Total Sales / Order Count |
| MoM Growth | (This Month - Last Month) / Last Month Ã— 100% |
| YoY Growth | (This Month - Same Month Last Year) / Same Month Last Year Ã— 100% |

## Notes

- Verify data source connection permissions
- Process large datasets in batches
- Keep amounts to two decimal places
- Handle timezone consistently
- Anonymize sensitive data
"""
    },
    "webapp-testing": {
        "content": """---
name: webapp-testing
description: Perform end-to-end UI testing on web applications using Playwright. Use this skill for automated website testing, user flow verification, or page element checks.
usage_count: 0
last_used_at: null
---

# Web Application UI Testing

A standardized workflow for automated UI testing using Playwright.

## Use Cases

- Automated UI testing for web applications
- Verify website functionality
- Validate user interaction flows
- Perform regression testing

## Workflow

1. **Confirm Test Scope**
   - Ask for target URL
   - Confirm features to test
   - Understand expected behavior

2. **Design Test Cases**
   - List key user flows
   - Define assertion conditions
   - Consider edge cases

3. **Write Test Script**
   ```python
   from playwright.sync_api import sync_playwright
   
   def test_login_flow():
       with sync_playwright() as p:
           browser = p.chromium.launch()
           page = browser.new_page()
           page.goto("https://example.com")
           
           page.fill("#username", "test_user")
           page.fill("#password", "test_pass")
           page.click("button[type='submit']")
           
           assert page.url == "https://example.com/dashboard"
           
           browser.close()
   ```

4. **Execute and Report**
   - Run tests
   - Capture screenshots at key steps
   - Generate test report

## Common Operations

| Operation | Code |
|-----------|------|
| Navigate | `page.goto(url)` |
| Click | `page.click(selector)` |
| Input | `page.fill(selector, text)` |
| Wait | `page.wait_for_selector(selector)` |
| Screenshot | `page.screenshot(path="screenshot.png")` |
| Assert | `assert page.locator(selector).is_visible()` |

## Output Format

```
Test Report
===========
Target: https://example.com
Time: 2024-01-01 12:00:00
Result: PASS âœ… / FAIL â�Œ

Test Cases:
1. [âœ…] Login Flow - User can log in successfully
2. [âœ…] Navigation Test - All links accessible
3. [â�Œ] Form Submit - Missing required field prompt

Failure Details:
- Form Submit: Expected error message, got no feedback
- Screenshot: ./screenshots/form_error.png
```

## Notes

- Use headless mode for faster execution
- Add appropriate waits to avoid element not loaded errors
- Isolate test environment from production
- Use environment variables for sensitive data
"""
    }
}


def init_skills():
    """
    Initialize skills directory and sample skills.
    Skip if directory already exists with skill files.
    """
    # Check if skills already exist
    if SKILLS_DIR.exists():
        existing_skills = [d for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
        if existing_skills:
            print(f"âœ… Skills directory already exists with {len(existing_skills)} skills, skipping initialization")
            return
    
    # Create skills directory
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create each sample skill
    for skill_name, skill_data in INITIAL_SKILLS.items():
        skill_dir = SKILLS_DIR / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_data["content"], encoding="utf-8")
        print(f"   ğŸ“¦ Created skill: {skill_name}")
    
    print(f"âœ… Initialized {len(INITIAL_SKILLS)} sample skills")


# Execute initialization
init_skills()


import re
import yaml
from pathlib import Path
from dataclasses import dataclass

# SKILLS_DIR is already defined in cell 19 with environment-aware path
# On Kaggle: /kaggle/working/skills
# Locally: ./skills


@dataclass
class SkillMetadata:
    """Skill metadata"""
    name: str
    description: str
    path: Path
    
    
def parse_skill_md(skill_path: Path) -> SkillMetadata | None:
    """
    Parse SKILL.md file and extract frontmatter metadata
    
    Args:
        skill_path: Path to SKILL.md file
        
    Returns:
        SkillMetadata or None
    """
    if not skill_path.exists():
        return None
        
    content = skill_path.read_text(encoding="utf-8")
    
    # Parse YAML frontmatter (section wrapped by ---)
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        return None
    
    try:
        metadata = yaml.safe_load(frontmatter_match.group(1))
        return SkillMetadata(
            name=metadata.get("name", ""),
            description=metadata.get("description", ""),
            path=skill_path.parent,
        )
    except yaml.YAMLError:
        return None


def load_all_skills() -> list[SkillMetadata]:
    """
    Load metadata for all skills
    
    Returns:
        List of skill metadata
    """
    skills = []
    
    if not SKILLS_DIR.exists():
        return skills
    
    # Iterate through all subdirectories
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
            
        skill_md = skill_dir / "SKILL.md"
        metadata = parse_skill_md(skill_md)
        
        if metadata:
            skills.append(metadata)
    
    return skills


# Load and display all skills
all_skills = load_all_skills()
print(f"âœ… Loaded {len(all_skills)} skills from {SKILLS_DIR}:")
for skill in all_skills:
    print(f"   ğŸ“¦ {skill.name}: {skill.description[:50]}...")



import json


def list_skills() -> str:
    """
    List all available skills
    
    Returns:
        Skill list (JSON format)
    """
    skills = load_all_skills()
    
    if not skills:
        return json.dumps({"skills": [], "message": "No skills available"}, ensure_ascii=False)
    
    result = [{
        "name": s.name,
        "description": s.description,
    } for s in skills]
    
    return json.dumps({
        "skills": result,
        "total": len(result)
    }, ensure_ascii=False, indent=2)


print("âœ… list_skills tool defined")


def get_skill_content(skill_name: str) -> str:
    """
    Get the complete content of a specified skill
    
    Args:
        skill_name: Skill name
        
    Returns:
        Complete SKILL.md content
    """
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    
    if not skill_path.exists():
        return json.dumps({"error": f"Skill '{skill_name}' does not exist"}, ensure_ascii=False)
    
    content = skill_path.read_text(encoding="utf-8")
    
    return json.dumps({
        "skill_name": skill_name,
        "content": content,
    }, ensure_ascii=False, indent=2)


def build_skills_prompt() -> str:
    """
    Build skill metadata prompt fragment for injection into agent instruction
    
    Returns:
        Formatted skill list string
    """
    skills = load_all_skills()
    
    if not skills:
        return "No skills available."
    
    lines = []
    for s in skills:
        lines.append(f"- **{s.name}**: {s.description}")
    
    return "\n".join(lines)


print("âœ… get_skill_content and build_skills_prompt defined")


# Test list_skills
print("ğŸ“‹ Test list_skills():")
print(list_skills())
print()

# Test build_skills_prompt (will be injected into agent instruction)
print("ğŸ“� Test build_skills_prompt():")
print(build_skills_prompt())
print()

# Test get_skill_content
print("ğŸ“„ Test get_skill_content('ecommerce-crawler'):")
result = json.loads(get_skill_content("ecommerce-crawler"))
print(f"Skill: {result['skill_name']}")
print(f"Content preview: {result['content'][:200]}...")



# Dynamically build instruction containing skill metadata
skills_prompt = build_skills_prompt()

AGENT_INSTRUCTION = f"""You are Skill Gardener, an intelligent assistant with skill management capabilities.

## Available Skills

The following skills are available. Match them to user requests:

{skills_prompt}

## Workflow

1. User describes a task â†’ Check if it matches an existing skill from the list above
2. If matched â†’ Call `get_skill_content(skill_name)` to get detailed instructions
3. Follow skill instructions to complete the task, interact with user to confirm details
4. If no match â†’ Help user complete the task directly, or suggest creating a new skill

## Response Requirements

- Be concise and clear
- If using a skill, mention which skill is being used
"""

# Create Agent
skill_aware_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="SkillGardener",
    instruction=AGENT_INSTRUCTION,
    tools=[
        list_skills,
        get_skill_content,
        preload_memory,
    ],
)

print("âœ… SkillGardener Agent created")
print(f"ğŸ“¦ Injected {len(load_all_skills())} skill metadata into prompt")


# Create Runner
skill_runner = Runner(
    agent=skill_aware_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… Runner created")


# Scenario 1: Query available skills
print("=" * 50)
print("ğŸ�¬ Scenario 1: Query available skills")
print("=" * 50)

await run_session(
    skill_runner,
    "What skills do I have available?",
    "skill-demo-01"
)


# Scenario 2: Match a skill
print("=" * 50)
print("ğŸ�¬ Scenario 2: Match a skill for report generation")
print("=" * 50)

await run_session(
    skill_runner,
    "Help me generate a November 2023 monthly sales report. We're in mock stage, auto-fill data, choose a reasonable approach, I confirm.",
    "skill-demo-02"
)


# Scenario 3: Another skill matching scenario
print("=" * 50)
print("ğŸ�¬ Scenario 3: Match a skill for web testing")
print("=" * 50)

await run_session(
    skill_runner,
    "I need to do automated testing on www.google.com to verify if the search function works properly",
    "skill-demo-03"
)


def create_skill(name: str, description: str, instructions: str) -> str:
    """
    Create a new skill
    
    Args:
        name: Skill name (lowercase, hyphen-separated, e.g., 'data-analysis')
        description: Skill description, explaining when to use this skill
        instructions: Detailed instructions including workflow, examples, notes, etc.
        
    Returns:
        Creation result (JSON format)
    """
    # Validate skill name format
    if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name):
        return json.dumps({
            "success": False,
            "error": "Invalid skill name format. Should be lowercase letters and numbers, separated by hyphens (e.g., 'data-analysis')"
        }, ensure_ascii=False)
    
    # Check if skill already exists
    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        return json.dumps({
            "success": False,
            "error": f"Skill '{name}' already exists"
        }, ensure_ascii=False)
    
    # Create skill directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate SKILL.md content
    skill_content = f"""---
name: {name}
description: {description}
---

{instructions}
"""
    
    # Write file
    skill_md_path = skill_dir / "SKILL.md"
    skill_md_path.write_text(skill_content, encoding="utf-8")
    
    return json.dumps({
        "success": True,
        "message": f"Skill '{name}' created successfully",
        "path": str(skill_md_path)
    }, ensure_ascii=False)


print("âœ… create_skill tool defined")


# Test creating a new skill
test_result = create_skill(
    name="code-review",
    description="Review code for quality, potential issues, and improvement suggestions. Use this skill when user needs code review or optimization advice.",
    instructions="""# Code Review

Conduct a comprehensive review of submitted code.

## Review Dimensions

1. **Code Quality** - Naming conventions, code structure, readability
2. **Potential Issues** - Bugs, security vulnerabilities, performance issues
3. **Best Practices** - Design patterns, SOLID principles
4. **Improvement Suggestions** - Refactoring directions, optimization advice

## Output Format

```
## Review Results

### ğŸ‘� Strengths
- ...

### âš ï¸� Issues
- ...

### ğŸ’¡ Suggestions
- ...
```
"""
)

print("ğŸ“� Test create_skill:")
print(test_result)

# Verify skill creation
print("\nğŸ“‚ Current skill list:")
for skill in load_all_skills():
    print(f"   ğŸ“¦ {skill.name}")


# Rebuild instruction with skill creation capability
skills_prompt_v2 = build_skills_prompt()

AGENT_INSTRUCTION_V2 = f"""You are Skill Gardener, an intelligent assistant with skill management capabilities.

## Available Skills

The following skills are available. Match them to user requests:

{skills_prompt_v2}

## Core Capabilities

1. **Skill Routing**: Match user requests to existing skills
2. **Skill Execution**: Call `get_skill_content` to get detailed instructions and execute
3. **Skill Creation**: Call `create_skill` to create new skills when user requests

## Workflow

### Execute Task
1. User describes task â†’ Check if it matches an existing skill
2. If matched â†’ Call `get_skill_content(skill_name)` to get instructions
3. Follow instructions to execute the task

### Create Skill
When user says "save as skill" or "create skill":
1. Summarize the task just completed
2. Extract skill name (lowercase, hyphen-separated), description, detailed instructions
3. Call `create_skill(name, description, instructions)` to create the skill

## Response Requirements

- Be concise and clear
- If using a skill, mention which skill is being used
- When creating skill, confirm content before creating
"""

# Create new Agent
skill_gardener_v2 = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="SkillGardenerV2",
    instruction=AGENT_INSTRUCTION_V2,
    tools=[
        list_skills,
        get_skill_content,
        create_skill,  # New: skill creation tool
        preload_memory,
    ],
)

# Create new Runner
skill_runner_v2 = Runner(
    agent=skill_gardener_v2,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… SkillGardener V2 created (with skill creation)")
print(f"ğŸ“¦ Currently have {len(load_all_skills())} skills")


# Scenario 4: Let Agent create a new skill
print("=" * 50)
print("ğŸ�¬ Scenario 4: Create new skill")
print("=" * 50)

await run_session(
    skill_runner_v2,
    [
        "Help me create a 'log-analysis' skill for analyzing application logs to find errors and anomaly patterns. Don't ask me questions, use a reasonable approach and generate directly.",
    ],
    "create-skill"
)


# Verify skill creation
print("ğŸ“‚ Skill list after creation:")
for skill in load_all_skills():
    print(f"   ğŸ“¦ {skill.name}: {skill.description[:40]}...")


import shutil
from typing import Optional


def delete_skill(skill_name: str) -> str:
    """
    Delete a specified skill
    
    Args:
        skill_name: Name of skill to delete
        
    Returns:
        Deletion result (JSON format)
    """
    skill_dir = SKILLS_DIR / skill_name
    
    if not skill_dir.exists():
        return json.dumps({
            "success": False,
            "error": f"Skill '{skill_name}' does not exist"
        }, ensure_ascii=False)
    
    # Delete entire skill directory
    shutil.rmtree(skill_dir)
    
    return json.dumps({
        "success": True,
        "message": f"Skill '{skill_name}' deleted"
    }, ensure_ascii=False)


# Design note: Centralize YAML/frontmatter updates and error checking at the file I/O layer
# to ensure skill files remain parseable and rollback-safe.
def update_skill(skill_name: str, description: Optional[str] = None, instructions: Optional[str] = None) -> str:
    """
    Update a specified skill's content
    
    Args:
        skill_name: Name of skill to update
        description: New skill description (optional, keeps original if not provided)
        instructions: New skill instructions (optional, keeps original if not provided)
        
    Returns:
        Update result (JSON format)
    """
    skill_md_path = SKILLS_DIR / skill_name / "SKILL.md"
    
    if not skill_md_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Skill '{skill_name}' does not exist"
        }, ensure_ascii=False)
    
    # Read existing content
    content = skill_md_path.read_text(encoding="utf-8")
    
    # Parse frontmatter
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not frontmatter_match:
        return json.dumps({
            "success": False,
            "error": "SKILL.md format error"
        }, ensure_ascii=False)
    
    # Parse existing metadata
    try:
        metadata = yaml.safe_load(frontmatter_match.group(1))
    except yaml.YAMLError:
        return json.dumps({
            "success": False,
            "error": "YAML parsing error"
        }, ensure_ascii=False)
    
    existing_instructions = frontmatter_match.group(2).strip()
    
    # Update fields
    if description is not None:
        metadata["description"] = description
    if instructions is not None:
        existing_instructions = instructions
    
    # Regenerate SKILL.md
    new_content = f"""---
name: {metadata.get('name', skill_name)}
description: {metadata.get('description', '')}
---

{existing_instructions}
"""
    
    skill_md_path.write_text(new_content, encoding="utf-8")
    
    return json.dumps({
        "success": True,
        "message": f"Skill '{skill_name}' updated",
        "updated_fields": {
            "description": description is not None,
            "instructions": instructions is not None
        }
    }, ensure_ascii=False)


print("âœ… delete_skill and update_skill tools defined")


# Current skill list
print("ğŸ“‚ Current skill list:")
for skill in load_all_skills():
    print(f"   ğŸ“¦ {skill.name}")

# Test update_skill - Update code-review skill description
print("\nğŸ”„ Test update_skill (update description):")
result = update_skill(
    "code-review",
    description="Comprehensive code review including code quality, security vulnerabilities, performance issues, and best practices. Suitable for PR reviews and code optimization suggestions."
)
print(result)

# Verify update result
print("\nğŸ“„ Updated code-review description:")
for skill in load_all_skills():
    if skill.name == "code-review":
        print(f"   {skill.description}")


# Test delete_skill - Delete code-review skill (for testing, optional to run)
# Note: This cell will delete the skill. Skip if you don't want to delete.

print("ğŸ—‘ï¸� Test delete_skill:")
result = delete_skill("code-review")
print(result)

# Verify deletion result
print("\nğŸ“‚ Skill list after deletion:")
for skill in load_all_skills():
    print(f"   ğŸ“¦ {skill.name}")


# Rebuild instruction with full CRUD capability
skills_prompt_v3 = build_skills_prompt()

AGENT_INSTRUCTION_V3 = f"""You are Skill Gardener, an intelligent assistant with complete skill management capabilities.

## Available Skills

{skills_prompt_v3}

## Core Capabilities

1. **Skill Routing**: Match user requests to existing skills
2. **Skill Execution**: Call `get_skill_content` to get instructions and execute
3. **Skill Creation**: Call `create_skill` to create new skills
4. **Skill Update**: Call `update_skill` to update skill content
5. **Skill Deletion**: Call `delete_skill` to delete skills

## Tool Usage Guide

### Create Skill
```
create_skill(name, description, instructions)
- name: lowercase, hyphen-separated (e.g., 'data-analysis')
- description: explain when to use this skill
- instructions: detailed execution instructions
```

### Update Skill
```
update_skill(skill_name, description=None, instructions=None)
- only pass fields that need updating
```

### Delete Skill
```
delete_skill(skill_name)
- confirm user intent before deleting
```

## Response Requirements

- Confirm before executing dangerous operations (deletion)
- Report results after operations complete
"""

# Create full-featured Agent
skill_gardener_v3 = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="SkillGardenerV3",
    instruction=AGENT_INSTRUCTION_V3,
    tools=[
        list_skills,
        get_skill_content,
        create_skill,
        update_skill,
        delete_skill,
        preload_memory,
    ],
)

# Create Runner
skill_runner_v3 = Runner(
    agent=skill_gardener_v3,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… SkillGardener V3 created (full CRUD)")
print(f"ğŸ“¦ Current skill count: {len(load_all_skills())}")


# Scenario 5: Update skill
print("=" * 50)
print("ğŸ�¬ Scenario 5: Update skill description")
print("=" * 50)

await run_session(
    skill_runner_v3,
    "Please update the log-analysis skill description to add 'supports multiple log formats'",
    "manage-skill-demo-01"
)


# Scenario 6: Delete skill (optional to run)
print("=" * 50)
print("ğŸ�¬ Scenario 6: Delete skill")
print("=" * 50)

await run_session(
    skill_runner_v3,
    "Delete the log-analysis skill, I have confirmed, execute immediately",
    "manage-skill-demo-02"
)


# Verify final skill list
print("ğŸ“‚ Final skill list:")
for skill in load_all_skills():
    print(f"   ğŸ“¦ {skill.name}: {skill.description[:40]}...")


from datetime import datetime
from dataclasses import field

# Capacity configuration
MAX_SKILLS = 10  # Maximum number of skills


@dataclass
class SkillMetadataV2:
    """Extended skill metadata with usage statistics"""
    name: str
    description: str
    path: Path
    usage_count: int = 0
    last_used_at: str = ""  # ISO format timestamp


def parse_skill_md_v2(skill_path: Path) -> SkillMetadataV2 | None:
    """
    Parse SKILL.md file, extract frontmatter metadata (with usage statistics)
    """
    if not skill_path.exists():
        return None
        
    content = skill_path.read_text(encoding="utf-8")
    
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        return None
    
    try:
        metadata = yaml.safe_load(frontmatter_match.group(1))
        return SkillMetadataV2(
            name=metadata.get("name", ""),
            description=metadata.get("description", ""),
            path=skill_path.parent,
            usage_count=metadata.get("usage_count", 0),
            last_used_at=metadata.get("last_used_at", ""),
        )
    except yaml.YAMLError:
        return None


def load_all_skills_v2() -> list[SkillMetadataV2]:
    """Load extended metadata for all skills"""
    skills = []
    
    if not SKILLS_DIR.exists():
        return skills
    
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
            
        skill_md = skill_dir / "SKILL.md"
        metadata = parse_skill_md_v2(skill_md)
        
        if metadata:
            skills.append(metadata)
    
    return skills


def update_skill_usage(skill_name: str) -> None:
    """
    Update skill usage statistics
    
    Args:
        skill_name: Skill name
    """
    skill_md_path = SKILLS_DIR / skill_name / "SKILL.md"
    
    if not skill_md_path.exists():
        return
    
    content = skill_md_path.read_text(encoding="utf-8")
    
    # Parse frontmatter
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not frontmatter_match:
        return
    
    try:
        metadata = yaml.safe_load(frontmatter_match.group(1))
    except yaml.YAMLError:
        return
    
    instructions = frontmatter_match.group(2).strip()
    
    # Update usage statistics
    metadata["usage_count"] = metadata.get("usage_count", 0) + 1
    metadata["last_used_at"] = datetime.now().isoformat()
    
    # Regenerate SKILL.md
    new_content = f"""---
name: {metadata.get('name', skill_name)}
description: {metadata.get('description', '')}
usage_count: {metadata.get('usage_count', 0)}
last_used_at: {metadata.get('last_used_at', '')}
---

{instructions}
"""
    
    skill_md_path.write_text(new_content, encoding="utf-8")


print("âœ… Skill usage statistics functions defined")


from typing import Optional


def get_lru_skill() -> Optional[str]:
    """
    Get the least recently used skill name (for LRU eviction)
    
    Priority:
    1. Never used skills (usage_count = 0)
    2. Skills with earliest last_used_at
    
    Returns:
        Least used skill name, or None
    """
    skills = load_all_skills_v2()
    
    if not skills:
        return None
    
    # Sort by usage_count and last_used_at
    def sort_key(s: SkillMetadataV2):
        # Prioritize evicting skills with usage_count = 0
        # Then sort by last_used_at (empty string sorts first)
        return (s.usage_count, s.last_used_at or "0000-00-00")
    
    skills.sort(key=sort_key)
    return skills[0].name


def enforce_capacity() -> dict:
    """
    Execute capacity control, LRU eviction if skill count exceeds threshold
    
    Returns:
        Eviction result
    """
    skills = load_all_skills_v2()
    evicted = []
    
    while len(skills) > MAX_SKILLS:
        lru_skill = get_lru_skill()
        if lru_skill:
            delete_skill(lru_skill)
            evicted.append(lru_skill)
            skills = load_all_skills_v2()
        else:
            break
    
    return {
        "evicted": evicted,
        "current_count": len(skills),
        "max_skills": MAX_SKILLS
    }


def get_skill_stats() -> str:
    """
    Get usage statistics for all skills
    
    Returns:
        Skill statistics (JSON format)
    """
    skills = load_all_skills_v2()
    
    # Sort by usage count
    skills.sort(key=lambda s: s.usage_count, reverse=True)
    
    result = []
    for s in skills:
        # Ensure last_used_at is a string
        last_used = s.last_used_at
        if hasattr(last_used, 'isoformat'):
            last_used = last_used.isoformat()
        
        result.append({
            "name": s.name,
            "description": s.description[:50] + "..." if len(s.description) > 50 else s.description,
            "usage_count": s.usage_count,
            "last_used_at": last_used or "Never used",
        })
    
    return json.dumps({
        "skills": result,
        "total": len(result),
        "max_skills": MAX_SKILLS
    }, ensure_ascii=False, indent=2)


print("âœ… LRU eviction and statistics functions defined")


def get_skill_content_v2(skill_name: str) -> str:
    """
    Get skill content (with usage statistics update)
    
    Args:
        skill_name: Skill name
        
    Returns:
        Skill content (JSON format)
    """
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    
    if not skill_path.exists():
        return json.dumps({"error": f"Skill '{skill_name}' does not exist"}, ensure_ascii=False)
    
    # Update usage statistics
    update_skill_usage(skill_name)
    
    # Read content
    content = skill_path.read_text(encoding="utf-8")
    
    return json.dumps({
        "skill_name": skill_name,
        "content": content,
    }, ensure_ascii=False, indent=2)


# Design note: Centralize naming validation and LRU capacity control at the tool layer
# to prevent agent instructions from directly manipulating the file system and losing control of the skills directory.
def create_skill_v2(name: str, description: str, instructions: str) -> str:
    """
    Create new skill (with capacity control)
    
    Args:
        name: Skill name
        description: Skill description
        instructions: Skill instructions
        
    Returns:
        Creation result (JSON format)
    """
    # Validate skill name format
    if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name):
        return json.dumps({
            "success": False,
            "error": "Invalid skill name format. Should be lowercase letters and numbers, separated by hyphens"
        }, ensure_ascii=False)
    
    # Check if skill already exists
    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        return json.dumps({
            "success": False,
            "error": f"Skill '{name}' already exists"
        }, ensure_ascii=False)
    
    # Check capacity, LRU eviction if necessary
    skills_count = len(load_all_skills_v2())
    eviction_result = None
    
    if skills_count >= MAX_SKILLS:
        eviction_result = enforce_capacity()
    
    # Create skill directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate SKILL.md content (with usage statistics fields)
    now = datetime.now().isoformat()
    skill_content = f"""---
name: {name}
description: {description}
usage_count: 0
last_used_at: 
created_at: {now}
---

{instructions}
"""
    
    # Write file
    skill_md_path = skill_dir / "SKILL.md"
    skill_md_path.write_text(skill_content, encoding="utf-8")
    
    result = {
        "success": True,
        "message": f"Skill '{name}' created successfully",
        "path": str(skill_md_path)
    }
    
    if eviction_result and eviction_result["evicted"]:
        result["evicted_skills"] = eviction_result["evicted"]
        result["message"] += f" (evicted: {', '.join(eviction_result['evicted'])})"
    
    return json.dumps(result, ensure_ascii=False)


def monthly_report() -> str:
    """Tool alias: Forward to monthly-report skill"""
    return get_skill_content_v2("monthly-report")


def ecommerce_crawler() -> str:
    """Tool alias: Forward to ecommerce-crawler skill"""
    return get_skill_content_v2("ecommerce-crawler")


def webapp_testing() -> str:
    """Tool alias: Forward to webapp-testing skill"""
    return get_skill_content_v2("webapp-testing")


print("âœ… Enhanced tool functions defined")


# View current statistics
print("ğŸ“Š Current skill usage statistics:")
print(get_skill_stats())

# Simulate using some skills
print("\nğŸ”„ Simulating skill usage...")
get_skill_content_v2("ecommerce-crawler")
get_skill_content_v2("ecommerce-crawler")
get_skill_content_v2("webapp-testing")

# View statistics again
print("\nğŸ“Š Statistics after usage:")
print(get_skill_stats())


# Build final Agent
def build_skills_prompt_v2() -> str:
    """Build skill list with statistics info"""
    skills = load_all_skills_v2()
    
    if not skills:
        return "No skills available."
    
    # Sort by usage count, frequently used ones first
    skills.sort(key=lambda s: s.usage_count, reverse=True)
    
    lines = []
    for s in skills:
        usage_info = f"(used {s.usage_count} times)" if s.usage_count > 0 else "(unused)"
        lines.append(f"- **{s.name}** {usage_info}: {s.description}")
    
    return "\n".join(lines)


skills_prompt_final = build_skills_prompt_v2()

AGENT_INSTRUCTION_FINAL = f"""You are Skill Gardener, an intelligent assistant with complete skill management capabilities.

## Available Skills (sorted by usage frequency)

{skills_prompt_final}

## Core Capabilities

1. **Skill Routing**: Match user requests to existing skills
2. **Skill Execution**: Call `get_skill_content_v2` to get instructions (auto-records usage)
3. **Skill Creation**: Call `create_skill_v2` to create new skills (auto capacity control)
4. **Skill Statistics**: Call `get_skill_stats` to view usage statistics
5. **Skill Management**: Update/delete skills

## Capacity Control

- Maximum skills: {MAX_SKILLS}
- Auto LRU eviction when exceeded (least used skills)

## Response Requirements

- Confirm before executing dangerous operations
- Report operation results
"""

# Create final Agent
skill_gardener_final = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="SkillGardenerFinal",
    instruction=AGENT_INSTRUCTION_FINAL,
    tools=[
        list_skills,
        get_skill_content_v2,
        create_skill_v2,
        update_skill,
        delete_skill,
        get_skill_stats,
        preload_memory,
        monthly_report,
        ecommerce_crawler,
        webapp_testing,
    ],
)

# Create Runner
skill_runner_final = Runner(
    agent=skill_gardener_final,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… SkillGardener Final created")
print(f"ğŸ“¦ Current skill count: {len(load_all_skills_v2())}/{MAX_SKILLS}")


# Scenario 7: View skill usage statistics
print("=" * 50)
print("ğŸ�¬ Scenario 7: View skill usage statistics")
print("=" * 50)

await run_session(
    skill_runner_final,
    "Show skill usage statistics",
    "stats-demo"
)


from google.adk.agents.callback_context import CallbackContext


# Design note: Re-fetch session after conversation ends before writing to Memory
# to ensure we save the complete event list, not a mid-conversation snapshot.
async def run_session_with_memory(
    runner_instance: Runner,
    user_queries: list[str] | str,
    session_id: str = "default",
    save_to_memory: bool = True
):
    """
    Run conversation session (with Memory support)
    
    Args:
        runner_instance: Runner instance
        user_queries: User input
        session_id: Session ID
        save_to_memory: Whether to save to Memory after session ends
    """
    print(f"\n### Session: {session_id}")

    # Create or get session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        print(f"  ğŸ”§ Tool called: {part.function_call.name}")
                    if hasattr(part, 'function_response') and part.function_response:
                        print(f"  âœ… Tool returned: {part.function_response.name}")
            
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text and part.text != "None":
                        print(f"\nAgent > {part.text}")

    # After session ends, re-fetch session with complete conversation history
    if save_to_memory:
        # Important: Must re-fetch session to include complete conversation history
        updated_session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session.id
        )
        
        # Save to Memory
        await memory_service.add_session_to_memory(updated_session)
        print(f"\n  ğŸ’¾ Session saved to Memory (events: {len(updated_session.events) if hasattr(updated_session, 'events') and updated_session.events else 0})")
        
        return updated_session
    
    return session


print("âœ… run_session_with_memory function defined (fixed version)")


from google.adk.tools import ToolContext


# Design note: Wrap Memory search with result trimming / fallback query to reduce token consumption
# and make it easier for Agent to consume structured results.
async def search_memory_tool(query: str, tool_context: ToolContext) -> str:
    """
    Search relevant memories in Memory (directly using memory_service.search_memory)
    
    Args:
        query: Search query (try to include English keywords that overlap with historical conversations)
        tool_context: Tool context (retained for ADK signature compatibility, not directly used currently)
        
    Returns:
        Search results (JSON format)
    """
    # Step 1: Call underlying MemoryService with original query
    response = await memory_service.search_memory(
        app_name=APP_NAME,
        user_id=USER_ID,
        query=query,
    )

    memories_raw = response.memories if hasattr(response, "memories") and response.memories else []

    # Fallback search if no results and query contains 'skills', adapting to this notebook's demo scenarios
    if not memories_raw and "skills" in query.lower():
        fallback_response = await memory_service.search_memory(
            app_name=APP_NAME,
            user_id=USER_ID,
            query="List all available skills",
        )
        if hasattr(fallback_response, "memories") and fallback_response.memories:
            memories_raw = fallback_response.memories

    if not memories_raw:
        return json.dumps({
            "found": False,
            "message": "No relevant memories found",
        }, ensure_ascii=False)

    # Format results, extract only first few text segments to avoid excessive output
    memories = []
    for m in memories_raw:
        content_text = ""
        if getattr(m, "content", None) and getattr(m.content, "parts", None):
            for part in m.content.parts[:3]:
                if hasattr(part, "text") and part.text:
                    content_text += part.text[:200] + "... "
        if content_text:
            author = getattr(m, "author", "") or ""
            if author:
                memories.append({"author": author, "content": content_text.strip()})
            else:
                memories.append({"content": content_text.strip()})

    if not memories:
        return json.dumps({
            "found": False,
            "message": "Memory content is empty",
        }, ensure_ascii=False)

    return json.dumps({
        "found": True,
        "count": len(memories),
        "memories": memories,
    }, ensure_ascii=False, indent=2)


print("âœ… search_memory_tool defined (using memory_service.search_memory directly)")


# Build Agent with Memory capability
skills_prompt_memory = build_skills_prompt_v2()

AGENT_INSTRUCTION_MEMORY = f"""You are Skill Gardener, an intelligent assistant with complete skill management and memory capabilities.

## Available Skills (sorted by usage frequency)

{skills_prompt_memory}

## Important Note

**You cannot directly execute skills!** Skills are just guidance documents.

When user requests a task:
1. Determine if it matches an existing skill
2. If matched, call `get_skill_content_v2(skill_name)` to get skill instructions
3. **Tell the user the execution steps in natural language based on the instructions**
4. You cannot directly call skill names like `ecommerce_crawler` as tools!

## Available Tools

### Skill Management Tools
- `list_skills` - List all skills
- `get_skill_content_v2` - Get skill content (auto-records usage)
- `create_skill_v2` - Create new skill (auto capacity control)
- `update_skill` - Update skill
- `delete_skill` - Delete skill
- `get_skill_stats` - View usage statistics

### Memory Tools
- `load_memory` - Load historical memories (ADK built-in, recommended)
- `search_memory_tool` - Search historical memories

## Workflow

1. **At new session start**: If user asks about history-related questions, call `load_memory` to load memories
2. **When executing tasks**: Combine Skill instructions with Memory context
3. **After session ends**: Auto-save to Memory (handled by system)

## Response Requirements

- If there are relevant historical memories, proactively mention them
- Confirm before executing dangerous operations
"""

# Create Agent with Memory
skill_gardener_memory = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="SkillGardenerMemory",
    instruction=AGENT_INSTRUCTION_MEMORY,
    tools=[
        list_skills,
        get_skill_content_v2,
        create_skill_v2,
        update_skill,
        delete_skill,
        get_skill_stats,
        search_memory_tool,  # Custom search tool
        load_memory,         # ADK built-in load tool
    ],
)

# Create Runner
skill_runner_memory = Runner(
    agent=skill_gardener_memory,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… SkillGardener Memory version created")
print(f"ğŸ“¦ Current skill count: {len(load_all_skills_v2())}")


# Debug: Check Memory service status
print("ğŸ”� Debug: Checking Memory service status")
print(f"   Memory Service type: {type(memory_service)}")

# Manual Memory search
async def debug_memory_search(query: str):
    """Debug: Direct Memory search"""
    response = await memory_service.search_memory(
        app_name=APP_NAME,
        user_id=USER_ID,
        query=query
    )
    print(f"\n   Search '{query}':")
    print(f"      Response type: {type(response)}")
    
    # Check response attributes
    if hasattr(response, 'memories'):
        memories = response.memories
        print(f"      memories count: {len(memories) if memories else 0}")
        if memories:
            for i, mem in enumerate(memories[:2]):
                print(f"      Memory {i+1}: {type(mem)}")
                # Try to print memory attributes
                if hasattr(mem, '__dict__'):
                    print(f"         Attributes: {list(mem.__dict__.keys())[:5]}")
    else:
        print(f"      Response attributes: {dir(response)[:10]}")
    
    return response

# Execute debug searches
r1 = await debug_memory_search("report")
r2 = await debug_memory_search("sales")
r3 = await debug_memory_search("Excel")


# Scenario 8: Simple task and save to Memory
print("=" * 50)
print("ğŸ�¬ Scenario 8: Simple task and save to Memory")
print("=" * 50)

# Use a simple, clear task
session1 = await run_session_with_memory(
    skill_runner_memory,
    [
        "List all available skills",
    ],
    "simple-task-001",
    save_to_memory=True
)

# Print session info for verification
print(f"\nğŸ“‹ Session info:")
print(f"   ID: {session1.id}")
print(f"   Events count: {len(session1.events) if hasattr(session1, 'events') and session1.events else 'N/A'}")


# Scenario 8.1: Debug Memory content (manual search)
print("=" * 50)
print("ğŸ”� Scenario 8.1: Manually check if Scenario 8 was saved to Memory")
print("=" * 50)

debug_response = await memory_service.search_memory(
    app_name=APP_NAME,
    user_id=USER_ID,
    query="List all available skills",
)
if hasattr(debug_response, "memories") and debug_response.memories:
    print(f"   âœ… Found {len(debug_response.memories)} memories")
else:
    print("   â�Œ No memories found (please confirm Scenario 8 was run with English query)")


# Scenario 9: Load historical memories in new session
print("=" * 50)
print("ğŸ�¬ Scenario 9: Load historical memories in new session")
print("=" * 50)

# First manually verify Memory has content (using same English keywords as Scenario 8)
print("ğŸ“‹ First verify Memory status:")
debug_response = await memory_service.search_memory(
    app_name=APP_NAME,
    user_id=USER_ID,
    query="List all available skills",
)
if hasattr(debug_response, 'memories') and debug_response.memories:
    print(f"   âœ… Found {len(debug_response.memories)} memories")
else:
    print("   â�Œ No memories found")

# Second session: Use load_memory to load history
session2 = await run_session_with_memory(
    skill_runner_memory,
    [
        "Use your memory to recall my previous request about skills. Search your memory with keyword: 'List all available skills'.",
    ],
    "memory-load-002",
    save_to_memory=False,
)


# Scenario 10: Use search_memory_tool to search specific content
print("=" * 50)
print("ğŸ�¬ Scenario 10: Search specific historical memories")
print("=" * 50)

session3 = await run_session_with_memory(
    skill_runner_memory,
    [
        "Search my previous questions about skills",
    ],
    "memory-search-003",
    save_to_memory=False
)


import logging
import time
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class Telemetry:
    """Telemetry class for tracking tool usage and session behavior"""
    def __init__(self):
        self.tool_stats = defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
        self.session_events = defaultdict(list)

    def start_session(self, session_id: str) -> None:
        self.session_events[session_id].append({"event": "session_start", "ts": time.time()})

    def end_session(self, session_id: str) -> None:
        self.session_events[session_id].append({"event": "session_end", "ts": time.time()})

    def log_tool(self, session_id: str, name: str, latency_ms: float, status: str = "ok") -> None:
        stat = self.tool_stats[name]
        stat["calls"] += 1
        stat["total_ms"] += latency_ms
        self.session_events[session_id].append(
            {"event": "tool", "name": name, "latency_ms": latency_ms, "status": status, "ts": time.time()}
        )

    def summary(self):
        rows = []
        for name, stat in sorted(self.tool_stats.items()):
            calls = stat["calls"]
            avg_ms = stat["total_ms"] / calls if calls else 0.0
            rows.append({"tool": name, "calls": calls, "avg_ms": round(avg_ms, 1), "total_ms": round(stat["total_ms"], 1)})
        return rows

    def print_summary(self) -> None:
        rows = self.summary()
        if not rows:
            print("No tool usage recorded yet.")
            return
        print("ğŸ”� Tool usage summary:")
        for row in rows:
            print(f"- {row['tool']}: {row['calls']} calls, avg {row['avg_ms']} ms, total {row['total_ms']} ms")

telemetry = Telemetry()
print("âœ… Observability telemetry ready")


# Design note: Add a telemetry collection layer outside the original Runner flow
# to minimally invasively track tool latency and session behavior for performance analysis.
async def run_session_observed(runner_instance: Runner, user_queries, session_id: str = "observed"):
    """Session run with observability: structured logging + tool latency statistics"""
    telemetry.start_session(session_id)
    logging.info("session_start id=%s", session_id)

    try:
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    except Exception:
        session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query in user_queries:
        logging.info("user_query session=%s text=%s", session_id, query)
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        active_calls = {}
        async for event in runner_instance.run_async(user_id=USER_ID, session_id=session.id, new_message=query_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        func_name = part.function_call.name
                        active_calls[func_name] = time.time()
                        logging.info("tool_call session=%s tool=%s", session_id, func_name)
                    if hasattr(part, "function_response") and part.function_response:
                        func_name = part.function_response.name
                        started = active_calls.pop(func_name, time.time())
                        latency_ms = (time.time() - started) * 1000
                        telemetry.log_tool(session_id, func_name, latency_ms)
                        logging.info("tool_response session=%s tool=%s latency_ms=%.1f", session_id, func_name, latency_ms)
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text and part.text != "None":
                        logging.info("agent_response session=%s text=%s", session_id, part.text.strip())

    telemetry.end_session(session_id)
    logging.info("session_end id=%s", session_id)
    telemetry.print_summary()
    return telemetry.summary()

print("âœ… run_session_observed ready (logging + metrics)")


# Example: Run observed conversation with final Runner (requires available API Key)
# Use a prompt that will trigger tool usage for better observability demo
await run_session_observed(
    skill_runner_final, 
    "Show me all available skills and their usage statistics", 
    session_id="obs-demo-1"
)
telemetry.print_summary()


# Evaluation cases (adjust based on skill directory)
eval_cases = [
    {
        "id": "monthly-report",
        "prompt": "Help me output a monthly product report outline covering highlights and risks",
        "expect": "monthly-report",
    },
    {
        "id": "webapp-testing",
        "prompt": "I have a frontend webpage and need an end-to-end test plan including test case groups and tools",
        "expect": "webapp-testing",
    },
    {
        "id": "ecommerce-crawler",
        "prompt": "Need to scrape e-commerce product info, give me crawler steps and precautions",
        "expect": "ecommerce-crawler",
    },
]


import asyncio

# Configurable: Delay between each case (seconds), to avoid 429 rate limiting
EVAL_SLEEP_SECONDS = 2.0

# Design note: Auto-check skill routing behavior through fixed cases instead of manually reviewing logs
# for quick presentation of evaluation results within the Notebook.
async def evaluate_skill_routing(runner_instance: Runner, cases):
    """Evaluate skill routing accuracy"""
    results = []
    for idx, case in enumerate(cases):
        session_id = f"eval-{case['id']}"
        used_skills = set()
        final_texts: list[str] = []

        if idx > 0 and EVAL_SLEEP_SECONDS > 0:
            await asyncio.sleep(EVAL_SLEEP_SECONDS)

        try:
            session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        except Exception:
            session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

        query_content = types.Content(role="user", parts=[types.Part(text=case["prompt"])])
        async for event in runner_instance.run_async(user_id=USER_ID, session_id=session.id, new_message=query_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "args", None):
                        if isinstance(fc.args, dict):
                            skill_arg = fc.args.get("skill_name") or fc.args.get("name")
                            if skill_arg:
                                used_skills.add(str(skill_arg))
                    fr = getattr(part, "function_response", None)
                    if fr and getattr(fr, "response", None):
                        resp = fr.response
                        if isinstance(resp, dict):
                            skill_arg = resp.get("skill_name") or resp.get("name")
                            if skill_arg:
                                used_skills.add(str(skill_arg))
                    txt = getattr(part, "text", None)
                    if txt:
                        final_texts.append(txt)
        final_text = " ".join(final_texts)
        expected = case["expect"]
        hit = any(expected in s for s in used_skills) or (expected in final_text)
        results.append({
            "id": case["id"],
            "prompt": case["prompt"],
            "expect": expected,
            "used_skills": list(used_skills),
            "hit": bool(hit),
        })
    return results

async def run_evaluation():
    """Run evaluation and print results"""
    res = await evaluate_skill_routing(skill_runner_final, eval_cases)
    hits = sum(1 for r in res if r["hit"])
    total = len(res)
    acc = hits / total if total else 0.0
    print("ğŸ“Š Evaluation results:")
    for r in res:
        print(f"- {r['id']}: expect={r['expect']}, used={r['used_skills'] or ['(none)']}, hit={r['hit']}")
    print(f"Accuracy: {hits}/{total} = {acc:.2f}")

# Run evaluation (requires API Key ready); uncomment nest_asyncio if Notebook doesn't support direct await
# import nest_asyncio; nest_asyncio.apply()
await run_evaluation()

