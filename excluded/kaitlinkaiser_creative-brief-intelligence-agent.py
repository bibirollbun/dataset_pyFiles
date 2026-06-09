import os
from kaggle_secrets import UserSecretsClient
from google import genai
from google.genai import types

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    print("âœ… Setup and authentication complete.")
    print("âœ“ API client initialized")
    
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# CREATIVE BRIEF INTELLIGENCE AGENT SYSTEM

!pip install google-genai pandas matplotlib seaborn --quiet

import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai
from google.genai import types
import os
import glob
import time 

print("âœ“ All packages imported")


# Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# Load Data from Files
print("ğŸ“‚ Loading data from files...")

DATA_PATH = '/kaggle/input/marketing-brief-documents/brief-documents/'

import os
import glob
import pandas as pd
import json

# Load all brief text files
brief_files = glob.glob(f'{DATA_PATH}briefs/*.txt')
print(f"Found {len(brief_files)} brief documents")

# Load client history 
try:
    client_history = pd.read_csv(f'{DATA_PATH}client_history.csv')
    # Check if CSV was parsed correctly (all columns in one string)
    if len(client_history.columns) == 1 and ',' in client_history.columns[0]:
        # CSV wasn't parsed correctly, try to fix it
        print("âš ï¸�  CSV parsing issue detected, attempting to fix...")
        # Read the first row to get column names
        with open(f'{DATA_PATH}client_history.csv', 'r') as f:
            first_line = f.readline().strip()
            columns = [col.strip() for col in first_line.split(',')]
        
        # Re-read with proper column names
        client_history = pd.read_csv(f'{DATA_PATH}client_history.csv', names=columns, skiprows=1)
        print(f"âœ“ Fixed CSV parsing - found {len(client_history.columns)} columns")
    print(f"âœ“ Loaded history for {len(client_history)} clients")
except Exception as e:
    print(f"âš ï¸�  Error loading client history: {e}")
    client_history = pd.DataFrame()  # Empty DataFrame as fallback

# Load routing rules
with open(f'{DATA_PATH}routing_rules.json', 'r') as f:
    routing_rules = json.load(f)
print("âœ“ Loaded routing rules")

# Display what we found
print("\nğŸ“§ Brief Documents:")
for filepath in brief_files:
    filename = os.path.basename(filepath)
    print(f"  â€¢ {filename}")


# Function to Read Brief from File
def load_brief_from_file(filepath):
    """Read a brief document and extract metadata"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    filename = os.path.basename(filepath)
    
    # Extract basic info from email format
    lines = content.split('\n')
    brief_data = {
        'filepath': filepath,
        'filename': filename,
        'content': content,
        'from_email': None,
        'subject': None,
        'date': None
    }
    
    # Parse email headers
    for line in lines[:5]:  # Check first 5 lines for headers
        if line.startswith('From:'):
            brief_data['from_email'] = line.replace('From:', '').strip()
        elif line.startswith('Subject:'):
            brief_data['subject'] = line.replace('Subject:', '').strip()
        elif line.startswith('Date:'):
            brief_data['date'] = line.replace('Date:', '').strip()
    
    return brief_data

# Test it on first file
test_brief = load_brief_from_file(brief_files[0])
print(f"\nğŸ“„ Sample Brief Loaded:")
print(f"File: {test_brief['filename']}")
print(f"From: {test_brief['from_email']}")
print(f"Subject: {test_brief['subject']}")
print(f"\nContent preview:\n{test_brief['content'][:200]}...")


# Agent 1 - Brief Extractor
print("Setting up Agent 1: Brief Extractor")

# Tool for extracting structured data from briefs
extract_brief_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name='extract_brief_structure',
            description='Extracts structured information from a messy marketing brief email',
            parameters={
                'type': 'object',
                'properties': {
                    'client': {'type': 'string', 'description': 'Client company name'},
                    'campaign_objective': {'type': 'string', 'description': 'Main campaign goal'},
                    'target_audience': {'type': 'string', 'description': 'Target demographic'},
                    'budget': {'type': 'number', 'description': 'Budget amount in dollars'},
                    'budget_status': {'type': 'string', 'enum': ['confirmed', 'estimated', 'TBD']},
                    'timeline': {'type': 'string', 'description': 'Timeline or launch date'},
                    'channels': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Marketing channels'},
                    'deliverables': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Required deliverables'},
                    'urgency': {'type': 'string', 'enum': ['urgent', 'high', 'normal', 'low']},
                    'special_notes': {'type': 'string', 'description': 'Any special requirements or context'}
                },
                'required': ['client', 'campaign_objective']
            }
        ),
        types.FunctionDeclaration(
            name='lookup_client_history',
            description='Retrieves historical campaign data for a client',
            parameters={
                'type': 'object',
                'properties': {
                    'client_name': {'type': 'string', 'description': 'Name of the client'}
                },
                'required': ['client_name']
            }
        )
    ]
)

def handle_extract_tool(tool_call, brief_content, client_history_df):
    """Handle tool calls for brief extraction"""
    function_name = tool_call.name
    args = dict(tool_call.args) if hasattr(tool_call, 'args') else {}
    
    print(f"   Function: {function_name}")
    print(f"   Args: {args}")
    
    if function_name == 'extract_brief_structure':
        # Agent has extracted structured data - just return it
        return {
            'status': 'success',
            'extracted_data': args,
            'message': 'Brief structure extracted successfully'
        }
    
    elif function_name == 'lookup_client_history':
        client_name = args.get('client_name', '')
        
        if not client_name:
            return {
                'status': 'error',
                'message': 'No client name provided'
            }
        
        # Check if client_history DataFrame is available and has data
        if client_history_df is None or client_history_df.empty:
            return {
                'status': 'no_data',
                'message': 'Client history data not available'
            }
        
        # Find the client column (should be first column or named 'client')
        client_column = None
        if 'client' in client_history_df.columns:
            client_column = 'client'
        elif len(client_history_df.columns) > 0:
            # Try first column
            first_col = client_history_df.columns[0]
            # Check if it's not the concatenated header
            if ',' not in first_col and len(first_col) < 50:
                client_column = first_col
            else:
                # If first column is the concatenated header, try to extract client from data
                # Look in the actual data rows
                for idx, row in client_history_df.iterrows():
                    first_val = str(row.iloc[0])
                    if ',' in first_val:
                        # Split the concatenated row
                        parts = first_val.split(',')
                        if len(parts) > 0 and parts[0].strip().lower() == client_name.lower():
                            # Found match in concatenated format
                            return {
                                'status': 'found',
                                'history': {'note': 'Found in concatenated format, parsing needed'},
                                'message': f"Found client {client_name} but data format needs parsing"
                            }
        
        if client_column is None:
            return {
                'status': 'error',
                'message': 'Could not identify client column in history data'
            }
        
        print(f"   Searching in column: {client_column}")
        
        # Look up in client history - case insensitive
        try:
            matching_clients = client_history_df[
                client_history_df[client_column].astype(str).str.lower().str.contains(client_name.lower(), na=False)
            ]
            
            if not matching_clients.empty:
                history = matching_clients.iloc[0].to_dict()
                # Clean NaN values - replace with None for JSON compatibility
                import math
                cleaned_history = {}
                for key, value in history.items():
                    # Check for NaN (pandas uses float('nan') which doesn't equal itself)
                    if isinstance(value, float) and math.isnan(value):
                        cleaned_history[key] = None
                    elif pd.isna(value):  # Also check for pandas NA values
                        cleaned_history[key] = None
                    else:
                        cleaned_history[key] = value
                
                return {
                    'status': 'found',
                    'history': cleaned_history,
                    'message': f"Found {cleaned_history.get('total_campaigns', 'unknown')} previous campaigns for {client_name}"
                }
            else:
                return {
                    'status': 'new_client',
                    'message': f'No previous history found for {client_name}'
                }
        except Exception as e:
            print(f"   Error searching client history: {e}")
            return {
                'status': 'error',
                'message': f'Error searching client history: {str(e)}'
            }
    
    return {'status': 'error', 'message': 'Unknown tool'}

print("Agent 1 tools configured")


# Agent 1 - Process Brief
def extract_brief_with_agent(brief_data):
    """Use Agent 1 to extract structured info from brief document"""
    
    print("\n" + "="*70)
    print("AGENT 1: Extracting Brief from " + brief_data['filename'])
    print("="*70)
    
    initial_prompt = """You are a marketing brief extraction agent. 

Read this email and extract all relevant campaign information into a structured format.

EMAIL CONTENT:
""" + brief_data['content'] + """

YOUR TASKS:
1. First, use lookup_client_history to see if we have worked with this client before
2. Then use extract_brief_structure to create a structured brief with all the key information
3. Fill in missing fields with typical values from client history if available
4. Flag any critical information that is missing

Be thorough and extract all details mentioned in the email."""

    extracted_data = None
    client_info = None
    contents = initial_prompt
    
    for turn in range(5):
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[extract_brief_tool],
                temperature=0.1,
                http_options=retry_config
            )
        )
        
        # Add error handling for API response
        if not response.candidates or len(response.candidates) == 0:
            print("âš ï¸�  No candidates in API response")
            break
            
        if not hasattr(response.candidates[0], 'content') or not response.candidates[0].content:
            print("âš ï¸�  No content in API response")
            break
        
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    print("\nAgent: " + part.text)
                
                if hasattr(part, 'function_call') and part.function_call:
                    tool_call = part.function_call
                    print("\nTool: " + tool_call.name)
                    
                    result = handle_extract_tool(tool_call, brief_data['content'], client_history)
                    print("   Result: " + result.get('message', 'Done'))
                    
                    if tool_call.name == 'extract_brief_structure':
                        extracted_data = result['extracted_data']
                    elif tool_call.name == 'lookup_client_history':
                        client_info = result.get('history')
                    
                    # Build conversation history for next turn
                    if isinstance(contents, str):
                        contents = [
                            types.Content(role='user', parts=[types.Part(text=initial_prompt)]),
                            response.candidates[0].content,
                            types.Content(
                                role='user',
                                parts=[types.Part(function_response=types.FunctionResponse(
                                    name=tool_call.name,
                                    response=result
                                ))]
                            )
                        ]
                    else:
                        contents.append(response.candidates[0].content)
                        contents.append(
                            types.Content(
                                role='user',
                                parts=[types.Part(function_response=types.FunctionResponse(
                                    name=tool_call.name,
                                    response=result
                                ))]
                            )
                        )
                    break
            else:
                break
        else:
            break
    
    return {
        'brief_data': extracted_data,
        'client_history': client_info,
        'original_file': brief_data['filename']
    }

# Test on first brief (remove this after testing)
result = extract_brief_with_agent(test_brief)
print("\nExtraction Complete!")
if result['brief_data']:
    print("\nExtracted Brief Data:")
    print(json.dumps(result['brief_data'], indent=2))
else:
    print("\nNo structured data extracted yet.")


# Agent 2 - Validation & Routing Agent
print("Setting up Agent 2: Validator & Router")

# Tool for validation and routing
validation_routing_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name='calculate_completeness_score',
            description='Calculates a completeness score for a brief based on required fields',
            parameters={
                'type': 'object',
                'properties': {
                    'brief_data': {'type': 'object', 'description': 'The extracted brief data to validate'},
                    'missing_fields': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of missing critical fields'},
                    'completeness_percentage': {'type': 'number', 'description': 'Completeness score 0-100'}
                },
                'required': ['completeness_percentage']
            }
        ),
        types.FunctionDeclaration(
            name='route_to_team',
            description='Routes brief to appropriate team based on budget, complexity, and urgency',
            parameters={
                'type': 'object',
                'properties': {
                    'brief_id': {'type': 'string', 'description': 'Brief identifier'},
                    'assigned_team': {'type': 'string', 'description': 'Team to handle this brief'},
                    'priority': {'type': 'string', 'enum': ['urgent', 'high', 'normal', 'low']},
                    'reasoning': {'type': 'string', 'description': 'Why this team was chosen'},
                    'requires_human_review': {'type': 'boolean', 'description': 'Whether human review is needed'}
                },
                'required': ['brief_id', 'assigned_team', 'requires_human_review']
            }
        )
    ]
)

def handle_validation_tool(tool_call, brief_data, routing_rules):
    """Handle validation and routing tool calls"""
    function_name = tool_call.name
    args = dict(tool_call.args) if hasattr(tool_call, 'args') else {}
    
    print(f"   Function: {function_name}")
    
    if function_name == 'calculate_completeness_score':
        completeness = args.get('completeness_percentage', 0)
        missing = args.get('missing_fields', [])
        
        return {
            'status': 'success',
            'completeness_score': completeness,
            'missing_fields': missing,
            'message': f"Brief is {completeness}% complete. Missing: {', '.join(missing) if missing else 'none'}"
        }
    
    elif function_name == 'route_to_team':
        # Normalize team names to use underscores (standardize format)
        assigned_team = args.get('assigned_team', 'standard_team')
        # Convert common variations to standard format
        team_normalization = {
            'Standard team': 'standard_team',
            'standard team': 'standard_team',
            'Junior team': 'junior_team',
            'junior team': 'junior_team',
            'Senior team': 'senior_team',
            'senior team': 'senior_team',
            'Executive review': 'executive_review',
            'executive review': 'executive_review'
        }
        assigned_team = team_normalization.get(assigned_team, assigned_team)
        
        return {
            'status': 'success',
            'routing_decision': {
                'brief_id': args.get('brief_id', 'unknown'),
                'assigned_team': assigned_team,
                'priority': args.get('priority', 'normal'),
                'reasoning': args.get('reasoning', 'No reasoning provided'),
                'requires_human_review': args.get('requires_human_review', False)
            },
            'message': f"Routed to {assigned_team}"
        }
    
    return {'status': 'error', 'message': 'Unknown tool'}

print("Agent 2 tools configured")


# Agent 2 - Validate and Route Brief
def validate_and_route_brief(extracted_result):
    """Use Agent 2 to validate completeness and route to appropriate team"""
    
    brief_data = extracted_result.get('brief_data')
    if not brief_data:
        print("No brief data to validate")
        return None
    
    print("\n" + "="*70)
    print("AGENT 2: Validating & Routing")
    print("="*70)
    
    # Build routing rules text from routing_rules.json
    routing_rules_text = "ROUTING RULES:\n"
    team_routing = routing_rules.get('team_routing', {})
    
    # Format team routing rules from JSON
    if 'junior_team' in team_routing:
        max_budget = team_routing['junior_team'].get('budget_max', 0)
        routing_rules_text += f"- Budget up to ${max_budget:,}: Junior team\n"
    
    if 'standard_team' in team_routing:
        min_budget = team_routing['standard_team'].get('budget_min', 0)
        max_budget = team_routing['standard_team'].get('budget_max', 0)
        routing_rules_text += f"- Budget ${min_budget:,}-${max_budget:,}: Standard team\n"
    
    if 'senior_team' in team_routing:
        min_budget = team_routing['senior_team'].get('budget_min', 0)
        exec_min = team_routing.get('executive_review', {}).get('budget_min', float('inf'))
        routing_rules_text += f"- Budget ${min_budget:,}-${exec_min-1:,}: Senior team\n"
    
    if 'executive_review' in team_routing:
        min_budget = team_routing['executive_review'].get('budget_min', 0)
        routing_rules_text += f"- Budget ${min_budget:,}+: Executive review\n"
    
    # Add auto-approve criteria if available
    auto_approve = routing_rules.get('auto_approve_criteria', {})
    if auto_approve.get('max_budget'):
        routing_rules_text += f"- Budget under ${auto_approve['max_budget']:,}: Can auto-approve if simple\n"
    
    prompt = f"""You are a brief validation and routing agent.

EXTRACTED BRIEF DATA:
{json.dumps(brief_data, indent=2)}

{routing_rules_text}

YOUR TASKS:
1. Use calculate_completeness_score to evaluate how complete this brief is
   - Check for: client, campaign_objective, target_audience, budget, timeline, channels, deliverables
   - Calculate a percentage score (0-100)
   - List any critical missing fields

2. Use route_to_team to assign this brief to the appropriate team
   - Consider: budget amount, complexity, urgency flags
   - Decide if human review is required
   - Explain your reasoning

Be thorough in your evaluation."""

    completeness_result = None
    routing_result = None
    contents = prompt
    
    for turn in range(5):
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[validation_routing_tool],
                temperature=0.1,
                http_options=retry_config
            )
        )
        
        # Add error handling for API response
        if not response.candidates or len(response.candidates) == 0:
            print("âš ï¸�  No candidates in API response")
            break
            
        if not hasattr(response.candidates[0], 'content') or not response.candidates[0].content:
            print("âš ï¸�  No content in API response")
            break
        
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    print("\nAgent: " + part.text)
                
                if hasattr(part, 'function_call') and part.function_call:
                    tool_call = part.function_call
                    print("\nTool: " + tool_call.name)
                    
                    result = handle_validation_tool(tool_call, brief_data, routing_rules)
                    print("   Result: " + result.get('message', 'Done'))
                    
                    if tool_call.name == 'calculate_completeness_score':
                        completeness_result = result
                    elif tool_call.name == 'route_to_team':
                        routing_result = result
                    
                    if isinstance(contents, str):
                        contents = [
                            types.Content(role='user', parts=[types.Part(text=prompt)]),
                            response.candidates[0].content,
                            types.Content(
                                role='user',
                                parts=[types.Part(function_response=types.FunctionResponse(
                                    name=tool_call.name,
                                    response=result
                                ))]
                            )
                        ]
                    else:
                        contents.append(response.candidates[0].content)
                        contents.append(
                            types.Content(
                                role='user',
                                parts=[types.Part(function_response=types.FunctionResponse(
                                    name=tool_call.name,
                                    response=result
                                ))]
                            )
                        )
                    break
            else:
                break
        else:
            break
    
    return {
        'completeness': completeness_result,
        'routing': routing_result,
        'original_brief': brief_data
    }
# Test on the extracted result 
validation_result = validate_and_route_brief(result)
print("\nValidation Complete!")
if validation_result:
    print("\nCompleteness Score: " + str(validation_result.get('completeness', {}).get('completeness_score', 'N/A')))
    if validation_result.get('routing'):
        routing = validation_result['routing']['routing_decision']
        print("Assigned Team: " + routing['assigned_team'])
        print("Priority: " + routing['priority'])
        print("Requires Review: " + str(routing['requires_human_review']))


# Process Briefs Through Complete System 
import random

print("="*70)
print("PROCESSING BRIEFS THROUGH AGENT SYSTEM")
print("="*70)

# Storage for all results
all_results = []

# STRATEGY: Process 2 briefs fully, simulate the rest
NUM_REAL_BRIEFS = 2

print(f"\nProcessing {NUM_REAL_BRIEFS} briefs with live agents")
print(f"Simulating {len(brief_files) - NUM_REAL_BRIEFS} briefs for demo purposes")
print("(In production, this would process all briefs with proper queuing)\n")

# Process first N briefs with real agents
for i, brief_file in enumerate(brief_files[:NUM_REAL_BRIEFS]):
    print(f"\n{'#'*70}")
    print(f"REAL PROCESSING - BRIEF {i+1} of {NUM_REAL_BRIEFS}")
    print(f"{'#'*70}")
    
    try:
        # Load the brief
        brief = load_brief_from_file(brief_file)
        
        # Agent 1: Extract
        print("\n[Agent 1: Extracting...]")
        extracted = extract_brief_with_agent(brief)
        
        # Wait 30 seconds between agents to avoid rate limits
        print("\nWaiting 30 seconds to respect rate limits...")
        time.sleep(30)
        
        # Agent 2: Validate & Route
        print("\n[Agent 2: Validating & Routing...]")
        validated = validate_and_route_brief(extracted)
        
        # Store combined result - normalize team name
        assigned_team = validated.get('routing', {}).get('routing_decision', {}).get('assigned_team', 'unknown') if validated else 'unknown'
        # Ensure team name is normalized
        team_normalization = {
            'Standard team': 'standard_team',
            'standard team': 'standard_team',
            'Junior team': 'junior_team',
            'junior team': 'junior_team',
            'Senior team': 'senior_team',
            'senior team': 'senior_team',
            'Executive review': 'executive_review',
            'executive review': 'executive_review'
        }
        assigned_team = team_normalization.get(assigned_team, assigned_team)
        
        result_record = {
            'filename': brief['filename'],
            'client': extracted.get('brief_data', {}).get('client', 'Unknown'),
            'budget': extracted.get('brief_data', {}).get('budget', 0),
            'completeness_score': validated.get('completeness', {}).get('completeness_score', 0) if validated else 0,
            'assigned_team': assigned_team,
            'priority': validated.get('routing', {}).get('routing_decision', {}).get('priority', 'normal') if validated else 'normal',
            'requires_review': validated.get('routing', {}).get('routing_decision', {}).get('requires_human_review', False) if validated else False,
            'extracted_data': extracted.get('brief_data'),
            'client_history': extracted.get('client_history'),
            'processing_type': 'REAL'
        }
        
        all_results.append(result_record)
        
        print(f"\n>>> REAL RESULT for {brief['filename']}:")
        print(f"    Client: {result_record['client']}")
        print(f"    Budget: ${result_record['budget']:,}")
        print(f"    Completeness: {result_record['completeness_score']}%")
        print(f"    Team: {result_record['assigned_team']}")
        print(f"    Priority: {result_record['priority']}")
        
        # Wait 30 seconds before next brief
        if i < NUM_REAL_BRIEFS - 1:
            print(f"\nWaiting 30 seconds before next brief...")
            time.sleep(30)
            
    except Exception as e:
        print(f"\nError processing {brief_file}: {str(e)}")
        continue

print("\n" + "="*70)
print(f"COMPLETED {len(all_results)} REAL AGENT PROCESSING")
print("="*70)

# Simulate remaining briefs with realistic data
print(f"\nGenerating simulated results for remaining {len(brief_files) - NUM_REAL_BRIEFS} briefs...")
print("(This demonstrates what the full system output would look like)\n")

# Helper function to apply routing logic from routing_rules.json
def apply_routing_logic(budget, routing_rules):
    """
    Apply routing logic using the routing_rules.json configuration
    This ensures consistency between real agent routing and simulated routing
    """
    team_routing = routing_rules['team_routing']
    
    # Apply budget-based routing rules from JSON
    if budget <= team_routing['junior_team']['budget_max']:
        team = 'junior_team'
        priority = random.choice(['normal', 'low'])
    elif budget <= team_routing['standard_team']['budget_max']:
        team = 'standard_team'
        priority = random.choice(['normal', 'high'])
    elif budget >= team_routing['senior_team']['budget_min'] and budget < team_routing['executive_review']['budget_min']:
        team = 'senior_team'
        priority = random.choice(['high', 'urgent'])
    else:
        team = 'executive_review'
        priority = 'urgent'
    
    return team, priority

# Realistic budget ranges per client (from client_history.csv)
client_budgets = {
    'Nike': [75000, 95000, 120000, 180000],
    'Adidas': [150000, 200000, 2500000],
    'Puma': [60000, 68000, 85000, 180000],
    'Under Armour': [120000, 150000, 200000],
    'Reebok': [1200, 8000, 45000, 55000]
}

# Process remaining briefs with simulation
for brief_file in brief_files[NUM_REAL_BRIEFS:]:
    brief = load_brief_from_file(brief_file)
    
    # Extract client from filename
    client = 'Unknown'
    budget = 50000
    
    # Determine client and select realistic budget
    if 'nike' in brief['filename'].lower():
        client = 'Nike'
        budget = random.choice(client_budgets['Nike'])
    elif 'adidas' in brief['filename'].lower():
        client = 'Adidas'
        budget = random.choice(client_budgets['Adidas'])
    elif 'puma' in brief['filename'].lower():
        client = 'Puma'
        budget = random.choice(client_budgets['Puma'])
    elif 'reebok' in brief['filename'].lower():
        client = 'Reebok'
        budget = random.choice(client_budgets['Reebok'])
    elif 'armour' in brief['filename'].lower():
        client = 'Under Armour'
        budget = random.choice(client_budgets['Under Armour'])
    
    # Apply routing logic using routing_rules.json 
    team, priority = apply_routing_logic(budget, routing_rules)
    
    # Determine review requirement using auto-approve criteria from JSON
    auto_approve_criteria = routing_rules['auto_approve_criteria']
    requires_review = budget > auto_approve_criteria['max_budget'] or random.choice([True, False])
    
    # Simulated result with realistic data
    simulated = {
        'filename': brief['filename'],
        'client': client,
        'budget': budget,
        'completeness_score': random.randint(65, 95),
        'assigned_team': team,
        'priority': priority,
        'requires_review': requires_review,
        'extracted_data': {'simulated': True},
        'client_history': None,
        'processing_type': 'SIMULATED'
    }
    all_results.append(simulated)
    
    print(f"  â€¢ {brief['filename']}: {client}, ${budget:,}, {team}, priority={priority}")

print("\n" + "="*70)
print(f"TOTAL RESULTS: {len(all_results)} briefs")
print(f"  - {NUM_REAL_BRIEFS} processed with live agents")
print(f"  - {len(all_results) - NUM_REAL_BRIEFS} simulated for demonstration")
print(f"  - All routing decisions use routing_rules.json logic")
print("="*70)


# Agent 3 - Tracking & Performance Metrics
print("\n" + "="*70)
print("AGENT 3: TRACKING & METRICS")
print("="*70)

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(all_results)

# Calculate key metrics
total_briefs = len(results_df)
avg_completeness = results_df['completeness_score'].mean()
total_budget = results_df['budget'].sum()
avg_budget = results_df['budget'].mean()

# Team distribution
team_counts = results_df['assigned_team'].value_counts()

# Priority distribution  
priority_counts = results_df['priority'].value_counts()

# Auto-approval rate (briefs that don't need review)
auto_approved = len(results_df[results_df['requires_review'] == False])
auto_approval_rate = (auto_approved / total_briefs * 100) if total_briefs > 0 else 0

# Completeness categories
high_completeness = len(results_df[results_df['completeness_score'] >= 80])
medium_completeness = len(results_df[(results_df['completeness_score'] >= 50) & (results_df['completeness_score'] < 80)])
low_completeness = len(results_df[results_df['completeness_score'] < 50])

print("\nKEY PERFORMANCE METRICS")
print("-" * 70)
print(f"Total Briefs Processed: {total_briefs}")
print(f"Average Completeness Score: {avg_completeness:.1f}%")
print(f"Total Budget Value: ${total_budget:,.0f}")
print(f"Average Budget per Brief: ${avg_budget:,.0f}")
print(f"Auto-Approval Rate: {auto_approval_rate:.1f}%")

print("\nTEAM ROUTING DISTRIBUTION")
print("-" * 70)
for team, count in team_counts.items():
    percentage = (count / total_briefs * 100)
    print(f"{team}: {count} briefs ({percentage:.1f}%)")

print("\nPRIORITY DISTRIBUTION")
print("-" * 70)
for priority, count in priority_counts.items():
    percentage = (count / total_briefs * 100)
    print(f"{priority}: {count} briefs ({percentage:.1f}%)")

print("\nCOMPLETENESS BREAKDOWN")
print("-" * 70)
print(f"High (80%+): {high_completeness} briefs")
print(f"Medium (50-79%): {medium_completeness} briefs")
print(f"Low (<50%): {low_completeness} briefs")

# Save metrics
metrics = {
    'total_briefs': total_briefs,
    'avg_completeness': avg_completeness,
    'total_budget': total_budget,
    'auto_approval_rate': auto_approval_rate,
    'team_distribution': team_counts.to_dict(),
    'priority_distribution': priority_counts.to_dict(),
    'completeness_breakdown': {
        'high': high_completeness,
        'medium': medium_completeness,
        'low': low_completeness
    }
}


# Create Visual Dashboard
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Creative Brief Intelligence Agent - Performance Dashboard', fontsize=16, fontweight='bold')

# Chart 1: Completeness Score Distribution
ax1 = axes[0, 0]
ax1.hist(results_df['completeness_score'], bins=10, color='skyblue', edgecolor='black')
ax1.axvline(avg_completeness, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_completeness:.1f}%')
ax1.set_xlabel('Completeness Score (%)')
ax1.set_ylabel('Number of Briefs')
ax1.set_title('Brief Completeness Distribution')
ax1.legend()

# Chart 2: Team Routing
ax2 = axes[0, 1]
team_counts.plot(kind='bar', ax=ax2, color='coral')
ax2.set_xlabel('Team')
ax2.set_ylabel('Number of Briefs')
ax2.set_title('Brief Routing by Team')
ax2.tick_params(axis='x', rotation=45)

# Chart 3: Budget Distribution by Team
ax3 = axes[0, 2]
team_budgets = results_df.groupby('assigned_team')['budget'].sum().sort_values()
team_budgets.plot(kind='barh', ax=ax3, color='lightgreen')
ax3.set_xlabel('Total Budget ($)')
ax3.set_ylabel('Team')
ax3.set_title('Total Budget by Team')

# Chart 4: Priority Levels
ax4 = axes[1, 0]
priority_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#ffcc99','#99ccff','#99ff99'])
ax4.set_ylabel('')
ax4.set_title('Priority Level Distribution')

# Chart 5: Auto-Approval vs Review Required
ax5 = axes[1, 1]
review_counts = results_df['requires_review'].value_counts()
review_labels = ['Auto-Approved', 'Needs Review']
colors = ['lightgreen', 'lightyellow']
ax5.pie([auto_approved, total_briefs - auto_approved], labels=review_labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax5.set_title('Approval Status')

# Chart 6: Completeness Categories
ax6 = axes[1, 2]
categories = ['High\n(80%+)', 'Medium\n(50-79%)', 'Low\n(<50%)']
values = [high_completeness, medium_completeness, low_completeness]
colors_cat = ['#2ecc71', '#f39c12', '#e74c3c']
ax6.bar(categories, values, color=colors_cat)
ax6.set_ylabel('Number of Briefs')
ax6.set_title('Completeness Quality')

plt.tight_layout()
plt.show()

print("\nDashboard generated successfully!")

