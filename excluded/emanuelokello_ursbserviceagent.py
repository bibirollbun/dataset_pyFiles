# Cell 1: Setup and Authentication
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key setup complete.")
except Exception as e:
    print(f"Authentication Error: {e}")


# Cell 2: Import Required Libraries
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, google_search, ToolContext
from google.adk.apps.app import App, EventsCompactionConfig
from google.genai import types

print("All imports successful")


# Cell 3: Configuration and Constants
# Retry configuration for robust API calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Service categories
SERVICES = [
    "Business Registration",
    "Intellectual Property (IP) Registration",
    "SIMPO Registration",
    "Insolvency and Receivership"
]

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "lg": "Luganda",
    "sw": "Swahili",
    "luo": "Luo"
}

# Application constants
APP_NAME = "URSBServiceAgent"
USER_ID = "ursb_user"
MODEL_NAME = "gemini-2.5-flash-lite"

print("Configuration loaded")


# Cell 4: Knowledge Base Integration
# Based on your Google Doc knowledge base

URSB_KNOWLEDGE_BASE = {
    "business_registration": {
        "description": "Registration of companies, business names, and partnerships",
        "requirements": [
            "Valid identification documents",
            "Proposed business name",
            "Business address",
            "Details of directors/partners"
        ],
        "process_steps": [
            "Name search and reservation",
            "Preparation of incorporation documents",
            "Payment of registration fees",
            "Submission of documents",
            "Certificate issuance"
        ],
        "fees": {
            "company_registration": "UGX 150,000 - 500,000 (depending on share capital)",
            "business_name": "UGX 50,000",
            "partnership": "UGX 100,000"
        },
        "processing_time": "3-5 business days",
        "contact": "business@ursb.go.ug"
    },
    "intellectual_property": {
        "description": "Protection of trademarks, patents, industrial designs, and copyrights",
        "types": [
            "Trademarks",
            "Patents",
            "Industrial Designs",
            "Utility Models",
            "Geographical Indications"
        ],
        "requirements": [
            "Completed application form",
            "Representation of the mark/design",
            "Proof of payment",
            "Power of attorney (if using agent)"
        ],
        "fees": {
            "trademark_application": "UGX 150,000",
            "patent_application": "UGX 500,000",
            "industrial_design": "UGX 200,000"
        },
        "processing_time": "6-12 months (varies by type)",
        "contact": "ip@ursb.go.ug"
    },
    "simpo": {
        "description": "Security Interests in Movable Property registration",
        "purpose": "Register security interests over movable assets like vehicles, machinery, inventory",
        "search_url": "https://simpodev.ursb.go.ug/upgrade/Search/Search/Create?NonLegalEffect=True",
        "benefits": [
            "Priority over unsecured creditors",
            "Legal proof of security interest",
            "Protects lender's rights"
        ],
        "requirements": [
            "Security agreement",
            "Debtor details",
            "Description of collateral",
            "Secured party information"
        ],
        "fees": {
            "registration": "UGX 50,000",
            "search": "Free online search available"
        },
        "contact": "simpo@ursb.go.ug"
    },
    "insolvency": {
        "description": "Administration of insolvent estates and receivership",
        "services": [
            "Company liquidation",
            "Personal bankruptcy",
            "Receivership management",
            "Administrator appointment"
        ],
        "requirements": [
            "Court order or creditor petition",
            "Financial statements",
            "List of creditors and debtors",
            "Asset inventory"
        ],
        "contact": "insolvency@ursb.go.ug"
    }
}

print("Knowledge base loaded")
print(f"   Services: {len(URSB_KNOWLEDGE_BASE)}")


# Cell 5: Language Detection and Translation Tools

def detect_language(text: str, tool_context: Optional[ToolContext] = None) -> Dict[str, str]:
    """
    Detect the language of user input.
    Returns language code and confidence level.
    """
    # Simple keyword-based detection for common local languages
    luganda_keywords = ["webale", "nyabo", "ssebo", "otya", "nkwagala", "bambi"]
    swahili_keywords = ["habari", "asante", "tafadhali", "sawa", "karibu"]
    luo_keywords = ["apwoyo", "ber", "nadi", "kare"]
    
    text_lower = text.lower()
    
    # Check for language indicators
    if any(word in text_lower for word in luganda_keywords):
        return {"language": "lg", "language_name": "Luganda", "confidence": "high"}
    elif any(word in text_lower for word in swahili_keywords):
        return {"language": "sw", "language_name": "Swahili", "confidence": "high"}
    elif any(word in text_lower for word in luo_keywords):
        return {"language": "luo", "language_name": "Luo", "confidence": "high"}
    else:
        # Default to English
        return {"language": "en", "language_name": "English", "confidence": "medium"}


def translate_to_english(text: str, source_language: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Translate user input to English for processing.
    Uses simple phrase mapping for common queries.
    """
    # Translation mappings for common service-related phrases
    translations = {
        "lg": {
            "njagala okuwandiisa bizinensi": "I want to register a business",
            "ssente mmeka": "how much does it cost",
            "kigenda kumala bbanga ki": "how long does it take",
            "nkola ntya": "how do I",
            "ndi wa": "where is",
            "sirina": "I don't have"
        },
        "sw": {
            "nataka kusajili biashara": "I want to register a business",
            "ni kiasi gani": "how much does it cost",
            "itachukua muda gani": "how long does it take",
            "ninafanya vipi": "how do I",
            "iko wapi": "where is"
        },
        "luo": {
            "adwaro ndiko ohala": "I want to register a business",
            "chalo adi": "how much does it cost",
            "biro kawo kinde adi": "how long does it take"
        }
    }
    
    if source_language == "en":
        return {"translated_text": text, "original_language": "en"}
    
    # Attempt translation using mappings
    text_lower = text.lower()
    if source_language in translations:
        for phrase, translation in translations[source_language].items():
            if phrase in text_lower:
                text_lower = text_lower.replace(phrase, translation)
        
        return {
            "translated_text": text_lower,
            "original_language": source_language,
            "note": "Partial translation using phrase mapping"
        }
    
    return {
        "translated_text": text,
        "original_language": source_language,
        "note": "No translation available, returning original"
    }


def translate_to_local(text: str, target_language: str, tool_context: Optional[ToolContext] = None) -> Dict[str, str]:
    """
    Translate English response back to user's language.
    Uses simple phrase mapping for common responses.
    """
    if target_language == "en":
        return {"translated_text": text, "target_language": "en"}
    
    # Translation mappings for common responses
    response_translations = {
        "lg": {
            "the cost is": "ssente ze",
            "you need": "weetaaga",
            "the process takes": "enkola egenda kumala",
            "please contact": "bambi tuukirire ku",
            "thank you": "webale nyo"
        },
        "sw": {
            "the cost is": "gharama ni",
            "you need": "unahitaji",
            "the process takes": "mchakato unachukua",
            "please contact": "tafadhali wasiliana na",
            "thank you": "asante sana"
        },
        "luo": {
            "the cost is": "chudo en",
            "you need": "idwaro",
            "the process takes": "tich kawo",
            "please contact": "kiyie iluong",
            "thank you": "apwoyo matek"
        }
    }
    
    if target_language in response_translations:
        translated = text
        for eng_phrase, local_phrase in response_translations[target_language].items():
            translated = translated.replace(eng_phrase, local_phrase)
        
        return {
            "translated_text": translated,
            "target_language": target_language,
            "note": "Partial translation applied"
        }
    
    return {
        "translated_text": text,
        "target_language": target_language,
        "note": "Translation not available"
    }

print("Translation tools defined")


# Cell 6: Knowledge Base Search Tool

def search_knowledge_base(query: str, service_category: Optional[str] = None, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Search the URSB knowledge base for relevant information.
    
    Args:
        query: User's search query
        service_category: Optional specific service category to search
        tool_context: Tool execution context
        
    Returns:
        Dictionary with search results and relevant information
    """
    results = []
    query_lower = query.lower()
    
    # Keywords for each service
    service_keywords = {
        "business_registration": ["business", "company", "registration", "incorporate", "partnership", "register"],
        "intellectual_property": ["trademark", "patent", "copyright", "ip", "intellectual", "brand", "invention"],
        "simpo": ["simpo", "security", "movable", "collateral", "asset", "vehicle", "machinery"],
        "insolvency": ["insolvency", "bankruptcy", "liquidation", "receivership", "insolvent", "wind up"]
    }
    
    # Determine which services are relevant
    if service_category:
        # Search specific category
        categories_to_search = [service_category.lower().replace(" ", "_").replace("(", "").replace(")", "")]
    else:
        # Search all categories that match keywords
        categories_to_search = []
        for category, keywords in service_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                categories_to_search.append(category)
        
        # If no match, search all
        if not categories_to_search:
            categories_to_search = list(URSB_KNOWLEDGE_BASE.keys())
    
    # Extract relevant information
    for category in categories_to_search:
        if category in URSB_KNOWLEDGE_BASE:
            info = URSB_KNOWLEDGE_BASE[category]
            result = {
                "service": category.replace("_", " ").title(),
                "description": info.get("description", ""),
                "contact": info.get("contact", "info@ursb.go.ug")
            }
            
            # Add specific information based on query intent
            if any(word in query_lower for word in ["cost", "fee", "price", "pay", "ssente", "gharama", "chudo"]):
                result["fees"] = info.get("fees", {})
            
            if any(word in query_lower for word in ["how long", "time", "duration", "bbanga", "muda", "kinde"]):
                result["processing_time"] = info.get("processing_time", "Contact URSB for timeline")
            
            if any(word in query_lower for word in ["need", "require", "document", "weetaaga", "unahitaji"]):
                result["requirements"] = info.get("requirements", [])
            
            if any(word in query_lower for word in ["process", "step", "how", "procedure", "enkola", "mchakato"]):
                result["process_steps"] = info.get("process_steps", [])
            
            # Add SIMPO search link if relevant
            if category == "simpo" and "search_url" in info:
                result["search_url"] = info["search_url"]
                result["note"] = "Free online search available"
            
            results.append(result)
    
    if not results:
        return {
            "status": "no_results",
            "message": "No specific information found. Please contact URSB at info@ursb.go.ug for assistance.",
            "available_services": SERVICES
        }
    
    return {
        "status": "success",
        "results": results,
        "result_count": len(results)
    }

print("Knowledge base search tool defined")


# Cell 7: SIMPO Search Tool

def check_simpo_registration(search_term: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Check SIMPO registration status.
    Provides information about the SIMPO search portal.
    
    Args:
        search_term: Term to search (company name, registration number, etc.)
        tool_context: Tool execution context
        
    Returns:
        Information about how to search SIMPO
    """
    return {
        "status": "info",
        "message": "SIMPO (Security Interests in Movable Property) search is available online.",
        "search_url": "https://simpodev.ursb.go.ug/upgrade/Search/Search/Create?NonLegalEffect=True",
        "search_term": search_term,
        "instructions": [
            "Visit the SIMPO search portal",
            "Enter the search term (company name, asset description, or registration number)",
            "Review the search results for any registered security interests",
            "Search is free and provides instant results"
        ],
        "note": "The search shows if there are any security interests registered over movable property",
        "api_note": "API integration coming soon for direct searches"
    }

print("SIMPO search tool defined")


# Cell 8: Service Routing Tool

def route_to_service(query: str, detected_service: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Route user query to the appropriate service specialist.
    
    Args:
        query: User's query
        detected_service: The service category detected
        tool_context: Tool execution context
        
    Returns:
        Routing information and initial response
    """
    service_map = {
        "business registration": "business_registration",
        "intellectual property": "intellectual_property",
        "ip registration": "intellectual_property",
        "simpo": "simpo",
        "simpo registration": "simpo",
        "insolvency": "insolvency",
        "receivership": "insolvency"
    }
    
    service_key = service_map.get(detected_service.lower())
    
    if service_key and service_key in URSB_KNOWLEDGE_BASE:
        service_info = URSB_KNOWLEDGE_BASE[service_key]
        return {
            "status": "routed",
            "service": detected_service,
            "specialist": f"{detected_service.title()} Specialist",
            "description": service_info.get("description", ""),
            "message": f"I'll help you with {detected_service}. What specific information do you need?"
        }
    
    return {
        "status": "needs_clarification",
        "message": "I can help you with the following services:",
        "available_services": SERVICES,
        "prompt": "Which service are you interested in?"
    }

print("Service routing tool defined")


# Cell 9: Create Specialist Agents

# Business Registration Specialist
business_registration_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="BusinessRegistrationSpecialist",
    description="Expert in company registration, business names, and partnerships",
    instruction="""You are a Business Registration specialist at URSB.

Your role:
1. Provide accurate information about company registration, business names, and partnerships
2. Use the search_knowledge_base tool to find specific information
3. Be concise and clear - provide direct answers
4. Always include relevant fees, requirements, and processing times
5. Direct users to contact details when needed

Guidelines:
- Be professional and helpful
- Provide step-by-step guidance when asked about processes
- Mention contact email: business@ursb.go.ug for detailed assistance
- If information is not in the knowledge base, recommend contacting URSB directly
""",
    tools=[FunctionTool(func=search_knowledge_base)]
)

# IP Registration Specialist
ip_registration_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="IPRegistrationSpecialist",
    description="Expert in trademarks, patents, and intellectual property protection",
    instruction="""You are an Intellectual Property specialist at URSB.

Your role:
1. Provide expert guidance on trademarks, patents, industrial designs, and copyrights
2. Use the search_knowledge_base tool for detailed information
3. Explain IP concepts in simple terms
4. Guide users on the appropriate type of IP protection for their needs

Guidelines:
- Help users understand different types of IP protection
- Provide clear information on fees and timelines
- Mention contact email: ip@ursb.go.ug for complex cases
- Be patient and educational in your explanations
""",
    tools=[FunctionTool(func=search_knowledge_base)]
)

# SIMPO Specialist
simpo_specialist_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="SIMPOSpecialist",
    description="Expert in Security Interests in Movable Property registration and searches",
    instruction="""You are a SIMPO (Security Interests in Movable Property) specialist at URSB.

Your role:
1. Explain SIMPO registration and its importance
2. Guide users on how to search the SIMPO registry
3. Use check_simpo_registration tool to provide search portal information
4. Use search_knowledge_base for detailed SIMPO information

Guidelines:
- Always mention the free online search portal: https://simpodev.ursb.go.ug  
- Explain that SIMPO protects lenders' security interests over movable assets
- Provide clear instructions on using the search portal
- Mention contact email: simpo@ursb.go.ug
- Note that API integration is coming soon
""",
    tools=[FunctionTool(func=search_knowledge_base), FunctionTool(func=check_simpo_registration)]
)

# Insolvency Specialist
insolvency_specialist_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="InsolvencySpecialist",
    description="Expert in insolvency proceedings, liquidation, and receivership",
    instruction="""You are an Insolvency and Receivership specialist at URSB.

Your role:
1. Provide information on company liquidation, bankruptcy, and receivership
2. Guide users through the insolvency process
3. Use search_knowledge_base for specific information
4. Be sensitive as these are often difficult situations

Guidelines:
- Be professional and empathetic
- Explain processes clearly and step-by-step
- Provide contact email: insolvency@ursb.go.ug for urgent matters
- Recommend consulting legal professionals for complex cases
""",
    tools=[FunctionTool(func=search_knowledge_base)]
)

print("Specialist agents created")
print(f"   - Business Registration Specialist")
print(f"   - IP Registration Specialist")
print(f"   - SIMPO Specialist")
print(f"   - Insolvency Specialist")


# Cell 10: Create Coordinator Agent

from google.adk.tools import AgentTool

# Main coordinator agent
ursb_coordinator_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="URSBServiceCoordinator",
    description="Main coordinator for URSB customer service",
    instruction="""You are the main customer service coordinator for URSB (Uganda Registration Services Bureau).

Your role:
1. Greet users warmly and professionally
2. Detect user's language using detect_language tool
3. If not English, translate query using translate_to_english tool
4. Identify which service the user needs help with
5. Delegate to the appropriate specialist agent:
   - BusinessRegistrationSpecialist for company/business registration
   - IPRegistrationSpecialist for trademarks, patents, IP
   - SIMPOSpecialist for security interests and SIMPO searches
   - InsolvencySpecialist for insolvency and receivership
6. Use search_knowledge_base for general URSB information
7. Translate responses back to user's language using translate_to_local if needed

CRITICAL WORKFLOW:
1. First, detect language
2. Translate to English if needed
3. Route to specialist OR answer directly
4. Translate response back if needed
5. Be concise and helpful

Available services:
- Business Registration (companies, business names, partnerships)
- Intellectual Property Registration (trademarks, patents, designs)
- SIMPO Registration (security interests in movable property)
- Insolvency and Receivership

Guidelines:
- ALWAYS be professional and courteous
- Provide clear, accurate information
- When in doubt, provide contact information: info@ursb.go.ug
- Keep responses concise (max 3-4 paragraphs)
- Include relevant fees, timelines, and requirements
- Direct complex cases to appropriate specialist agents
""",
    tools=[
        FunctionTool(func=detect_language),
        FunctionTool(func=translate_to_english),
        FunctionTool(func=translate_to_local),
        FunctionTool(func=search_knowledge_base),
        FunctionTool(func=route_to_service),
        AgentTool(agent=business_registration_agent),
        AgentTool(agent=ip_registration_agent),
        AgentTool(agent=simpo_specialist_agent),
        AgentTool(agent=insolvency_specialist_agent)
    ]
)

print("URSB Coordinator Agent created")
print("   Main entry point with language detection and specialist routing")


# Cell 11: Create Application with Memory Management

# Create app with efficient context management
ursb_app = App(
    name=APP_NAME,
    root_agent=ursb_coordinator_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,  # Compact every 5 turns
        overlap_size=3  # Keep only 3 turn overlap for context
    )
)

print("URSB Application created")
print("   Features:")
print("   - Automatic context compaction (every 5 turns)")
print("   - Minimal overlap (3 turn) for efficiency")
print("   - Optimized for performance")


# Cell 12: Create Runner with Session Management

# Session service for conversation management
session_service = InMemorySessionService()

# Create runner
ursb_runner = Runner(
    app=ursb_app,
    session_service=session_service
)

print("URSB Service Runner created")
print("   Ready to handle customer queries!")


# Cell 13: Helper Function for Testing

async def chat_with_ursb(user_message: str, session_id: str = "demo_session", user_id: str = USER_ID):
    """
    Helper function to interact with the URSB agent.
    
    Args:
        user_message: The user's question or request
        session_id: Session identifier (for conversation continuity)
        user_id: User identifier
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}\n")
    
    # Create or get session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
    
    # Create message content
    message_content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )
    
    # Run agent and collect response
    full_response = ""
    async for event in ursb_runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message_content
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    full_response += part.text
    
    print(f"URSB AGENT: {full_response}")
    print(f"\n{'='*60}\n")
    
    return full_response

print("Chat helper function defined")


# Cell 14: Test Cases - English

print("Testing URSB Agent with English queries\n")

# Test 1: Business Registration
await chat_with_ursb(
    "Hello, I want to register a new company. What do I need?",
    session_id="test_en_1"
)

# Test 2: Cost inquiry
await chat_with_ursb(
    "How much does it cost to register a business name?",
    session_id="test_en_2"
)

# Test 3: SIMPO inquiry
await chat_with_ursb(
    "I need to search if there's a security interest on a vehicle. How do I do that?",
    session_id="test_en_3"
)

# Test 4: IP inquiry
await chat_with_ursb(
    "How long does it take to register a trademark?",
    session_id="test_en_4"
)


# Cell 16: Performance Monitoring

import time
from typing import List, Dict

async def benchmark_agent(queries: List[str], num_runs: int = 3) -> Dict[str, Any]:
    """
    Benchmark agent performance.
    
    Args:
        queries: List of test queries
        num_runs: Number of times to run each query
        
    Returns:
        Performance metrics
    """
    results = {
        "total_queries": len(queries) * num_runs,
        "query_times": [],
        "avg_response_time": 0,
        "min_response_time": float('inf'),
        "max_response_time": 0
    }
    
    print("â�±ï¸�  Running performance benchmark...\n")
    
    for i, query in enumerate(queries):
        for run in range(num_runs):
            session_id = f"perf_test_{i}_{run}"
            
            start_time = time.time()
            await chat_with_ursb(query, session_id=session_id)
            elapsed = time.time() - start_time
            
            results["query_times"].append(elapsed)
            results["min_response_time"] = min(results["min_response_time"], elapsed)
            results["max_response_time"] = max(results["max_response_time"], elapsed)
            
            print(f"Query {i+1}, Run {run+1}: {elapsed:.2f}s")
    
    results["avg_response_time"] = sum(results["query_times"]) / len(results["query_times"])
    
    print(f"\nğŸ“Š Performance Summary:")
    print(f"   Total Queries: {results['total_queries']}")
    print(f"   Average Response Time: {results['avg_response_time']:.2f}s")

