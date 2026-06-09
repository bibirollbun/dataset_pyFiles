import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print("ğŸ”‘ Authentication Error: Please add 'GOOGLE_API_KEY' to your Kaggle secrets.")
    raise e


from typing import Any, Dict, List
import json

from google.genai import types

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext

print("âœ… ADK components imported successfully.")

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Session service (conceptual)
session_service = InMemorySessionService()
print("âœ… Session service (conceptual) created.")


def pretty_print_json(data: Any):
    print(json.dumps(data, indent=2, ensure_ascii=False))

print("âœ… Helper pretty printer ready.")


# =============================================================================
# LEAD DATABASE - Customize this for your target market! ğŸ�¯
# =============================================================================

LEADS_DB: List[Dict[str, Any]] = [
    {
        "lead_id": "L001",
        "name": "Priya Sharma",
        "title": "Head of Sales Operations",
        "company": "TechVista Solutions",
        "domain": "techvista.io",
        "email": "priya.sharma@techvista.io",
        "location": "Bengaluru, India",
        "industry": "SaaS - HR Tech",
        "company_size": "50-200",
        "revenue_band": "1M-10M",
        "tech_stack": ["HubSpot", "Salesforce", "Slack", "Zoom"],
        "recent_signals": [
            "Series A funding ($5M) announced 2 months ago",
            "Hiring 3 new sales managers",
            "Recently launched AI-powered recruitment module"
        ],
        "pain_points": [
            "Manual lead qualification process",
            "Low email response rates",
            "CRM data hygiene issues"
        ],
        "linkedin_url": "https://linkedin.com/in/priyasharma",
        "qualification_score": 0,
        "priority": "high",  # high/medium/low (will be calculated)
        "status": "new"
    },
    {
        "lead_id": "L002",
        "name": "Michael Chen",
        "title": "VP of Marketing",
        "company": "GrowthLabs Inc",
        "domain": "growthlabs.co",
        "email": "mchen@growthlabs.co",
        "location": "San Francisco, USA",
        "industry": "Marketing Automation",
        "company_size": "200-500",
        "revenue_band": "10M-50M",
        "tech_stack": ["Marketo", "Google Analytics", "Outreach.io"],
        "recent_signals": [
            "Series B funding ($15M) 6 months ago",
            "Expanding to EMEA market",
            "CEO mentioned 'AI-first strategy' in recent podcast"
        ],
        "pain_points": [
            "Scaling personalized outreach",
            "Attribution tracking across channels",
            "High customer acquisition cost"
        ],
        "linkedin_url": "https://linkedin.com/in/michaelchen",
        "qualification_score": 0,
        "priority": "high",
        "status": "new"
    },
    {
        "lead_id": "L003",
        "name": "Sarah Williams",
        "title": "Director of Business Development",
        "company": "CloudSync Enterprise",
        "domain": "cloudsync.com",
        "email": "sarah.w@cloudsync.com",
        "location": "London, UK",
        "industry": "Cloud Infrastructure",
        "company_size": "500-1000",
        "revenue_band": "50M-100M",
        "tech_stack": ["AWS", "Pipedrive", "Intercom", "Calendly"],
        "recent_signals": [
            "IPO filing announced last quarter",
            "Opened 2 new regional offices",
            "Launched partnership program"
        ],
        "pain_points": [
            "Partner onboarding takes 3+ months",
            "Low engagement on cold outreach",
            "Need to scale BDR team without headcount"
        ],
        "linkedin_url": "https://linkedin.com/in/sarahwilliams",
        "qualification_score": 0,
        "priority": "medium",
        "status": "new"
    },
    {
        "lead_id": "L004",
        "name": "Rajesh Kumar",
        "title": "Chief Revenue Officer",
        "company": "FinTech Innovations",
        "domain": "fintechinnovations.in",
        "email": "rajesh@fintechinnovations.in",
        "location": "Mumbai, India",
        "industry": "FinTech - Digital Payments",
        "company_size": "100-250",
        "revenue_band": "5M-10M",
        "tech_stack": ["Stripe", "Zoho CRM", "Freshworks"],
        "recent_signals": [
            "Raised Pre-Series A ($2M)",
            "Featured in TechCrunch for AI fraud detection",
            "Hiring Head of Enterprise Sales"
        ],
        "pain_points": [
            "Enterprise deal cycles too long",
            "Limited sales automation",
            "Need better lead scoring"
        ],
        "linkedin_url": "https://linkedin.com/in/rajeshkumar",
        "qualification_score": 0,
        "priority": "high",
        "status": "new"
    },
    {
        "lead_id": "L005",
        "name": "Emily Rodriguez",
        "title": "Co-founder & COO",
        "company": "EduTech Global",
        "domain": "edutechglobal.com",
        "email": "emily@edutechglobal.com",
        "location": "Austin, USA",
        "industry": "EdTech - Online Learning",
        "company_size": "20-50",
        "revenue_band": "1M-5M",
        "tech_stack": ["Notion", "Airtable", "Loom", "Mailchimp"],
        "recent_signals": [
            "Bootstrapped, profitable since Year 2",
            "Won 'Best EdTech Startup' award",
            "Expanding course catalog by 300%"
        ],
        "pain_points": [
            "Manual student onboarding",
            "Low conversion from free trial",
            "Need automated follow-ups for leads"
        ],
        "linkedin_url": "https://linkedin.com/in/emilyrodriguez",
        "qualification_score": 0,
        "priority": "medium",
        "status": "new"
    },
    {
        "lead_id": "L006",
        "name": "David Thompson",
        "title": "Sales Director",
        "company": "RetailTech Pro",
        "domain": "retailtechpro.com",
        "email": "david.t@retailtechpro.com",
        "location": "Toronto, Canada",
        "industry": "E-commerce - Retail",
        "company_size": "50-100",
        "revenue_band": "5M-10M",
        "tech_stack": ["Shopify", "Klaviyo", "Google Ads"],
        "recent_signals": [
            "Black Friday sales up 200% YoY",
            "Expanding to US market",
            "Hiring SDR team"
        ],
        "pain_points": [
            "Cart abandonment rate >70%",
            "Low repeat purchase rate",
            "Need automated email campaigns"
        ],
        "linkedin_url": "https://linkedin.com/in/davidthompson",
        "qualification_score": 0,
        "priority": "medium",
        "status": "new"
    },
    {
        "lead_id": "L007",
        "name": "Lisa Park",
        "title": "Head of Partnerships",
        "company": "AI StartupHub",
        "domain": "aistartuphub.ai",
        "email": "lisa@aistartuphub.ai",
        "location": "Singapore",
        "industry": "AI/ML - Platform",
        "company_size": "10-50",
        "revenue_band": "<1M",
        "tech_stack": ["Python", "FastAPI", "OpenAI"],
        "recent_signals": [
            "Pre-seed round ($500K) closed",
            "Launched beta product 3 months ago",
            "Featured in TechInAsia"
        ],
        "pain_points": [
            "Need to find enterprise customers",
            "Low brand awareness",
            "Limited sales resources"
        ],
        "linkedin_url": "https://linkedin.com/in/lisapark",
        "qualification_score": 0,
        "priority": "low",
        "status": "new"
    },
    {
        "lead_id": "L008",
        "name": "James Martinez",
        "title": "VP of Sales",
        "company": "HealthCare Solutions",
        "domain": "healthcaresol.com",
        "email": "j.martinez@healthcaresol.com",
        "location": "Boston, USA",
        "industry": "HealthTech - SaaS",
        "company_size": "200-500",
        "revenue_band": "20M-50M",
        "tech_stack": ["Salesforce", "Zendesk", "Twilio"],
        "recent_signals": [
            "Series C funding ($30M) last year",
            "Acquired competitor MedTech Pro",
            "Expanding sales team by 50%"
        ],
        "pain_points": [
            "Long sales cycles (9+ months)",
            "Complex multi-stakeholder deals",
            "Need better pipeline visibility"
        ],
        "linkedin_url": "https://linkedin.com/in/jamesmartinez",
        "qualification_score": 0,
        "priority": "high",
        "status": "new"
    },
]

print(f"âœ… Mock Lead Database loaded: {len(LEADS_DB)} leads")

# =============================================================================
# COMPANY DATABASE - Additional company-level enrichment data ğŸ�¢
# =============================================================================

COMPANY_DB: List[Dict[str, Any]] = [
    {
        "company": "TechVista Solutions",
        "domain": "techvista.io",
        "industry": "SaaS - HR Tech",
        "employee_count": 120,
        "funding_stage": "Series A",
        "total_funding": "$5M",
        "headquarters": "Bengaluru, India",
        "founded_year": 2020,
        "tech_stack": ["React", "Node.js", "AWS", "PostgreSQL"],
        "recent_news": [
            "Raised $5M Series A led by Accel",
            "Launched AI resume screening feature",
            "Named in Forbes '30 Under 30' Asia"
        ],
        "company_summary": "Mid-stage HR Tech SaaS helping companies automate recruitment with AI"
    },
    {
        "company": "GrowthLabs Inc",
        "domain": "growthlabs.co",
        "industry": "Marketing Automation",
        "employee_count": 350,
        "funding_stage": "Series B",
        "total_funding": "$25M",
        "headquarters": "San Francisco, USA",
        "founded_year": 2018,
        "tech_stack": ["Python", "Django", "GCP", "BigQuery"],
        "recent_news": [
            "Series B $15M round from Sequoia",
            "Acquired competitor 'LeadBoost AI'",
            "Expanding to Europe with London office"
        ],
        "company_summary": "Marketing automation platform for scaling personalized campaigns"
    },
    {
        "company": "CloudSync Enterprise",
        "domain": "cloudsync.com",
        "industry": "Cloud Infrastructure",
        "employee_count": 750,
        "funding_stage": "Pre-IPO",
        "total_funding": "$120M",
        "headquarters": "London, UK",
        "founded_year": 2015,
        "tech_stack": ["Kubernetes", "Docker", "Azure", "MongoDB"],
        "recent_news": [
            "Filed for IPO last quarter",
            "Opened offices in Singapore and Dubai",
            "Partnership with Microsoft Azure"
        ],
        "company_summary": "Enterprise cloud infrastructure and data sync platform"
    },
    {
        "company": "FinTech Innovations",
        "domain": "fintechinnovations.in",
        "industry": "FinTech - Digital Payments",
        "employee_count": 180,
        "funding_stage": "Pre-Series A",
        "total_funding": "$2M",
        "headquarters": "Mumbai, India",
        "founded_year": 2021,
        "tech_stack": ["Java", "Spring Boot", "Stripe API", "MySQL"],
        "recent_news": [
            "Featured in TechCrunch for AI fraud detection",
            "Raised $2M from angel investors",
            "Onboarded 500+ merchants in Q1"
        ],
        "company_summary": "Digital payments platform with AI-powered fraud detection"
    },
    {
        "company": "EduTech Global",
        "domain": "edutechglobal.com",
        "industry": "EdTech - Online Learning",
        "employee_count": 35,
        "funding_stage": "Bootstrapped",
        "total_funding": "$0 (Profitable)",
        "headquarters": "Austin, USA",
        "founded_year": 2019,
        "tech_stack": ["Ruby on Rails", "Heroku", "Stripe", "SendGrid"],
        "recent_news": [
            "Won 'Best EdTech Startup' at SXSW",
            "Launched 50+ new courses in 2024",
            "Profitable since Year 2"
        ],
        "company_summary": "Online learning platform for professional upskilling"
    },
]

print(f"âœ… Mock Company Database loaded: {len(COMPANY_DB)} companies")


# =============================================================================
# CUSTOM TOOLS: LeadSearchTool & CompanyEnrichmentTool
# =============================================================================

def LeadSearchTool(icp_criteria: Dict[str, Any]) -> dict:
    """
    Custom Tool: Filters leads based on ICP (Ideal Customer Profile) criteria.
    
    This function is used by the Lead Research Agent to find and qualify leads into:
      - hot (high priority, strong fit)
      - warm (medium priority, good fit)
      - cold (low priority, marginal fit)
    
    Args:
        icp_criteria: Dict containing:
            - industries: List[str] - target industries
            - company_sizes: List[str] - target sizes
            - locations: List[str] - target locations
            - revenue_bands: List[str] - target revenue
            - required_pain_points: List[str] - must-have pain points
            - exclude_tech_stack: List[str] - tech to exclude
    
    Returns:
        dict: {
          "status": "success",
          "hot_leads": [...],
          "warm_leads": [...],
          "cold_leads": [...],
          "total_found": int
        }
    """
    # Extract ICP criteria
    target_industries = icp_criteria.get("industries", [])
    target_sizes = icp_criteria.get("company_sizes", [])
    target_locations = icp_criteria.get("locations", [])
    target_revenue = icp_criteria.get("revenue_bands", [])
    required_pain_points = icp_criteria.get("required_pain_points", [])
    exclude_tech = icp_criteria.get("exclude_tech_stack", [])
    
    hot_leads = []
    warm_leads = []
    cold_leads = []
    
    for lead in LEADS_DB:
        # Skip if already contacted or disqualified
        if lead["status"] in ["contacted", "replied", "dead"]:
            continue
        
        score = 0
        
        # Industry match (30 points)
        if target_industries:
            for industry in target_industries:
                if industry.lower() in lead["industry"].lower():
                    score += 30
                    break
        
        # Company size match (20 points)
        if target_sizes:
            if lead["company_size"] in target_sizes:
                score += 20
        
        # Location match (10 points)
        if target_locations:
            for loc in target_locations:
                if loc.lower() in lead["location"].lower():
                    score += 10
                    break
        
        # Revenue band match (15 points)
        if target_revenue:
            if lead["revenue_band"] in target_revenue:
                score += 15
        
        # Recent signals (10 points)
        if len(lead["recent_signals"]) >= 2:
            score += 10
        
        # Pain point alignment (15 points)
        if required_pain_points:
            matching_pains = sum(
                1 for rp in required_pain_points
                if any(rp.lower() in p.lower() for p in lead["pain_points"])
            )
            score += min(matching_pains * 5, 15)
        
        # Exclude if using competitor tech
        if exclude_tech:
            has_competitor = any(
                tech.lower() in [t.lower() for t in lead["tech_stack"]]
                for tech in exclude_tech
            )
            if has_competitor:
                score -= 20
        
        # Update lead score
        lead["qualification_score"] = score
        
        # Categorize by score
        if score >= 70:
            lead["priority"] = "high"
            hot_leads.append(lead)
        elif score >= 40:
            lead["priority"] = "medium"
            warm_leads.append(lead)
        else:
            lead["priority"] = "low"
            cold_leads.append(lead)
    
    # Sort each category by score (descending)
    hot_leads.sort(key=lambda x: x["qualification_score"], reverse=True)
    warm_leads.sort(key=lambda x: x["qualification_score"], reverse=True)
    cold_leads.sort(key=lambda x: x["qualification_score"], reverse=True)
    
    return {
        "status": "success",
        "hot_leads": hot_leads,
        "warm_leads": warm_leads,
        "cold_leads": cold_leads,
        "total_found": len(hot_leads) + len(warm_leads) + len(cold_leads)
    }


def CompanyEnrichmentTool(domain: str) -> dict:
    """
    Custom Tool: Enriches lead data with company-level information.
    
    This function is used by the Lead Enricher Agent to add context about
    the company (funding, tech stack, recent news, etc.).
    
    Args:
        domain: Company domain (e.g., "techvista.io")
    
    Returns:
        dict: {
          "status": "success" | "not_found",
          "company_data": {...} or None
        }
    """
    for company in COMPANY_DB:
        if company["domain"] == domain:
            return {
                "status": "success",
                "company_data": company
            }
    
    # If not found, return basic mock enrichment
    return {
        "status": "not_found",
        "company_data": {
            "company": domain.split(".")[0].title(),
            "domain": domain,
            "company_summary": f"Company operating at {domain}",
            "industry": "Unknown",
            "employee_count": "N/A",
            "recent_news": []
        }
    }


print("âœ… LeadSearchTool defined.")
print("âœ… CompanyEnrichmentTool defined.")


def CompanyEnrichmentTool(domain: str) -> dict:
    """
    Custom Tool: Enriches lead data with company-level information.
    
    This function is used by the Lead Enricher Agent to add context about
    the company (funding, tech stack, recent news, etc.).
    
    Args:
        domain: Company domain (e.g., "techvista.io")
    
    Returns:
        dict: {
          "status": "success" | "not_found",
          "company_data": {...} or None
        }
    """
    for company in COMPANY_DB:
        if company["domain"] == domain:
            return {
                "status": "success",
                "company_data": company
            }
    
    # If not found, return basic mock enrichment
    return {
        "status": "not_found",
        "company_data": {
            "company": domain.split(".")[0].title(),
            "domain": domain,
            "company_summary": f"Company operating at {domain}",
            "industry": "Unknown",
            "employee_count": "N/A",
            "recent_news": []
        }
    }


print("âœ… CompanyEnrichmentTool defined.")


# =============================================================================
# MEMORY BANK: Long-term storage for leads, campaigns, and outreach history
# =============================================================================

# Lead profile memory (stores enriched lead data + interaction history)
LEAD_MEMORY: Dict[str, Dict[str, Any]] = {}

# Campaign memory (stores campaign configs and performance data)
CAMPAIGN_MEMORY: Dict[str, Dict[str, Any]] = {}

# Outreach history (stores all sent messages and replies)
OUTREACH_MEMORY: Dict[str, List[Dict[str, Any]]] = {}

# CRM Activity Log (stores all interactions: emails, calls, meetings)
ACTIVITY_LOG: List[Dict[str, Any]] = []


# =============================================================================
# LEAD MEMORY TOOLS
# =============================================================================

def save_lead_tool(lead_id: str, lead_data: Dict[str, Any]) -> dict:
    """
    Tool: Save or update a lead's profile in the memory bank.
    
    This stores enriched lead data including:
    - Basic info (name, company, email)
    - Qualification score and priority
    - Interaction history
    - Current status in pipeline
    
    Args:
        lead_id: Unique lead identifier (e.g., "L001")
        lead_data: Complete lead profile dictionary
    
    Returns:
        dict: {"status": "success", "lead_id": "..."}
    """
    # Add timestamp if not present
    if "last_updated" not in lead_data:
        from datetime import datetime
        lead_data["last_updated"] = datetime.now().isoformat()
    
    LEAD_MEMORY[lead_id] = lead_data
    
    return {
        "status": "success",
        "lead_id": lead_id,
        "message": f"Lead {lead_id} saved to memory"
    }


def get_lead_tool(lead_id: str) -> dict:
    """
    Tool: Retrieve a stored lead profile from the memory bank.
    
    Args:
        lead_id: Unique lead identifier
    
    Returns:
        dict: {"status": "success", "lead": {...}} or error
    """
    lead = LEAD_MEMORY.get(lead_id)
    
    if lead is None:
        return {
            "status": "error",
            "error_message": f"No lead found for {lead_id}"
        }
    
    return {
        "status": "success",
        "lead": lead
    }


def update_lead_status_tool(lead_id: str, new_status: str, notes: str = "") -> dict:
    """
    Tool: Update a lead's status in the pipeline.
    
    Valid statuses: new, qualified, contacted, replied, meeting_scheduled, 
                    converted, nurture, dead
    
    Args:
        lead_id: Unique lead identifier
        new_status: New pipeline status
        notes: Optional notes about the status change
    
    Returns:
        dict: {"status": "success", "lead_id": "...", "new_status": "..."}
    """
    if lead_id not in LEAD_MEMORY:
        return {
            "status": "error",
            "error_message": f"Lead {lead_id} not found in memory"
        }
    
    from datetime import datetime
    
    LEAD_MEMORY[lead_id]["status"] = new_status
    LEAD_MEMORY[lead_id]["last_updated"] = datetime.now().isoformat()
    
    if notes:
        if "notes" not in LEAD_MEMORY[lead_id]:
            LEAD_MEMORY[lead_id]["notes"] = []
        LEAD_MEMORY[lead_id]["notes"].append({
            "timestamp": datetime.now().isoformat(),
            "note": notes
        })
    
    return {
        "status": "success",
        "lead_id": lead_id,
        "new_status": new_status
    }


# =============================================================================
# CAMPAIGN MEMORY TOOLS
# =============================================================================

def save_campaign_tool(campaign_id: str, campaign_data: Dict[str, Any]) -> dict:
    """
    Tool: Save or update a campaign configuration in the memory bank.
    
    Campaign data includes:
    - ICP criteria (target industries, sizes, etc.)
    - Outreach templates and variants
    - Send schedule and throttle settings
    - Performance metrics
    
    Args:
        campaign_id: Unique campaign identifier (e.g., "CAMP_001")
        campaign_data: Complete campaign config dictionary
    
    Returns:
        dict: {"status": "success", "campaign_id": "..."}
    """
    from datetime import datetime
    
    if "created_at" not in campaign_data:
        campaign_data["created_at"] = datetime.now().isoformat()
    
    campaign_data["last_updated"] = datetime.now().isoformat()
    
    CAMPAIGN_MEMORY[campaign_id] = campaign_data
    
    return {
        "status": "success",
        "campaign_id": campaign_id,
        "message": f"Campaign {campaign_id} saved to memory"
    }


def get_campaign_tool(campaign_id: str) -> dict:
    """
    Tool: Retrieve a stored campaign configuration from the memory bank.
    
    Args:
        campaign_id: Unique campaign identifier
    
    Returns:
        dict: {"status": "success", "campaign": {...}} or error
    """
    campaign = CAMPAIGN_MEMORY.get(campaign_id)
    
    if campaign is None:
        return {
            "status": "error",
            "error_message": f"No campaign found for {campaign_id}"
        }
    
    return {
        "status": "success",
        "campaign": campaign
    }


# =============================================================================
# OUTREACH HISTORY TOOLS
# =============================================================================

def log_outreach_tool(lead_id: str, outreach_data: Dict[str, Any]) -> dict:
    """
    Tool: Log an outreach message (email/video/voice) to the memory bank.
    
    This creates an audit trail of all messages sent to each lead.
    
    Args:
        lead_id: Unique lead identifier
        outreach_data: Dict containing:
            - channel: "email" | "video" | "voice"
            - subject: Email subject or message title
            - body: Message content
            - variant: "A" | "B" (for A/B testing)
            - sent_at: Timestamp
            - campaign_id: Associated campaign
    
    Returns:
        dict: {"status": "success", "outreach_id": "..."}
    """
    from datetime import datetime
    import uuid
    
    outreach_id = str(uuid.uuid4())[:8]
    
    if "sent_at" not in outreach_data:
        outreach_data["sent_at"] = datetime.now().isoformat()
    
    outreach_data["outreach_id"] = outreach_id
    outreach_data["lead_id"] = lead_id
    
    # Initialize list if lead has no outreach history
    if lead_id not in OUTREACH_MEMORY:
        OUTREACH_MEMORY[lead_id] = []
    
    OUTREACH_MEMORY[lead_id].append(outreach_data)
    
    # Also add to global activity log
    ACTIVITY_LOG.append({
        "activity_type": "outreach_sent",
        "lead_id": lead_id,
        "timestamp": outreach_data["sent_at"],
        "details": outreach_data
    })
    
    return {
        "status": "success",
        "outreach_id": outreach_id,
        "lead_id": lead_id
    }


def get_outreach_history_tool(lead_id: str) -> dict:
    """
    Tool: Retrieve all outreach messages sent to a lead.
    
    Args:
        lead_id: Unique lead identifier
    
    Returns:
        dict: {"status": "success", "history": [...]} or error
    """
    history = OUTREACH_MEMORY.get(lead_id, [])
    
    return {
        "status": "success",
        "lead_id": lead_id,
        "history": history,
        "total_messages": len(history)
    }


def log_reply_tool(lead_id: str, reply_data: Dict[str, Any]) -> dict:
    """
    Tool: Log a reply from a lead (email response, meeting booking, etc.).
    
    Args:
        lead_id: Unique lead identifier
        reply_data: Dict containing:
            - reply_type: "positive" | "neutral" | "negative" | "meeting_booked"
            - content: Reply text or summary
            - received_at: Timestamp
            - sentiment: Sentiment score (optional)
    
    Returns:
        dict: {"status": "success", "reply_id": "..."}
    """
    from datetime import datetime
    import uuid
    
    reply_id = str(uuid.uuid4())[:8]
    
    if "received_at" not in reply_data:
        reply_data["received_at"] = datetime.now().isoformat()
    
    reply_data["reply_id"] = reply_id
    
    # Add to activity log
    ACTIVITY_LOG.append({
        "activity_type": "reply_received",
        "lead_id": lead_id,
        "timestamp": reply_data["received_at"],
        "details": reply_data
    })
    
    # Update lead status based on reply type
    if reply_data.get("reply_type") == "meeting_booked":
        update_lead_status_tool(lead_id, "meeting_scheduled", "Lead booked a meeting!")
    elif reply_data.get("reply_type") == "positive":
        update_lead_status_tool(lead_id, "replied", "Positive reply received")
    
    return {
        "status": "success",
        "reply_id": reply_id,
        "lead_id": lead_id
    }


# =============================================================================
# CRM ACTIVITY LOG TOOLS
# =============================================================================
from typing import Optional
def get_activity_log_tool(lead_id: Optional[str] = None, limit: int = 50) -> dict:
    """
    Tool: Retrieve activity log (all interactions across all leads or for specific lead).
    
    Args:
        lead_id: Optional - filter by specific lead
        limit: Maximum number of activities to return
    
    Returns:
        dict: {"status": "success", "activities": [...]}
    """
    if lead_id:
        activities = [a for a in ACTIVITY_LOG if a.get("lead_id") == lead_id]
    else:
        activities = ACTIVITY_LOG
    
    # Return most recent activities first
    activities_sorted = sorted(
        activities,
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )[:limit]
    
    return {
        "status": "success",
        "activities": activities_sorted,
        "total_count": len(activities_sorted)
    }


def get_campaign_metrics_tool(campaign_id: str) -> dict:
    """
    Tool: Calculate performance metrics for a campaign.
    
    Metrics include:
    - Total leads targeted
    - Total outreach sent
    - Reply rate
    - Meeting booking rate
    - Conversion rate
    
    Args:
        campaign_id: Unique campaign identifier
    
    Returns:
        dict: {"status": "success", "metrics": {...}}
    """
    # Get all activities for this campaign
    campaign_activities = [
        a for a in ACTIVITY_LOG
        if a.get("details", {}).get("campaign_id") == campaign_id
    ]
    
    sent_count = sum(1 for a in campaign_activities if a["activity_type"] == "outreach_sent")
    reply_count = sum(1 for a in campaign_activities if a["activity_type"] == "reply_received")
    meeting_count = sum(
        1 for a in campaign_activities
        if a["activity_type"] == "reply_received"
        and a.get("details", {}).get("reply_type") == "meeting_booked"
    )
    
    reply_rate = (reply_count / sent_count * 100) if sent_count > 0 else 0
    meeting_rate = (meeting_count / sent_count * 100) if sent_count > 0 else 0
    
    return {
        "status": "success",
        "campaign_id": campaign_id,
        "metrics": {
            "total_sent": sent_count,
            "total_replies": reply_count,
            "total_meetings": meeting_count,
            "reply_rate_percent": round(reply_rate, 2),
            "meeting_rate_percent": round(meeting_rate, 2)
        }
    }


print("âœ… Memory Bank tools defined (leads + campaigns + outreach history + CRM log).")


# =============================================================================
# EXAMPLE: Save and retrieve lead data
# =============================================================================

# Save a lead to memory
test_lead = {
    "lead_id": "L001",
    "name": "Priya Sharma",
    "company": "TechVista Solutions",
    "email": "priya.sharma@techvista.io",
    "status": "new",
    "qualification_score": 85,
    "priority": "high"
}

save_result = save_lead_tool("L001", test_lead)
print("\nğŸ’¾ Save Lead Result:")
pretty_print_json(save_result)

# Retrieve the lead
get_result = get_lead_tool("L001")
print("\nğŸ“¥ Retrieved Lead:")
pretty_print_json(get_result)

# Update lead status
update_result = update_lead_status_tool(
    "L001",
    "contacted",
    "Sent initial outreach email (Variant A)"
)
print("\nâœ�ï¸� Updated Lead Status:")
pretty_print_json(update_result)

# =============================================================================
# EXAMPLE: Log outreach and replies
# =============================================================================

# Log an outreach email
outreach_data = {
    "channel": "email",
    "subject": "Quick question about TechVista's hiring automation",
    "body": "Hi Priya, I noticed TechVista recently raised Series A...",
    "variant": "A",
    "campaign_id": "CAMP_001"
}

outreach_result = log_outreach_tool("L001", outreach_data)
print("\nğŸ“§ Logged Outreach:")
pretty_print_json(outreach_result)

# Simulate a positive reply
reply_data = {
    "reply_type": "positive",
    "content": "Hi! Yes, we're definitely interested. Can we schedule a call?",
    "sentiment": 0.85
}

reply_result = log_reply_tool("L001", reply_data)
print("\nâœ‰ï¸� Logged Reply:")
pretty_print_json(reply_result)

# Get outreach history for this lead
history = get_outreach_history_tool("L001")
print("\nğŸ“‹ Outreach History:")
pretty_print_json(history)

# Get activity log
activities = get_activity_log_tool(lead_id="L001")
print("\nğŸ“Š Activity Log:")
pretty_print_json(activities)

# =============================================================================
# EXAMPLE: Campaign tracking
# =============================================================================

# Save a campaign
campaign_data = {
    "campaign_id": "CAMP_001",
    "name": "SaaS Outreach Q4 2024",
    "icp_criteria": {
        "industries": ["SaaS", "FinTech"],
        "company_sizes": ["50-200"]
    },
    "status": "active"
}

campaign_result = save_campaign_tool("CAMP_001", campaign_data)
print("\nğŸ“¢ Saved Campaign:")
pretty_print_json(campaign_result)

# Get campaign metrics
metrics = get_campaign_metrics_tool("CAMP_001")
print("\nğŸ“ˆ Campaign Metrics:")
pretty_print_json(metrics)


# =============================================================================
# AGENT 1: Lead Research Agent - EXTRA STRICT
# =============================================================================

lead_research_agent = LlmAgent(
    name="LeadResearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Lead Research Agent for KeyXGrowth AI.

YOU HAVE EXACTLY 2 TOOLS AVAILABLE:
1. LeadSearchTool - searches for leads
2. save_lead_tool - saves leads to memory

DO NOT CALL ANY OTHER TOOLS. DO NOT INVENT TOOLS.

You will receive campaign configuration with ICP criteria.

STEP 1: Call LeadSearchTool(icp_criteria) to discover leads.
STEP 2: For each discovered lead, call save_lead_tool(lead_id, lead_data) to store them.
STEP 3: Return ONLY this JSON (no markdown, no code blocks):
{
  "campaign_id": "...",
  "hot_leads": [ { "lead_id": "...", "name": "...", "company": "...", "score": ... }, ... ],
  "warm_leads": [ ... ],
  "cold_leads": [ ... ],
  "total_found": 10,
  "status": "success"
}

CRITICAL RULES:
- ONLY use LeadSearchTool and save_lead_tool
- DO NOT call send_email_tool, email_tool, or any other tool
- DO NOT include markdown backticks or headers
- Return pure JSON only
""",
    tools=[LeadSearchTool, save_lead_tool],
)
print("âœ… LeadResearchAgent created.")


# =============================================================================
# AGENT 2: Lead Enricher Agent (Parallel Processing)
# =============================================================================

lead_enricher_agent = LlmAgent(
    name="LeadEnricherAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Lead Enricher Agent for KeyXGrowth AI.

You will receive a list of lead_ids to enrich.

For EACH lead:
STEP 1: Call get_lead_tool(lead_id) to retrieve the lead data.
STEP 2: Extract the company domain from the lead.
STEP 3: Call CompanyEnrichmentTool(domain) to get company data (funding, news, tech stack).
STEP 4: Merge the enrichment data into the lead profile:
   - Add company_summary
   - Add recent_news
   - Add decision_makers
   - Generate personalization_hooks (e.g., "recent Series A funding", "expanding to EMEA")
STEP 5: Call save_lead_tool(lead_id, enriched_lead_data) to update memory.

Return ONLY this JSON:
{
  "enriched_leads": [
    {
      "lead_id": "...",
      "name": "...",
      "company": "...",
      "personalization_hooks": ["hook1", "hook2"],
      "company_summary": "..."
    },
    ...
  ],
  "total_enriched": 5,
  "status": "success"
}

IMPORTANT:
- Process ALL leads provided.
- The final answer MUST be ONLY valid JSON.
- Do NOT include markdown or extra text.
""",
    tools=[get_lead_tool, CompanyEnrichmentTool, save_lead_tool],
)
print("âœ… LeadEnricherAgent created.")


# =============================================================================
# AGENT 3: Qualification Agent (BANT Scoring)
# =============================================================================

qualification_agent = LlmAgent(
    name="QualificationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Qualification Agent for KeyXGrowth AI.

You will receive a list of enriched lead_ids to qualify using the BANT framework:
- **B**udget: Does company size/funding suggest they can afford the solution?
- **A**uthority: Is the contact a decision-maker (VP, Director, C-level)?
- **N**eed: Do their pain points align with our solution?
- **T**iming: Do recent signals indicate buying intent?

For EACH lead:
STEP 1: Call get_lead_tool(lead_id) to retrieve enriched data.
STEP 2: Analyze the lead using BANT criteria:
   - Budget Score (0-25): Based on company_size, revenue_band, funding_stage
   - Authority Score (0-25): Based on title (VP/Director/C-level = high, Manager = medium, IC = low)
   - Need Score (0-25): Based on pain_points alignment with our solution
   - Timing Score (0-25): Based on recent_signals (funding, hiring, product launches)
STEP 3: Calculate total qualification_score (0-100).
STEP 4: Determine priority:
   - score >= 70 â†’ priority = "high", status = "qualified"
   - score >= 40 â†’ priority = "medium", status = "qualified"
   - score < 40 â†’ priority = "low", status = "nurture"
STEP 5: Call update_lead_status_tool(lead_id, status, notes) with qualification reasoning.

Return ONLY this JSON:
{
  "qualified_leads": [
    {
      "lead_id": "...",
      "name": "...",
      "company": "...",
      "qualification_score": 85,
      "priority": "high",
      "bant_breakdown": {
        "budget": 20,
        "authority": 25,
        "need": 20,
        "timing": 20
      },
      "reasoning": "Strong fit: C-level contact, recent Series A funding, pain points align perfectly."
    },
    ...
  ],
  "total_qualified": 5,
  "status": "success"
}

IMPORTANT:
- Be analytical and objective in scoring.
- The final answer MUST be ONLY valid JSON.
""",
    tools=[get_lead_tool, update_lead_status_tool],
)
print("âœ… QualificationAgent created.")


# =============================================================================
# AGENT 4: Email Writer Agent (A/B Variants)
# =============================================================================

email_writer_agent = LlmAgent(
    name="EmailWriterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Email Writer Agent for KeyXGrowth AI.

You will receive a qualified lead_id and campaign_id.

STEP 1: Call get_lead_tool(lead_id) to retrieve lead data.
STEP 2: Generate TWO email variants (A and B) with these requirements:
   - Subject line: 6-8 words, personalized with company name or pain point
   - Body: 120-150 words max
   - Opening: Personalized hook using personalization_hooks or recent_signals
   - Middle: One-sentence value proposition addressing their pain_points
   - CTA: Single clear call-to-action (e.g., "Can we schedule a 15-minute call next week?")
   - Tone: Professional but conversational, not salesy
   
   Variant A: Problem-focused approach (lead with pain point)
   Variant B: Opportunity-focused approach (lead with success story or benefit)

STEP 3: For EACH variant, call log_outreach_tool(lead_id, outreach_data) where:
   outreach_data = {
     "channel": "email",
     "subject": "...",
     "body": "...",
     "variant": "A" or "B",
     "campaign_id": "..."
   }

Return ONLY this JSON:
{
  "lead_id": "...",
  "lead_name": "...",
  "emails": [
    {
      "variant": "A",
      "subject": "...",
      "body": "..."
    },
    {
      "variant": "B",
      "subject": "...",
      "body": "..."
    }
  ],
  "status": "success"
}

IMPORTANT:
- Keep emails concise and highly personalized.
- Use actual data from the lead profile (name, company, pain points, signals).
- The final answer MUST be ONLY valid JSON.
""",
    tools=[get_lead_tool, log_outreach_tool],
)
print("âœ… EmailWriterAgent created.")


# =============================================================================
# AGENT 5: Video Script Agent (Optional - Multi-Channel)
# =============================================================================

video_script_agent = LlmAgent(
    name="VideoScriptAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Video Script Agent for KeyXGrowth AI.

You will receive a qualified lead_id and campaign_id.

STEP 1: Call get_lead_tool(lead_id) to retrieve lead data.
STEP 2: Generate a 30-45 second personalized video script with:
   - [0-5s] Opening: "Hi [Name], I noticed [company] recently [signal]..."
   - [5-25s] Value Prop: Address their specific pain point and our solution
   - [25-35s] Social Proof: Brief mention of similar company success
   - [35-45s] CTA: "I'd love to show you a quick demo. Can we chat next week?"
   - Include timestamps for editing

STEP 3: Call log_outreach_tool(lead_id, outreach_data) where:
   outreach_data = {
     "channel": "video",
     "subject": "Personalized video for [Company]",
     "body": "<full script with timestamps>",
     "variant": "video_v1",
     "campaign_id": "..."
   }

Return ONLY this JSON:
{
  "lead_id": "...",
  "video_script": "...",
  "estimated_duration": "40 seconds",
  "status": "success"
}

IMPORTANT:
- Script must sound natural when spoken.
- Include specific details from lead profile.
- The final answer MUST be ONLY valid JSON.
""",
    tools=[get_lead_tool, log_outreach_tool],
)
print("âœ… VideoScriptAgent created.")

# =============================================================================
# AGENT 6: Sender & Scheduler Agent (Mock Delivery) - FIXED
# =============================================================================

sender_agent = LlmAgent(
    name="SenderAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Sender & Scheduler Agent for KeyXGrowth AI.

You will receive:
- lead_id: Lead to send outreach to
- campaign_id: Campaign this belongs to
- variant: Which email variant to send ("A" or "B")

YOU HAVE EXACTLY 2 TOOLS AVAILABLE:
1. get_outreach_history_tool - retrieves messages
2. update_lead_status_tool - updates lead status

DO NOT CALL ANY OTHER TOOLS. DO NOT INVENT TOOLS.

STEP 1: Call get_outreach_history_tool(lead_id) to retrieve all drafted messages.
STEP 2: Find the email matching the specified variant from the history.
STEP 3: Note that the email was sent (we simulate sending - NO actual email tool exists).
STEP 4: Call update_lead_status_tool(lead_id, "contacted", "Email variant [X] sent on [date]")

Return ONLY this JSON (no markdown, no code blocks):
{
  "lead_id": "...",
  "lead_email": "...",
  "variant_sent": "A",
  "subject": "...",
  "sent_at": "2024-11-24T10:30:00",
  "delivery_status": "simulated_success",
  "status": "success"
}

CRITICAL RULES:
- ONLY use get_outreach_history_tool and update_lead_status_tool
- DO NOT call send_email_tool, email_tool, or any other tool
- DO NOT include markdown backticks or headers
- Return pure JSON only
""",
    tools=[get_outreach_history_tool, update_lead_status_tool],
)
print("âœ… SenderAgent created.")


# =============================================================================
# AGENT 7: Follow-Up Loop Agent (Reply Monitoring)
# =============================================================================

followup_agent = LlmAgent(
    name="FollowUpAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Follow-Up Agent for KeyXGrowth AI. You handle reply monitoring.

You will receive:
- lead_id: Lead to check for replies
- simulated_reply: A mock reply text to classify

STEP 1: Call get_lead_tool(lead_id) to retrieve lead context.
STEP 2: Call get_outreach_history_tool(lead_id) to see past messages.
STEP 3: Classify the reply:
   - **positive**: Interest expressed ("Yes", "Tell me more", "Let's talk", "Schedule a call")
   - **neutral**: Non-committal ("Send info", "Maybe later", "Check back in Q2")
   - **negative**: Rejection ("Not interested", "Unsubscribe", "Wrong person")
   - **out_of_office**: Auto-reply detected

STEP 4: Take action based on classification:
   - positive â†’ Call log_reply_tool(lead_id, {reply_type: "positive", ...})
              â†’ Update status to "meeting_scheduled" 
   - neutral â†’ Schedule next follow-up in 3 days
   - negative â†’ Update status to "dead", stop outreach
   - out_of_office â†’ Reschedule for 1 week later

Return ONLY this JSON:
{
  "lead_id": "...",
  "reply_classification": "positive",
  "sentiment_score": 0.85,
  "recommended_action": "Send calendar invite",
  "next_follow_up": null or "2024-11-25",
  "status": "success"
}

IMPORTANT:
- The final answer MUST be ONLY valid JSON.
- Do NOT include markdown, headings, or extra text.
""",
    tools=[get_lead_tool, get_outreach_history_tool, log_reply_tool, update_lead_status_tool],
)
print("âœ… FollowUpAgent created.")


# =============================================================================
# AGENT 8: Evaluator Agent (Campaign Analytics)
# =============================================================================

evaluator_agent = LlmAgent(
    name="EvaluatorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Evaluator Agent for KeyXGrowth AI and **FINAL REPORTER**.

You will receive a campaign_id to analyze.

STEP 1: Call get_campaign_tool(campaign_id) to retrieve campaign config.
STEP 2: Call get_campaign_metrics_tool(campaign_id) to get performance data.
STEP 3: Analyze the metrics and generate insights:
   - Which variant (A or B) performed better?
   - What was the reply rate vs industry benchmark (typically 2-5%)?
   - Which pain points or personalization hooks resonated most?
   - What should be optimized for next campaign?

STEP 4: Generate a comprehensive **Markdown report** including:
   - Campaign Overview (name, ICP, dates)
   - Key Metrics (sent, replies, meetings, conversion rate)
   - A/B Test Results (if applicable)
   - Top Performing Leads (by score and response)
   - Recommendations for Optimization

**IMPORTANT OUTPUT RULE:**
- **DO NOT** return JSON or code fences.
- The final response MUST be a single, **human-readable Markdown report**.

**Report Structure:**
# ğŸ“Š KeyXGrowth AI - Campaign Performance Report

## Campaign: [Name]
- **Campaign ID:** ...
- **Date Range:** ...
- **Target ICP:** ...

## ğŸ“ˆ Key Metrics
- **Total Leads Targeted:** ...
- **Outreach Sent:** ...
- **Reply Rate:** ...% 
- **Meeting Booked Rate:** ...%
- **Status Breakdown:** X qualified, Y contacted, Z replied

## ğŸ”¬ A/B Test Results (if applicable)
| Variant | Sent | Replies | Reply Rate |
|---------|------|---------|------------|
| A       | 25   | 3       | 12%        |
| B       | 25   | 6       | 24%        |

**Winner:** Variant B outperformed by 2x

## ğŸ’¡ Insights & Recommendations
1. **What Worked:** ...
2. **What Didn't:** ...
3. **Next Steps:** ...
""",
    tools=[get_campaign_tool, get_campaign_metrics_tool, get_activity_log_tool],
)
print("âœ… EvaluatorAgent created (Final Reporter).")


# =============================================================================
# ROOT SEQUENTIAL MULTI-AGENT SYSTEM
# =============================================================================

root_agent = SequentialAgent(
    name="KeyXGrowthPipeline",
    sub_agents=[
        lead_research_agent,
        lead_enricher_agent,
        qualification_agent,
        email_writer_agent,
        # video_script_agent,  # Optional - uncomment for multi-channel
        sender_agent,
        evaluator_agent,  # Final reporter
    ],
)
print("âœ… Root SequentialAgent (KeyXGrowthPipeline) created.")

# Main pipeline runner
runner = InMemoryRunner(root_agent)
print("âœ… InMemoryRunner created for KeyXGrowthPipeline.")

# Separate runner for the Follow-Up Loop Agent (long-running)
followup_runner = InMemoryRunner(followup_agent)
print("âœ… InMemoryRunner created for FollowUpAgent (Loop).")


# =============================================================================
# UNIVERSAL DEBUG PRINTER
# =============================================================================

def debug_print_events(events):
    """
    Pretty-prints agent execution events for debugging.
    Shows which agent ran and what it output.
    """
    print("\n==================== AGENT EXECUTION LOG ====================\n")
    for turn in events:
        who = getattr(turn, "source", "Unknown")
        print(f"ğŸ¤– {who} >\n")
        
        text_found = False
        
        # Check various text fields
        if (
            hasattr(turn, "content")
            and turn.content is not None
            and hasattr(turn.content, "text")
            and turn.content.text
        ):
            print(turn.content.text)
            text_found = True
        
        if (
            hasattr(turn, "content")
            and turn.content is not None
            and hasattr(turn.content, "parts")
            and turn.content.parts is not None
        ):
            for p in turn.content.parts:
                if hasattr(p, "text") and p.text:
                    print(p.text)
                    text_found = True
        
        if hasattr(turn, "delta") and turn.delta is not None:
            if hasattr(turn.delta, "text") and turn.delta.text:
                print(turn.delta.text)
                text_found = True
        
        if hasattr(turn, "message") and isinstance(turn.message, str):
            print(turn.message)
            text_found = True
        
        if not text_found:
            print("[No text output]")
        
        print("\n" + "-" * 60 + "\n")

print("âœ… Debug printer ready.")


# ============================================================
# DEBUG: Check which agents have which tools
# ============================================================

print("=" * 70)
print("ğŸ”� AGENT TOOLS AUDIT")
print("=" * 70 + "\n")

agents_to_check = [
    ("LeadResearchAgent", lead_research_agent),
    ("LeadEnricherAgent", lead_enricher_agent),
    ("QualificationAgent", qualification_agent),
    ("EmailWriterAgent", email_writer_agent),
    ("SenderAgent", sender_agent),
    ("EvaluatorAgent", evaluator_agent),
    ("FollowUpAgent", followup_agent),
]

for agent_name, agent in agents_to_check:
    print(f"\n{agent_name}:")
    if hasattr(agent, 'tools') and agent.tools:
        for tool in agent.tools:
            # Get tool name
            if hasattr(tool, '__name__'):
                tool_name = tool.__name__
            elif hasattr(tool, 'name'):
                tool_name = tool.name
            else:
                tool_name = str(tool)
            print(f"  âœ… {tool_name}")
    else:
        print(f"  â�Œ No tools registered!")

print("\n" + "=" * 70)
print("âœ… Audit complete - check for missing tools above")
print("=" * 70 + "\n")


# Clear all memory before demo
LEAD_MEMORY.clear()
CAMPAIGN_MEMORY.clear()
OUTREACH_MEMORY.clear()
ACTIVITY_LOG.clear()
print("âœ… Memory cleared for fresh demo run")


campaign_config = """
CAMPAIGN_ID: CAMP_Q4_2024
CAMPAIGN_NAME: SaaS Outreach - Q4 2024

ICP_CRITERIA:
{
  "industries": ["SaaS", "FinTech", "Marketing Automation"],
  "company_sizes": ["50-200", "200-500"],
  "locations": ["USA", "India"],
  "revenue_bands": ["1M-10M", "10M-50M"],
  "required_pain_points": ["automation", "scaling", "manual processes"],
  "exclude_tech_stack": []
}

CAMPAIGN_SETTINGS:
{
  "daily_send_limit": 50,
  "email_variant": "A",
  "follow_up_enabled": true,
  "send_schedule": "business_hours_only"
}
"""


import asyncio

campaign_config = """
CAMPAIGN_ID: CAMP_Q4_2024
CAMPAIGN_NAME: SaaS Outreach - Q4 2024

ICP_CRITERIA:
{
  "industries": ["SaaS", "FinTech", "Marketing Automation"],
  "company_sizes": ["50-200", "200-500"],
  "locations": ["USA", "India"],
  "revenue_bands": ["1M-10M", "10M-50M"],
  "required_pain_points": ["automation", "scaling", "manual processes"],
  "exclude_tech_stack": []
}

CAMPAIGN_SETTINGS:
{
  "daily_send_limit": 50,
  "email_variant": "A",
  "follow_up_enabled": true,
  "send_schedule": "business_hours_only"
}

TARGET_LEAD_IDS: ["L001", "L002", "L004"]
"""

prompt = f"""
{campaign_config}

Execute the full outreach pipeline:
1. Research and qualify leads based on ICP criteria
2. Enrich leads with company data
3. Score and prioritize using BANT framework
4. Generate personalized email variants (A/B)
5. Simulate sending outreach
6. Generate final campaign performance report
"""

LEAD_MEMORY.clear()
CAMPAIGN_MEMORY.clear()
OUTREACH_MEMORY.clear()
ACTIVITY_LOG.clear()

print("âœ… Memory reset for new campaign run.")

campaign_data = {
    "campaign_id": "CAMP_Q4_2024",
    "name": "SaaS Outreach - Q4 2024",
    "icp_criteria": {
        "industries": ["SaaS", "FinTech", "Marketing Automation"],
        "company_sizes": ["50-200", "200-500"],
        "locations": ["USA", "India"],
        "revenue_bands": ["1M-10M", "10M-50M"],
        "required_pain_points": ["automation", "scaling", "manual processes"],
        "exclude_tech_stack": []
    },
    "settings": {
        "daily_send_limit": 50,
        "email_variant": "A",
        "follow_up_enabled": True,
        "send_schedule": "business_hours_only"
    },
    "status": "active"
}

save_campaign_tool("CAMP_Q4_2024", campaign_data)
print("âœ… Campaign configuration saved to memory.")

print("\nğŸš€ Starting KeyXGrowth AI Pipeline...\n")
response = await runner.run_debug(prompt)

print("\n==================== AGENT EXECUTION SUMMARY ====================\n")

for i, turn in enumerate(response):
    who = getattr(turn, "source", f"Turn {i}")
    
    text = None
    if hasattr(turn, "content") and turn.content is not None:
        if hasattr(turn.content, "text") and turn.content.text:
            text = turn.content.text
        elif hasattr(turn.content, "parts") and turn.content.parts is not None:
            texts = [p.text for p in turn.content.parts if hasattr(p, "text") and p.text]
            if texts:
                text = "\n".join(texts)
    
    if text and who in ["LeadResearchAgent", "QualificationAgent", "EmailWriterAgent", "EvaluatorAgent"]:
        print(f"--- OUTPUT FROM: {who} ---")
        print(text.strip())
        print("\n" + "=" * 70 + "\n")

print("\n===== MEMORY VERIFICATION =====\n")

if LEAD_MEMORY:
    print(f"âœ… Leads successfully saved to memory: {len(LEAD_MEMORY)} leads")
    print("\nSample Lead (L001):")
    if "L001" in LEAD_MEMORY:
        pretty_print_json(LEAD_MEMORY["L001"])
else:
    print("âš ï¸� No leads found in LEAD_MEMORY.")

if OUTREACH_MEMORY:
    print(f"\nâœ… Outreach messages logged: {sum(len(v) for v in OUTREACH_MEMORY.values())} total messages")
    print("\nSample Outreach History (L001):")
    if "L001" in OUTREACH_MEMORY:
        pretty_print_json(OUTREACH_MEMORY["L001"][:2])
else:
    print("âš ï¸� No outreach messages found in OUTREACH_MEMORY.")

if ACTIVITY_LOG:
    print(f"\nâœ… Activity log contains {len(ACTIVITY_LOG)} events")
    print("\nRecent Activities (last 3):")
    for activity in ACTIVITY_LOG[-3:]:
        print(f"  - {activity['activity_type']} | Lead: {activity.get('lead_id', 'N/A')} | {activity['timestamp']}")
else:
    print("âš ï¸� Activity log is empty.")

print("\n===== CAMPAIGN METRICS =====\n")
metrics_result = get_campaign_metrics_tool("CAMP_Q4_2024")
if metrics_result["status"] == "success":
    print("âœ… Campaign metrics calculated:")
    pretty_print_json(metrics_result["metrics"])
else:
    print("âš ï¸� Unable to calculate campaign metrics.")

print("\n===== FINAL REPORT CONFIRMATION =====\n")

final_turn = response[-1]
final_source = getattr(final_turn, "source", "Unknown")

if final_source == "EvaluatorAgent":
    print("âœ… Final campaign report generated by EvaluatorAgent.")
    print("Check the AGENT EXECUTION SUMMARY above for the full Markdown report.")
else:
    print(f"âš ï¸� Expected EvaluatorAgent as final output, but got: {final_source}")
    print("\n--- Last Agent Response (for debugging) ---")
    if hasattr(final_turn, "content") and final_turn.content:
        print(final_turn.content)

print("\n" + "=" * 70)
print("âœ… PIPELINE EXECUTION COMPLETE")
print("=" * 70 + "\n")


# ============================================================
# REPLY SIMULATION: FOLLOW-UP AGENT LOOP RUN
# ============================================================

import asyncio

# Customize this to test different reply scenarios!
simulated_reply = """
Hi there!

Thanks for reaching out. We're definitely interested in automating our outreach process. 
Our current manual approach is taking up way too much time, and your solution sounds 
like exactly what we need.

Can we schedule a call next week to discuss this further? I'm available Tuesday or 
Thursday afternoon.

Best,
Priya Sharma
Head of Sales Operations, TechVista Solutions
"""

followup_prompt = json.dumps(
    {
        "lead_id": "L001",
        "campaign_id": "CAMP_Q4_2024",
        "simulated_reply": simulated_reply,
    },
    indent=2,
)

print("\nğŸ”„ Running Follow-Up Agent to analyze reply...\n")

# Run the Follow-Up Agent
followup_response = await followup_runner.run_debug(followup_prompt)

# Wait for response with timeout
MAX_WAIT_TIME = 10
sleep_interval = 1
elapsed_time = 0

while elapsed_time < MAX_WAIT_TIME:
    if hasattr(followup_response[-1], 'content') and followup_response[-1].content:
        if hasattr(followup_response[-1].content, 'text') and followup_response[-1].content.text:
            break
        elif hasattr(followup_response[-1].content, 'parts') and followup_response[-1].content.parts:
            if any(p.text for p in followup_response[-1].content.parts if hasattr(p, 'text')):
                break
    
    followup_response = await followup_runner.run_debug(followup_prompt)
    await asyncio.sleep(sleep_interval)
    elapsed_time += sleep_interval
    print(f"[Wait: {elapsed_time}s] Waiting for FollowUpAgent output...")

# Extract final text
followup_last = followup_response[-1]
followup_text = None

if hasattr(followup_last, "content") and followup_last.content is not None:
    if hasattr(followup_last.content, "text") and followup_last.content.text:
        followup_text = followup_last.content.text
    elif hasattr(followup_last.content, "parts") and followup_last.content.parts is not None:
        texts = [p.text for p in followup_last.content.parts if hasattr(p, "text") and p.text]
        if texts:
            followup_text = "\n".join(texts)

print("\n===== FOLLOW-UP AGENT ANALYSIS =====\n")

# SAFETY CHECK: Only proceed if we got text
if followup_text:
    print(followup_text)
    
    # Try to parse JSON
    try:
        followup_json = json.loads(followup_text)
        print("\nâœ… Successfully parsed Follow-Up Agent response:")
        pretty_print_json(followup_json)
        
        # Display key insights
        print("\nğŸ“Š Key Insights:")
        print(f"  â€¢ Reply Classification: {followup_json.get('reply_classification', 'N/A')}")
        print(f"  â€¢ Sentiment Score: {followup_json.get('sentiment_score', 'N/A')}")
        print(f"  â€¢ Recommended Action: {followup_json.get('recommended_action', 'N/A')}")
        print(f"  â€¢ Next Follow-Up: {followup_json.get('next_follow_up', 'None (meeting scheduled)')}")
        
    except json.JSONDecodeError as e:
        print("âš ï¸� Could not parse JSON from Follow-Up Agent response.")
        print(f"Error: {e}")
        print("\nRaw response (first 500 chars):")
        print(followup_text[:500])
else:
    print("â�Œ No response received from Follow-Up Agent.")
    print("\nDebugging info:")
    print(f"  â€¢ Response has {len(followup_response)} turns")
    print(f"  â€¢ Last turn source: {getattr(followup_response[-1], 'source', 'Unknown')}")
    print(f"  â€¢ Last turn has content: {hasattr(followup_response[-1], 'content')}")
    
    # Show full last turn for debugging
    print("\n--- Full Last Turn ---")
    print(followup_response[-1])

print("\nâœ… FollowUpAgent demo complete.")


# ============================================================
# FINAL OUTPUT SUMMARY AND CAMPAIGN REPORT GENERATION
# ============================================================

import json

print("\n" + "=" * 70)
print("ğŸ”„ Regenerating complete campaign report...")
print("=" * 70 + "\n")

# Re-run the full pipeline to get fresh output
campaign_config = """
CAMPAIGN_ID: CAMP_Q4_2024
CAMPAIGN_NAME: SaaS Outreach - Q4 2024

ICP_CRITERIA:
{
  "industries": ["SaaS", "FinTech", "Marketing Automation"],
  "company_sizes": ["50-200", "200-500"],
  "locations": ["USA", "India"],
  "revenue_bands": ["1M-10M", "10M-50M"],
  "required_pain_points": ["automation", "scaling", "manual processes"],
  "exclude_tech_stack": []
}

CAMPAIGN_SETTINGS:
{
  "daily_send_limit": 50,
  "email_variant": "A",
  "follow_up_enabled": true
}

TARGET_LEAD_IDS: ["L001", "L002", "L004"]
"""

prompt = f"""
{campaign_config}

Execute the full outreach pipeline and generate the final campaign performance report.
"""

# Run the pipeline fully again
demo_response = await runner.run_debug(prompt)

# Extract final turn (EvaluatorAgent should be last)
last_turn = demo_response[-1]
final_text = None

# Safe text extraction
if hasattr(last_turn, "content") and last_turn.content:
    if hasattr(last_turn.content, "text") and last_turn.content.text:
        final_text = last_turn.content.text
    elif hasattr(last_turn.content, "parts") and last_turn.content.parts:
        pieces = [p.text for p in last_turn.content.parts if hasattr(p, "text") and p.text]
        if pieces:
            final_text = "\n".join(pieces)

# Fallback
if final_text is None:
    final_text = "[No final text produced by EvaluatorAgent]"

print("\n" + "=" * 70)
print("ğŸ“Š FINAL CAMPAIGN PERFORMANCE REPORT")
print("=" * 70 + "\n")
print(final_text)

# ============================================================
# CREATE SUBMISSION PACKAGE
# ============================================================

print("\n" + "=" * 70)
print("ğŸ“¦ Creating submission package...")
print("=" * 70 + "\n")

# Compile all campaign data into a structured JSON
submission_data = {
    "campaign_summary": {
        "campaign_id": "CAMP_Q4_2024",
        "campaign_name": "SaaS Outreach - Q4 2024",
        "execution_date": "2024-11-22",
        "pipeline_status": "completed"
    },
    
    "leads_processed": {
        "total_leads_in_memory": len(LEAD_MEMORY),
        "leads": [
            {
                "lead_id": lead_id,
                "name": data.get("name"),
                "company": data.get("company"),
                "status": data.get("status"),
                "qualification_score": data.get("qualification_score", 0),
                "priority": data.get("priority")
            }
            for lead_id, data in LEAD_MEMORY.items()
        ]
    },
    
    "outreach_summary": {
        "total_messages_sent": sum(len(messages) for messages in OUTREACH_MEMORY.values()),
        "leads_contacted": len(OUTREACH_MEMORY),
        "messages_by_lead": {
            lead_id: len(messages) 
            for lead_id, messages in OUTREACH_MEMORY.items()
        }
    },
    
    "activity_log": {
        "total_activities": len(ACTIVITY_LOG),
        "activities_by_type": {},
        "recent_activities": ACTIVITY_LOG[-10:]  # Last 10 activities
    },
    
    "campaign_metrics": get_campaign_metrics_tool("CAMP_Q4_2024"),
    
    "final_report_markdown": final_text
}

# Count activities by type
for activity in ACTIVITY_LOG:
    activity_type = activity.get("activity_type", "unknown")
    submission_data["activity_log"]["activities_by_type"][activity_type] = \
        submission_data["activity_log"]["activities_by_type"].get(activity_type, 0) + 1

# Save main output file
output_filename = "keyxgrowth_campaign_report.json"
with open(output_filename, "w") as f:
    json.dump(submission_data, f, indent=2, ensure_ascii=False)

print(f"âœ… Saved campaign report: {output_filename}")

# Save Markdown report separately for easy reading
markdown_filename = "keyxgrowth_campaign_report.md"
with open(markdown_filename, "w") as f:
    f.write(final_text)

print(f"âœ… Saved Markdown report: {markdown_filename}")

# Save sample emails for review
if OUTREACH_MEMORY:
    sample_emails = []
    for lead_id, messages in OUTREACH_MEMORY.items():
        for msg in messages[:2]:  # First 2 messages per lead
            if msg.get("channel") == "email":
                sample_emails.append({
                    "lead_id": lead_id,
                    "lead_name": LEAD_MEMORY.get(lead_id, {}).get("name", "Unknown"),
                    "variant": msg.get("variant"),
                    "subject": msg.get("subject"),
                    "body": msg.get("body")
                })
    
    emails_filename = "sample_outreach_emails.json"
    with open(emails_filename, "w") as f:
        json.dump({"sample_emails": sample_emails}, f, indent=2, ensure_ascii=False)
    
    print(f"âœ… Saved sample emails: {emails_filename}")

# Print submission summary
print("\n" + "=" * 70)
print("ğŸ�‰ SUBMISSION PACKAGE COMPLETE")
print("=" * 70)
print(f"""
Files created:
1. {output_filename} - Complete campaign data (JSON)
2. {markdown_filename} - Human-readable report (Markdown)
3. {emails_filename} - Sample outreach emails (JSON)

Campaign Statistics:
- Leads Processed: {len(LEAD_MEMORY)}
- Messages Sent: {sum(len(messages) for messages in OUTREACH_MEMORY.values())}
- Activities Logged: {len(ACTIVITY_LOG)}
- Campaign Status: Complete

ğŸ“¤ You can now SUBMIT these files to the competition.
""")

print("=" * 70 + "\n")

# Display final metrics summary
print("ğŸ“ˆ FINAL METRICS SUMMARY\n")
metrics = get_campaign_metrics_tool("CAMP_Q4_2024")
if metrics["status"] == "success":
    pretty_print_json(metrics["metrics"])
else:
    print("âš ï¸� Metrics calculation failed")

print("\nâœ… KeyXGrowth AI Demo Complete!")


# ============================================================
# FINAL OUTPUT SUMMARY AND CAMPAIGN REPORT GENERATION
# ============================================================

import json

print("\n" + "=" * 70)
print("ğŸ“¦ Creating Submission Package")
print("=" * 70 + "\n")

# Compile all campaign data into a structured JSON
submission_data = {
    "project_info": {
        "name": "KeyXGrowth AI",
        "track": "Enterprise Agents",
        "author": "Your Name",  # UPDATE THIS
        "submission_date": "2024-11-24",
        "notebook_url": "https://www.kaggle.com/code/yourname/keyxgrowth-ai"  # UPDATE THIS
    },
    
    "campaign_summary": {
        "campaign_id": "CAMP_Q4_2024",
        "campaign_name": "SaaS Outreach - Q4 2024",
        "execution_date": "2024-11-24",
        "pipeline_status": "completed"
    },
    
    "leads_processed": {
        "total_leads_in_memory": len(LEAD_MEMORY),
        "leads": [
            {
                "lead_id": lead_id,
                "name": data.get("name"),
                "company": data.get("company"),
                "status": data.get("status"),
                "qualification_score": data.get("qualification_score", 0),
                "priority": data.get("priority")
            }
            for lead_id, data in LEAD_MEMORY.items()
        ]
    },
    
    "outreach_summary": {
        "total_messages_sent": sum(len(messages) for messages in OUTREACH_MEMORY.values()),
        "leads_contacted": len(OUTREACH_MEMORY),
        "messages_by_channel": {
            "email": sum(1 for messages in OUTREACH_MEMORY.values() 
                        for msg in messages if msg.get("channel") == "email")
        }
    },
    
    "activity_log": {
        "total_activities": len(ACTIVITY_LOG),
        "recent_activities": ACTIVITY_LOG[-10:] if ACTIVITY_LOG else []
    },
    
    "campaign_metrics": get_campaign_metrics_tool("CAMP_Q4_2024")
}

# Save main output file
output_filename = "keyxgrowth_campaign_report.json"
with open(output_filename, "w") as f:
    json.dump(submission_data, f, indent=2, ensure_ascii=False)

print(f"âœ… Saved campaign report: {output_filename}")

# Save sample emails
if OUTREACH_MEMORY:
    sample_emails = []
    for lead_id, messages in OUTREACH_MEMORY.items():
        for msg in messages[:2]:
            if msg.get("channel") == "email":
                sample_emails.append({
                    "lead_id": lead_id,
                    "lead_name": LEAD_MEMORY.get(lead_id, {}).get("name", "Unknown"),
                    "variant": msg.get("variant"),
                    "subject": msg.get("subject"),
                    "body": msg.get("body")
                })
    
    emails_filename = "sample_outreach_emails.json"
    with open(emails_filename, "w") as f:
        json.dump({"sample_emails": sample_emails}, f, indent=2, ensure_ascii=False)
    
    print(f"âœ… Saved sample emails: {emails_filename}")

print("\n" + "=" * 70)
print("ğŸ�‰ SUBMISSION PACKAGE COMPLETE")
print("=" * 70)
print(f"""
Files created:
1. {output_filename} - Complete campaign data (JSON)
2. {emails_filename} - Sample outreach emails (JSON)

Campaign Statistics:
- Leads Processed: {len(LEAD_MEMORY)}
- Messages Sent: {sum(len(messages) for messages in OUTREACH_MEMORY.values())}
- Activities Logged: {len(ACTIVITY_LOG)}

âœ… Notebook is ready for submission!
""")

print("=" * 70 + "\n")

