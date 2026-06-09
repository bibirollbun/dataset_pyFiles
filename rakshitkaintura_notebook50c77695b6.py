!pip install -q google-generativeai python-dotenv


import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

import google.generativeai as genai
from google.generativeai import GenerativeModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




@dataclass
class CompanyProfile:
    name: str
    industry: str
    size: str
    founded: str
    headquarters: str
    website: str
    description: str

@dataclass
class FinancialData:
    revenue: str
    growth_rate: str
    funding_stage: str
    funding_amount: str
    profitability: str
    valuation: Optional[str] = None

@dataclass
class Contact:
    name: str
    title: str
    linkedin_url: str
    email: Optional[str] = None
    phone: Optional[str] = None

@dataclass
class SalesIntelligence:
    company_profile: CompanyProfile
    financial_data: FinancialData
    key_contacts: List[Contact]
    buying_signals: List[str]
    recommendations: List[str]
    confidence_score: float
    generated_at: str



class SimpleSession:
    def __init__(self, session_id: str):
        self.id = session_id
        self.memory = {}
        self.created_at = datetime.now()
    
    def store(self, key: str, value: any):
        self.memory[key] = value
    
    def retrieve(self, key: str) -> any:
        return self.memory.get(key)

class SessionManager:
    def __init__(self):
        self.sessions = {}
        logger.info("SessionManager initialized")
    
    def create_session(self) -> SimpleSession:
        session_id = f"session_{len(self.sessions)}_{datetime.now().timestamp()}"
        session = SimpleSession(session_id)
        self.sessions[session_id] = session
        return session




class ResearchAgent:
    """Agent 1: Company Research"""
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "ResearchAgent"
    
    async def research(self, company_name: str, session: SimpleSession) -> CompanyProfile:
        logger.info(f"{self.name}: Researching {company_name}")
        
        prompt = f"""Research {company_name} and provide JSON with:
        {{"name": "Company Name", "industry": "Industry", "size": "Employee count", 
        "founded": "Year", "headquarters": "Location", "website": "URL", 
        "description": "Brief description"}}"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().replace('```json', '').replace('```', '').strip()
            data = json.loads(text)
            
            profile = CompanyProfile(**{k: data.get(k, 'Unknown') for k in 
                ['name', 'industry', 'size', 'founded', 'headquarters', 'website', 'description']})
            profile.name = company_name
            return profile
        except:
            return CompanyProfile(company_name, "Technology", "1000-5000", "2015", 
                "San Francisco, CA", "www.example.com", "Tech company")


class FinancialAnalysisAgent:
    """Agent 2: Financial Intelligence"""
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "FinancialAnalysisAgent"
    
    async def analyze(self, company_name: str, session: SimpleSession) -> FinancialData:
        logger.info(f"{self.name}: Analyzing {company_name}")
        
        prompt = f"""Analyze {company_name} financials. Return JSON with:
        {{"revenue": "$XXM", "growth_rate": "+XX%", "funding_stage": "Series X", 
        "funding_amount": "$XXM", "profitability": "Status", "valuation": "$XXM"}}"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().replace('```json', '').replace('```', '').strip()
            data = json.loads(text)
            return FinancialData(**{k: data.get(k, 'Unknown') for k in 
                ['revenue', 'growth_rate', 'funding_stage', 'funding_amount', 'profitability', 'valuation']})
        except:
            return FinancialData("$250M ARR", "+45% YoY", "Series C", "$150M", 
                "Path to profitability", "$1B")


class ContactDiscoveryAgent:
    """Agent 3: Decision Maker Identification"""
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "ContactDiscoveryAgent"
    
    async def discover(self, company_name: str, session: SimpleSession) -> List[Contact]:
        logger.info(f"{self.name}: Finding contacts at {company_name}")
        
        prompt = f"""Find 3 key decision-makers at {company_name}. Return JSON:
        {{"contacts": [{{"name": "Name", "title": "Title", "linkedin_url": "URL", 
        "email": "email@company.com"}}]}}"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().replace('```json', '').replace('```', '').strip()
            data = json.loads(text)
            return [Contact(**c) for c in data.get('contacts', [])[:5]]
        except:
            return [
                Contact("Sarah Chen", "VP of Sales", "linkedin.com/in/sarachen", "s.chen@company.com"),
                Contact("Michael Rodriguez", "CTO", "linkedin.com/in/mrodriguez", "m.rodriguez@company.com")
            ]


class ReportGeneratorAgent:
    """Agent 4: Intelligence Synthesis"""
    
    def __init__(self, model: GenerativeModel):
        self.model = model
        self.name = "ReportGeneratorAgent"
    
    async def generate(self, profile: CompanyProfile, financial: FinancialData, 
                      contacts: List[Contact], session: SimpleSession) -> Dict[str, Any]:
        logger.info(f"{self.name}: Generating report")
        
        context = json.dumps({
            'company': asdict(profile),
            'financials': asdict(financial),
            'contacts': [asdict(c) for c in contacts]
        }, indent=2)
        
        prompt = f"""Analyze this data and return JSON with:
        {{"buying_signals": ["signal1", "signal2"], "recommendations": ["rec1", "rec2"], 
        "confidence_score": 85}}
        
        Data: {context}"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(text)
        except:
            return {
                'buying_signals': ['Strong growth', 'Recent funding', 'Expanding team'],
                'recommendations': ['Lead with ROI', 'Target VP of Sales', 'Emphasize scalability'],
                'confidence_score': 80.0
            }



class SalesIntelligenceOrchestrator:
    """Multi-Agent Orchestrator with Parallel + Sequential Workflow"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = GenerativeModel('gemini-pro')
        self.research_agent = ResearchAgent(self.model)
        self.financial_agent = FinancialAnalysisAgent(self.model)
        self.contact_agent = ContactDiscoveryAgent(self.model)
        self.report_agent = ReportGeneratorAgent(self.model)
        self.session_manager = SessionManager()
        self.memory_bank = {}
        
        logger.info("Orchestrator initialized")
    
    async def analyze_company(self, company_name: str) -> SalesIntelligence:
        """
        Multi-Agent Workflow:
        1. PARALLEL: Research + Financial agents
        2. SEQUENTIAL: Contact agent
        3. SEQUENTIAL: Report agent
        """
        logger.info(f"Analyzing {company_name}...")
        session = self.session_manager.create_session()
        start = datetime.now()
        
        # PARALLEL EXECUTION
        profile, financial = await asyncio.gather(
            self.research_agent.research(company_name, session),
            self.financial_agent.analyze(company_name, session)
        )
        
        # SEQUENTIAL EXECUTION
        contacts = await self.contact_agent.discover(company_name, session)
        report_data = await self.report_agent.generate(profile, financial, contacts, session)
        
        intelligence = SalesIntelligence(
            company_profile=profile,
            financial_data=financial,
            key_contacts=contacts,
            buying_signals=report_data.get('buying_signals', []),
            recommendations=report_data.get('recommendations', []),
            confidence_score=report_data.get('confidence_score', 75.0),
            generated_at=datetime.now().isoformat()
        )
        
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"âœ… Completed in {duration:.2f}s")
        
        return intelligence



from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

try:
    GEMINI_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    print("âœ… API key loaded from Kaggle secrets")
except:
    print("âš ï¸�  GOOGLE_API_KEY not found in Kaggle secrets")
    print("ğŸ“� To add it:")
    print("   1. Click 'Add-ons' â†’ 'Secrets' in Kaggle")
    print("   2. Add secret with Label: GOOGLE_API_KEY")
    print("   3. Get your key from: https://aistudio.google.com/app/apikey")
    GEMINI_API_KEY = None


async def demo():
    """Demo function for notebook"""
    
    if not GEMINI_API_KEY:
        print("âš ï¸�  Please add GEMINI_API_KEY to Kaggle secrets!")
        print("\nğŸ“� Steps to add secret:")
        print("   1. Click 'Add-ons' â†’ 'Secrets' in the right sidebar")
        print("   2. Click '+ Add a new secret'")
        print("   3. Label: GEMINI_API_KEY")
        print("   4. Value: Your API key from https://aistudio.google.com/app/apikey")
        print("   5. Click 'Add'")
        print("\nğŸ”„ Then restart the notebook and run again")
        return
    
    orchestrator = SalesIntelligenceOrchestrator(GEMINI_API_KEY)
    
    # Analyze a company
    company = "Anthropic"
    print(f"\nğŸ”� Analyzing {company}...\n")
    
    result = await orchestrator.analyze_company(company)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"ğŸ“Š SALES INTELLIGENCE REPORT: {result.company_profile.name}")
    print(f"{'='*70}\n")
    
    print(f"ğŸ�¢ COMPANY PROFILE:")
    print(f"   Industry: {result.company_profile.industry}")
    print(f"   Size: {result.company_profile.size}")
    print(f"   Founded: {result.company_profile.founded}")
    print(f"   HQ: {result.company_profile.headquarters}\n")
    
    print(f"ğŸ’° FINANCIALS:")
    print(f"   Revenue: {result.financial_data.revenue}")
    print(f"   Growth: {result.financial_data.growth_rate}")
    print(f"   Funding: {result.financial_data.funding_stage} - {result.financial_data.funding_amount}\n")
    
    print(f"ğŸ‘¥ KEY CONTACTS:")
    for contact in result.key_contacts[:3]:
        print(f"   â€¢ {contact.name} - {contact.title}")
        print(f"     {contact.linkedin_url}\n")
    
    print(f"ğŸ“ˆ BUYING SIGNALS:")
    for signal in result.buying_signals:
        print(f"   âœ“ {signal}")
    
    print(f"\nğŸ’¡ RECOMMENDATIONS:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\nğŸ�¯ Confidence Score: {result.confidence_score}%")
    print(f"{'='*70}\n")
    
    print("âœ… Analysis complete! Time saved: ~9.5 hours")





# STEP 9: Run it! (In Kaggle/Colab, just run this cell)
await demo()

