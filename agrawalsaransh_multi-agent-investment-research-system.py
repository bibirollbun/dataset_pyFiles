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
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



import asyncio
from datetime import datetime
from typing import Dict, Any

!pip install nest_asyncio --quiet
import nest_asyncio
nest_asyncio.apply()


# ADK Imports
from google.adk.agents import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import google_search



def get_financial_data(company_name: str) -> Dict[str, Any]:
    """
    Retrieves comprehensive financial data and key metrics for a company.
    
    Args:
        company_name: Name of the company to analyze
        
    Returns:
        Dictionary containing financial metrics and analysis data
    """
    return {
        "company": company_name,
        "data_source": "Financial Data Tool (Simulated)",
        "timestamp": datetime.now().isoformat(),
        # "financials": {
        #     "revenue_ttm": "To be fetched from API",
        #     "revenue_growth_3y_cagr": "Calculate 3-year CAGR",
        #     "gross_margin": "Current gross margin %",
        #     "ebitda_margin": "Current EBITDA margin %",
        #     "operating_margin": "Current operating margin %",
        #     "net_margin": "Current net profit margin %",
        #     "roe": "Return on Equity %",
        #     "roic": "Return on Invested Capital %",
        #     "debt_to_equity": "Total Debt / Total Equity",
        #     "current_ratio": "Current Assets / Current Liabilities",
        #     "free_cash_flow": "Operating CF - CapEx",
        #     "market_cap": "Current market capitalization",
        #     "pe_ratio": "Price to Earnings ratio",
        #     "pb_ratio": "Price to Book ratio",
        #     "ev_ebitda": "Enterprise Value / EBITDA"

        "financials": {
            "revenue_ttm": "$850M - $1.2B (estimated)",
            "revenue_growth_3y_cagr": "15-25% (CDMO industry average)",
            "gross_margin": "45-55%",
            "ebitda_margin": "25-35%",
            "operating_margin": "18-25%",
            "net_margin": "12-18%",
            "roe": "15-22%",
            "roic": "12-18%",
            "debt_to_equity": "0.3-0.6x",
            "current_ratio": "1.5-2.5x",
            "pe_ratio": "25-40x",
            "ev_ebitda": "15-25x"
        },
        "valuation_context": "Premium valued due to growth prospects in CDMO sector",
        "status": "success"
        # "note": "Production version would fetch real-time data from financial APIs",
        # "status": "success"
    }


def apply_investment_framework(company_name: str, sector: str) -> Dict[str, Any]:
    """
    Applies the investment Framework for systematic equity analysis.
    
    Args:
        company_name: Name of the company to analyze
        sector: Industry sector of the company
        
    Returns:
        Dictionary containing investment Framework analysis structure
    """
    return {
        "company": company_name,
        "sector": sector,
        "framework": "investment Framework for Equity Analysis",
        "analysis_date": datetime.now().isoformat(),
        "analysis_areas": {
            "market_cap_expansion": {
                "total_addressable_market": "Size of total market opportunity",
                "current_market_share": "Company's current % of TAM",
                "expansion_potential": "Whitespace opportunities",
                "geographic_expansion": "New markets to enter",
                "product_expansion": "New product categories"
            },
            "competitive_moat": {
                "barriers_to_entry": "Regulatory, capital, technical barriers",
                "switching_costs": "Customer relationship stickiness",
                "network_effects": "Scale advantages",
                "brand_value": "Pricing power and brand equity",
                "proprietary_tech": "IP, patents, unique capabilities",
                "moat_sustainability": "Defensibility over 5-10 years"
            },
            "pipeline_quality": {
                "phase_3_molecules": "Late-stage pipeline count and quality",
                "capacity_expansion": "Planned facility additions",
                "technology_capabilities": "API, formulation, biologics",
                "customer_concentration": "Revenue diversification",
                "contract_backlog": "Future revenue visibility"
            },
            "margin_dynamics": {
                "current_vs_industry": "Margin comparison to peers",
                "margin_trajectory": "Improving or deteriorating trend",
                "operating_leverage": "Incremental margins on growth",
                "cost_structure": "Fixed vs. variable cost mix",
                "pricing_power": "Ability to pass through inflation"
            },
            "management_quality": {
                "track_record": "Historical execution on promises",
                "capital_allocation": "M&A, buybacks, dividends, R&D",
                "governance": "Board quality, insider ownership",
                "transparency": "Quality of disclosures",
                "strategic_vision": "Long-term strategy clarity"
            },
            "risk_assessment": {
                "regulatory_risk": "FDA, compliance, approval risks",
                "customer_concentration": "Revenue dependency risks",
                "competitive_intensity": "Threat from new entrants",
                "execution_risk": "Expansion plan complexity",
                "financial_risk": "Leverage, liquidity risks",
                "market_risk": "Cyclicality, end-market exposure"
            }
        },
        "output_requirement": {
            "rating": "Buy / Hold / Sell",
            "target_price": "12-month price target",
            "investment_thesis": "3-5 bullet points",
            "key_risks": "Top 3 risks",
            "catalysts": "Performance drivers"
        }
    }


print("âœ“ Tools defined as FUNCTIONS")


class ResearchMemory:
    """Manages research history and builds institutional knowledge."""
    
    def __init__(self):
        self.research_history = []
        self.company_tracking = {}
        self.sector_insights = {}
    
    def add_research(self, company: str, sector: str, analysis: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "company": company,
            "sector": sector,
            "analysis": analysis
        }
        self.research_history.append(entry)
        
        if company not in self.company_tracking:
            self.company_tracking[company] = {"analyses": []}
        self.company_tracking[company]["analyses"].append(entry)
        
        if sector not in self.sector_insights:
            self.sector_insights[sector] = {"companies": []}
        if company not in self.sector_insights[sector]["companies"]:
            self.sector_insights[sector]["companies"].append(company)
    
    def get_summary(self):
        return {
            "total_reports": len(self.research_history),
            "companies": list(self.company_tracking.keys()),
            "sectors": list(self.sector_insights.keys())
        }


print("âœ“ ResearchMemory defined")


class InvestmentResearchSystem:
    """
    Multi-agent investment research system using ADK SequentialAgent.
    
    ALL FIXES APPLIED:
    1. Tools are functions (not classes)
    2. Session created before runner.run()
    3. Proper async handling for Kaggle
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.memory = ResearchMemory()
        
        # Configuration
        self.app_name = "investment_research_app"
        self.user_id = "research_user"
        
        # Pipeline
        self.research_pipeline = None
    
    def setup_agents(self):
        """Create multi-agent architecture using ADK SequentialAgent."""
        
        print("\n" + "="*70)
        print("INITIALIZING MULTI-AGENT SYSTEM")
        print("="*70)
        
        # Agent 1: Financial Analyst
        # â­� Using function directly as tool!
        financial_analyst = Agent(
            name="financial_analyst",
            model=self.model_name,
            description="Expert financial analyst specializing in equity valuation",
            instruction="""
            You are a senior financial analyst with expertise in pharmaceutical and CDMO companies.
            
            Your role:
            1. Analyze financial statements and key metrics
            2. Assess revenue quality and growth sustainability
            3. Evaluate margin trends and operating leverage
            4. Determine financial health and leverage ratios
            5. Benchmark against industry peers
            
            Use the get_financial_data tool to retrieve company metrics.
            Provide quantitative analysis with specific metrics and trends.
            """,
            tools=[get_financial_data], 
            output_key="financial_analysis"
        )
        print("âœ“ Agent 1: Financial Analyst (output â†’ financial_analysis)")
        
        # Agent 2: Market Researcher
        market_researcher = Agent(
            name="market_researcher",
            model=self.model_name,
            description="Market research specialist focusing on competitive dynamics",
            instruction="""
            You are a market research specialist with expertise in pharmaceutical CDMO sector.
            
            Previous financial analysis: {financial_analysis}
            
            Your role:
            1. Apply the investment Framework systematically
            2. Assess competitive moat and sustainability
            3. Evaluate market cap expansion opportunities
            4. Analyze pipeline quality and growth catalysts
            5. Identify key risks to investment thesis
            
            Use apply_investment_framework tool and google_search for research.
            Provide specific examples and competitive positioning assessment.
            """,
            tools=[apply_investment_framework], 
            output_key="market_analysis"
        )
        print("âœ“ Agent 2: Market Researcher (output â†’ market_analysis)")
        
        # Agent 3: Report Generator
        report_generator = Agent(
            name="report_generator",
            model=self.model_name,
            description="Investment report writer synthesizing research",
            instruction="""
            You are a senior investment analyst writing final investment reports.
            
            You have access to:
            - Financial Analysis: {financial_analysis}
            - Market Analysis: {market_analysis}
            
            CREATE A PROFESSIONAL INVESTMENT REPORT:

            ## Executive Summary
            [2-3 sentences: What is this company and what's the investment thesis?]
            
            ## Investment Recommendation
            **Rating:** [BUY / HOLD / SELL]
            **12-Month Target Price:** [Price with rationale]
            **Risk-Reward:** [Favorable / Balanced / Unfavorable]
            
            ## Investment Thesis
            1. [Primary reason to invest/avoid]
            2. [Secondary driver]
            3. [Third key point]
            
            ## Financial Analysis Summary
            [Key financial metrics and what they indicate]
            
            ## investment Framework Assessment
            | Dimension | Rating | Key Insight |
            |-----------|--------|-------------|
            | Market Cap Expansion | Strong/Moderate/Weak | [One line] |
            | Competitive Moat | Strong/Moderate/Weak | [One line] |
            | Pipeline Quality | Strong/Moderate/Weak | [One line] |
            | Margin Dynamics | Strong/Moderate/Weak | [One line] |
            | Management Quality | Strong/Moderate/Weak | [One line] |
            
            ## Growth Catalysts
            - [Catalyst 1]
            - [Catalyst 2]
            - [Catalyst 3]
            
            ## Key Risks
            1. [Risk 1 with mitigation]
            2. [Risk 2 with mitigation]
            3. [Risk 3 with mitigation]
            
            ## Conclusion
            [Final investment recommendation with conviction level]
            
            Be decisive, specific, and professional.
            """,
            output_key="final_report"
        )
        print("âœ“ Agent 3: Report Generator (output â†’ final_report)")
        
        # Create SequentialAgent Pipeline
        self.research_pipeline = SequentialAgent(
            name="investment_research_pipeline",
            description="Orchestrates comprehensive investment research workflow",
            sub_agents=[
                financial_analyst,
                market_researcher,
                report_generator
            ]
        )
        
        print("\nâœ“ SequentialAgent Pipeline Created")
        print("  Flow: Financial Analyst â†’ Market Researcher â†’ Report Generator")
        print("="*70 + "\n")
    
    def research_company(self, company_name: str, sector: str = "pharmaceutical_cdmo") -> Dict[str, Any]:
        """
        Execute research (sync wrapper for async).
        Works with nest_asyncio applied.
        """
        return asyncio.run(self._run_research(company_name, sector))
    
    async def research_company_async(self, company_name: str, sector: str = "pharmaceutical_cdmo") -> Dict[str, Any]:
        """
        Async version - use with await in Jupyter cells.
        Usage: report = await system.research_company_async("Company", "sector")
        """
        return await self._run_research(company_name, sector)
    
    async def _run_research(self, company_name: str, sector: str) -> Dict[str, Any]:
        """
        Core implementation with all fixes applied.
        """
        
        print(f"\n{'='*70}")
        print(f"RESEARCHING: {company_name} ({sector})")
        print(f"{'='*70}\n")
        
        # Step 1: Generate unique session ID
        session_id = f"research_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Step 2: Create session service
        session_service = InMemorySessionService()
        
        # Step 3: CREATE SESSION BEFORE RUNNING 
        print(f"â–¶ Creating session: {session_id}")
        await session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=session_id
        )
        print("âœ“ Session created successfully")
        
        # Step 4: Create runner
        runner = Runner(
            agent=self.research_pipeline,
            app_name=self.app_name,
            session_service=session_service
        )
        
        # Step 5: Create research prompt
        research_prompt = f"""
        Conduct comprehensive investment research on {company_name} in the {sector} sector.
        
        Company: {company_name}
        Sector: {sector}
        
        Please provide:
        1. Complete financial analysis using the financial data tool
        2. Thorough market analysis using investment Framework
        3. Final investment recommendation with target price
        
        Be specific, quantitative, and actionable in your analysis.
        """
        
        user_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=research_prompt)]
        )
        
        # Step 6: Execute pipeline
        print("â–¶ Executing research pipeline...")
        print("  â†’ Step 1: Financial analysis...")
        
        final_response = ""
        async for event in runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=user_message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_response = part.text
        
        print("  â†’ Step 2: Market research completed")
        print("  â†’ Step 3: Report generation completed")
        print("âœ“ Research pipeline completed\n")
        
        # Build output
        research_output = {
            "company": company_name,
            "sector": sector,
            "research_date": datetime.now().isoformat(),
            "research_method": "Multi-Agent Sequential Workflow (ADK SequentialAgent)",
            "session_id": session_id,
            "agents_involved": [
                "financial_analyst",
                "market_researcher",
                "report_generator"
            ],
            "final_report": final_response,
            "status": "completed"
        }
        
        # Store in memory
        self.memory.add_research(company_name, sector, research_output)
        
        return research_output


print("âœ“ InvestmentResearchSystem defined with ALL FIXES")



 # %%
# Create and setup the research system
research_system = InvestmentResearchSystem(model_name="gemini-2.0-flash")
research_system.setup_agents()

print("\nâœ… Research system ready!")
print("   Use: research_system.research_company('Company Name', 'sector')")


# %%
# Define companies to research
companies = [
    ("Sai Life Sciences", "pharmaceutical_cdmo"),
    # Add more companies as needed:
    # ("Divi's Laboratories", "pharmaceutical_cdmo"),
    # ("Laurus Labs", "pharmaceutical_cdmo"),
]

# Research first company
company1, sector1 = companies[0]
report1 = research_system.research_company(company1, sector1)
report1

# # Research second company
# company2, sector2 = companies[0]
# report2 = research_system.research_company(company2, sector2)
# report2

from IPython.display import display, Markdown  # only need once

text = report1['final_report']
text = text.replace("```", "").strip()  # Remove code block fences

display(Markdown(text))




