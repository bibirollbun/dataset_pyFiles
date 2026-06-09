"""
Install required packages for AI Agent system
"""

print("Installing packages... (2-3 minutes)")

!pip install -q google-genai==0.2.0
!pip install -q pandas==2.1.0
!pip install -q matplotlib==3.8.0

print("âœ… All packages installed!")


 #  AIzaSyBPV0339EHPEaa2cGtAuoiUiaka0TgRmMo


"""
Import libraries and set up Google Gemini API
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import json

# Google Generative AI
from google import genai
from google.genai import types

# Data processing
import pandas as pd

print("âœ… Libraries imported successfully")


# SET API KEY


# Replace with your actual API key from Google AI Studio
GEMINI_API_KEY = "AIzaSyBPV0339EHPEaa2cGtAuoiUiaka0TgRmMo" 


# Set environment variable
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

print("âœ… Gemini API initialized")
print(f"Using model: gemini-2.0-flash")


"""
Agent System Components - Part 1
Includes: Logging system and Data Collector Agent
"""
# OBSERVABILITY: Agent Logger


class AgentLogger:
    """
    Logging system for tracking all agent activities.
    Demonstrates: Observability concept
    """
    
    def __init__(self):
        self.logs = []
        self.start_time = datetime.now()
    
    def log(self, agent_name: str, action: str, details: str = "", level: str = "INFO"):
        """Log an agent action."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        log_entry = {
            "timestamp": timestamp,
            "elapsed_seconds": elapsed,
            "agent": agent_name,
            "action": action,
            "details": details,
            "level": level
        }
        
        self.logs.append(log_entry)
        
        # Print to console
        icon = {"INFO": "â„¹ï¸�", "SUCCESS": "âœ…", "ERROR": "â�Œ", "WARNING": "âš ï¸�"}.get(level, "ğŸ“�")
        print(f"[{timestamp}] {icon} {agent_name}: {action}")
        if details:
            print(f"           Details: {details[:100]}...")
    
    def get_summary(self) -> str:
        """Get summary of all agent activities."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        return f"""
Agent Activity Summary:
- Total actions logged: {len(self.logs)}
- Total execution time: {total_time:.2f} seconds
- Agents involved: {len(set(log['agent'] for log in self.logs))}
"""

# Create global logger instance
logger = AgentLogger()

print("âœ… AgentLogger initialized")


# MULTI-AGENT SYSTEM: Agent 1 - Data Collector

class DataCollectorAgent:
    """
    Agent responsible for understanding queries and identifying data needs.
    Role: Determines what logistics KPIs are relevant for the query.
    """
    
    def __init__(self):
        self.model = "gemini-2.0-flash-exp"
        self.client = client
        logger.log("DataCollectorAgent", "Initialized", level="SUCCESS")
    
    def collect_data_requirements(self, user_query: str) -> str:
        """
        Analyze user query and determine data requirements.
        
        Args:
            user_query: Natural language logistics question
            
        Returns:
            Structured list of required data points and KPIs
        """
        logger.log("DataCollectorAgent", f"Processing query: {user_query[:50]}...")
        
        prompt = f"""You are a logistics data specialist with expertise in:
- Supply chain KPIs
- Fleet management metrics
- Delivery performance indicators
- Cost analysis

User Query: "{user_query}"

Your task: Identify what specific logistics data points and KPIs are needed to answer this query.

Provide a structured list of:
1. Required data points
2. Relevant KPIs to calculate
3. Time period needed
4. Any comparative benchmarks

Format your response clearly with sections."""
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            result = response.text
            logger.log("DataCollectorAgent", "Data requirements identified", level="SUCCESS")
            return result
            
        except Exception as e:
            logger.log("DataCollectorAgent", f"Error: {e}", level="ERROR")
            return f"Error collecting data requirements: {e}"

print("âœ… DataCollectorAgent defined")



"""
Agent System Components - Part 2
Includes: Analyst Agent and Report Generator Agent
"""

# MULTI-AGENT SYSTEM: Agent 2 - Analyst

class AnalystAgent:
    """
    Agent responsible for analyzing logistics data and finding insights.
    Role: Performs quantitative analysis and identifies patterns.
    """
    
    def __init__(self):
        self.model = "gemini-2.0-flash-exp"
        self.client = client
        logger.log("AnalystAgent", "Initialized", level="SUCCESS")
    
    def analyze_data(self, data_requirements: str, sample_data: Optional[str] = None) -> str:
        """
        Analyze logistics data based on requirements.
        
        Args:
            data_requirements: What data points to analyze
            sample_data: Optional sample data to work with
            
        Returns:
            Analysis findings with insights and recommendations
        """
        logger.log("AnalystAgent", "Starting analysis...")
        
        # Create sample logistics data if not provided
        if sample_data is None:
            sample_data = """
Sample Logistics Data (Q3 2025):
- Total Deliveries: 1,234
- On-Time Deliveries: 1,073 (87%)
- Average Delivery Time: 2.3 days
- Total Cost: $55,530
- Cost per Delivery: $45
- Fleet Utilization: 76%

Zone Breakdown:
- Zone 1: 234 deliveries, 92% on-time, $42/delivery
- Zone 2: 456 deliveries, 89% on-time, $44/delivery
- Zone 3: 298 deliveries, 81% on-time, $56/delivery (HIGH COST!)
- Zone 4: 189 deliveries, 88% on-time, $43/delivery
- Zone 5: 157 deliveries, 78% on-time, $47/delivery (LOW ON-TIME!)
"""
        
        prompt = f"""You are a logistics business analyst with 10+ years experience in:
- Fleet optimization
- Route efficiency analysis
- Cost reduction strategies
- Performance improvement

Data Requirements:
{data_requirements}

Available Data:
{sample_data}

Your task: Provide comprehensive analysis including:
1. Key Performance Metrics Summary
2. Performance Assessment (strengths/weaknesses)
3. Problem Areas Identified
4. Root Cause Analysis
5. Optimization Opportunities (with estimated impact)

Use data-driven insights. Be specific with numbers and percentages.

Analysis Report:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            result = response.text
            logger.log("AnalystAgent", "Analysis completed", f"Generated {len(result)} chars", "SUCCESS")
            return result
            
        except Exception as e:
            logger.log("AnalystAgent", f"Error during analysis: {e}", level="ERROR")
            return f"Error during analysis: {e}"

print("âœ… AnalystAgent defined")

# MULTI-AGENT SYSTEM: Agent 3 - Report Generator

class ReportGeneratorAgent:
    """
    Agent responsible for creating executive-ready reports.
    Role: Transforms analysis into actionable business reports.
    """
    
    def __init__(self):
        self.model = "gemini-2.0-flash-exp"
        self.client = client
        logger.log("ReportGeneratorAgent", "Initialized", level="SUCCESS")
    
    def generate_report(self, analysis_findings: str, report_format: str = "executive") -> str:
        """
        Generate professional business report from analysis.
        
        Args:
            analysis_findings: Raw analysis from AnalystAgent
            report_format: Type of report (executive, detailed, summary)
            
        Returns:
            Formatted business report
        """
        logger.log("ReportGeneratorAgent", f"Generating {report_format} report...")
        
        prompt = f"""You are an executive business report writer specializing in logistics.

Analysis Findings:
{analysis_findings}

Create a professional {report_format.upper()} REPORT with:

1. EXECUTIVE SUMMARY (2-3 sentences of key takeaways)

2. KEY FINDINGS (bullet points, data-driven)

3. RECOMMENDATIONS (numbered list with estimated impact)
   - Include cost savings estimates
   - Include timeline for implementation
   - Prioritize by impact

4. NEXT ACTIONS (specific, actionable steps)

Use professional business language. Be concise but comprehensive.
Include specific numbers and percentages from the analysis.

REPORT:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            result = response.text
            logger.log("ReportGeneratorAgent", "Report generated successfully", level="SUCCESS")
            return result
            
        except Exception as e:
            logger.log("ReportGeneratorAgent", f"Error: {e}", level="ERROR")
            return f"Error generating report: {e}"

print("âœ… ReportGeneratorAgent defined")


"""
Main Agent Orchestrator
Coordinates all agents in sequential pipeline
"""

# MAIN ORCHESTRATOR CLASS

class LogisticsIntelligenceAgent:
    """
    Main orchestrator for Logistics Intelligence Agent system.
    
    Demonstrates:
    - Multi-agent system (3 sequential agents)
    - Agent coordination and orchestration
    - End-to-end workflow automation
    """
    
    def __init__(self):
        print("\n" + "=" * 60)
        print("   LOGISTICS INTELLIGENCE AGENT SYSTEM")
        print("=" * 60)
        
        logger.log("Orchestrator", "Initializing agent system...")
        
        # Initialize all agents
        self.data_collector = DataCollectorAgent()
        self.analyst = AnalystAgent()
        self.reporter = ReportGeneratorAgent()
        
        logger.log("Orchestrator", "All agents initialized", level="SUCCESS")
        print("=" * 60)
        print()
    
    def process_query(self, user_query: str, include_sample_data: bool = True) -> Dict:
        """
        Process logistics query through multi-agent pipeline.
        
        Pipeline:
        1. Data Collector identifies what data is needed
        2. Analyst performs analysis on available data
        3. Reporter creates executive report
        
        Args:
            user_query: Natural language logistics question
            include_sample_data: Whether to use sample data
            
        Returns:
            Dictionary with all outputs and metadata
        """
        print(f"\n{'='*60}")
        print(f"ğŸ”� USER QUERY: {user_query}")
        print(f"{'='*60}\n")
        
        logger.log("Orchestrator", "Starting query processing", user_query)
        
        # STAGE 1: Data Collection
        print("\nğŸ“¦ STAGE 1: Data Collection")
        print("â”€" * 60)
        data_requirements = self.data_collector.collect_data_requirements(user_query)
        print(data_requirements)
        
        # STAGE 2: Analysis
        print("\n\nğŸ“Š STAGE 2: Analysis")
        print("â”€" * 60)
        analysis = self.analyst.analyze_data(
            data_requirements,
            sample_data="Use sample data" if include_sample_data else None
        )
        print(analysis)
        
        # STAGE 3: Report Generation
        print("\n\nğŸ“� STAGE 3: Report Generation")
        print("â”€" * 60)
        report = self.reporter.generate_report(analysis, report_format="executive")
        print(report)
        
        # Compile results
        result = {
            "query": user_query,
            "data_requirements": data_requirements,
            "analysis": analysis,
            "final_report": report,
            "processing_time": f"{(datetime.now() - logger.start_time).total_seconds():.2f}s",
            "logs": logger.logs
        }
        
        logger.log("Orchestrator", "Query processing complete", level="SUCCESS")
        
        return result
    
    def get_activity_summary(self) -> str:
        """Get summary of all agent activities."""
        return logger.get_summary()

print("âœ… LogisticsIntelligenceAgent orchestrator defined")


"""
Test the Logistics Intelligence Agent
Run multiple example queries to demonstrate functionality
"""

print("\n" + "=" * 60)
print("   TESTING LOGISTICS INTELLIGENCE AGENT")
print("=" * 60)

# Create agent instance
agent = LogisticsIntelligenceAgent()

# Test queries based on real logistics scenarios
test_queries = [
    "Analyze delivery performance for last quarter and identify improvement areas",
    "What are the main cost drivers in our logistics operations and how can we reduce them?",
    "Compare zone performance and recommend resource reallocation",
]

# Run first test query
print("\n\n" + "ğŸ�¯" * 30)
print("RUNNING TEST QUERY 1")
print("ğŸ�¯" * 30)

result = agent.process_query(test_queries[0])

# Display final report prominently
print("\n\n" + "=" * 60)
print("   ğŸ“„ FINAL EXECUTIVE REPORT")
print("=" * 60)
print(result['final_report'])

# Show activity summary
print("\n\n" + "=" * 60)
print("   ğŸ“Š AGENT ACTIVITY SUMMARY")
print("=" * 60)
print(agent.get_activity_summary())

print("\nâœ… TEST COMPLETE - Agent working successfully!")


"""
This is Optional: Save agent outputs for documentation
"""

# Save results to JSON for your writeup
import json

output_data = {
    "project": "Logistics Intelligence Agent",
    "track": "Enterprise Agents",
    "author": "Oluwatobi Elusakin",
    "demo_query": result['query'],
    "execution_time": result['processing_time'],
    "concepts_demonstrated": [
        "Multi-agent system (3 sequential agents)",
        "Observability (comprehensive logging)",
        "Tools (Gemini 2.0 Flash LLM)"
    ],
    "final_report": result['final_report']
}

# Display for documentation
print("=" * 60)
print("PROJECT DATA FOR SUBMISSION:")
print("=" * 60)
print(json.dumps(output_data, indent=2))

# Can also save to file
with open('agent_demo_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\nâœ… Results saved for submission documentation")


