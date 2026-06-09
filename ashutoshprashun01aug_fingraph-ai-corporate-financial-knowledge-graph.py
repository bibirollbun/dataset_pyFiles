# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install PyPDF2
!pip install --upgrade google-adk


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

import os
import io
import json
import pandas as pd
import PyPDF2
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
from dataclasses import dataclass, asdict
print("imported")


class TOONFormatter:
    """
    TOON format reduces tokens by 30-60% compared to JSON[citation:1]
    Perfect for AI agents where token efficiency matters[citation:7]
    """
    
    @staticmethod
    def dict_to_toon(data: Dict, indent: int = 0) -> str:
        """Convert Python dict to TOON format"""
        lines = []
        indent_str = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.append(TOONFormatter.dict_to_toon(value, indent + 1))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Tabular array format for uniform objects[citation:7]
                fields = list(value[0].keys())
                lines.append(f"{indent_str}{key}[{len(value)}]{{{','.join(fields)}}}:")
                for item in value:
                    row_values = [str(item[field]) for field in fields]
                    lines.append(f"{indent_str}  {','.join(row_values)}")
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}[{len(value)}]: {','.join(str(v) for v in value)}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def toon_to_dict(toon_text: str) -> Dict:
        """Convert TOON format back to Python dict (simplified)"""
        result = {}
        lines = toon_text.strip().split('\n')
        stack = [(result, 0)]  # (current_dict, indent_level)
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line:
                i += 1
                continue
                
            # Count indentation
            indent = len(line) - len(line.lstrip())
            current_level = indent // 2
            
            # Remove indentation
            line = line.strip()
            
            if ':' in line and '[' not in line and '{' not in line:
                # Simple key-value pair
                key, value = line.split(':', 1)
                key, value = key.strip(), value.strip()
                
                # Find correct parent based on indentation
                while stack and stack[-1][1] >= current_level:
                    stack.pop()
                
                parent_dict, _ = stack[-1]
                parent_dict[key] = value
                
            elif '[' in line and '{' in line:
                # Tabular array format
                key_part, rest = line.split(':', 1)
                key = key_part.split('[')[0].strip()
                
                # Extract fields
                fields_start = rest.find('{') + 1
                fields_end = rest.find('}')
                fields = [f.strip() for f in rest[fields_start:fields_end].split(',')]
                
                # Find correct parent
                while stack and stack[-1][1] >= current_level:
                    stack.pop()
                
                parent_dict, _ = stack[-1]
                parent_dict[key] = []
                stack.append(({}, current_level))
                
                # Parse rows
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().endswith(':'):
                    row_line = lines[i].strip()
                    values = [v.strip() for v in row_line.split(',')]
                    row_dict = {fields[j]: values[j] for j in range(min(len(fields), len(values)))}
                    parent_dict[key].append(row_dict)
                    i += 1
                i -= 1  # Adjust for outer loop increment
                
            elif line.endswith(':'):
                # Nested object
                key = line[:-1].strip()
                
                # Find correct parent
                while stack and stack[-1][1] >= current_level:
                    stack.pop()
                
                parent_dict, _ = stack[-1]
                new_dict = {}
                parent_dict[key] = new_dict
                stack.append((new_dict, current_level))
            
            i += 1
        
        return result



class DocumentProcessor:
    """Process uploaded financial documents"""
    
    @staticmethod
    def extract_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF documents"""
        text = ""
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        
        return text
    
    @staticmethod
    def extract_from_csv(file_content: bytes) -> str:
        """Extract data from CSV files"""
        df = pd.read_csv(io.BytesIO(file_content))
        return f"CSV Data: {len(df)} rows, {len(df.columns)} columns\n" + df.head().to_string()
    
    @staticmethod
    def extract_from_excel(file_content: bytes) -> str:
        """Extract data from Excel files"""
        df = pd.read_excel(io.BytesIO(file_content))
        return f"Excel Data: {len(df)} rows, {len(df.columns)} columns\n" + df.head().to_string()

# ============================================
# Financial Analysis Tools
# ============================================
class FinancialAnalysisTools:
    """Core financial analysis tools for the agent"""
    
    # Tool 1: Document Processing Tool
    @staticmethod
    def process_financial_document(
        file_name: str,
        file_content: bytes,
        document_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Process uploaded financial documents (PDF, CSV, Excel)
        Returns extracted text and metadata in TOON format
        """
        try:
            # Determine file type
            if file_name.lower().endswith('.pdf'):
                content = DocumentProcessor.extract_from_pdf(file_content)
                doc_type = "pdf"
            elif file_name.lower().endswith('.csv'):
                content = DocumentProcessor.extract_from_csv(file_content)
                doc_type = "csv"
            elif file_name.lower().endswith(('.xlsx', '.xls')):
                content = DocumentProcessor.extract_from_excel(file_content)
                doc_type = "excel"
            else:
                content = file_content.decode('utf-8', errors='ignore')
                doc_type = "text"
            
            # Classify document type
            if document_type == "auto":
                if any(term in file_name.lower() for term in ['invoice', 'bill']):
                    doc_class = "invoice"
                elif any(term in file_name.lower() for term in ['balance', 'financial']):
                    doc_class = "balance_sheet"
                elif any(term in file_name.lower() for term in ['contract', 'agreement']):
                    doc_class = "contract"
                elif any(term in file_name.lower() for term in ['purchase', 'po']):
                    doc_class = "purchase_order"
                else:
                    doc_class = "financial_document"
            else:
                doc_class = document_type
            
            # Create structured response
            result = {
                "status": "success",
                "document_id": str(uuid.uuid4()),
                "file_name": file_name,
                "document_type": doc_class,
                "content_preview": content[:1000] + "..." if len(content) > 1000 else content,
                "content_length": len(content),
                "processed_at": datetime.now().isoformat(),
                "entities_found": FinancialAnalysisTools._count_entities(content)
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "file_name": file_name
            }
    
    @staticmethod
    def _count_entities(content: str) -> Dict[str, int]:
        """Count potential financial entities in text"""
        content_lower = content.lower()
        return {
            "companies": len([w for w in content_lower.split() if any(term in w for term in ['inc', 'ltd', 'corp', 'llc'])]),
            "amounts": len([w for w in content_lower.split() if '$' in w or 'usd' in w]),
            "dates": len([w for w in content_lower.split() if any(term in w for term in ['2024', '2025', 'jan', 'feb', 'mar', 'q1', 'q2', 'q3', 'q4'])]),
            "products": len([w for w in content_lower.split() if any(term in w for term in ['item', 'product', 'service', 'sku'])]),
        }
    
    # Tool 2: Cash Conversion Cycle Analysis
    @staticmethod
    def analyze_cash_conversion_cycle(
        period: str = "last quarter",
        inventory_days: Optional[float] = None,
        receivables_days: Optional[float] = None,
        payables_days: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze Cash Conversion Cycle (CCC)
        CCC = DIO + DSO - DPO
        """
        # Default values for demonstration
        if inventory_days is None:
            inventory_days = 45.0  # Days Inventory Outstanding
        if receivables_days is None:
            receivables_days = 60.0  # Days Sales Outstanding
        if payables_days is None:
            payables_days = 30.0  # Days Payables Outstanding
        
        ccc = inventory_days + receivables_days - payables_days
        
        # Analyze trends
        previous_ccc = ccc - 5.0  # Simulating improvement
        
        # Identify root causes
        root_causes = []
        if inventory_days > 50:
            root_causes.append(f"High inventory levels (DIO = {inventory_days} days)")
        if receivables_days > 45:
            root_causes.append(f"Slow customer payments (DSO = {receivables_days} days)")
        if payables_days < 35:
            root_causes.append(f"Quick supplier payments (DPO = {payables_days} days)")
        
        if not root_causes:
            root_causes.append("Normal operating range")
        
        # Generate recommendations
        recommendations = []
        if inventory_days > 50:
            recommendations.append("Optimize inventory management and reduce stock levels")
        if receivables_days > 45:
            recommendations.append("Implement stricter credit terms and improve collections process")
        if payables_days < 35:
            recommendations.append("Negotiate longer payment terms with key suppliers")
        
        return {
            "analysis": "cash_conversion_cycle",
            "period": period,
            "metrics": {
                "days_inventory_outstanding": round(inventory_days, 2),
                "days_sales_outstanding": round(receivables_days, 2),
                "days_payables_outstanding": round(payables_days, 2),
                "cash_conversion_cycle": round(ccc, 2)
            },
            "trend_analysis": {
                "current_ccc": round(ccc, 2),
                "previous_ccc": round(previous_ccc, 2),
                "change": round(ccc - previous_ccc, 2),
                "direction": "improving" if ccc < previous_ccc else "worsening"
            },
            "root_causes": root_causes,
            "recommendations": recommendations,
            "formula": "CCC = DIO + DSO - DPO",
            "interpretation": f"A CCC of {round(ccc, 1)} days means it takes {round(ccc, 1)} days to convert inventory investments into cash flows."
        }
    
    # Tool 3: Budget Variance Analysis
    @staticmethod
    def analyze_budget_variances(
        period: str = "Q4 2024",
        cost_centers: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze which cost centers overshot budgets and why
        """
        # Default cost centers for demonstration
        if cost_centers is None:
            cost_centers = [
                {
                    "name": "Marketing",
                    "budget": 500000,
                    "actual": 620000,
                    "variance": 120000,
                    "variance_percent": 24.0,
                    "primary_reasons": ["Unexpected ad campaign", "Agency fees exceeded estimates"],
                    "largest_expense": "Digital Advertising - $250,000"
                },
                {
                    "name": "IT Department",
                    "budget": 300000,
                    "actual": 350000,
                    "variance": 50000,
                    "variance_percent": 16.7,
                    "primary_reasons": ["Hardware refresh cycle", "Cybersecurity software renewal"],
                    "largest_expense": "Server Upgrade - $120,000"
                },
                {
                    "name": "Operations",
                    "budget": 800000,
                    "actual": 750000,
                    "variance": -50000,
                    "variance_percent": -6.3,
                    "primary_reasons": ["Process efficiency gains", "Vendor contract renegotiation"],
                    "largest_expense": "Logistics Optimization - $200,000"
                },
                {
                    "name": "Research & Development",
                    "budget": 400000,
                    "actual": 450000,
                    "variance": 50000,
                    "variance_percent": 12.5,
                    "primary_reasons": ["New product prototype", "Additional testing resources"],
                    "largest_expense": "Prototype Development - $180,000"
                }
            ]
        
        # Calculate summary metrics
        overshot_centers = [cc for cc in cost_centers if cc["variance"] > 0]
        total_overspent = sum(cc["variance"] for cc in overshot_centers)
        
        return {
            "analysis": "budget_variance",
            "period": period,
            "cost_centers_analyzed": len(cost_centers),
            "centers_over_budget": len(overshot_centers),
            "total_overspent": total_overspent,
            "detailed_analysis": cost_centers,
            "summary": {
                "highest_variance_percent": max(cc["variance_percent"] for cc in cost_centers),
                "lowest_variance_percent": min(cc["variance_percent"] for cc in cost_centers),
                "average_variance_percent": sum(cc["variance_percent"] for cc in cost_centers) / len(cost_centers)
            },
            "recommendations": [
                "Review and approve all expenses over $50,000",
                "Implement quarterly budget review meetings",
                "Consider reallocating underspent budgets to critical areas",
                "Set up variance alerts for departments exceeding 15% variance"
            ]
        }
    
    # Tool 4: Financial Entity Extraction
    @staticmethod
    def extract_financial_entities(
        document_text: str,
        entity_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract financial entities from document text
        Returns entities in TOON format for efficiency[citation:3]
        """
        if entity_types is None:
            entity_types = ["companies", "amounts", "dates", "products", "contract_terms"]
        
        # Simplified entity extraction (in production, use NER or LLM)
        entities = {
            "companies": [],
            "amounts": [],
            "dates": [],
            "products": [],
            "contract_terms": []
        }
        
        lines = document_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Extract amounts
            if '$' in line:
                parts = line.split('$')
                for i in range(1, len(parts)):
                    amount_part = parts[i].split()[0] if parts[i].split() else parts[i]
                    try:
                        amount = float(''.join(filter(str.isdigit, amount_part)))
                        entities["amounts"].append({
                            "amount": amount,
                            "currency": "USD",
                            "context": line[:100]
                        })
                    except:
                        pass
            
            # Extract dates
            date_keywords = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                           '2024', '2025', 'q1', 'q2', 'q3', 'q4']
            if any(keyword in line_lower for keyword in date_keywords):
                entities["dates"].append({
                    "text": line.strip(),
                    "type": "date_reference"
                })
            
            # Extract company names (simplified)
            company_indicators = ['inc', 'ltd', 'corp', 'llc', 'gmbh', 'pte']
            words = line.split()
            for i, word in enumerate(words):
                if any(indicator in word.lower() for indicator in company_indicators):
                    # Get 2 words before as potential company name
                    company_name = ' '.join(words[max(0, i-2):i+1])
                    entities["companies"].append({
                        "name": company_name,
                        "type": "company"
                    })
        
        # Filter to requested entity types
        filtered_entities = {et: entities[et] for et in entity_types if et in entities}
        
        return {
            "status": "success",
            "entities_found": sum(len(v) for v in filtered_entities.values()),
            "entities_by_type": filtered_entities,
            "toon_format": TOONFormatter.dict_to_toon(filtered_entities)
        }
    
    # Tool 5: Financial Query Tool
    @staticmethod
    def answer_financial_query(
        query: str,
        context_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Answer CFO-level financial queries using available data
        """
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['cash conversion', 'ccc', 'working capital']):
            return FinancialAnalysisTools.analyze_cash_conversion_cycle()
        
        elif any(term in query_lower for term in ['budget', 'overshot', 'over budget', 'variance']):
            return FinancialAnalysisTools.analyze_budget_variances()
        
        elif any(term in query_lower for term in ['profit', 'revenue', 'margin']):
            return {
                "query": query,
                "analysis": "profitability",
                "metrics": {
                    "gross_margin": "42%",
                    "operating_margin": "28%",
                    "net_margin": "18%",
                    "revenue_growth": "15% YoY"
                },
                "insights": [
                    "Margins have improved by 3% compared to last quarter",
                    "Revenue growth is strong but slowing",
                    "Cost optimization opportunities identified in operations"
                ]
            }
        
        elif any(term in query_lower for term in ['entity', 'extract', 'document']):
            return {
                "query": query,
                "response": "Use the extract_financial_entities tool with document text",
                "example_usage": "extract_financial_entities(document_text='invoice content...')"
            }
        
        else:
            return {
                "query": query,
                "response": "I can help with: cash conversion cycle, budget variances, profitability analysis, and entity extraction.",
                "available_tools": [
                    "process_financial_document",
                    "analyze_cash_conversion_cycle", 
                    "analyze_budget_variances",
                    "extract_financial_entities"
                ]
            }



class CorporateFinancialAgent:
    """
    Main Corporate Financial Knowledge Graph Agent
    Built using Google ADK following Kaggle tutorial patterns
    """
    
    def _init_(self, api_key: str = None):
        # Set API key for Google AI Studio
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Create tools using FunctionTool (alternative to AgentConfig)
        self.tools = [
            FunctionTool.create(
                FinancialAnalysisTools,
                "process_financial_document",
                description="Process uploaded financial documents (PDF, CSV, Excel) and extract text"
            ),
            FunctionTool.create(
                FinancialAnalysisTools,
                "analyze_cash_conversion_cycle",
                description="Analyze Cash Conversion Cycle (CCC) - calculate DIO, DSO, DPO and identify root causes"
            ),
            FunctionTool.create(
                FinancialAnalysisTools,
                "analyze_budget_variances",
                description="Analyze which cost centers overshot budgets and identify reasons"
            ),
            FunctionTool.create(
                FinancialAnalysisTools,
                "extract_financial_entities",
                description="Extract financial entities (companies, amounts, dates) from document text"
            ),
            FunctionTool.create(
                FinancialAnalysisTools,
                "answer_financial_query",
                description="Answer CFO-level financial queries using available tools and data"
            )
        ]
        
        # Create the agent using LlmAgent builder pattern[citation:2]
        self.agent = Agent.builder() \
            .name("corporate_financial_agent") \
            .model(Gemini(model="gemini-2.0-flash")) \
            .description("Corporate Financial Knowledge Graph Agent that processes documents, extracts entities, and answers CFO-level queries") \
            .instruction("""
            You are a Corporate Financial Knowledge Graph Agent with these capabilities:
            
            1. DOCUMENT PROCESSING:
               - Process uploaded financial documents (invoices, contracts, balance sheets, purchase orders)
               - Extract text and identify document types
            
            2. FINANCIAL ANALYSIS:
               - Analyze Cash Conversion Cycle (CCC = DIO + DSO - DPO)
               - Identify why CCC worsened and provide recommendations
               - Analyze budget variances across cost centers
               - Identify departments overshooting budgets and root causes
            
            3. ENTITY EXTRACTION:
               - Extract financial entities (companies, amounts, dates, products)
               - Use TOON format for efficient data representation[citation:1]
            
            4. CFO-LEVEL QUERIES:
               - Answer questions like "Why did our cash conversion cycle worsen?"
               - Provide actionable insights with data-driven evidence
            
            Always use the appropriate tools for each task. Format responses clearly.
            Use TOON format when returning structured data for better token efficiency.[citation:7]
            """) \
            .tools(self.tools) \
            .build()
        
        # Create runner for execution
        self.runner = InMemoryRunner(self.agent)
    
    async def process_query(self, user_id: str, query: str) -> str:
        """Process a user query through the agent"""
        try:
            # Create session
            session = self.runner.session_service().create_session(
                self.agent.name, 
                user_id
            )
            
            # Run the agent
            result = await self.runner.run_async(
                user_id=user_id,
                session_id=session.id,
                user_content=types.Content.from_text(query)
            )
            
            # Extract and format response
            response = ""
            for event in result:
                if hasattr(event, 'stringify_content'):
                    response += event.stringify_content() + "\n"
            
            return response if response else "No response generated"
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def process_with_tools(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Direct tool execution for testing"""
        tool_map = {
            "process_document": FinancialAnalysisTools.process_financial_document,
            "analyze_ccc": FinancialAnalysisTools.analyze_cash_conversion_cycle,
            "analyze_budget": FinancialAnalysisTools.analyze_budget_variances,
            "extract_entities": FinancialAnalysisTools.extract_financial_entities,
            "answer_query": FinancialAnalysisTools.answer_financial_query
        }
        
        if tool_name in tool_map:
            return tool_map[tool_name](**kwargs)
        else:
            return {"error": f"Tool {tool_name} not found"}




def demonstrate_agent():
    """Demonstrate the agent's capabilities for Kaggle notebook"""
    print("=" * 60)
    print("ğŸ�¢ Corporate Financial Knowledge Graph Agent")
    print("Built for Kaggle with Google ADK & TOON Format")
    print("=" * 60)
    
    # Initialize agent (use your API key)
    # agent = CorporateFinancialAgent(api_key="YOUR_API_KEY_HERE")
    
    # For demonstration without API key, use direct tools
    print("\nâ¿¡ Cash Conversion Cycle Analysis:")
    ccc_result = FinancialAnalysisTools.analyze_cash_conversion_cycle()
    print(f"CCC Analysis Result:")
    print(f"Cash Conversion Cycle: {ccc_result['metrics']['cash_conversion_cycle']} days")
    print(f"Root Causes: {', '.join(ccc_result['root_causes'])}")
    
    print("\n" + "-" * 60)
    print("\nâ¿¢ Budget Variance Analysis:")
    budget_result = FinancialAnalysisTools.analyze_budget_variances()
    overspent = [cc for cc in budget_result['detailed_analysis'] if cc['variance'] > 0]
    print(f"Departments Over Budget: {len(overspent)}")
    for dept in overspent[:2]:  # Show first 2
        print(f"  â€¢ {dept['name']}: +${dept['variance']:,} ({dept['variance_percent']}%)")
    
    print("\n" + "-" * 60)
    print("\nâ¿£ TOON Format Demonstration[citation:3]:")
    sample_data = {
        "financial_entities": [
            {"type": "company", "name": "ABC Corp", "amount": 50000},
            {"type": "company", "name": "XYZ Ltd", "amount": 75000}
        ],
        "analysis_period": "Q4 2024",
        "total_amount": 125000
    }
    
    toon_output = TOONFormatter.dict_to_toon(sample_data)
    print("TOON Format Output (Token Efficient):")
    print(toon_output)
    
    print("\n" + "-" * 60)
    print("\nâ¿¤ Example CFO Queries You Can Ask:")
    print("â€¢ 'Why did our cash conversion cycle worsen last quarter?'")
    print("â€¢ 'Which cost centers overshot budgets and why?'")
    print("â€¢ 'Analyze our working capital efficiency'")
    print("â€¢ 'Extract entities from this financial document'")
    
    print("\n" + "=" * 60)
    print("âœ… Agent is ready for financial analysis!")

# ============================================
# Main Execution for Kaggle
# ============================================
if __name__ == "__main__":
    # This runs when executed in Kaggle notebook
    demonstrate_agent()
    
    # Example: Create a simple agent instance for notebook use
    try:
        # Try to get API key from Kaggle secrets
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("GOOGLE_API_KEY")
        
        if api_key:
            print(f"\nğŸ”‘ API Key found! Initializing agent...")
          
        else:
            print("\nâš   No API key found. Using demonstration mode.")
            print("To use full agent capabilities, add your Google API key to Kaggle secrets.")
            
    except ImportError:
        print("\nğŸ“˜ Running in demonstration mode")
        print("For full functionality, run this in Kaggle with GoogleÂ APIÂ key")


# ============================================
# FINANCIAL TEST DATA FOR CORPORATE AGENT
# ============================================

# 1. SAMPLE INVOICE DATA
sample_invoice_text = """
INVOICE

Invoice Number: INV-2024-001
Date: January 15, 2024
Due Date: February 15, 2024

BILL TO:
TechNova Corporation
123 Innovation Drive
San Francisco, CA 94105

FROM:
Quantum Solutions Inc.
456 Tech Park Avenue
Austin, TX 78701

DESCRIPTION OF SERVICES:
Item 1: Cloud Infrastructure Services
Quantity: 1 month
Unit Price: $25,000.00
Total: $25,000.00

Item 2: AI Model Training Hours
Quantity: 150 hours
Unit Price: $200.00
Total: $30,000.00

Item 3: Technical Support
Quantity: 1 month
Unit Price: $5,000.00
Total: $5,000.00

SUBTOTAL: $60,000.00
TAX (8.25%): $4,950.00
TOTAL DUE: $64,950.00

Payment Terms: Net 30
Late Fee: 1.5% per month

Contact: billing@quantumsolutions.com
Phone: (512) 555-0123
"""

# 2. BALANCE SHEET DATA
sample_balance_sheet = """
TECHNOVA CORPORATION
CONSOLIDATED BALANCE SHEET
As of December 31, 2024
(Amounts in thousands of USD)

ASSETS
Current Assets:
  Cash and Cash Equivalents: $450,000
  Accounts Receivable: $320,000
  Inventory: $280,000
  Prepaid Expenses: $50,000
  Total Current Assets: $1,100,000

Non-Current Assets:
  Property, Plant & Equipment: $850,000
  Intangible Assets: $300,000
  Investments: $150,000
  Total Non-Current Assets: $1,300,000

TOTAL ASSETS: $2,400,000

LIABILITIES AND SHAREHOLDERS' EQUITY
Current Liabilities:
  Accounts Payable: $210,000
  Short-term Debt: $150,000
  Accrued Expenses: $90,000
  Total Current Liabilities: $450,000

Non-Current Liabilities:
  Long-term Debt: $400,000
  Deferred Tax Liability: $100,000
  Total Non-Current Liabilities: $500,000

TOTAL LIABILITIES: $950,000

Shareholders' Equity:
  Common Stock: $200,000
  Retained Earnings: $1,250,000
  Total Shareholders' Equity: $1,450,000

TOTAL LIABILITIES AND EQUITY: $2,400,000

FINANCIAL RATIOS:
Current Ratio: 2.44
Debt-to-Equity: 0.66
Inventory Turnover: 6.5
Days Sales Outstanding: 45 days
Days Payable Outstanding: 30 days
"""

# 3. CONTRACT DATA
sample_contract_text = """
SOFTWARE LICENSE AGREEMENT

Agreement Date: October 1, 2024
Effective Date: November 1, 2024
Term: 36 months
Parties:
1. TECHNOVA CORPORATION ("Licensee")
   Address: 123 Innovation Drive, San Francisco, CA 94105
   
2. CLOUDSCAPE TECHNOLOGIES INC. ("Licensor")
   Address: 789 Cloud Way, Seattle, WA 98101

SCOPE OF LICENSE:
Licensor grants Licensee a non-exclusive license to use:
- Enterprise Cloud Platform
- AI Analytics Suite
- Data Security Module

FEES AND PAYMENT:
Annual License Fee: $250,000
Implementation Fee: $75,000 (one-time)
Support & Maintenance: 20% of license fee annually
Total Contract Value: $975,000 over 3 years

Payment Schedule:
Year 1: $325,000 (includes implementation)
Year 2: $300,000
Year 3: $350,000

TERMINATION:
Either party may terminate with 90 days written notice.
Early termination fee: 50% of remaining contract value.

SIGNATURES:
___________________________
John Smith, CFO
TechNova Corporation

___________________________
Sarah Johnson, CEO
Cloudscape Technologies Inc.
"""

# 4. PURCHASE ORDER DATA
sample_purchase_order = """
PURCHASE ORDER

PO Number: PO-2024-089
Date: November 20, 2024
Vendor: Global Hardware Solutions
Vendor ID: GHS-7890
Ship To: TechNova Corporation Warehouse, 456 Storage Lane, Reno, NV 89501

ITEMS ORDERED:
1. Server Rack Units
   SKU: SRU-4500
   Quantity: 10
   Unit Price: $8,500
   Total: $85,000
   
2. Network Switches
   SKU: NS-2400
   Quantity: 5
   Unit Price: $4,200
   Total: $21,000
   
3. Storage Arrays
   SKU: SA-8800
   Quantity: 3
   Unit Price: $12,000
   Total: $36,000
   
4. Backup Power Supplies
   SKU: BPS-1500
   Quantity: 8
   Unit Price: $3,750
   Total: $30,000

SUBTOTAL: $172,000
SHIPPING: $2,500
TAX: $13,585
ORDER TOTAL: $188,085

Delivery Terms: FOB Destination
Payment Terms: Net 45
Required Delivery Date: December 15, 2024

Approved By: Maria Rodriguez, IT Director
"""

# 5. FINANCIAL METRICS FOR CASH CONVERSION CYCLE ANALYSIS
financial_metrics_data = {
    "Q3_2024": {
        "inventory": 250000,  # in thousands
        "accounts_receivable": 280000,
        "accounts_payable": 180000,
        "cost_of_goods_sold": 1625000,  # Annualized
        "net_credit_sales": 1950000,  # Annualized
        "days_in_period": 90
    },
    "Q4_2024": {
        "inventory": 280000,  # Increased from Q3
        "accounts_receivable": 320000,  # Increased from Q3
        "accounts_payable": 210000,  # Increased from Q3
        "cost_of_goods_sold": 1750000,  # Annualized
        "net_credit_sales": 2100000,  # Annualized
        "days_in_period": 90
    }
}

# 6. BUDGET VS ACTUAL DATA BY COST CENTER
budget_variance_data = [
    {
        "cost_center": "Marketing",
        "budget": 500000,
        "actual": 620000,
        "variance": 120000,
        "variance_percent": 24.0,
        "major_expenses": [
            {"description": "Digital Advertising Campaign", "amount": 250000},
            {"description": "Agency Fees", "amount": 150000},
            {"description": "Event Sponsorship", "amount": 120000}
        ],
        "reason": "Unplanned Q4 brand campaign and higher agency costs"
    },
    {
        "cost_center": "Research & Development",
        "budget": 800000,
        "actual": 850000,
        "variance": 50000,
        "variance_percent": 6.25,
        "major_expenses": [
            {"description": "AI Model Development", "amount": 300000},
            {"description": "Prototype Testing", "amount": 250000},
            {"description": "Research Equipment", "amount": 150000}
        ],
        "reason": "Additional prototype iterations required"
    },
    {
        "cost_center": "Information Technology",
        "budget": 300000,
        "actual": 350000,
        "variance": 50000,
        "variance_percent": 16.67,
        "major_expenses": [
            {"description": "Server Hardware Upgrade", "amount": 120000},
            {"description": "Cybersecurity Software", "amount": 80000},
            {"description": "Cloud Infrastructure", "amount": 75000}
        ],
        "reason": "Unplanned hardware refresh and security upgrades"
    },
    {
        "cost_center": "Operations",
        "budget": 600000,
        "actual": 550000,
        "variance": -50000,
        "variance_percent": -8.33,
        "major_expenses": [
            {"description": "Logistics Optimization", "amount": 200000},
            {"description": "Facility Maintenance", "amount": 150000},
            {"description": "Supply Chain Management", "amount": 120000}
        ],
        "reason": "Process efficiencies and vendor renegotiation"
    }
]

# 7. SIMULATED FILE CONTENT FOR UPLOAD TESTING
# (Binary representations of the above documents)
import io

def create_test_files():
    """Create simulated file content for testing upload functionality"""
    
    # Invoice as simulated PDF content
    invoice_bytes = sample_invoice_text.encode('utf-8')
    
    # Balance sheet as simulated Excel data
    balance_sheet_df = pd.DataFrame({
        'Account': ['Cash', 'Accounts Receivable', 'Inventory', 'Accounts Payable', 
                   'Long-term Debt', 'Equity'],
        'Amount': [450000, 320000, 280000, 210000, 400000, 1450000],
        'Type': ['Asset', 'Asset', 'Asset', 'Liability', 'Liability', 'Equity']
    })
    balance_sheet_excel = io.BytesIO()
    with pd.ExcelWriter(balance_sheet_excel, engine='openpyxl') as writer:
        balance_sheet_df.to_excel(writer, sheet_name='Balance Sheet', index=False)
    balance_sheet_bytes = balance_sheet_excel.getvalue()
    
    # Purchase order as CSV
    po_df = pd.DataFrame({
        'Item': ['Server Rack Units', 'Network Switches', 'Storage Arrays', 'Backup Power'],
        'Quantity': [10, 5, 3, 8],
        'Unit_Price': [8500, 4200, 12000, 3750],
        'Total': [85000, 21000, 36000, 30000]
    })
    po_csv = po_df.to_csv(index=False).encode('utf-8')
    
    return {
        'invoice.pdf': invoice_bytes,
        'balance_sheet.xlsx': balance_sheet_bytes,
        'purchase_order.csv': po_csv,
        'contract.txt': sample_contract_text.encode('utf-8')
    }

# 8. CFO QUERY EXAMPLES
cfo_queries = [
    "Why did our cash conversion cycle increase from 60 to 75 days last quarter?",
    "Which departments exceeded their budgets in Q4 and what were the main reasons?",
    "Analyze our working capital efficiency and suggest improvements",
    "Extract all vendor information from our recent contracts",
    "Compare our current ratio with industry average of 2.0",
    "Identify the top 3 expenses that caused Marketing to go over budget",
    "Calculate our Days Sales Outstanding and recommend collection strategies",
    "What is our total contractual commitment for software licenses next year?",
    "Analyze inventory turnover trend and suggest optimization",
    "Provide a summary of all outstanding purchase orders above $50,000"
]

# 9. EXPECTED ENTITIES TO EXTRACT
expected_entities = {
    "companies": [
        "TechNova Corporation",
        "Quantum Solutions Inc.", 
        "Cloudscape Technologies Inc.",
        "Global Hardware Solutions"
    ],
    "amounts": [
        64950,  # Invoice total
        2400000,  # Total assets
        975000,  # Contract value
        188085  # PO total
    ],
    "dates": [
        "January 15, 2024",
        "February 15, 2024", 
        "December 31, 2024",
        "October 1, 2024",
        "November 1, 2024",
        "November 20, 2024",
        "December 15, 2024"
    ],
    "products": [
        "Cloud Infrastructure Services",
        "AI Model Training Hours",
        "Technical Support",
        "Enterprise Cloud Platform",
        "AI Analytics Suite",
        "Server Rack Units",
        "Network Switches"
    ]
}

# ============================================
# TEST FUNCTIONS FOR THE AGENT
# ============================================
def run_comprehensive_tests():
    """Run comprehensive tests on the financial agent"""
    
    print("ğŸ§ª COMPREHENSIVE FINANCIAL AGENT TESTS")
    print("=" * 60)
    
    # Test 1: Document Processing
    print("\nâ¿¡ TESTING DOCUMENT PROCESSING:")
    print("-" * 40)
    
    # Simulate processing each document type
    test_files = create_test_files()
    
    print(f"Generated {len(test_files)} test files:")
    for filename, content in test_files.items():
        print(f"  âœ“ {filename}: {len(content):,} bytes")
    
    # Test 2: Entity Extraction
    print("\nâ¿¢ TESTING ENTITY EXTRACTION:")
    print("-" * 40)
    
    # Combine all document text
    all_text = "\n".join([
        sample_invoice_text,
        sample_balance_sheet,
        sample_contract_text,
        sample_purchase_order
    ])
    
    print(f"Combined document text: {len(all_text):,} characters")
    print(f"Expected to extract:")
    print(f"  â€¢ {len(expected_entities['companies'])} companies")
    print(f"  â€¢ {len(expected_entities['amounts'])} monetary amounts")
    print(f"  â€¢ {len(expected_entities['dates'])} dates")
    print(f"  â€¢ {len(expected_entities['products'])} products")
    
    # Test 3: Cash Conversion Cycle Analysis
    print("\nâ¿£ TESTING CASH CONVERSION CYCLE ANALYSIS:")
    print("-" * 40)
    
    q3 = financial_metrics_data["Q3_2024"]
    q4 = financial_metrics_data["Q4_2024"]
    
    # Calculate DIO (Days Inventory Outstanding)
    q3_dio = (q3["inventory"] / q3["cost_of_goods_sold"]) * 365
    q4_dio = (q4["inventory"] / q4["cost_of_goods_sold"]) * 365
    
    # Calculate DSO (Days Sales Outstanding)  
    q3_dso = (q3["accounts_receivable"] / q3["net_credit_sales"]) * 365
    q4_dso = (q4["accounts_receivable"] / q4["net_credit_sales"]) * 365
    
    # Calculate DPO (Days Payables Outstanding)
    q3_dpo = (q3["accounts_payable"] / q3["cost_of_goods_sold"]) * 365
    q4_dpo = (q4["accounts_payable"] / q4["cost_of_goods_sold"]) * 365
    
    # Calculate CCC
    q3_ccc = q3_dio + q3_dso - q3_dpo
    q4_ccc = q4_dio + q4_dso - q4_dpo
    
    print(f"Q3 2024 CCC: {q3_ccc:.1f} days")
    print(f"Q4 2024 CCC: {q4_ccc:.1f} days")
    print(f"Change: +{q4_ccc - q3_ccc:.1f} days (WORSENING)")
    print(f"\nComponent Analysis:")
    print(f"  DIO: {q3_dio:.1f} â†’ {q4_dio:.1f} days (+{q4_dio - q3_dio:.1f})")
    print(f"  DSO: {q3_dso:.1f} â†’ {q4_dso:.1f} days (+{q4_dso - q3_dso:.1f})")
    print(f"  DPO: {q3_dpo:.1f} â†’ {q4_dpo:.1f} days (+{q4_dpo - q3_dpo:.1f})")
    
    # Test 4: Budget Variance Analysis
    print("\nâ¿¤ TESTING BUDGET VARIANCE ANALYSIS:")
    print("-" * 40)
    
    total_budget = sum(item["budget"] for item in budget_variance_data)
    total_actual = sum(item["actual"] for item in budget_variance_data)
    total_variance = total_actual - total_budget
    
    print(f"Total Budget: ${total_budget:,}")
    print(f"Total Actual: ${total_actual:,}")
    print(f"Total Variance: ${total_variance:,} ({total_variance/total_budget*100:.1f}%)")
    
    print(f"\nDepartments Over Budget:")
    for dept in budget_variance_data:
        if dept["variance"] > 0:
            print(f"  â€¢ {dept['cost_center']}: +${dept['variance']:,} ({dept['variance_percent']}%)")
            print(f"    Reason: {dept['reason']}")
    
    # Test 5: TOON Format Demonstration
    print("\nâ¿¥ TESTING TOON FORMAT EFFICIENCY:")
    print("-" * 40)
    
    # Create sample financial data
    sample_data = {
        "quarter": "Q4 2024",
        "revenue": 1250000,
        "expenses": 950000,
        "profit": 300000,
        "departments": [
            {"name": "Marketing", "budget": 500000, "actual": 620000},
            {"name": "R&D", "budget": 800000, "actual": 850000},
            {"name": "IT", "budget": 300000, "actual": 350000}
        ]
    }
    
    # Convert to JSON and TOON for comparison
    import json
    json_str = json.dumps(sample_data, indent=2)
    toon_str = TOONFormatter.dict_to_toon(sample_data)
    
    print(f"JSON size: {len(json_str)} characters")
    print(f"TOON size: {len(toon_str)} characters")
    print(f"Token savings: {(1 - len(toon_str)/len(json_str))*100:.1f}%")
    
    print(f"\nTOON Output Preview:")
    print(toon_str[:200] + "...")
    
    print("\n" + "=" * 60)
    print("âœ… ALL TEST DATA READY FOR AGENT VALIDATION!")
    print("\nTo test the agent, use:")
    print("1. await agent.process_query('test_user', cfo_queries[0])")
    print("2. tools.extract_financial_entities(all_text)")
    print("3. tools.analyze_cash_conversion_cycle()")
    print("4. tools.analyze_budget_variances()")

# ============================================
# QUICK TEST SCRIPT
# ============================================
def quick_test_agent_tools():
    """Quick test of agent tools with the provided data"""
    
    print("ğŸš€ QUICK AGENT TOOL TEST")
    print("=" * 50)
    
    # Test entity extraction
    print("\nğŸ“„ Testing Entity Extraction from Invoice:")
    print("-" * 40)
    
    # Use the entity extraction tool
    from your_agent_module import FinancialAnalysisTools
    
    entities = FinancialAnalysisTools.extract_financial_entities(
        document_text=sample_invoice_text,
        entity_types=["companies", "amounts", "dates"]
    )
    
    print(f"Found {entities['entities_found']} entities")
    print(f"Companies: {len(entities['entities_by_type'].get('companies', []))}")
    print(f"Amounts: {len(entities['entities_by_type'].get('amounts', []))}")
    
    # Test CCC analysis
    print("\nğŸ’° Testing Cash Conversion Cycle Analysis:")
    print("-" * 40)
    
    ccc_result = FinancialAnalysisTools.analyze_cash_conversion_cycle(
        period="Q4 2024",
        inventory_days=45.0,
        receivables_days=60.0,
        payables_days=30.0
    )
    
    print(f"CCC: {ccc_result['metrics']['cash_conversion_cycle']} days")
    print(f"Root Causes: {ccc_result['root_causes'][0]}")
    
    # Test budget analysis
    print("\nğŸ“Š Testing Budget Variance Analysis:")
    print("-" * 40)
    
    budget_result = FinancialAnalysisTools.analyze_budget_variances()
    
    overspent = [d for d in budget_result['detailed_analysis'] if d['variance'] > 0]
    print(f"Departments over budget: {len(overspent)}")
    for dept in overspent[:2]:
        print(f"  â€¢ {dept['name']}: +${dept['variance']:,}")
    
    print("\nâœ… Quick tests completed successfully!")


if __name__ == "__main__":
    # Run comprehensive tests
    run_comprehensive_tests()
    
    # Show Kaggle example
    print("\n" + "=" * 60)
    
    # Provide summary
    print("\n" + "=" * 60)
    print("ğŸ“¦ TEST DATA SUMMARY")
    print("-" * 60)
    print(f"â€¢ Invoice: {len(sample_invoice_text):,} chars")
    print(f"â€¢ Balance Sheet: {len(sample_balance_sheet):,} chars") 
    print(f"â€¢ Contract: {len(sample_contract_text):,} chars")
    print(f"â€¢ Purchase Order: {len(sample_purchase_order):,} chars")
    print(f"â€¢ CFO Queries: {len(cfo_queries)} examples")
    print(f"â€¢ Budget Data: {len(budget_variance_data)} departments")
    print(f"â€¢ Financial Metrics: Q3 & Q4 2024")
    



import json
import pandas as pd
import io
from datetime import datetime

OUTPUT_FILE = "financial_agent_test_output.txt"

# Utility writer
def write(text):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def create_test_files():
    invoice_bytes = sample_invoice_text.encode("utf-8")

    balance_sheet_df = pd.DataFrame({
        'Account': ['Cash', 'Accounts Receivable', 'Inventory', 'Accounts Payable',
                    'Long-term Debt', 'Equity'],
        'Amount': [450000, 320000, 280000, 210000, 400000, 1450000],
        'Type': ['Asset', 'Asset', 'Asset', 'Liability', 'Liability', 'Equity']
    })

    balance_sheet_excel = io.BytesIO()
    with pd.ExcelWriter(balance_sheet_excel, engine='openpyxl') as writer:
        balance_sheet_df.to_excel(writer, sheet_name='Balance Sheet', index=False)
    balance_sheet_bytes = balance_sheet_excel.getvalue()

    po_df = pd.DataFrame({
        'Item': ['Server Rack Units', 'Network Switches', 'Storage Arrays', 'Backup Power'],
        'Quantity': [10, 5, 3, 8],
        'Unit_Price': [8500, 4200, 12000, 3750],
        'Total': [85000, 21000, 36000, 30000]
    })
    po_csv = po_df.to_csv(index=False).encode('utf-8')

    return {
        'invoice.pdf': invoice_bytes,
        'balance_sheet.xlsx': balance_sheet_bytes,
        'purchase_order.csv': po_csv,
        'contract.txt': sample_contract_text.encode('utf-8')
    }


# =============================
# MAIN TEST EXECUTION
# =============================
def run_comprehensive_tests():
    open(OUTPUT_FILE, "w").close()  # clear file

    write("ğŸ§ª COMPREHENSIVE FINANCIAL AGENT TESTS")
    write("=" * 60)

    # TEST 1 â€” DOCUMENT PROCESSING
    write("\nâ¿¡ TESTING DOCUMENT PROCESSING:")
    write("-" * 40)

    test_files = create_test_files()
    write(f"Generated {len(test_files)} test files:")
    for filename, content in test_files.items():
        write(f"  âœ“ {filename}: {len(content):,} bytes")

    # TEST 2 â€” ENTITY EXTRACTION (text only)
    write("\nâ¿¢ TESTING ENTITY EXTRACTION:")
    write("-" * 40)

    all_text = "\n".join([
        sample_invoice_text,
        sample_balance_sheet,
        sample_contract_text,
        sample_purchase_order
    ])

    write(f"Combined document text length: {len(all_text):,} characters")

    write("\nExpected Entity Types:")
    write("  â€¢ companies")
    write("  â€¢ amounts")
    write("  â€¢ dates")
    write("  â€¢ products")

    # TEST 3 â€” CASH CONVERSION CYCLE ANALYSIS
    write("\nâ¿£ TESTING CASH CONVERSION CYCLE ANALYSIS:")
    write("-" * 40)

    q3 = financial_metrics_data["Q3_2024"]
    q4 = financial_metrics_data["Q4_2024"]

    q3_dio = (q3["inventory"] / q3["cost_of_goods_sold"]) * 365
    q4_dio = (q4["inventory"] / q4["cost_of_goods_sold"]) * 365

    q3_dso = (q3["accounts_receivable"] / q3["net_credit_sales"]) * 365
    q4_dso = (q4["accounts_receivable"] / q4["net_credit_sales"]) * 365

    q3_dpo = (q3["accounts_payable"] / q3["cost_of_goods_sold"]) * 365
    q4_dpo = (q4["accounts_payable"] / q4["cost_of_goods_sold"]) * 365

    q3_ccc = q3_dio + q3_dso - q3_dpo
    q4_ccc = q4_dio + q4_dso - q4_dpo

    write(f"Q3 CCC: {q3_ccc:.1f} days")
    write(f"Q4 CCC: {q4_ccc:.1f} days")
    write(f"Change: +{q4_ccc - q3_ccc:.1f} days")

    write("\nComponent Changes:")
    write(f"  DIO: {q3_dio:.1f} â†’ {q4_dio:.1f} (+{q4_dio - q3_dio:.1f})")
    write(f"  DSO: {q3_dso:.1f} â†’ {q4_dso:.1f} (+{q4_dso - q3_dso:.1f})")
    write(f"  DPO: {q3_dpo:.1f} â†’ {q4_dpo:.1f} (+{q4_dpo - q3_dpo:.1f})")

    # TEST 4 â€” BUDGET VARIANCE ANALYSIS
    write("\nâ¿¤ TESTING BUDGET VARIANCE ANALYSIS:")
    write("-" * 40)

    total_budget = sum(x["budget"] for x in budget_variance_data)
    total_actual = sum(x["actual"] for x in budget_variance_data)

    write(f"Total Budget: ${total_budget:,}")
    write(f"Total Actual: ${total_actual:,}")
    write(f"Variance: ${total_actual - total_budget:,}")

    write("\nDepartments Over Budget:")
    for d in budget_variance_data:
        if d["variance"] > 0:
            write(f"  â€¢ {d['cost_center']}: +${d['variance']:,}")
            write(f"    Reason: {d['reason']}")

    # TEST 5 â€” TOON formatting demo
    write("\nâ¿¥ TESTING TOON FORMAT:")
    write("-" * 40)

    sample_data = {
        "quarter": "Q4 2024",
        "revenue": 1250000,
        "expenses": 950000,
        "profit": 300000,
        "departments": [
            {"name": "Marketing", "budget": 500000, "actual": 620000},
            {"name": "R&D", "budget": 800000, "actual": 850000},
            {"name": "IT", "budget": 300000, "actual": 350000}
        ]
    }

    json_str = json.dumps(sample_data, indent=2)
    toon_str = TOONFormatter.dict_to_toon(sample_data)

    write(f"JSON size: {len(json_str)} chars")
    write(f"TOON size: {len(toon_str)} chars")

    write("\nTOON Preview:")
    write(toon_str[:200] + "...")

    write("\n" + "=" * 60)
    write("ğŸ�‰ ALL TESTS COMPLETE")
    write(f"Saved to: {OUTPUT_FILE}")





run_comprehensive_tests()


import pandas as pd
import io
import json
from datetime import datetime
import csv

# ============================================
# FINANCIAL TEST DATA FOR CORPORATE AGENT
# ============================================

# 1. SAMPLE INVOICE DATA
sample_invoice_text = """
INVOICE

Invoice Number: INV-2024-001
Date: January 15, 2024
Due Date: February 15, 2024

BILL TO:
TechNova Corporation
123 Innovation Drive
San Francisco, CA 94105

FROM:
Quantum Solutions Inc.
456 Tech Park Avenue
Austin, TX 78701

DESCRIPTION OF SERVICES:
Item 1: Cloud Infrastructure Services
Quantity: 1 month
Unit Price: $25,000.00
Total: $25,000.00

Item 2: AI Model Training Hours
Quantity: 150 hours
Unit Price: $200.00
Total: $30,000.00

Item 3: Technical Support
Quantity: 1 month
Unit Price: $5,000.00
Total: $5,000.00

SUBTOTAL: $60,000.00
TAX (8.25%): $4,950.00
TOTAL DUE: $64,950.00

Payment Terms: Net 30
Late Fee: 1.5% per month

Contact: billing@quantumsolutions.com
Phone: (512) 555-0123
"""

# Generate Invoice CSV
def generate_invoice_csv():
    invoice_data = {
        'Invoice Header': [
            ['Field', 'Value'],
            ['Invoice Number', 'INV-2024-001'],
            ['Date', 'January 15, 2024'],
            ['Due Date', 'February 15, 2024'],
            ['Bill To', 'TechNova Corporation'],
            ['From', 'Quantum Solutions Inc.'],
            ['Subtotal', '$60,000.00'],
            ['Tax (8.25%)', '$4,950.00'],
            ['Total Due', '$64,950.00'],
            ['Payment Terms', 'Net 30'],
            ['Contact', 'billing@quantumsolutions.com'],
            ['Phone', '(512) 555-0123']
        ],
        'Line Items': [
            ['Item', 'Description', 'Quantity', 'Unit Price', 'Total'],
            ['1', 'Cloud Infrastructure Services', '1 month', '$25,000.00', '$25,000.00'],
            ['2', 'AI Model Training Hours', '150 hours', '$200.00', '$30,000.00'],
            ['3', 'Technical Support', '1 month', '$5,000.00', '$5,000.00'],
            ['', '', '', 'SUBTOTAL', '$60,000.00']
        ]
    }
    
    # Save to CSV
    with open('invoice_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(invoice_data['Invoice Header'])
    
    with open('invoice_line_items.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(invoice_data['Line Items'])
    
    return invoice_data

# 2. BALANCE SHEET DATA
sample_balance_sheet = """
TECHNOVA CORPORATION
CONSOLIDATED BALANCE SHEET
As of December 31, 2024
(Amounts in thousands of USD)

ASSETS
Current Assets:
  Cash and Cash Equivalents: $450,000
  Accounts Receivable: $320,000
  Inventory: $280,000
  Prepaid Expenses: $50,000
  Total Current Assets: $1,100,000

Non-Current Assets:
  Property, Plant & Equipment: $850,000
  Intangible Assets: $300,000
  Investments: $150,000
  Total Non-Current Assets: $1,300,000

TOTAL ASSETS: $2,400,000

LIABILITIES AND SHAREHOLDERS' EQUITY
Current Liabilities:
  Accounts Payable: $210,000
  Short-term Debt: $150,000
  Accrued Expenses: $90,000
  Total Current Liabilities: $450,000

Non-Current Liabilities:
  Long-term Debt: $400,000
  Deferred Tax Liability: $100,000
  Total Non-Current Liabilities: $500,000

TOTAL LIABILITIES: $950,000

Shareholders' Equity:
  Common Stock: $200,000
  Retained Earnings: $1,250,000
  Total Shareholders' Equity: $1,450,000

TOTAL LIABILITIES AND EQUITY: $2,400,000

FINANCIAL RATIOS:
Current Ratio: 2.44
Debt-to-Equity: 0.66
Inventory Turnover: 6.5
Days Sales Outstanding: 45 days
Days Payable Outstanding: 30 days
"""

# Generate Balance Sheet CSV
def generate_balance_sheet_csv():
    balance_sheet_data = {
        'Assets': [
            ['Category', 'Account', 'Amount (in thousands)'],
            ['Current Assets', 'Cash and Cash Equivalents', '450,000'],
            ['Current Assets', 'Accounts Receivable', '320,000'],
            ['Current Assets', 'Inventory', '280,000'],
            ['Current Assets', 'Prepaid Expenses', '50,000'],
            ['Current Assets', 'Total Current Assets', '1,100,000'],
            ['Non-Current Assets', 'Property, Plant & Equipment', '850,000'],
            ['Non-Current Assets', 'Intangible Assets', '300,000'],
            ['Non-Current Assets', 'Investments', '150,000'],
            ['Non-Current Assets', 'Total Non-Current Assets', '1,300,000'],
            ['', 'TOTAL ASSETS', '2,400,000']
        ],
        'Liabilities and Equity': [
            ['Category', 'Account', 'Amount (in thousands)'],
            ['Current Liabilities', 'Accounts Payable', '210,000'],
            ['Current Liabilities', 'Short-term Debt', '150,000'],
            ['Current Liabilities', 'Accrued Expenses', '90,000'],
            ['Current Liabilities', 'Total Current Liabilities', '450,000'],
            ['Non-Current Liabilities', 'Long-term Debt', '400,000'],
            ['Non-Current Liabilities', 'Deferred Tax Liability', '100,000'],
            ['Non-Current Liabilities', 'Total Non-Current Liabilities', '500,000'],
            ['', 'TOTAL LIABILITIES', '950,000'],
            ['Shareholders Equity', 'Common Stock', '200,000'],
            ['Shareholders Equity', 'Retained Earnings', '1,250,000'],
            ['Shareholders Equity', 'Total Shareholders Equity', '1,450,000'],
            ['', 'TOTAL LIABILITIES AND EQUITY', '2,400,000']
        ],
        'Financial Ratios': [
            ['Ratio', 'Value', 'Unit'],
            ['Current Ratio', '2.44', 'ratio'],
            ['Debt-to-Equity', '0.66', 'ratio'],
            ['Inventory Turnover', '6.5', 'times'],
            ['Days Sales Outstanding', '45', 'days'],
            ['Days Payable Outstanding', '30', 'days']
        ]
    }
    
    # Save to CSV files
    for sheet_name, data in balance_sheet_data.items():
        filename = f'balance_sheet_{sheet_name.lower().replace(" ", "_")}.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    return balance_sheet_data

# 3. CONTRACT DATA
sample_contract_text = """
SOFTWARE LICENSE AGREEMENT

Agreement Date: October 1, 2024
Effective Date: November 1, 2024
Term: 36 months
Parties:
1. TECHNOVA CORPORATION ("Licensee")
   Address: 123 Innovation Drive, San Francisco, CA 94105
   
2. CLOUDSCAPE TECHNOLOGIES INC. ("Licensor")
   Address: 789 Cloud Way, Seattle, WA 98101

SCOPE OF LICENSE:
Licensor grants Licensee a non-exclusive license to use:
- Enterprise Cloud Platform
- AI Analytics Suite
- Data Security Module

FEES AND PAYMENT:
Annual License Fee: $250,000
Implementation Fee: $75,000 (one-time)
Support & Maintenance: 20% of license fee annually
Total Contract Value: $975,000 over 3 years

Payment Schedule:
Year 1: $325,000 (includes implementation)
Year 2: $300,000
Year 3: $350,000

TERMINATION:
Either party may terminate with 90 days written notice.
Early termination fee: 50% of remaining contract value.

SIGNATURES:
___________________________
John Smith, CFO
TechNova Corporation

___________________________
Sarah Johnson, CEO
Cloudscape Technologies Inc.
"""

# Generate Contract CSV
def generate_contract_csv():
    contract_data = [
        ['Contract Field', 'Value'],
        ['Agreement Date', 'October 1, 2024'],
        ['Effective Date', 'November 1, 2024'],
        ['Term', '36 months'],
        ['Licensee Name', 'TECHNOVA CORPORATION'],
        ['Licensee Address', '123 Innovation Drive, San Francisco, CA 94105'],
        ['Licensor Name', 'CLOUDSCAPE TECHNOLOGIES INC.'],
        ['Licensor Address', '789 Cloud Way, Seattle, WA 98101'],
        ['Annual License Fee', '$250,000'],
        ['Implementation Fee', '$75,000'],
        ['Support & Maintenance', '20% of license fee annually'],
        ['Total Contract Value', '$975,000'],
        ['Payment Schedule Year 1', '$325,000'],
        ['Payment Schedule Year 2', '$300,000'],
        ['Payment Schedule Year 3', '$350,000'],
        ['Termination Notice Period', '90 days'],
        ['Early Termination Fee', '50% of remaining contract value'],
        ['Licensee Signatory', 'John Smith, CFO'],
        ['Licensor Signatory', 'Sarah Johnson, CEO']
    ]
    
    with open('contract_details.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(contract_data)
    
    # License scope data
    license_scope = [
        ['Module', 'Type'],
        ['Enterprise Cloud Platform', 'Software'],
        ['AI Analytics Suite', 'Software'],
        ['Data Security Module', 'Software']
    ]
    
    with open('contract_license_scope.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(license_scope)
    
    return contract_data

# 4. PURCHASE ORDER DATA
sample_purchase_order = """
PURCHASE ORDER

PO Number: PO-2024-089
Date: November 20, 2024
Vendor: Global Hardware Solutions
Vendor ID: GHS-7890
Ship To: TechNova Corporation Warehouse, 456 Storage Lane, Reno, NV 89501

ITEMS ORDERED:
1. Server Rack Units
   SKU: SRU-4500
   Quantity: 10
   Unit Price: $8,500
   Total: $85,000
   
2. Network Switches
   SKU: NS-2400
   Quantity: 5
   Unit Price: $4,200
   Total: $21,000
   
3. Storage Arrays
   SKU: SA-8800
   Quantity: 3
   Unit Price: $12,000
   Total: $36,000
   
4. Backup Power Supplies
   SKU: BPS-1500
   Quantity: 8
   Unit Price: $3,750
   Total: $30,000

SUBTOTAL: $172,000
SHIPPING: $2,500
TAX: $13,585
ORDER TOTAL: $188,085

Delivery Terms: FOB Destination
Payment Terms: Net 45
Required Delivery Date: December 15, 2024

Approved By: Maria Rodriguez, IT Director
"""

# Generate Purchase Order CSV
def generate_purchase_order_csv():
    po_header = [
        ['Field', 'Value'],
        ['PO Number', 'PO-2024-089'],
        ['Date', 'November 20, 2024'],
        ['Vendor', 'Global Hardware Solutions'],
        ['Vendor ID', 'GHS-7890'],
        ['Ship To', 'TechNova Corporation Warehouse, 456 Storage Lane, Reno, NV 89501'],
        ['Subtotal', '$172,000'],
        ['Shipping', '$2,500'],
        ['Tax', '$13,585'],
        ['Order Total', '$188,085'],
        ['Delivery Terms', 'FOB Destination'],
        ['Payment Terms', 'Net 45'],
        ['Required Delivery Date', 'December 15, 2024'],
        ['Approved By', 'Maria Rodriguez, IT Director']
    ]
    
    po_items = [
        ['Item', 'Description', 'SKU', 'Quantity', 'Unit Price', 'Total'],
        ['1', 'Server Rack Units', 'SRU-4500', '10', '$8,500', '$85,000'],
        ['2', 'Network Switches', 'NS-2400', '5', '$4,200', '$21,000'],
        ['3', 'Storage Arrays', 'SA-8800', '3', '$12,000', '$36,000'],
        ['4', 'Backup Power Supplies', 'BPS-1500', '8', '$3,750', '$30,000'],
        ['', '', '', '', 'SUBTOTAL', '$172,000']
    ]
    
    with open('purchase_order_header.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(po_header)
    
    with open('purchase_order_items.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(po_items)
    
    return {'header': po_header, 'items': po_items}

# 5. FINANCIAL METRICS FOR CASH CONVERSION CYCLE ANALYSIS
financial_metrics_data = {
    "Q3_2024": {
        "inventory": 250000,  # in thousands
        "accounts_receivable": 280000,
        "accounts_payable": 180000,
        "cost_of_goods_sold": 1625000,  # Annualized
        "net_credit_sales": 1950000,  # Annualized
        "days_in_period": 90
    },
    "Q4_2024": {
        "inventory": 280000,  # Increased from Q3
        "accounts_receivable": 320000,  # Increased from Q3
        "accounts_payable": 210000,  # Increased from Q3
        "cost_of_goods_sold": 1750000,  # Annualized
        "net_credit_sales": 2100000,  # Annualized
        "days_in_period": 90
    }
}

# Generate Financial Metrics CSV
def generate_financial_metrics_csv():
    metrics_data = []
    for period, metrics in financial_metrics_data.items():
        row = [period]
        row.extend([metrics[key] for key in ['inventory', 'accounts_receivable', 'accounts_payable', 
                                            'cost_of_goods_sold', 'net_credit_sales', 'days_in_period']])
        metrics_data.append(row)
    
    headers = ['Period', 'Inventory', 'Accounts Receivable', 'Accounts Payable', 
              'Cost of Goods Sold', 'Net Credit Sales', 'Days in Period']
    
    all_data = [headers] + metrics_data
    
    with open('financial_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)
    
    # Calculate CCC metrics
    ccc_data = []
    for period, metrics in financial_metrics_data.items():
        dio = (metrics['inventory'] / metrics['cost_of_goods_sold']) * 365
        dso = (metrics['accounts_receivable'] / metrics['net_credit_sales']) * 365
        dpo = (metrics['accounts_payable'] / metrics['cost_of_goods_sold']) * 365
        ccc = dio + dso - dpo
        
        ccc_data.append([
            period,
            round(dio, 2),
            round(dso, 2),
            round(dpo, 2),
            round(ccc, 2)
        ])
    
    ccc_headers = ['Period', 'DIO (Days)', 'DSO (Days)', 'DPO (Days)', 'CCC (Days)']
    ccc_all_data = [ccc_headers] + ccc_data
    
    with open('cash_conversion_cycle.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(ccc_all_data)
    
    return {'metrics': all_data, 'ccc': ccc_all_data}

# 6. BUDGET VS ACTUAL DATA BY COST CENTER
budget_variance_data = [
    {
        "cost_center": "Marketing",
        "budget": 500000,
        "actual": 620000,
        "variance": 120000,
        "variance_percent": 24.0,
        "major_expenses": [
            {"description": "Digital Advertising Campaign", "amount": 250000},
            {"description": "Agency Fees", "amount": 150000},
            {"description": "Event Sponsorship", "amount": 120000}
        ],
        "reason": "Unplanned Q4 brand campaign and higher agency costs"
    },
    {
        "cost_center": "Research & Development",
        "budget": 800000,
        "actual": 850000,
        "variance": 50000,
        "variance_percent": 6.25,
        "major_expenses": [
            {"description": "AI Model Development", "amount": 300000},
            {"description": "Prototype Testing", "amount": 250000},
            {"description": "Research Equipment", "amount": 150000}
        ],
        "reason": "Additional prototype iterations required"
    },
    {
        "cost_center": "Information Technology",
        "budget": 300000,
        "actual": 350000,
        "variance": 50000,
        "variance_percent": 16.67,
        "major_expenses": [
            {"description": "Server Hardware Upgrade", "amount": 120000},
            {"description": "Cybersecurity Software", "amount": 80000},
            {"description": "Cloud Infrastructure", "amount": 75000}
        ],
        "reason": "Unplanned hardware refresh and security upgrades"
    },
    {
        "cost_center": "Operations",
        "budget": 600000,
        "actual": 550000,
        "variance": -50000,
        "variance_percent": -8.33,
        "major_expenses": [
            {"description": "Logistics Optimization", "amount": 200000},
            {"description": "Facility Maintenance", "amount": 150000},
            {"description": "Supply Chain Management", "amount": 120000}
        ],
        "reason": "Process efficiencies and vendor renegotiation"
    }
]

# Generate Budget Variance CSV
def generate_budget_variance_csv():
    # Main budget variance data
    headers = ['Cost Center', 'Budget ($)', 'Actual ($)', 'Variance ($)', 'Variance (%)', 'Reason']
    rows = []
    
    for item in budget_variance_data:
        rows.append([
            item['cost_center'],
            f"${item['budget']:,}",
            f"${item['actual']:,}",
            f"${item['variance']:,}",
            f"{item['variance_percent']}%",
            item['reason']
        ])
    
    all_data = [headers] + rows
    
    with open('budget_variance_summary.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)
    
    # Detailed expense data
    expense_headers = ['Cost Center', 'Expense Description', 'Amount ($)']
    expense_rows = []
    
    for item in budget_variance_data:
        for expense in item['major_expenses']:
            expense_rows.append([
                item['cost_center'],
                expense['description'],
                f"${expense['amount']:,}"
            ])
    
    expense_all_data = [expense_headers] + expense_rows
    
    with open('budget_expense_details.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(expense_all_data)
    
    # Summary statistics
    total_budget = sum(item['budget'] for item in budget_variance_data)
    total_actual = sum(item['actual'] for item in budget_variance_data)
    total_variance = total_actual - total_budget
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Budget', f"${total_budget:,}"],
        ['Total Actual', f"${total_actual:,}"],
        ['Total Variance', f"${total_variance:,}"],
        ['Overall Variance %', f"{(total_variance/total_budget*100):.2f}%"],
        ['Departments Over Budget', str(len([i for i in budget_variance_data if i['variance'] > 0]))],
        ['Departments Under Budget', str(len([i for i in budget_variance_data if i['variance'] < 0]))],
        ['Highest Variance %', f"{max(i['variance_percent'] for i in budget_variance_data)}%"],
        ['Lowest Variance %', f"{min(i['variance_percent'] for i in budget_variance_data)}%"]
    ]
    
    with open('budget_summary_stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(summary_data)
    
    return {'summary': all_data, 'expenses': expense_all_data, 'stats': summary_data}

# 8. CFO QUERY EXAMPLES
cfo_queries = [
    "Why did our cash conversion cycle increase from 60 to 75 days last quarter?",
    "Which departments exceeded their budgets in Q4 and what were the main reasons?",
    "Analyze our working capital efficiency and suggest improvements",
    "Extract all vendor information from our recent contracts",
    "Compare our current ratio with industry average of 2.0",
    "Identify the top 3 expenses that caused Marketing to go over budget",
    "Calculate our Days Sales Outstanding and recommend collection strategies",
    "What is our total contractual commitment for software licenses next year?",
    "Analyze inventory turnover trend and suggest optimization",
    "Provide a summary of all outstanding purchase orders above $50,000"
]

# Generate CFO Queries CSV
def generate_cfo_queries_csv():
    headers = ['Query ID', 'CFO Query', 'Category']
    
    categories = {
        0: 'Cash Flow Analysis',
        1: 'Budget Analysis',
        2: 'Working Capital',
        3: 'Vendor Management',
        4: 'Financial Ratios',
        5: 'Expense Analysis',
        6: 'Accounts Receivable',
        7: 'Contract Management',
        8: 'Inventory Management',
        9: 'Purchase Orders'
    }
    
    rows = []
    for i, query in enumerate(cfo_queries):
        rows.append([f'Q{i+1:03d}', query, categories.get(i, 'General Analysis')])
    
    all_data = [headers] + rows
    
    with open('cfo_queries.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)
    
    return all_data

#9. EXPECTED ENTITIES TO EXTRACT
expected_entities = {
    "companies": [
        "TechNova Corporation",
        "Quantum Solutions Inc.", 
        "Cloudscape Technologies Inc.",
        "Global Hardware Solutions"
    ],
    "amounts": [
        64950,  # Invoice total
        2400000,  # Total assets
        975000,  # Contract value
        188085  # PO total
    ],
    "dates": [
        "January 15, 2024",
        "February 15, 2024", 
        "December 31, 2024",
        "October 1, 2024",
        "November 1, 2024",
        "November 20, 2024",
        "December 15, 2024"
    ],
    "products": [
        "Cloud Infrastructure Services",
        "AI Model Training Hours",
        "Technical Support",
        "Enterprise Cloud Platform",
        "AI Analytics Suite",
        "Server Rack Units",
        "Network Switches"
    ]
}

# Generate Expected Entities CSV
def generate_expected_entities_csv():
    # Flatten the dictionary into rows
    headers = ['Entity Type', 'Entity Value', 'Source Document']
    
    # Map entities to their likely source documents
    source_map = {
        'companies': ['All Documents'],
        'amounts': ['Invoice', 'Balance Sheet', 'Contract', 'Purchase Order'],
        'dates': ['Invoice', 'Contract', 'Purchase Order'],
        'products': ['Invoice', 'Contract', 'Purchase Order']
    }
    
    rows = []
    for entity_type, entities in expected_entities.items():
        for i, entity in enumerate(entities):
            source = source_map.get(entity_type, ['Various'])[i % len(source_map.get(entity_type, ['Various']))]
            rows.append([entity_type, entity, source])
    
    all_data = [headers] + rows
    
    with open('expected_entities.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)
    
    # Also create summary by type
    summary_headers = ['Entity Type', 'Count', 'Examples']
    summary_rows = []
    
    for entity_type, entities in expected_entities.items():
        examples = ', '.join(str(e) for e in entities[:3]) + ('...' if len(entities) > 3 else '')
        summary_rows.append([entity_type, len(entities), examples])
    
    summary_data = [summary_headers] + summary_rows
    
    with open('entity_summary.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(summary_data)
    
    return {'detailed': all_data, 'summary': summary_data}

# ============================================
# MAIN FUNCTION TO GENERATE ALL CSV FILES
# ============================================
def generate_all_csv_files():
    """Generate all CSV files for testing"""
    
    print("ğŸ“Š GENERATING FINANCIAL TEST DATA CSV FILES")
    print("=" * 60)
    
    files_generated = []
    
    # Generate each CSV file
    print("\nâ¿¡ Generating Invoice CSV files...")
    invoice_data = generate_invoice_csv()
    files_generated.extend(['invoice_data.csv', 'invoice_line_items.csv'])
    print(f"   âœ“ Created: invoice_data.csv ({len(invoice_data['Invoice Header'])} rows)")
    print(f"   âœ“ Created: invoice_line_items.csv ({len(invoice_data['Line Items'])} rows)")
    
    print("\nâ¿¢ Generating Balance Sheet CSV files...")
    balance_data = generate_balance_sheet_csv()
    files_generated.extend(['balance_sheet_assets.csv', 'balance_sheet_liabilities_and_equity.csv', 'balance_sheet_financial_ratios.csv'])
    print(f"   âœ“ Created: balance_sheet_assets.csv ({len(balance_data['Assets'])} rows)")
    print(f"   âœ“ Created: balance_sheet_liabilities_and_equity.csv ({len(balance_data['Liabilities and Equity'])} rows)")
    print(f"   âœ“ Created: balance_sheet_financial_ratios.csv ({len(balance_data['Financial Ratios'])} rows)")
    
    print("\nâ¿£ Generating Contract CSV files...")
    contract_data = generate_contract_csv()
    files_generated.extend(['contract_details.csv', 'contract_license_scope.csv'])
    print(f"   âœ“ Created: contract_details.csv ({len(contract_data)} rows)")
    print(f"   âœ“ Created: contract_license_scope.csv (4 rows)")
    
    print("\nâ¿¤ Generating Purchase Order CSV files...")
    po_data = generate_purchase_order_csv()
    files_generated.extend(['purchase_order_header.csv', 'purchase_order_items.csv'])
    print(f"   âœ“ Created: purchase_order_header.csv ({len(po_data['header'])} rows)")
    print(f"   âœ“ Created: purchase_order_items.csv ({len(po_data['items'])} rows)")
    
    print("\nâ¿¥ Generating Financial Metrics CSV files...")
    metrics_data = generate_financial_metrics_csv()
    files_generated.extend(['financial_metrics.csv', 'cash_conversion_cycle.csv'])
    print(f"   âœ“ Created: financial_metrics.csv (3 rows)")
    print(f"   âœ“ Created: cash_conversion_cycle.csv (3 rows)")
    
    print("\nâ¿¦ Generating Budget Variance CSV files...")
    budget_data = generate_budget_variance_csv()
    files_generated.extend(['budget_variance_summary.csv', 'budget_expense_details.csv', 'budget_summary_stats.csv'])
    print(f"   âœ“ Created: budget_variance_summary.csv (5 rows)")
    print(f"   âœ“ Created: budget_expense_details.csv (13 rows)")
    print(f"   âœ“ Created: budget_summary_stats.csv (9 rows)")
    
    print("\nâ¿§ Generating CFO Queries CSV file...")
    queries_data = generate_cfo_queries_csv()
    files_generated.append('cfo_queries.csv')
    print(f"   âœ“ Created: cfo_queries.csv ({len(queries_data)} rows)")
    
    print("\nâ¿¨ Generating Expected Entities CSV files...")
    entities_data = generate_expected_entities_csv()
    files_generated.extend(['expected_entities.csv', 'entity_summary.csv'])
    print(f"   âœ“ Created: expected_entities.csv ({len(entities_data['detailed'])} rows)")
    print(f"   âœ“ Created: entity_summary.csv ({len(entities_data['summary'])} rows)")
    
    # Create a manifest file
    manifest_data = [
        ['File Name', 'Description', 'Rows', 'Generated At'],
        ['invoice_data.csv', 'Invoice header information', len(invoice_data['Invoice Header']), datetime.now().isoformat()],
        ['invoice_line_items.csv', 'Invoice line items', len(invoice_data['Line Items']), datetime.now().isoformat()],
        ['balance_sheet_assets.csv', 'Balance sheet assets section', len(balance_data['Assets']), datetime.now().isoformat()],
        ['balance_sheet_liabilities_and_equity.csv', 'Balance sheet liabilities and equity', len(balance_data['Liabilities and Equity']), datetime.now().isoformat()],
        ['balance_sheet_financial_ratios.csv', 'Financial ratios from balance sheet', len(balance_data['Financial Ratios']), datetime.now().isoformat()],
        ['contract_details.csv', 'Contract details and terms', len(contract_data), datetime.now().isoformat()],
        ['contract_license_scope.csv', 'Software modules in contract', 4, datetime.now().isoformat()],
        ['purchase_order_header.csv', 'Purchase order header information', len(po_data['header']), datetime.now().isoformat()],
        ['purchase_order_items.csv', 'Purchase order line items', len(po_data['items']), datetime.now().isoformat()],
        ['financial_metrics.csv', 'Quarterly financial metrics', 3, datetime.now().isoformat()],
        ['cash_conversion_cycle.csv', 'Cash conversion cycle calculations', 3, datetime.now().isoformat()],
        ['budget_variance_summary.csv', 'Budget vs actual by cost center', 5, datetime.now().isoformat()],
        ['budget_expense_details.csv', 'Detailed expense breakdown', 13, datetime.now().isoformat()],
        ['budget_summary_stats.csv', 'Budget analysis statistics', 9, datetime.now().isoformat()],
        ['cfo_queries.csv', 'Sample CFO queries for testing', len(queries_data), datetime.now().isoformat()],
        ['expected_entities.csv', 'Expected entities to extract', len(entities_data['detailed']), datetime.now().isoformat()],
        ['entity_summary.csv', 'Entity extraction summary', len(entities_data['summary']), datetime.now().isoformat()]
    ]
    
    with open('csv_files_manifest.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(manifest_data)
    
    files_generated.append('csv_files_manifest.csv')
    
    print("\n" + "=" * 60)
    print(f"âœ… GENERATED {len(files_generated)} CSV FILES")
    print("=" * 60)
    
    print("\nğŸ“� Generated Files:")
    for i, filename in enumerate(sorted(files_generated), 1):
        print(f"  {i:2d}. {filename}")
    
    print(f"\nğŸ“Š Total Files: {len(files_generated)}")
    print(f"ğŸ•� Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return files_generated

# ============================================
# READ AND DISPLAY CSV CONTENT
# ============================================
def display_csv_preview(filename, num_rows=5):
    """Display a preview of a CSV file"""
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
            
        print(f"\nğŸ“„ {filename} Preview (first {min(num_rows, len(rows))} rows):")
        print("-" * 80)
        
        for i, row in enumerate(rows[:num_rows]):
            print(f"Row {i+1}: {row}")
        
        print(f"\nTotal rows in file: {len(rows)}")
        return rows
    except FileNotFoundError:
        print(f"â�Œ File not found: {filename}")
        return None

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Generate all CSV files
    generated_files = generate_all_csv_files()
    
    # Display previews of key files
    print("\n\nğŸ”� CSV FILE PREVIEWS")
    print("=" * 60)
    
    # Preview key files
    key_files = [
        'budget_variance_summary.csv',
        'cash_conversion_cycle.csv', 
        'cfo_queries.csv',
        'entity_summary.csv'
    ]
    
    for filename in key_files:
        display_csv_preview(filename, 3)
    
    print("\n" + "=" * 60)
    print("ğŸ�¯ CSV GENERATION COMPLETE!")
    print("\nUse these CSV files to test your Corporate Financial Knowledge Graph Agent.")

