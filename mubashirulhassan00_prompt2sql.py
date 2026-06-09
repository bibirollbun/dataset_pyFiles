# Kaggle notebooks come with ADK pre-installed
# For local development: !pip install google-adk

print("âœ“ Dependencies ready")


import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, AgentTool
from google.genai import types

# Load API key from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ“ API Key loaded successfully")
except Exception as e:
    print(f"âš ï¸� Error: {e}")
    print("Make sure GOOGLE_API_KEY is added to Kaggle Secrets!")

print("âœ“ All imports successful")



# Configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Use gemini-2.5-flash-lite (stable, GA, lowest cost)
MODEL_NAME = "gemini-2.5-flash-lite"

print(f"âœ“ Model: {MODEL_NAME}")
print(f"âœ“ Retry attempts: {retry_config.attempts}")
print("âœ“ This is the same model used in all Kaggle course notebooks!")



# Data storage class
class DataStore:
    """Centralized data storage for the agent system"""
    def __init__(self):
        self.df = None
        self.filename = None
        self.upload_time = None
        
data_store = DataStore()

def load_csv_data(file_path: str) -> Dict[str, Any]:
    """
    Load CSV file into the data store.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary with status and data information
    """
    try:
        data_store.df = pd.read_csv(file_path)
        data_store.filename = os.path.basename(file_path)
        data_store.upload_time = datetime.now().isoformat()
        
        return {
            "status": "success",
            "filename": data_store.filename,
            "rows": len(data_store.df),
            "columns": len(data_store.df.columns),
            "column_names": list(data_store.df.columns)
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def get_basic_statistics() -> Dict[str, Any]:
    """
    Generate comprehensive statistical summary of the loaded dataset.
    
    Returns:
        Dictionary with descriptive statistics
    """
    if data_store.df is None:
        return {"status": "error", "error_message": "No data loaded"}
    
    try:
        df = data_store.df
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            numeric_stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "missing": int(df[col].isna().sum())
            }
        
        return {
            "status": "success",
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "numeric_stats": numeric_stats
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def query_data(query_description: str) -> Dict[str, Any]:
    """
    Execute natural language queries on the data.
    
    Args:
        query_description: Natural language description of what to find
        
    Returns:
        Dictionary with query results
    """
    if data_store.df is None:
        return {"status": "error", "error_message": "No data loaded"}
    
    try:
        df = data_store.df
        query_lower = query_description.lower()
        
        # Simple pattern matching for common queries
        if "top" in query_lower or "highest" in query_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result = df.nlargest(5, numeric_cols[0])
                return {
                    "status": "success",
                    "result_type": "top_rows",
                    "data": result.to_dict('records')[:5]
                }
        
        elif "average" in query_lower or "mean" in query_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            averages = {col: float(df[col].mean()) for col in numeric_cols}
            return {
                "status": "success",
                "result_type": "averages",
                "data": averages
            }
        
        else:
            # Default: return first 5 rows
            return {
                "status": "success",
                "result_type": "sample",
                "data": df.head(5).to_dict('records')
            }
            
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def get_data_summary() -> Dict[str, Any]:
    """
    Get a quick overview of the loaded dataset.
    
    Returns:
        Dictionary with dataset overview
    """
    if data_store.df is None:
        return {"status": "error", "error_message": "No data loaded"}
    
    try:
        df = data_store.df
        return {
            "status": "success",
            "filename": data_store.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(3).to_dict('records')
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

print("âœ“ Custom tools created:")
print("  - load_csv_data")
print("  - get_basic_statistics")
print("  - query_data")
print("  - get_data_summary")



# Recreate agents with proper instructions
MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Statistics Agent - FIXED with better instructions
statistics_agent = LlmAgent(
    name="statistics_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    description="Specialist in statistical analysis and descriptive statistics.",
    instruction="""
    You are a statistical analysis expert.
    
    When asked to analyze data:
    1. Call the get_basic_statistics tool
    2. Analyze the statistical results returned
    3. Write a clear summary of the key statistics
    4. Highlight important patterns like means, medians, and ranges
    
    Always provide a text response summarizing your findings.
    """,
    tools=[FunctionTool(func=get_basic_statistics)]
)

print(f"âœ“ Statistics Agent created with {MODEL_NAME}")

# Query Agent - FIXED
query_agent = LlmAgent(
    name="query_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    description="Specialist in answering specific data queries.",
    instruction="""
    You are a data query specialist.
    
    When asked to query data:
    1. Call the query_data tool with the user's question
    2. Analyze the results returned
    3. Provide a clear answer based on the data
    4. Include specific numbers and facts from the results
    
    Always provide a text response with your findings.
    """,
    tools=[FunctionTool(func=query_data)]
)

print(f"âœ“ Query Agent created with {MODEL_NAME}")

# Insights Agent - FIXED (no tools, just synthesis)
insights_agent = LlmAgent(
    name="insights_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    description="Specialist in generating business insights from data.",
    instruction="""
    You are a business intelligence expert.
    
    Based on the information provided to you:
    1. Identify key business trends and patterns
    2. Provide actionable recommendations
    3. Highlight opportunities and risks
    4. Focus on strategic decision-making
    
    Synthesize the information into clear business insights.
    """
)

print(f"âœ“ Insights Agent created with {MODEL_NAME}")
print(f"\nğŸ¤– All agents ready with {MODEL_NAME}!")



# Simplified coordinator - remove sub-agents temporarily to test
coordinator_agent = LlmAgent(
    name="data_analyst_coordinator",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    description="Coordinates data analysis workflow.",
    instruction="""
    You are a data analyst coordinator.
    
    When the user asks for analysis:
    1. First, call get_data_summary to understand the dataset
    2. Then, call get_basic_statistics to analyze the numbers
    3. Finally, provide a comprehensive analysis covering:
       - Dataset overview (rows, columns, data types)
       - Statistical patterns (means, distributions, ranges)
       - Business insights and recommendations
    
    Present your findings in a clear, structured format.
    """,
    tools=[
        FunctionTool(func=get_data_summary),
        FunctionTool(func=get_basic_statistics),
        FunctionTool(func=query_data)
    ]
)

print(f"âœ“ Simplified Coordinator created with {MODEL_NAME}")
print("âœ“ Using direct tools instead of sub-agents for reliability")



# Session Service for maintaining conversation state
session_service = InMemorySessionService()

print("âœ“ Session Service initialized")

# Memory Manager - Tracks user preferences and history
class MemoryManager:
    """
    Custom memory system for tracking analysis context.
    Demonstrates ADK Concept #4: Memory Management
    """
    
    def __init__(self):
        self.user_preferences = {}
        self.analysis_history = []
        self.session_metadata = {}
    
    def add_analysis(self, query: str, result: str):
        """Record an analysis in history"""
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result[:200]  # Store truncated result
        })
    
    def get_recent_analyses(self, n: int = 5) -> List[Dict]:
        """Retrieve recent analysis history"""
        return self.analysis_history[-n:]
    
    def set_preference(self, key: str, value: Any):
        """Store user preference"""
        self.user_preferences[key] = value
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve user preference"""
        return self.user_preferences.get(key, default)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context for agents"""
        return {
            "total_analyses": len(self.analysis_history),
            "recent_queries": [a["query"] for a in self.analysis_history[-3:]],
            "preferences": self.user_preferences,
            "current_dataset": data_store.filename
        }

# Initialize memory manager
memory = MemoryManager()

print("âœ“ Memory Manager initialized")
print("  Features: analysis history, user preferences, context tracking")



# Recreate runner with the fixed coordinator
from google.adk.runners import InMemoryRunner

runner = InMemoryRunner(
    agent=coordinator_agent
)

print(f"âœ“ InMemoryRunner initialized")
print("ğŸš€ System ready!")



async def run_analysis(query: str):
    """
    Run an analysis query through the agent system.
    
    Args:
        query: Natural language query about the data
    """
    print("\n" + "="*80)
    print(f"ğŸ“Š QUERY: {query}")
    print("="*80)
    
    print("\nğŸ¤– Agent Response:")
    print("-" * 80)
    
    # Run analysis using InMemoryRunner
    response_text = ""
    try:
        # Use run_debug and capture only the text output
        response = await runner.run_debug(query)
        
        # Extract just the text from the response
        if response:
            for event in response:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text = part.text
                            # Already printed by run_debug, so we don't print again
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        response_text = f"Error: {e}"
    
    print("-" * 80)
    
    # Store in memory
    if response_text:
        memory.add_analysis(query, response_text)
    
    # Don't print the raw response object
    return response_text

def display_system_status():
    """Display current system status and context"""
    print("\n" + "="*80)
    print("ğŸ“ˆ SYSTEM STATUS")
    print("="*80)
    
    # Data status
    if data_store.df is not None:
        print(f"âœ“ Dataset Loaded: {data_store.filename}")
        print(f"  Rows: {len(data_store.df):,}")
        print(f"  Columns: {len(data_store.df.columns)}")
        col_preview = ', '.join(list(data_store.df.columns)[:5])
        if len(data_store.df.columns) > 5:
            col_preview += "..."
        print(f"  Columns: {col_preview}")
    else:
        print("âš ï¸�  No dataset loaded")
    
    # Memory status
    context = memory.get_context_summary()
    print(f"\nâœ“ Analysis History: {context['total_analyses']} queries")
    if context['recent_queries']:
        print("  Recent queries:")
        for q in context['recent_queries']:
            print(f"    - {q[:60]}...")  # Truncate long queries
    
    print("="*80)

print("âœ“ Helper functions defined (cleaned output)")



# Create sample dataset for demonstration
sample_data = pd.DataFrame({
    'customer_id': range(1, 101),
    'customer_name': [f'Customer_{i}' for i in range(1, 101)],
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'total_revenue': np.random.uniform(1000, 50000, 100).round(2),
    'num_orders': np.random.randint(1, 50, 100),
    'satisfaction_score': np.random.uniform(3.0, 5.0, 100).round(1),
    'customer_since': pd.date_range('2020-01-01', periods=100, freq='3D')
})

# Add some derived columns
sample_data['avg_order_value'] = (sample_data['total_revenue'] / sample_data['num_orders']).round(2)
sample_data['is_premium'] = sample_data['total_revenue'] > 30000

# Save to CSV
sample_file = '/tmp/sample_sales_data.csv'
sample_data.to_csv(sample_file, index=False)

print("âœ“ Sample dataset created")
print(f"  File: {sample_file}")
print(f"  Records: {len(sample_data)}")
print("\nğŸ“Š Sample data preview:")
print(sample_data.head(10))



# Load the sample data
result = load_csv_data(sample_file)
print("\nğŸ“� Data Loading Result:")
print(json.dumps(result, indent=2))

# Display system status
display_system_status()



# Run first analysis
await run_analysis(
    "Provide a comprehensive statistical analysis of this sales dataset. "
    "Focus on revenue patterns, customer behavior, and regional differences."
)



# Run second analysis
await run_analysis(
    "Who are the top 5 customers by revenue? "
    "What patterns do you notice about these high-value customers?"
)



# Run third analysis
await run_analysis(
    "Based on the data, what strategic recommendations would you make "
    "to improve revenue and customer satisfaction?"
)



# Show final system status with all analyses
display_system_status()

print("\nâœ¨ Demo completed successfully!")
print(f"Total analyses performed: {len(memory.get_recent_analyses())}")



# Uncomment and modify to use your own data:

# Step 1: Update with your file path
your_file_path = '/kaggle/input/sample-sales/sales.csv'

print(f"Loading file: {your_file_path}")


# Uncomment and modify to use your own data:

# Step 2: Load your data
result = load_csv_data(your_file_path)
print(json.dumps(result, indent=2))


# Uncomment and modify to use your own data:

# Step 3: Run your custom query
await run_analysis("How many distinct products are there?")




