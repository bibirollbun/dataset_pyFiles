# =====================================================================
# GenBI: AI Business Intelligence Assistant (Kaggle Master Edition)
# Stack: Google Gemini SDK, Plotly, Statsmodels, LangChain
# Model: Gemini 2.5 Flash (Updated for Nov 2025 Stability)
# =====================================================================

# 1. INSTALLATION
# (Uncomment the line below if running in a new session)
# !pip install -q -U google-generativeai langchain langchain-community plotly statsmodels pydantic

import os
import pandas as pd
import numpy as np
import json
import re
import warnings
from typing import List, Optional

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display, Markdown

# AI & LangChain
import google.generativeai as genai
from langchain.schema import Generation, LLMResult, SystemMessage, HumanMessage
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import Tool
from pydantic import Field

# Modeling
from statsmodels.tsa.arima.model import ARIMA

# Configuration
warnings.filterwarnings('ignore')
pio.renderers.default = 'iframe' # Essential for Kaggle display

# =====================================================================
# 2. ROBUST GEMINI PROVIDER (Gemini 2.5)
# =====================================================================

class GeminiProvider(BaseLLM):
    """
    Custom LangChain wrapper for Google Gemini API via SDK.
    Updated to use 'gemini-2.5-flash' to avoid 404 deprecation errors.
    """
    model_name: str = Field(default="gemini-2.5-flash", description="Gemini model")
    temperature: float = Field(default=0.7, description="Temperature")
    api_key: str = Field(..., description="API Key")
    
    class Config: arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    @property
    def _llm_type(self) -> str: return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Gemini 2.5 supports higher output tokens (8192)
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=8192
                )
            )
            return response.text
        except Exception as e:
            return f"API Error: {str(e)}"

    def _generate(self, prompts: List[str], stop=None, **kwargs) -> LLMResult:
        return LLMResult(generations=[[Generation(text=self._call(p))] for p in prompts])

# =====================================================================
# 3. DATA UTILITIES (Load & Demo)
# =====================================================================

def create_demo_data() -> pd.DataFrame:
    """Generates synthetic e-commerce data for testing/demo."""
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    categories = np.random.choice(['Electronics', 'Fashion', 'Home', 'Services'], 100)
    sales = np.random.randint(100, 5000, 100) + (np.arange(100) * 20) # Upward trend
    profit = sales * np.random.uniform(0.1, 0.5, 100)
    
    return pd.DataFrame({
        'Date': dates,
        'Category': categories,
        'Sales': sales,
        'Profit': profit,
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })

def load_data(file_path: str) -> pd.DataFrame:
    """Loads file and auto-parses dates."""
    if file_path.endswith('.csv'): df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')): df = pd.read_excel(file_path)
    else: raise ValueError("Unsupported format")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try: df[col] = pd.to_datetime(df[col])
            except: pass
    return df.fillna(0)

def get_data_schema(df: pd.DataFrame) -> str:
    schema = [f"Rows: {df.shape[0]}, Cols: {df.shape[1]}"]
    for col in df.columns:
        schema.append(f"â€¢ {col} ({df[col].dtype})")
    return "\n".join(schema)

# =====================================================================
# 4. FUNCTIONAL TOOLS (The "Brain" Logic)
# =====================================================================

class AnalysisToolkit:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_stats(self, query: str) -> str:
        try:
            if query == 'overview': return self.data.describe(include='all').to_markdown()
            if query == 'correlation': return self.data.select_dtypes('number').corr().round(2).to_markdown()
            if query == 'missing': return self.data.isna().sum()[self.data.isna().sum()>0].to_markdown()
            if query in self.data.columns: return self.data[query].describe().to_markdown()
            return "Invalid stats request."
        except Exception as e: return f"Error: {e}"

    def filter_data(self, query: str) -> str:
        try:
            p = json.loads(query)
            mask = pd.Series([True] * len(self.data), index=self.data.index)
            for cond in p.get("conditions", []):
                col, op, val = cond['column'], cond['operator'], cond['value']
                if op == "==": mask &= self.data[col] == val
                elif op == ">": mask &= self.data[col] > val
                elif op == "<": mask &= self.data[col] < val
            
            res = self.data[mask]
            return f"Filtered: {len(res)} rows.\n{res.head().to_markdown()}"
        except Exception as e: return f"Filter Error: {e}"

    def plot_data(self, query: str) -> str:
        try:
            p = json.loads(query)
            df = self.data.copy()
            
            # Aggregation logic
            if p.get('agg') == 'sum': df = df.groupby(p['x'], as_index=False)[p['y']].sum()
            elif p.get('agg') == 'mean': df = df.groupby(p['x'], as_index=False)[p['y']].mean()

            # Plotting
            type = p.get('chart_type', 'bar')
            if type == 'bar': fig = px.bar(df, x=p['x'], y=p['y'], color=p.get('color'), title=p.get('title'))
            elif type == 'line': fig = px.line(df, x=p['x'], y=p['y'], color=p.get('color'), title=p.get('title'))
            elif type == 'scatter': fig = px.scatter(df, x=p['x'], y=p['y'], color=p.get('color'), title=p.get('title'))
            elif type == 'pie': fig = px.pie(df, names=p['x'], values=p['y'], title=p.get('title'))
            elif type == 'histogram': fig = px.histogram(df, x=p['x'], color=p.get('color'), title=p.get('title'))
            
            fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            fig.show()
            return "Chart displayed successfully."
        except Exception as e: return f"Plot Error: {e}"

    def forecast(self, query: str) -> str:
        try:
            p = json.loads(query)
            df = self.data.copy()
            df[p['time_col']] = pd.to_datetime(df[p['time_col']])
            ts = df.sort_values(p['time_col']).set_index(p['time_col'])[p['target_col']]
            
            # Resample daily
            ts = ts.resample('D').sum().fillna(0)
            
            model = ARIMA(ts, order=(1,1,1))
            fit = model.fit()
            forecast = fit.forecast(steps=int(p['periods']))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Historical'))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast', line=dict(color='red')))
            fig.update_layout(title=f"ARIMA Forecast for {p['target_col']}", template="plotly_white")
            fig.show()
            return f"Forecast generated for {p['periods']} periods."
        except Exception as e: return f"Forecast Error: {e}"

# =====================================================================
# 5. AGENT CONFIGURATION
# =====================================================================

from langchain.prompts import BaseChatPromptTemplate

SYSTEM_PROMPT = """You are GenBI, an expert Data Analyst AI running on Gemini 2.5.
SCHEMA:
{schema}

INSTRUCTIONS:
1. Always use tools.
2. For "show", "plot" -> use `plot_data`. Return JSON.
3. For "predict" -> use `forecast`.
4. Output Markdown.

FORMAT:
Thought: [reasoning]
Action: [tool_name]
Action Input: [json_string]
"""

class AgentPrompt(BaseChatPromptTemplate):
    tools: List[Tool]
    schema: str
    input_variables: List[str] = ["query", "chat_history"]

    def format_messages(self, **kwargs) -> List[HumanMessage]:
        tools_str = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        sys = SYSTEM_PROMPT.format(schema=self.schema) + "\nTOOLS:\n" + tools_str
        msgs = [SystemMessage(content=sys)]
        if kwargs.get("chat_history"): msgs.extend(kwargs["chat_history"])
        msgs.append(HumanMessage(content=f"Query: {kwargs['query']}"))
        return msgs

class DataAgent:
    def __init__(self, llm, tools, prompt, memory):
        self.chain = LLMChain(llm=llm, prompt=prompt)
        self.tools = {t.name: t for t in tools}
        self.memory = memory

    def run(self, query):
        hist = self.memory.load_memory_variables({})["chat_history"]
        res = self.chain.run(query=query, chat_history=hist)
        
        # Tool Execution Loop
        if "Action:" in res:
            try:
                action = re.search(r"Action: (\w+)", res)
                inp = re.search(r"Action Input: (\{.*\}|[\w\s_]+)", res, re.DOTALL)
                
                if action and inp:
                    tool_name = action.group(1).strip()
                    tool_inp = inp.group(1).strip()
                    
                    if tool_name in self.tools:
                        display(Markdown(f"*âš™ï¸� Running {tool_name}...*"))
                        output = self.tools[tool_name].run(tool_inp)
                        
                        # Summary pass
                        final = self.chain.run(query=f"Tool Output: {output}\nSummarize:", chat_history=hist)
                        self.memory.save_context({"input": query}, {"output": final})
                        return final
            except Exception as e: return f"Error: {e}"
        
        self.memory.save_context({"input": query}, {"output": res})
        return res

def create_agent(llm, data):
    tk = AnalysisToolkit(data)
    tools = [
        Tool(name="get_stats", func=tk.get_stats, description="Get stats. Input: 'overview' or col name."),
        Tool(name="filter_data", func=tk.filter_data, description="Filter data. JSON conditions."),
        Tool(name="plot_data", func=tk.plot_data, description="Plot. JSON: {chart_type, x, y, agg, color}."),
        Tool(name="forecast", func=tk.forecast, description="Forecast. JSON: {time_col, target_col, periods}.")
    ]
    prompt = AgentPrompt(tools=tools, schema=get_data_schema(data))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return DataAgent(llm, tools, prompt, memory)

# =====================================================================
# 6. MAIN EXECUTION
# =====================================================================

def main():
    display(Markdown("# ğŸš€ GenBI: Kaggle Master Edition (Gemini 2.5)"))
    display(Markdown("---"))
    
    # 1. API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: 
        print("Note: In Kaggle, use 'Add-ons > Secrets' or paste below.")
        api_key = input("Enter Gemini API Key: ")
    
    try:
        # UPDATED: Using Gemini 2.5 Flash as the standard for late 2025
        llm = GeminiProvider(api_key=api_key, model_name="gemini-2.5-flash")
        display(Markdown("**âœ… Gemini 2.5 AI Connected**"))
    except Exception as e:
        display(Markdown(f"**â�Œ Connection Failed:** {e}"))
        return

    # 2. Data
    file_path = input("Enter dataset path (or PRESS ENTER for Demo Data): ").strip()
    
    try:
        if not file_path:
            display(Markdown("**âš ï¸� No file provided. Generating SYNTHETIC DEMO DATA...**"))
            data = create_demo_data()
        else:
            data = load_data(file_path)
        display(Markdown(f"**âœ… Data Loaded:** {data.shape[0]} rows"))
        display(data.head())
    except Exception as e:
        display(Markdown(f"**â�Œ Error:** {e}"))
        return

    # 3. Start
    agent = create_agent(llm, data)
    display(Markdown("---"))
    display(Markdown("### ğŸ¤– Agent Ready!"))

    while True:
        q = input("Query (or 'exit'): ")
        if q.lower() in ['exit', 'quit']: break
        
        display(Markdown(f"**You:** {q}"))
        display(Markdown("**Thinking...**"))
        res = agent.run(q)
        display(Markdown(f"**GenBI:** {res}"))
        display(Markdown("---"))

if __name__ == "__main__":
    main()

