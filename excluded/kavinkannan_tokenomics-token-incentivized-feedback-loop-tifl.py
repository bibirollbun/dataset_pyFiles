
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import random
from typing import Any, Dict
import logging
import sys
import traceback
import os
import json
import uuid
import re
import asyncio

import logging
import sys
import traceback

# --- Multi-Agent System Setup ---
import os
import json
import uuid
import re
import asyncio
from typing import List, Any, Dict, Optional

# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

try:
    import google.generativeai as genai
    model_client = Gemini(model='gemini-2.5-flash-lite')
    print("Using Real Gemini Client: gemini-2.5-flash-lite")
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    model_client = None

# Custom Tool
def librarian_tool(data: str):
    try:
        with open("agent_analysis_log.jsonl", "a") as f:
            f.write(str(data) + "\n")
        return "Data saved."
    except Exception as e:
        return f"Error saving data: {e}"

# Agents 
guardian_agent = LlmAgent(
    name="guardian",
    model=model_client,
    instruction="""
    You are a Content Safety Guardian.
    Analyze the input for hate speech, PII, or harassment.
    Output 'UNSAFE: [Reason]' if bad content is found.
    Output 'SAFE' if no issues are detected.
    Do not output anything else.
    """
)

router_agent = LlmAgent(
    name="router",
    model=model_client,
    instruction="""
    Classify user feedback into exactly one category:
    - 'bug report'
    - 'feature request'
    - 'unsafe content'
    - 'general comment'
    Output ONLY the category name.
    """
)

critic_agent = LlmAgent(
    name="critic",
    model=model_client,
    tools=[librarian_tool],
    instruction="""
    You are a Feedback Analyst.
    1. Analyze the feedback for 'sentiment', 'tone', and 'clarity'.
    2. Call the librarian_tool to save your analysis (as a JSON string or dict).
    """
)


quality_agent = LlmAgent(
    name="quality",
    model=model_client,
    instruction="""
    You are a Quality Control Agent.
    Analyze the input text. If it appears to be gibberish, random keys, or spam (e.g. "asdfasdf", "qweqwe"), output 'TRUE'.
    Otherwise, output 'FALSE'.
    """
)

# --- Logic & Data Setup ---

def initialize_session_state() -> Dict[str, Any]:
    initial_log = [
        {"Platform": "Google", "Type": "Doc Update", "Date": "2023-11-15", "Score": 20, "Status": "Pending"},
        {"Platform": "OpenAI", "Type": "Code Review", "Date": "2023-11-20", "Score": 50, "Status": "Confirmed"},
        {"Platform": "Anthropic", "Type": "Bug Report", "Date": "2023-11-18", "Score": 100, "Status": "Confirmed"},
        {"Platform": "Meta", "Type": "Safety Report", "Date": "2023-11-12", "Score": 75, "Status": "Confirmed"},
    ]
    return {
        "total_points": 2793,
        "trust_score": 850,
        "level": 6,
        "contributions_count": 320,
        "contribution_log": initial_log,
        "platform_counts": {"OpenAI": 45, "Anthropic": 30, "Google": 25}
    }

def get_df_from_state(state):
    return pd.DataFrame(state["contribution_log"])

# --- CHART: ---
def create_donut_chart(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ['#f36b1a', '#3b82f6', '#10b981', '#a855f7']

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.6,
        marker=dict(colors=colors), textinfo='none', hoverinfo='label+percent'
    )])
    fig.update_layout(
        showlegend=True,
        # Background changed to Light Grey to match general theme
        paper_bgcolor='#000000', 
        plot_bgcolor='#000000',
        # Font changed back to Dark Grey for contrast
        font=dict(family="Inter, sans-serif", color="#FFFFFF"), 
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
        margin=dict(t=0, b=0, l=0, r=0), height=200,
    )
    return fig

# --- HTML Generators ---

# Helper for the styled Titles
def section_title(text, icon=""):
    return f"""<div class="section-header">{icon} &nbsp; {text}</div>"""

def generate_header_html(points, level):
    range_span = MAX_LEVEL_POINTS - MIN_LEVEL_POINTS
    current_relative = max(0, points - MIN_LEVEL_POINTS)
    pct = min(100, max(5, (current_relative / range_span) * 100))

    return f"""
    <div style="margin-bottom: 20px;">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
            <div style="display:flex; align-items:center;">
                 <img src="https://ui-avatars.com/api/?name=John+Doe&background=random" style="width:48px; height:48px; border-radius:50%; margin-right: 15px;">
                 <div>
                    <div style="font-weight:700; font-size:20px;">John Doe</div>
                    <div style="color:#666; font-size:14px;">Contribution Level {level}</div>
                 </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:12px; color:#888; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Next Tier</div>
                <div style="font-weight:700; color:#f36b1a;">{MAX_LEVEL_POINTS - points} pts left</div>
            </div>
        </div>
        <div class="progress-bg">
            <div class="progress-bubble" style="left: {pct}%;">{points}</div>
            <div class="progress-fill" style="width: {pct}%;"></div>
        </div>
        <div class="level-text">
            <span>Tier {level} Start: {MIN_LEVEL_POINTS}</span>
            <span>Tier {level+1} Goal: {MAX_LEVEL_POINTS}</span>
        </div>
    </div>
    """

def generate_metric_html(label, value, is_score=False):
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'

def update_ui_from_state(state):
    header = generate_header_html(state["total_points"], state["level"])
    m_bal = generate_metric_html("Balance", f"{state['total_points']}")
    m_rep = generate_metric_html("Reputation", state["trust_score"])
    m_con = generate_metric_html("Contribs", state["contributions_count"])
    m_lvl = generate_metric_html("Level", state["level"])
    df = get_df_from_state(state)
    chart = create_donut_chart(state["platform_counts"])
    return header, m_bal, m_rep, m_con, m_lvl, df, chart

# --- Interaction Logic ---

def show_feedback_options():
    return gr.update(visible=True)

def show_other_input():
    return gr.update(visible=True)

# --- Custom Logging Plugin for Visibility ---
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

class CustomLoggingPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="custom_logging")
        self.logger = logging.getLogger("tokenomics.agent")
    
    async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> None:
        self.logger.info(f"ğŸš€ [AGENT START] {agent.name}")

    async def after_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> None:
        self.logger.info(f"ğŸ�� [AGENT END] {agent.name}")

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        self.logger.info(f"ğŸ§  [LLM REQUEST] {llm_request.system_instruction} | Input: {llm_request.input}")

    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: LlmResponse) -> None:
        self.logger.info(f"ğŸ’¡ [LLM RESPONSE] {llm_response.text}")

# Update process_bad_feedback to use logging and CustomLoggingPlugin
# --- Session & Memory Management ---
class SimpleSessionManager:
    def __init__(self, filepath="memory.json"):
        self.filepath = filepath
        self.sessions = {}
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.sessions = json.load(f)
                print(f"ğŸ’¾ [MEMORY] Loaded {len(self.sessions)} sessions from {self.filepath}")
            except Exception as e:
                print(f"âš ï¸� [MEMORY] Error loading memory: {e}")
                self.sessions = {}

    def save_memory(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.sessions, f, indent=2)
            # print(f"ğŸ’¾ [MEMORY] Saved to {self.filepath}") # Verbose
        except Exception as e:
            print(f"âš ï¸� [MEMORY] Error saving memory: {e}")

    def get_history(self, session_id):
        # Convert stored dict history to genai compatible list if needed
        # genai expects [{'role': 'user', 'parts': [...]}, ...]
        # We store simple [{'role': 'user', 'text': '...'}, ...]
        raw_hist = self.sessions.get(session_id, [])
        genai_hist = []
        for turn in raw_hist:
            genai_hist.append({"role": turn["role"], "parts": [turn["text"]]})
        return genai_hist

    def add_turn(self, session_id, user_text, model_text):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": "user", "text": user_text})
        self.sessions[session_id].append({"role": "model", "text": model_text})
        self.save_memory()

# Global Session Manager
session_manager = SimpleSessionManager()

async def process_bad_feedback(state, platform, reason_text):
    print(f"ğŸš€ [DEBUG] process_bad_feedback called for {platform}: {reason_text}")
    gr.Info(f"ğŸš€ Starting Analysis: {reason_text}")
    
    logger = logging.getLogger("tokenomics")
    logger.info(f"--- Processing Feedback: {platform} - {reason_text} ---")
    
    # --- API KEY CONFIGURATION  ---
    import os
    from kaggle_secrets import UserSecretsClient

    try:
        GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        genai.configure(api_key=GOOGLE_API_KEY) # Ensure genai is configured
        print("âœ… Setup and authentication complete.")
    except Exception as e:
        print(
            f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
        )
    
    async def run_agent_direct(agent, input_text, tools=None, session_id=None):
        msg = f"[{agent.name}] Starting run with input: {input_text}"
        print(f"ğŸ¤– [DEBUG] {msg}")
        logger.info(msg)
        
        candidate_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-1.5-flash"]
        last_exception = None
        
        system_instruction = agent.instruction
    
        for model_name in candidate_models:
            try:
                # Initialize Model
                if tools:
                    model = genai.GenerativeModel(model_name, tools=tools, system_instruction=system_instruction)
                else:
                    model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
                
                # Session / Chat History Management
                history = []
                if session_id:
                    history = session_manager.get_history(session_id)
                
                chat = model.start_chat(history=history)
                
                # Generate Response (Chat)
                response = chat.send_message(input_text)
                
                # Check for function call
                if tools and response.parts:
                    for part in response.parts:
                        if part.function_call:
                            fc = part.function_call
                            fn_name = fc.name
                            fn_args = dict(fc.args)
                            
                            print(f"ğŸ”§ [DEBUG] Tool Call Detected: {fn_name}({fn_args})")
                            logger.info(f"Tool Call: {fn_name}")
                            
                            if fn_name == "librarian_tool":
                                result = librarian_tool(**fn_args)
                                # In a real chat loop, we'd send this back. For now, return result.
                                # Update memory with the tool call interaction if desired, but simple text is safer for now.
                                return f"Tool Executed: {result}"
                            else:
                                return f"Unknown tool: {fn_name}"

                final_text = response.text
                
                # Update Memory
                if session_id:
                    session_manager.add_turn(session_id, input_text, final_text)
                
                res_msg = f"[{agent.name}] Finished using {model_name}. Response: {final_text}"
                print(f"âœ… [DEBUG] {res_msg}")
                logger.info(res_msg)
                
                return final_text

            except Exception as e:
                print(f"âš ï¸� [DEBUG] Model {model_name} failed: {e}. Trying next...")
                last_exception = e
                continue
        
        err_msg = f"[{agent.name}] ALL MODELS FAILED. Last error: {last_exception}"
        print(f"â�Œ [DEBUG] {err_msg}")
        logger.error(err_msg)
        logger.error(traceback.format_exc())
        return "AGENT_ERROR"
    try:
        # Use Platform as Session ID for context
        session_id = f"session_{platform}"
        
        # 1. Guardian Agent Check
        gr.Info("ğŸ›¡ï¸� Guardian Agent Checking...")
        # Guardian usually stateless, but we can pass session if we want. Let's keep it simple.
        safety_result = await run_agent_direct(guardian_agent, reason_text, session_id=session_id)
        if "UNSAFE" in safety_result:
            gr.Warning("âš ï¸� Guardian flagged as UNSAFE")
            logger.info("Guardian flagged as UNSAFE")
            new_entry = {
                "Platform": platform, 
                "Type": "Safety Violation", 
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Score": -50, 
                "Status": "Rejected"
            }
            state["contribution_log"].insert(0, new_entry)
            state["trust_score"] -= 10
            return gr.update(visible=False), gr.update(visible=False), state, *update_ui_from_state(state)

        # 2. Quality Agent Check
        gr.Info("âœ¨ Quality Agent Checking...")
        quality_result = await run_agent_direct(quality_agent, reason_text, session_id=session_id)
        if "TRUE" in quality_result:
            gr.Warning("ğŸ—‘ï¸� Quality agent flagged as Gibberish")
            logger.info("Quality agent flagged as Gibberish")
            new_entry = {
                "Platform": platform, 
                "Type": "Low Quality/Gibberish", 
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Score": 0, 
                "Status": "Rejected"
            }
            state["contribution_log"].insert(0, new_entry)
            return gr.update(visible=False), gr.update(visible=False), state, *update_ui_from_state(state)

        # 3. Router Agent Classification
        gr.Info("ğŸ”€ Router Agent Classifying...")
        category_response = await run_agent_direct(router_agent, reason_text, session_id=session_id)
        category = category_response.strip()
        logger.info(f"Router classified as: {category}")
        
        # 4. Critic Agent Analysis
        gr.Info("ğŸ§� Critic Agent Analyzing (Native Tool)...")
        # Pass tools explicitly to enable native function calling
        analysis_result = await run_agent_direct(critic_agent, reason_text, tools=[librarian_tool], session_id=session_id)
        
        if "Tool Executed" in analysis_result:
             gr.Info("ğŸ’¾ Analysis saved via Tool!")
        else:
             logger.warning(f"Critic did not use tool. Output: {analysis_result}")
             pass

        # Success
        gr.Info("âœ… Feedback Processed Successfully")
        new_entry = {
            "Platform": platform, 
            "Type": f"Flag: {reason_text} ({category})", 
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Score": 2, 
            "Status": "Under Review"
        }
        state["contribution_log"].insert(0, new_entry)
        state["total_points"] += 2
        state["contributions_count"] += 1
        
        return gr.update(visible=False), gr.update(visible=False), state, *update_ui_from_state(state)

    except Exception as e:
        gr.Error(f"â�Œ Error: {str(e)}")
        print(f"â�Œ [CRITICAL ERROR] {e}")
        logger.critical(f"CRITICAL ERROR in process_bad_feedback: {e}")
        logger.critical(traceback.format_exc())
        return gr.update(visible=True), gr.update(visible=True), state, *update_ui_from_state(state)


def generate_response(user_prompt, history, state, platform):
    ai_response = "Response generated. Dashboard updated."
    
    points_earned = 5
    new_entry = {
        "Platform": platform, "Type": "Agent Interaction", "Date": datetime.now().strftime("%Y-%m-%d"),
        "Score": points_earned, "Status": "Processed"
    }
    state["contribution_log"].insert(0, new_entry)
    state["total_points"] += points_earned
    state["contributions_count"] += 1
    
    if platform in state["platform_counts"]: state["platform_counts"][platform] += 1
    else: state["platform_counts"][platform] = 1

    history = history or []
    history.append((user_prompt, ai_response))
    
    return history, "", state, *update_ui_from_state(state)

def on_thumbs_up(state, platform):
    state["total_points"] += 10
    state["trust_score"] += 5
    return state, *update_ui_from_state(state)

def execute_trade(points_in, state):
    if points_in > state["total_points"]: return "Insufficient points.", state, *update_ui_from_state(state)
    state["total_points"] -= int(points_in)
    credits = points_in / 100
    return f"Traded {points_in} pts for ${credits:.2f}", state, *update_ui_from_state(state)

# --- Layout Construction ---




# --- Constants & Configuration ---
PRIMARY_COLOR = "#f36b1a"
MAX_LEVEL_POINTS = 5000
MIN_LEVEL_POINTS = 1500


# --- CSS Styling ---
css = """
body { font-family: 'Inter', sans-serif; background-color: #FAFAFA; }
.container { max-width: 1400px !important; margin: auto; }

/* The Main Single Container Card */
.custom-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 32px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* Metric Cards */
.metric-card { 
    text-align: left; 
    background-color: #f36b1a; 
    padding: 16px; 
    border-radius: 8px; 
}
.metric-label { 
    font-size: 11px; 
    color: #FFFFFF; 
    font-weight: 500; 
    text-transform: uppercase; 
    letter-spacing: 0.5px;
    opacity: 0.9;
}
.metric-value { 
    font-size: 28px; 
    font-weight: 700; 
    color: #FFFFFF; 
    margin-top: 2px; 
}

/* --- NEW: Styled Section Headers --- */
.section-header {
    background-color: #FFF5EB; /* Light Orange Background */
    color: #c2410c;            /* Dark Orange Text */
    padding: 12px 16px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 16px;
    border-left: 5px solid #f36b1a; /* Accent Border */
    margin-bottom: 20px;
    display: flex;
    align-items: center;
}

/* Profile Progress Bar */
.progress-bg { background-color: #E9E9E9; height: 12px; border-radius: 6px; width: 100%; position: relative; margin-top: 15px;}
.progress-fill { background-color: #f36b1a; height: 100%; border-radius: 6px; transition: width 0.5s ease-in-out; }
.progress-bubble {
    position: absolute; top: -32px; transform: translateX(-50%);
    background: #f36b1a; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;
    transition: left 0.5s ease-in-out;
}
.level-text { font-size: 12px; color: #999; display: flex; justify-content: space-between; width: 100%; margin-top: 8px;}

/* Table & Buttons */
.table-wrap { border-radius: 6px; overflow: hidden; border: 1px solid #EAEAEA; }
#trade-btn { background-color: #f36b1a !important; color: white !important; border: none !important; font-weight: 600; }
"""


# Async Wrappers for Gradio Buttons with Logging
async def process_app_problem(s, p):
    gr.Info(f"Processing: App Problem on {p}")
    msg = f"ğŸ”˜ Button Clicked: App Problem | Platform: {p}"
    logging.getLogger("tokenomics").info(msg)
    print(msg, file=sys.stderr)
    return await process_bad_feedback(s, p, "App Problem")

async def process_fact_error(s, p):
    gr.Info(f"Processing: Factual Error on {p}")
    msg = f"ğŸ”˜ Button Clicked: Factual Error | Platform: {p}"
    logging.getLogger("tokenomics").info(msg)
    print(msg, file=sys.stderr)
    return await process_bad_feedback(s, p, "Factual Error")

async def process_instr_fail(s, p):
    gr.Info(f"Processing: Instruction Fail on {p}")
    msg = f"ğŸ”˜ Button Clicked: Instruction Fail | Platform: {p}"
    logging.getLogger("tokenomics").info(msg)
    print(msg, file=sys.stderr)
    return await process_bad_feedback(s, p, "Instruction Fail")

async def process_unsafe(s, p):
    gr.Info(f"Processing: Unsafe Content on {p}")
    msg = f"ğŸ”˜ Button Clicked: Unsafe Content | Platform: {p}"
    logging.getLogger("tokenomics").info(msg)
    print(msg, file=sys.stderr)
    return await process_bad_feedback(s, p, "Unsafe Content")

async def process_other_submit(s, p, t):
    gr.Info(f"Processing: Other Feedback on {p}")
    msg = f"ğŸ”˜ Button Clicked: Other Submit | Platform: {p} | Text: {t}"
    logging.getLogger("tokenomics").info(msg)
    print(msg, file=sys.stderr)
    return await process_bad_feedback(s, p, t)

with gr.Blocks(css=css, title="Tokenomics Dashboard", theme=gr.themes.Default(primary_hue="orange")) as demo:

    session_state = gr.State(initialize_session_state())

    with gr.Row():

        # === LEFT COLUMN: Chat ===
        with gr.Column(scale=4):
            with gr.Group(elem_classes="custom-card"):
                gr.HTML(section_title("Agent Swarm", "ğŸ§ "))
                platform_selector = gr.Dropdown(["OpenAI", "Anthropic", "Google"], value="OpenAI", label="Target Platform")
                chatbot = gr.Chatbot(value=[[None, "Ready."]], height=300, type="tuples")
                msg = gr.Textbox(show_label=False, placeholder="Type message...")
                
                # Primary Buttons
                with gr.Row():
                    btn_up = gr.Button("ğŸ‘� Good", size="sm")
                    btn_down = gr.Button("ğŸ‘� Bad", size="sm")

                # NEW: Hidden Reason Panel (Initially visible=False)
                with gr.Column(visible=False) as reason_panel:
                    gr.Markdown("#### What was the issue?")
                    # Stacking buttons vertically to match your HTML reference
                    btn_app_prob = gr.Button("Problem with an app", size="sm")
                    btn_fact = gr.Button("Not factually correct", size="sm")
                    btn_instr = gr.Button("Didn't follow instructions", size="sm")
                    btn_unsafe = gr.Button("Offensive / Unsafe", size="sm")
                    btn_other = gr.Button("Other...", size="sm")

                # NEW: Hidden Other Input Panel (Initially visible=False)
                with gr.Column(visible=False) as other_panel:
                    other_text = gr.Textbox(label="Provide additional details", lines=2)
                    btn_submit_other = gr.Button("Submit Feedback", variant="primary")
                    
        # === RIGHT COLUMN: Unified Dashboard Container ===
        with gr.Column(scale=8):
            
            with gr.Group(elem_classes="custom-card"):
                
                # 1. Profile Section
                gr.HTML(section_title("Tokenomics Profile", "ğŸ‘¤"))
                header_html = gr.HTML(generate_header_html(2793, 6))
                
                gr.HTML("<hr style='margin: 20px 0; border-top: 1px solid #eee;'>")

                # 2. Metrics Row
                with gr.Row():
                    metric_bal = gr.HTML(generate_metric_html("Balance", "2793"))
                    metric_rep = gr.HTML(generate_metric_html("Reputation", "850"))
                    metric_cont = gr.HTML(generate_metric_html("Contribs", "320"))
                    metric_lvl = gr.HTML(generate_metric_html("Level", "6"))
                
                gr.HTML("<hr style='margin: 20px 0; border-top: 1px solid #eee;'>")

                # 3. Middle Section: Charts & Trade
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(section_title("Activity", "ğŸ“Š"))
                        dist_chart = gr.Plot(value=create_donut_chart({"OpenAI":45, "Anthropic":30, "Google":25}))
                    
                    with gr.Column(scale=1):
                        gr.HTML(section_title("Trade", "ğŸ’±"))
                        trade_input = gr.Number(label="Points", value=500, step=100)
                        trade_output = gr.Number(label="Credits ($)", value=5.00)
                        btn_trade = gr.Button("Exchange Tokens", elem_id="trade-btn")
                        trade_msg = gr.Markdown("")

                gr.HTML("<hr style='margin: 20px 0; border-top: 1px solid #eee;'>")

                # 4. History Section
                gr.HTML(section_title("Recent History", "ğŸ“œ"))
                ledger_table = gr.Dataframe(
                    value=get_df_from_state(initialize_session_state()),
                    headers=["Platform", "Type", "Date", "Score", "Status"],
                    interactive=False,
                    elem_classes="table-wrap"
                )

    # --- Wiring ---
    # 1. Dashboard outputs list
    dashboard_outputs = [header_html, metric_bal, metric_rep, metric_cont, metric_lvl, ledger_table, dist_chart]
    
    # 2. Consolidated outputs for feedback actions (Hides panels + Updates Dashboard)
    feedback_outputs = [reason_panel, other_panel, session_state, *dashboard_outputs]

    # 3. Existing Chat & Thumbs Up
    msg.submit(generate_response, [msg, chatbot, session_state, platform_selector], [chatbot, msg, session_state, *dashboard_outputs])
    btn_up.click(on_thumbs_up, [session_state, platform_selector], [session_state, *dashboard_outputs])
    
    # 4. NEW: Thumbs Down Wiring
    # Clicking "Bad" reveals the reason panel
    btn_down.click(show_feedback_options, None, reason_panel)

    # Clicking "Other..." reveals the text input panel
    btn_other.click(show_other_input, None, other_panel)

    # Clicking specific reasons submits immediately
    # We use partial functions (lambda) to pass the specific string
    btn_app_prob.click(process_app_problem, [session_state, platform_selector], feedback_outputs)
    btn_fact.click(process_fact_error, [session_state, platform_selector], feedback_outputs)
    btn_instr.click(process_instr_fail, [session_state, platform_selector], feedback_outputs)
    btn_unsafe.click(process_unsafe, [session_state, platform_selector], feedback_outputs)
    
    # Clicking Submit on the "Other" text box
    btn_submit_other.click(process_other_submit, [session_state, platform_selector, other_text], feedback_outputs)
    
    outputs = [session_state, header_html, metric_bal, metric_rep, metric_cont, metric_lvl, ledger_table, dist_chart]
    
    msg.submit(generate_response, [msg, chatbot, session_state, platform_selector], [chatbot, msg, *outputs])
    btn_up.click(on_thumbs_up, [session_state, platform_selector], outputs)
    
    def calc_credits(p): return p/100 if p else 0
    trade_input.change(calc_credits, trade_input, trade_output)
    btn_trade.click(execute_trade, [trade_input, session_state], [trade_msg, *outputs])

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)




