import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# --- CONFIGURATION ---
# ⚠️ REPLACE WITH YOUR ACTUAL API KEY
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY" 

genai.configure(api_key=GOOGLE_API_KEY)

CONFIG = {
    "agent_name": "EarthGuardian",
    "model": "models/gemini-2.0-flash", # You can also use 'gemini-1.5-flash'
    "temperature": 0.3
}


# --- DOMAIN-SPECIFIC TOOLS ---

def propose_sustainability_solutions(issue_type: str, location: str, budget_level: str) -> str:
    """Generates specific eco-friendly solutions for a given problem and location."""
    prompt = (
        f"Act as an Environmental Engineer. Propose 5 concrete solutions for:\n"
        f"Issue: {issue_type}\n"
        f"Location: {location}\n"
        f"Budget Constraint: {budget_level}\n\n"
        f"Include: Implementation steps, expected impact, and necessary technology."
    )
    # We use a separate model call here to generate the specific content for the tool
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text

def draft_action_plan(goal: str, duration_months: int, resources: str) -> str:
    """Creates a timeline for environmental projects."""
    prompt = (
        f"Create a phased environmental action plan:\n"
        f"Goal: {goal}\n"
        f"Timeline: {duration_months} months\n"
        f"Available Resources: {resources}\n\n"
        f"Provide: Monthly milestones, key performance indicators (KPIs), and risk assessments."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text

def analyze_pollution_data(pollutant_type: str, reading_value: float, standard_limit: float) -> str:
    """Analyzes pollution levels and suggests immediate mitigation."""
    prompt = (
        f"Analyze this pollution data:\n"
        f"Pollutant: {pollutant_type}\n"
        f"Current Reading: {reading_value}\n"
        f"Safety Limit: {standard_limit}\n\n"
        f"Provide: Severity assessment, health risks, and immediate emergency measures."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text

def calculate_carbon_impact(activity: str, scale: str) -> str:
    """Estimates carbon footprint or savings."""
    prompt = (
        f"Estimate the carbon impact (positive or negative) for:\n"
        f"Activity: {activity}\n"
        f"Scale: {scale}\n\n"
        f"Provide: Estimated CO2e tons/year, comparison to average benchmarks, and reduction tips."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text


# --- FUNCTION DECLARATIONS (API SCHEMA) ---

function_declarations = [
    FunctionDeclaration(
        name="propose_sustainability_solutions",
        description="Suggests solutions for environmental issues like waste, water, or energy.",
        parameters={
            "type": "object",
            "properties": {
                "issue_type": {"type": "string", "description": "The environmental problem (e.g., 'plastic waste', 'drought')"},
                "location": {"type": "string", "description": "Target city or region"},
                "budget_level": {"type": "string", "description": "Low, Medium, or High"}
            },
            "required": ["issue_type", "location", "budget_level"]
        }
    ),
    FunctionDeclaration(
        name="draft_action_plan",
        description="Creates a timeline for executing environmental projects.",
        parameters={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The main objective (e.g., 'Plant 1000 trees')"},
                "duration_months": {"type": "integer", "description": "Time available in months"},
                "resources": {"type": "string", "description": "Manpower or funding available"}
            },
            "required": ["goal", "duration_months", "resources"]
        }
    ),
    FunctionDeclaration(
        name="analyze_pollution_data",
        description="Analyzes pollution metrics against safety standards.",
        parameters={
            "type": "object",
            "properties": {
                "pollutant_type": {"type": "string", "description": "e.g., PM2.5, Lead, CO2"},
                "reading_value": {"type": "number", "description": "Measured value"},
                "standard_limit": {"type": "number", "description": "Regulatory limit"}
            },
            "required": ["pollutant_type", "reading_value", "standard_limit"]
        }
    ),
    FunctionDeclaration(
        name="calculate_carbon_impact",
        description="Calculates carbon footprint of activities.",
        parameters={
            "type": "object",
            "properties": {
                "activity": {"type": "string", "description": "e.g., 'Switching to solar', 'Driving diesel truck'"},
                "scale": {"type": "string", "description": "e.g., '10 households', '500 miles'"}
            },
            "required": ["activity", "scale"]
        }
    )
]

# Wrap them in a Tool object
tools = Tool(function_declarations=function_declarations)


# --- UTILITY CLASSES ---

@dataclass
class ConversationMemory:
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    
    def get_context(self) -> str:
        # Returns the last 5 messages to keep context fresh without overloading the token limit
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.messages[-5:]])

@dataclass
class AgentLogger:
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def info(self, event: str, **kwargs):
        print(f"[INFO] {event} | {kwargs}")
        self.logs.append({"level": "INFO", "event": event, "details": kwargs})
        
    def error(self, event: str, **kwargs):
        print(f"[ERROR] {event} | {kwargs}")
        self.logs.append({"level": "ERROR", "event": event, "details": kwargs})


# --- MAIN AGENT CLASS (FIXED) ---

class EcoAgent:
    def __init__(self, config: Dict, tools: Tool):
        self.config = config
        self.memory = ConversationMemory()
        self.logger = AgentLogger()
        self.model = genai.GenerativeModel(model_name=config['model'], tools=[tools])
        
        self.function_map = {
            "propose_sustainability_solutions": propose_sustainability_solutions,
            "draft_action_plan": draft_action_plan,
            "analyze_pollution_data": analyze_pollution_data,
            "calculate_carbon_impact": calculate_carbon_impact
        }

    def run(self, user_query: str) -> str:
        self.logger.info("Query Received", query=user_query)
        self.memory.add_message("user", user_query)
        
        system_prompt = (
            f"You are {self.config['agent_name']}, an expert AI focused on solving Earth's environmental issues. "
            f"Use your tools to analyze data, propose solutions, and calculate impacts. "
            f"Context: {self.memory.get_context()}"
        )

        chat = self.model.start_chat()
        response = chat.send_message(f"{system_prompt}\n\nUser Query: {user_query}")

        final_response = "Processing..." # Default state

        try:
            # Check the first part of the response to see if it's a function call
            # We do NOT access response.text yet
            part = response.candidates[0].content.parts[0]
            
            # CASE 1: The model wants to call a function
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                func_name = fc.name
                func_args = dict(fc.args)
                
                self.logger.info("Tool Triggered", tool=func_name, args=func_args)
                
                if func_name in self.function_map:
                    # 1. Execute the tool
                    tool_result = self.function_map[func_name](**func_args)
                    
                    # 2. Send result back to Gemini to get the final text answer
                    final_response_obj = chat.send_message(
                        f"Tool Output for {func_name}: {tool_result}\n"
                        "Synthesize this into a helpful response for the user."
                    )
                    final_response = final_response_obj.text
            
            # CASE 2: The model just replied with text (no tool needed)
            else:
                final_response = response.text

        except Exception as e:
            self.logger.error("Execution Failure", error=str(e))
            final_response = "I encountered an error while processing the tool output."

        self.memory.add_message("agent", final_response)
        return final_response


# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    # Initialize the agent
    agent = EcoAgent(CONFIG, tools)
    
    # Test 1: Suggesting a solution
    print("\n--- Query 1: Waste Management ---")
    response = agent.run("How can we reduce plastic waste in Mumbai on a low budget?")
    print(f"Agent: {response}\n")
    
    # Test 2: Calculating Impact
    print("\n--- Query 2: Carbon Footprint ---")
    response = agent.run("What is the carbon impact of switching a small village of 50 homes to solar power?")
    print(f"Agent: {response}\n")

