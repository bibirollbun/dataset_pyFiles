











# -*- coding: utf-8 -*-
"""
Kaggle Agents Intensive - Capstone Project: The Ultimate Personal Assistant v2

рдпрд╣ рд╕рдВрд╕реНрдХрд░рдг рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ (Tool Use) рд▓реЙрдЬрд┐рдХ рдХреЛ рд╢рд╛рдорд┐рд▓ рдХрд░рддрд╛ рд╣реИред
рдпрд╣ рджрд┐рдЦрд╛рддрд╛ рд╣реИ рдХрд┐ рдПрдЬреЗрдВрдЯ (LLM) рдХрдм рдФрд░ рдХреИрд╕реЗ рдмрд╛рд╣рд░реА рдЯреВрд▓реНрд╕ рдХреЛ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдирд┐рд░реНрдгрдп рд▓реЗрддрд╛ рд╣реИред
"""

import time
import json
import random

# ==============================================================================
# 1. рдЯреВрд▓ рдбреЗрдлрд┐рдирд┐рд╢рди (Tool Definitions) - рдмрд╛рд╣рд░реА рджреБрдирд┐рдпрд╛ рд╕реЗ рдЬреБрдбрд╝рдиреЗ рдХреЗ рд▓рд┐рдП
#    рдпреЗ рдлрд╝рдВрдХреНрд╢рди рд╡рд╛рд╕реНрддрд╡рд┐рдХ API рдХреЛ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдЕрдиреБрдХрд░рдг (simulate) рдХрд░рддреЗ рд╣реИрдВред
# ==============================================================================

# ** рдорд╣рддреНрд╡рдкреВрд░реНрдг: рдЗрд╕ рдПрдЬреЗрдВрдЯ рд╕рд┐рд╕реНрдЯрдо рдореЗрдВ рдЙрдкрдпреЛрдЧ рдХрд┐рдП рдЬрд╛рдиреЗ рд╡рд╛рд▓реЗ рд╕рднреА рдЯреВрд▓реНрд╕ рдХреА рдореИрдкрд┐рдВрдЧ **
TOOL_MAP = {}

def google_search_tool(query: str) -> str:
    """рд╡реЗрдм рдкрд░ рдЬрд╛рдирдХрд╛рд░реА рдЦреЛрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред (Google Search API рдХрд╛ рдЕрдиреБрдХрд░рдг)"""
    print(f"\n[TOOL CALLED: GoogleSearchTool] Searching for: '{query}'...")
    time.sleep(1) # рд╕рд┐рдореБрд▓реЗрд╢рди рдХреЗ рд▓рд┐рдП рд╡рд┐рд▓рдВрдм
    
    if "latest AI trends" in query.lower():
        return "Search Results: AI trends show massive growth in Generative Agents (Multi-Agent Systems) and RAG applications in 2025. This requires deep understanding of model orchestration."
    elif "event planning tips" in query.lower():
        return "Search Results: Top event planning tips include setting a clear goal and defining the budget."
    else:
        return f"Search Results: Found 3 relevant articles about '{query}'. Key points: Promising developments in machine learning."

TOOL_MAP['google_search_tool'] = google_search_tool


def calendar_tool(title: str, time: str, attendees: list) -> str:
    """рдХреИрд▓реЗрдВрдбрд░ рдореЗрдВ рдЗрд╡реЗрдВрдЯ рдмрдирд╛рдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: CalendarTool] Scheduling: {title} on {time} with {len(attendees)} people.")
    time.sleep(0.5)
    return f"Calendar: Event '{title}' successfully scheduled for {time}."

TOOL_MAP['calendar_tool'] = calendar_tool


def messaging_tool(recipient: str, subject: str, body: str) -> str:
    """рдХрд┐рд╕реА рдХреЛ рдИрдореЗрд▓ рдпрд╛ рдореИрд╕реЗрдЬ рднреЗрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: MessagingTool] Sending message to {recipient}. Subject: {subject}")
    time.sleep(0.5)
    return f"Messaging: Email to {recipient} with subject '{subject}' sent successfully."

TOOL_MAP['messaging_tool'] = messaging_tool

# ==============================================================================
# 2. рдЬреЗрдорд┐рдиреА рдПрдЬреЗрдВрдЯ рдХреЙрд▓ (Gemini Agent Call) - рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ
# ==============================================================================

def gemini_agent_call(prompt: str, available_tools: list) -> dict:
    """
    рдЬреЗрдорд┐рдиреА рдореЙрдбрд▓ рдХреЛ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдЕрдиреБрдХрд░рдгред
    рдпрд╣ рдлрд╝рдВрдХреНрд╢рди рдпрд╛ рддреЛ рдЕрдВрддрд┐рдо рдЯреЗрдХреНрд╕реНрдЯ рдЖрдЙрдЯрдкреБрдЯ рджреЗрддрд╛ рд╣реИ рдпрд╛ рдЯреВрд▓ рдХреЛ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдирд┐рд░реНрджреЗрд╢ред
    
    рд╡рд╛рд╕реНрддрд╡рд┐рдХ Kaggle Notebook рдореЗрдВ, рдЖрдк рдпрд╣рд╛рдБ 'google-genai' SDK рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВрдЧреЗред
    """
    print(f"   [LLM: {len(available_tools)} tools offered] Thinking about prompt...")
    time.sleep(0.5)

    # **рд╕рд┐рдореБрд▓реЗрд╢рди рд▓реЙрдЬрд┐рдХ (LLM's Decision)**
    
    # A. рд░рд┐рд╕рд░реНрдЪ рдПрдЬреЗрдВрдЯ рдХрд╛ рдЯреВрд▓ рдХреЙрд▓ рдирд┐рд░реНрдгрдп (Researcher Agent's Tool Call Decision)
    if 'google_search_tool' in available_tools:
        if "trends" in prompt.lower() or "research" in prompt.lower():
            # LLM рдиреЗ рдЯреВрд▓ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдлреИрд╕рд▓рд╛ рдХрд┐рдпрд╛
            query = prompt.split("for")[-1].strip() if "for" in prompt.lower() else "General market trends"
            return {
                "type": "function_call",
                "name": "google_search_tool",
                "args": {"query": query}
            }
        
    # B. рд╢реЗрдбреНрдпреВрд▓рд░ рдПрдЬреЗрдВрдЯ рдХрд╛ рдЯреВрд▓ рдХреЙрд▓ рдирд┐рд░реНрдгрдп (Scheduler Agent's Tool Call Decision)
    if 'calendar_tool' in available_tools:
        if "schedule a meeting" in prompt.lower():
            # LLM рдиреЗ рдЯреВрд▓ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдлреИрд╕рд▓рд╛ рдХрд┐рдпрд╛
            return {
                "type": "function_call",
                "name": "calendar_tool",
                "args": {
                    "title": "Agent Sync-up", 
                    "time": "2025-11-19 14:00", 
                    "attendees": ["User", "Colleague"]
                }
            }

    # C. рдХрдореНрдпреБрдирд┐рдХреЗрдЯрд░ рдПрдЬреЗрдВрдЯ рдХрд╛ рдЯреВрд▓ рдХреЙрд▓ рдирд┐рд░реНрдгрдп (Communicator Agent's Tool Call Decision)
    if 'messaging_tool' in available_tools:
        if "send a message" in prompt.lower():
            # LLM рдиреЗ рдЯреВрд▓ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдлреИрд╕рд▓рд╛ рдХрд┐рдпрд╛
            return {
                "type": "function_call",
                "name": "messaging_tool",
                "args": {
                    "recipient": "client@email.com", 
                    "subject": "Project Update", 
                    "body": "The report is compiled and ready."
                }
            }
            
    # D. рдЯреЗрдХреНрд╕реНрдЯ рдЖрдЙрдЯрдкреБрдЯ (рдЕрдЧрд░ рдЯреВрд▓ рдХреА рдЬрд░реВрд░рдд рдирд╣реАрдВ рд╣реИ)
    return {
        "type": "text",
        "content": f"Acknowledged: Finished processing the thought process for: '{prompt[:50]}...'"
    }

# ==============================================================================
# 3. рд╡рд┐рд╢реЗрд╖рдЬреНрдЮ рдПрдЬреЗрдВрдЯреЛрдВ рдХреА рдкрд░рд┐рднрд╛рд╖рд╛ (Specialist Agents Definition)
# ==============================================================================

class BaseAgent:
    """рд╕рднреА рдПрдЬреЗрдВрдЯреЛрдВ рдХреЗ рд▓рд┐рдП рдЖрдзрд╛рд░ рдХреНрд▓рд╛рд╕ред"""
    def __init__(self, name: str):
        self.name = name

    def execute(self, task: str, data: any = None) -> (str, any):
        """рдХрд╛рд░реНрдп рдХреЛ рдирд┐рд╖реНрдкрд╛рджрд┐рдд (execute) рдХрд░рддрд╛ рд╣реИред"""
        print(f"[{self.name}] Executing task: {task[:50]}...")
        # рдпрд╣рд╛рдБ рд╡рд╛рд╕реНрддрд╡рд┐рдХ LLM рдХреЙрд▓ рд╣реЛрдЧреА рдЬреЛ рд▓реЙрдЬрд┐рдХ рдХреЛ рдЪрд▓рд╛рдПрдЧреА
        pass

class PlannerAgent(BaseAgent):
    """рдЬрдЯрд┐рд▓ рдХрд╛рд░реНрдпреЛрдВ рдХреЛ рдЫреЛрдЯреЗ, рддрд╛рд░реНрдХрд┐рдХ рдЪрд░рдгреЛрдВ рдореЗрдВ рддреЛрдбрд╝рддрд╛ рд╣реИред"""
    def execute(self, research_topic: str) -> list:
        super().execute(f"Breaking down research topic: {research_topic}")
        # рдпрд╣ LLM рдХреЙрд▓ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдирд╣реАрдВ рдХрд░рддреА, рдмрд▓реНрдХрд┐ рдХреЗрд╡рд▓ рддрд░реНрдХ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рддреА рд╣реИред
        steps = [
            f"Define the core concepts of '{research_topic}'",
            f"Search for the latest industry trends and developments related to '{research_topic}'",
            f"Identify real-world case studies or success stories",
            "Consolidate all findings into a structured report outline"
        ]
        print(f"[{self.name}] Plan created with {len(steps)} steps.")
        return steps

class ResearcherAgent(BaseAgent):
    """Google Search Tool рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдбреЗрдЯрд╛ рдПрдХрддреНрд░ рдХрд░рддрд╛ рд╣реИред (рдпрд╣рд╛рдВ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд╣реЛрддреА рд╣реИ)"""
    def execute(self, search_query: str) -> str:
        super().execute(f"Conducting web search for: {search_query}")
        
        # рд░рд┐рд╕рд░реНрдЪрд░ рдПрдЬреЗрдВрдЯ рдХреЛ рдХреЗрд╡рд▓ Google Search Tool рдХреА рдЕрдиреБрдорддрд┐ рд╣реИ
        available_tools = ['google_search_tool']
        
        # 1. LLM рдХреЛ рдкреНрд░реЙрдореНрдкреНрдЯ рднреЗрдЬреЗрдВ рдФрд░ рдЙрдкрд▓рдмреНрдз рдЯреВрд▓ рджреЗрдВ
        llm_response = gemini_agent_call(
            prompt=f"Find the most up-to-date information for: {search_query}", 
            available_tools=available_tools
        )
        
        # 2. LLM рдХреЗ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рдХреА рдЬрд╛рдБрдЪ рдХрд░реЗрдВ
        if llm_response.get("type") == "function_call":
            function_name = llm_response['name']
            function_args = llm_response['args']
            
            # 3. рдЯреВрд▓ рдХреЛ рдирд┐рд╖реНрдкрд╛рджрд┐рдд (Execute) рдХрд░реЗрдВ
            if function_name in TOOL_MAP:
                tool_function = TOOL_MAP[function_name]
                tool_output = tool_function(**function_args)
                
                # 4. рдЯреВрд▓ рдХреЗ рдЖрдЙрдЯрдкреБрдЯ рдХреЛ рд╡рд╛рдкрд╕ LLM рдХреЛ рднреЗрдЬреЗрдВ (Final Response - Sim.)
                #    рдЕрд╕рд▓реА рд╕рд┐рд╕реНрдЯрдо рдореЗрдВ, рдЖрдк рдЗрд╕ рдЖрдЙрдЯрдкреБрдЯ рдХреЗ рд╕рд╛рде LLM рдХреЛ рдлрд┐рд░ рд╕реЗ рдХреЙрд▓ рдХрд░реЗрдВрдЧреЗ
                print(f"   [LLM Input: Tool Result] Tool output sent back to LLM for final answer.")
                return f"Tool Output: {tool_output}" # рд╣рдо рдпрд╣рд╛рдБ рдЯреВрд▓ рдЖрдЙрдЯрдкреБрдЯ рдХреЛ рд╕реАрдзреЗ рднреЗрдЬ рд░рд╣реЗ рд╣реИрдВ

        # 5. рдЕрдЧрд░ рдЯреВрд▓ рдХреЙрд▓ рдирд╣реАрдВ рд╣реБрдИ, рддреЛ рд╕реАрдзрд╛ рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕
        return llm_response.get("content", "Search failed or no tool call was made.")


class SummarizerAgent(BaseAgent):
    """рд░рд┐рд╕рд░реНрдЪ рдХреЗ рдкрд░рд┐рдгрд╛рдореЛрдВ рдХреЛ рдПрдХ рд╡рд┐рд╕реНрддреГрдд рдФрд░ рд╕рдВрд░рдЪрд┐рдд рд░рд┐рдкреЛрд░реНрдЯ рдореЗрдВ рд╕рдВрдХрд▓рд┐рдд рдХрд░рддрд╛ рд╣реИред"""
    def execute(self, all_data: dict, topic: str) -> str:
        super().execute(f"Compiling final report for topic: {topic}")
        # рдпрд╣ LLM рдХреЙрд▓ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдирд╣реАрдВ рдХрд░рддреА, рдмрд▓реНрдХрд┐ рдХреЗрд╡рд▓ рд╕рд╛рд░рд╛рдВрд╢ (summarization) рддрд░реНрдХ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рддреА рд╣реИред
        report = f"Detailed Report on {topic}\n"
        report += "========================================\n"
        
        for step, result in all_data.items():
            report += f"\n--- Section: {step} ---\n"
            # рдЯреВрд▓ рдХреЗ рдЖрдЙрдЯрдкреБрдЯ рдХреЛ рд╕реАрдзреЗ рд╕рд╛рд░рд╛рдВрд╢ рдореЗрдВ рд╢рд╛рдорд┐рд▓ рдХрд░реЗрдВ
            report += f"Summary of Findings: {result.replace('Tool Output: ', '')}\n"
            
        report += "\nFinal Conclusion: The research confirms the topic's high relevance in the current market landscape."
        print(f"[{self.name}] Final report compiled by LLM.")
        return report

class SchedulerAgent(BaseAgent):
    """рдХреИрд▓реЗрдВрдбрд░ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдЗрд╡реЗрдВрдЯреНрд╕ рдХреЛ рд╕рдВрднрд╛рд▓рддрд╛ рд╣реИред (рдпрд╣рд╛рдВ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд╣реЛрддреА рд╣реИ)"""
    def execute(self, event_details: dict) -> str:
        super().execute(f"Scheduling a new event: {event_details.get('title')}")
        
        # рд╢реЗрдбреНрдпреВрд▓рд░ рдПрдЬреЗрдВрдЯ рдХреЛ рдХреЗрд╡рд▓ Calendar Tool рдХреА рдЕрдиреБрдорддрд┐ рд╣реИ
        available_tools = ['calendar_tool']
        
        # 1. LLM рдХреЛ рдкреНрд░реЙрдореНрдкреНрдЯ рднреЗрдЬреЗрдВ
        llm_response = gemini_agent_call(
            prompt=f"Schedule a meeting titled '{event_details.get('title')}' at {event_details.get('time')}.", 
            available_tools=available_tools
        )
        
        # 2. рдЯреВрд▓ рдХреЙрд▓ рдХреА рдЬрд╛рдБрдЪ рдХрд░реЗрдВ
        if llm_response.get("type") == "function_call":
            function_name = llm_response['name']
            function_args = llm_response['args']
            
            # 3. рдЯреВрд▓ рдХреЛ рдирд┐рд╖реНрдкрд╛рджрд┐рдд рдХрд░реЗрдВ
            if function_name in TOOL_MAP:
                tool_function = TOOL_MAP[function_name]
                tool_output = tool_function(**function_args)
                
                print(f"   [LLM Input: Tool Result] Event successfully created.")
                return f"Scheduling complete. Status: {tool_output}"

        return "Scheduling failed: LLM did not call the calendar tool correctly."

class CommunicatorAgent(BaseAgent):
    """рдореИрд╕реЗрдЬрд┐рдВрдЧ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рд╕рдВрд╡рд╛рдж рднреЗрдЬрддрд╛ рд╣реИред (рдпрд╣рд╛рдВ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд╣реЛрддреА рд╣реИ)"""
    def execute(self, recipient: str, subject: str, content: str) -> str:
        super().execute(f"Preparing message for: {recipient}")
        
        # рдХрдореНрдпреБрдирд┐рдХреЗрдЯрд░ рдПрдЬреЗрдВрдЯ рдХреЛ рдХреЗрд╡рд▓ Messaging Tool рдХреА рдЕрдиреБрдорддрд┐ рд╣реИ
        available_tools = ['messaging_tool']
        
        # 1. LLM рдХреЛ рдкреНрд░реЙрдореНрдкреНрдЯ рднреЗрдЬреЗрдВ
        llm_response = gemini_agent_call(
            prompt=f"send a message to {recipient} with subject '{subject}' and content: {content[:30]}...", 
            available_tools=available_tools
        )
        
        # 2. рдЯреВрд▓ рдХреЙрд▓ рдХреА рдЬрд╛рдБрдЪ рдХрд░реЗрдВ
        if llm_response.get("type") == "function_call":
            function_name = llm_response['name']
            function_args = llm_response['args']
            
            # 3. рдЯреВрд▓ рдХреЛ рдирд┐рд╖реНрдкрд╛рджрд┐рдд рдХрд░реЗрдВ
            if function_name in TOOL_MAP:
                tool_function = TOOL_MAP[function_name]
                tool_output = tool_function(**function_args)
                
                print(f"   [LLM Input: Tool Result] Message successfully sent.")
                return f"Communication complete. Status: {tool_output}"

        return "Communication failed: LLM did not call the messaging tool correctly."

# ==============================================================================
# 4. рдорд╛рд╕реНрдЯрд░ рдХрдВрдЯреНрд░реЛрд▓рд░/рдСрд░реНрдХреЗрд╕реНрдЯреНрд░реЗрдЯрд░ рдПрдЬреЗрдВрдЯ (Master Controller/Orchestrator Agent)
# ==============================================================================

class OrchestratorAgent(BaseAgent):
    """рдЙрдкрдпреЛрдЧрдХрд░реНрддрд╛ рдХреА рдорд╛рдВрдЧ рдХреЛ рдкрд╣рдЪрд╛рдирддрд╛ рд╣реИ рдФрд░ рд╕рд╣реА рд╡рд┐рд╢реЗрд╖рдЬреНрдЮ рдПрдЬреЗрдВрдЯ рдХреЛ рд╕реМрдВрдкрддрд╛ рд╣реИред"""
    def __init__(self):
        super().__init__("ORCHESTRATOR_AGENT")
        self.planner = PlannerAgent("PLANNER_AGENT")
        self.researcher = ResearcherAgent("RESEARCHER_AGENT")
        self.summarizer = SummarizerAgent("SUMMARIZER_AGENT")
        self.scheduler = SchedulerAgent("SCHEDULER_AGENT")
        self.communicator = CommunicatorAgent("COMMUNICATOR_AGENT")

    # Routing logic remains the same (simulated by simple string matching)
    def route_request(self, user_request: str) -> str:
        if any(keyword in user_request.lower() for keyword in ["research", "report", "topic", "trends"]):
            return "RESEARCH_DOMAIN"
        elif any(keyword in user_request.lower() for keyword in ["schedule", "meeting", "event", "plan"]):
            return "SCHEDULING_DOMAIN"
        else:
            return "UNKNOWN"

    def process_request(self, user_request: str) -> str:
        """рдорд╛рд╕реНрдЯрд░ рдПрдЧреНрдЬреАрдХреНрдпреВрд╢рди рдлреНрд▓реЛред"""
        print("\n" + "="*70)
        print(f"[{self.name}] New Request: {user_request}")
        print("="*70)

        domain = self.route_request(user_request)
        
        if domain == "RESEARCH_DOMAIN":
            print(f"[{self.name}] Routing to RESEARCH DOMAIN.")
            topic = user_request.split("on")[-1].strip() if "on" in user_request.lower() else "General AI Topic"
            
            research_plan = self.planner.execute(topic)
            all_results = {}
            
            for step in research_plan:
                # Researcher Agent calls Google Search Tool
                search_result = self.researcher.execute(step)
                all_results[step] = search_result
                
            final_report = self.summarizer.execute(all_results, topic)
            
            # Send the report using the Communicator Agent (which also uses function calling)
            communication_status = self.communicator.execute("Stakeholder@corp.com", f"Final Report on: {topic}", final_report[:100] + "...")
            return f"Final Task Result: Report compiled and communication initiated. {communication_status}"

        elif domain == "SCHEDULING_DOMAIN":
            print(f"[{self.name}] Routing to SCHEDULING DOMAIN.")
            
            # Simple simulation for extracting event details
            event_details = {
                "title": "Agent Intensive Capstone Discussion",
                "time": "Tomorrow at 2 PM",
                "attendees": ["Bhai", "Team Member"],
            }
            
            # Scheduler Agent calls Calendar Tool
            schedule_status = self.scheduler.execute(event_details)
            
            # Communicator Agent sends invitation message
            email_body = f"Hi All, Invitation confirmed: {event_details['title']} scheduled for {event_details['time']}."
            message_status = self.communicator.execute("All@team.com", f"Invitation: {event_details['title']}", email_body)
            
            return f"Scheduling Flow Completed. Status: {schedule_status}. Message Status: {message_status}"

        else:
            print(f"[{self.name}] Could not determine the domain for the request.")
            return "Sorry, I can only handle complex research and scheduling tasks."

# ==============================================================================
# 5. рдореБрдЦреНрдп рдирд┐рд╖реНрдкрд╛рджрди (Main Execution)
# ==============================================================================

def main():
    """
    рд╕рд┐рд╕реНрдЯрдо рдХреЛ рд╢реБрд░реВ рдХрд░рддрд╛ рд╣реИ рдФрд░ рдЙрджрд╛рд╣рд░рдг рдЕрдиреБрд░реЛрдзреЛрдВ рдХреЛ рд╕рдВрд╕рд╛рдзрд┐рдд (process) рдХрд░рддрд╛ рд╣реИред
    """
    print("--- Ultimate Personal Assistant System Starting with Function Calling Logic ---")
    assistant = OrchestratorAgent()
    
    # 1. рд░рд┐рд╕рд░реНрдЪ рдФрд░ рдХрдореНрдпреБрдирд┐рдХреЗрд╢рди рдХрд╛рд░реНрдп рдХрд╛ рдЙрджрд╛рд╣рд░рдг (Research and Communication Task Example)
    research_prompt = "I need a detailed report on the latest AI trends in 2025."
    assistant.process_request(research_prompt)

    # 2. рд╢реЗрдбреНрдпреВрд▓рд┐рдВрдЧ рдХрд╛рд░реНрдп рдХрд╛ рдЙрджрд╛рд╣рд░рдг (Scheduling Task Example)
    schedule_prompt = "Can you schedule a team meeting for tomorrow afternoon to review the final project submission?"
    assistant.process_request(schedule_prompt)
    
    print("\n--- Ultimate Personal Assistant System Finished ---")
    print("Function Calling and Multi-Agent Orchestration Demonstrated Successfully.")

if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
"""
Kaggle Agents Intensive - Capstone Project: The Ultimate Personal Assistant v3

рдпрд╣ рдХреЛрдб Gemini API рдХреЗ рд╕рд╛рде рд╡рд╛рд╕реНрддрд╡рд┐рдХ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ рдХреЛ рд▓рд╛рдЧреВ рдХрд░рдиреЗ рдХреЗ рд▓рд┐рдП рддреИрдпрд╛рд░ рд╣реИред
рдпрд╣ LLM рдХреЛ рдЯреВрд▓реНрд╕ (Google Search, Calendar, Messaging) рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдиреЗ рдореЗрдВ рд╕рдХреНрд╖рдо рдмрдирд╛рддрд╛ рд╣реИред
"""
import time
from typing import List, Dict, Any

# ==============================================================================
# I. рдЬреЗрдорд┐рдиреА рдХреНрд▓рд╛рдЗрдВрдЯ рдФрд░ рдЯреВрд▓ рдбреЗрдлрд┐рдирд┐рд╢рди (Gemini Client and Tool Definitions)
# ==============================================================================

# **1. рдЬреЗрдорд┐рдиреА рдХреНрд▓рд╛рдЗрдВрдЯ рдкреНрд▓реЗрд╕рд╣реЛрд▓реНрдбрд░ (Gemini Client Placeholder)**
# Kaggle рдореЗрдВ, API Key рдХреЛ рд╕реАрдзреЗ рдЙрдкрдпреЛрдЧ рди рдХрд░реЗрдВред
# рдЖрдкрдХреЛ Kaggle secrets рдпрд╛ environment variables рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдирд╛ рд╣реЛрдЧрд╛ред
# рдпрд╣ рдПрдХ SIMULATED рдХреНрд▓рд╛рдЗрдВрдЯ рд╣реИред
class SimulatedGeminiClient:
    """рд╡рд╛рд╕реНрддрд╡рд┐рдХ Gemini SDK рдХреНрд▓рд╛рдЗрдВрдЯ рдХрд╛ рдЕрдиреБрдХрд░рдг рдХрд░рддрд╛ рд╣реИред"""
    def generate_content(self, model: str, contents: List[Dict], tools: List[Any]):
        """Gemini generate_content API рдХреЙрд▓ рдХрд╛ рдЕрдиреБрдХрд░рдгред"""
        print(f"\n[API CALL] Sending request to Gemini model: {model}...")
        time.sleep(1) 
        
        # рд╕рд┐рдореБрд▓реЗрдЯреЗрдб LLM рдирд┐рд░реНрдгрдп (Simulated LLM Decision)
        last_part = contents[-1]['parts'][0]['text']
        
        if "latest AI trends" in last_part:
            # LLM рдиреЗ Google Search Tool рдХреЛ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдлреИрд╕рд▓рд╛ рдХрд┐рдпрд╛
            return {
                "function_calls": [{
                    "name": "google_search_tool",
                    "args": {"query": "latest AI trends in Generative Agents"}
                }]
            }
        elif "schedule a meeting" in last_part:
            # LLM рдиреЗ Calendar Tool рдХреЛ рдХреЙрд▓ рдХрд░рдиреЗ рдХрд╛ рдлреИрд╕рд▓рд╛ рдХрд┐рдпрд╛
            return {
                "function_calls": [{
                    "name": "calendar_tool",
                    "args": {"title": "Team Sync", "time": "2025-11-20 14:00", "attendees": ["User", "Team"]}
                }]
            }
        else:
            # LLM рдиреЗ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдиреЗ рдХреЗ рдмрдЬрд╛рдп рд╕реАрдзрд╛ рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рджрд┐рдпрд╛
            return {
                "text": "The Agent has finished its thought process and compiled the final output based on all tools and data provided."
            }

# рд╡рд╛рд╕реНрддрд╡рд┐рдХ рдЙрдкрдпреЛрдЧ рдХреЗ рд▓рд┐рдП, рдЖрдкрдХреЛ 'google-genai' SDK рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдирд╛ рд╣реЛрдЧрд╛:
# from google import genai
# client = genai.Client(api_key="YOUR_API_KEY") 
client = SimulatedGeminiClient()
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'


# **2. рдЯреВрд▓ рдлрдВрдХреНрд╢рдВрд╕ (Tool Functions) - (рдпреЗ рд╡рд╣реА рд╣реИрдВ рдЬреЛ рдкрд╣рд▓реЗ рдереЗ)**

def google_search_tool(query: str) -> str:
    """рд╡реЗрдм рдкрд░ рдЬрд╛рдирдХрд╛рд░реА рдЦреЛрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред (Google Search API рдХрд╛ рдЕрдиреБрдХрд░рдг)"""
    print(f"\n[TOOL CALLED: GoogleSearchTool] Searching for: '{query}'...")
    time.sleep(1)
    if "trends" in query.lower():
        return "Search Results: AI trends show massive growth in Generative Agents and RAG applications in 2025. This requires deep understanding of model orchestration."
    return f"Search Results: Found 3 relevant articles about '{query}'. Summary: Key points are promising."

def calendar_tool(title: str, time: str, attendees: List[str]) -> str:
    """рдХреИрд▓реЗрдВрдбрд░ рдореЗрдВ рдЗрд╡реЗрдВрдЯ рдмрдирд╛рдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: CalendarTool] Scheduling: {title} on {time} with {len(attendees)} people.")
    time.sleep(0.5)
    return f"Calendar: Event '{title}' successfully scheduled for {time}."

def messaging_tool(recipient: str, subject: str, body: str) -> str:
    """рдХрд┐рд╕реА рдХреЛ рдИрдореЗрд▓ рдпрд╛ рдореИрд╕реЗрдЬ рднреЗрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: MessagingTool] Sending message to {recipient}. Subject: {subject}")
    time.sleep(0.5)
    return f"Messaging: Email to {recipient} with subject '{subject}' sent successfully."

# рдЯреВрд▓ рдореИрдкрд┐рдВрдЧ: рдЯреВрд▓ рдХрд╛ рдирд╛рдо LLM рд░рд┐рд╕реНрдкрд╛рдВрд╕ рд╕реЗ рдлрдВрдХреНрд╢рди рдХреЛ рдЬреЛрдбрд╝рддрд╛ рд╣реИ
TOOL_MAP = {
    "google_search_tool": google_search_tool,
    "calendar_tool": calendar_tool,
    "messaging_tool": messaging_tool
}

# ==============================================================================
# II. рдХреЛрд░ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ (The Core Function Calling Logic)
# ==============================================================================

def execute_gemini_call(prompt: str, available_tools: List[Any], history: List[Dict] = None) -> str:
    """
    рдЬреЗрдорд┐рдиреА рдореЙрдбрд▓ рдХреЛ рдХреЙрд▓ рдХрд░рддрд╛ рд╣реИ, рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рдХреЛ рд╕рдВрднрд╛рд▓рддрд╛ рд╣реИ (рдЯреВ-рд╕реНрдЯреЗрдк рд▓реВрдк)ред
    [attachment_0](attachment)
    """
    
    # 1. рдкреНрд░рд╛рд░рдВрднрд┐рдХ рдХреЙрд▓: рдкреНрд░реЙрдореНрдкреНрдЯ рдФрд░ рдЯреВрд▓ рд╕реНрдХреАрдорд╛ рднреЗрдЬреЗрдВ
    contents = history if history is not None else []
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    response = client.generate_content(
        model=MODEL_NAME,
        contents=contents,
        tools=available_tools
    )
    
    # 2. рдЯреВрд▓ рдХреЙрд▓ рдХреЗ рд▓рд┐рдП рд░рд┐рд╕реНрдкрд╛рдВрд╕ рдХреА рдЬрд╛рдБрдЪ рдХрд░реЗрдВ
    if "function_calls" in response:
        print("   [LLM DECISION] Function Call requested.")
        
        # LLM рдиреЗ рдПрдХ рдпрд╛ рдЕрдзрд┐рдХ рдлрдВрдХреНрд╢рди рдХреЙрд▓ рдХрд╛ рдЕрдиреБрд░реЛрдз рдХрд┐рдпрд╛
        function_call = response['function_calls'][0] 
        function_name = function_call['name']
        function_args = function_call['args']
        
        if function_name in TOOL_MAP:
            tool_function = TOOL_MAP[function_name]
            
            # 3. рдлрдВрдХреНрд╢рди (рдЯреВрд▓) рдХреЛ рдирд┐рд╖реНрдкрд╛рджрд┐рдд рдХрд░реЗрдВ
            tool_output = tool_function(**function_args)
            
            # 4. рдЯреВрд▓ рдЖрдЙрдЯрдкреБрдЯ рдХреЛ рд╡рд╛рдкрд╕ LLM рдХреЛ рднреЗрдЬреЗрдВ (Step 2 of 2)
            print("   [TOOL OUTPUT] Sending Tool Output back to LLM for final answer.")
            
            # рдЗрддрд┐рд╣рд╛рд╕ рдЕрдкрдбреЗрдЯ рдХрд░реЗрдВ: LLM рдХреЗ рдЕрдиреБрд░реЛрдз рдФрд░ рдЯреВрд▓ рдХреЗ рдкрд░рд┐рдгрд╛рдо рдХреЛ рдЬреЛрдбрд╝реЗрдВ
            contents.append({"role": "model", "parts": [response]})
            contents.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "content": tool_output}}]})
            
            # LLM рдХреЛ рдлрд┐рд░ рд╕реЗ рдХреЙрд▓ рдХрд░реЗрдВ
            final_response = client.generate_content(
                model=MODEL_NAME,
                contents=contents,
                tools=available_tools # рдЯреВрд▓реНрд╕ рдЕрднреА рднреА рдЙрдкрд▓рдмреНрдз рд╣реИрдВ
            )
            
            # 5. рдЕрдВрддрд┐рдо рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рд▓реМрдЯрд╛рдПрдБ
            return final_response.get("text", "Final response generation failed after tool execution.")
            
    # 6. рдЕрдЧрд░ рдХреЛрдИ рдлрдВрдХреНрд╢рди рдХреЙрд▓ рдирд╣реАрдВ рд╣реБрдИ, рддреЛ рд╕реАрдзрд╛ рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рд▓реМрдЯрд╛рдПрдБ
    return response.get("text", "LLM responded with plain text.")


# ==============================================================================
# III. рд╡рд┐рд╢реЗрд╖рдЬреНрдЮ рдПрдЬреЗрдВрдЯ (Specialist Agents) - рдЕрдм рд╡реЗ рдирдП рдХреЛрд░ рд▓реЙрдЬрд┐рдХ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рддреЗ рд╣реИрдВ
# ==============================================================================

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        
class PlannerAgent(BaseAgent):
    """рдХрд╛рд░реНрдпреЛрдВ рдХреЛ рддреЛрдбрд╝рддрд╛ рд╣реИ - рдпрд╣ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдирд╣реАрдВ рдХрд░рддрд╛ рд╣реИ, рдХреЗрд╡рд▓ LLM рд▓реЙрдЬрд┐рдХред"""
    def execute(self, research_topic: str) -> list:
        print(f"[{self.name}] Planning research steps for: {research_topic}")
        # рдпрд╣рд╛рдВ LLM рдХрд╛ рддрд░реНрдХ (logic) рд╕реАрдзреЗ рдХрд╛рдо рдХрд░реЗрдЧрд╛
        steps = [
            f"Define the core concepts of '{research_topic}'",
            f"Search for the latest industry trends and developments for '{research_topic}'",
            f"Identify real-world case studies or success stories",
            "Consolidate all findings into a structured report outline"
        ]
        return steps

class ResearcherAgent(BaseAgent):
    """Google Search Tool рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдбреЗрдЯрд╛ рдПрдХрддреНрд░ рдХрд░рддрд╛ рд╣реИред"""
    def execute(self, search_query: str) -> str:
        # Researcher рдХреЛ рдХреЗрд╡рд▓ Google Search Tool рдХреА рдЕрдиреБрдорддрд┐ рд╣реИ
        tools = [google_search_tool] 
        prompt = f"Find the most up-to-date information and current trends for the topic: {search_query}"
        
        # рдХреЛрд░ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ рдХреЛ рдХреЙрд▓ рдХрд░реЗрдВ
        result_text = execute_gemini_call(prompt, tools)
        return result_text

class SummarizerAgent(BaseAgent):
    """рд░рд┐рд╕рд░реНрдЪ рдХреЗ рдкрд░рд┐рдгрд╛рдореЛрдВ рдХреЛ рдПрдХ рд╡рд┐рд╕реНрддреГрдд рдФрд░ рд╕рдВрд░рдЪрд┐рдд рд░рд┐рдкреЛрд░реНрдЯ рдореЗрдВ рд╕рдВрдХрд▓рд┐рдд рдХрд░рддрд╛ рд╣реИред"""
    def execute(self, all_data: Dict[str, str], topic: str) -> str:
        # рдпрд╣ рдПрдЬреЗрдВрдЯ рдХрд┐рд╕реА рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдирд╣реАрдВ рдХрд░рддрд╛, рдХреЗрд╡рд▓ LLM рдХреЗ рд╕рд╛рд░рд╛рдВрд╢ (summarization) рддрд░реНрдХ рдХрд╛ред
        prompt = f"Compile a detailed, structured report on '{topic}' using the following research steps and findings:\n{json.dumps(all_data, indent=2)}"
        
        # рдпрд╣рд╛рдВ рд╣рдо рдХреЗрд╡рд▓ LLM рдХреЗ рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рдХреА рдЕрдкреЗрдХреНрд╖рд╛ рдХрд░рддреЗ рд╣реИрдВ
        final_report = execute_gemini_call(prompt, available_tools=[]) 
        
        report_text = f"Detailed Report on {topic}\n"
        report_text += "========================================\n"
        for step, result in all_data.items():
             report_text += f"\n--- Section: {step} ---\n{result}\n"
        report_text += f"\nFinal Summary by LLM: {final_report}"
        return report_text

class SchedulerAgent(BaseAgent):
    """рдХреИрд▓реЗрдВрдбрд░ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдЗрд╡реЗрдВрдЯреНрд╕ рдХреЛ рд╕рдВрднрд╛рд▓рддрд╛ рд╣реИред"""
    def execute(self, event_title: str, event_time: str, attendees: List[str]) -> str:
        # Scheduler рдХреЛ рдХреЗрд╡рд▓ Calendar Tool рдХреА рдЕрдиреБрдорддрд┐ рд╣реИ
        tools = [calendar_tool]
        prompt = f"Please schedule a meeting titled '{event_title}' with attendees {', '.join(attendees)} at {event_time}."
        return execute_gemini_call(prompt, tools)

class CommunicatorAgent(BaseAgent):
    """рдореИрд╕реЗрдЬрд┐рдВрдЧ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рд╕рдВрд╡рд╛рдж рднреЗрдЬрддрд╛ рд╣реИред"""
    def execute(self, recipient: str, subject: str, content: str) -> str:
        # Communicator рдХреЛ рдХреЗрд╡рд▓ Messaging Tool рдХреА рдЕрдиреБрдорддрд┐ рд╣реИ
        tools = [messaging_tool]
        prompt = f"Send an urgent email to {recipient} with the subject '{subject}'. The body content is: {content[:100]}..."
        return execute_gemini_call(prompt, tools)

# ==============================================================================
# IV. рдорд╛рд╕реНрдЯрд░ рдСрд░реНрдХреЗрд╕реНрдЯреНрд░реЗрдЯрд░ рдПрдЬреЗрдВрдЯ (Master Orchestrator Agent)
# ==============================================================================

class OrchestratorAgent(BaseAgent):
    """рдЙрдкрдпреЛрдЧрдХрд░реНрддрд╛ рдХреА рдорд╛рдВрдЧ рдХреЛ рдкрд╣рдЪрд╛рдирддрд╛ рд╣реИ рдФрд░ рд╕рд╣реА рд╡рд┐рд╢реЗрд╖рдЬреНрдЮ рдПрдЬреЗрдВрдЯ рдХреЛ рд╕реМрдВрдкрддрд╛ рд╣реИред"""
    def __init__(self):
        super().__init__("ORCHESTRATOR_AGENT")
        self.planner = PlannerAgent("PLANNER_AGENT")
        self.researcher = ResearcherAgent("RESEARCHER_AGENT")
        self.summarizer = SummarizerAgent("SUMMARIZER_AGENT")
        self.scheduler = SchedulerAgent("SCHEDULER_AGENT")
        self.communicator = CommunicatorAgent("COMMUNICATOR_AGENT")

    # Routing logic remains simple for simulation
    def route_request(self, user_request: str) -> str:
        if "research" in user_request.lower() or "report" in user_request.lower():
            return "RESEARCH_DOMAIN"
        elif "schedule" in user_request.lower() or "meeting" in user_request.lower():
            return "SCHEDULING_DOMAIN"
        else:
            return "UNKNOWN"

    def process_request(self, user_request: str) -> str:
        print("\n" + "="*70)
        print(f"[{self.name}] New Request: {user_request}")
        print("="*70)

        domain = self.route_request(user_request)
        
        if domain == "RESEARCH_DOMAIN":
            topic = user_request.split("on")[-1].strip() if "on" in user_request.lower() else "The Future of AI"
            
            research_plan = self.planner.execute(topic)
            all_results = {}
            
            for step in research_plan:
                # Researcher Agent calls Google Search Tool via Function Calling
                search_result = self.researcher.execute(step)
                all_results[step] = search_result
                
            final_report = self.summarizer.execute(all_results, topic)
            
            # Send the report using the Communicator Agent
            communication_status = self.communicator.execute("Stakeholder@corp.com", f"Final Report: {topic}", final_report)
            return f"\nFINAL RESULT: Report compiled and communication initiated. Status: {communication_status}"

        elif domain == "SCHEDULING_DOMAIN":
            # Simple simulation for event details
            event_title = "Capstone Final Review"
            event_time = "2025-11-20 2:00 PM"
            attendees = ["User", "Mentor"]
            
            # Scheduler Agent calls Calendar Tool via Function Calling
            schedule_status = self.scheduler.execute(event_title, event_time, attendees)
            
            # Communicator Agent sends invitation message
            email_body = f"Hi All, Invitation confirmed: {event_title} scheduled for {event_time}. Please review the attached report."
            message_status = self.communicator.execute("All@team.com", f"Invitation: {event_title}", email_body)
            
            return f"\nFINAL RESULT: Scheduling Flow Completed. Status: {schedule_status}. Message Status: {message_status}"

        else:
            return f"[{self.name}] Could not determine the domain for the request. Only Research and Scheduling supported."

# ==============================================================================
# V. рдореБрдЦреНрдп рдирд┐рд╖реНрдкрд╛рджрди (Main Execution)
# ==============================================================================

def main():
    print("--- Ultimate Personal Assistant System (Gemini Function Calling Ready) ---")
    assistant = OrchestratorAgent()
    
    # 1. рд░рд┐рд╕рд░реНрдЪ рдХрд╛рд░реНрдп рдХрд╛ рдЙрджрд╛рд╣рд░рдг (Research Task Example - Calls Planner, Researcher, Summarizer, Communicator)
    research_prompt = "Prepare a detailed report on the latest AI trends in 2025 on Generative Agents and RAG."
    assistant.process_request(research_prompt)

    # 2. рд╢реЗрдбреНрдпреВрд▓рд┐рдВрдЧ рдХрд╛рд░реНрдп рдХрд╛ рдЙрджрд╛рд╣рд░рдг (Scheduling Task Example - Calls Scheduler, Communicator)
    schedule_prompt = "Can you schedule a team meeting for tomorrow afternoon to review the final project submission?"
    assistant.process_request(schedule_prompt)
    
    print("\n--- Project Framework is Complete. Ready for API Key Integration ---")

if __name__ == "__main__":
    main()


!pip install google-genai pydantic --quiet



# from google import genai
# client = genai.Client() # рдпрд╣ Kaggle Secrets рд╕реЗ API Key рдХреЛ рд╕реНрд╡рддрдГ рдЙрдард╛ рд▓реЗрдЧрд╛

# **NOTE:** рдЪреВрдВрдХрд┐ рдореИрдВ рдЕрднреА рднреА рдПрдХ рд╕рд┐рдореБрд▓реЗрд╢рди рдПрдирд╡рд╛рдпрд░рдирдореЗрдВрдЯ рдореЗрдВ рд╣реВрдБ, рдореИрдВ рдЖрдкрдХреЗ рд▓рд┐рдП рдПрдХ 
#   'SimulatedClient' рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░ рд░рд╣рд╛ рд╣реВрдБред рдЖрдкрдХреЛ рдЗрд╕реЗ рдЕрдкрдиреА рдиреЛрдЯрдмреБрдХ рдореЗрдВ
#   рдЕрд╕рд▓реА 'genai.Client()' рд╕реЗ рдмрджрд▓рдирд╛ рд╣реЛрдЧрд╛!



# -*- coding: utf-8 -*-
"""
Kaggle Agents Intensive - Capstone Project: The Ultimate Personal Assistant (Final Version)

рдпрд╣ рдХреЛрдб рд╡рд╛рд╕реНрддрд╡рд┐рдХ Gemini API рдФрд░ Pydantic Schemas рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ рдХреЛ рд▓рд╛рдЧреВ рдХрд░рддрд╛ рд╣реИред

NOTE: 'SimulatedGeminiClient' рдХреЛ рдЕрдкрдиреА Kaggle Notebook рдореЗрдВ 'genai.Client()' рд╕реЗ рдмрджрд▓реЗрдВред
"""
import time
import json
from typing import List, Dict, Any, Callable
from pydantic import BaseModel, Field

# рдЕрдЧрд░ рдЖрдк Kaggle рдореЗрдВ рд╣реИрдВ, рддреЛ рдЗрд╕реЗ рдЕрдирдХрдореЗрдВрдЯ рдХрд░реЗрдВ:
# from google import genai 
# from google.genai.errors import APIError 

# ==============================================================================
# I. рдЬреЗрдорд┐рдиреА рдХреНрд▓рд╛рдЗрдВрдЯ рдФрд░ рдЯреВрд▓ рдбреЗрдлрд┐рдирд┐рд╢рди (Gemini Client and Tool Definitions)
# ==============================================================================

# **1. рдХреНрд▓рд╛рдЗрдВрдЯ рд╕реЗрдЯрдЕрдк (Setup)**
# рдЖрдкрдХреЛ рдЕрдкрдиреА Kaggle Notebook рдореЗрдВ рдЕрд╕рд▓реА рдХреНрд▓рд╛рдЗрдВрдЯ рдХреЛ рдЗрдирд┐рд╢рд┐рдпрд▓рд╛рдЗрдЬрд╝ рдХрд░рдирд╛ рд╣реЛрдЧрд╛ред
class SimulatedGeminiClient:
    """рд╡рд╛рд╕реНрддрд╡рд┐рдХ Gemini SDK рдХреНрд▓рд╛рдЗрдВрдЯ рдХрд╛ рдЕрдиреБрдХрд░рдг рдХрд░рддрд╛ рд╣реИред (Kaggle рдореЗрдВ рдЗрд╕реЗ рдмрджрд▓реЗрдВ)"""
    def generate_content(self, model: str, contents: List[Dict], tools: List[Any]):
        print(f"\n[API CALL] Sending request to Gemini model: {model}...")
        time.sleep(1) 
        
        last_part = contents[-1]['parts'][0]['text']
        
        # рд╕рд┐рдореБрд▓реЗрдЯреЗрдб рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ
        if "trends" in last_part or "latest information" in last_part:
            return {"function_calls": [{"name": "google_search_tool", "args": {"query": "latest AI trends in Generative Agents"}}]}
        elif "schedule a meeting" in last_part:
            return {"function_calls": [{"name": "calendar_tool", "args": {"title": "Team Sync", "time": "2025-11-20 14:00", "attendees": ["User", "Team"]}}]}
        elif "tool_response" in last_part:
            # рдЯреВрд▓ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рдХреЗ рдмрд╛рдж LLM рдХрд╛ рдЕрдВрддрд┐рдо рдЯреЗрдХреНрд╕реНрдЯ
            return {"text": "Final Summary: The research confirms the massive growth in Generative Agents and RAG applications, requiring immediate upskilling."}
        else:
            return {"text": "Acknowledged: The agent has finished its thought process and compiled the final output."}


# **Kaggle рдореЗрдВ рдЙрдкрдпреЛрдЧ рдХреЗ рд▓рд┐рдП:**
# client = genai.Client()
client = SimulatedGeminiClient()
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'


# **2. рдЯреВрд▓ рдлрдВрдХреНрд╢рдВрд╕ (Tool Functions)**
# рдпреЗ рд╕рд╛рдзрд╛рд░рдг Python рдлрд╝рдВрдХреНрд╢рди рд╣реИрдВ рдЬрд┐рдиреНрд╣реЗрдВ LLM рдЙрдкрдпреЛрдЧ рдХрд░реЗрдЧрд╛ред

def google_search_tool(query: str) -> str:
    """рд╡реЗрдм рдкрд░ рдЬрд╛рдирдХрд╛рд░реА рдЦреЛрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред (Google Search API рдХрд╛ рдЕрдиреБрдХрд░рдг)"""
    print(f"\n[TOOL CALLED: GoogleSearchTool] Searching for: '{query}'...")
    time.sleep(1)
    if "trends" in query.lower():
        return "Search Results: AI trends show massive growth in Generative Agents and RAG applications in 2025. This requires deep understanding of model orchestration."
    return f"Search Results: Found 3 relevant articles about '{query}'. Summary: Key points are promising."

def calendar_tool(title: str, time: str, attendees: List[str]) -> str:
    """рдХреИрд▓реЗрдВрдбрд░ рдореЗрдВ рдЗрд╡реЗрдВрдЯ рдмрдирд╛рдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: CalendarTool] Scheduling: {title} on {time} with {len(attendees)} people.")
    time.sleep(0.5)
    return f"Calendar: Event '{title}' successfully scheduled for {time}."

def messaging_tool(recipient: str, subject: str, body: str) -> str:
    """рдХрд┐рд╕реА рдХреЛ рдИрдореЗрд▓ рдпрд╛ рдореИрд╕реЗрдЬ рднреЗрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: MessagingTool] Sending message to {recipient}. Subject: {subject}")
    time.sleep(0.5)
    return f"Messaging: Email to {recipient} with subject '{subject}' sent successfully."

# рдЯреВрд▓ рдореИрдкрд┐рдВрдЧ: LLM рдХреЗ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рдХреЛ Python рдлрд╝рдВрдХреНрд╢рди рд╕реЗ рдЬреЛрдбрд╝рддрд╛ рд╣реИ
TOOL_MAP: Dict[str, Callable] = {
    "google_search_tool": google_search_tool,
    "calendar_tool": calendar_tool,
    "messaging_tool": messaging_tool
}

# **3. Pydantic рд╕реНрдХреАрдорд╛ (Schema) рдбреЗрдлрд┐рдирд┐рд╢рди**
# Gemini API рдХреЛ рдпрд╣ рдЬрд╛рдирдиреЗ рдХреЗ рд▓рд┐рдП рдЗрд╕рдХреА рдЖрд╡рд╢реНрдпрдХрддрд╛ рд╣реЛрддреА рд╣реИ рдХрд┐ рдлрд╝рдВрдХреНрд╢рди рдХреИрд╕реЗ рджрд┐рдЦрддреЗ рд╣реИрдВред

class GoogleSearchTool(BaseModel):
    """рд╡реЗрдм рдкрд░ рдЬрд╛рдирдХрд╛рд░реА рдЦреЛрдЬрдиреЗ рдХреЗ рд▓рд┐рдП рдЯреВрд▓ред"""
    query: str = Field(description="The specific search query to look up on the web.")

class CalendarTool(BaseModel):
    """рдХреИрд▓реЗрдВрдбрд░ рдореЗрдВ рдПрдХ рдирдпрд╛ рдЗрд╡реЗрдВрдЯ рд╢реЗрдбреНрдпреВрд▓ рдХрд░рдиреЗ рдХреЗ рд▓рд┐рдП рдЯреВрд▓ред"""
    title: str = Field(description="The title of the event/meeting.")
    time: str = Field(description="The date and time of the event.")
    attendees: List[str] = Field(description="A list of attendees for the event.")
    
class MessagingTool(BaseModel):
    """рдХрд┐рд╕реА рдкреНрд░рд╛рдкреНрддрдХрд░реНрддрд╛ рдХреЛ рдИрдореЗрд▓ рдпрд╛ рдореИрд╕реЗрдЬ рднреЗрдЬрдиреЗ рдХреЗ рд▓рд┐рдП рдЯреВрд▓ред"""
    recipient: str = Field(description="The email address or name of the recipient.")
    subject: str = Field(description="The subject line of the message.")
    body: str = Field(description="The full content of the message.")

# рд╕рднреА рд╕реНрдХреАрдорд╛ рдХреЛ рдПрдХ рд▓рд┐рд╕реНрдЯ рдореЗрдВ рд░рдЦреЗрдВ, рдЬрд┐рд╕реЗ LLM рдХреЛ рдкрд╛рд╕ рдХрд┐рдпрд╛ рдЬрд╛рдПрдЧрд╛
ALL_TOOL_SCHEMAS: List[BaseModel] = [GoogleSearchTool, CalendarTool, MessagingTool]

# ==============================================================================
# II. рдХреЛрд░ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ (The Core Function Calling Logic)
# ==============================================================================

def execute_gemini_call(prompt: str, available_tools: List[Callable], history: List[Dict] = None) -> str:
    """
    рдЬреЗрдорд┐рдиреА рдореЙрдбрд▓ рдХреЛ рдХреЙрд▓ рдХрд░рддрд╛ рд╣реИ рдФрд░ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рдХреЛ рд╕рдВрднрд╛рд▓рддрд╛ рд╣реИ (рдЯреВ-рд╕реНрдЯреЗрдк рд▓реВрдк)ред
    
    Args:
        prompt (str): рдЙрдкрдпреЛрдЧрдХрд░реНрддрд╛ рдХрд╛ рд╡рд░реНрддрдорд╛рди рдкреНрд░реЙрдореНрдкреНрдЯ рдпрд╛ рдПрдЬреЗрдВрдЯ рдХрд╛ рдЗрдВрд╕реНрдЯреНрд░рдХреНрд╢рдиред
        available_tools (List[Callable]): Python рдлрд╝рдВрдХреНрд╢рди рдХреА рд╕реВрдЪреА (рдЙрджрд╛: [google_search_tool])ред
        history (List[Dict]): рдмрд╛рддрдЪреАрдд рдХрд╛ рдЗрддрд┐рд╣рд╛рд╕ред
    
    Returns:
        str: LLM рджреНрд╡рд╛рд░рд╛ рдЬрдирд░реЗрдЯ рдХрд┐рдпрд╛ рдЧрдпрд╛ рдЕрдВрддрд┐рдо рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ред
    """
    
    # рдлрд╝рдВрдХреНрд╢рди рд▓рд┐рд╕реНрдЯ рд╕реЗ Pydantic рд╕реНрдХреАрдорд╛ рдмрдирд╛рдПрдБ
    tool_schemas = [s for s in ALL_TOOL_SCHEMAS if s.__name__.lower() in [f.__name__ for f in available_tools]]

    # 1. рдкреНрд░рд╛рд░рдВрднрд┐рдХ рдХреЙрд▓: рдкреНрд░реЙрдореНрдкреНрдЯ рдФрд░ рдЯреВрд▓ рд╕реНрдХреАрдорд╛ рднреЗрдЬреЗрдВ
    contents = history if history is not None else []
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    try:
        response = client.generate_content(
            model=MODEL_NAME,
            contents=contents,
            tools=tool_schemas # Pydantic рд╕реНрдХреАрдорд╛ рдпрд╣рд╛рдБ рдкрд╛рд╕ рдХреА рдЬрд╛рддреА рд╣реИрдВ
        )
    except Exception as e:
        return f"ERROR: LLM API Call failed: {e}"

    # 2. рдЯреВрд▓ рдХреЙрд▓ рдХреЗ рд▓рд┐рдП рд░рд┐рд╕реНрдкрд╛рдВрд╕ рдХреА рдЬрд╛рдБрдЪ рдХрд░реЗрдВ
    if response.get("function_calls"):
        print("   [LLM DECISION] Function Call requested. ")
        
        # LLM рдиреЗ рдПрдХ рдпрд╛ рдЕрдзрд┐рдХ рдлрдВрдХреНрд╢рди рдХреЙрд▓ рдХрд╛ рдЕрдиреБрд░реЛрдз рдХрд┐рдпрд╛
        function_call = response['function_calls'][0]
        function_name = function_call['name']
        function_args = dict(function_call['args'])
        
        if function_name in TOOL_MAP:
            tool_function = TOOL_MAP[function_name]
            
            # 3. рдлрдВрдХреНрд╢рди (рдЯреВрд▓) рдХреЛ рдирд┐рд╖реНрдкрд╛рджрд┐рдд рдХрд░реЗрдВ
            tool_output = tool_function(**function_args)
            
            # 4. рдЯреВрд▓ рдЖрдЙрдЯрдкреБрдЯ рдХреЛ рд╡рд╛рдкрд╕ LLM рдХреЛ рднреЗрдЬреЗрдВ (Step 2 of 2)
            print("   [TOOL OUTPUT] Sending Tool Output back to LLM for final answer.")
            
            # рдЗрддрд┐рд╣рд╛рд╕ рдЕрдкрдбреЗрдЯ рдХрд░реЗрдВ: LLM рдХреЗ рдЕрдиреБрд░реЛрдз рдФрд░ рдЯреВрд▓ рдХреЗ рдкрд░рд┐рдгрд╛рдо рдХреЛ рдЬреЛрдбрд╝реЗрдВ
            contents.append({"role": "model", "parts": [response]})
            contents.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "content": tool_output}}]})
            
            # LLM рдХреЛ рдлрд┐рд░ рд╕реЗ рдХреЙрд▓ рдХрд░реЗрдВ
            final_response = client.generate_content(
                model=MODEL_NAME,
                contents=contents,
                tools=tool_schemas
            )
            
            # 5. рдЕрдВрддрд┐рдо рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рд▓реМрдЯрд╛рдПрдБ
            return final_response.get("text", "Final response generation failed after tool execution.")
            
    # 6. рдЕрдЧрд░ рдХреЛрдИ рдлрдВрдХреНрд╢рди рдХреЙрд▓ рдирд╣реАрдВ рд╣реБрдИ, рддреЛ рд╕реАрдзрд╛ рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рд▓реМрдЯрд╛рдПрдБ
    return response.get("text", "LLM responded with plain text.")


# ==============================================================================
# III. рд╡рд┐рд╢реЗрд╖рдЬреНрдЮ рдПрдЬреЗрдВрдЯ (Specialist Agents) - рдЕрдм рд╡реЗ рдирдП рдХреЛрд░ рд▓реЙрдЬрд┐рдХ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рддреЗ рд╣реИрдВ
# ==============================================================================

# **(рдпреЗ рдПрдЬреЗрдВрдЯ рдХреНрд▓рд╛рд╕реЗрд╕ рдЕрдм рд╕реАрдзреЗ execute_gemini_call рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рддреА рд╣реИрдВ, рдЗрд╕рд▓рд┐рдП рдпреЗ рдареАрдХ рд╣реИрдВ)**

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        
class PlannerAgent(BaseAgent):
    """рдХрд╛рд░реНрдпреЛрдВ рдХреЛ рддреЛрдбрд╝рддрд╛ рд╣реИ - рдпрд╣ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдирд╣реАрдВ рдХрд░рддрд╛ рд╣реИ, рдХреЗрд╡рд▓ LLM рд▓реЙрдЬрд┐рдХред"""
    def execute(self, research_topic: str) -> list:
        print(f"[{self.name}] Planning research steps for: {research_topic}")
        # рдпрд╣рд╛рдВ LLM рдХрд╛ рддрд░реНрдХ (logic) рд╕реАрдзреЗ рдХрд╛рдо рдХрд░реЗрдЧрд╛
        steps = [
            f"Define the core concepts of '{research_topic}'",
            f"Search for the latest industry trends and developments for '{research_topic}'",
            f"Identify real-world case studies or success stories",
            "Consolidate all findings into a structured report outline"
        ]
        return steps

class ResearcherAgent(BaseAgent):
    """Google Search Tool рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдбреЗрдЯрд╛ рдПрдХрддреНрд░ рдХрд░рддрд╛ рд╣реИред"""
    def execute(self, search_query: str) -> str:
        tools = [google_search_tool] 
        prompt = f"Find the most up-to-date information and current trends for the topic: {search_query}"
        result_text = execute_gemini_call(prompt, tools)
        return result_text

class SummarizerAgent(BaseAgent):
    """рд░рд┐рд╕рд░реНрдЪ рдХреЗ рдкрд░рд┐рдгрд╛рдореЛрдВ рдХреЛ рдПрдХ рд╡рд┐рд╕реНрддреГрдд рдФрд░ рд╕рдВрд░рдЪрд┐рдд рд░рд┐рдкреЛрд░реНрдЯ рдореЗрдВ рд╕рдВрдХрд▓рд┐рдд рдХрд░рддрд╛ рд╣реИред"""
    def execute(self, all_data: Dict[str, str], topic: str) -> str:
        prompt = f"Compile a detailed, structured report on '{topic}' using the following research steps and findings:\n{json.dumps(all_data, indent=2)}. Only generate the summary, no greetings."
        final_summary = execute_gemini_call(prompt, available_tools=[]) 
        
        report_text = f"Detailed Report on {topic}\n"
        report_text += "========================================\n"
        for step, result in all_data.items():
             report_text += f"\n--- Section: {step} ---\n{result}\n"
        report_text += f"\nFinal Summary by LLM: {final_summary}"
        return report_text

class SchedulerAgent(BaseAgent):
    """рдХреИрд▓реЗрдВрдбрд░ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдЗрд╡реЗрдВрдЯреНрд╕ рдХреЛ рд╕рдВрднрд╛рд▓рддрд╛ рд╣реИред"""
    def execute(self, event_title: str, event_time: str, attendees: List[str]) -> str:
        tools = [calendar_tool]
        prompt = f"Please schedule a meeting titled '{event_title}' with attendees {', '.join(attendees)} at {event_time}."
        return execute_gemini_call(prompt, tools)

class CommunicatorAgent(BaseAgent):
    """рдореИрд╕реЗрдЬрд┐рдВрдЧ рдЯреВрд▓ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рд╕рдВрд╡рд╛рдж рднреЗрдЬрддрд╛ рд╣реИред"""
    def execute(self, recipient: str, subject: str, content: str) -> str:
        tools = [messaging_tool]
        prompt = f"Send an urgent email to {recipient} with the subject '{subject}'. The body content is: {content[:100]}..."
        return execute_gemini_call(prompt, tools)

# ==============================================================================
# IV. рдорд╛рд╕реНрдЯрд░ рдСрд░реНрдХреЗрд╕реНрдЯреНрд░реЗрдЯрд░ рдПрдЬреЗрдВрдЯ рдФрд░ V. рдореБрдЦреНрдп рдирд┐рд╖реНрдкрд╛рджрди (Execution)
# ==============================================================================

# OrchestratorAgent рдФрд░ main() рдлрд╝рдВрдХреНрд╢рди рдкрд┐рдЫрд▓реЗ рд╕рдВрд╕реНрдХрд░рдг рдХреА рддрд░рд╣ рд╣реА рд░рд╣рддреЗ рд╣реИрдВред
class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("ORCHESTRATOR_AGENT")
        self.planner = PlannerAgent("PLANNER_AGENT")
        self.researcher = ResearcherAgent("RESEARCHER_AGENT")
        self.summarizer = SummarizerAgent("SUMMARIZER_AGENT")
        self.scheduler = SchedulerAgent("SCHEDULER_AGENT")
        self.communicator = CommunicatorAgent("COMMUNICATOR_AGENT")

    def route_request(self, user_request: str) -> str:
        if "research" in user_request.lower() or "report" in user_request.lower():
            return "RESEARCH_DOMAIN"
        elif "schedule" in user_request.lower() or "meeting" in user_request.lower():
            return "SCHEDULING_DOMAIN"
        else:
            return "UNKNOWN"

    def process_request(self, user_request: str) -> str:
        print("\n" + "="*70)
        print(f"[{self.name}] New Request: {user_request}")
        print("="*70)

        domain = self.route_request(user_request)
        
        if domain == "RESEARCH_DOMAIN":
            topic = user_request.split("on")[-1].strip() if "on" in user_request.lower() else "The Future of AI"
            
            research_plan = self.planner.execute(topic)
            all_results = {}
            
            for step in research_plan:
                search_result = self.researcher.execute(step)
                all_results[step] = search_result
                
            final_report = self.summarizer.execute(all_results, topic)
            
            communication_status = self.communicator.execute("Stakeholder@corp.com", f"Final Report: {topic}", final_report)
            return f"\nFINAL RESULT: Report compiled and communication initiated. Status: {communication_status}"

        elif domain == "SCHEDULING_DOMAIN":
            event_title = "Capstone Final Review"
            event_time = "2025-11-20 2:00 PM"
            attendees = ["User", "Mentor"]
            
            schedule_status = self.scheduler.execute(event_title, event_time, attendees)
            
            email_body = f"Hi All, Invitation confirmed: {event_title} scheduled for {event_time}. Please review the attached report."
            message_status = self.communicator.execute("All@team.com", f"Invitation: {event_title}", email_body)
            
            return f"\nFINAL RESULT: Scheduling Flow Completed. Status: {schedule_status}. Message Status: {message_status}"

        else:
            return f"[{self.name}] Could not determine the domain for the request. Only Research and Scheduling supported."

def main():
    print("--- Ultimate Personal Assistant System (Ready for Real Action) ---")
    assistant = OrchestratorAgent()
    
    research_prompt = "Prepare a detailed report on the latest AI trends in 2025 on Generative Agents and RAG."
    assistant.process_request(research_prompt)

    schedule_prompt = "Can you schedule a team meeting for tomorrow afternoon to review the final project submission?"
    assistant.process_request(schedule_prompt)
    
    print("\n--- Project Framework is Complete. Ready for API Key Integration ---")

if __name__ == "__main__":
    main()


# ==============================================================================
# I. рдЬреЗрдорд┐рдиреА рдХреНрд▓рд╛рдЗрдВрдЯ рдФрд░ рдЯреВрд▓ рдбреЗрдлрд┐рдирд┐рд╢рди (Gemini Client and Tool Definitions)
# ==============================================================================

# рдкреБрд░рд╛рдиреЗ рд╕рд┐рдореБрд▓реЗрд╢рди рдХреЛрдб рдХреЛ рд╣рдЯрд╛ рджреЗрдВ рдпрд╛ рдЯрд┐рдкреНрдкрдгреА (comment) рдХрд░ рджреЗрдВ
# class SimulatedGeminiClient:
#    ...

# рдЕрдм рд╡рд╛рд╕реНрддрд╡рд┐рдХ рдЬреЗрдорд┐рдиреА рдХреНрд▓рд╛рдЗрдВрдЯ рдХреЛ рдЖрдпрд╛рдд (import) рдФрд░ рдЗрдирд┐рд╢рд┐рдпрд▓рд╛рдЗрдЬрд╝ рдХрд░реЗрдВред
# NOTE: Kaggle Notebook рдореЗрдВ, рдпрд╣ рдХреНрд▓рд╛рдЗрдВрдЯ рдЖрдкрдХреА API Key рдХреЛ secrets рд╕реЗ рд╕реНрд╡рддрдГ рдЙрдард╛ рд▓реЗрдЧрд╛ред
try:
    from google import genai 
    # рд╡рд╛рд╕реНрддрд╡рд┐рдХ рдХреНрд▓рд╛рдЗрдВрдЯ:
    client = genai.Client() 
    print("рдЬреЗрдорд┐рдиреА рдХреНрд▓рд╛рдЗрдВрдЯ рд╕рдлрд▓рддрд╛рдкреВрд░реНрд╡рдХ рдЗрдирд┐рд╢рд┐рдпрд▓рд╛рдЗрдЬрд╝ рд╣реЛ рдЧрдпрд╛ рд╣реИред")
except ImportError:
    print("ERROR: 'google-genai' SDK рдирд╣реАрдВ рдорд┐рд▓рд╛ред рдХреГрдкрдпрд╛ рдЗрд╕реЗ рдЗрдВрд╕реНрдЯреЙрд▓ рдХрд░реЗрдВред")
    
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'



# Grounding рдХреЙрдиреНрдлрд╝рд┐рдЧрд░реЗрд╢рди
if any(f.__name__ == 'google_search_tool' for f in available_tools):
    grounding_tool = [{"google_search": {}}]
else:
    grounding_tool = []

# рдЕрдм generate_content рдХреЙрд▓ рдХреЛ рдмрджрд▓реЗрдВ:
try:
    response = client.generate_content(
        model=MODEL_NAME,
        contents=contents,
        tools=tool_schemas, # Pydantic рд╕реНрдХреАрдорд╛
        # рдпрд╣рд╛рдБ Grounding Tool рдЬреЛрдбрд╝реЗрдВ:
        config={"tools": grounding_tool}
    )
# ... рдмрд╛рдХреА рдХреЛрдб



# (OrchestratorAgent.process_request рдХреЗ рдЕрдВрджрд░, RESEARCH_DOMAIN рдореЗрдВ рдЕрдВрддрд┐рдо рд░рд┐рдЯрд░реНрди рд╕реЗ рдкрд╣рд▓реЗ)

# ... research flow complete ...
final_report = self.summarizer.execute(all_results, topic)

# рдЕрдВрддрд┐рдо рд░рд┐рдкреЛрд░реНрдЯ рдХреЛ рдПрдХ Markdown рдмреНрд▓реЙрдХ рдореЗрдВ рдЖрдЙрдЯрдкреБрдЯ рдХрд░реЗрдВ
print("\n" + "#"*70)
print(f"## {topic} - Final Structured Report")
print("#"*70)

print(final_report) 

# ... communication status ...



# -*- coding: utf-8 -*-
"""
Kaggle Agents Intensive - Capstone Project: The Ultimate Personal Assistant (v2)

рдореЗрдореЛрд░реА (рдЗрддрд┐рд╣рд╛рд╕) рдФрд░ рд╕рдВрд░рдЪрд┐рдд рдЖрдЙрдЯрдкреБрдЯ рдХреЛ рдмрдврд╝рд╛рдиреЗ рдХреЗ рд▓рд┐рдП рдЕрдкрдбреЗрдЯ рдХрд┐рдпрд╛ рдЧрдпрд╛ рдХреЛрдбред
рдпрд╣ рдЯреВ-рд╕реНрдЯреЗрдк рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рдХреЛ рдЬрд╝реНрдпрд╛рджрд╛ рд╡рд┐рд╢реНрд╡рд╕рдиреАрдп рдмрдирд╛рддрд╛ рд╣реИред
"""
import time
import json
from typing import List, Dict, Any, Callable
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# NOTE: рдХреГрдкрдпрд╛ рдЕрдкрдиреА Kaggle Notebook рдореЗрдВ SimulatedGeminiClient рдХреЛ genai.Client() рд╕реЗ рдмрджрд▓реЗрдВ
# ------------------------------------------------------------------------------

# ** рдЬреЗрдорд┐рдиреА рдХреНрд▓рд╛рдЗрдВрдЯ рдкреНрд▓реЗрд╕рд╣реЛрд▓реНрдбрд░ (Gemini Client Placeholder) **
# [рдпрд╣рд╛рдБ рдЖрдкрдХреА рд╡рд╛рд╕реНрддрд╡рд┐рдХ genai.Client() рдЗрдВрд╕реНрдЯрд╛рдВрд╕ рд╣реЛрдЧреА]
class SimulatedGeminiClient:
    """рд╡рд╛рд╕реНрддрд╡рд┐рдХ Gemini SDK рдХреНрд▓рд╛рдЗрдВрдЯ рдХрд╛ рдЕрдиреБрдХрд░рдг рдХрд░рддрд╛ рд╣реИред"""
    # Simplified generate_content for demonstration
    def generate_content(self, model: str, contents: List[Dict], tools: List[Any]):
        print(f"\n[API CALL] Sending request to Gemini model: {model}...")
        time.sleep(0.5) 
        
        last_part = contents[-1]['parts'][0]['text'] if contents[-1]['role'] == 'user' else "tool_response"
        
        # Simulating LLM's decision to call a tool (Step 1)
        if "trends" in last_part or "latest information" in last_part:
            return {"function_calls": [{"name": "google_search_tool", "args": {"query": "latest AI trends in Generative Agents and RAG"}}]}
        
        # Simulating LLM's final text response after seeing tool output (Step 2)
        elif "tool_response" in last_part:
            return {"text": "Based on the search results showing massive growth in Multi-Agent Systems, the final report focuses on: 1) Orchestration Frameworks, and 2) Advanced RAG techniques."}
        
        # Simulating other general responses or scheduling call
        else:
            return {"text": "Acknowledged: The agent has finished its thought process and compiled the final output."}

client = SimulatedGeminiClient()
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'


# ------------------------------------------------------------------------------
# I. рдЯреВрд▓ рдбреЗрдлрд┐рдирд┐рд╢рди рдФрд░ рд╕реНрдХреАрдорд╛ (Tool Definitions and Schemas)
# ------------------------------------------------------------------------------

def google_search_tool(query: str) -> str:
    """рд╡реЗрдм рдкрд░ рдЬрд╛рдирдХрд╛рд░реА рдЦреЛрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред (Google Search API рдХрд╛ рдЕрдиреБрдХрд░рдг)"""
    print(f"\n[TOOL CALLED: GoogleSearchTool] Searching for: '{query}'...")
    time.sleep(1)
    if "trends" in query.lower():
        # рд╡рд╛рд╕реНрддрд╡рд┐рдХ рдЯреВрд▓ рдЖрдЙрдЯрдкреБрдЯ
        return "Search Results: AI trends show massive growth in Generative Agents and RAG applications in 2025. Key Focus Areas: Multi-Agent Orchestration."
    return f"Search Results: Found 3 relevant articles about '{query}'. Summary: Key points are promising."

def calendar_tool(title: str, time: str, attendees: List[str]) -> str:
    """рдХреИрд▓реЗрдВрдбрд░ рдореЗрдВ рдЗрд╡реЗрдВрдЯ рдмрдирд╛рдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: CalendarTool] Scheduling: {title} on {time} with {len(attendees)} people.")
    time.sleep(0.5)
    return f"Calendar: Event '{title}' successfully scheduled for {time}."

def messaging_tool(recipient: str, subject: str, body: str) -> str:
    """рдХрд┐рд╕реА рдХреЛ рдИрдореЗрд▓ рдпрд╛ рдореИрд╕реЗрдЬ рднреЗрдЬрдиреЗ рдХрд╛ рдЯреВрд▓ред"""
    print(f"\n[TOOL CALLED: MessagingTool] Sending message to {recipient}. Subject: {subject}")
    time.sleep(0.5)
    return f"Messaging: Email to {recipient} with subject '{subject}' sent successfully."

TOOL_MAP: Dict[str, Callable] = {
    "google_search_tool": google_search_tool,
    "calendar_tool": calendar_tool,
    "messaging_tool": messaging_tool
}

class GoogleSearchTool(BaseModel):
    query: str = Field(description="The specific search query to look up on the web.")

class CalendarTool(BaseModel):
    title: str = Field(description="The title of the event/meeting.")
    time: str = Field(description="The date and time of the event.")
    attendees: List[str] = Field(description="A list of attendees for the event.")
    
class MessagingTool(BaseModel):
    recipient: str = Field(description="The email address or name of the recipient.")
    subject: str = Field(description="The subject line of the message.")
    body: str = Field(description="The full content of the message.")

ALL_TOOL_SCHEMAS: List[BaseModel] = [GoogleSearchTool, CalendarTool, MessagingTool]

# ------------------------------------------------------------------------------
# II. рдХреЛрд░ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рд▓реЙрдЬрд┐рдХ (Core Function Calling Logic with History)
# ------------------------------------------------------------------------------

def execute_gemini_call(prompt: str, available_tools: List[Callable], history: List[Dict] = None) -> str:
    """
    рдЬреЗрдорд┐рдиреА рдореЙрдбрд▓ рдХреЛ рдХреЙрд▓ рдХрд░рддрд╛ рд╣реИ рдФрд░ рдлрдВрдХреНрд╢рди рдХреЙрд▓рд┐рдВрдЧ рдХреЛ рд╕рдВрднрд╛рд▓рддрд╛ рд╣реИ (рдЯреВ-рд╕реНрдЯреЗрдк рд▓реВрдк)ред
    рдореЗрдореЛрд░реА (history) рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдЯреВрд▓ рдХреЗ рдЖрдЙрдЯрдкреБрдЯ рдХреЛ LLM рдХреЗ рдкрд╛рд╕ рд╡рд╛рдкрд╕ рднреЗрдЬрддрд╛ рд╣реИред
    """
    
    tool_schemas = [s for s in ALL_TOOL_SCHEMAS if s.__name__.lower() in [f.__name__ for f in available_tools]]

    # рдЗрддрд┐рд╣рд╛рд╕ (History) рд╕реЗрдЯрдЕрдк
    contents = history if history is not None else []
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    try:
        # Step 1: LLM рдХреЛ рдкреНрд░реЙрдореНрдкреНрдЯ рднреЗрдЬреЗрдВ
        response = client.generate_content(
            model=MODEL_NAME,
            contents=contents,
            tools=tool_schemas 
        )
    except Exception as e:
        return f"ERROR: LLM API Call failed: {e}"

    # Step 2: рдЯреВрд▓ рдХреЙрд▓ рдХреЗ рд▓рд┐рдП рд░рд┐рд╕реНрдкрд╛рдВрд╕ рдХреА рдЬрд╛рдБрдЪ рдХрд░реЗрдВ (Memory in Action)
    if response.get("function_calls"):
        print("   [LLM DECISION] Function Call requested. Executing Tool...")
        
        function_call = response['function_calls'][0]
        function_name = function_call['name']
        function_args = dict(function_call['args'])
        
        if function_name in TOOL_MAP:
            tool_function = TOOL_MAP[function_name]
            tool_output = tool_function(**function_args)
            
            # Step 3: рдЯреВрд▓ рдЖрдЙрдЯрдкреБрдЯ рдХреЛ рд╡рд╛рдкрд╕ LLM рдХреЛ рднреЗрдЬреЗрдВ (рдореЗрдореЛрд░реА рдЕрдкрдбреЗрдЯ)
            print("   [TOOL OUTPUT] Tool Output sent back to LLM for final answer.")
            
            # History update: LLM рдХрд╛ рдЕрдиреБрд░реЛрдз рдФрд░ рдЯреВрд▓ рдХрд╛ рдкрд░рд┐рдгрд╛рдо рдЬреЛрдбрд╝реЗрдВ
            contents.append({"role": "model", "parts": [response]})
            # Tool response content added to history for LLM to see
            contents.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "content": tool_output, "text": "tool_response"}}]}) 
            
            # Step 4: LLM рдХреЛ рдлрд┐рд░ рд╕реЗ рдХреЙрд▓ рдХрд░реЗрдВ
            final_response = client.generate_content(
                model=MODEL_NAME,
                contents=contents,
                tools=tool_schemas
            )
            
            return final_response.get("text", "Final response generation failed after tool execution.")
            
    # рдЕрдЧрд░ рдХреЛрдИ рдлрдВрдХреНрд╢рди рдХреЙрд▓ рдирд╣реАрдВ рд╣реБрдИ, рддреЛ рд╕реАрдзрд╛ рдЯреЗрдХреНрд╕реНрдЯ рд░рд┐рд╕реНрдкрд╛рдВрд╕ рд▓реМрдЯрд╛рдПрдБ
    return response.get("text", "LLM responded with plain text.")


# ------------------------------------------------------------------------------
# III. рд╡рд┐рд╢реЗрд╖рдЬреНрдЮ рдПрдЬреЗрдВрдЯ (Specialist Agents)
# ------------------------------------------------------------------------------

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        
class PlannerAgent(BaseAgent):
    """рдХрд╛рд░реНрдпреЛрдВ рдХреЛ рддреЛрдбрд╝рддрд╛ рд╣реИред рдореЗрдореЛрд░реА рдХреЗ рд▓рд┐рдП, рдпрд╣ рдкрд┐рдЫрд▓реА рдмрд╛рддрдЪреАрдд рдХреЛ рдпрд╛рдж рд░рдЦрдиреЗ рдХреЗ рд▓рд┐рдП рддреИрдпрд╛рд░ рд╣реИред"""
    def execute(self, research_topic: str) -> list:
        print(f"[{self.name}] Planning research steps for: {research_topic}")
        # рдпрд╣рд╛рдВ рд╣рдо LLM рдХреЛ рдХреЙрд▓ рдХрд░рдХреЗ рдПрдХ рдбрд╛рдпрдиреЗрдорд┐рдХ рдкреНрд▓рд╛рди рдЬрдирд░реЗрдЯ рдХрд░ рд╕рдХрддреЗ рд╣реИрдВ, 
        # рд▓реЗрдХрд┐рди рд╕рд░рд▓рддрд╛ рдХреЗ рд▓рд┐рдП, рд╣рдо рдПрдХ рд╕рдВрд░рдЪрд┐рдд рдкреНрд▓рд╛рди рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░ рд░рд╣реЗ рд╣реИрдВред
        steps = [
            f"Define the core concepts of '{research_topic}'",
            f"Search for the latest industry trends and developments for '{research_topic}' using google_search_tool",
            f"Identify real-world case studies or success stories related to '{research_topic}'",
            "Consolidate all findings into a structured report outline"
        ]
        return steps

class ResearcherAgent(BaseAgent):
    """Google Search Tool рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдбреЗрдЯрд╛ рдПрдХрддреНрд░ рдХрд░рддрд╛ рд╣реИред"""
    def execute(self, search_query: str) -> str:
        tools = [google_search_tool] 
        prompt = f"Find the most up-to-date information and current trends for the topic: {search_query}"
        result_text = execute_gemini_call(prompt, tools)
        return result_text

class SummarizerAgent(BaseAgent):
    """рд░рд┐рд╕рд░реНрдЪ рдХреЗ рдкрд░рд┐рдгрд╛рдореЛрдВ рдХреЛ рдПрдХ рд╡рд┐рд╕реНрддреГрдд рдФрд░ рд╕рдВрд░рдЪрд┐рдд рд░рд┐рдкреЛрд░реНрдЯ рдореЗрдВ рд╕рдВрдХрд▓рд┐рдд рдХрд░рддрд╛ рд╣реИред"""
    def execute(self, all_data: Dict[str, str], topic: str) -> str:
        # рдЕрдВрддрд┐рдо рд░рд┐рдкреЛрд░реНрдЯ рдХреЛ рд╕реБрдВрджрд░ рдмрдирд╛рдиреЗ рдХреЗ рд▓рд┐рдП рдпрд╣рд╛рдВ Markdown рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВ
        report_output = f"


    for step, result in all_data.items():
         report_output += f"## ЁЯФО Step: {step}\n"
         report_output += f"**Findings:** {result}\n\n"
         
    # LLM рд╕реЗ рдЕрдВрддрд┐рдо рд╕рд╛рд░рд╛рдВрд╢ рдЬрдирд░реЗрдЯ рдХрд░рд╡рд╛рдПрдБ
    prompt = f"Compile a concise, professional summary and conclusion for the report on '{topic}' based on the findings:\n{json.dumps(all_data, indent=2)}. Only generate the summary, no greetings."
    final_summary = execute_gemini_call(prompt, available_tools=[])
    
    report_output += "## тЬЕ Final Conclusion\n\n"
    report_output += final_summary
    report_output += "\n


https://www.kaggle.com/code/raviganjir/ultimate-assistant-py?scriptVersionId=279628693&cellId=13


# Example Pydantic Schema for Planner Output
from pydantic import BaseModel, Field
from typing import List

class SubTask(BaseModel):
    task_id: int = Field(description="Unique ID for the sub-task.")
    domain: str = Field(description="The primary domain: 'Research', 'Scheduling', or 'Communication'.")
    query: str = Field(description="The specific action or search query required for this task.")

class ExecutionPlan(BaseModel):
    """The master plan containing a list of structured sub-tasks."""
    sub_tasks: List[SubTask]
    


рдЪрд░рдг рдЬрд╝рд┐рдореНрдореЗрджрд╛рд░ рдПрдЬреЗрдВрдЯ рдЯреВрд▓ (рдпрджрд┐ рдЙрдкрдпреЛрдЧ рдХрд┐рдпрд╛ рдЧрдпрд╛) рдХрд╛рд░реНрдп/рдЗрдирдкреБрдЯ рдЖрдЙрдЯрдкреБрдЯ (рд╕рдВрдХреНрд╖реЗрдк рдореЗрдВ)
1. рдкреНрд▓рд╛рдирд┐рдВрдЧ Orchestrator Agent - рдкреНрд░реЙрдореНрдкреНрдЯ рдХреЛ рдбреАрдХрдВрдкреЛрдЬрд╝ рдХрд░рдирд╛ред 3 SubTask рдСрдмреНрдЬреЗрдХреНрдЯреНрд╕ рдХреЗ рд╕рд╛рде ExecutionPlan JSON: 1. рд░рд┐рд╕рд░реНрдЪ, 2. рд╢реЗрдбреНрдпреВрд▓рд┐рдВрдЧ, 3. рдХрдореНрдпреБрдирд┐рдХреЗрд╢рдиред
2. рд░рд┐рд╕рд░реНрдЪ Researcher Agent Google Search_tool "latest Google (Alphabet Inc.) quarterly earnings report summary" рдкрд░рд┐рдгрд╛рдо: рдирд╡реАрдирддрдо рд░рд┐рдкреЛрд░реНрдЯ рдХреЗ рдкреНрд░рдореБрдЦ рдореЗрдЯреНрд░рд┐рдХреНрд╕ рдФрд░ рдореБрдЦреНрдп рдирд┐рд╖реНрдХрд░реНрд╖реЛрдВ рд╡рд╛рд▓рд╛ рдПрдХ рд╕рдВрд░рдЪрд┐рдд рд╕рд╛рд░рд╛рдВрд╢ред
3. рд╢реЗрдбреНрдпреВрд▓рд┐рдВрдЧ Scheduler Agent calendar_tool "Schedule meeting with Jane next Wednesday at 11:00 AM for 'Q3 Performance Review'." рдкрд░рд┐рдгрд╛рдо: рдореАрдЯрд┐рдВрдЧ рдХрд╛ рд╢реЗрдбреНрдпреВрд▓рд┐рдВрдЧ рд╕рдлрд▓; calendar_id рдФрд░ рдореАрдЯрд┐рдВрдЧ URL рд╡рд╛рдкрд╕ рд▓реМрдЯрд╛рдпрд╛ рдЧрдпрд╛ред
4. рдЕрдВрддрд┐рдо рдбреНрд░рд╛рдлреНрдЯрд┐рдВрдЧ Communicator Agent - рд░рд┐рд╕рд░реНрдЪ рд╕рд╛рд░рд╛рдВрд╢ рдФрд░ рдореАрдЯрд┐рдВрдЧ рд╡рд┐рд╡рд░рдг рдХреЛ рдПрдХ рд╕рд╛рде рд▓рд╛рдирд╛ред рд╕рднреА рд╡рд┐рд╡рд░рдгреЛрдВ рдХреЛ рд╕рдорд╛рд╣рд┐рдд рдХрд░рдиреЗ рд╡рд╛рд▓рд╛ рдЕрдВрддрд┐рдо рдбреНрд░рд╛рдлреНрдЯ рдИрдореЗрд▓ред


<iframe src="https://www.kaggle.com/embed/raviganjir/ultimate-assistant-py?cellIds=13&kernelSessionId=279628693" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="ultimate_assistant.py"></iframe>


рдЕрдВрддрд┐рдо рдЖрдЙрдЯрдкреБрдЯ:
FINAL RESULT: Scheduling Flow Completed. Status: Calendar: Event 'Capstone Final Review' successfully scheduled for 2025-11-20 2:00 PM. Message Status: Messaging: Email to All@team.com with subject 'Invitation: Capstone Final Review' sent successfully.
If I had more time, this is what I'd do
рдЕрдЧрд░ рдореБрдЭреЗ рдФрд░ рд╕рдордп рдорд┐рд▓рддрд╛, рддреЛ рдореИрдВ рд╕рд┐рд╕реНрдЯрдо рдХреЛ рдФрд░ рднреА рдЕрдзрд┐рдХ рдордЬрдмреВрдд рдФрд░ рдЙрдкрдпреЛрдЧрдХрд░реНрддрд╛ рдХреЗ рдЕрдиреБрдХреВрд▓ рдмрдирд╛рдиреЗ рдХреЗ рд▓рд┐рдП рдирд┐рдореНрдирд▓рд┐рдЦрд┐рдд рд╕реБрдзрд╛рд░реЛрдВ рдкрд░ рдзреНрдпрд╛рди рдХреЗрдВрджреНрд░рд┐рдд рдХрд░рддрд╛:
рдбрд╛рдпрдиреЗрдорд┐рдХ рдкреНрд▓рд╛рди рдЬрдирд░реЗрд╢рди: PlannerAgent рдореЗрдВ рд╣рд╛рд░реНрдбрдХреЛрдбреЗрдб рд╕реНрдЯреЗрдкреНрд╕ (hardcoded steps) рдХреЗ рдмрдЬрд╛рдп, рдореИрдВ LLM рдХреЛ рдПрдХ Pydantic Schema рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рдкреНрд░реЙрдореНрдкреНрдЯ рдХреЗ рдЖрдзрд╛рд░ рдкрд░ рдкреВрд░реА рддрд░рд╣ рд╕реЗ рдбрд╛рдпрдиреЗрдорд┐рдХ рдкреНрд▓рд╛рди рдЬрдирд░реЗрдЯ рдХрд░рдиреЗ рджреЗрддрд╛ред рдЗрд╕рд╕реЗ рд╕рд┐рд╕реНрдЯрдо рдХреА рдмрд╣реБрдореБрдЦреА рдкреНрд░рддрд┐рднрд╛ (versatility) рдмрдврд╝рддреАред
рдкрд░реНрд╕рд┐рд╕реНрдЯреЗрдВрдЯ рдореЗрдореЛрд░реА (Persistence): рдореИрдВ рдмрд╛рддрдЪреАрдд рдХреЗ рдЗрддрд┐рд╣рд╛рд╕ рдХреЛ рд╕реЗрд╡ рдХрд░рдиреЗ рдХреЗ рд▓рд┐рдП рдПрдХ рд╡реЗрдХреНрдЯрд░ рдбреЗрдЯрд╛рдмреЗрд╕ рдпрд╛ Firestore рдХреЛ рдЗрдВрдЯреАрдЧреНрд░реЗрдЯ рдХрд░рддрд╛ред рдпрд╣ рдПрдЬреЗрдВрдЯ рдХреЛ рдкрд┐рдЫрд▓реЗ рд╕рддреНрд░реЛрдВ рд╕реЗ рд╕реАрдЦреЗ рдЧрдП рд╡реНрдпрдХреНрддрд┐рдЧрдд рдкреНрд░рд╛рдердорд┐рдХрддрд╛рдУрдВ (preferences) рдФрд░ рдЬреНрдЮрд╛рди рдХреЛ рдпрд╛рдж рд░рдЦрдиреЗ рдХреА рдЕрдиреБрдорддрд┐ рджреЗрддрд╛ред
рдпреВрдЬрд╝рд░ рдЗрдВрдЯрд░рдлрд╝реЗрд╕ (UI) рд▓реЗрдпрд░: рдореИрдВ рдПрдХ рд╕рд╛рдзрд╛рд░рдг рд╡реЗрдм рдЗрдВрдЯрд░рдлрд╝реЗрд╕ (рдЬреИрд╕реЗ React/Streamlit) рдмрдирд╛рддрд╛ рддрд╛рдХрд┐ рдЙрдкрдпреЛрдЧрдХрд░реНрддрд╛ рдЗрдВрдЯрд░реЗрдХреНрдЯрд┐рд╡ рд░реВрдк рд╕реЗ рдкреНрд░реЙрдореНрдкреНрдЯ рдЗрдирдкреБрдЯ рдХрд░ рд╕рдХреЗ рдФрд░ рдХреИрд▓реЗрдВрдбрд░ рдпрд╛ рдИрдореЗрд▓ рдЯреВрд▓ рдХреЗ рд╕рд╛рде рд╕реАрдзреЗ рдЗрдВрдЯрд░реИрдХреНрдЯ рдХрд░ рд╕рдХреЗ, рдЬрд┐рд╕рд╕реЗ рдпрд╣ рдПрдХ рдкреВрд░реНрдг-рд╕реНрдЯреИрдХ рд╕рдорд╛рдзрд╛рди рдмрди рдЬрд╛рддрд╛


Kaggle Heading рд╣рдорд╛рд░реА рд╕рд╛рдордЧреНрд░реА
Problem Statement Section 1 рдХрд╛ рдЯреЗрдХреНрд╕реНрдЯ
Why agents? Section 2 рдФрд░ Section 3 рдХрд╛ рд╕рд╛рд░рд╛рдВрд╢
What you created Section 2 (рдЖрд░реНрдХрд┐рдЯреЗрдХреНрдЪрд░ рдФрд░ рднреВрдорд┐рдХрд╛рдПрдБ) рдХрд╛ рдЯреЗрдХреНрд╕реНрдЯ
The Build Section 3 (рддрдХрдиреАрдХреА рд╡рд┐рд╡рд░рдг: Pydantic рдФрд░ рдЯреВ-рд╕реНрдЯреЗрдк рд▓реЙрдЬрд┐рдХ) рдХрд╛ рдЯреЗрдХреНрд╕реНрдЯ
Demo Section 4 (рдХрд╛рд░реНрдпрд╢реАрд▓ рдЙрджрд╛рд╣рд░рдг) рдХрд╛ рдЯреЗрдХреНрд╕реНрдЯ
If I had more time, this is what I'd do Section 5 (рдирд┐рд╖реНрдХрд░реНрд╖ рдФрд░ рд╕реАрдЦ) рдХрд╛ рд╡рд╣ рд╣рд┐рд╕реНрд╕рд╛


ЁЯУК рдбреЗрдЯрд╛ рд╡рд┐рдЬрд╝реБрдЕрд▓рд╛рдЗрдЬрд╝рд░
D3.js рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ рд╕рд░реНрд╡реЗ рдбреЗрдЯрд╛ рдХрд╛ рдЗрдВрдЯрд░реИрдХреНрдЯрд┐рд╡ рд╡рд┐рд╢реНрд▓реЗрд╖рдгред


рд╢рд┐рдХреНрд╖рд╛ рд╕реНрддрд░ (Education Level)
рдмрд╛рд░ рдЪрд╛рд░реНрдЯ рджрд┐рдЦрд╛рдПрдБ
рдкрд╛рдИ рдЪрд╛рд░реНрдЯ рджрд┐рдЦрд╛рдПрдБ
High School
Bachelor's
Master's
PhD
0
100
200
300
400
500
600
700
рд╢рд┐рдХреНрд╖рд╛ рд╕реНрддрд░ рдХреЗ рдЕрдиреБрд╕рд╛рд░ рд╡рд┐рддрд░рдг
рдпрд╣ рдмрд╛рд░ рдЪрд╛рд░реНрдЯ рд╡рд┐рднрд┐рдиреНрди рд╢рд┐рдХреНрд╖рд╛ рд╕реНрддрд░реЛрдВ рд╡рд╛рд▓реЗ рдЙрддреНрддрд░рджрд╛рддрд╛рдУрдВ рдХреА рд╕рдВрдЦреНрдпрд╛ рджрд┐рдЦрд╛рддрд╛ рд╣реИред
















