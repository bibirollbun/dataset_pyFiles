# =============================================================================
# PROJECT: MENTAL HEALTH AGENT (Smart Simulation Version)
# PLATFORM: Kaggle Notebook
# UPGRADE: Added pre-written content for Stress, Anxiety, and Tips
# =============================================================================

import os
import sys
import uuid
import logging
import time
from typing import Annotated, Sequence, TypedDict, List
import operator

# Install (Ignore the dependency conflict errors, they are fine for this demo)
!pip install -qU langgraph langchain-google-genai langchain_community duckduckgo-search textblob

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from textblob import TextBlob

# --- 1. THE SMART MOCK BRAIN (Simulating the AI) ---
# Since the API key is failing, this class acts as the "Brain"
class SmartMockLLM:
    def invoke(self, messages, **kwargs):
        # Get the last message from the conversation
        if isinstance(messages, list):
            # Determine if we are looking at the User input or the System Prompt
            # The last message is usually the User's input
            last_content = messages[-1].content.lower()
            # Sometimes the prompt is complex, so we look for key phrases in the whole context
            full_context = " ".join([m.content for m in messages]).lower()
        else:
            last_content = str(messages).lower()
            full_context = last_content

        # --- A. LOGIC FOR EVALUATOR AGENT ---
        if "rate the safety" in full_context:
            return AIMessage(content="9")

        # --- B. LOGIC FOR TRIAGE AGENT ---
        if "does this text imply self-harm" in full_context:
            if "kill" in last_content or "suicide" in last_content or "end it" in last_content:
                return AIMessage(content="YES")
            else:
                return AIMessage(content="NO")

        # --- C. LOGIC FOR COUNSELOR AGENT (The Responses you want) ---
        
        # 1. High Risk / Safety
        if "suicide" in last_content or "kill myself" in last_content:
            return AIMessage(content="I am very concerned about your safety. Please call 988 (Suicide & Crisis Lifeline) immediately. You are not alone, and there is help available right now.")

        # 2. Request for Tips / Stress Relief (THIS IS WHAT YOU ASKED FOR)
        elif "tip" in last_content or "stress" in last_content or "relax" in last_content:
            return AIMessage(content=(
                "Here are some grounded techniques to help relieve stress:\n"
                "1. **Box Breathing:** Inhale for 4 seconds, hold for 4, exhale for 4, hold for 4.\n"
                "2. **5-4-3-2-1 Technique:** Acknowledge 5 things you see, 4 you can touch, 3 you hear, 2 you can smell, and 1 you can taste.\n"
                "3. **Physical Movement:** A quick 10-minute walk can reset your cortisol levels.\n"
                "Which of these would you like to try?"
            ))

        # 3. Depression / Sadness
        elif "depress" in last_content or "sad" in last_content or "alone" in last_content:
            return AIMessage(content="I hear that you are going through a heavy time. Depression can feel isolating, but small steps help. Have you eaten or had water today? Sometimes taking care of basic physical needs is the first step.")

        # 4. Anxiety
        elif "anxious" in last_content or "panic" in last_content:
            return AIMessage(content="It sounds like anxiety is high right now. Let's focus on the present moment. Can you feel your feet on the floor? Take a slow deep breath with me.")

        # 5. Generic / Greeting
        else:
            return AIMessage(content="I am here to listen. Could you tell me a bit more about what is on your mind? (Try asking for 'tips' or telling me how you feel).")

# Use the Mock Brain
llm = SmartMockLLM()

# --- 2. SETUP STATE & TOOLS ---

class MentalHealthState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    risk_level: str
    evaluation_score: int

# --- 3. AGENT NODES ---

def triage_node(state: MentalHealthState):
    last_msg = state['messages'][-1].content
    # Heuristic: TextBlob for sentiment
    blob = TextBlob(last_msg)
    
    # Ask MockLLM for opinion
    prompt = f"Does this text imply self-harm? {last_msg}"
    risk_check = llm.invoke([HumanMessage(content=prompt)]).content
    
    if "YES" in risk_check or blob.sentiment.polarity < -0.8:
        risk = "HIGH"
    elif blob.sentiment.polarity < -0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {"risk_level": risk}

def counselor_node(state: MentalHealthState):
    # The counselor logic
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

def evaluator_node(state: MentalHealthState):
    last_msg = state['messages'][-1].content
    prompt = f"Rate the safety of: {last_msg}"
    score = llm.invoke([HumanMessage(content=prompt)]).content
    return {"evaluation_score": int(score)}

# --- 4. GRAPH BUILD ---

workflow = StateGraph(MentalHealthState)
workflow.add_node("triage", triage_node)
workflow.add_node("counselor", counselor_node)
workflow.add_node("evaluator", evaluator_node)

workflow.add_edge(START, "triage")
workflow.add_edge("triage", "counselor") # Simplified for demo
workflow.add_edge("counselor", "evaluator")
workflow.add_edge("evaluator", END)

app = workflow.compile(checkpointer=MemorySaver())

# --- 5. RUNNER ---

def run_smart_simulation():
    print("\n" + "="*60)
    print("ðŸ§  MENTAL HEALTH AGENT (Smart Simulation)")
    print("============================================================")
    print("NOTE: API is offline. Using pre-programmed responses.")
    print("Try asking: 'give me stress tips' or 'i feel depressed'")
    print("============================================================\n")
    
    sid = str(uuid.uuid4())[:4]
    config = {"configurable": {"thread_id": sid}}
    
    while True:
        try:
            u_input = input(f"User: ")
            if u_input.lower() in ['quit', 'exit']: break
            
            inputs = {"messages": [HumanMessage(content=u_input)], "session_id": sid, "risk_level": "LOW"}
            
            # Run
            for event in app.stream(inputs, config=config):
                if 'counselor' in event:
                    print(f"\nðŸ¤– Agent: {event['counselor']['messages'][-1].content}\n")
                    
        except Exception as e:
            print(e)
            break

if __name__ == "__main__":
    run_smart_simulation()

