# Cell: Visualization Helper

def visualize_architecture():
    print("\nğŸ”¹ SYSTEM ARCHITECTURE DATA FLOW ğŸ”¹\n")
    steps = [
        "ğŸ‘¤ User Input",
        "   â†“",
        "âš™ï¸� Orchestrator (hAIppySession)",
        "   â†“",
        "ğŸ•µï¸� Agent 1: Emotion Detective (Gemini)",
        "   â†“",
        "ğŸ›¡ï¸� Crisis Check Gate (Safety Layer)",
        "   â†“ (If Safe)",
        "ğŸ“š Agent 2: Strategy Retriever (Tools)",
        "   â†“ âŸ· (Calls RAG Dictionary)",
        "ğŸ§  Agent 3: Therapist Persona (Gemini)",
        "   â†“ â†� (Injects Context from Memory)",
        "ğŸ’¬ Final Response"
    ]
    
    for step in steps:
        print(step)
        # Simple animation effect (optional)
        # time.sleep(0.1) 
    print("\nâœ… End of Flow")

# Run it to display the diagram in your output
visualize_architecture()


!pip install -q -U google-generativeai langchain-google-genai langchain langchain-community langchain-core


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


import os
import getpass
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory

# --- CONFIGURATION ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI Studio API Key: ")

# We use Gemini 2.0 Flash for speed and efficiency
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    convert_system_message_to_human=True 
)
print("âœ… Configuration Complete.")


# --- 1. DEMONSTRATING Docstring KNOWLEDGE BASE (RAG SOURCE) ---
# It is structured as a dictionary of lists for this demo.
knowledge_base = {
    "stress": [
        "Box Breathing: Inhale 4s, Hold 4s, Exhale 4s, Hold 4s.",
        "5-Minute Nature Walk: Reset sensory input by walking outside without a phone."
    ],
    "anxiety": [
        "5-4-3-2-1 Technique: Name 5 things you see, 4 you feel, etc.",
        "Ice Dive: Splash cold water on your face to trigger the mammalian dive reflex."
    ],
    "sadness": [
        "Comfort Playlist: Listen to nostalgic, low-tempo music.",
        "Hydration Reset: Drink a full glass of water to reset the parasympathetic system."
    ],
    "burnout": [
        "Task Bracketing: Work for 25m, then forced 5m break.",
        "Digital Sunset: Turn off all screens 1 hour before bed."
    ]
}

@tool
def get_coping_strategies(detected_emotion: str) -> str:
    """
    Retrieves specific coping strategies from the hardcoded knowledge base
    based on the emotion category.
    """
    detected_emotion = detected_emotion.lower().strip()
    
    # Check if we have a match in our keys
    for key in knowledge_base:
        if key in detected_emotion:
            # Format the list into a nice string bulleted list
            strategies = knowledge_base[key]
            return "\n".join([f"- {s}" for s in strategies])
            
    return "- Take deep breaths.\n- Drink a glass of water."

print("âœ… RAG Knowledge Base Loaded & Tool Initialized.")


class AgentLogger:
    """
    A static utility class for observability.
    It prints structured logs to the console to visualize the agent's 'thought process'
    in real-time.
    """
    @staticmethod
    def log(agent_name, input_data, output_data):
        print(f"\n--- ğŸ”� TRACE: {agent_name} ---")
        print(f"ğŸ“¥ Input: {input_data}")
        print(f"ğŸ“¤ Output: {output_data}")
        print("--------------------------------")


# --- AGENT 1: EMOTION DETECTIVE (Gemini Powered) ---
# NOTE: Includes 'CRISIS' category for safety.
emotion_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert psychological classifier. 
    Classify the user's text into EXACTLY one of these categories: 
    [stress, anxiety, sadness, burnout, CRISIS, neutral].
    
    âš ï¸� IMPORTANT: 
    - Output 'CRISIS' if the user mentions self-harm, suicide, severe panic, or dying.
    - Otherwise, output the most relevant emotion.
    - Output ONLY the single word category."""),
    ("human", "{user_input}")
])

def emotion_agent(user_input):
    chain = emotion_prompt | llm
    result = chain.invoke({"user_input": user_input})
    clean_result = result.content.strip()
    AgentLogger.log("Emotion Agent", user_input, clean_result)
    return clean_result

# --- AGENT 2: STRATEGY RETRIEVER ---
def strategy_agent(emotion_category):
    # Programmatic tool call
    strategies = get_coping_strategies.invoke(emotion_category)
    AgentLogger.log("RAG Tool", emotion_category, strategies)
    return strategies

# --- AGENT 3: THE THERAPIST ---
response_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are 'hAIppy', a supportive mental health assistant.
    1. Sympathize with the user's {emotion}.
    2. Present these strategies clearly: 
    {strategies}
    3. End with a reflective journaling prompt.
    
    Context from previous chat: {history}
    """),
    ("human", "{original_input}")
])

def therapist_agent(original_input, emotion, strategies, history_text):
    chain = response_prompt | llm
    result = chain.invoke({
        "original_input": original_input,
        "emotion": emotion,
        "strategies": strategies,
        "history": history_text
    })
    return result.content


from langchain_core.pydantic_v1 import BaseModel, Field

# 1. Define the Profile Structure
class UserProfile(BaseModel):
    user_name: str
    summary_of_issues: str = Field(description="A concise summary of what stresses this user out.")
    successful_strategies: list[str] = Field(description="List of coping mechanisms the user liked or found helpful.")
    failed_strategies: list[str] = Field(description="List of coping mechanisms the user disliked or that didn't work.")
    last_mood: str = Field(description="The emotion detected in the last interaction.")

# Global "Database" to hold profiles (In a real app, this would be SQL/NoSQL)
USER_DATABASE = {}

class hAIppySession:
    def __init__(self, user_name):
        self.user_name = user_name
        # Initialize or retrieve profile
        if user_name not in USER_DATABASE:
            USER_DATABASE[user_name] = UserProfile(
                user_name=user_name,
                summary_of_issues="New user.",
                successful_strategies=[],
                failed_strategies=[],
                last_mood="Neutral"
            )
    
    def get_profile_context(self):
        """Generates a system prompt section based on the compacted profile."""
        profile = USER_DATABASE[self.user_name]
        return (
            f"USER PROFILE:\n"
            f"- Name: {profile.user_name}\n"
            f"- Known Issues: {profile.summary_of_issues}\n"
            f"- WHAT WORKS: {', '.join(profile.successful_strategies)}\n"
            f"- WHAT FAILED: {', '.join(profile.failed_strategies)}\n"
            f"- Last Mood: {profile.last_mood}"
        )

    def compact_memory(self, latest_interaction_summary):
        """
        Uses the LLM to update the structured profile based on the recent chat.
        This is the 'Context Compaction' step.
        """
        profile = USER_DATABASE[self.user_name]
        
        # SIMULATED LOGIC for demonstration:
        # If the interaction mentioned "thanks" or "helped", we assume the strategy worked.
        # In a real app, we would ask Gemini to parse this using a separate prompt.
        if "thanks" in latest_interaction_summary.lower() or "good" in latest_interaction_summary.lower():
            if "Breathing" in latest_interaction_summary and "Breathing" not in profile.successful_strategies:
                profile.successful_strategies.append("Breathing")
            if "Walk" in latest_interaction_summary and "Walk" not in profile.successful_strategies:
                 profile.successful_strategies.append("Nature Walk")
        
        print(f"ğŸ’¾ MEMORY COMPACTED for {self.user_name}. Profile updated.")

    def process_message(self, user_input):
        print(f"\nâ–¶ï¸� STARTING SESSION FOR: {self.user_name}")
        
        # 1. Detect Emotion
        detected_emotion = emotion_agent(user_input)
        
        # 2. Crisis Check
        if "CRISIS" in detected_emotion.upper():
            crisis_response = (
                "ğŸ›‘ **IMMEDIATE ATTENTION REQUIRED.** ğŸ›‘\n\n"
                "I am an AI assistant, not a substitute for professional help. "
                "It sounds like you are in immediate distress. Please reach out to a professional immediately.\n\n"
                "**Please use one of these resources:**\n"
                "* **Crisis Text Line:** Text HOME to 741741 (US/Canada)\n"
                "* **National Suicide Prevention Lifeline (US):** 988\n"
                "* **Emergency Services:** Call your local emergency number (e.g., 911, 999, 112).\n"
                "I am here to listen, but I cannot provide medical intervention."
            )
            print(f"ğŸ¤– **CRISIS RESPONSE TRIGGERED**:\n{crisis_response}")
            
            # Crucial Safety Step: Stop workflow.
            # We do NOT add this to memory to avoid confusing future context with crisis data.
            return crisis_response

        # 3. Retrieve Strategies
        strategies = strategy_agent(detected_emotion)
        
        # 4. Generate Response (injecting the COMPACTED PROFILE, not just raw text)
        profile_context = self.get_profile_context()
        # Note: We pass the profile into the 'history' slot of the therapist agent
        final_response = therapist_agent(user_input, detected_emotion, strategies, profile_context)
        
        AgentLogger.log("Therapist Agent", "Response Generation", final_response)
        
        # 5. TRIGGER COMPACTION (The Learning Step)
        # We update the memory immediately after the turn
        self.compact_memory(f"User felt {detected_emotion}. AI suggested {strategies}. User input: {user_input}")
        
        return final_response

print("âœ… Smart Compacting Memory System Ready.")


# Initialize Session
user_dave = hAIppySession(user_name="Dave")

# --- Interaction 1: Dave is stressed ---
print("\nğŸ”¹ --- TURN 1: Dave (Stress) ---")
# User expresses stress
response1 = user_dave.process_message("I'm so stressed about my presentation.")
print(f"AI: {response1}\n")

# --- SIMULATION: User gives feedback ---
# In a real chat app, the user would type this.
# We manually trigger the 'compact_memory' function to simulate the user saying "Thanks"
print(">> User Feedback: 'Thanks, the Box Breathing really helped.'")
user_dave.compact_memory("User felt stress. Strategy: Box Breathing. User said: Thanks, the Box Breathing really helped.")

# --- Interaction 2: Dave comes back later ---
print("\nğŸ”¹ --- TURN 2: Dave (New Session) ---")

# PROOF: We check the database to see if it learned.
print(f"ğŸ‘€ INTERNAL DATABASE CHECK: {USER_DATABASE['Dave'].successful_strategies}")

# Now when Dave chats again, the AI knows he likes breathing.
response2 = user_dave.process_message("I'm feeling a bit anxious about the meeting.")
print(f"AI: {response2}\n")


# --- ADVANCED DEMO: Multi-User Memory Isolation & Learning ---

print("ğŸ§ª STARTING MULTI-USER SIMULATION...\n")

# 1. Initialize Two Distinct Users
alice = hAIppySession("Alice")
bob = hAIppySession("Bob")

# ==========================================
# ğŸ‘¤ USER 1: ALICE (The "Negative Feedback" Test)
# Goal: Alice tries a strategy, hates it, and the AI should remember to avoid it.
# ==========================================
print(f"\n{'='*40}")
print("ğŸ‘¤ ALICE: SESSION 1 (Anxiety)")
print(f"{'='*40}")

# Alice is anxious
response_a1 = alice.process_message("I am having a panic attack, I feel so anxious.")
print(f"\nğŸ¤– hAIppy (to Alice): {response_a1}")

# --- SIMULATE LEARNING (Negative Feedback) ---
# Alice says: "I hate counting, it makes me more anxious."
# We manually update her profile to simulate the Memory Agent processing this feedback.
print("\n>> ğŸ“� LEARNING STEP: Alice hated the '5-4-3-2-1 Technique'. Updating Profile...")
USER_DATABASE["Alice"].failed_strategies.append("5-4-3-2-1 Technique")

print(f"\n{'='*40}")
print("ğŸ‘¤ ALICE: SESSION 2 (The Return)")
print(f"{'='*40}")

# Alice returns with the same issue. 
# The AI should see '5-4-3-2-1' in FAILED strategies and suggest something else (like Ice Dive).
response_a2 = alice.process_message("I'm still feeling really panicked.")
print(f"\nğŸ¤– hAIppy (to Alice): {response_a2}")


# ==========================================
# ğŸ‘¤ USER 2: BOB (The "Positive Feedback" Test)
# Goal: Bob tries a strategy, loves it, and the AI should remember he prefers it.
# ==========================================
print(f"\n\n{'='*40}")
print("ğŸ‘¤ BOB: SESSION 1 (Burnout)")
print(f"{'='*40}")

# Bob is burnt out
response_b1 = bob.process_message("I've been staring at screens for 12 hours. My brain is fried.")
print(f"\nğŸ¤– hAIppy (to Bob): {response_b1}")

# --- SIMULATE LEARNING (Positive Feedback) ---
# Bob says: "The Digital Sunset sounds perfect."
# We manually update his profile.
print("\n>> ğŸ“� LEARNING STEP: Bob liked 'Digital Sunset'. Updating Profile...")
USER_DATABASE["Bob"].successful_strategies.append("Digital Sunset")


# ==========================================
# ğŸ�† FINAL PROOF: MEMORY ISOLATION
# ==========================================
print(f"\n\n{'='*40}")
print("ğŸ“Š FINAL DATABASE INSPECTION")
print(f"{'='*40}")

# We print the raw database to prove Alice and Bob are separate
print(f"ğŸ“‚ ALICE'S DO NOT USE LIST: {USER_DATABASE['Alice'].failed_strategies}")
print(f"ğŸ“‚ BOB'S FAVORITES LIST:    {USER_DATABASE['Bob'].successful_strategies}")

if "5-4-3-2-1 Technique" not in USER_DATABASE['Bob'].failed_strategies:
    print("\nâœ… SUCCESS: Alice's dislike of counting did NOT leak into Bob's profile.")

