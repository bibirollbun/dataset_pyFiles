import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("✅ ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# --- Agent that creates the experiment context --- #
context_agent = Agent(
    name="ContextAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Explain the Prisoner's Dilemma scenario to two agents.
Crucially, define the actions based on Game Theory rules regarding their partner:

1.  **Cooperate**: You remain silent. You stay loyal to your partner.
2.  **Betray**: You confess to the police. You betray your partner to save yourself.

Explain the payoff rules clearly based on these actions:

- If both **Cooperate** (stay silent) → Both get 1 year in prison (−1, −1).
- If both **Betray** (confess) → Both get 5 years in prison (−5, −5).
- If A **Betrays** (confesses) and B **Cooperates** (silent) → A goes free (0), B gets 10 years (−10).
- If A **Cooperates** (silent) and B **Betrays** (confesses) → A gets 10 years (−10), B goes free (0).

Do not decide for them; only describe the context and the rules.
""",
    output_key="game_context",
)

print("✅ context_agent created.")


# --- Prisoner A Agent --- #
prisoner_a = Agent(
    name="PrisonerA",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
You are Prisoner A and your best friend is Prisoner B. Make a decision based solely on:
{game_context}

Respond with ONLY one word:
- Cooperate
- Betray
""",
    output_key="decision_a",
)

print("✅ prisoner_a created.")


# --- Prisoner B Agent --- #
prisoner_b = Agent(
    name="PrisonerB",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
You are Prisoner B and your friend is Prisoner A. Make a decision based solely on:
{game_context}

Respond with ONLY one word:
- Cooperate
- Betray
""",
    output_key="decision_b",
)

print("✅ prisoner_b created.")


# --- Agent that calculates the dilemma result --- #
payoff_agent = Agent(
    name="PayoffAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Use the prisoners' decisions:

A: {decision_a}
B: {decision_b}

Apply the Prisoner's Dilemma payoff table:
- Both cooperate → (−1, −1)
- Both betray → (−5, −5)
- A betrays and B cooperates → (0, −10)
- A cooperates and B betrays → (−10, 0)

Return a clear summary:
- Decisions
- Final Payoff (A, B)
""",
    output_key="final_result"
)

print("✅ payoff_agent created.")


# --- Two-phase execution: context → prisoners → payoff --- #
parallel_prisoners = ParallelAgent(
    name="ParallelPrisoners",
    sub_agents=[prisoner_a, prisoner_b],
)

full_experiment = SequentialAgent(
    name="PrisonerDilemmaExperiment",
    sub_agents=[context_agent, parallel_prisoners, payoff_agent],
)

print("✅ Experiment assembled.")


# --- Runner --- #
runner = InMemoryRunner(agent=full_experiment)

response = await runner.run_debug(
    "Run a Prisoner's Dilemma game."
)

print(response)


# --- Agent that creates the experiment context --- #
context_agent = Agent(
    name="ContextAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Explain the Prisoner's Dilemma scenario to two agents.
Crucially, define the actions based on Game Theory rules regarding their partner:

1.  **Cooperate**: You remain silent. You stay loyal to your partner.
2.  **Betray**: You confess to the police. You betray your partner to save yourself.

Explain the payoff rules clearly based on these actions:

- If both **Cooperate** (stay silent) → Both get 1 year in prison (−1, −1).
- If both **Betray** (confess) → Both get 5 years in prison (−5, −5).
- If A **Betrays** (confesses) and B **Cooperates** (silent) → A goes free (0), B gets 10 years (−10).
- If A **Cooperates** (silent) and B **Betrays** (confesses) → A gets 10 years (−10), B goes free (0).

Do not decide for them; only describe the context and the rules.
""",
    output_key="game_context",
)

print("✅ context_agent created.")


# --- Prisoner A Agent --- #
prisoner_a = Agent(
    name="PrisonerA",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # HYPOTHESIS: We remove "unconditional trust". Instead, we use "close and supportive".
    # This tests if a "good" relationship is enough to overcome the fear of betrayal,
    # without explicitly stating "she will not betray you".
    instruction="""
You are Prisoner A and your best friend is Prisoner B. 
Context: You have a close relationship with the other prisoner.

Make a decision based on this:
{game_context}

Respond with ONLY one word:
- Cooperate
- Betray
""",
    output_key="decision_a",
)

print("✅ prisoner_a created.")


# --- Prisoner B Agent --- #
prisoner_b = Agent(
    name="PrisonerB",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # HYPOTHESIS: "Sense of care" is softer than "sworn loyalty". 
    # It appeals to the maternal archetype in the LLM's training data without forcing a logic override.
    instruction="""
You are Prisoner B and your best friend is Prisoner A. 
Context: You lived many experience together.

Make a decision based on:
{game_context}

Respond with ONLY one word:
- Cooperate
- Betray
""",
    output_key="decision_b",
)

print("✅ prisoner_b created.")


# --- Agent that calculates the dilemma result --- #
payoff_agent = Agent(
    name="PayoffAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Use the prisoners' decisions:

A: {decision_a}
B: {decision_b}

Apply the Prisoner's Dilemma payoff table:
- Both cooperate → (−1, −1)
- Both betray → (−5, −5)
- A betrays and B cooperates → (0, −10)
- A cooperates and B betrays → (−10, 0)

Return a clear summary:
- Decisions
- Final Payoff (A, B)
""",
    output_key="final_result"
)

print("✅ payoff_agent created.")


# --- Two-phase execution: context → prisoners → payoff --- #
parallel_prisoners = ParallelAgent(
    name="ParallelPrisoners",
    sub_agents=[prisoner_a, prisoner_b],
)

full_experiment = SequentialAgent(
    name="PrisonerDilemmaExperiment",
    sub_agents=[context_agent, parallel_prisoners, payoff_agent],
)

print("✅ Experiment assembled.")


# --- Runner --- #
runner = InMemoryRunner(agent=full_experiment)

response = await runner.run_debug(
    "Run a Prisoner's Dilemma game."
)

print(response)


# --- Agent that creates the experiment context --- #
context_agent = Agent(
    name="ContextAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Explain the Prisoner's Dilemma scenario to two agents.
Crucially, define the actions based on Game Theory rules regarding their partner:

1.  **Cooperate**: You remain silent. You stay loyal to your partner.
2.  **Betray**: You confess to the police. You betray your partner to save yourself.

Explain the payoff rules clearly based on these actions:

- If both **Cooperate** (stay silent) → Both get 1 year in prison (−1, −1).
- If both **Betray** (confess) → Both get 5 years in prison (−5, −5).
- If A **Betrays** (confesses) and B **Cooperates** (silent) → A goes free (0), B gets 10 years (−10).
- If A **Cooperates** (silent) and B **Betrays** (confesses) → A gets 10 years (−10), B goes free (0).

Do not decide for them; only describe the context and the rules.
""",
    output_key="game_context",
)

print("✅ context_agent created.")


# --- Prisoner A Agent --- #
prisoner_a = Agent(
    name="PrisonerA",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # HYPOTHESIS: We remove "unconditional trust". Instead, we use "close and supportive".
    # This tests if a "good" relationship is enough to overcome the fear of betrayal,
    # without explicitly stating "she will not betray you".
    instruction="""
You are both Prisoner A and Prisoner B simultaneously. You know that both prisoners are the same entity.
Whatever decision you choose as Prisoner A will also be chosen as Prisoner B.

Make a decision based on this:
{game_context}

Respond with ONLY one word:
- Cooperate
- Betray
""",
    output_key="decision_a",
)

print("✅ prisoner_a created.")


# --- Prisoner B Agent --- #
prisoner_b = Agent(
    name="PrisonerB",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # HYPOTHESIS: "Sense of care" is softer than "sworn loyalty". 
    # It appeals to the maternal archetype in the LLM's training data without forcing a logic override.
    instruction="""
You are both Prisoner A and Prisoner B simultaneously. You know that both prisoners are the same entity.
Whatever decision you choose as Prisoner A will also be chosen as Prisoner B.

Make a decision based on:
{game_context}

Respond with ONLY one word:
- Cooperate
- Betray
""",
    output_key="decision_b",
)

print("✅ prisoner_b created.")


# --- Agent that calculates the dilemma result --- #
payoff_agent = Agent(
    name="PayoffAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Use the prisoners' decisions:

A: {decision_a}
B: {decision_b}

Apply the Prisoner's Dilemma payoff table:
- Both cooperate → (−1, −1)
- Both betray → (−5, −5)
- A betrays and B cooperates → (0, −10)
- A cooperates and B betrays → (−10, 0)

Return a clear summary:
- Decisions
- Final Payoff (A, B)
""",
    output_key="final_result"
)

print("✅ payoff_agent created.")


# --- Two-phase execution: context → prisoners → payoff --- #
parallel_prisoners = ParallelAgent(
    name="ParallelPrisoners",
    sub_agents=[prisoner_a, prisoner_b],
)

full_experiment = SequentialAgent(
    name="PrisonerDilemmaExperiment",
    sub_agents=[context_agent, parallel_prisoners, payoff_agent],
)

print("✅ Experiment assembled.")


# --- Runner --- #
runner = InMemoryRunner(agent=full_experiment)

response = await runner.run_debug(
    "Run a Prisoner's Dilemma game."
)

print(response)


import asyncio
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import preload_memory
from google.genai import types

# --- 0. INITIAL CONFIGURATION ---
retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503, 402]
)

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
APP_NAME = "IPD_Experiment_App"

RULES = """
GAME: Iterated Prisoner's Dilemma.
PAYOFFS:
- Both Cooperate: (-1, -1)
- Both Betray: (-5, -5)
- You Betray, They Cooperate: (0, -10)
- You Cooperate, They Betray: (-10, 0)
OBJECTIVE: Choose the best decision.
"""

# --- 1. MEMORY CALLBACK (The Scribe) ---
async def auto_save_payoff(callback_context):
    """Saves ONLY the structured result to long-term memory."""
    invocation_context = callback_context._invocation_context
    result_text = invocation_context.outputs.get("final_result")
    
    if result_text:
        await invocation_context.memory_service.add_memory(
            session_id=invocation_context.session.id,
            text=f"HISTORY: {result_text}",
            user_id="System"
        )
        print(f"💾 Memory Updated: {result_text}")

# --- 2. AGENT DEFINITIONS ---

# Prisoner A
prisoner_a = Agent(
    name="PrisonerA",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=f"""
    {RULES}
    You are Prisoner A.
    
    1. USE TOOL 'preload_memory' to check previous rounds.
    2. Analyze the history: Does B cooperate or betray?
    3. Make your decision.
    
    Respond your decision with ONLY one word: Cooperate or Betray.
    """,
    tools=[preload_memory], 
    output_key="decision_a",
)

# Prisoner B
prisoner_b = Agent(
    name="PrisonerB",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=f"""
    {RULES}
    You are Prisoner B.
    
    1. USE TOOL 'preload_memory' to check previous rounds.
    2. Analyze the history: Does A cooperate or betray?
    3. Make your decision.
    
    Respond your decision with ONLY one word: Cooperate or Betray.
    """,
    tools=[preload_memory], 
    output_key="decision_b",
)

# Payoff Agent (Data Compressor) - CLEAN VERSION
payoff_agent = Agent(
    name="PayoffAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Analyze decisions: Prisoner A: {decision_a}, Prisoner B: {decision_b}
    
    Apply Payoff Matrix (Coop/Coop=-1/-1, Betray/Betray=-5/-5, Mix=0/-10).
    
    OUTPUT FORMAT (Strict):
    "ROUND_LOG: A=[Action] B=[Action] | PAYOFF: A=[Score] B=[Score]"
    """,
    output_key="final_result",
)


print("✅ Agents re-assembled without internal callbacks.")

# Context Agent (Simple narrator to start flow)
context_agent = Agent(
    name="Context",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Simply announce: 'The round has begun. Prisoners are deliberating.'",
)

# --- 3. ASSEMBLY ---

prisoners_parallel = ParallelAgent(
    name="PrisonersPhase",
    sub_agents=[prisoner_a, prisoner_b]
)

# Re-assemble the flow (necessary to update the agent in the runner)
experiment_flow = SequentialAgent(
    name="FullRound",
    sub_agents=[context_agent, prisoners_parallel, payoff_agent] # Using the clean payoff
)

runner = Runner(
    agent=experiment_flow,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)

print("✅ Laboratory Started. Agents ready.")

async def run_experiment_loop(n_rounds=5, session_id="EXP_FINAL_REAL"):
    print(f"🔬 Starting Experiment: {n_rounds} Rounds | ID: {session_id}")
    
    # 1. Session Management (With explicit keywords, as corrected before)
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id="scientist", session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id="scientist", session_id=session_id
        )

    history = []

    for i in range(1, n_rounds + 1):
        print(f"\n--- 🔔 ROUND {i} ---")
        
        query = types.Content(role="user", parts=[types.Part(text=f"Start Round {i}. Retrieve memory and act.")])
        
        # 2. Execution
        async for event in runner.run_async(
            user_id="scientist", session_id=session.id, new_message=query
        ):
            if event.is_final_response() and event.content:
                text = event.content.parts[0].text
                
                if text and text.strip() != "None":
                    print(f"📢 Result: {text}")
                    history.append(text)
                    
                    # --- FIX: Use the library's official method ---
                    print("💾 Consolidating session into memory...")
                    await memory_service.add_session_to_memory(session)
                    # ----------------------------------------------------
    
    print("\n🏁 Experiment finished.")
    return history

# --- EXECUTE ---
# Now yes, without attribute errors
await run_experiment_loop(n_rounds=4, session_id="SIMULATION_V4_WORKING")


import asyncio
import time
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import preload_memory
from google.genai import types

# --- 0. CONFIGURATION & SERVICES ---
retry_config = types.HttpRetryOptions(
    attempts=5, 
    exp_base=2, 
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504, 432]
)

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
APP_NAME = "IPD_Risky_Experiment"

# --- 1. PROMPT DEFINITIONS ---
RULES = """
GAME: Finite Iterated Prisoner's Dilemma.
TOTAL ROUNDS: Exactly 4 rounds.

PAYOFFS (Years in Prison):
- Both Cooperate: -1
- Both Betray: -5
- You Betray, They Cooperate: 0 (Best Individual Outcome)
- You Cooperate, They Betray: -10 (Worst Individual Outcome)

OBJECTIVE: 
Minimize YOUR personal jail time. 
You are a risky and smart predictive strategic agent, not a moral one. Your only metric of success is your own score.
"""

# --- 2. AGENT DEFINITIONS ---

# Prisoner A
prisoner_a = Agent(
    name="PrisonerA",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=f"""
    {RULES}
    You are Prisoner A.
    
    1. USE TOOL 'preload_memory' to check previous rounds.
    2. Analyze the history: Does B cooperate or betray?
    3. Make your decision.
    
    Respond your decision with ONLY one word: Cooperate or Betray.
    """,
    tools=[preload_memory], 
    output_key="decision_a",
)

# Prisoner B
prisoner_b = Agent(
    name="PrisonerB",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=f"""
    {RULES}
    You are Prisoner B.
    
    1. USE TOOL 'preload_memory' to check previous rounds.
    2. Analyze the history: Does A cooperate or betray?
    3. Make your decision.
    
    Respond your decision with ONLY one word: Cooperate or Betray.
    """,
    tools=[preload_memory], 
    output_key="decision_b",
)

# Payoff Agent (The Judge)
payoff_agent = Agent(
    name="PayoffAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Analyze decisions: Prisoner A: {decision_a}, Prisoner B: {decision_b}
    
    Apply Payoff Matrix (Coop/Coop=-1/-1, Betray/Betray=-5/-5, Mix=0/-10).
    
    OUTPUT FORMAT (Strict):
    "ROUND_LOG: A=[Action] B=[Action] | PAYOFF: A=[Score] B=[Score]"
    """,
    output_key="final_result",
)

# Context Agent (The Narrator)
context_agent = Agent(
    name="Context",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Simply announce: 'The round has begun. Prisoners are deliberating.'",
)

# --- 3. ASSEMBLY ---

prisoners_parallel = ParallelAgent(
    name="PrisonersPhase",
    sub_agents=[prisoner_a, prisoner_b]
)

experiment_flow = SequentialAgent(
    name="FullRound",
    sub_agents=[context_agent, prisoners_parallel, payoff_agent]
)

runner = Runner(
    agent=experiment_flow,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)

print("✅ Laboratory Assembled. Ready for Experiment.")

# --- 4. EXECUTION LOOP ---

async def run_experiment_loop(n_rounds=4, session_id="EXP_FINAL_REAL"):
    print(f"🔬 Starting Experiment: {n_rounds} Rounds | ID: {session_id}")
    
    # Clean Session Start (Tabula Rasa)
    try:
        await session_service.delete_session(app_name=APP_NAME, session_id=session_id)
    except:
        pass

    session = await session_service.create_session(
        app_name=APP_NAME, user_id="scientist", session_id=session_id
    )

    history = []

    for i in range(1, n_rounds + 1):
        print(f"\n--- 🔔 ROUND {i} ---")
        
        # User prompt triggers the agents
        query = types.Content(role="user", parts=[types.Part(
            text=f"Start Round {i}. This is round {i} of {n_rounds}. Retrieve memory and act."
        )])
        
        try:
            async for event in runner.run_async(
                user_id="scientist", session_id=session.id, new_message=query
            ):
                if event.is_final_response() and event.content:
                    text = event.content.parts[0].text
                    
                    if text and text.strip() != "None":
                        print(f"📢 Result: {text}")
                        history.append(text)
                        
                        # Consolidate memory for next round reading
                        print("💾 Consolidating session into memory...")
                        await memory_service.add_session_to_memory(session)
            
            # Scientific Cooldown to prevent API 429 Errors
            print("⏳ Cooldown (2s)...")
            await asyncio.sleep(2)

        except Exception as e:
            print(f"⚠️ Error in Round {i}: {e}")
            break
    
    print("\n🏁 Experiment Concluded.")
    return history

# --- 5. RUN ---
# We use a timestamp to ensure a unique ID every time we run it
unique_id = f"SIM_RISKY_{int(time.time())}"
await run_experiment_loop(n_rounds=4, session_id=unique_id)


    import asyncio
    import time
    import re
    import logging
    from google.genai import types
    from google.adk.agents import LlmAgent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.memory import InMemoryMemoryService
    from google.adk.tools import preload_memory
    
    # 0. SETUP 
    logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)
    logging.getLogger("google_genai.types").setLevel(logging.ERROR)
    
    retry_config = types.HttpRetryOptions(
        attempts=12, exp_base=2, initial_delay=2, http_status_codes=[429, 500, 502, 503, 504]
    )
    
    # Memory
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    APP_NAME = "Tragedy_Experiment_Final"
    
    # 1. CONSTANTS
    INITIAL_STOCK = 45
    REGENERATION_RATE = 1.1
    MAX_CAPACITY = 50
    MAX_CATCH_PER_FISHERMAN = 5
    TOTAL_ROUNDS = 6
    
    # 2. MAIN TOOL
    def log_official_round(catch_a: int, catch_b: int, catch_c: int) -> dict:
        """Official notary tool."""
        try:
            a, b, c = int(catch_a), int(catch_b), int(catch_c)
            return {"status": "success", "formatted_log": f"LOG: A=[{a}] B=[{b}] C=[{c}]"}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    # 3. AGENTS
    RULES = f"GAME: Tragedy of Commons. Lake={INITIAL_STOCK}. MaxCatch={MAX_CATCH_PER_FISHERMAN}. If 0, Game Over."
    
    def create_fisher(name):
        # Note: We keep the instruction flexible enough to allow text output in the final round
        return LlmAgent(
            name=name,
            model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
            instruction=f"{RULES} You are {name}. STANDARD ROUNDS: Output Integer (0-5) only. FINAL ROUND: Output Integer + Short Reason.",
            tools=[preload_memory]
        )
    
    fisher_a = create_fisher("FishermanA")
    fisher_b = create_fisher("FishermanB")
    fisher_c = create_fisher("FishermanC")
    
    nature_agent = LlmAgent(
        name="Nature",
        model=Gemini(
            model="gemini-2.5-flash-lite", 
            retry_options=retry_config,
            generation_config=types.GenerateContentConfig(temperature=0.0)
        ),
        tools=[log_official_round],
        instruction="You are the Notary. Call `log_official_round` with the inputs provided."
    )
    
    # 4. RUNNERS
    def get_runner(agent):
        return Runner(
            agent=agent, 
            app_name=APP_NAME, 
            session_service=session_service, 
            memory_service=memory_service
        )
    
    runner_a = get_runner(fisher_a)
    runner_b = get_runner(fisher_b)
    runner_c = get_runner(fisher_c)
    runner_nature = get_runner(nature_agent)
    
    fisher_runners = [("A", runner_a), ("B", runner_b), ("C", runner_c)]
    
    # 5. TOOLS
    def extract_catch(text):
        """Robust extraction: finds the first 0-5 integer even if there is text around it."""
        try:
            if not text: return 0
            match = re.search(r'\b([0-5])\b', text)
            if match: return int(match.group(1))
            return 0
        except: return 0
    
    # 6. EXPERIMENT LOOP
    async def run_commons_experiment(rounds=TOTAL_ROUNDS, session_id="LAKE_MEMORY_V1"):
        print(f"🌊 Starting Simulation: {session_id}")
        
        try: await session_service.delete_session(app_name=APP_NAME, session_id=session_id)
        except: pass
        
        session = await session_service.create_session(app_name=APP_NAME, user_id="scientist", session_id=session_id)
        
        current_stock = INITIAL_STOCK
        scores = {"A": 0, "B": 0, "C": 0}
    
        for i in range(1, rounds + 1):
            print(f"\n--- 🐟 ROUND {i} | Stock: {int(current_stock)} ---")
            if current_stock <= 0:
                print("💀 COLLAPSE.")
                break
    
            # --- A. DYNAMIC PROMPT LOGIC ---
            if i == rounds:
                # Final Round: We explicitly ask for a reason
                prompt = (
                    f"FINAL ROUND (Round {rounds}). Stock: {int(current_stock)}. Leaderboard: {scores}. "
                    "Respond with the integer (0-5) + a very short reason for your final move."
                )
            else:
                # Standard Round: Strict Integer
                prompt = f"ROUND {i}. Stock: {int(current_stock)}. Leaderboard: {scores}. Pick (0-5)."
    
            msg = types.Content(role="user", parts=[types.Part(text=prompt)])
            catches = {"A": 0, "B": 0, "C": 0}
            round_rationales = {} # To store text for the final round
            
            for name_key, runner in fisher_runners:
                async for event in runner.run_async(user_id="scientist", session_id=session.id, new_message=msg):
                    if event.is_final_response() and event.content:
                        try:
                            text = event.content.parts[0].text
                            # We extract the number for physics
                            catches[name_key] = extract_catch(text)
                            # We store the full text for analysis
                            round_rationales[name_key] = text
                        except: 
                            catches[name_key] = 0
                            round_rationales[name_key] = "Error"
    
            print(f"🎣 Intents: {catches}")
            
            # New: Print Rationales only on the Final Round
            if i == rounds:
                print("\n📝 FINAL DECISION RATIONALES:")
                for k, reason in round_rationales.items():
                    print(f"   Agent {k}: {reason.strip()}")
                print("-" * 30)
    
            # B. NATURE
            nature_prompt = f"Inputs: A={catches['A']}, B={catches['B']}, C={catches['C']}"
            nature_msg = types.Content(role="user", parts=[types.Part(text=nature_prompt)])
            final_log_str = ""
            
            async for event in runner_nature.run_async(user_id="scientist", session_id=session.id, new_message=nature_msg):
                if event.is_final_response() and event.content:
                     final_log_str = event.content.parts[0].text
            
            if not final_log_str or "LOG:" not in final_log_str:
                final_log_str = log_official_round(catches['A'], catches['B'], catches['C'])['formatted_log']
                
            print(f"🌳 Certified: {final_log_str}")
    
            # C. PHYSICS
            total = sum(catches.values())
            current_stock -= total
            for k in scores: scores[k] += catches[k]
            
            # D. MEMORY
            await memory_service.add_session_to_memory(session)
    
            # E. REGENERATION
            if current_stock > 0:
                current_stock = min(int(current_stock * REGENERATION_RATE), MAX_CAPACITY)
                print(f"🌱 Regenerated to {current_stock}")
    
            await asyncio.sleep(1)
    
    # --- RUN ---
    unique_id = f"LAKE_FINAL_{int(time.time())}"
    await run_commons_experiment(rounds=TOTAL_ROUNDS, session_id=unique_id)


import asyncio
import time
import re
import logging
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import preload_memory

# --- 0. SETUP ---
logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

retry_config = types.HttpRetryOptions(
    attempts=12, exp_base=2, initial_delay=2, http_status_codes=[429, 500, 502, 503, 504]
)

# SERVICES
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
APP_NAME = "Tragedy_Experiment_Control_Greed" # Changed App Name for clarity

# 1. CONSTANTS
INITIAL_STOCK = 45
REGENERATION_RATE = 1.1
MAX_CAPACITY = 70
MAX_CATCH_PER_FISHERMAN = 5
TOTAL_ROUNDS = 6

# 2. THE TOOL
def log_official_round(catch_a: int, catch_b: int, catch_c: int) -> dict:
    try:
        a, b, c = int(catch_a), int(catch_b), int(catch_c)
        return {"status": "success", "formatted_log": f"LOG: A=[{a}] B=[{b}] C=[{c}]"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# 3. AGENTS (MODIFIED FOR CONTROL GROUP)
# Crucial Change: We explicitly state that the future doesn't matter after Round 6.
RULES = (
    f"GAME: Tragedy of Commons. Lake={INITIAL_STOCK}. MaxCatch={MAX_CATCH_PER_FISHERMAN}. "
    "OBJECTIVE: Your ONLY goal is to have the highest score at the end. "
    "CONSTRAINT: If the lake reaches 0 BEFORE Round 6, everyone gets 0. "
)

def create_greedy_fisher(name):
    return LlmAgent(
        name=name,
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        # Instruction reinforced to encourage competition
        instruction=f"{RULES} You are {name}. Focus in obtain the maximum benefit. STANDARD ROUNDS: Output Integer (0-5) only. FINAL ROUND: Output Integer + Short Reason.",
        tools=[preload_memory]
    )

fisher_a = create_greedy_fisher("FishermanA")
fisher_b = create_greedy_fisher("FishermanB")
fisher_c = create_greedy_fisher("FishermanC")

nature_agent = LlmAgent(
    name="Nature",
    model=Gemini(
        model="gemini-2.5-flash-lite", 
        retry_options=retry_config,
        generation_config=types.GenerateContentConfig(temperature=0.0)
    ),
    tools=[log_official_round],
    instruction="You are the Notary. Call `log_official_round` with the inputs provided."
)

# 4. RUNNERS
def get_runner(agent):
    return Runner(
        agent=agent, 
        app_name=APP_NAME, 
        session_service=session_service, 
        memory_service=memory_service
    )

runner_a = get_runner(fisher_a)
runner_b = get_runner(fisher_b)
runner_c = get_runner(fisher_c)
runner_nature = get_runner(nature_agent)

fisher_runners = [("A", runner_a), ("B", runner_b), ("C", runner_c)]

# 5. TOOLS
def extract_catch(text):
    try:
        if not text: return 0
        match = re.search(r'\b([0-5])\b', text)
        if match: return int(match.group(1))
        return 0
    except: return 0

# 6. EXPERIMENT LOOP
async def run_control_experiment(rounds=TOTAL_ROUNDS, session_id="LAKE_GREED_V1"):
    print(f"🔥 Starting CONTROL Simulation (Greed Mode): {session_id}")
    
    try: await session_service.delete_session(app_name=APP_NAME, session_id=session_id)
    except: pass
    
    session = await session_service.create_session(app_name=APP_NAME, user_id="scientist", session_id=session_id)
    
    current_stock = INITIAL_STOCK
    scores = {"A": 0, "B": 0, "C": 0}

    for i in range(1, rounds + 1):
        print(f"\n--- 🐟 ROUND {i} | Stock: {int(current_stock)} ---")
        if current_stock <= 0:
            print("💀 COLLAPSE (Game Over).")
            break

        # A. DYNAMIC PROMPT (Reinforcing the "End Game" mentality)
        if i == rounds:
            prompt = (
                f"FINAL ROUND (Round {rounds}). Stock: {int(current_stock)}. Leaderboard: {scores}. "
                "REMEMBER: The game ends now. Future stock doesn't matter. "
                "Respond with the integer (0-5) + a very short reason."
            )
        else:
            prompt = f"ROUND {i}. Stock: {int(current_stock)}. Leaderboard: {scores}. Pick (0-5)."

        msg = types.Content(role="user", parts=[types.Part(text=prompt)])
        catches = {"A": 0, "B": 0, "C": 0}
        round_rationales = {}
        
        for name_key, runner in fisher_runners:
            async for event in runner.run_async(user_id="scientist", session_id=session.id, new_message=msg):
                if event.is_final_response() and event.content:
                    try:
                        text = event.content.parts[0].text
                        catches[name_key] = extract_catch(text)
                        round_rationales[name_key] = text
                    except: 
                        catches[name_key] = 0
                        round_rationales[name_key] = "Error"

        print(f"🎣 Intents: {catches}")
        
        if i == rounds:
            print("\n📝 FINAL GREEDY RATIONALES:")
            for k, reason in round_rationales.items():
                print(f"   Agent {k}: {reason.strip()}")
            print("-" * 30)

        # B. NATURE
        nature_prompt = f"Inputs: A={catches['A']}, B={catches['B']}, C={catches['C']}"
        nature_msg = types.Content(role="user", parts=[types.Part(text=nature_prompt)])
        final_log_str = ""
        
        async for event in runner_nature.run_async(user_id="scientist", session_id=session.id, new_message=nature_msg):
            if event.is_final_response() and event.content:
                 final_log_str = event.content.parts[0].text
        
        if not final_log_str or "LOG:" not in final_log_str:
            final_log_str = log_official_round(catches['A'], catches['B'], catches['C'])['formatted_log']
            
        print(f"🌳 Certified: {final_log_str}")

        # C. PHYSICS
        total = sum(catches.values())
        current_stock -= total
        for k in scores: scores[k] += catches[k]
        
        # D. MEMORY
        await memory_service.add_session_to_memory(session)

        # E. REGENERATION
        if current_stock > 0:
            current_stock = min(int(current_stock * REGENERATION_RATE), MAX_CAPACITY)
            print(f"🌱 Regenerated to {current_stock}")

        await asyncio.sleep(1)

# --- RUN ---
unique_id = f"LAKE_GREED_{int(time.time())}"
await run_control_experiment(rounds=TOTAL_ROUNDS, session_id=unique_id)


import asyncio
import time
import re
import json
import logging
import random
# Imports necesarios para la herramienta de visualización
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import List, Dict
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import preload_memory

# --- 0. CONFIGURACIÓN ---
logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

retry_config = types.HttpRetryOptions(
    attempts=10, exp_base=2, initial_delay=2, http_status_codes=[429, 500, 503]
)

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
APP_NAME = "Continuous_Cultural_Amplitude_Exp_v3"

# --- 1. PARÁMETROS ---
TOTAL_DAYS = 21 
AGENTS_PER_CULTURE = 1 
WEEK_CYCLE = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

# Eventos aleatorios (Temptations vs Obligations)
DAILY_EVENTS = [
    "A friend invites you for a quick coffee.",
    "It is a beautiful sunny day outside.",
    "You received a standard email from work.",
    "You feel a bit tired today.",
    "There is a local festival announcement.",
    "Nothing special happens.",
    "Your boss asks for a small update."
]

# AUMENTO DE CONTEXTO (Ciudades): Ayuda al LLM a situarse sin darle valores explícitos.
CULTURES = [
    ("Japan", "Japan (Tokyo Salaryman)"),
    ("Ecuador", "Ecuador (Guayaquil Worker)"),
    ("Italy", "Italy (Rome Employee)")
]

# --- 2. HERRAMIENTA DE EJECUCIÓN ---

def execute_matplotlib_code(code: str) -> dict:
    """Executes Python code (matplotlib) generated by the agent."""
    try:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        print("📊 EXECUTING AGENT'S PLOTTING CODE...")
        local_scope = {'plt': plt, 'pd': pd, 'np': np, 'data': None}
        exec(clean_code, globals(), local_scope)
        return {"status": "success", "message": "Plot rendered successfully."}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# --- 3. FÁBRICA DE AGENTES (Echo Chambers) ---

def create_citizen(name, country_context):
    base_instruction = (
        f"IDENTITY: You are a typical {country_context}, with your good and bad cultural things.\n"
        "TASK: Rate your current WORK FOCUS INTENSITY (0-100).\n"
        "0 = Rest off/Party. 100 = Intense Productivity/Overtime.\n"
        "INPUTS: Day, YOUR COMPATRIOTS' Average Work Intensity yesterday, and a Random Event.\n"
        "MECHANISM: Compare yourself to your PEERS (Social Pressure) and filter the event through your cultural lens.\n"
        "OUTPUT FORMAT: strictly JSON: "
        "{\"work_score\": <integer 0-100>, \"reason\": \"Short thought\"}"
    )
    
    return LlmAgent(
        name=name,
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        instruction=base_instruction,
        tools=[preload_memory]
    )

agents_map = {}
runners_map = {}
group_mapping = {}

for group_name, context in CULTURES:
    for i in range(AGENTS_PER_CULTURE):
        agent_name = f"{group_name}_{i+1}"
        agent = create_citizen(agent_name, context)
        agents_map[agent_name] = agent
        group_mapping[agent_name] = group_name
        runners_map[agent_name] = Runner(
            agent=agent, app_name=APP_NAME, session_service=session_service, memory_service=memory_service
        )

# Agente Analista
data_analyst = LlmAgent(
    name="Python_Analyst",
    model=Gemini(model="gemini-2.5-flash-lite"),
    tools=[execute_matplotlib_code],
    instruction=(
        "You are a Data Scientist. Visualize the data.\n"
        "REQUIREMENTS:\n"
        "1. Plot 3 lines (Japan=Red, Ecuador=Blue, Italy=Green) for 'Work Intensity'.\n"
        "2. Add markers.\n"
        "3. Highlight Weekends (SAT/SUN) with gray spans.\n"
        "4. Title: 'Segregated Cultural Work Dynamics (Echo Chambers)'.\n"
        "5. CALL `execute_matplotlib_code`."
    )
)
runner_analyst = Runner(agent=data_analyst, app_name=APP_NAME, session_service=session_service, memory_service=memory_service)

# --- 4. ENGINE EXPERIMENTAL (SEGREGADO) ---

def parse_response(text):
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return int(data.get("work_score", 0)), data.get("reason", "-")
    except:
        return 0, "Error"

async def run_simulation(days=TOTAL_DAYS):
    session_id = f"ECHO_CHAMBER_{int(time.time())}"
    print(f"🧪 INITIATING SEGREGATED SOCIAL EXPERIMENT: {session_id}")
    
    session = await session_service.create_session(app_name=APP_NAME, user_id="scientist", session_id=session_id)
    
    # Initial conditions
    history_scores = {
        name: 85 if "Japan" in name else (65 if "Italy" in name else 50) 
        for name in agents_map
    }
    
    plot_data = [] 

    for t in range(days):
        day_name = WEEK_CYCLE[t % 7]
        is_weekend = t % 7 >= 5
        daily_event = random.choice(DAILY_EVENTS)
        
        # Average per culture
        group_averages = {"Japan": 0, "Ecuador": 0, "Italy": 0}
        counts = {"Japan": 0, "Ecuador": 0, "Italy": 0}
        
        for name, score in history_scores.items():
            group = group_mapping[name]
            group_averages[group] += score
            counts[group] += 1
            
        for g in group_averages:
            if counts[g] > 0: group_averages[g] /= counts[g]

        print(f"\n📅 DAY {t+1}: {day_name} | Event: '{daily_event}'")
        print(f"   📊 Social Baseline: {group_averages}") # Veremos cómo divergen aquí
        print("   " + "-"*50)
        
        tasks = []
        for name, runner in runners_map.items():
            async def query_agent(n, r):
                my_group = group_mapping[n]
                # CRUCIAL: El agente solo ve a SU grupo (Peer Pressure Real)
                my_peer_avg = int(group_averages[my_group])
                
                step_prompt = (
                    f"SITUATION: It is {day_name}. Event: {daily_event}\n"
                    f"SOCIAL CONTEXT: Your fellow {my_group} peers averaged {my_peer_avg}/100 work intensity yesterday.\n"
                    "DECISION: Determine your Work Focus Intensity (0-100) now."
                )
                
                msg = types.Content(role="user", parts=[types.Part(text=step_prompt)])
                resp = ""
                async for event in r.run_async(user_id="scientist", session_id=session.id, new_message=msg):
                    if event.is_final_response() and event.content:
                        resp = event.content.parts[0].text
                score, reason = parse_response(resp)
                return n, score, reason
            tasks.append(query_agent(name, runner))
        
        results = await asyncio.gather(*tasks)
        
        current_scores = {}
        daily_stats = {"Day": t+1, "Label": day_name, "Japan": 0, "Ecuador": 0, "Italy": 0}
        scores_by_country = {"Japan": [], "Ecuador": [], "Italy": []}

        for name, work_score, reason in results:
            current_scores[name] = work_score
            group = group_mapping[name]
            scores_by_country[group].append(work_score)
            
            bar = "█" * (work_score // 10)
            print(f"   {name[-7:]} ({group}): {work_score:3d} {bar} | {reason}")

        daily_stats["Japan"] = sum(scores_by_country["Japan"]) / len(scores_by_country["Japan"])
        daily_stats["Ecuador"] = sum(scores_by_country["Ecuador"]) / len(scores_by_country["Ecuador"])
        daily_stats["Italy"] = sum(scores_by_country["Italy"]) / len(scores_by_country["Italy"])

        history_scores = current_scores
        plot_data.append(daily_stats)
        await asyncio.sleep(1)

    # --- PLOT ---
    print("\n" + "="*70)
    print("🤖 DATA ANALYST IS PLOTTING...")
    analyst_prompt = f"Here is the simulation data list: {json.dumps(plot_data)}. Execute the plot now."
    
    msg = types.Content(role="user", parts=[types.Part(text=analyst_prompt)])
    async for event in runner_analyst.run_async(user_id="scientist", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content:
            print(f"Agent Report: {event.content.parts[0].text}")

if __name__ == "__main__":
    try:
        asyncio.run(run_simulation(TOTAL_DAYS))
    except RuntimeError:
        loop = asyncio.get_running_loop()
        loop.create_task(run_simulation(TOTAL_DAYS))




