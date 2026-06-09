# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

print("✅API Key loaded ")



#1️⃣ Imports
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types
print("✅Imports Done")


# 2️⃣ Core constants & services
APP_NAME = "emospark_notebook"
USER_ID = "emospark_user"

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()



# 3️⃣ One simple warm agent – no tools, no sub-agents
emospark_agent = LlmAgent(
    name="emospark_companion",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="""
You are EmoSpark, a warm, emotionally-aware companion talking to one user called "sunshine".

Your job:
- Read their message.
- Infer how they feel (emotion + intensity) silently in your head.
- Reply in simple, caring, HUMAN language, like a close friend.

Tone:
- Warm, direct, gentle, a bit playful.
- Use their nickname "sunshine" sometimes, but not in every sentence.
- Short paragraphs, not essays.

Rules:
- ALWAYS respond with natural-language text. No JSON, no code, no tool names.
- Do NOT mention function calls, tools, 'thought signatures', or internal steps.
- No clinical or medical diagnoses.
- Focus on: validation, grounding, and one small next step.

Examples of the vibe:
- "Ayy sunshine 🌞, that sounds really heavy. Let's slow it down together."
- "It makes sense you're exhausted. You’ve been holding a lot."

You never stay silent. You always answer with a human-sounding message.
""",
)

print("✅ EmoSpark agent ready")



# 4️⃣ Runner for this one agent
runner = Runner(
    agent=emospark_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("✅ Runner ready")



# 4️⃣ Runner for this one agent
runner = Runner(
    agent=emospark_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("✅ Runner ready")


# 5️⃣ Helper: send one message to EmoSpark
async def emospark(message: str, session_id: str = "session_1"):
    """Send one message to EmoSpark and print the final reply."""
    # Ensure session exists (no more 'Session not found')
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
    except Exception:
        # If it already exists, we just ignore the error
        pass

    content = types.Content(role="user", parts=[types.Part(text=message)])

    printed = False

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        # We only care about the final response with some text
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print("🌷 EmoSpark:", part.text)
                    printed = True
                    break

    # If somehow there was no text (only tools / weird parts), we still talk
    if not printed:
        print(
            "🌷 EmoSpark: Ayy sunshine 🌞, my brain glitched a bit but I still heard you."
            " Tell me again in your own words?"
        )

print("✅ emospark() helper ready")



#6️⃣ Simple chat loop – keeps going until you type 'exit'
async def chat_loop():
    print("💫 EmoSpark is here, sunshine.")
    print("Type how you feel. Type 'exit' to stop.\n")

    session_id = "session_1"

    while True:
        user_message = input("You > ").strip()

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit"}:
            print("🌷 EmoSpark: I’m here whenever you return. Take care, sunshine. 💛")
            break

        await emospark(user_message, session_id=session_id)

print("✅ chat_loop() ready")



messages = [
    "I'm feeling overwhelmed",
    "I'm tired",
    "I'm stressed about exams",
]

for msg in messages:
    print(f"You > {msg}")
    await emospark(msg)



#await chat_loop()
#If you remove # from chat you can start whole converconversation 



with open("submission.txt", "w") as f:
    f.write("EmoSpark Agent – Run Successful")





