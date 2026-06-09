import os
from kaggle_secrets import UserSecretsClient

# Load Google API key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API key loaded successfully")
except Exception as e:
    print("â�Œ Failed to load API Key:", e)



!pip install -q google-adk[a2a]

from google import genai
import json
from pathlib import Path

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
print("âœ… Libraries imported")



LOG_FILE = "/kaggle/input/input-file-for-ai-project/server.log"
THREAD_FILE = "/kaggle/input/input-file-for-ai-project/threaddump.txt"
GC_FILE = "/kaggle/input/input-file-for-ai-project/gc.log"

print("ğŸ“‚ Log File:", LOG_FILE)
print("ğŸ“‚ Thread Dump File:", THREAD_FILE)
print("ğŸ“‚ GC File:", GC_FILE)



def load_text(file_path):
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")



def ask_model(prompt, text):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, text]
    )
    return response.text



PROMPT_LOG = "You are an expert production support engineer. Analyze the logs and summarize errors, warnings, latency issues, and DB failures. Provide RCA + recommended fixes."
PROMPT_THREAD = "You are a JVM expert. Analyze the thread dump and detect deadlocks, blocks, CPU spikes, RUNNABLE loops, and GC threads. Provide RCA."
PROMPT_GC = "You are a Java GC specialist. Analyze the GC log for memory leaks, high pause times, full GC frequency, and heap usage patterns. Provide RCA + fixes."



def run_agent():
    results = {}

    print("ğŸ”� Analyzing server.logâ€¦")
    results["log_analysis"] = ask_model(PROMPT_LOG, load_text(LOG_FILE))

    print("ğŸ”� Analyzing thread dumpâ€¦")
    results["thread_analysis"] = ask_model(PROMPT_THREAD, load_text(THREAD_FILE))

    print("ğŸ”� Analyzing gc.logâ€¦")
    results["gc_analysis"] = ask_model(PROMPT_GC, load_text(GC_FILE))

    return results



result = run_agent()

print("\nğŸ�‰ FINAL OUTPUT\n")
print(json.dumps(result, indent=2))


