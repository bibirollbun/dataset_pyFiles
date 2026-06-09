from kaggle_secrets import UserSecretsClient
import os
user_secrets = UserSecretsClient()
_api = user_secrets.get_secret("GOOGLE_API_KEY")
assert _api, "Add GOOGLE_API_KEY via Add-ons -> Secrets"
os.environ["GOOGLE_API_KEY"] = _api
print("Secret loaded")


# !pip install google-genai numpy --upgrade


import os
import numpy as np
from typing import List, Dict
from google import genai
from google.genai import types
from datetime import datetime

client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])


# Thresholds
SIMILARITY_THRESHOLD = 0.85  # If 85% similar, it's a duplicate/loop
MAX_JUNK_STREAK = 2          # Max low-value videos allowed in a row


# --- CONCEPT: SESSIONS & MEMORY ---
class SessionMemory:
    def __init__(self):
        self.history: List[Dict] = [] # Stores full metadata
        self.vector_store: List[List[float]] = [] # Stores embeddings
        self.streak_count = 0

    def add_video(self, metadata, embedding):
        self.history.append(metadata)
        self.vector_store.append(embedding)

    def get_last_embedding(self):
        if not self.vector_store:
            return None
        return self.vector_store[-1]

# --- HELPER: COSINE SIMILARITY ---
def cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0.0
    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    return dot_product / (norm_a * norm_b)


# --- CONCEPT : MULTI-AGENT SYSTEM ---

class AnalystAgent:
    """
    Role: Turn raw text into Mathematical Meaning (Embeddings) & Structured Data.
    """
    def analyze(self, video_title, video_desc):
        # 1. Generate Embedding for Similarity Check using Google GenAI
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=f"{video_title} {video_desc}"
        )
        embedding = response.embeddings[0].values

        # 2. Evaluate Educational Value using Gemini 2.5 Flash
        prompt = f"""
        Analyze this video content for a toddler. 
        Return strictly 'TRUE' if it teaches a hard skill (math, alphabet, science, social skills).
        Return 'FALSE' if it is passive entertainment, unboxing, or brain-rot.
        
        Title: {video_title}
        Description: {video_desc}
        """
        
        eval_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        is_educational = "TRUE" in eval_resp.text.upper()
        
        return embedding, is_educational


class GuardianAgent:
    """
    Role: The Judge. Checks Memory and enforces rules.
    """
    def __init__(self, memory: SessionMemory):
        self.memory = memory

    def decide(self, title, embedding, is_educational):
        decision_log = {}
        
        # Check 1: Is it educational?
        if is_educational:
            self.memory.streak_count = 0 # Reset streak
            self.memory.add_video({"title": title, "type": "EDU"}, embedding)
            decision_log = {"action": "ALLOW", "reason": "Educational Content"}
            return decision_log

        # Check 2: Is it a "Zombie Loop"? (Similarity Check)
        last_vec = self.memory.get_last_embedding()
        similarity = cosine_similarity(embedding, last_vec)
        
        decision_log["similarity_score"] = round(similarity, 2)
        
        if similarity > SIMILARITY_THRESHOLD:
            # Content is basically the same as the last video
            self.memory.streak_count += 1
        else:
            # Content is junk, but different junk. Increase streak anyway.
            self.memory.streak_count += 1

        # Final Decision
        if self.memory.streak_count > MAX_JUNK_STREAK:
            decision_log["action"] = "BLOCK_AND_SWAP"
            decision_log["reason"] = f"Zombie Loop Detected (Streak: {self.memory.streak_count})"
            # We do NOT add blocked videos to memory
        else:
            decision_log["action"] = "ALLOW"
            decision_log["reason"] = "Entertainment Quota Available"
            self.memory.add_video({"title": title, "type": "FUN"}, embedding)
            
        return decision_log


# --- CONCEPT : OBSERVABILITY ---
def log_observability(trace_id, inputs, outputs):
    """
    Simulates sending logs to a dashboard (e.g., LangSmith or Datadog)
    """
    print(f"\n[OBSERVABILITY LOG ID: {trace_id}]")
    print(f"timestamp: {datetime.now().isoformat()}")
    print(f"inputs: {inputs}")
    print(f"decision: {outputs}")
    print("-" * 40)


# --- MAIN ORCHESTRATION (SIMULATING DEPLOYMENT) ---

def main():
    # Initialize System
    memory = SessionMemory()
    analyst = AnalystAgent()
    guardian = GuardianAgent(memory)

    # Simulation: A child watching a playlist
    playlist = [
        ("Learn to Count to 10", "Educational math video for toddlers counting apples."),
        ("Surprise Egg Unboxing Red", "Opening 50 red eggs with plastic toys."),
        ("Surprise Egg Unboxing Blue", "Opening blue eggs with toys inside."), # Similar to prev
        ("Surprise Egg Unboxing Giant", "Opening a giant egg."),               # Loop detected here!
        ("Learn Colors with Fruits", "Learning red, blue, and green with fruits.")
    ]

    print("ğŸš€ KidGuard AI Agent Started...\n")

    for i, (title, desc) in enumerate(playlist):
        trace_id = f"vid_{i}"
        
        # 1. Analyst Agent Work
        embedding, is_edu = analyst.analyze(title, desc)
        
        # 2. Guardian Agent Work
        decision = guardian.decide(title, embedding, is_edu)
        
        # 3. Observability Logging
        log_inputs = {"title": title, "is_educational": is_edu}
        log_observability(trace_id, log_inputs, decision)

        # 4. Action
        if decision["action"] == "BLOCK_AND_SWAP":
            print(f"ğŸ›‘ INTERVENTION: '{title}' was blocked!")
            print("âœ… SWAPPING CONTENT: Playing 'Solar System for Kids' instead.\n")
            # In a real app, we would trigger the swap logic here
            # Reset streak after forced intervention
            memory.streak_count = 0 
        else:
            print(f"â–¶ï¸� PLAYING: {title}\n")

if __name__ == "__main__":
    main()

