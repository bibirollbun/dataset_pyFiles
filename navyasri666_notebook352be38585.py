import os
import sys
import logging
import json
import time
import google.generativeai as genai
from typing import Dict, Any

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SubZero] - %(levelname)s - %(message)s')
logger = logging.getLogger("SubZero_Core")

class AgentError(Exception):
    pass

# --- 1. SMART MODEL SELECTION ---
def get_best_model_name(api_key, keyword="flash"):
    """
    Automatically finds a working model name for the user's API key.
    """
    # Fallback to the most standard stable model
    default_fallback = "models/gemini-1.5-flash"
    
    try:
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())
        
        # Filter 1: Must support generateContent
        valid_models = [
            m.name for m in all_models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Filter 2: Exclude 'learnlm' or 'experimental' models that tend to 404 or be restricted
        # We want STABLE models for the competition
        safe_models = [
            m for m in valid_models 
            if "learnlm" not in m and "experimental" not in m
        ]
        
        # Strategy 1: Look for "gemini-1.5-flash" specific versions (002, 001)
        # These are production-ready and high-rate-limit
        matches = [m for m in safe_models if "gemini-1.5-flash" in m]
        
        if matches:
            # Sort to get the latest (usually 002 comes after 001)
            matches.sort(reverse=True) 
            return matches[0]
        
        # Strategy 2: Look for any "flash" model if 1.5-flash isn't found
        matches = [m for m in safe_models if keyword in m]
        if matches:
            matches.sort(reverse=True)
            return matches[0]

        # Strategy 3: Last resort, anything available
        if safe_models:
            return safe_models[0]
            
    except Exception as e:
        logger.warning(f"Could not list models ({e}). Using default fallback.")

    return default_fallback

# --- 2. TOOLS ---
def search_cancellation_policy(company_name: str):
    """Retrieves the cancellation and refund policy for a specific company."""
    logger.info(f"ğŸ› ï¸� [TOOL USE] Searching policy for: {company_name}")
    
    knowledge_base = {
        "netflix": "Netflix Policy: No refunds for partial months. Access continues until the billing cycle ends. Cancellation must be done via the Account page.",
        "adobe": "Adobe Policy: Full refund if cancelled within 14 days of initial order. After 14 days, 50% cancellation fee of remaining contract obligation applies.",
        "gym shark": "Gym Shark Policy: 30-day written notice required. No partial refunds.",
        "spotify": "Spotify Policy: Reverts to Free account at end of billing cycle. No refunds for already paid months.",
        "unknown": "General Consumer Law: If services were not rendered or were defective, you may be entitled to a chargeback. Recommend contacting support immediately."
    }
    
    key = company_name.lower().strip()
    for k, v in knowledge_base.items():
        if k in key or key in k:
            return v
    return knowledge_base["unknown"]

subzero_tools = [search_cancellation_policy]

# --- 3. AGENT PIPELINE ---
class SubZeroPipeline:
    def __init__(self, api_key: str, worker_id: str, writer_id: str):
        if not api_key:
            raise ValueError("API Key must be provided.")
        
        genai.configure(api_key=api_key)
        self.worker_id = worker_id
        self.writer_id = writer_id

        try:
            # Agent A: The Auditor (Fast, JSON extraction, NO TOOLS)
            self.auditor_model = genai.GenerativeModel(model_name=worker_id)

            # Agent B: The Researcher (Fast, Uses Tools)
            self.researcher_model = genai.GenerativeModel(model_name=worker_id, tools=subzero_tools)
            
            # Agent C: The Scribe (Creative)
            self.writer_model = genai.GenerativeModel(model_name=writer_id)
            
            logger.info(f"Pipeline initialized with: {worker_id} & {writer_id}")
        except Exception as e:
            logger.error(f"Error initializing models.")
            raise e

    def _retry_api_call(self, func, *args, **kwargs):
        """Helper to retry API calls when Rate Limits (429) are hit."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Catch Quota/Rate Limit errors
                if "429" in str(e) or "Quota" in str(e) or "quota" in str(e):
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"âš ï¸� Rate Limit Hit. Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise AgentError("Max retries exceeded due to API Rate Limits.")

    def run(self, user_complaint: str) -> Dict[str, Any]:
        """Executes the sequential multi-agent workflow."""
        logger.info("ğŸŸ¢ --- STARTING SUB-ZERO PIPELINE ---")
        
        results = {
            "original_input": user_complaint,
            "audit": None,
            "policy": None,
            "final_draft": None
        }

        try:
            # Step 1: Audit
            logger.info(f"Step 1: Auditor Agent running...")
            audit_prompt = f"""
            Analyze the following consumer complaint. 
            Extract the 'Company Name' and 'Amount'.
            Return the result as JSON.
            Complaint: "{user_complaint}"
            """
            
            # Use retry wrapper
            audit_resp = self._retry_api_call(
                self.auditor_model.generate_content,
                audit_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            audit_data = json.loads(audit_resp.text)
            results['audit'] = audit_data
            
            # PAUSE to be polite to the API
            time.sleep(1)

            # Step 2: Research
            logger.info("Step 2: Researcher Agent running...")
            company_name = audit_data.get("Company Name", "Unknown")
            
            research_chat = self.researcher_model.start_chat(enable_automatic_function_calling=True)
            research_prompt = f"""
            I have a dispute with {company_name}. 
            Use the `search_cancellation_policy` tool to find their specific refund terms.
            Summarize the policy findings for me.
            """
            
            # Use retry wrapper
            research_resp = self._retry_api_call(
                research_chat.send_message,
                research_prompt
            )
            results['policy'] = research_resp.text

            # PAUSE to be polite to the API
            time.sleep(1)

            # Step 3: Scribe
            logger.info(f"Step 3: Scribe Agent running...")
            scribe_prompt = f"""
            You are a professional legal concierge. Draft a formal cancellation and refund request email.
            
            CONTEXT:
            - User's Complaint: {user_complaint}
            - Extracted Facts: {audit_data}
            - Company Policy Found: {results['policy']}
            
            INSTRUCTIONS:
            - If the policy allows a refund (e.g. within 14 days), demand it citing the policy.
            - If the policy is strict, argue based on "accidental purchase".
            - Tone: Firm, Professional, Polite.
            """
            
            # Use retry wrapper
            scribe_resp = self._retry_api_call(
                self.writer_model.generate_content,
                scribe_prompt
            )
            results['final_draft'] = scribe_resp.text
            
            logger.info("ğŸ�� Pipeline executed successfully.")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise AgentError(f"SubZero crashed: {e}")

# --- 4. EXECUTION ---
def get_api_key():
    """Retrieves API Key from Kaggle Secrets, Colab UserData, or Environment."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key: return api_key
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("GOOGLE_API_KEY")
    except: pass
        
    try:
        from google.colab import userdata
        return userdata.get('GOOGLE_API_KEY')
    except: pass
    return None

def main():
    print("ğŸ¤– Initializing SubZero Protocol (v5 - Flash Only Mode)...")
    
    # 1. Get the Key
    api_key = get_api_key()
    if not api_key:
        print("â�Œ CRITICAL ERROR: API Key not found.")
        print("INSTRUCTIONS: Add 'GOOGLE_API_KEY' to your Kaggle Secrets.")
        return

    # 2. Auto-Detect Models
    print("ğŸ”� Detecting available models for your API Key...")
    
    # TARGET: gemini-2.5-flash (or best available flash) for BOTH agents
    flash_model = get_best_model_name(api_key, "flash")
    
    print(f"âœ… Selected Model (All Agents): {flash_model}")

    # 3. Initialize the Agent
    try:
        # We pass the same flash model for both worker and writer to avoid rate limits
        agent_system = SubZeroPipeline(api_key, flash_model, flash_model)
    except Exception as e:
        print(f"â�Œ Failed to initialize agents: {e}")
        return

    # 4. Define Real-World Test Cases
    test_cases = [
        "I noticed a $54 charge from Adobe on my statement today. I just bought this subscription 3 days ago and I don't want it.",
        "I want to cancel my Gym Shark membership. I've been a member for 2 months but I'm moving."
    ]

    # 5. Run the Pipeline
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"ğŸ“¥ TEST CASE {i}: {case}")
        print(f"{'='*40}")
        
        try:
            output = agent_system.run(case)
            
            print(f"\nğŸ”� [AUDIT DATA] (Agent A)")
            print(output['audit'])
            
            print(f"\nğŸ“š [POLICY FOUND] (Agent B + Tool)")
            print(output['policy'])
            
            print(f"\nâœ‰ï¸� [FINAL DRAFT] (Agent C)")
            print(f"{'-'*20}")
            print(output['final_draft'])
            print(f"{'-'*20}")
            
            # Moderate pause between test cases
            time.sleep(3)
            
        except Exception as e:
            print(f"â�Œ Execution failed: {e}")

if __name__ == "__main__":
    main()

