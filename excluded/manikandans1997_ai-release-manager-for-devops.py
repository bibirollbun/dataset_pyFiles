%pip install nest_asyncio asyncio  google.generativeai google-genai


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY1")
    os.environ["GOOGLE_API_KEY1"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import asyncio
from dataclasses import dataclass, field
from typing import List
import google.generativeai as genai

print("âœ… components imported successfully.")


if not GOOGLE_API_KEY:
    print("âš ï¸�  WARNING: GOOGLE_API_KEY not found in environment variables.")
    print("   Please set it using: export GOOGLE_API_KEY='your_key'")
    
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

async def ask_gemini(prompt: str) -> str:
    """Async wrapper for Gemini calls with basic error handling."""
    if not GOOGLE_API_KEY:
        return "[Mock Output] API Key missing - skipping AI call."
    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

# -----------------------------------------------------------
# 2. STATE & MEMORY (Requirement: Sessions/State)
# -----------------------------------------------------------

@dataclass
class ReleaseState:
    """Shared memory object passed between agents."""
    pr_description: str = ""
    code_changes: str = ""
    risk_level: str = ""
    security_issues: str = ""
    static_analysis: str = ""
    test_results: List[str] = field(default_factory=list)
    final_decision: str = ""

# Global state instance
state = ReleaseState()

# -----------------------------------------------------------
# 3. AGENT DEFINITIONS (Requirement: Multi-Agent System)
# -----------------------------------------------------------

class PRAnalysisAgent:
    """
    Type: Sequential Agent
    Role: First line of defense. Reads the PR text to gauge intent.
    """
    async def run(self, pr_text: str):
        state.pr_description = pr_text
        print("\n[1] ğŸ“‹ PR Analysis Agent: Analyzing intent...")

        prompt = f"""
        You are a DevOps PR Analysis Agent.
        Analyze this PR description and classify the Risk Level (Low/Medium/High).
        
        PR Text:
        {pr_text}
        
        Output format: Risk: [Level] - [One sentence summary]
        """
        result = await ask_gemini(prompt)
        state.risk_level = result
        print(f"    â†³ {result}")

class StaticReviewAgent:
    """
    Type: Parallel Agent A
    Role: Checks for messy code patterns concurrently with security.
    """
    async def run(self, code: str):
        print("    [2A] ğŸ‘“ Static Review Agent: Checking syntax & patterns...")
        prompt = f"""
        You are a Static Code Reviewer. Review this Python code for bugs or smells.
        Be extremely concise (bullet points).
        
        Code:
        {code}
        """
        return await ask_gemini(prompt)

class SecurityScanAgent:
    """
    Type: Parallel Agent B
    Role: Checks for vulnerabilities concurrently with static analysis.
    """
    async def run(self, code: str):
        print("    [2B] ğŸ›¡ï¸�  Security Agent: Scanning for vulnerabilities...")
        prompt = f"""
        You are a Security Scanner. Scan this code for secrets, injection, or unsafe logic.
        Return "SAFE" if no issues, or a warning list if issues found.
        
        Code:
        {code}
        """
        return await ask_gemini(prompt)

class TestRunnerAgent:
    """
    Type: Loop Agent
    Role: Runs a simulated loop of tests until completion or failure.
    """
    async def run(self):
        print("\n[3] ğŸ”„ Test Runner Agent: Initiating test loop...")
        
        test_suites = ["Unit Tests", "Integration Tests", "Performance Smoke Test"]
        
        for suite in test_suites:
            # Simulation: We ask Gemini to "simulate" the test result based on the risk level
            prompt = f"""
            You are a CI Simulator. 
            The current risk level is: {state.risk_level}.
            Simulate the result for '{suite}'.
            Return either "PASS" or "FAIL - [Reason]".
            """
            result = await ask_gemini(prompt)
            state.test_results.append(f"{suite}: {result}")
            print(f"    â†³ {suite} ... {result}")

class ReleaseDecisionAgent:
    """
    Type: Sequential Agent (Final)
    Role: Synthesizes all previous agent outputs to make a go/no-go decision.
    """
    async def run(self):
        print("\n[4] ğŸ§  Release Manager: Synthesizing final decision...")

        prompt = f"""
        You are the Chief Release Manager. Make a Go/No-Go decision based on these reports:

        1. Risk Assessment: {state.risk_level}
        2. Static Analysis: {state.static_analysis}
        3. Security Scan: {state.security_issues}
        4. Test Results: {state.test_results}

        Format:
        DECISION: [APPROVE / REJECT]
        REASON: [Summary]
        ACTION: [Next git command to run]
        """
        decision = await ask_gemini(prompt)
        state.final_decision = decision
        print(f"\n{'='*40}\nFINAL OUTPUT\n{'='*40}\n{decision}\n{'='*40}")

# -----------------------------------------------------------
# 4. ORCHESTRATOR
# -----------------------------------------------------------

class ReleaseManagerOrchestrator:
    async def run_pipeline(self, pr_text: str, code_text: str):
        print(f"ğŸš€ OPSMIND CI/CD BRAIN STARTING...")
        
        # 1. Sequential: PR Analysis
        await PRAnalysisAgent().run(pr_text)

        # 2. Parallel: Fan-out for Code & Security review
        print("\n[2] âš¡ Starting Parallel Analysis...")
        static_agent = StaticReviewAgent()
        security_agent = SecurityScanAgent()
        
        # asyncio.gather runs these at the same time
        results = await asyncio.gather(
            static_agent.run(code_text), 
            security_agent.run(code_text)
        )
        state.static_analysis = results[0]
        state.security_issues = results[1]
        print(f"    â†³ Static Analysis Complete")
        print(f"    â†³ Security Scan Complete")

        # 3. Loop: Run Tests
        await TestRunnerAgent().run()

        # 4. Sequential: Final Decision
        await ReleaseDecisionAgent().run()

# -----------------------------------------------------------
# 5. ENTRY POINT
# -----------------------------------------------------------

if __name__ == "__main__":
    scenarios = [
        {
            "name": "High Risk (Reject)",
            "pr": "Refactoring the login logic to allow unauthenticated access for debug mode.",
            "code": """
def login(user, password):
    # DEBUG BACKDOOR - REMOVE BEFORE PROD
    if user == "admin" and password == "debug_mode":
        return True

    if db.check_password(user, password):
        return True
    return False
"""
        },
        {
            "name": "Low Risk (Approve)",
            "pr": "Optimize database queries for faster report generation, no functional changes.",
            "code": """
def generate_report(user_id):
    data = db.fetch_user_data(user_id)
    summary = summarize(data)
    return summary
"""
        },
        {
            "name": "Medium Risk (Conditional Approval)",
            "pr": "Add feature to export user data in CSV, includes handling optional fields.",
            "code": """
def export_user_data(users):
    csv_data = "id,name,email\\n"
    for u in users:
        csv_data += f"{u.id},{u.name},{u.email}\\n"
    return csv_data
"""
        }
    ]

    orchestrator = ReleaseManagerOrchestrator()
    
    for scenario in scenarios:
        print(f"\n\n=== Running Scenario: {scenario['name']} ===\n")
        
        # Handle Jupyter/Interactive Loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(orchestrator.run_pipeline(scenario["pr"], scenario["code"]))
        else:
            asyncio.run(orchestrator.run_pipeline(scenario["pr"], scenario["code"]))


