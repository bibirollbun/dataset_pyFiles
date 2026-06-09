# --- STEP 1: SETUP ---
!pip install -q -U google-generativeai

import google.generativeai as genai
import time
import json
from datetime import datetime

# 1. SETUP API KEY (Using Kaggle Secrets)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
my_secret = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=my_secret)

print("âœ… CyberSentinel Environment Ready.")


for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)


# --- STEP 2: DEFINE TOOLS & MEMORY ---

# [Requirement 1: Tools]
# A mock tool simulating a Threat Intelligence API (like VirusTotal)
def check_threat_intel(ip_address):
    """
    Queries a mock Threat Intelligence database for IP reputation.
    """
    # Simulated Data for Demonstration
    mock_database = {
        "192.168.1.50": {"risk_score": 0, "category": "Safe", "owner": "Local Network"},
        "203.0.113.5":  {"risk_score": 85, "category": "Malicious", "threat": "Botnet Command & Control"},
        "198.51.100.23": {"risk_score": 45, "category": "Suspicious", "threat": "Known VPN Exit Node"}
    }
    
    result = mock_database.get(ip_address, {"risk_score": 10, "category": "Unknown", "threat": "None"})
    return result

# [Requirement 2: Memory/State]
# Keeps track of IPs we have already analyzed to avoid duplicate work
class IncidentMemory:
    def __init__(self):
        self.analyzed_ips = {} # Stores {ip: verdict}

    def has_analyzed(self, ip):
        return ip in self.analyzed_ips

    def store_verdict(self, ip, verdict):
        self.analyzed_ips[ip] = verdict

# [Requirement 3: Observability]
# Structured logging for security auditing
def audit_log(phase, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"ğŸ›¡ï¸� [SOC-AUDIT] {timestamp} | {phase.upper()} | {message}")

# --- STEP 3: THE AGENT CLASS ---

class CyberSentinelAgent:
    def __init__(self):
        self.memory = IncidentMemory()
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
    def investigate_ip(self, ip_address):
        audit_log("START", f"Received request to investigate IP: {ip_address}")
        
        # 1. MEMORY CHECK
        if self.memory.has_analyzed(ip_address):
            audit_log("MEMORY HIT", f"IP {ip_address} was already analyzed. Retrieving cached verdict.")
            return f"Skipping analysis. Previous Verdict: {self.memory.analyzed_ips[ip_address]}"
        
        # 2. TOOL USAGE
        audit_log("TOOL USE", f"Querying Threat Intel Database for {ip_address}...")
        intel_data = check_threat_intel(ip_address)
        audit_log("DATA RX", f"Intelligence received: {intel_data}")
        
        # 3. REASONING (LLM)
        # We feed the tool data into the LLM to generate a human-readable report
        prompt = f"""
        You are CyberSentinel, a Tier 1 SOC Analyst.
        
        TASK: Analyze the following network intelligence and provide a Verdict.
        
        TARGET IP: {ip_address}
        INTEL DATA: {intel_data}
        
        RULES:
        - If Risk Score > 80: Verdict is CRITICAL. Recommend blocking firewall immediately.
        - If Risk Score > 40: Verdict is WARNING. Recommend monitoring traffic.
        - If Risk Score < 40: Verdict is SAFE. No action needed.
        
        OUTPUT FORMAT:
        Verdict: [CRITICAL/WARNING/SAFE]
        Analysis: [One sentence explanation]
        Action: [Recommended technical action]
        """
        
        response = self.model.generate_content(prompt)
        final_report = response.text.strip()
        
        # 4. UPDATE MEMORY
        self.memory.store_verdict(ip_address, final_report)
        audit_log("COMPLETE", "Analysis finished and stored in memory.")
        
        return final_report

print("âœ… CyberSentinel Agent loaded.")


# --- STEP 4: EXECUTION ---

sentinel = CyberSentinelAgent()

print("\n--- ğŸš¨ INCIDENT 1: Suspicious Botnet Traffic ---")
# We simulate an alert coming in for a known bad IP
report1 = sentinel.investigate_ip("203.0.113.5")
print(f"\nğŸ“„ REPORT:\n{report1}\n")
time.sleep(1)

print("\n--- ğŸŸ¢ INCIDENT 2: Local Traffic ---")
# We simulate a safe local IP
report2 = sentinel.investigate_ip("192.168.1.50")
print(f"\nğŸ“„ REPORT:\n{report2}\n")
time.sleep(1)

print("\n--- ğŸ§  INCIDENT 3: Duplicate Check (Memory Test) ---")
# We ask about the FIRST IP again. The agent should NOT run the tool, but check memory.
report3 = sentinel.investigate_ip("203.0.113.5")
print(f"\nğŸ“„ REPORT:\n{report3}\n")

