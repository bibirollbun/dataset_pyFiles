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


!pip install google-generativeai



import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Load API key
key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=key)

# Pick the model
model = genai.GenerativeModel("gemini-2.5-flash")



class GeminiAgent:
    def __init__(self, name, model="gemini-2.5-flash"):
        self.name = name
        self.model = genai.GenerativeModel(model)
        self.memory = []

    def run(self, input_data):
        # Save memory
        self.memory.append(input_data)

        response = self.model.generate_content(input_data)
        return response.text



log_agent = GeminiAgent("LogAnalyzer")
threat_agent = GeminiAgent("ThreatClassifier")
mitigation_agent = GeminiAgent("MitigationAdvisor")
report_agent = GeminiAgent("ReportGenerator")



log_analyzer_prompt = """
You are a cybersecurity log analysis agent.
Task: Extract ONLY suspicious or abnormal events.
Return the result as a bullet list.

Here are the logs:
"""

threat_classifier_prompt = """
You are a cybersecurity threat classifier.

Task:
For each suspicious event, classify:
- threat type
- severity (Low/Medium/High)
Return JSON list:
[
 {"event": "...", "type": "...", "severity": "..."},
]
Events:
"""

mitigation_prompt = """
You are a cybersecurity mitigation advisor.
Given classified threats, suggest ONE mitigation per threat.
Return bullet list of actions in the same order.

Threats:
"""

report_prompt = """
You are a cybersecurity incident report generator.
Write a professional report using:
- events
- threat types
- severities
- mitigation actions

Include:
- report date
- summary
- detailed findings
- recommended actions
Format it cleanly.

Data:
"""



# -----------------------------
# Mock Logs Block
# -----------------------------
# These are the sample logs that our CyberSentinel agents will analyze.
# No user input required; you can modify/add logs as needed.

mock_logs = [
    "2025-11-27 08:01: Failed login from 192.168.1.105",
    "2025-11-27 08:03: Access to sensitive file 'armdoc.xlsx' outside work hours",
    "2025-11-27 08:05: Failed login from 192.168.1.105",
    "2025-11-27 08:10: Suspicious outbound connection to 203.0.113.45",
    "2025-11-27 08:15: Unauthorized USB device connected",
    "2025-11-27 08:20: Multiple failed password attempts for admin account"
]

print("âœ… Mock logs loaded:")
for log in mock_logs:
    print("-", log)




joined_logs = "\n".join(mock_logs)

# Step 1 â€” Log Analyzer
suspicious_events = log_agent.run(log_analyzer_prompt + joined_logs)
print("ğŸ”� Suspicious Events:\n", suspicious_events, "\n")

# Step 2 â€” Threat Classification
classified_threats = threat_agent.run(threat_classifier_prompt + suspicious_events)
print("âš ï¸� Classified Threats:\n", classified_threats, "\n")

# Step 3 â€” Mitigation Advisor
mitigation_actions = mitigation_agent.run(mitigation_prompt + classified_threats)
print("ğŸ›¡ Recommended Mitigation:\n", mitigation_actions, "\n")

# Step 4 â€” Incident Report
final_report = report_agent.run(
    report_prompt + 
    "\nTHREATS:\n" + classified_threats + 
    "\nMITIGATION:\n" + mitigation_actions
)

print("ğŸ“„ FINAL INCIDENT REPORT:\n")
print(final_report)



print("ğŸ§  LogAnalyzer Memory:", log_agent.memory, "\n")
print("ğŸ§  ThreatClassifier Memory:", threat_agent.memory, "\n")
print("ğŸ§  MitigationAdvisor Memory:", mitigation_agent.memory, "\n")
print("ğŸ§  ReportGenerator Memory:", report_agent.memory, "\n")


