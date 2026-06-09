

# ---------------------------
# 1️⃣ Imports & Setup
# ---------------------------
import datetime
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from termcolor import colored
from IPython.display import display, HTML

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# 2️⃣ Simulated LLM & API Functions
# ---------------------------
def llm_generate(prompt):
    """Simulate LLM response"""
    time.sleep(0.5)
    return f"LLM Response: {prompt}"

def get_cve_info(threat_name):
    """Simulate OpenAPI call for CVE info"""
    return f"CVE info for {threat_name}"

# ---------------------------
# 3️⃣ Memory Bank
# ---------------------------
class MemoryBank:
    def __init__(self):
        self.short_term = {}
        self.long_term = {"parsed_logs": [], "threats": [], "alerts": []}

    def save_short_term(self, key, value):
        self.short_term[key] = value

    def save_long_term(self, key, value):
        self.long_term[key].append(value)

    def get_long_term(self, key):
        return self.long_term.get(key, [])

memory = MemoryBank()

# ---------------------------
# 4️⃣ Log Parser Agent
# ---------------------------
class LogParserAgent:
    def __init__(self, logs):
        self.logs = logs

    def parse_logs(self):
        logging.info("Parsing logs...")
        parsed = []
        for log in self.logs:
            risk = "Low"
            if "FAILED LOGIN" in log:
                risk = "Medium"
            if "INTRUSION" in log:
                risk = "High"
            explanation = llm_generate(f"Explain threat level '{risk}' for log: {log}")
            parsed_entry = {"log": log, "risk": risk, "explanation": explanation}
            parsed.append(parsed_entry)
            memory.save_long_term("parsed_logs", parsed_entry)
        logging.info("Logs parsed successfully.")
        return parsed

# ---------------------------
# 5️⃣ Threat Analysis Agent
# ---------------------------
class ThreatAnalysisAgent:
    def __init__(self, parsed_logs):
        self.parsed_logs = parsed_logs

    def analyze_threats(self):
        logging.info("Analyzing threats...")
        for entry in self.parsed_logs:
            threat_name = entry["log"].split()[0]
            classification = llm_generate(f"Classify threat for log: {entry['log']}")
            severity = {"Low": 1, "Medium": 2, "High": 3}[entry["risk"]]
            cve_info = get_cve_info(threat_name)
            threat_entry = {
                "log": entry["log"],
                "classification": classification,
                "severity": severity,
                "cve_info": cve_info
            }
            memory.save_long_term("threats", threat_entry)
            time.sleep(0.1)
            logging.info(f"Analyzed log: {entry['log']}")
        logging.info("Threat analysis completed.")
        return memory.get_long_term("threats")

# ---------------------------
# 6️⃣ Alert & Response Agent
# ---------------------------
class AlertResponseAgent:
    def __init__(self, threats):
        self.threats = threats

    def generate_alert(self, threat):
        alert_msg = f"ALERT: {threat['classification']} detected in log '{threat['log']}'. Recommended action: Review CVE: {threat['cve_info']}"
        memory.save_long_term("alerts", alert_msg)
        logging.info(f"{alert_msg}")

    def process_alerts_parallel(self):
        logging.info("Generating alerts in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.generate_alert, self.threats)
        logging.info("All alerts generated.")

# ---------------------------
# 7️⃣ Color-coded Display Function (Terminal)
# ---------------------------
def display_colored_logs(parsed_logs, threats, alerts):
    print("\n=== PARSED LOGS ===")
    for log in parsed_logs:
        color = {"Low": "green", "Medium": "yellow", "High": "red"}[log["risk"]]
        print(colored(log, color))

    print("\n=== THREATS ===")
    for threat in threats:
        color = {1: "green", 2: "yellow", 3: "red"}[threat["severity"]]
        print(colored(threat, color))

    print("\n=== ALERTS ===")
    for alert in alerts:
        log_text = alert.split(" detected")[0]
        parsed_entry = next((l for l in parsed_logs if l["log"] in log_text), None)
        color = "green"
        if parsed_entry:
            color = {"Low": "green", "Medium": "yellow", "High": "red"}[parsed_entry["risk"]]
        print(colored(alert, color))

# ---------------------------
# 8️⃣ HTML Color
# ---------------------------
def display_html_logs(parsed_logs, threats, alerts):
    html_content = "<h3>PARSED LOGS</h3><ul>"
    for log in parsed_logs:
        color = {"Low": "green", "Medium": "orange", "High": "red"}[log["risk"]]
        html_content += f"<li style='color:{color}'>{log}</li>"
    html_content += "</ul>"

    html_content += "<h3>THREATS</h3><ul>"
    for threat in threats:
        color = {1: "green", 2: "orange", 3: "red"}[threat["severity"]]
        html_content += f"<li style='color:{color}'>{threat}</li>"
    html_content += "</ul>"

    html_content += "<h3>ALERTS</h3><ul>"
    for alert in alerts:
        log_text = alert.split(" detected")[0]
        parsed_entry = next((l for l in parsed_logs if l["log"] in log_text), None)
        color = "green"
        if parsed_entry:
            color = {"Low": "green", "Medium": "orange", "High": "red"}[parsed_entry["risk"]]
        html_content += f"<li style='color:{color}'>{alert}</li>"
    html_content += "</ul>"

    display(HTML(html_content))

# ---------------------------
# 9️⃣ Sample Logs Input
# ---------------------------
sample_logs = [
    "FAILED LOGIN from 192.168.1.10",
    "INTRUSION detected on port 22",
    "NORMAL ACTIVITY user1 login"
]

# ---------------------------
# 10️⃣ Run Agents Sequentially
# ---------------------------
parser_agent = LogParserAgent(sample_logs)
parsed_logs = parser_agent.parse_logs()

analysis_agent = ThreatAnalysisAgent(parsed_logs)
threats = analysis_agent.analyze_threats()

alert_agent = AlertResponseAgent(threats)
alert_agent.process_alerts_parallel()

# ---------------------------
# 11️⃣ Display Color-coded Output
# ---------------------------
# Terminal output
display_colored_logs(memory.get_long_term("parsed_logs"),
                     memory.get_long_term("threats"),
                     memory.get_long_term("alerts"))

# Notebook HTML output
display_html_logs(memory.get_long_term("parsed_logs"),
                  memory.get_long_term("threats"),
                  memory.get_long_term("alerts"))

# ---------------------------
# 12️⃣ Save Final Data to JSON
# ---------------------------
with open("cybersecurity_agents_output.json", "w") as f:
    json.dump(memory.long_term, f, indent=4)

