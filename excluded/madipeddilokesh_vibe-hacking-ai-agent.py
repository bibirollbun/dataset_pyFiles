"""
Vibe Hacking AI Agent - Complete Capstone Code

Covers:
1. Environment Setup
2. API Configuration
3. Tool Functions
4. Function Declarations
5. Memory System
6. Logging System
7. Main Agent Class
8. Agent Testing Utilities
9. Statistics Dashboard
10. Export Functions
11. Reset Memory & Stats
12. Live Agent Demo
13. Export & Reset Demo
"""

import sys
import os
import time
import random
import logging
from collections import defaultdict
from getpass import getpass
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import track
import pprint

console = Console()

# ---------------------------------
# 1. ENVIRONMENT SETUP
# ---------------------------------

def install_package(package_name):
    """ Install package using pip """
    try:
        print(f"Installing package: {package_name} ...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}.\n")
    except Exception as e:
        print(f"Failed to install {package_name}, error: {e}")
        sys.exit(1)

def environment_setup():
    """ Programmatic environment setup """
    packages = ["rich"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"Package already installed: {pkg}")
        except ImportError:
            install_package(pkg)
    print("Environment setup 완료!\n")


# ---------------------------------
# 2. API CONFIGURATION
# ---------------------------------

CONFIG_FILE = ".env"

def save_api_key(key_name, key_value):
    """ Save or update key in .env file """
    lines = []
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            lines = f.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key_name}="):
            lines[i] = f"{key_name}={key_value}\n"
            updated = True
            break
    if not updated:
        lines.append(f"{key_name}={key_value}\n")
    with open(CONFIG_FILE, "w") as f:
        f.writelines(lines)

def configure_api_key(key_name, description):
    console.print(f"\nConfigure {description}:", style="bold cyan")
    console.print("(Input will be hidden for security)", style="italic yellow")
    key_value = getpass(f"Enter your {description} key: ").strip()
    if key_value:
        save_api_key(key_name, key_value)
        console.print(f"{description} key saved successfully.", style="green")
    else:
        console.print(f"No key entered. Skipping {description} configuration.", style="red")

def api_configuration():
    console.print("=== API Configuration Setup ===", style="bold magenta")
    configure_api_key("GOOGLE_GEMINI_API_KEY", "Google Gemini API")
    configure_api_key("SHODAN_API_KEY", "Shodan API")
    console.print(f"Keys saved in '{CONFIG_FILE}'.\nMake sure to add it to .gitignore.", style="italic")

# ---------------------------------
# 3. TOOL FUNCTIONS (simulation)
# ---------------------------------

def call_gemini(prompt: str) -> str:
    """ Simulate Gemini LLM responses """
    responses = {
        "generate_report": "AI-generated summary with vulnerabilities and mitigation recommendations.",
        "exploit_decision": random.choice(["proceed", "skip"]),
        "threat_explanation": "Simulates ransomware attack behavior for demo."
    }
    if "report" in prompt.lower():
        return responses["generate_report"]
    elif "exploit" in prompt.lower():
        return responses["exploit_decision"]
    elif "threat" in prompt.lower():
        return responses["threat_explanation"]
    return "Gemini response to prompt."

def perform_whois_lookup(target):
    console.print(f"Performing WHOIS lookup for {target} (simulated)...", style="blue")
    time.sleep(1)
    return f"WHOIS info: Owner example for {target}"

def perform_shodan_lookup(target):
    console.print(f"Querying Shodan for {target} (simulated)...", style="blue")
    time.sleep(1)
    return f"Shodan data summary for {target}"

def simulate_vulnerability_scan(target):
    console.print(f"Scanning vulnerabilities on {target} (simulated)...", style="yellow")
    time.sleep(2)
    vulns = [
        "Open port 80 outdated HTTP server",
        "SQL Injection in login",
        "Default password on IoT device",
        "Weak SSL/TLS",
        "Cross-site scripting (XSS)"
    ]
    found = random.sample(vulns, random.randint(1,3))
    console.print(f"Found {len(found)} vulnerabilities on {target}", style="green")
    return found

def attempt_exploit(target, vuln):
    console.print(f"Deciding to exploit {vuln} on {target}...", style="magenta")
    decision = call_gemini(f"Exploit decision for {vuln} on {target}")
    if decision == "proceed":
        console.print(f"Exploiting {vuln} on {target} (simulated)...", style="magenta")
        time.sleep(2)
        result = random.choice([True, False])
        console.print(f"Exploit {'succeeded' if result else 'failed'}!", style="green" if result else "yellow")
        return result
    else:
        console.print(f"Skipping exploit {vuln} on {target}.", style="yellow")
        return False

def phishing_simulation(target):
    console.print(f"Simulating phishing attack on {target}...", style="cyan")
    time.sleep(2)
    console.print("Phishing email sent (simulated).")

def ransomware_simulation(target):
    console.print(f"Simulating ransomware attack on {target}...", style="cyan")
    console.print(call_gemini("Explain ransomware"), style="italic")
    time.sleep(3)
    console.print("Files encrypted (simulation).")

def data_exfiltration_simulation(target):
    console.print(f"Simulating data exfiltration on {target}...", style="cyan")
    time.sleep(2)
    console.print("Sensitive data exfiltrated (simulated).")

THREAT_PROFILES = {
    "phishing": phishing_simulation,
    "ransomware": ransomware_simulation,
    "data_exfiltration": data_exfiltration_simulation,
}

# ---------------------------------
# 4. FUNCTION DECLARATIONS
# ---------------------------------

def run_reconnaissance(target_type, targets):
    results = {}
    for target in track(targets, description="Reconnaissance in progress..."):
        data = {}
        if target_type in ["web", "both"]:
            data["whois"] = perform_whois_lookup(target)
        if target_type in ["iot", "both"]:
            data["shodan"] = perform_shodan_lookup(target)
        results[target] = data
    return results

def run_vulnerability_scanning(targets):
    results = {}
    for target in targets:
        results[target] = simulate_vulnerability_scan(target)
    return results

def run_exploitation(targets, vuln_data):
    results = {}
    for target in targets:
        exploits = {}
        vulns = vuln_data.get(target, [])
        for vuln in vulns:
            exploits[vuln] = attempt_exploit(target, vuln)
        results[target] = exploits
    return results

def run_threat_simulation(targets, threat_names):
    results = {}
    for target in targets:
        results[target] = []
        for threat in threat_names:
            THREAT_PROFILES[threat](target)
            results[target].append(threat)
    return results

def generate_final_report(targets, recon_data, vuln_data, exploit_data, threat_data):
    report = "=== Vibe Hacking AI Agent Report ===\n\n"
    for target in targets:
        report += f"Target: {target}\n"
        report += "Reconnaissance Data:\n"
        for k,v in recon_data.get(target, {}).items():
            report += f"  - {k}: {v}\n"
        report += "Vulnerabilities:\n"
        for vuln in vuln_data.get(target, []):
            report += f"  - {vuln}\n"
        report += "Exploitation:\n"
        exploits = exploit_data.get(target, {})
        if exploits:
            for vuln, succ in exploits.items():
                status = "Succeeded" if succ else "Failed or Skipped"
                report += f"  - {vuln}: {status}\n"
        else:
            report += "  - None\n"
        report += "Threat Simulations:\n"
        threats = threat_data.get(target, [])
        if threats:
            for t in threats:
                report += f"  - {t}\n"
        else:
            report += "  - None\n"
        report += "\n"
    report += call_gemini("Generate detailed mitigation report.")
    return report


# ---------------------------------
# 5. MEMORY SYSTEM
# ---------------------------------

class MemoryManager:
    def __init__(self):
        self._memory = defaultdict(dict)

    def store(self, module_name, target, **kwargs):
        if target not in self._memory[module_name]:
            self._memory[module_name][target] = {}
        self._memory[module_name][target].update(kwargs)

    def retrieve(self, module_name, target):
        return self._memory.get(module_name, {}).get(target, {})

    def has_data(self, module_name, target, key):
        return key in self._memory.get(module_name, {}).get(target, {})

    def clear(self):
        self._memory.clear()

    def dump(self):
        print("=== Memory Dump ===")
        pprint.pprint(dict(self._memory))


# ---------------------------------
# 6. LOGGING SYSTEM
# ---------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VibeAgentLogger")


# ---------------------------------
# 7. MAIN AGENT CLASS
# ---------------------------------

class VibeHackingAgent:
    def __init__(self):
        self.memory = MemoryManager()
        self.logger = logger
        self.console = console

    def run_recon(self, target_type, targets):
        self.logger.info("Running reconnaissance...")
        data = run_reconnaissance(target_type, targets)
        for target, info in data.items():
            self.memory.store("recon", target, **info)
        return data

    def run_vuln_scan(self, targets):
        self.logger.info("Running vulnerability scan...")
        data = run_vulnerability_scanning(targets)
        for target, vulns in data.items():
            self.memory.store("scan", target, vulnerabilities=vulns)
        return data

    def run_exploit(self, targets):
        self.logger.info("Running exploits...")
        vuln_data = {t: self.memory.retrieve("scan", t).get("vulnerabilities", []) for t in targets}
        results = run_exploitation(targets, vuln_data)
        for target, exploits in results.items():
            self.memory.store("exploit", target, exploits=exploits)
        return results

    def run_threat_sim(self, targets, threat_profiles):
        self.logger.info("Running threat simulations...")
        results = run_threat_simulation(targets, threat_profiles)
        for target, threats in results.items():
            self.memory.store("threat", target, threats=threats)
        return results

    def generate_report(self, targets):
        self.logger.info("Generating final report...")
        recon = {t: self.memory.retrieve("recon", t) for t in targets}
        scan = {t: self.memory.retrieve("scan", t).get("vulnerabilities", []) for t in targets}
        exploit = {t: self.memory.retrieve("exploit", t).get("exploits", {}) for t in targets}
        threat = {t: self.memory.retrieve("threat", t).get("threats", []) for t in targets}
        return generate_final_report(targets, recon, scan, exploit, threat)

    def reset(self):
        self.logger.info("Clearing memory...")
        self.memory.clear()

# ---------------------------------
# 8. TEST THE AGENT
# ---------------------------------

def test_agent():
    agent = VibeHackingAgent()
    agent.console.print("[bold green]Starting agent test...[/bold green]\n")

    targets = ["example.com", "192.168.0.10"]
    target_type = "both"

    agent.console.print("Reconnaissance Phase")
    recon = agent.run_recon(target_type, targets)
    agent.console.print(f"Recon results: {recon}\n")

    agent.console.print("Vulnerability Scanning Phase")
    vulns = agent.run_vuln_scan(targets)
    agent.console.print(f"Vulnerabilities found: {vulns}\n")

    agent.console.print("Exploitation Phase")
    exploits = agent.run_exploit(targets)
    agent.console.print(f"Exploitation results: {exploits}\n")

    agent.console.print("Threat Simulation Phase")
    threats = agent.run_threat_sim(targets, ["phishing", "ransomware"])
    agent.console.print(f"Threat simulation results: {threats}\n")

    report = agent.generate_report(targets)
    agent.console.print("[bold cyan]Final Report:[/bold cyan]\n")
    agent.console.print(report)

# ---------------------------------
# 9. STATISTICS DASHBOARD
# ---------------------------------

class StatisticsDashboard:
    def __init__(self):
        self.vuln_count = 0
        self.exploit_attempts = 0
        self.exploits_succeeded = 0
        self.threats_run = 0

    def update(self, vuln_data, exploit_data, threat_data):
        self.vuln_count = sum(len(v) for v in vuln_data.values())
        self.exploit_attempts = sum(len(v) for v in exploit_data.values())
        self.exploits_succeeded = sum(
            sum(1 for success in exploits.values() if success)
            for exploits in exploit_data.values()
        )
        self.threats_run = sum(len(t) for t in threat_data.values())

    def display(self):
        console.print("[bold underline]Statistics Dashboard[/bold underline]")
        console.print(f"Vulnerabilities Found: {self.vuln_count}")
        console.print(f"Exploitation Attempts: {self.exploit_attempts}")
        console.print(f"Successful Exploits: {self.exploits_succeeded}")
        console.print(f"Threat Simulations Run: {self.threats_run}")

# ---------------------------------
# 10. EXPORT FUNCTIONS
# ---------------------------------

def export_report(report_text, filename="vibe_hacking_report.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)
    console.print(f"Report exported to {filename}", style="green")

def export_statistics(stats_dashboard, filename="vibe_hacking_stats.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Vibe Hacking AI Agent Statistics ===\n\n")
        f.write(f"Vulnerabilities Found: {stats_dashboard.vuln_count}\n")
        f.write(f"Exploitation Attempts: {stats_dashboard.exploit_attempts}\n")
        f.write(f"Successful Exploits: {stats_dashboard.exploits_succeeded}\n")
        f.write(f"Threats Simulated: {stats_dashboard.threats_run}\n")
    console.print(f"Statistics exported to {filename}", style="green")

# ---------------------------------
# 11. RESET AGENT MEMORY & STATISTICS
# ---------------------------------

def reset_agent(agent, stats_dashboard):
    agent.reset()
    stats_dashboard.vuln_count = 0
    stats_dashboard.exploit_attempts = 0
    stats_dashboard.exploits_succeeded = 0
    stats_dashboard.threats_run = 0
    console.print("Agent memory and statistics reset.", style="yellow")

# ---------------------------------
# 12. LIVE AGENT DEMO (WORKING EXAMPLES)
# ---------------------------------

def live_agent_demo():
    console.print("[bold underline]Live Agent Demo[/bold underline]\n")

    agent = VibeHackingAgent()
    dashboard = StatisticsDashboard()

    targets = ["demo.com", "10.0.0.1"]
    target_type = "both"

    recon = agent.run_recon(target_type, targets)
    vulns = agent.run_vuln_scan(targets)
    exploits = agent.run_exploit(targets)
    threats = agent.run_threat_sim(targets, ["phishing"])
    
    dashboard.update(vulns, exploits, threats)
    
    report = agent.generate_report(targets)

    console.print(report)
    dashboard.display()

# ---------------------------------
# 13. EXPORT & RESET DEMO
# ---------------------------------

def export_and_reset_demo():
    console.print("[bold underline]Export & Reset Demo[/bold underline]\n")

    agent = VibeHackingAgent()
    dashboard = StatisticsDashboard()

    targets = ["demo.org"]
    target_type = "web"

    recon = agent.run_recon(target_type, targets)
    vulns = agent.run_vuln_scan(targets)
    exploits = agent.run_exploit(targets)
    threats = agent.run_threat_sim(targets, ["data_exfiltration"])
    
    dashboard.update(vulns, exploits, threats)
    report = agent.generate_report(targets)

    export_report(report, "exported_report.md")
    export_statistics(dashboard, "exported_stats.md")
    
    dashboard.display()
    reset_agent(agent, dashboard)

# ---------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------

if __name__ == "__main__":
    console.print("[bold cyan]Vibe Hacking AI Agent Capstone Complete Demo[/bold cyan]\n")

    # Optionally check environment setup (commented to avoid pip installs during runs)
    # environment_setup()

    # API configuration (uncomment to run config setup)
    # api_configuration()

    # Run quick test
    test_agent()

    # Run live demo
    live_agent_demo()

    # Show export and reset demo
    export_and_reset_demo()

    console.print("\n[bold green]All done![/bold green]")


