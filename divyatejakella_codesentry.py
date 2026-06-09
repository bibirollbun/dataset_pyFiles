# Install dependencies
!pip install -q -U streamlit google-adk google-generativeai bandit pyngrok


# Setup Gemini API Key from Kaggle Secrets
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    NGROK_KEY = UserSecretsClient().get_secret("NGrok")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["NGROK_KEY"] = NGROK_KEY
    print("âœ… Gemini API and Ngrok key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: {e}")
    print("Please add 'GOOGLE_API_KEY' to your Kaggle secrets.")


# Import all dependencies
import os
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any
from IPython.display import display, Markdown

# ADK imports (for new features)
try:
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    ADK_AVAILABLE = True
    print("âœ… ADK framework available")
except ImportError:
    ADK_AVAILABLE = False
    print("â„¹ï¸�  ADK not available - using legacy mode")

# Core genai imports (always available)
import google.generativeai as genai
from google.api_core import retry

print("âœ… All dependencies imported successfully.")


# Configure API Key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("âš ï¸� Warning: API_KEY is missing.")
else:
    genai.configure(api_key=API_KEY)

# Enhanced retry configuration
if ADK_AVAILABLE:
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )
    print("âœ… Using ADK retry configuration")
else:
    retry_config = None

# Legacy retry policy (always available)
retry_policy = {"retry": retry.Retry(
    initial=1.0,
    multiplier=2.0,
    maximum=60.0,
    timeout=300.0
)}

# Initialize models
flash_config = genai.GenerationConfig(temperature=0.7)
json_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)

model_flash = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    generation_config=flash_config
)

model_json = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    generation_config=json_config
)

print("âœ… Models initialized with retry logic!")


# Global Team Context - Acts as the "Memory" for CodeSentry
TEAM_CONTEXT = {
    "project_name": "SecureDataPlatform",
    "version": "3.0-ultimate",
    "last_updated": datetime.now().isoformat(),
    "allowed_patterns": [
        {
            "id": "B303",
            "reason": "MD5 used for file integrity checks only, not authentication",
            "file_scope": "utils.py",
            "confidence": 0.9,
            "added_by": "manual",
            "date_added": datetime.now().isoformat()
        }
    ],
    "coding_style": "Prefer clear variable names. Use type hinting. Follow PEP 8.",
    "security_level": "high",
    "feedback_history": [],
    "scan_statistics": {
        "total_scans": 0,
        "threats_found": 0,
        "false_positives_suppressed": 0,
        "avg_scan_time": 0.0
    }
}

# Structured output schema for compliance reports
SECURITY_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "vulnerability": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "severity": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                "cwe_id": {"type": "string"},
                "line_number": {"type": "integer"},
                "description": {"type": "string"}
            }
        },
        "risk_assessment": {
            "type": "object",
            "properties": {
                "is_exploitable": {"type": "boolean"},
                "attack_vector": {"type": "string"},
                "business_impact": {"type": "string"},
                "cvss_score": {"type": "number"}
            }
        },
        "remediation": {
            "type": "object",
            "properties": {
                "fix_complexity": {"type": "string", "enum": ["trivial", "moderate", "complex"]},
                "estimated_time": {"type": "string"},
                "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]}
            }
        }
    }
}

print("âœ… Team context initialized")


class MetricsCollector:
    """Tracks CodeSentry performance metrics for ROI calculation"""
    
    def __init__(self):
        self.scans = 0
        self.real_threats_found = 0
        self.false_alarms_suppressed = 0
        self.fixes_applied = 0
        self.fixes_rejected = 0
        self.scan_times = []
    
    def record_scan(self, is_threat, fix_applied, scan_time):
        """Records metrics for each scan execution"""
        self.scans += 1
        self.scan_times.append(scan_time)
        
        if is_threat:
            self.real_threats_found += 1
            if fix_applied:
                self.fixes_applied += 1
            else:
                self.fixes_rejected += 1
        else:
            self.false_alarms_suppressed += 1
    
    def get_summary(self):
        """Generates performance summary with ROI calculation"""
        total_alerts = self.real_threats_found + self.false_alarms_suppressed
        precision = (self.real_threats_found / total_alerts * 100) if total_alerts > 0 else 0
        avg_time = sum(self.scan_times) / len(self.scan_times) if self.scan_times else 0
        
        time_saved_fixes = self.fixes_applied * 15
        time_saved_false_alarms = self.false_alarms_suppressed * 5
        total_time_saved = time_saved_fixes + time_saved_false_alarms
        roi_dollars = total_time_saved * 50 / 60
        
        return f"""
### ğŸ�¯ CodeSentry Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Scans** | {self.scans} |
| **Real Threats Found** | {self.real_threats_found} |
| **False Alarms Suppressed** | {self.false_alarms_suppressed} |
| **Precision** | {precision:.1f}% |
| **Fixes Auto-Applied** | {self.fixes_applied}/{self.real_threats_found if self.real_threats_found > 0 else 1} |
| **Avg Scan Time** | {avg_time:.2f}s |

### ğŸ’° ROI Analysis
- Time saved from auto-fixes: {time_saved_fixes} min
- Time saved from suppressing false alarms: {time_saved_false_alarms} min
- **Total developer time saved: {total_time_saved} minutes**
- **Cost savings (at $50/hr): ${roi_dollars:.2f}**
"""

# Initialize global metrics
metrics = MetricsCollector()
print("âœ… Metrics system initialized")


def run_security_scan(code_snippet: str) -> str:
    """
    Tool: Runs Bandit security scanner on code
    Returns: JSON report of vulnerabilities
    """
    filename = "temp_scan.py"
    with open(filename, "w") as f:
        f.write(code_snippet)
    
    try:
        result = subprocess.run(
            ["bandit", "-r", filename, "-f", "json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except Exception as e:
        return json.dumps({"error": str(e)})


class ScannerAgent:
    """Detects security vulnerabilities using static analysis tools"""
    
    def scan(self, code: str) -> Dict:
        """Executes security scan and parses results"""
        print("ğŸ•µï¸� Scanner Agent: Analyzing code with Bandit...")
        raw_report = run_security_scan(code)
        
        try:
            report_data = json.loads(raw_report)
            issues = report_data.get('results', [])
            
            if not issues:
                return {"status": "clean", "issues": []}
            
            summary = {
                "status": "vulnerabilities_found",
                "count": len(issues),
                "issues": issues,
                "raw_report": raw_report
            }
            
            print(f"   âš ï¸�  Found {len(issues)} potential issue(s)")
            return summary
            
        except json.JSONDecodeError:
            return {"status": "error", "raw_report": raw_report}

print("âœ… Scanner Agent ready")


class ContextAgent:
    """
    Checks team history to see if vulnerability is already known/allowed.
    IMPORTANT: Handles vulnerability families (e.g., pickle has B301, B302, B403, B404, B305)
    """
    
    VULN_FAMILIES = {
        "pickle": ["B301", "B302", "B403", "B404", "B305"],
        "yaml": ["B506", "B605"],
        "eval": ["B307", "B102"],
        "md5": ["B303", "B324"],
        "sql": ["B608", "B609"],
    }
    
    def _get_related_ids(self, vuln_id: str) -> list:
        """Returns all vulnerability IDs in the same family"""
        for family, ids in self.VULN_FAMILIES.items():
            if vuln_id in ids:
                return ids
        return [vuln_id]
    
    def get_context(self, filename: str, vuln_id: str) -> Dict:
        """Searches memory for team-specific context"""
        print(f"ğŸ§  Context Agent: Checking team history for {vuln_id} in {filename}...")
        
        related_ids = self._get_related_ids(vuln_id)
        
        for rule in TEAM_CONTEXT["allowed_patterns"]:
            if rule["id"] in related_ids:
                if rule["file_scope"] == "*" or rule["file_scope"] in filename:
                    print(f"   ğŸ“‹ Found matching pattern for {rule['id']}: {rule['reason']}")
                    return {
                        "allowed": True,
                        "reason": rule["reason"],
                        "confidence": rule.get("confidence", 0.5)
                    }
        
        similar_feedback = [
            f for f in TEAM_CONTEXT["feedback_history"]
            if f["vuln_id"] in related_ids and not f["accepted"]
        ]
        
        if len(similar_feedback) >= 2:
            print(f"   ğŸ“Š Found {len(similar_feedback)} rejected fixes for related issues")
            return {
                "allowed": True,
                "reason": f"Team rejected related fixes {len(similar_feedback)} times",
                "confidence": 0.7
            }
        
        print("   ğŸ†• No prior context found - treating as new")
        return {
            "allowed": False,
            "reason": "No historical pattern found",
            "confidence": 0.0
        }

print("âœ… Context Agent ready")


class AnalyzerAgent:
    """
    The 'Brain' - decides if vulnerability is true/false positive.
    Uses Gemini for intelligent analysis beyond pattern matching.
    """
    
    def analyze(self, code: str, scan_summary: Dict, context_result: Dict) -> Dict:
        """Performs intelligent vulnerability analysis"""
        print("ğŸ¤” Analyzer Agent: Assessing severity and validity...")
        
        if scan_summary["status"] == "clean":
            return {"is_threat": False, "reasoning": "No issues detected", "severity": "None"}
        
        issue = scan_summary["issues"][0] if scan_summary["issues"] else {}
        
        prompt = f"""You are a Senior Security Architect. Analyze this potential vulnerability.

CODE SNIPPET:
{code}

SECURITY TOOL REPORT:
- Test ID: {issue.get('test_id', 'Unknown')}
- Issue: {issue.get('issue_text', 'No description')}
- Severity: {issue.get('issue_severity', 'Unknown')}
- Confidence: {issue.get('issue_confidence', 'Unknown')}

TEAM CONTEXT:
- Allowed: {context_result['allowed']}
- Reason: {context_result['reason']}
- Historical Confidence: {context_result['confidence']}

SECURITY POLICY:
- Project: {TEAM_CONTEXT['project_name']}
- Security Level: {TEAM_CONTEXT['security_level']}

TASK:
1. If team context says 'allowed' with confidence > 0.6, mark as FALSE POSITIVE
2. If code uses the flagged function for non-security purposes, mark FALSE POSITIVE
3. If this is genuinely exploitable, mark TRUE POSITIVE

Output ONLY valid JSON:
{{
    "is_threat": boolean,
    "reasoning": "detailed explanation",
    "severity": "Low/Medium/High/Critical",
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = model_flash.generate_content(prompt, request_options=retry_policy)
            text = response.text.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(text)
            
            print(f"   ğŸ“Š Decision: {'ğŸš¨ THREAT' if analysis['is_threat'] else 'âœ… SAFE'}")
            return analysis
            
        except Exception as e:
            print(f"   âš ï¸�  Analysis error: {e}")
            return {
                "is_threat": True,
                "reasoning": f"Analysis failed, assuming threat for safety: {e}",
                "severity": "High",
                "confidence": 0.5
            }

print("âœ… Analyzer Agent ready")


class EducatorAgent:
    """Explains vulnerabilities in developer-friendly language (no jargon)"""
    
    def explain(self, code: str, scan_summary: Dict, analysis: Dict) -> str:
        """Generates plain-English explanation"""
        print("ğŸ�“ Educator Agent: Translating security-speak to plain English...")
        
        issue = scan_summary["issues"][0] if scan_summary["issues"] else {}
        
        prompt = f"""You are a Senior Engineer explaining security to a junior developer.

CODE:
{code}

VULNERABILITY:
- Type: {issue.get('test_id', 'Unknown')}
- Description: {issue.get('issue_text', 'Unknown')}
- Severity: {analysis['severity']}

TASK: Explain in 2-3 sentences:
1. What the vulnerability is (no jargon)
2. Why it's dangerous (real-world example)
3. How an attacker could exploit it

Be concise, clear, and helpful. No condescending tone."""
        
        response = model_flash.generate_content(prompt, request_options=retry_policy)
        explanation = response.text.strip()
        
        print("   âœ… Explanation generated")
        return explanation

print("âœ… Educator Agent ready")


class FixerAgent:
    """Generates secure code that matches team style"""
    
    def fix(self, code: str, explanation: str) -> str:
        """Generates secure code replacement"""
        print("ğŸ› ï¸� Fixer Agent: Generating secure code...")
        
        prompt = f"""You are a Python Expert. Rewrite this code to fix the security vulnerability.

ORIGINAL CODE:
{code}

VULNERABILITY:
{explanation}

TEAM CODING STYLE:
{TEAM_CONTEXT['coding_style']}

REQUIREMENTS:
1. Maintain original functionality
2. Follow team's coding style
3. Add a comment explaining the security improvement
4. Use modern Python best practices

Output ONLY the fixed Python code (no explanations, no markdown)."""
        
        response = model_flash.generate_content(prompt, request_options=retry_policy)
        fixed_code = response.text.strip().replace("```python", "").replace("```", "").strip()
        
        print("   âœ… Secure code generated")
        return fixed_code

print("âœ… Fixer Agent ready")


class ReportingAgent:
    """Generates structured compliance-ready reports"""
    
    def generate_report(self, code: str, analysis: Dict, explanation: str, fixed_code: str) -> Dict:
        """Creates machine-readable compliance report"""
        print("ğŸ“Š Reporting Agent: Creating structured report...")
        
        prompt = f"""Generate a security compliance report following this EXACT JSON schema:
{json.dumps(SECURITY_REPORT_SCHEMA, indent=2)}

ANALYSIS:
{json.dumps(analysis, indent=2)}

VULNERABILITY EXPLANATION:
{explanation}

FIXED CODE AVAILABLE: {len(fixed_code) > 0}

INSTRUCTIONS:
- Map severity to one of: Low, Medium, High, Critical
- Estimate CVSS score (0-10)
- Set priority: P0 (Critical), P1 (High), P2 (Medium), P3 (Low)
- Estimate fix time: "5-15 min", "15-30 min", "30-60 min", "1-2 hours"
- Set complexity: trivial, moderate, complex

Output ONLY valid JSON matching the schema."""
        
        try:
            response = model_json.generate_content(prompt, request_options=retry_policy)
            report = json.loads(response.text)
            print("   âœ… Structured report generated")
            return report
        except Exception as e:
            print(f"   âš ï¸�  Report generation failed: {e}")
            return {"error": str(e)}

print("âœ… Reporting Agent ready")


class FeedbackAgent:
    """
    Learns from developer feedback to reduce false positives over time.
    This is the KEY DIFFERENTIATOR from traditional SAST tools.
    """
    
    def collect_feedback(self, vuln_id: str, code_context: str, accepted: bool, reason: str = "") -> str:
        """Records developer feedback for learning"""
        print(f"ğŸ“ˆ Feedback Agent: Recording developer feedback...")
        
        feedback = {
            "vuln_id": vuln_id,
            "code_pattern": self._extract_pattern(code_context),
            "accepted": accepted,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        TEAM_CONTEXT["feedback_history"].append(feedback)
        
        if not accepted:
            rejection_count = sum(
                1 for f in TEAM_CONTEXT["feedback_history"]
                if f["vuln_id"] == vuln_id and not f["accepted"]
            )
            
            if rejection_count >= 2:
                existing = [p for p in TEAM_CONTEXT["allowed_patterns"] if p["id"] == vuln_id]
                if not existing:
                    TEAM_CONTEXT["allowed_patterns"].append({
                        "id": vuln_id,
                        "reason": f"Auto-learned: Rejected {rejection_count}x - {reason}",
                        "file_scope": "*",
                        "confidence": min(0.6 + (rejection_count * 0.1), 0.9),
                        "added_by": "auto-learning",
                        "date_added": datetime.now().isoformat()
                    })
                    print(f"   ğŸ§  Learning applied: {vuln_id} added to allowed patterns")
        
        acceptance_rate = self._get_acceptance_rate(vuln_id)
        print(f"   ğŸ“Š Acceptance rate for {vuln_id}: {acceptance_rate}")
        
        return f"Feedback recorded. {acceptance_rate}"
    
    def _extract_pattern(self, code: str) -> str:
        lines = code.strip().split('\n')
        return lines[0] if lines else ""
    
    def _get_acceptance_rate(self, vuln_id: str) -> str:
        relevant = [f for f in TEAM_CONTEXT["feedback_history"] if f["vuln_id"] == vuln_id]
        if not relevant:
            return "No history"
        
        accepted_count = sum(1 for f in relevant if f["accepted"])
        total = len(relevant)
        percentage = (accepted_count / total) * 100
        return f"{accepted_count}/{total} ({percentage:.0f}%)"

print("âœ… Feedback Agent ready")


class UltimateOrchestrator:
    """
    Master controller with session management and observability.
    Implements multi-step planning based on scan results.
    """
    
    def __init__(self):
        self.scanner = ScannerAgent()
        self.context = ContextAgent()
        self.analyzer = AnalyzerAgent()
        self.educator = EducatorAgent()
        self.fixer = FixerAgent()
        self.reporter = ReportingAgent()
        self.feedback = FeedbackAgent()
        
        # Session and audit tracking
        self.sessions = {}
        self.audit_log = []
    
    def create_plan(self, scan_summary: Dict) -> Dict:
        """Creates execution plan based on scan results"""
        if scan_summary["status"] == "clean":
            return {
                "needs_context_check": False,
                "needs_analysis": False,
                "needs_fix": False,
                "needs_report": False,
                "reasoning": "No vulnerabilities detected"
            }
        
        issue = scan_summary["issues"][0] if scan_summary["issues"] else {}
        severity = issue.get("issue_severity", "MEDIUM").upper()
        
        plan = {
            "needs_context_check": True,
            "needs_analysis": True,
            "needs_fix": severity in ["HIGH", "MEDIUM"],
            "needs_report": severity in ["HIGH", "CRITICAL"],
            "reasoning": f"Detected {severity} severity issue - full pipeline needed"
        }
        
        print(f"ğŸ“‹ Orchestrator Plan: {plan['reasoning']}")
        return plan
    
    def execute(self, filename: str, code: str) -> Dict:
        """Main execution pipeline"""
        import time
        start_time = time.time()
        
        print("\n" + "="*70)
        print(f"ğŸ�¯ CodeSentry Ultimate analyzing: {filename}")
        print("="*70 + "\n")
        
        results = {}
        
        # Step 1: Scan
        scan_summary = self.scanner.scan(code)
        results["scan"] = scan_summary
        
        if scan_summary["status"] == "clean":
            print("\nâœ… No security issues detected!")
            metrics.record_scan(is_threat=False, fix_applied=False, scan_time=time.time()-start_time)
            return results
        
        # Step 2: Create execution plan
        plan = self.create_plan(scan_summary)
        results["plan"] = plan
        
        vuln_id = scan_summary["issues"][0].get("test_id", "Unknown") if scan_summary["issues"] else "Unknown"
        
        # Step 3: Check context
        context_result = {"allowed": False, "confidence": 0.0}
        if plan["needs_context_check"]:
            context_result = self.context.get_context(filename, vuln_id)
            results["context"] = context_result
        
        # Short-circuit if high-confidence allowance
        if context_result.get("allowed", False) and context_result.get("confidence", 0.0) > 0.6:
            print(f"\nâœ… Alert suppressed by team pattern: {context_result['reason']}")
            print(f"   Confidence: {context_result['confidence']*100:.0f}%")
            analysis = {
                "is_threat": False,
                "reasoning": f"Team-approved (confidence: {context_result['confidence']*100:.0f}%): {context_result['reason']}",
                "severity": "None",
                "confidence": context_result["confidence"]
            }
            results["analysis"] = analysis
            metrics.record_scan(is_threat=False, fix_applied=False, scan_time=time.time()-start_time)
            return results
        
        # Step 4: Analyze
        if plan["needs_analysis"]:
            analysis = self.analyzer.analyze(code, scan_summary, context_result)
            results["analysis"] = analysis
        else:
            analysis = {"is_threat": False, "reasoning": "Skipped analysis"}
        
        if not analysis["is_threat"]:
            print(f"\nâœ… Alert suppressed: {analysis['reasoning']}")
            metrics.record_scan(is_threat=False, fix_applied=False, scan_time=time.time()-start_time)
            return results
        
        # Step 5: Explain
        explanation = self.educator.explain(code, scan_summary, analysis)
        results["explanation"] = explanation
        
        # Step 6: Fix
        fixed_code = ""
        if plan["needs_fix"]:
            fixed_code = self.fixer.fix(code, explanation)
            results["fixed_code"] = fixed_code
        
        # Step 7: Report
        if plan["needs_report"]:
            report = self.reporter.generate_report(code, analysis, explanation, fixed_code)
            results["compliance_report"] = report
        
        metrics.record_scan(is_threat=True, fix_applied=len(fixed_code) > 0, scan_time=time.time()-start_time)
        return results

print("âœ… Ultimate Orchestrator ready")


def display_results(results: Dict):
    """Pretty print CodeSentry results"""
    if results.get("scan", {}).get("status") == "clean":
        display(Markdown("### âœ… No Issues Found\n\nCode passed security scan!"))
        return
    
    analysis = results.get("analysis", {})
    
    if not analysis.get("is_threat", False):
        display(Markdown(f"### âœ… False Positive Suppressed\n\n**Reason:** {analysis.get('reasoning', 'N/A')}"))
        return
    
    output = f"### ğŸš¨ Security Vulnerability Detected\n\n"
    output += f"**Severity:** {analysis.get('severity', 'Unknown')}  \n"
    output += f"**Confidence:** {analysis.get('confidence', 0)*100:.0f}%  \n\n"
    
    if "explanation" in results:
        output += f"#### ğŸ“– What's Wrong\n\n{results['explanation']}\n\n"
    
    if "fixed_code" in results:
        output += f"#### ğŸ› ï¸� Proposed Fix\n\n```python\n{results['fixed_code']}\n```\n\n"
    
    if "compliance_report" in results:
        report = results["compliance_report"]
        if "remediation" in report:
            rem = report["remediation"]
            output += f"#### ğŸ“‹ Remediation Details\n\n"
            output += f"- **Priority:** {rem.get('priority', 'N/A')}  \n"
            output += f"- **Complexity:** {rem.get('fix_complexity', 'N/A')}  \n"
            output += f"- **Estimated Time:** {rem.get('estimated_time', 'N/A')}  \n\n"
    
    display(Markdown(output))

print("âœ… Display helpers ready")


orchestrator = UltimateOrchestrator()

dangerous_code = """
import hashlib

def store_password(password):
    # SECURITY ISSUE: MD5 is cryptographically broken!
    return hashlib.md5(password.encode()).hexdigest()

user_pass = "supersecret123"
hashed = store_password(user_pass)
print(f"Stored password hash: {hashed}")
"""

results1 = orchestrator.execute("auth_service.py", dangerous_code)
display_results(results1)


safe_checksum_code = """
import hashlib

def calculate_file_etag(file_data):
    # This is fine: MD5 for file deduplication, not security
    return hashlib.md5(file_data).hexdigest()

file_content = b"large file content here"
etag = calculate_file_etag(file_content)
"""

results2 = orchestrator.execute("utils.py", safe_checksum_code)
display_results(results2)


print("\n" + "="*70)
print("ğŸ“ˆ DEMONSTRATING LEARNING CAPABILITY")
print("="*70 + "\n")

feedback_agent = FeedbackAgent()

print("\n--- ğŸ—“ï¸� WEEK 1: First Pickle Encounter ---\n")

pickle_code = """
import pickle

def load_internal_config(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
"""

results_w1 = orchestrator.execute("config_loader.py", pickle_code)
display_results(results_w1)

print("\nğŸ‘¨â€�ğŸ’» Developer: 'This is for internal configs only - rejecting fix'\n")
feedback_agent.collect_feedback("B301", pickle_code, False, "Internal config files only")

print("\n--- ğŸ—“ï¸� WEEK 2: Similar Pattern ---\n")
print("ğŸ‘¨â€�ğŸ’» Developer: 'Same situation - rejecting again'\n")
feedback_agent.collect_feedback("B301", pickle_code, False, "Internal ML models only")

print("\n--- ğŸ—“ï¸� WEEK 4: System Has Learned! ---\n")
results_w4 = orchestrator.execute("cache_manager.py", pickle_code)
display_results(results_w4)

print("\nğŸ�‰ CodeSentry learned from feedback! False positive automatically suppressed.")


sql_injection_code = """
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # CRITICAL: SQL Injection vulnerability!
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    
    return cursor.fetchone()
"""

results3 = orchestrator.execute("database.py", sql_injection_code)
display_results(results3)


display(Markdown(metrics.get_summary()))


# ============================================================================
# CodeSentry for Kaggle - Complete Setup
# Run this cell first to create the Streamlit app file
# ============================================================================

import os

# Create the Streamlit app code (using raw string to avoid quote issues)
streamlit_app_code = r'''# ============================================================================
# CodeSentry Production - Kaggle Ready
# Full feedback loop, persistence, metrics, and proper state management
# ============================================================================

import streamlit as st
import os
import json
import subprocess
from datetime import datetime
from typing import Dict, List
import google.generativeai as genai
from google.api_core import retry
import pickle
from pathlib import Path

# ============================================================================
# PERSISTENCE LAYER
# ============================================================================

class PersistenceManager:
    """Handles saving/loading team context to disk"""
    
    def __init__(self, storage_path="codesentry_data.pkl"):
        self.storage_path = storage_path
    
    def save(self, data: Dict):
        """Save team context to disk"""
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            st.warning(f"Could not save data: {e}")
    
    def load(self) -> Dict:
        """Load team context from disk"""
        if Path(self.storage_path).exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load data: {e}")
        return None

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CodeSentry Security Scanner",
    page_icon="ğŸ›¡ï¸�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Setup
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("âš ï¸� GOOGLE_API_KEY not found!")
    st.info("Set it with: os.environ['GOOGLE_API_KEY'] = 'your-key'")
    st.stop()

genai.configure(api_key=API_KEY)

flash_config = genai.GenerationConfig(temperature=0.7)
json_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)

model_flash = genai.GenerativeModel(model_name='gemini-2.5-flash-lite', generation_config=flash_config)
model_json = genai.GenerativeModel(model_name='gemini-2.5-flash-lite', generation_config=json_config)

retry_policy = {"retry": retry.Retry(initial=1.0, multiplier=2.0, maximum=60.0, timeout=300.0)}

# ============================================================================
# SESSION STATE INITIALIZATION WITH PERSISTENCE
# ============================================================================

persistence = PersistenceManager()

# Load saved data or initialize fresh
if 'team_context' not in st.session_state:
    saved_data = persistence.load()
    
    if saved_data:
        st.session_state.team_context = saved_data
        st.toast("âœ… Loaded previous session data", icon="ğŸ“‚")
    else:
        st.session_state.team_context = {
            "project_name": "SecureDataPlatform",
            "version": "4.0-production",
            "last_updated": datetime.now().isoformat(),
            "allowed_patterns": [
                {
                    "id": "B303",
                    "reason": "MD5 for integrity checks only, not authentication",
                    "file_scope": "utils.py",
                    "confidence": 0.9,
                    "added_by": "manual",
                    "date_added": datetime.now().isoformat()
                }
            ],
            "coding_style": "Prefer clear variable names. Use type hinting. Follow PEP 8.",
            "security_level": "high",
            "feedback_history": [],
            "scan_statistics": {
                "total_scans": 0,
                "real_threats_found": 0,
                "false_positives_suppressed": 0,
                "fixes_applied": 0,
                "total_scan_time": 0.0
            }
        }

if 'last_scan_result' not in st.session_state:
    st.session_state.last_scan_result = None

if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

if 'show_feedback_form' not in st.session_state:
    st.session_state.show_feedback_form = False

if 'feedback_type' not in st.session_state:
    st.session_state.feedback_type = None

# ============================================================================
# CORE SECURITY SCANNER
# ============================================================================

def run_security_scan(code_snippet: str) -> str:
    filename = "temp_scan.py"
    with open(filename, "w") as f:
        f.write(code_snippet)
    try:
        result = subprocess.run(["bandit", "-r", filename, "-f", "json"], capture_output=True, text=True, timeout=10)
        return result.stdout
    except Exception as e:
        return json.dumps({"error": str(e)})

class ScannerAgent:
    def scan(self, code: str) -> Dict:
        raw_report = run_security_scan(code)
        try:
            report_data = json.loads(raw_report)
            issues = report_data.get('results', [])
            return {"status": "clean", "issues": []} if not issues else {
                "status": "vulnerabilities_found",
                "count": len(issues),
                "issues": issues,
                "raw_report": raw_report
            }
        except json.JSONDecodeError:
            return {"status": "error", "raw_report": raw_report}

class ContextAgent:
    VULN_FAMILIES = {
        "pickle": ["B301", "B302", "B403", "B404", "B305"],
        "yaml": ["B506", "B605"],
        "eval": ["B307", "B102"],
        "md5": ["B303", "B324"],
        "sql": ["B608", "B609"],
    }
    
    def _get_related_ids(self, vuln_id: str) -> list:
        for family, ids in self.VULN_FAMILIES.items():
            if vuln_id in ids:
                return ids
        return [vuln_id]
    
    def get_context(self, filename: str, vuln_id: str) -> Dict:
        related_ids = self._get_related_ids(vuln_id)
        
        for rule in st.session_state.team_context["allowed_patterns"]:
            if rule["id"] in related_ids and (rule["file_scope"] == "*" or rule["file_scope"] in filename):
                return {
                    "allowed": True,
                    "reason": rule["reason"],
                    "confidence": rule.get("confidence", 0.5)
                }
        
        similar_feedback = [
            f for f in st.session_state.team_context["feedback_history"]
            if f["vuln_id"] in related_ids and not f["accepted"]
        ]
        
        if len(similar_feedback) >= 2:
            return {
                "allowed": True,
                "reason": f"Team rejected related fixes {len(similar_feedback)} times",
                "confidence": 0.7
            }
        
        return {"allowed": False, "reason": "No pattern found", "confidence": 0.0}

class AnalyzerAgent:
    # Critical vulnerability patterns that should always be High/Critical
    CRITICAL_PATTERNS = {
        "B608": "SQL injection",  # Hardcoded SQL
        "B609": "SQL injection",  # String SQL formatting
        "B201": "Flask debug mode",
        "B301": "Pickle deserialization",
        "B506": "YAML unsafe load",
        "B102": "exec() usage",
        "B307": "eval() usage",
    }
    
    def analyze(self, code: str, scan_summary: Dict, context_result: Dict) -> Dict:
        if scan_summary["status"] == "clean":
            return {"is_threat": False, "reasoning": "No issues detected", "severity": "None"}
        
        issue = scan_summary["issues"][0] if scan_summary["issues"] else {}
        vuln_id = issue.get('test_id', 'Unknown')
        reported_severity = issue.get('issue_severity', 'MEDIUM').upper()
        
        # Override severity for known critical vulnerabilities
        if vuln_id in self.CRITICAL_PATTERNS:
            # Check if it's actually exploitable
            if self._is_sql_injection_exploitable(code, vuln_id):
                reported_severity = "CRITICAL"
            elif vuln_id in ["B608", "B609"]:  # SQL injection variants
                reported_severity = "HIGH"  # At minimum High for SQL injection
        
        prompt = f"""You are a Senior Security Architect. Analyze this potential vulnerability.

CODE SNIPPET:
{code}

SECURITY TOOL REPORT:
- Test ID: {vuln_id}
- Issue: {issue.get('issue_text', 'No description')}
- Reported Severity: {reported_severity}
- Confidence: {issue.get('issue_confidence', 'Unknown')}

TEAM CONTEXT:
- Allowed: {context_result['allowed']}
- Reason: {context_result['reason']}
- Historical Confidence: {context_result['confidence']}

CRITICAL CONTEXT:
- SQL Injection (B608, B609): ALWAYS High or Critical if user input involved
- Pickle/YAML unsafe load (B301, B506): ALWAYS High if loading untrusted data
- eval/exec (B102, B307): ALWAYS High if user input involved
- Flask debug mode in production (B201): ALWAYS Critical

ANALYSIS RULES:
1. If team context says 'allowed' with confidence > 0.6, mark as FALSE POSITIVE
2. If code uses the flagged function for non-security purposes (e.g., MD5 for checksums), mark FALSE POSITIVE
3. If SQL injection with user input possible: Mark as HIGH or CRITICAL
4. If deserialization with untrusted data: Mark as HIGH or CRITICAL
5. If genuinely exploitable with realistic attack vector: Mark TRUE POSITIVE with appropriate severity

Output ONLY valid JSON:
{{
    "is_threat": boolean,
    "reasoning": "detailed explanation of exploitability",
    "severity": "Low/Medium/High/Critical",
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = model_flash.generate_content(prompt, request_options=retry_policy)
            analysis = json.loads(response.text.replace("```json", "").replace("```", "").strip())
            
            # Safety check: Ensure SQL injection is never downgraded below High
            if vuln_id in ["B608", "B609"] and analysis["is_threat"]:
                if analysis["severity"] in ["Low", "Medium"]:
                    analysis["severity"] = "High"
                    analysis["reasoning"] += " [Severity upgraded: SQL injection is always High minimum]"
            
            return analysis
            
        except Exception as e:
            # Fail-safe with appropriate severity
            default_severity = "Critical" if vuln_id in self.CRITICAL_PATTERNS else "High"
            return {
                "is_threat": True,
                "reasoning": f"Analysis failed, assuming threat for safety: {e}",
                "severity": default_severity,
                "confidence": 0.5
            }
    
    def _is_sql_injection_exploitable(self, code: str, vuln_id: str) -> bool:
        """Check if SQL injection is actually exploitable"""
        if vuln_id not in ["B608", "B609"]:
            return False
        
        # Check for signs of user input
        user_input_indicators = [
            "input(",
            "request.",
            "argv",
            "environ",
            "get(",
            "post(",
            "form",
            "args",
            "data",
            "json",
            "query",
            "params"
        ]
        
        code_lower = code.lower()
        return any(indicator in code_lower for indicator in user_input_indicators)

class EducatorAgent:
    # Pre-written explanations for critical vulnerabilities
    CRITICAL_EXPLANATIONS = {
        "B608": "This code builds SQL queries by directly inserting user input into the query string. An attacker could inject malicious SQL commands (like `' OR '1'='1`) to bypass authentication, steal all database records, or even delete your entire database. This is one of the most dangerous web vulnerabilities - it's literally giving attackers direct database access.",
        
        "B609": "This code uses string formatting to build SQL queries with user input. An attacker could inject SQL commands by providing malicious input like `admin' --` to bypass login checks, `'; DROP TABLE users; --` to delete data, or use UNION queries to steal sensitive information. This vulnerability has been used in countless real-world breaches.",
        
        "B301": "This code uses pickle.load() which can execute arbitrary code during deserialization. An attacker who controls the pickle file can craft it to run any Python code they want on your server - install backdoors, steal credentials, or take complete control of your system. Never unpickle data from untrusted sources.",
        
        "B506": "This code uses yaml.load() without SafeLoader, which can execute arbitrary Python code. An attacker can craft a YAML file that runs malicious code when loaded - this could give them complete control of your server, steal data, or install malware. Always use yaml.safe_load() instead.",
        
        "B102": "This code uses exec() which executes arbitrary Python code from a string. If an attacker can control what goes into exec(), they can run any code they want - read files, steal credentials, install backdoors, or completely take over your system. This is essentially giving attackers a Python shell.",
        
        "B307": "This code uses eval() which evaluates arbitrary Python expressions. An attacker who controls the input can execute malicious code - they could import os and run system commands, read sensitive files, or modify your application's behavior in dangerous ways. eval() should almost never be used with untrusted input."
    }
    
    def explain(self, code: str, scan_summary: Dict, analysis: Dict) -> str:
        issue = scan_summary["issues"][0] if scan_summary["issues"] else {}
        vuln_id = issue.get('test_id', 'Unknown')
        
        # Use pre-written explanation for critical vulnerabilities
        if vuln_id in self.CRITICAL_EXPLANATIONS:
            return self.CRITICAL_EXPLANATIONS[vuln_id]
        
        # Otherwise, generate explanation with AI
        prompt = f"""Explain this vulnerability in plain English (2-3 sentences):
CODE: {code}
VULNERABILITY: {vuln_id} - {issue.get('issue_text')}
SEVERITY: {analysis['severity']}

Requirements:
1. Explain WHAT the vulnerability is (no jargon)
2. Explain WHY it's dangerous (real-world example)
3. Explain HOW an attacker could exploit it
4. Be specific about the actual risk, not generic warnings"""
        
        return model_flash.generate_content(prompt, request_options=retry_policy).text.strip()

class FixerAgent:
    def fix(self, code: str, explanation: str) -> str:
        prompt = f"""Fix this security issue:
ORIGINAL: {code}
PROBLEM: {explanation}
STYLE: {st.session_state.team_context['coding_style']}
Output only the fixed Python code (no markdown)."""
        return model_flash.generate_content(prompt, request_options=retry_policy).text.strip().replace("```python", "").replace("```", "").strip()

class MetricsCollector:
    @staticmethod
    def calculate_roi() -> Dict:
        stats = st.session_state.team_context["scan_statistics"]
        time_saved_fixes = stats["fixes_applied"] * 15
        time_saved_false_alarms = stats["false_positives_suppressed"] * 5
        total_time_saved = time_saved_fixes + time_saved_false_alarms
        cost_savings = (total_time_saved / 60) * 50
        total_alerts = stats["real_threats_found"] + stats["false_positives_suppressed"]
        precision = (stats["real_threats_found"] / total_alerts * 100) if total_alerts > 0 else 0
        avg_scan_time = stats["total_scan_time"] / stats["total_scans"] if stats["total_scans"] > 0 else 0
        
        return {
            "time_saved_fixes": time_saved_fixes,
            "time_saved_false_alarms": time_saved_false_alarms,
            "total_time_saved": total_time_saved,
            "cost_savings": cost_savings,
            "precision": precision,
            "avg_scan_time": avg_scan_time
        }

class UltimateOrchestrator:
    def __init__(self):
        self.scanner = ScannerAgent()
        self.context = ContextAgent()
        self.analyzer = AnalyzerAgent()
        self.educator = EducatorAgent()
        self.fixer = FixerAgent()
    
    def execute(self, filename: str, code: str) -> Dict:
        import time
        start_time = time.time()
        
        results = {}
        scan_summary = self.scanner.scan(code)
        results["scan"] = scan_summary
        
        if scan_summary["status"] == "clean":
            scan_time = time.time() - start_time
            results["scan_time"] = scan_time
            st.session_state.team_context["scan_statistics"]["total_scans"] += 1
            st.session_state.team_context["scan_statistics"]["total_scan_time"] += scan_time
            return results
        
        vuln_id = scan_summary["issues"][0].get("test_id", "Unknown")
        context_result = self.context.get_context(filename, vuln_id)
        results["context"] = context_result
        
        if context_result.get("allowed", False) and context_result.get("confidence", 0.0) > 0.6:
            results["analysis"] = {
                "is_threat": False,
                "reasoning": f"Team-approved: {context_result['reason']}",
                "severity": "None"
            }
            scan_time = time.time() - start_time
            results["scan_time"] = scan_time
            st.session_state.team_context["scan_statistics"]["total_scans"] += 1
            st.session_state.team_context["scan_statistics"]["false_positives_suppressed"] += 1
            st.session_state.team_context["scan_statistics"]["total_scan_time"] += scan_time
            return results
        
        analysis = self.analyzer.analyze(code, scan_summary, context_result)
        results["analysis"] = analysis
        
        if not analysis["is_threat"]:
            scan_time = time.time() - start_time
            results["scan_time"] = scan_time
            st.session_state.team_context["scan_statistics"]["total_scans"] += 1
            st.session_state.team_context["scan_statistics"]["false_positives_suppressed"] += 1
            st.session_state.team_context["scan_statistics"]["total_scan_time"] += scan_time
            return results
        
        results["explanation"] = self.educator.explain(code, scan_summary, analysis)
        results["fixed_code"] = self.fixer.fix(code, results["explanation"])
        
        scan_time = time.time() - start_time
        results["scan_time"] = scan_time
        st.session_state.team_context["scan_statistics"]["total_scans"] += 1
        st.session_state.team_context["scan_statistics"]["real_threats_found"] += 1
        st.session_state.team_context["scan_statistics"]["total_scan_time"] += scan_time
        
        return results

def submit_feedback(vulnerability_id: str, user_accepts: bool, reason: str):
    feedback = {
        "vuln_id": vulnerability_id,
        "accepted": user_accepts,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }
    
    st.session_state.team_context["feedback_history"].append(feedback)
    
    if not user_accepts:
        rejection_count = sum(
            1 for f in st.session_state.team_context["feedback_history"]
            if f["vuln_id"] == vulnerability_id and not f["accepted"]
        )
        
        if rejection_count >= 2:
            existing = [p for p in st.session_state.team_context["allowed_patterns"] if p["id"] == vulnerability_id]
            if not existing:
                st.session_state.team_context["allowed_patterns"].append({
                    "id": vulnerability_id,
                    "reason": f"Auto-learned: Rejected {rejection_count}x - {reason}",
                    "file_scope": "*",
                    "confidence": min(0.6 + (rejection_count * 0.1), 0.9),
                    "added_by": "auto-learning",
                    "date_added": datetime.now().isoformat()
                })
                
                st.session_state.team_context["scan_statistics"]["real_threats_found"] -= 1
                st.session_state.team_context["scan_statistics"]["false_positives_suppressed"] += 1
                
                st.toast(f"ğŸ§  Learning applied! {vulnerability_id} will be suppressed in future scans", icon="âœ…")
                return f"âœ… Learning applied! {vulnerability_id} added to allowed patterns with {min(0.6 + (rejection_count * 0.1), 0.9)*100:.0f}% confidence."
    else:
        st.session_state.team_context["scan_statistics"]["fixes_applied"] += 1
    
    relevant = [f for f in st.session_state.team_context["feedback_history"] if f["vuln_id"] == vulnerability_id]
    accepted_count = sum(1 for f in relevant if f["accepted"])
    total = len(relevant)
    percentage = (accepted_count / total) * 100 if total > 0 else 0
    
    persistence.save(st.session_state.team_context)
    
    return f"âœ… Feedback recorded! Acceptance rate for {vulnerability_id}: {accepted_count}/{total} ({percentage:.0f}%)"

# UI - SIDEBAR
with st.sidebar:
    st.header("ğŸ“Š Team Statistics")
    
    stats = st.session_state.team_context["scan_statistics"]
    roi = MetricsCollector.calculate_roi()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Scans", stats["total_scans"])
        st.metric("Real Threats", stats["real_threats_found"])
        st.metric("Precision", f"{roi['precision']:.1f}%")
    with col2:
        st.metric("FP Suppressed", stats["false_positives_suppressed"])
        st.metric("Fixes Applied", stats["fixes_applied"])
        st.metric("Avg Scan Time", f"{roi['avg_scan_time']:.2f}s")
    
    st.divider()
    st.header("ğŸ’° ROI Analysis")
    st.metric("Time Saved", f"{roi['total_time_saved']} min")
    st.metric("Cost Savings", f"${roi['cost_savings']:.2f}")
    
    with st.expander("ğŸ“ˆ Breakdown"):
        st.write(f"**Auto-fixes:** {roi['time_saved_fixes']} min saved")
        st.write(f"**FP Suppression:** {roi['time_saved_false_alarms']} min saved")
        st.write(f"**Rate:** $50/hour developer time")
    
    st.divider()
    st.header("ğŸ§ª Quick Examples")
    
    if st.button("ğŸ”´ Insecure MD5", use_container_width=True):
        st.session_state.example_code = """import hashlib
password = "supersecret123"
hash = hashlib.md5(password.encode()).hexdigest()
print(f"Hash: {hash}")"""
        st.rerun()
    
    if st.button("ğŸ”´ Pickle Vuln", use_container_width=True):
        st.session_state.example_code = """import pickle
with open('user_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)"""
        st.rerun()
    
    if st.button("ğŸ”´ YAML Unsafe", use_container_width=True):
        st.session_state.example_code = """import yaml
with open('config.yml') as f:
    config = yaml.load(f)"""
        st.rerun()
    
    if st.button("ğŸŸ¢ Safe SHA-256", use_container_width=True):
        st.session_state.example_code = """import hashlib
password = "secret"
hash = hashlib.sha256(password.encode()).hexdigest()
print(hash)"""
        st.rerun()
    
    st.divider()
    st.header("ğŸ§  Team Memory")
    
    with st.expander(f"Allowed Patterns ({len(st.session_state.team_context['allowed_patterns'])})"):
        for pattern in st.session_state.team_context["allowed_patterns"]:
            st.write(f"**{pattern['id']}** - {pattern['reason'][:50]}...")
            st.caption(f"Confidence: {pattern['confidence']*100:.0f}% | Added: {pattern['added_by']}")
    
    with st.expander(f"Feedback History ({len(st.session_state.team_context['feedback_history'])})"):
        for fb in reversed(st.session_state.team_context["feedback_history"][-5:]):
            icon = "âœ…" if fb["accepted"] else "â�Œ"
            st.write(f"{icon} **{fb['vuln_id']}** - {fb['reason'][:40]}...")
            st.caption(f"{fb['timestamp'][:10]}")
    
    st.divider()
    st.header("ğŸ’¾ Data Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Data", use_container_width=True):
            persistence.save(st.session_state.team_context)
            st.toast("âœ… Data saved!", icon="ğŸ’¾")
    
    with col2:
        if st.button("Reset All", use_container_width=True):
            if st.button("âš ï¸� Confirm Reset"):
                st.session_state.clear()
                st.rerun()

# UI - MAIN AREA
st.title("ğŸ›¡ï¸� CodeSentry Security Scanner")
st.caption("AI-powered Python security analysis with adaptive learning")

code_input = st.text_area(
    "Paste your Python code here:",
    height=250,
    value=st.session_state.get('example_code', ''),
    placeholder="import hashlib\npassword = 'test'\nhash = hashlib.md5(password.encode()).hexdigest()"
)

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    scan_button = st.button("ğŸ”� Scan Code", type="primary", use_container_width=True)

with col2:
    if st.session_state.last_scan_result:
        if st.button("ğŸ—‘ï¸� Clear Results", use_container_width=True):
            st.session_state.last_scan_result = None
            st.session_state.feedback_submitted = False
            st.session_state.show_feedback_form = False
            st.rerun()

if scan_button and code_input:
    with st.spinner("ğŸ•µï¸� Analyzing code for security vulnerabilities..."):
        try:
            orchestrator = UltimateOrchestrator()
            results = orchestrator.execute("snippet.py", code_input)
            
            st.session_state.last_scan_result = results
            st.session_state.scan_history.append({
                "timestamp": datetime.now().isoformat(),
                "code": code_input[:100] + "..." if len(code_input) > 100 else code_input,
                "status": results.get("scan", {}).get("status", "unknown")
            })
            st.session_state.feedback_submitted = False
            st.session_state.show_feedback_form = False
            
            persistence.save(st.session_state.team_context)
            st.rerun()
            
        except Exception as e:
            st.error(f"â�Œ Analysis failed: {str(e)}")

# DISPLAY RESULTS
if st.session_state.last_scan_result:
    results = st.session_state.last_scan_result
    
    if results.get("scan", {}).get("status") == "clean":
        st.success("### âœ… No Security Issues Found")
        st.info("Your code passed the security scan! No vulnerabilities detected.")
    
    elif results.get("analysis", {}).get("is_threat") == False:
        st.success("### âœ… Alert Suppressed")
        st.info(f"**Reason:** {results.get('analysis', {}).get('reasoning', 'N/A')}")
    
    else:
        analysis = results.get("analysis", {})
        severity = analysis.get("severity", "Unknown")
        
        if severity == "Critical":
            st.error(f"### ğŸš¨ Security Vulnerability Detected - {severity}")
        elif severity == "High":
            st.warning(f"### âš ï¸� Security Vulnerability Detected - {severity}")
        else:
            st.info(f"### â„¹ï¸� Security Vulnerability Detected - {severity}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Severity", severity)
        with col2:
            st.metric("Confidence", f"{analysis.get('confidence', 0)*100:.0f}%")
        with col3:
            vuln_id = results.get("scan", {}).get("issues", [{}])[0].get("test_id", "Unknown")
            st.metric("Vulnerability ID", vuln_id)
        
        st.divider()
        
        if "explanation" in results:
            st.subheader("ğŸ“– What's Wrong")
            st.write(results["explanation"])
        
        if "fixed_code" in results:
            st.subheader("ğŸ› ï¸� Proposed Fix")
            st.code(results["fixed_code"], language="python")
        
        st.divider()
        
        if not st.session_state.feedback_submitted:
            st.subheader("ğŸ’¬ Provide Feedback")
            st.caption("Help CodeSentry learn from your team's decisions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("âœ… Accept - Real Vulnerability", use_container_width=True, type="primary", key="accept_btn"):
                    st.session_state.show_feedback_form = True
                    st.session_state.feedback_type = "accept"
                    st.rerun()
            
            with col2:
                if st.button("â�Œ Reject - False Positive", use_container_width=True, key="reject_btn"):
                    st.session_state.show_feedback_form = True
                    st.session_state.feedback_type = "reject"
                    st.rerun()
        
        if st.session_state.show_feedback_form and not st.session_state.feedback_submitted:
            st.write("---")
            with st.form("feedback_form", clear_on_submit=True):
                st.write("### ğŸ“� Additional Context")
                
                if st.session_state.feedback_type == "accept":
                    st.info("You're confirming this is a real security vulnerability.")
                else:
                    st.warning("You're marking this as a false positive.")
                
                feedback_reason = st.text_area(
                    "Why are you accepting/rejecting this finding?",
                    placeholder="e.g., 'We only use MD5 for file checksums, not password hashing'",
                    help="This helps CodeSentry learn your team's security policies"
                )
                
                submit_col1, submit_col2 = st.columns([1, 3])
                with submit_col1:
                    submit_feedback_btn = st.form_submit_button("Submit Feedback", use_container_width=True)
                with submit_col2:
                    cancel_btn = st.form_submit_button("Cancel", use_container_width=True)
                
                if cancel_btn:
                    st.session_state.show_feedback_form = False
                    st.session_state.feedback_type = None
                    st.rerun()
                
                if submit_feedback_btn:
                    if feedback_reason:
                        user_accepts = (st.session_state.feedback_type == "accept")
                        message = submit_feedback(vuln_id, user_accepts, feedback_reason)
                        
                        st.success(message)
                        
                        if not user_accepts:
                            st.info("ğŸ§  CodeSentry will remember this pattern and reduce similar false positives in the future!")
                        
                        st.session_state.feedback_submitted = True
                        st.session_state.show_feedback_form = False
                        st.session_state.feedback_type = None
                        
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("Please provide a reason for your decision")
        
        if st.session_state.feedback_submitted:
            st.success("âœ… Feedback recorded! Stats updated in sidebar.")

st.divider()
st.caption("ğŸ’¡ **Tip:** CodeSentry learns from your feedback. The more you use it, the smarter it becomes!")
st.caption(f"ğŸ”§ **Team:** {st.session_state.team_context['project_name']} v{st.session_state.team_context['version']} | Last updated: {st.session_state.team_context['last_updated'][:10]}")
'''

# Write the file
with open("codesentry_app.py", "w") as f:
    f.write(streamlit_app_code)

print("âœ… CodeSentry app created successfully!")


!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip



!unzip ngrok-stable-linux-amd64.zip



from pyngrok import ngrok
ngrok.set_auth_token(NGROK_KEY)  


import subprocess
import time
from pyngrok import ngrok

# Start Streamlit in background
proc = subprocess.Popen(
    ["streamlit", "run", "codesentry_app.py", "--server.port", "8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for Streamlit to start
time.sleep(10)

# Create ngrok tunnel
public_url = ngrok.connect(8501)
print(f"ğŸŒ� CodeSentry is running at: {public_url}")
print(f"ğŸ“± Click the link above to open the app!")


print("\n" + "="*70)
print("ğŸ§  TEAM KNOWLEDGE BASE (Memory System)")
print("="*70 + "\n")

print(f"Project: {TEAM_CONTEXT['project_name']}")
print(f"Version: {TEAM_CONTEXT['version']}")
print(f"Security Level: {TEAM_CONTEXT['security_level'].upper()}")
print(f"\nAllowed Patterns (Learned from {len(TEAM_CONTEXT['feedback_history'])} feedback items):\n")

for i, pattern in enumerate(TEAM_CONTEXT['allowed_patterns'], 1):
    print(f"{i}. [{pattern['id']}] {pattern['reason']}")
    print(f"   Confidence: {pattern['confidence']*100:.0f}% | Scope: {pattern['file_scope']}")
    print(f"   Added by: {pattern.get('added_by', 'unknown')} on {pattern.get('date_added', 'N/A')[:10]}\n")

if TEAM_CONTEXT['feedback_history']:
    accepted = sum(1 for f in TEAM_CONTEXT['feedback_history'] if f['accepted'])
    total = len(TEAM_CONTEXT['feedback_history'])
    print(f"\nğŸ“Š Overall Fix Acceptance Rate: {accepted}/{total} ({accepted/total*100:.0f}%)")

