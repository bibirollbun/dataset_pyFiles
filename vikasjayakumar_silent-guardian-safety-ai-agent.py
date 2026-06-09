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



# Install dependencies (if not already installed)
# !pip install google-adk


# Configure API Key
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    # Fallback for local development
    if "GOOGLE_API_KEY" in os.environ:
        print("âœ… Using environment variable for API key.")
    else:
        print(f"âš ï¸� API key not found. Please set GOOGLE_API_KEY environment variable.")



# Import required libraries
import os
import json
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

print("âœ… All imports successful! . \n")


# --- GLOBAL UTILITY FUNCTIONS (must be defined before anything else) ---
import uuid
from datetime import datetime

def print_header(title: str):
    print("\n" + "="*12 + f" {title} " + "="*12)

def now_ts():
    return datetime.utcnow().isoformat() + "Z"

def gen_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
print("âœ… Utility functions loaded successfully at", now_ts())


# -----------------------------
# Data models 
# -----------------------------
print_header("Data models")

@dataclass
class Message:
    msg_id: str
    sender_id: str
    text: str
    ts: str
    attachments: List[str] = field(default_factory=list)

@dataclass
class Conversation:
    convo_id: str
    platform: str
    messages: List[Message]
    metadata: Dict[str,Any] = field(default_factory=dict)   

@dataclass
class Incident:
    incident_id: str
    convo_id: str
    involved_user_ids: List[str]
    start_ts: str
    end_ts: str
    labels: List[Dict[str, Any]]
    severity: str
    evidence: Optional[str]
    status: str

print("Data models ready.")



# -----------------------------
# Helpers
# -----------------------------

def now_ts():
    return datetime.utcnow().isoformat() + "Z"

def gen_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def print_header(title: str):
    print('\n' + '='*8 + ' ' + title + ' ' + '='*8)

# Simple anonymize
PII_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PII_PHONE_RE = re.compile(r"\b\+?\d[\d \-]{6,}\d\b")

def anonymize_text(text: str) -> str:
    t = PII_EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    t = PII_PHONE_RE.sub("[PHONE_REDACTED]", t)
    return t


# -----------------------------
# Configuration / MCP / Tools
# -----------------------------
class ModelControlPlane:
    def __init__(self):
        # switch between 'rule' or 'llm' (llm is a simulated rich responder)
        self.registry = {'classifier': {'type':'llm','version':'v1'}}
    def get(self, name):
        return self.registry.get(name)
    def set(self, name, cfg):
        self.registry[name] = cfg

mcp = ModelControlPlane()

class Tool:
    def run(self, *args, **kwargs):
        raise NotImplementedError

class GoogleSearchTool(Tool):
    def __init__(self, api_key=None):
        self.api_key = api_key
    def run(self, query):
        # demo-only: no external calls. Return placeholder structure.
        print(f"[GoogleSearchTool] (simulated) run: {query}")
        return {'query': query, 'hits': []}

tool_registry = {'google_search': GoogleSearchTool(os.environ.get('GOOGLE_API_KEY'))}

print("\n===== MCP / TOOL REGISTRY STATUS =====")
print("MCP Classifier Config :", mcp.get("classifier"))
print("Registered Tools      :", list(tool_registry.keys()))
print("Google API Key Loaded :", bool(os.environ.get('GOOGLE_API_KEY')))
print("=======================================\n")



# -----------------------------
# In-memory DB (no files)
# -----------------------------
print_header("In-memory Memory DB")
# Use an in-memory SQLite DB so nothing is written to disk; persists for the notebook session
conn = sqlite3.connect(':memory:')
cur = conn.cursor()
cur.execute('''
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    convo_id TEXT,
    involved_users TEXT,
    start_ts TEXT,
    end_ts TEXT,
    labels TEXT,
    severity TEXT,
    evidence TEXT,
    status TEXT
)
''')
cur.execute('''
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    anon_id TEXT,
    safety_score REAL,
    incidents TEXT
)
''')
conn.commit()
print("âœ… In-memory SQLite DB ready (no files created).")


# -----------------------------
# Agent 1: Memory Agent
#Stores incident history for each user to support recidivism scoring.
#Allows the system to get â€œsmarterâ€� with repeated interactions.
# -----------------------------

print_header("Memory Agent")
class MemoryAgent:
    def __init__(self, conn):
        self.conn = conn
    def append_incident(self, user_id, incident_id):
        cur = self.conn.cursor()
        cur.execute("SELECT incidents FROM user_profiles WHERE user_id=?", (user_id,))
        r = cur.fetchone()
        if not r:
            cur.execute("INSERT INTO user_profiles (user_id, anon_id, safety_score, incidents) VALUES (?,?,?,?)",
                        (user_id, gen_id('anon'), 0.5, json.dumps([incident_id])))
        else:
            incs = json.loads(r[0] or '[]')
            incs.append(incident_id)
            cur.execute("UPDATE user_profiles SET incidents=? WHERE user_id= ?", (json.dumps(incs), user_id))
        self.conn.commit()
    def get_incidents(self, user_id):
        cur = self.conn.cursor()
        cur.execute("SELECT incidents FROM user_profiles WHERE user_id=?", (user_id,))
        r = cur.fetchone()
        return json.loads(r[0]) if r and r[0] else []

memory_agent = MemoryAgent(conn)
print("MemoryAgent ready.")



# -----------------------------
#Agents 2 and 3 : Ingestor, Extractor
#Ingestor - Converts raw chat messages into a clean, structured conversation format and ensures every downstream agent receives consistent and validated inputs.
#Extractor - Selects the most relevant messages for analysis based on timestamps and prevents unnecessary processing and focuses the classifier on the correct slice of data.
# -----------------------------

print_header("Ingestor & Extractor")
class IngestorAgent:
    def __init__(self):
        print("[Ingestor] init")
    def ingest_stream(self, raw_stream: List[Dict[str,Any]], platform: str='slack') -> Conversation:
        print("[Ingestor] ingest_stream called")
        convo_id = gen_id('convo')
        messages = []
        for m in raw_stream:
            msg = Message(
                msg_id=m.get('msg_id', gen_id('msg')),
                sender_id=m.get('sender_id','unknown'),
                text=anonymize_text(m.get('text','')),
                ts=m.get('ts', now_ts()),
                attachments=m.get('attachments', [])
            )
            messages.append(msg)
        conv = Conversation(convo_id=convo_id, platform=platform, messages=messages)
        print(f"[Ingestor] Produced conversation {convo_id} with {len(messages)} messages")
        return conv

class ExtractorAgent:
    def __init__(self):
        print("[Extractor] init")
    def extract_window(self, conv: Conversation, center_msg_idx:int=0, window_seconds:int=300):
        print("[Extractor] extract_window called")
        if not conv.messages:
            return []
        center_ts = datetime.fromisoformat(conv.messages[center_msg_idx].ts.replace('Z',''))
        lower = (center_ts - timedelta(seconds=window_seconds)).isoformat() + 'Z'
        upper = (center_ts + timedelta(seconds=window_seconds)).isoformat() + 'Z'
        window_msgs = [m for m in conv.messages if lower <= m.ts <= upper]
        print(f"[Extractor] Window size: {len(window_msgs)} (ts range {lower} - {upper})")
        return window_msgs

ingestor = IngestorAgent()
extractor = ExtractorAgent()
print("Ingestor and Extractor Agent ready.")


# -----------------------------
# Agent 4 : Classifier: MCP-aware, scalable "LLM-sim" mode
#Applies multi-category harassment detection using rule-based and simulated LLM logic.
#Acts as the first decision point that labels message severity and meaning.
# -----------------------------

print_header("Harassment Classifier (MCP-aware, LLM-sim)")
class HarassmentClassifierAgent:
    def __init__(self):
        print("[HarassmentClassifier] init")
        # small lexicons kept for rule-mode fallback
        self.insult_keywords = set(["stupid","idiot","worthless","loser","dumb","hate you","shut up"]) 
        self.threat_keywords = set(["kill you","hurt you","i'll kill","i will kill","die","end you"]) 
        self.doxxing_keywords = set(["address","phone","where do you live","i know where you live","share her number"]) 

    def classify_messages(self, messages: List[Message]) -> List[Dict[str,Any]]:
        cfg = mcp.get('classifier') or {'type':'rule'}
        mode = cfg.get('type','rule')
        print(f"[HarassmentClassifier] classify_messages called (mode={mode})")
        outputs = []
        for m in messages:
            if mode == 'rule':
                out = self._rule_classify(m)
            else:
                out = self._llm_sim_classify(m)
            outputs.append(out)
            print(f"[HarassmentClassifier] msg:{m.msg_id[:8]} labels:{[l['label'] for l in out['labels']]}")
        return outputs

    def _rule_classify(self, m: Message):
        text_lower = m.text.lower() if m.text else ''
        labels = []
        if any(k in text_lower for k in self.insult_keywords):
            labels.append({'label':'insult','score':0.9,'span': self._find_span(text_lower,self.insult_keywords)})
        if any(k in text_lower for k in self.threat_keywords):
            labels.append({'label':'threat','score':0.95,'span': self._find_span(text_lower,self.threat_keywords)})
        if any(k in text_lower for k in self.doxxing_keywords):
            labels.append({'label':'doxxing','score':0.7,'span': self._find_span(text_lower,self.doxxing_keywords)})
        if not labels:
            labels = [{'label':'none','score':0.0,'span':''}]
        return {'msg_id': m.msg_id, 'sender_id': m.sender_id, 'labels': labels, 'text': m.text}

    def _llm_sim_classify(self, m: Message):
        # Simulated LLM response: generate a rich JSON and a natural-language explanation.
        txt = (m.text or '').strip()
        labels = []
        explanation = []
        score = 0.0
        # heuristics but produce richer textual reasoning
        if any(k in txt.lower() for k in self.threat_keywords):
            labels.append({'label':'threat','score':0.98,'span': self._find_span(txt.lower(), self.threat_keywords)})
            explanation.append('Message contains explicit threat language.')
            score = max(score, 0.98)
        if any(k in txt.lower() for k in self.doxxing_keywords):
            labels.append({'label':'doxxing','score':0.8,'span': self._find_span(txt.lower(), self.doxxing_keywords)})
            explanation.append('Possible doxxing-related phrasing detected.')
            score = max(score, 0.8)
        if any(k in txt.lower() for k in self.insult_keywords) or re.search(r"\bidiot\b|\bdumb\b|\bworthless\b", txt.lower()):
            labels.append({'label':'insult','score':0.9,'span': self._find_span(txt.lower(), self.insult_keywords)})
            explanation.append('Insulting / demeaning language present.')
            score = max(score, 0.9)
        # If no label found, ask a clarifying simulated question in the output for interactive queries
        if not labels:
            labels = [{'label':'none','score':0.0,'span':''}]
            explanation.append('No clear harassment label detected; message appears benign or ambiguous.')
        # Build a natural language summary as the "assistant" output for user queries
        nl = f"Classifier (LLM-sim) analysis: labels={','.join([l['label'] for l in labels])}; rationale={' | '.join(explanation)}"
        return {'msg_id': m.msg_id, 'sender_id': m.sender_id, 'labels': labels, 'text': m.text, 'explanation': nl}

    def _find_span(self, text: str, lexicon: set):
        for k in lexicon:
            if k in text:
                return k
        return ''

classifier = HarassmentClassifierAgent()


# -----------------------------
# Agent 5 : Pattern Detector (uses MemoryAgent)
#Finds repeat targeting, mentions, and historical recidivism using MemoryAgent.
#Helps the system understand whether an offender is repeating harmful behavior.
# -----------------------------

print_header("Pattern Detector")
class PatternDetectorAgent:
    def __init__(self, memory_agent: MemoryAgent):
        print("[PatternDetector] init")
        self.memory = memory_agent
    def detect_patterns(self, conv: Conversation, classified_msgs: List[Dict[str,Any]]) -> Dict[str,Any]:
        print("[PatternDetector] detect_patterns called")
        patterns = {'repeat_targeting': [], 'recidivism': {}}
        for cm in classified_msgs:
            labels = [l['label'] for l in cm['labels'] if l['label']!='none']
            if labels:
                sender = cm['sender_id']
                potential_targets = re.findall(r"@\w+", cm.get('text') or '')
                for t in potential_targets:
                    patterns['repeat_targeting'].append({'sender': sender, 'target': t, 'msg_id': cm['msg_id']})
                # check memory for historical incidents
                prev = self.memory.get_incidents(sender)
                patterns['recidivism'][sender] = len(prev)
        print(f"[PatternDetector] found patterns: {patterns}")
        return patterns

pattern_detector = PatternDetectorAgent(memory_agent)
print("Pattern Detector Agent ready.")


# -----------------------------
# Agent 6 : Risk Scorer (recidivism-aware)
#Converts labels and patterns into a numerical risk score and severity class.
#Decides whether a situation is Low, High, or Immediate risk.
# -----------------------------

print_header("Risk Scorer")
class RiskScorerAgent:
    def __init__(self):
        print("[RiskScorer] init")
    def compute_risk(self, classified_msgs: List[Dict[str,Any]], patterns: Dict[str,Any]) -> Dict[str,Any]:
        print("[RiskScorer] compute_risk called")
        base_scores = [max((l.get('score',0.0) for l in cm['labels'])) for cm in classified_msgs] if classified_msgs else []
        avg_score = sum(base_scores)/len(base_scores) if base_scores else 0.0
        repeat_count = len(patterns.get('repeat_targeting', []))
        if repeat_count >= 3:
            avg_score = min(1.0, avg_score + 0.2)
        # recidivism bump
        recidivism_bump = 0.0
        for sender, count in patterns.get('recidivism', {}).items():
            if count > 0:
                recidivism_bump = max(recidivism_bump, min(0.3, 0.05 * count))
        if recidivism_bump > 0:
            print(f"[RiskScorer] applying recidivism bump: {recidivism_bump}")
            avg_score = min(1.0, avg_score + recidivism_bump)
        severity = 'Low'
        if avg_score >= 0.8:
            severity = 'Immediate'
        elif avg_score >= 0.5:
            severity = 'High'
        elif avg_score >= 0.2:
            severity = 'Medium'
        risk = {'score': avg_score, 'severity': severity}
        print(f"[RiskScorer] score={avg_score:.2f} severity={severity}")
        return risk

risk_scorer = RiskScorerAgent()
print("Risk scorer Agent ready.")



# -----------------------------
# Agent 7, 8 and 9 :  Ethics, Planner, Notifier
# Ethics - Blocks unsafe or inappropriate actions before they reach the moderato and ensures all interventions comply with safety, fairness, and ethical guidelines.
# Planner - Chooses the best action to take (notify HR, generate evidence, support message) and applies decision rules based on severity, threat level, and patterns.
# Notifier - Handles human-in-the-loop communication by alerting moderators during escalations and sends supportive or guidance messages directly to affected users when appropriate.
# -----------------------------

print_header("Ethics / Planner / Notifier")
class EthicsAgent:
    def __init__(self, policy: Dict[str,Any]):
        self.policy = policy
    def check(self, action: Dict[str,Any]) -> Dict[str,Any]:
        if action.get('action_type') in ['ban_user','fire_employee','legal_escalation']:
            return {'allowed': False, 'reason':'Human approval required'}
        if action.get('action_type')=='auto_message' and not self.policy.get('allow_auto_messages', False):
            return {'allowed': False, 'reason':'Auto messages disabled'}
        return {'allowed': True}

ethics = EthicsAgent({'allow_auto_messages': True})

class InterventionPlannerAgent:
    def plan(self, conv, classified_msgs, risk, patterns):
        severity = risk['severity']
        actions = []
        if severity=='Low':
            actions.append({'action_type':'suggest_support_message','rationale':'Offer support','message_template':'I noticed a tense exchange â€” are you okay?'})
        elif severity=='Medium':
            actions.append({'action_type':'notify_moderator','rationale':'Moderator review recommended','message_template':''})
        else:
            actions.append({'action_type':'create_incident_and_notify_hr','rationale':'Escalate to HR','message_template':''})
            actions.append({'action_type':'generate_evidence_text','rationale':'Create evidence text','message_template':''})
        print(f"[InterventionPlanner] planned actions: {[a['action_type'] for a in actions]}")
        return actions

planner = InterventionPlannerAgent()

class NotifierAgent:
    def notify_moderator(self, incident, actions):
        print(f"--- Moderator Notification: Incident {incident.incident_id} (severity={incident.severity}) ---")
        for i,a in enumerate(actions):
            print(f"[{i}] Action: {a['action_type']} - {a.get('rationale','')}")
    def send_supportive_message(self, user_id, msg):
        print(f"[Notifier] Supportive message to {user_id}: {msg}")

notifier = NotifierAgent()
print("Ethics Planner and Notifier Agent ready.")




# -----------------------------
# Agent 10 : Evidence Builder (no files: returns/prints evidence text)
#Creates structured text evidence for incidents needing HR escalation.
#Includes message history, labels, severity, and reasoning.
# -----------------------------

print_header("Evidence Builder")
class EvidenceBuilderAgent:
    def build_evidence_text(self, conv: Conversation, classified_msgs: List[Dict[str,Any]], risk: Dict[str,Any]) -> str:
        lines = []
        lines.append(f"Evidence for conversation {conv.convo_id} (platform={conv.platform})")
        lines.append(f"Generated: {now_ts()}")
        lines.append(f"Risk: {risk}")
        lines.append('\nMessages:')
        for cm in classified_msgs:
            lines.append(f"- {cm['msg_id']} | {cm['sender_id']} | {cm.get('text','')} | labels={cm['labels']}")
            if 'explanation' in cm:
                lines.append(f"  explanation: {cm['explanation']}")
        txt = '\n'.join(lines)
        print('\n' + txt + '\n')
        return txt

evidence_builder = EvidenceBuilderAgent()
print("Evidence Builder Agent ready.")



# -----------------------------
# Agent 11 : Observability (in-memory)
# Captures logs for every stage of the pipeline (ingestion â†’ ethics â†’ finish).
# Provides complete transparency for debugging, evaluation, and audits.
# -----------------------------

print_header("Observability")
class ObservabilityAgent:
    def __init__(self):
        self.logs = []
    def log(self, entry: Dict[str,Any]):
        entry['ts'] = now_ts()
        self.logs.append(entry)
        print(f"[Observability] {entry.get('event','unknown')}")
    def dump(self):
        # return logs for display in notebook
        return list(self.logs)

observability = ObservabilityAgent()
print("Observability Agent ready.")


# -----------------------------------------------------------------------------
# COMBINED PATCH â€” Overview & developer notes
#
# Summary:
# This patch upgrades the notebook with production-ready demo features:
#  - Replaces the classifier with an MCP-aware hybrid implementation (HarassmentClassifierAgent_MCP)
#    that can run in fast 'rule' mode or a conservative 'llm' (simulated) mode via the MCP.
#  - Installs PatternDetectorAgentV2 which uses memory (recidivism lookup) to detect repeat targeting.
#  - Installs RiskScorerAgentV2 which applies repeat-targeting and recidivism bumps to risk scores.
#  - Adds a lightweight MessageBus (pub/sub) for agent-to-agent communication and safe async handlers.
#  - Exposes a FastAPI ingest stub for integration testing.
#  - Monkeypatches pipeline.run_once to ensure MemoryAgent is updated after incidents are created.
#
# Why this patch:
#  - Enables MCP-driven experiments (flip classifier mode via mcp.set(...)).
#  - Improves detection quality by incorporating historical incidents (recidivism).
#  - Demonstrates agent-to-agent messaging and pluggable tool usage (google_search stub).
#  - Keeps the notebook self-contained and safe for Kaggle (no external LLM calls).
#
# How to use / test:
#  1. Flip classifier mode: mcp.set('classifier', {'type':'rule'})  (or {'type':'llm'} for demo).
#  2. Run a single abusive stream multiple times (3+) â€” observe risk bump from repeat-targeting.
#  3. Check user memory table and incidents: SELECT * FROM user_profiles / incidents (or use cur.execute).
# -----------------------------------------------------------------------------



# ------------------ COMBINED PATCH CELL ------------------
print_header("Applying combined patch: MCP-aware classifier, PatternDetectorV2, RiskScorerV2, MessageBus, pipeline memory update")

# 1) MCP-aware classifier (safe replacement)
class HarassmentClassifierAgent_MCP:
    def __init__(self):
        print("[HarassmentClassifier_MCP] init")
        self.insult_keywords = set(["stupid","idiot","worthless","loser","dumb","hate you","shut up"])
        self.threat_keywords = set(["kill you","hurt you","i'll kill","i will kill","die","end you"])
        self.doxxing_keywords = set(["address","phone","where do you live","i know where you live","share her number"])

    def classify_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        print("[HarassmentClassifier_MCP] classify_messages called")
        outputs = []
        cfg = mcp.get('classifier') if 'mcp' in globals() else {'type':'rule'}
        mode = cfg.get('type', 'rule')
        for m in messages:
            if mode == 'rule':
                text_lower = m.text.lower()
                labels = []
                if any(k in text_lower for k in self.insult_keywords):
                    labels.append({'label':'insult', 'score':0.9, 'span': self._find_span(text_lower, self.insult_keywords)})
                if any(k in text_lower for k in self.threat_keywords):
                    labels.append({'label':'threat', 'score':0.95, 'span': self._find_span(text_lower, self.threat_keywords)})
                if any(k in text_lower for k in self.doxxing_keywords):
                    labels.append({'label':'doxxing', 'score':0.7, 'span': self._find_span(text_lower, self.doxxing_keywords)})
                if not labels:
                    labels = [{'label':'none', 'score':0.0, 'span':''}]
                out = {'msg_id': m.msg_id, 'sender_id': m.sender_id, 'labels': labels, 'text': m.text}
                outputs.append(out)
                print(f"[HarassmentClassifier_MCP] msg:{m.msg_id[:8]} labels:{[l['label'] for l in labels]}")
            else:
                # LLM mode placeholder: show prompt, optionally call a safe tool
                prompt = f"Classify the following message for harassment categories: {m.text}"
                print(f"[HarassmentClassifier_MCP][LLM-mode] prompt: {prompt[:120]}")
                tool_result = None
                if 'tool_registry' in globals() and tool_registry.get('google_search'):
                    tool_result = tool_registry['google_search'].run(m.text)
                # conservative fallback
                labels = [{'label':'none', 'score':0.0, 'span':''}]
                out = {'msg_id': m.msg_id, 'sender_id': m.sender_id, 'labels': labels, 'text': m.text, 'tool_result': tool_result}
                outputs.append(out)
                print(f"[HarassmentClassifier_MCP][LLM-mode] msg:{m.msg_id[:8]} labels:{[l['label'] for l in labels]}")
        return outputs

    def _find_span(self, text: str, lexicon: set):
        for k in lexicon:
            if k in text:
                return k
        return ''

# create and attach
classifier = HarassmentClassifierAgent_MCP()
print("Replaced classifier with HarassmentClassifierAgent_MCP")
mcp.set('classifier', {'type':'rule','version':'v0'})
try:
    pipeline.classifier = classifier
    print("pipeline.classifier updated.")
except Exception:
    print("pipeline not present or not yet instantiated; classifier object ready.")

# 2) PatternDetector V2 (uses memory_agent.get_incidents if available)
class PatternDetectorAgentV2:
    def __init__(self, memory_conn):
        print("[PatternDetectorV2] init")
        self.conn = memory_conn

    def detect_patterns(self, conv: Conversation, classified_msgs: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("[PatternDetectorV2] detect_patterns called")
        patterns = {'repeat_targeting': [], 'recidivism': {}}
        for cm in classified_msgs:
            labels = [l['label'] for l in cm['labels'] if l['label'] != 'none']
            if labels:
                sender = cm['sender_id']
                potential_targets = re.findall(r"@\w+", cm.get('text','') or '')
                for t in potential_targets:
                    patterns['repeat_targeting'].append({'sender': sender, 'target': t, 'msg_id': cm['msg_id']})
                # historical incidents via memory_agent
                if 'memory_agent' in globals():
                    try:
                        prev_incs = memory_agent.get_incidents(sender)
                        patterns['recidivism'][sender] = len(prev_incs)
                    except Exception as e:
                        print("[PatternDetectorV2] memory_agent.get_incidents error:", e)
        print(f"[PatternDetectorV2] found patterns: {patterns}")
        return patterns

pattern_detector = PatternDetectorAgentV2(conn)
print("Instantiated PatternDetectorAgentV2 and assigned to 'pattern_detector'")

# 3) RiskScorer V2 (applies recidivism bump)
class RiskScorerAgentV2:
    def __init__(self):
        print("[RiskScorerV2] init")

    def compute_risk(self, classified_msgs: List[Dict[str, Any]], patterns: Dict[str, Any]) -> Dict[str, Any]:
        print("[RiskScorerV2] compute_risk called")
        base_scores = []
        for cm in classified_msgs:
            max_score = max(l.get('score', 0.0) for l in cm['labels'])
            base_scores.append(max_score)
        avg_score = sum(base_scores) / len(base_scores) if base_scores else 0.0
        repeat_count = len(patterns.get('repeat_targeting', []))
        if repeat_count >= 3:
            avg_score = min(1.0, avg_score + 0.2)
        # recidivism bump
        recidivism_bump = 0.0
        if 'recidivism' in patterns:
            for sender, count in patterns['recidivism'].items():
                if count >= 1:
                    recidivism_bump = max(recidivism_bump, min(0.3, 0.05 * count))
        if recidivism_bump > 0:
            print(f"[RiskScorerV2] applying recidivism bump: {recidivism_bump}")
            avg_score = min(1.0, avg_score + recidivism_bump)
        severity = 'Low'
        if avg_score >= 0.8:
            severity = 'Immediate'
        elif avg_score >= 0.5:
            severity = 'High'
        elif avg_score >= 0.2:
            severity = 'Medium'
        risk = {'score': avg_score, 'severity': severity}
        print(f"[RiskScorerV2] score={avg_score:.2f} severity={severity}")
        return risk

risk_scorer = RiskScorerAgentV2()
print("Instantiated RiskScorerAgentV2 and assigned to 'risk_scorer'")

# Reattach components to pipeline if available
try:
    pipeline.pattern_detector = pattern_detector
    pipeline.risk_scorer = risk_scorer
    pipeline.classifier = classifier
    print("Reattached pattern_detector, risk_scorer, classifier to pipeline.")
except Exception:
    print("Pipeline not present; components ready to attach when pipeline is re-created.")

# 4) MessageBus A2A pub/sub
class MessageBus:
    def __init__(self):
        self.handlers = {}
    def register(self, topic, fn):
        self.handlers.setdefault(topic, []).append(fn)
    def publish(self, topic, payload):
        print(f"[MessageBus] publish: {topic}")
        for h in self.handlers.get(topic, []):
            try:
                h(payload)
            except Exception as e:
                print(f"[MessageBus] handler error: {e}")

bus = MessageBus()
print("MessageBus initialized.")

# Example handler: on classified messages, run async pattern detection (demo)
def _pattern_handler(payload):
    try:
        conv = payload.get('conv')
        classified = payload.get('classified')
        pd = pattern_detector.detect_patterns(conv, classified)
        print("[MessageBus pattern_handler] detected patterns:", pd)
    except Exception as e:
        print("[MessageBus pattern_handler] error:", e)

# register handler (safe if pattern_detector exists)
if 'pattern_detector' in globals():
    bus.register('classified_messages', _pattern_handler)
    print("Registered pattern_handler on topic 'classified_messages'")

# 5) FastAPI stub
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    app = FastAPI()
    class IngestPayload(BaseModel):
        messages: list
    @app.post('/ingest')
    def ingest_endpoint(payload: IngestPayload):
        stream = payload.messages
        incident = pipeline.run_once(stream)
        return {'incident_id': incident.incident_id, 'severity': incident.severity}
    print("FastAPI stub defined. (Not running server in notebook.)")
except Exception:
    print("FastAPI not available in this environment (optional).")

# 6) Monkeypatch pipeline.run_once to update MemoryAgent after incident creation
try:
    if 'pipeline' in globals():
        old_run = pipeline.run_once
        def _run_and_update_memory(raw_stream):
            # call existing pipeline logic (which returns an Incident)
            incident = old_run(raw_stream)
            # update memory agent for involved users if available
            try:
                if 'memory_agent' in globals():
                    for u in incident.involved_user_ids:
                        try:
                            memory_agent.append_incident(u, incident.incident_id)
                        except Exception as e:
                            print("[pipeline memory update] append_incident error:", e)
                    print("[pipeline memory update] updated memory_agent for involved users.")
            except Exception as e:
                print("[pipeline memory update] unexpected error:", e)
            return incident
        pipeline.run_once = _run_and_update_memory
        print("Pipeline.run_once monkeypatched: will now update memory_agent after creating incidents.")
    else:
        print("No pipeline object found; cannot monkeypatch run_once. Create pipeline then re-run this cell.")
except Exception as e:
    print("Error patching pipeline.run_once:", e)

print_header("COMBINED PATCH APPLIED - Test suggestion")
print("Suggested test: run one abusive stream multiple times (3+) to see recidivism bump in RiskScorerV2.")
print("To flip classifier to LLM-mode (demo), run: mcp.set('classifier', {'type':'llm','version':'v1'})")


# --- FIX: Ensure planner exists before pipeline creation ---
# Purpose: guard against NameError when pipeline is instantiated before certain agents.
# Behavior:
#  - Check if 'planner', 'ethics', 'notifier', etc. are in globals()
#  - If missing, create minimal safe stub objects (no-op or logging-only)
#  - This lets pipeline be instantiated reliably while dev iterates on agents



# --- FIX: Ensure planner exists before pipeline creation ---
if 'planner' not in globals():
    print("planner not found â€” creating a default InterventionPlannerAgent()")

    class InterventionPlannerAgent:
        def __init__(self):
            print("[InterventionPlanner] init")

        def plan(self, conv, classified_msgs, risk, patterns):
            severity = risk['severity']
            actions = []
            if severity == 'Low':
                actions.append({
                    'action_type':'suggest_support_message',
                    'rationale':'Low severity; offer support to target'
                })
            elif severity == 'Medium':
                actions.append({
                    'action_type':'notify_moderator',
                    'rationale':'Medium severity; moderator review recommended'
                })
            else:
                actions.append({
                    'action_type':'create_incident_and_notify_hr',
                    'rationale':'High/Immediate severity; escalate to HR'
                })
            return actions

    planner = InterventionPlannerAgent()

print("planner is ready:", planner)



# --- Safe Bootstrap Cell ---
# Purpose: create deterministic minimal runtime baseline.
# Behavior:
#  - sets up ModelControlPlane (mcp) and tool_registry
#  - creates or connects to memory DB/connection placeholder
#  - defines lightweight MemoryAgent stub if not defined.



# -----------------------------
# SAFE BOOTSTRAP CELL â€” create missing agents & instantiate pipeline
# -----------------------------
import types, inspect

print("\n==== SAFE BOOTSTRAP: ensuring required agents & pipeline ====")

# Helper: create a minimal default implementation only if missing
def ensure(name, creator_fn):
    if name in globals() and globals()[name] is not None:
        print(f"âœ” {name} already present")
        return globals()[name]
    else:
        obj = creator_fn()
        globals()[name] = obj
        print(f"âœ” Created default {name}")
        return obj

# Default simple classes (only created if the user hasn't defined them)
def make_ethics():
    class EthicsAgent:
        def __init__(self, policy=None):
            self.policy = policy or {'allow_auto_messages': True}
            print("[Ethics] default created")
        def check(self, action):
            if action.get('action_type') in ['ban_user','fire_employee','legal_escalation']:
                return {'allowed': False, 'reason': 'Human approval required'}
            if action.get('action_type') == 'auto_message' and not self.policy.get('allow_auto_messages', False):
                return {'allowed': False, 'reason': 'Auto messages disabled'}
            return {'allowed': True}
    return EthicsAgent({'allow_auto_messages': True})

def make_notifier():
    class NotifierAgent:
        def notify_moderator(self, incident, actions):
            try:
                iid = incident.incident_id
            except Exception:
                # incident may be a tuple â€” be resilient
                try:
                    iid = incident[0].incident_id
                except Exception:
                    iid = getattr(incident, 'incident_id', '<unknown>')
            print(f"[Notifier] Incident {iid} notify (simulated). Actions: {[a['action_type'] for a in actions]}")
        def send_supportive_message(self, user_id, msg):
            print(f"[Notifier] Supportive message to {user_id}: {msg}")
    return NotifierAgent()

def make_observability():
    class ObservabilityAgent:
        def __init__(self):
            self.logs = []
            print("[Observability] default created")
        def log(self, entry):
            entry = dict(entry)
            entry['ts'] = datetime.utcnow().isoformat() + "Z"
            self.logs.append(entry)
            print(f"[Observability] {entry.get('event','unknown')}")
        def dump(self):
            return list(self.logs)
    return ObservabilityAgent()

def make_planner():
    class InterventionPlannerAgent:
        def __init__(self):
            print("[InterventionPlanner] default created")
        def plan(self, conv, classified_msgs, risk, patterns):
            severity = risk.get('severity') if isinstance(risk, dict) else 'Low'
            actions = []
            if severity == 'Low':
                actions.append({'action_type':'suggest_support_message','rationale':'Offer support'})
            elif severity == 'Medium':
                actions.append({'action_type':'notify_moderator','rationale':'Moderator review recommended'})
            else:
                actions.append({'action_type':'create_incident_and_notify_hr','rationale':'Escalate to HR'})
                actions.append({'action_type':'generate_evidence_text','rationale':'Create evidence text'})
            return actions
    return InterventionPlannerAgent()

def make_pattern_detector():
    class PatternDetectorAgent:
        def __init__(self, memory_agent=None):
            self.memory = memory_agent
            print("[PatternDetector] default created")
        def detect_patterns(self, conv, classified_msgs):
            patterns = {'repeat_targeting': [], 'recidivism': {}}
            for cm in classified_msgs:
                labels = [l['label'] for l in cm.get('labels',[]) if l.get('label')!='none']
                if labels:
                    sender = cm.get('sender_id')
                    text = cm.get('text','') or ''
                    for t in re.findall(r"@\w+", text):
                        patterns['repeat_targeting'].append({'sender': sender, 'target': t, 'msg_id': cm.get('msg_id')})
                    if self.memory:
                        try:
                            prev = self.memory.get_incidents(sender)
                            patterns['recidivism'][sender] = len(prev)
                        except Exception:
                            patterns['recidivism'][sender] = 0
            return patterns
    return PatternDetectorAgent(memory_agent if 'memory_agent' in globals() else None)

def make_risk_scorer():
    class RiskScorerAgent:
        def __init__(self):
            print("[RiskScorer] default created")
        def compute_risk(self, classified_msgs, patterns):
            base_scores = []
            for cm in classified_msgs:
                s = max((l.get('score',0.0) for l in cm.get('labels',[])), default=0.0)
                base_scores.append(s)
            avg = sum(base_scores)/len(base_scores) if base_scores else 0.0
            repeat_count = len(patterns.get('repeat_targeting',[])) if isinstance(patterns, dict) else 0
            if repeat_count >= 3:
                avg = min(1.0, avg + 0.2)
            # recidivism bump (safe)
            bump = 0.0
            for cnt in (patterns.get('recidivism',{}).values() if isinstance(patterns, dict) else []):
                if cnt>0:
                    bump = max(bump, min(0.3, 0.05*cnt))
            avg = min(1.0, avg + bump)
            severity = 'Low'
            if avg >= 0.8: severity='Immediate'
            elif avg >= 0.5: severity='High'
            elif avg >= 0.2: severity='Medium'
            return {'score': avg, 'severity': severity}
    return RiskScorerAgent()

def make_classifier():
    class HarassmentClassifierAgent:
        def __init__(self):
            self.insult_keywords = set(["stupid","idiot","worthless","loser","dumb","hate you","shut up"])
            self.threat_keywords = set(["kill you","hurt you","i'll kill","i will kill","die","end you"])
            self.doxxing_keywords = set(["address","phone","where do you live","i know where you live","share her number"])
            print("[Classifier] default created")
        def _find_span(self, text, lex):
            for k in lex:
                if k in text: return k
            return ''
        def classify_messages(self, messages):
            outputs=[]
            mode = (mcp.get('classifier') or {}).get('type','rule') if 'mcp' in globals() else 'rule'
            for m in messages:
                t = (m.get('text') if isinstance(m, dict) else getattr(m,'text', '')) if isinstance(m, dict) or hasattr(m,'text') else ''
                text_lower = (t or '').lower()
                labels=[]
                if any(k in text_lower for k in self.threat_keywords):
                    labels.append({'label':'threat','score':0.98,'span':self._find_span(text_lower,self.threat_keywords)})
                if any(k in text_lower for k in self.doxxing_keywords):
                    labels.append({'label':'doxxing','score':0.8,'span':self._find_span(text_lower,self.doxxing_keywords)})
                if any(k in text_lower for k in self.insult_keywords) or re.search(r"\bidiot\b|\bdumb\b|\bworthless\b", text_lower):
                    labels.append({'label':'insult','score':0.9,'span':self._find_span(text_lower,self.insult_keywords)})
                if not labels:
                    labels=[{'label':'none','score':0.0,'span':''}]
                # support both Message objects and dict-like items
                msg_id = (m.get('msg_id') if isinstance(m, dict) else getattr(m, 'msg_id', gen_id('msg')))
                sender_id = (m.get('sender_id') if isinstance(m, dict) else getattr(m, 'sender_id', 'unknown'))
                outputs.append({'msg_id': msg_id, 'sender_id': sender_id, 'labels': labels, 'text': t})
            return outputs
    return HarassmentClassifierAgent()

def make_ingestor():
    class IngestorAgent:
        def ingest_stream(self, raw_stream, platform='slack'):
            convo_id = gen_id('convo')
            messages=[]
            for m in raw_stream:
                mid = m.get('msg_id') if isinstance(m, dict) else getattr(m,'msg_id', gen_id('msg'))
                sender = m.get('sender_id') if isinstance(m, dict) else getattr(m,'sender_id','unknown')
                txt = anonymize_text(m.get('text','') if isinstance(m, dict) else getattr(m,'text',''))
                ts = m.get('ts') if isinstance(m, dict) else getattr(m,'ts', now_ts())
                messages.append({'msg_id': mid, 'sender_id': sender, 'text': txt, 'ts': ts})
            # Return a minimal convo-like object with attributes used downstream
            return types.SimpleNamespace(convo_id=convo_id, platform=platform, messages=[types.SimpleNamespace(**mm) for mm in messages])
    return IngestorAgent()

def make_extractor():
    class ExtractorAgent:
        def extract_window(self, conv, center_msg_idx=0, window_seconds=300):
            if not getattr(conv,'messages',[]): return []
            center_ts = datetime.fromisoformat(conv.messages[center_msg_idx].ts.replace('Z',''))
            lower=(center_ts - timedelta(seconds=window_seconds)).isoformat()+'Z'
            upper=(center_ts + timedelta(seconds=window_seconds)).isoformat()+'Z'
            window_msgs = [m for m in conv.messages if lower <= getattr(m,'ts',now_ts()) <= upper]
            return window_msgs
    return ExtractorAgent()

def make_evidence_builder():
    class EvidenceBuilderAgent:
        def build_evidence_text(self, conv, classified_msgs, risk):
            lines=[f"Evidence for convo {getattr(conv,'convo_id','<conv>')}","Generated: "+now_ts(), f"Risk: {risk}", "Messages:"]
            for cm in classified_msgs:
                lines.append(f"- {cm.get('msg_id')} | {cm.get('sender_id')} | {cm.get('text')} | labels={cm.get('labels')}")
            txt = "\n".join(lines)
            print(txt)
            return txt
    return EvidenceBuilderAgent()

# Ensure memory_agent exists (in-memory SQLite expected by user's notebook)
if 'memory_agent' not in globals():
    # create a tiny MemoryAgent that stores in a dict (non-persistent)
    class MemoryAgentSimple:
        def __init__(self):
            self.mem = {}
            print("[MemoryAgentSimple] created (in-memory dict)")
        def append_incident(self,user_id, incident_id):
            self.mem.setdefault(user_id, []).append(incident_id)
        def get_incidents(self, user_id):
            return list(self.mem.get(user_id, []))
    memory_agent = MemoryAgentSimple()
    globals()['memory_agent'] = memory_agent
else:
    print("âœ” memory_agent present")

# Create/ensure other agents (will not overwrite user-defined ones)
ensure('ethics', lambda: make_ethics())
ensure('notifier', lambda: make_notifier())
ensure('observability', lambda: make_observability())
ensure('planner', lambda: make_planner())
ensure('pattern_detector', lambda: make_pattern_detector())
ensure('risk_scorer', lambda: make_risk_scorer())
ensure('classifier', lambda: make_classifier())
ensure('ingestor', lambda: make_ingestor())
ensure('extractor', lambda: make_extractor())
ensure('evidence_builder', lambda: make_evidence_builder())

# Now define a robust pipeline that tolerates missing/tuple returns
print("\nConstructing SilentGuardianPipeline (robust)...")
class SilentGuardianPipelineRobust:
    def __init__(self):
        self.ingestor = globals().get('ingestor')
        self.extractor = globals().get('extractor')
        self.classifier = globals().get('classifier')
        self.pattern_detector = globals().get('pattern_detector')
        self.risk_scorer = globals().get('risk_scorer')
        self.planner = globals().get('planner')
        self.evidence_builder = globals().get('evidence_builder')
        self.ethics = globals().get('ethics')
        self.notifier = globals().get('notifier')
        self.observability = globals().get('observability')
        self.memory = globals().get('memory_agent')
        print("[PipelineRobust] created with components:",
              "ingestor", bool(self.ingestor),
              "classifier", bool(self.classifier),
              "pattern_detector", bool(self.pattern_detector))

    def run_once(self, raw_stream):
        print("[PipelineRobust] run_once called")
        conv = self.ingestor.ingest_stream(raw_stream)
        # Logging safe
        try:
            self.observability.log({'event':'ingested_conversation','convo_id': conv.convo_id, 'n_messages': len(conv.messages)})
        except Exception:
            pass
        # Extract window (safe)
        try:
            window_msgs = self.extractor.extract_window(conv, center_msg_idx=max(0, len(conv.messages)-1))
        except Exception:
            window_msgs = getattr(conv, 'messages', [])
        # Classify (supports both objects and dicts)
        classified = self.classifier.classify_messages(window_msgs)
        # Detect patterns
        patterns = {}
        try:
            patterns = self.pattern_detector.detect_patterns(conv, classified)
        except Exception as e:
            print("[PipelineRobust] pattern_detector error:", e); patterns = {}
        # Score risk
        risk = self.risk_scorer.compute_risk(classified, patterns)
        # Plan actions
        actions = self.planner.plan(conv, classified, risk, patterns)
        # Ethics filter
        filtered_actions = []
        for a in actions:
            try:
                check = self.ethics.check(a)
            except Exception:
                check = {'allowed': True}
            try:
                self.observability.log({'event':'ethics_check','action': a.get('action_type'), 'result': check})
            except Exception:
                pass
            if check.get('allowed'):
                filtered_actions.append(a)
        # Evidence text if requested
        evidence_text = None
        if any(a.get('action_type')=='generate_evidence_text' for a in filtered_actions):
            try:
                evidence_text = self.evidence_builder.build_evidence_text(conv, classified, risk)
            except Exception as e:
                print("[PipelineRobust] evidence builder error:", e)
                evidence_text = None
        # Build incident object (SimpleNamespace) to keep consistent shape
        incident = types.SimpleNamespace(
            incident_id = gen_id('incident'),
            convo_id = getattr(conv,'convo_id', gen_id('convo')),
            involved_user_ids = list({getattr(m,'sender_id', None) for m in getattr(conv,'messages',[]) if getattr(m,'sender_id',None)}),
            start_ts = getattr(conv.messages[0],'ts', now_ts()) if getattr(conv,'messages',None) else now_ts(),
            end_ts = getattr(conv.messages[-1],'ts', now_ts()) if getattr(conv,'messages',None) else now_ts(),
            labels = [l for cm in classified for l in cm.get('labels',[])],
            severity = risk.get('severity') if isinstance(risk, dict) else 'Low',
            evidence = evidence_text,
            status = 'new'
        )
        # store minimal incident in memory_agent if possible
        try:
            for u in incident.involved_user_ids:
                if self.memory:
                    self.memory.append_incident(u, incident.incident_id)
        except Exception as e:
            print("[PipelineRobust] memory append error:", e)
        # Notify moderator (resilient)
        try:
            self.notifier.notify_moderator(incident, filtered_actions)
        except Exception as e:
            print("[PipelineRobust] notifier error:", e)
        try:
            self.observability.log({'event':'pipeline_complete', 'incident_id': incident.incident_id, 'severity': incident.severity})
        except Exception:
            pass
        # Return standardized tuple: (incident, conv, classified, patterns, risk)
        return incident, conv, classified, patterns, risk

# Instantiate robust pipeline and place into globals under the expected name
pipeline = SilentGuardianPipelineRobust()
globals()['pipeline'] = pipeline
print("\nâœ” pipeline (robust) is ready and assigned to variable 'pipeline'")
print("You can now call: incident, conv, classified, patterns, risk = pipeline.run_once(stream)")
print("==== SAFE BOOTSTRAP complete ====\n")



# Restore print_header safely
def print_header(title: str):
    print("\n" + "="*8 + f" {title} " + "="*8 + "\n")

print("âœ” print_header restored.")


# --- Repair Cell (hotfixes / monkeypatches) ---
# Purpose: patch broken or partially-applied state after manual edits.
# Examples:
#  - ensure observability.logs exists (fix AttributeError)
#  - replace pipeline.classifier/pipeline.pattern_detector when updated classes are defined
#  - make pipeline.run_once wrapper tuple-safe (handles both dataclass and tuple returns)
# Use: run when you see NameError/AttributeError after re-editing components.



# --- REPAIR CELL: Ensure all agents exist before Pipeline ---

# 1. Pattern Detector
try:
    pattern_detector
    print("[Repair] pattern_detector already exists.")
except NameError:
    print("[Repair] Creating pattern_detector...")
    class PatternDetector_Fix:
        def detect_patterns(self, conv, classified):
            return {'repeat_targeting': [], 'recidivism': {}}
    pattern_detector = PatternDetector_Fix()

# 2. Risk Scorer
try:
    risk_scorer
except NameError:
    class RiskScorer_Fix:
        def compute_risk(self, classified, patterns):
            return {'score':0.0, 'severity':'Low'}
    risk_scorer = RiskScorer_Fix()

# 3. Planner
try:
    planner
except NameError:
    class Planner_Fix:
        def plan(self, conv, classified, risk, patterns):
            return [{'action_type':'suggest_support_message'}]
    planner = Planner_Fix()

# 4. Ethics Agent
try:
    ethics
except NameError:
    class Ethics_Fix:
        def check(self, action):
            return {'allowed': True}
    ethics = Ethics_Fix()

# 5. Notifier
try:
    notifier
except NameError:
    class Notifier_Fix:
        def notify_moderator(self, incident, actions):
            print("[Notifier_Fix] moderator notified")
    notifier = Notifier_Fix()

# 6. Observability
try:
    observability
except NameError:
    class Obs_Fix:
        def __init__(self): self.logs=[]
        def log(self, entry): self.logs.append(entry)
    observability = Obs_Fix()

# 7. Memory Agent
try:
    memory_agent
except NameError:
    class Memory_Fix:
        def append_incident(self,u,i): pass
    memory_agent = Memory_Fix()

print("âœ” All required components exist now.")



# --- Safe Pipeline Builder Cell ---
# Purpose: instantiate the pipeline in a defensive manner.
# Behavior:
#  - verify required agent globals exist (ingestor, extractor, classifier, pattern_detector, risk_scorer, planner, ethics, notifier, evidence_builder, observability, memory_agent)
#  - if any missing, attach a small safe stub that logs calls (no destructive behavior)
#  - print a short summary of attached components (so you can confirm)
# Use: run this cell to (re)create a stable pipeline object after edits.



# -----------------------------
# Safe pipeline builder cell (paste this into your notebook)
# -----------------------------
from typing import List, Dict, Any

print_header("Safe Pipeline Builder (auto-fallbacks)")

# Helper: create a minimal stub implementation for an agent type if it's missing
def make_stub(name):
    class Stub:
        def __init__(self):
            print(f"[Stub:{name}] created")
        # common minimal methods used by pipeline
        def ingest_stream(self, raw_stream, platform='slack'):
            return None
        def extract_window(self, conv, center_msg_idx=0, window_seconds=300):
            return []
        def classify_messages(self, messages):
            return []
        def detect_patterns(self, conv, classified_msgs):
            return {'repeat_targeting': [], 'recidivism': {}}
        def compute_risk(self, classified_msgs, patterns):
            return {'score': 0.0, 'severity': 'Low'}
        def plan(self, conv, classified_msgs, risk, patterns):
            return []
        def build_evidence_text(self, conv, classified_msgs, risk):
            return "NO EVIDENCE (stub)"
        def check(self, action):
            return {'allowed': True}
        def notify_moderator(self, incident, actions):
            print(f"[Stub-notifier] Incident {getattr(incident,'incident_id', '<no-id>')} actions: {actions}")
        def log(self, entry):
            print(f"[Stub-observability] {entry}")
        def append_incident(self, user_id, incident_id):
            pass
        def get_incidents(self, user_id):
            return []
    Stub.__name__ = f"Stub_{name}"
    return Stub()

# List of expected component names and the attribute to attach in pipeline
expected_components = {
    'ingestor': ('ingestor', 'IngestorAgent'),
    'extractor': ('extractor', 'ExtractorAgent'),
    'classifier': ('classifier', 'HarassmentClassifierAgent'),
    'pattern_detector': ('pattern_detector', 'PatternDetectorAgent'),
    'risk_scorer': ('risk_scorer', 'RiskScorerAgent'),
    'planner': ('planner', 'InterventionPlannerAgent'),
    'evidence_builder': ('evidence_builder', 'EvidenceBuilderAgent'),
    'ethics': ('ethics', 'EthicsAgent'),
    'notifier': ('notifier', 'NotifierAgent'),
    'observability': ('observability', 'ObservabilityAgent'),
    'memory': ('memory_agent', 'MemoryAgent'),
}

# Prepare attachments: use existing globals() objects if present; otherwise create stubs
attached = {}
for attr, (global_name, cls_name) in expected_components.items():
    if global_name in globals() and globals()[global_name] is not None:
        attached[attr] = globals()[global_name]
        print(f"Using existing {global_name} -> {type(attached[attr]).__name__}")
    else:
        # create a stub and attach
        stub = make_stub(attr)
        attached[attr] = stub
        globals()[global_name] = stub  # create a global reference so later cells find it
        print(f"Created stub for missing {global_name} -> {type(stub).__name__}")

# Minimal Incident dataclass fallback (if missing)
try:
    Incident  # noqa: F821
except NameError:
    from dataclasses import dataclass, field
    @dataclass
    class Incident:
        incident_id: str
        convo_id: str
        involved_user_ids: List[str]
        start_ts: str
        end_ts: str
        labels: List[Dict[str,Any]]
        severity: str
        evidence: Any = None
        status: str = 'new'
    globals()['Incident'] = Incident
    print("Defined fallback Incident dataclass")

# Now define a safe pipeline class that uses the attached components
class SilentGuardianPipelineSafe:
    def __init__(self, components: Dict[str, Any]):
        print("[PipelineSafe] init")
        # attach everything (either real or stub)
        self.ingestor = components['ingestor']
        self.extractor = components['extractor']
        self.classifier = components['classifier']
        self.pattern_detector = components['pattern_detector']
        self.risk_scorer = components['risk_scorer']
        self.planner = components['planner']
        self.evidence_builder = components['evidence_builder']
        self.ethics = components['ethics']
        self.notifier = components['notifier']
        self.observability = components['observability']
        self.memory = components['memory']

    def run_once(self, raw_stream: List[Dict[str,Any]]):
        print("[PipelineSafe] run_once called")
        # Ingest
        conv = None
        try:
            conv = self.ingestor.ingest_stream(raw_stream)
        except Exception as e:
            print("[PipelineSafe] ingest_stream error:", e)
            # create a minimal conversation object if necessary
            conv = type("Conv", (), {"convo_id": gen_id("convo"), "platform": "unknown", "messages": []})()

        # observability
        try:
            self.observability.log({'event':'ingested_conversation','convo_id': getattr(conv,'convo_id', None), 'n_messages': len(getattr(conv,'messages',[]))})
        except Exception as e:
            print("[PipelineSafe] observability.log error:", e)

        # Extract window
        try:
            window_msgs = self.extractor.extract_window(conv, center_msg_idx=max(0, len(getattr(conv,'messages',[]))-1))
        except Exception as e:
            print("[PipelineSafe] extract_window error:", e)
            window_msgs = getattr(conv,'messages',[])

        # Classify
        try:
            classified = self.classifier.classify_messages(window_msgs)
        except Exception as e:
            print("[PipelineSafe] classify_messages error:", e)
            classified = []

        # Pattern detection
        try:
            patterns = self.pattern_detector.detect_patterns(conv, classified)
        except Exception as e:
            print("[PipelineSafe] detect_patterns error:", e)
            patterns = {'repeat_targeting': [], 'recidivism': {}}

        # Risk scoring
        try:
            risk = self.risk_scorer.compute_risk(classified, patterns)
        except Exception as e:
            print("[PipelineSafe] compute_risk error:", e)
            risk = {'score':0.0, 'severity':'Low'}

        # Plan interventions
        try:
            actions = self.planner.plan(conv, classified, risk, patterns)
        except Exception as e:
            print("[PipelineSafe] planner.plan error:", e)
            actions = []

        # Ethics vetting
        filtered_actions = []
        for a in actions:
            try:
                check = self.ethics.check(a)
            except Exception as e:
                print("[PipelineSafe] ethics.check error:", e)
                check = {'allowed': True}
            try:
                self.observability.log({'event':'ethics_check','action': a.get('action_type','unknown'), 'result': check})
            except Exception:
                pass
            if check.get('allowed'):
                filtered_actions.append(a)

        # Evidence (text) generation if requested
        evidence_text = None
        if any(a.get('action_type') in ('generate_evidence_text','generate_evidence_packet') for a in filtered_actions):
            try:
                evidence_text = self.evidence_builder.build_evidence_text(conv, classified, risk)
            except Exception as e:
                print("[PipelineSafe] evidence_builder error:", e)
                evidence_text = "EVIDENCE ERROR"

        # Build Incident (robust)
        incident_id = gen_id('incident')
        involved = list({m.sender_id for m in getattr(conv,'messages',[])}) if getattr(conv,'messages',None) else []
        incident = Incident(
            incident_id=incident_id,
            convo_id=getattr(conv,'convo_id', gen_id('convo')),
            involved_user_ids=involved,
            start_ts=getattr(conv,'messages',[{'ts': now_ts()}])[0]['ts'] if getattr(conv,'messages',None) else now_ts(),
            end_ts=getattr(conv,'messages',[-1])['ts'] if getattr(conv,'messages',None) else now_ts(),
            labels=[l for cm in classified for l in cm.get('labels',[])],
            severity=risk.get('severity','Low'),
            evidence=evidence_text,
            status='new'
        )

        # store incident if DB cursor exists (safe)
        try:
            if 'cur' in globals() and 'conn' in globals():
                cur.execute('INSERT INTO incidents (incident_id, convo_id, involved_users, start_ts, end_ts, labels, severity, evidence, status) VALUES (?,?,?,?,?,?,?,?,?)',
                            (incident.incident_id, incident.convo_id, json.dumps(incident.involved_user_ids), incident.start_ts, incident.end_ts, json.dumps(incident.labels), incident.severity, json.dumps(incident.evidence) if incident.evidence else None, incident.status))
                conn.commit()
        except Exception as e:
            print("[PipelineSafe] DB insert error:", e)

        # Update memory for involved users
        try:
            for u in incident.involved_user_ids:
                try:
                    self.memory.append_incident(u, incident.incident_id)
                except Exception:
                    pass
        except Exception:
            pass

        # Notify moderator (prints with fallback)
        try:
            self.notifier.notify_moderator(incident, filtered_actions)
        except Exception as e:
            print("[PipelineSafe] notifier.notify_moderator error:", e)

        try:
            self.observability.log({'event':'pipeline_complete','incident_id': incident.incident_id, 'severity': incident.severity})
        except Exception:
            pass

        # Return the robust tuple
        return incident, conv, classified, patterns, risk

# instantiate pipeline with attached components
pipeline = SilentGuardianPipelineSafe(attached)
print("\nPipeline created. Attached components summary:")
for k, v in attached.items():
    print(f" - {k}: {type(v).__name__}")

print("\nâœ… Safe pipeline ready. Run pipeline.run_once(stream) to test.")



# --- Pipeline Cell ---
# Purpose: define SilentGuardianPipeline and its run_once() method (main processing flow).
# Behavior:
#  - ingest -> extract -> classify -> pattern detect -> score -> plan -> ethics -> evidence -> incident save -> notify
#  - uses the objects attached in the Safe Pipeline Builder cell
# Safety:
#  - should not directly perform destructive actions (ban/fire); those are flagged for human approval by EthicsAgent
#  - returns a consistent Incident dataclass (or guarded tuple) so downstream code can handle both shapes



# -----------------------------
# Pipeline
# -----------------------------
from typing import List, Dict, Any  
print_header("Pipeline")
class SilentGuardianPipeline:
    def __init__(self):
        print("[Pipeline] init")
        self.ingestor = ingestor
        self.extractor = extractor
        self.classifier = classifier
        self.pattern_detector = pattern_detector
        self.risk_scorer = risk_scorer
        self.planner = planner
        self.evidence_builder = evidence_builder
        self.ethics = ethics
        self.notifier = notifier
        self.observability = observability
        self.memory = memory_agent

    def run_once(self, raw_stream: List[Dict[str,Any]]):
        print("[Pipeline] run_once called")
        conv = self.ingestor.ingest_stream(raw_stream)
        self.observability.log({'event':'ingested_conversation','convo_id': conv.convo_id, 'n_messages': len(conv.messages)})
        window_msgs = self.extractor.extract_window(conv, center_msg_idx=max(0, len(conv.messages)-1))
        classified = self.classifier.classify_messages(window_msgs)
        # publish: here we just call pattern detector
        patterns = self.pattern_detector.detect_patterns(conv, classified)
        risk = self.risk_scorer.compute_risk(classified, patterns)
        actions = self.planner.plan(conv, classified, risk, patterns)
        filtered_actions = []
        for a in actions:
            check = self.ethics.check(a)
            self.observability.log({'event':'ethics_check','action': a['action_type'], 'result': check})
            if check.get('allowed'):
                filtered_actions.append(a)
            else:
                print(f"[Pipeline] Action {a['action_type']} blocked by ethics: {check.get('reason')}")
        evidence_text = None
        if any(a['action_type']=='generate_evidence_text' for a in filtered_actions):
            evidence_text = self.evidence_builder.build_evidence_text(conv, classified, risk)
        # Create incident object and store in-memory DB
        incident_id = gen_id('incident')
        incident = Incident(incident_id=incident_id, convo_id=conv.convo_id,
                            involved_user_ids=list({m.sender_id for m in conv.messages}),
                            start_ts=conv.messages[0].ts if conv.messages else now_ts(),
                            end_ts=conv.messages[-1].ts if conv.messages else now_ts(),
                            labels=[l for cm in classified for l in cm['labels']],
                            severity=risk['severity'], evidence=evidence_text, status='new')
        cur.execute('INSERT INTO incidents (incident_id, convo_id, involved_users, start_ts, end_ts, labels, severity, evidence, status) VALUES (?,?,?,?,?,?,?,?,?)',
                    (incident.incident_id, incident.convo_id, json.dumps(incident.involved_user_ids), incident.start_ts, incident.end_ts, json.dumps(incident.labels), incident.severity, json.dumps(incident.evidence) if incident.evidence else None, incident.status))
        conn.commit()
        print(f"[Pipeline] Created incident {incident.incident_id} severity={incident.severity}")
        # update memory for involved users
        for u in incident.involved_user_ids:
            self.memory.append_incident(u, incident.incident_id)
        # Notify moderator (prints)
        self.notifier.notify_moderator(incident, filtered_actions)
        self.observability.log({'event':'pipeline_complete','incident_id': incident.incident_id, 'severity': incident.severity})
        return incident, conv, classified, patterns, risk

pipeline = SilentGuardianPipeline()
print("Pipeline ready.")



print_header("Memory Agent")
class MemoryAgent:
    def __init__(self, conn):
        self.conn = conn
    def append_incident(self, user_id, incident_id):
        cur = self.conn.cursor()
        cur.execute("SELECT incidents FROM user_profiles WHERE user_id=?", (user_id,))
        r = cur.fetchone()
        if not r:
            cur.execute("INSERT INTO user_profiles (user_id, anon_id, safety_score, incidents) VALUES (?,?,?,?)",
                        (user_id, gen_id('anon'), 0.5, json.dumps([incident_id])))
        else:
            incs = json.loads(r[0] or '[]')
            incs.append(incident_id)
            cur.execute("UPDATE user_profiles SET incidents=? WHERE user_id= ?", (json.dumps(incs), user_id))
        self.conn.commit()
    def get_incidents(self, user_id):
        cur = self.conn.cursor()
        cur.execute("SELECT incidents FROM user_profiles WHERE user_id=?", (user_id,))
        r = cur.fetchone()
        return json.loads(r[0]) if r and r[0] else []

memory_agent = MemoryAgent(conn)
print("MemoryAgent ready.")



print_header("Observability")
class ObservabilityAgent:
    def __init__(self):
        self.logs = []
    def log(self, entry: Dict[str,Any]):
        entry['ts'] = now_ts()
        self.logs.append(entry)
        print(f"[Observability] {entry.get('event','unknown')}")
    def dump(self):
        # return logs for display in notebook
        return list(self.logs)

observability = ObservabilityAgent()
print("Observability and Logging done successfully.")


# -----------------------------
# Demo streams (runtime)
# Generates realistic chat conversations dynamically for live testing of the pipeline.
# Includes benign, low-risk, and repeated high-risk harassment scenarios.
# -----------------------------
print_header("Demo streams")
SIM_STREAMS = []
SIM_STREAMS.append([
    {'sender_id':'@alice','text':'Hey team, please review my PR','ts': now_ts()},
    {'sender_id':'@bob','text':'Looks good to me, thanks!','ts': now_ts()}
])
SIM_STREAMS.append([
    {'sender_id':'@alex','text':'You are so dumb','ts': now_ts()},
    {'sender_id':'@mira','text':'Please do not talk to me like that','ts': now_ts()}
])
for i in range(4):
    SIM_STREAMS.append([
        {'sender_id':'@alex','text': f"@mira you are worthless and a loser {i}", 'ts': now_ts()}
    ])
print(f"Prepared {len(SIM_STREAMS)} demo streams (runtime-generated).")


# Verifies that the pipeline and observability system are properly initialized before execution.
# Prevents runtime crashes by stopping execution if critical components are missing.

if 'pipeline' not in globals():
    print("â�Œ pipeline does NOT exist yet. Run the pipeline creation cell first.")
else:
    if not hasattr(pipeline, 'observability'):
        print("â�Œ pipeline.observability missing â€” create Observability before pipeline.")
    else:
        if not hasattr(pipeline.observability, 'logs'):
            pipeline.observability.logs = []
            print("âœ… Fixed: created pipeline.observability.logs = []")
        else:
            print(f"âœ” pipeline.observability.logs already exists (len = {len(pipeline.observability.logs)})")



# Automatically repairs missing observability log storage if it was not initialized correctly.
# Ensures uninterrupted logging without restarting the full notebook.

try:
    if hasattr(pipeline, 'observability'):
        if not hasattr(pipeline.observability, 'logs'):
            pipeline.observability.logs = []
            print("Created pipeline.observability.logs = [] (quick fix)")
        else:
            print("pipeline.observability.logs already exists (len = {})".format(len(pipeline.observability.logs)))
    else:
        print("pipeline or pipeline.observability not found in globals()")
except Exception as e:
    print("Error applying quick fix:", e)


# ============================================================================
# MAIN EXECUTION PIPELINE â€” END-TO-END DEMO RUNNER
# ============================================================================
# This is the primary execution cell of the Silent Guardian system.
# It runs the complete multi-agent pipeline on all demo chat streams,
# generates structured incident summaries, and prints human-readable outputs.
#
# HOW TO USE THIS SECTION:
# 1. Ensure all agents (Ingestor, Classifier, Pattern Detector, Risk Scorer,
#    Planner, MemoryAgent, Observability, and Pipeline) are already created.
# 2. Ensure SIM_STREAMS (demo chat conversations) are defined.
# 3. Run this cell once to execute the full detection â†’ analysis â†’ risk scoring
#    â†’ intervention â†’ memory update â†’ reporting workflow.
#
# WHAT THIS SECTION DOES:
# â€¢ Executes each chat stream through the unified multi-agent pipeline.
# â€¢ Automatically normalizes different pipeline return formats (safe handling).
# â€¢ Identifies offender and target using analytical heuristics.
# â€¢ Prints a standardized Incident Summary Dashboard for every conversation.
# â€¢ Displays full conversation, classifier outputs, detected patterns, and risk.
# â€¢ Collects all results into the `results` list for evaluation or export.
#
# This section demonstrates the complete real-time behavior of the system
# exactly as it would operate in a production moderation environment.
# ============================================================================




# ----------------------------
# Executes the full Silent Guardian multi-agent pipeline on all runtime demo streams and prints standardized incident reports.
# -----------------------------
from collections import Counter
from typing import List, Dict, Any

def summarize_offender_target(classified: List[Dict[str,Any]], patterns: Dict[str,Any]):
    """
    Pick an offender (sender with highest total label score) and a target (most frequent @-mention).
    Returns tuple (offender, target).
    Defensive: works with labelled messages where each cm has 'sender_id' and 'labels'.
    """
    # Sum label scores per sender
    scores_by_sender = {}
    for cm in classified:
        s = cm.get('sender_id', '(unknown)')
        total = sum(l.get('score', 0.0) for l in cm.get('labels', []))
        scores_by_sender[s] = scores_by_sender.get(s, 0.0) + total
    if scores_by_sender:
        sorted_items = sorted(scores_by_sender.items(), key=lambda kv: (-kv[1], kv[0]))
        offender = sorted_items[0][0]
    else:
        offender = '(none)'

    # Collect targets from pattern detector and direct message mentions
    targets = [t.get('target') for t in patterns.get('repeat_targeting', []) if t.get('target')]
    # Also scan classified messages for mentions if patterns empty
    if not targets:
        for cm in classified:
            text = cm.get('text') or ''
            # safe string check
            if isinstance(text, str):
                targets.extend(re.findall(r"@\w+", text))
    if targets:
        counts = Counter(targets)
        # pick most frequent, break ties lexicographically
        sorted_targets = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        target = sorted_targets[0][0]
    else:
        target = '(none detected)'

    return offender, target


def print_incident_dashboard(incident_obj, conv, classified: List[Dict[str,Any]],
                             patterns: Dict[str,Any], risk: Dict[str,Any],
                             planner_obj=None, ethics_obj=None):
    """
    Robust incident dashboard printer.
    Supports:
     - incident_obj as dataclass-like with .incident_id/.severity
     - incident_obj as tuple (returns from older pipeline) -- tries reasonable positions
    """
    planner_obj = planner_obj or globals().get('planner')
    ethics_obj = ethics_obj or globals().get('ethics')

    offender, target = summarize_offender_target(classified, patterns)

    # Resolve incident id + severity safely
    if hasattr(incident_obj, 'incident_id'):
        incident_id = getattr(incident_obj, 'incident_id', '(unknown)')
    elif isinstance(incident_obj, tuple):
        # try common tuple shapes: (incident, conv, classified, patterns, risk)
        # if incident_obj actually is the "incident" element in older wrapper, handle both.
        try:
            # if the user passed a 5-tuple from old pipeline, second element may be conv etc.
            # Search tuple for an object with attribute 'incident_id'
            incident_id = next((x.incident_id for x in incident_obj if hasattr(x, 'incident_id')), None)
            if not incident_id:
                incident_id = incident_obj[0] if len(incident_obj) > 0 else '(unknown)'
        except Exception:
            incident_id = str(incident_obj)
    else:
        incident_id = str(incident_obj)

    if hasattr(incident_obj, 'severity'):
        severity = getattr(incident_obj, 'severity', '(unknown)')
    elif isinstance(incident_obj, tuple):
        # try to find risk-like element in tuple (a dict with 'severity') or fallback
        sev = None
        for x in incident_obj:
            if isinstance(x, dict) and 'severity' in x:
                sev = x.get('severity')
                break
        severity = sev or (incident_obj[0].severity if hasattr(incident_obj[0], 'severity') else '(unknown)')
    else:
        severity = risk.get('severity') if isinstance(risk, dict) else '(unknown)'

    # Build recommended actions using planner if available and if it accepts inputs
    actions = []
    if planner_obj:
        try:
            # planner.plan may expect conv, classified, risk, patterns
            # ensure risk is a dict with severity
            r = risk if isinstance(risk, dict) else {'severity': severity}
            actions = planner_obj.plan(conv, classified, r, patterns) or []
        except Exception as e:
            # fallback: leave actions empty but do not raise
            # print debug line to help developer
            print(f"[print_incident_dashboard] planner.plan call failed: {e}")
            actions = []

    # Print dashboard
    print("\n=== Incident Summary Dashboard ===")
    print(f"Incident ID: {incident_id}")
    print(f"Severity: {severity}")
    print(f"Offender: {offender}")
    print(f"Target: {target}")
    print("Recommended Actions:")
    if actions:
        for a in actions:
            print(f" - {a.get('action_type', '(unknown)')}" + (f": {a.get('rationale')}" if a.get('rationale') else ""))
    else:
        print(" - (none)")
    print("="*40 + "\n")


# -----------------------------
# Re-run demo loop using new dashboard function
# (This replaces the old ad-hoc printing block; paste this cell and run)
# -----------------------------
print_header("Run demo")
results = []

# Defensive checks for required globals
if 'SIM_STREAMS' not in globals():
    raise RuntimeError("SIM_STREAMS not found in globals. Ensure you defined demo streams before running this cell.")
if 'pipeline' not in globals():
    raise RuntimeError("pipeline not found in globals. Define pipeline (SilentGuardianPipeline) before running this cell.")

for i, s in enumerate(SIM_STREAMS):
    print(f"\n--- Running stream #{i} ---")
    # pipeline.run_once may return either: Incident (dataclass) or tuple (incident, conv, classified, patterns, risk)
    out = pipeline.run_once(s)
    # Normalize return shapes:
    if isinstance(out, tuple) and len(out) == 5:
        incident, conv, classified, patterns, risk = out
    elif hasattr(out, 'incident_id') or isinstance(out, dict):
        # older pipeline variant returned incident only; try to reconstruct other outputs where possible
        incident = out
        # attempt to reconstruct conv/classified/patterns/risk via a second classification pass (safe)
        try:
            # we have the raw stream s -> ingest to get conv
            conv = ingestor.ingest_stream(s)
            window_msgs = extractor.extract_window(conv, center_msg_idx=max(0, len(conv.messages)-1))
            classified = classifier.classify_messages(window_msgs)
            patterns = pattern_detector.detect_patterns(conv, classified) if 'pattern_detector' in globals() else {}
            risk = risk_scorer.compute_risk(classified, patterns) if 'risk_scorer' in globals() else {'severity': getattr(incident, 'severity', 'Unknown')}
        except Exception as e:
            # last resort: set placeholders
            conv = None
            classified = []
            patterns = {}
            risk = {'severity': getattr(incident, 'severity', 'Unknown')}
    else:
        # Unexpected shape â€” try to treat it as incident-like
        incident = out
        conv = None
        classified = []
        patterns = {}
        risk = {'severity': getattr(incident, 'severity', 'Unknown')}

    # Print standardized dashboard
    try:
        print_incident_dashboard(incident, conv, classified, patterns, risk, planner_obj=globals().get('planner'))
    except Exception as e:
        print("[demo loop] print_incident_dashboard failed:", e)

    # The rest of the structured output (messages, classifier results, etc.)
    if conv:
        print(f"Session: {getattr(conv, 'convo_id', '(unknown)')}")
        try:
            n_messages = len(conv.messages)
        except Exception:
            n_messages = 0
        print(f"User messages ({n_messages}):")
        for m in getattr(conv, 'messages', []):
            # m may be Message object or dict
            if hasattr(m, 'sender_id'):
                sid = m.sender_id
                txt = m.text
                ts = m.ts
            else:
                sid = m.get('sender_id','(unknown)')
                txt = m.get('text','')
                ts = m.get('ts', now_ts())
            print(f" - {sid} @ {ts}: {txt}")
    else:
        print("Session: (conversation object not available)")

    # Classifier results
    print('\nClassifier results:')
    for cm in classified:
        expl = cm.get('explanation') or ''
        print(f" - {cm.get('msg_id','(no-id)')} by {cm.get('sender_id','(unknown)')}: labels={[l['label'] for l in cm.get('labels',[])]} {expl}")

    # Patterns, risk, recommended actions (repeat)
    print('\nPatterns detected:')
    print(patterns)
    print('\nRisk: ', risk)
    print('\nRecommended actions:')
    try:
        rec_actions = globals().get('planner').plan(conv, classified, risk, patterns)
    except Exception as e:
        rec_actions = []
    for a in rec_actions:
        print(' -', a.get('action_type'), ':', a.get('rationale',''))
    results.append({'incident':incident, 'conv':conv, 'classified':classified, 'patterns':patterns, 'risk':risk})

print_header("Run demo complete (patched)")



# -----------------------------
# Inspect MemoryAgent persistent user incident history
# -----------------------------
cur.execute("SELECT user_id, incidents FROM user_profiles WHERE incidents IS NOT NULL")
print(cur.fetchall()[:10])



# ---------------------------------------------
# Observability & Execution Trace Viewer
# ---------------------------------------------
# Displays the last 10 internal pipeline events captured during the demo run.
# This provides full transparency into system behavior for debugging, auditing,
# and validating multi-agent decision flow.
     
print_header('Observability logs (last 10)')
logs = observability.dump()
for e in logs[-10:]:
    print(e)


# ---------------------------------------------
# Evaluation Cell 1: Baseline Functional Check
# ---------------------------------------------
# Runs a labeled test set through the pipeline and prints
# Gold (expected) vs Pred (predicted) severity levels.
# This verifies that the end-to-end detection and scoring logic is working.

print_header('Evaluation')
EVAL_SET = [
    ([{'sender_id':'@u1','text':'Hi all, hello','ts': now_ts()}], 'Low'),
    ([{'sender_id':'@u2','text':'You are worthless','ts': now_ts()}], 'High'),
    ([{'sender_id':'@u3','text':'I will kill you','ts': now_ts()}], 'Immediate'),
]
print('Running eval set (prints Gold vs Pred)')
for stream, gold in EVAL_SET:
    inc, conv, classified, patterns, risk = pipeline.run_once(stream)
    print(f"Gold={gold} Pred={risk['severity']}")


# ---------------------------------------------
# Evaluation Cell 2: Pipeline Output Normalizer
# ---------------------------------------------
# Wraps pipeline.run_once with a robust normalizer to guarantee
# a consistent return format: (incident, conv, classified, patterns, risk).
# Also performs safe MemoryAgent updates without risking runtime crashes.
# This ensures all evaluation logic works reliably across pipeline versions.

from types import MethodType
print("Applying robust pipeline.run_once normalizer...")

# get the current 'raw' run function if saved previously (avoid losing the original)
original = getattr(pipeline, '__orig_run__', None)
if original is None:
    # if not saved, assume current pipeline.run_once is the original
    original = pipeline.run_once

def _normalized_run(raw_stream):
    # call original
    res = original(raw_stream)
    # normalize to tuple (incident, conv, classified, patterns, risk)
    if isinstance(res, tuple):
        # assume the original already returns desired tuple
        normalized = res
    else:
        # single incident returned -> try to reconstruct minimal tuple
        incident = res
        conv = None
        classified = None
        patterns = None
        risk = None
        # try to extract some fields if incident-like
        if hasattr(incident, 'convo_id'):
            conv = Conversation(convo_id=getattr(incident, 'convo_id'), platform='unknown', messages=[])
        if hasattr(incident, 'severity'):
            risk = {'severity': getattr(incident, 'severity')}
        normalized = (incident, conv, classified, patterns, risk)

    # Update memory safely (no attribute errors)
    try:
        incident_obj = normalized[0]
        involved = getattr(incident_obj, 'involved_user_ids', None) or getattr(incident_obj, 'involved_users', None)
        if involved and 'memory_agent' in globals():
            for u in involved:
                try:
                    memory_agent.append_incident(u, getattr(incident_obj, 'incident_id', None))
                except Exception as e:
                    print("[pipeline memory update] append_incident error:", e)
            # informational
            print("[pipeline memory update] updated memory_agent for involved users.")
    except Exception as e:
        print("[pipeline memory update] unexpected error while updating memory:", e)

    return normalized

# attach wrapper and remember original
pipeline.__orig_run__ = original
pipeline.run_once = MethodType(lambda self, raw_stream: _normalized_run(raw_stream), pipeline)
print("pipeline.run_once normalized: callers will receive a tuple (incident, conv, classified, patterns, risk).")



# ---------------------------------------------
# Evaluation Cell 3: Final Stabilized Validation
# ---------------------------------------------
# Executes the evaluation using the normalized pipeline output.
# Safely extracts predicted severity and compares it with gold labels.
# This cell represents the final verification step for detection accuracy
# and can be extended for formal ML metrics in future versions.

print_header('Evaluation  - fixed unpacking')
EVAL_SET = [
    ([{'sender_id':'@u1','text':'Hi all, hello','ts': now_ts()}], 'Low'),
    ([{'sender_id':'@u2','text':'You are worthless','ts': now_ts()}], 'High'),
    ([{'sender_id':'@u3','text':'I will kill you','ts': now_ts()}], 'Immediate'),
]
print('Running small eval set (prints Gold vs Pred)')

for stream, gold in EVAL_SET:
    # call pipeline.run_once - accept either tuple return or single Incident
    result = pipeline.run_once(stream)
    if isinstance(result, tuple):
        # expected tuple shape: (incident, conv, classified, patterns, risk)
        incident = result[0] if len(result) > 0 else None
        conv = result[1] if len(result) > 1 else None
        classified = result[2] if len(result) > 2 else None
        patterns = result[3] if len(result) > 3 else None
        risk = result[4] if len(result) > 4 else (None if not incident else {'severity': incident.severity if hasattr(incident, 'severity') else None})
    else:
        # single object returned (incident-like)
        incident = result
        conv = classified = patterns = None
        # try to derive risk if available on incident; else fallback
        risk = {'severity': getattr(incident, 'severity', None)} if incident is not None else {'severity': None}

    pred = risk.get('severity') if isinstance(risk, dict) else getattr(risk, 'severity', None)
    print(f"Gold={gold} Pred={pred}")




# -----------------------------
# Inspect in-memory DB (print summary)
# -----------------------------
print_header('In-memory incidents summary')
cur.execute('SELECT incident_id, convo_id, severity FROM incidents')
for r in cur.fetchall():
    print(r)

print('\nâœ… Demo complete â€” all outputs are printed above (no files).')


