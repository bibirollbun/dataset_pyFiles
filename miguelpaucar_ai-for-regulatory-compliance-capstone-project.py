# ===============================================================
# 1) Install required packages (Run once in Kaggle)
# ===============================================================
!pip install --upgrade google-generativeai
!pip install unidecode
!pip install python-docx
!pip install pypdf




# ===============================================================
# 2) Imports & Authentication
# ===============================================================
import os
import re
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import unidecode
from kaggle_secrets import UserSecretsClient

# Load API Key from Kaggle Secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Google API Key loaded successfully.")
except Exception as e:
    raise RuntimeError("â�Œ ERROR: Please add GOOGLE_API_KEY in Kaggle Secrets.") from e


# Gemini API
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Document Readers
from docx import Document
from pypdf import PdfReader


# ===============================================================
# 3) Document Loader (PDF, DOCX, TXT)
# ===============================================================
def load_document(file_path: str) -> str:
    """
    Loads text from PDF, DOCX, or TXT formats.
    """
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text

    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")



# ===============================================================
# 4) PII Redaction
# ===============================================================
PII_PATTERNS = [
    (r'\b\d{8,11}\b', '[REDACTED-NID]'),
    (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[REDACTED-EMAIL]'),
    (r'\+?\d[\d\-\s]{7,}\d', '[REDACTED-PHONE]'),
]

def redact_pii(text: str) -> str:
    cleaned = text
    for pattern, repl in PII_PATTERNS:
        cleaned = re.sub(pattern, repl, cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ===============================================================
# 5) SBS RULES 
# ===============================================================

# Based on: Resolution SBS NÂ° 00975-2025 (Regulation on Economic Group, Affiliation, Operational Limits).


SAMPLE_RULES = [
    # Article 1 - Scope
    {
        "id": "A1-01",
        "article": "Article 1 - Scope",
        "description": "Document does not state applicability to companies indicated in Article 1 (institutions under letters A,B,C of Art.16, Cofide, Banco de la NaciÃ³n, Agrobanco, Fondo Mivivienda).",
        "keywords": ["scope", "applicable", "Cofide", "Banco de la NaciÃ³n", "Agrobanco", "Fondo Mivivienda"],
        "severity": "medium",
        "points": 2
    },
    {
        "id": "A1-02",
        "article": "Article 1 - Scope",
        "description": "No mention of exceptions or special applicability clauses (e.g., Bank of the Nation, Cofide exceptions).",
        "keywords": ["not applicable", "exception", "exemption", "does not apply"],
        "severity": "low",
        "points": 1
    },

    # Article 2 - Definitions
    {
        "id": "A2-01",
        "article": "Article 2 - Definitions",
        "description": "Missing definitions for key terms: 'economic group', 'control', 'linked parties', 'conglomerate' (must include definitions used by the regulation).",
        "keywords": ["economic group", "control", "linked", "conglomerate", "definitions"],
        "severity": "high",
        "points": 5
    },
    {
        "id": "A2-02",
        "article": "Article 2 - Definitions",
        "description": "Missing definition or treatment of special terms: 'principal officers', 'holdings', 'autonomous patrimony'.",
        "keywords": ["principal officers", "holding", "autonomous patrimony", "patrimonio autÃ³nomo"],
        "severity": "medium",
        "points": 3
    },

    # Article 3 / 4 - Economic Group & Control (from Chapter II)
    {
        "id": "A3-01",
        "article": "Article 3 - Economic Group",
        "description": "No clear identification method for economic group composition (direct/indirect control, joint decision-making persons).",
        "keywords": ["group economic", "economic group", "control direct", "control indirect"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A3-02",
        "article": "Article 4 - Control",
        "description": "Lack of policy to identify direct vs indirect control and tests (50%+ voting, designation powers).",
        "keywords": ["control", "50% voting", "designate remove", "indirect control"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A3-03",
        "article": "Article 4 - Control",
        "description": "No process for applying supervisory presumptions of control if information is incomplete.",
        "keywords": ["presumption of control", "supervisory presumption", "insufficient info"],
        "severity": "medium",
        "points": 3
    },

    # Article 6-9 - Conglomerate definitions
    {
        "id": "A6-01",
        "article": "Article 6/7/8 - Conglomerates",
        "description": "Policy missing to classify groups into financial conglomerate, mixed conglomerate, or non-financial conglomerate.",
        "keywords": ["conglomerate", "financial conglomerate", "mixed", "non-financial"],
        "severity": "medium",
        "points": 3
    },

    # Article 10 - Limit to directors & employees (Art.201)
    {
        "id": "A10-01",
        "article": "Article 10 - Directors & Employees limit (Art. 201)",
        "description": "No monitoring process for loans to directors, employees and related persons (includes spouses/relatives).",
        "keywords": ["directors", "employees", "limit", "Art. 201", "spouse", "relatives"],
        "severity": "high",
        "points": 5
    },
    {
        "id": "A10-02",
        "article": "Article 10 - Directors & Employees limit (Art. 201)",
        "description": "No updated database of directors/employees (and relatives) for limit tracking.",
        "keywords": ["database", "directors database", "employee registry"],
        "severity": "medium",
        "points": 3
    },

    # Article 11-19 - Linked parties & Art.202 application
    {
        "id": "A11-01",
        "article": "Article 11 - Linked parties criteria",
        "description": "No formal criteria to classify linked parties (ownership, significant influence, integration to economic group).",
        "keywords": ["linked parties", "vinculados", "ownership", "significant influence"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A11-02",
        "article": "Article 11 - Linked parties",
        "description": "Missing presumption rules when shareholder info is not available (presume linkage unless proven otherwise).",
        "keywords": ["presume linkage", "no shareholder info", "presumption"],
        "severity": "medium",
        "points": 3
    },
    {
        "id": "A19-01",
        "article": "Article 19 - Limits computation (Art.202)",
        "description": "No methodology documented for computing Art.202 limits (inclusions/exclusions, treatment of report operations, syndicated loans proportion).",
        "keywords": ["compute limit", "Article 202", "syndicated", "mitigants", "reporting operations"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A19-02",
        "article": "Article 19 - Limits computation",
        "description": "Missing policy on non-application of counterparty substitution for Art.202 (29.8 reference).",
        "keywords": ["no substitution", "Article 202", "counterparty substitution"],
        "severity": "medium",
        "points": 3
    },

    # Article 20-23 - Application to linked parties, group of connected counterparties
    {
        "id": "A20-01",
        "article": "Article 20 - Application of Art.203/204 to linked parties",
        "description": "No statement that funding to linked parties must comply with Art.204 rules (i.e., follow Chapter V requirements).",
        "keywords": ["Article 204", "linked funding", "compliance with Chapter V"],
        "severity": "high",
        "points": 5
    },
    {
        "id": "A21-01",
        "article": "Article 21 - Group of connected counterparties (risk unit)",
        "description": "No criteria or tests for considering counterparties as connected by single risk (control, interdependence).",
        "keywords": ["connected counterparties", "risk unit", "interdependence", "control"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A22-01",
        "article": "Article 22 - Interdependence tests",
        "description": "Missing minimum interdependence tests (50% revenue transactions, guarantee relationships, shared management or funding dependency).",
        "keywords": ["interdependence", "50% revenue", "guarantee", "shared management"],
        "severity": "high",
        "points": 5
    },

    # Article 23 - Calculation methods & exclusions
    {
        "id": "A23-01",
        "article": "Article 23 - Computation and exclusions",
        "description": "No guidance on exclusions from exposure calculation (exclusions referenced in para 23.5) or treatment of mutual funds / investment funds.",
        "keywords": ["exclusions", "23.5", "mutual funds", "investment funds", "compute exposure"],
        "severity": "medium",
        "points": 3
    },
    {
        "id": "A23-02",
        "article": "Article 23 - Syndicated loans and proportional mitigants",
        "description": "No method for prorating mitigants in syndicated loans (article 19.7/23.8 references).",
        "keywords": ["syndicated loans", "proportional mitigants", "proration"],
        "severity": "medium",
        "points": 3
    },

    # Article 24-26 - State and sovereign exposures
    {
        "id": "A24-01",
        "article": "Article 24 - State counterparties (Peru)",
        "description": "Lack of classification rules for public-sector counterparties (Treasury, BCRP, regional/local governments treated as individual counterparties or groups).",
        "keywords": ["Treasury", "BCRP", "regional government", "state counterparties"],
        "severity": "medium",
        "points": 3
    },
    {
        "id": "A25-01",
        "article": "Article 25 - Limits vs Peruvian State",
        "description": "No policy noting that sovereign Peruvian financing is not subject to limits (25.1) and other state entity treatments.",
        "keywords": ["sovereign Peru", "not subject to limits", "article 25"],
        "severity": "medium",
        "points": 2
    },
    {
        "id": "A26-01",
        "article": "Article 26 - Sovereign external exposures",
        "description": "No limits for sovereign exposures by risk classification (Risk I: 60%, Risk II/III: 25%, Risk IV: 15%) and selection criteria for external ratings.",
        "keywords": ["sovereign exposure", "risk I", "60%", "risk II", "25%", "risk IV", "15%"],
        "severity": "high",
        "points": 6
    },

    # Article 27 - Guarantee funds and credit-insurance patrimonies
    {
        "id": "A27-01",
        "article": "Article 27 - Guarantees by law and credit-insurance patrimonies",
        "description": "No limit documented for coverages granted by statutory guarantee funds or credit-insurance patrimonies (max 25% of level 1 effective capital).",
        "keywords": ["guarantee fund", "credit insurance", "25%", "Article 27"],
        "severity": "medium",
        "points": 3
    },

    # Article 28 - Large exposures calculation
    {
        "id": "A28-01",
        "article": "Article 28 - Large exposures calculation",
        "description": "Missing policy to calculate large exposures using articles 21-24 criteria (connected counterparties, state rules).",
        "keywords": ["large exposure", "calculation", "Article 28", "21-24"],
        "severity": "high",
        "points": 5
    },

    # Article 29-30 - Counterparty substitution and FX / market movements
    {
        "id": "A29-01",
        "article": "Article 29 - Counterparty substitution",
        "description": "Documents allow substitution of counterparty for certain guarantees only if documentation supports automatic payment and eligibility (29.1-29.4).",
        "keywords": ["substitution", "counterparty substitution", "guarantees", "automatic payment", "29.1", "29.4"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A30-01",
        "article": "Article 30 - FX computation & changes in client condition",
        "description": "No rule specifying use of daily accounting exchange rate published by SBS for foreign currency computations or treatment for changes in client condition/patrimony decreases.",
        "keywords": ["exchange rate", "daily accounting rate", "SBS published", "foreign currency", "30.1"],
        "severity": "medium",
        "points": 3
    },

    # Article 31-34 - Reporting & data updates (reports 20, 20-A, 21)
    {
        "id": "A31-01",
        "article": "Article 31/32/33 - Reporting on groups & clients (Reports 20/20-A)",
        "description": "Missing semiannual Report 20 and 20-A submission rules (timing, content: top 20 exposures, those >=10% of Tier 1 capital before substitution).",
        "keywords": ["Report 20", "Report 20-A", "semiannual", "top 20 exposures", "10% Tier 1"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A34-01",
        "article": "Article 34 - Linked party financing report (Report 21)",
        "description": "Missing quarterly Report 21 submission for financings to linked parties and linked entities (15 calendar days after quarter close).",
        "keywords": ["Report 21", "quarterly", "linked financings", "15 days"],
        "severity": "high",
        "points": 6
    },
    {
        "id": "A35-01",
        "article": "Article 35 - Sworn statement responsibility",
        "description": "No declaration that chapter submission information is sworn and the board/management are responsible for truthfulness and timeliness (Article 35).",
        "keywords": ["sworn statement", "responsibility", "board", "management", "Article 35"],
        "severity": "medium",
        "points": 3
    },

    # Article 36-38 - Supervision, internal audit & external audit
    {
        "id": "A36-01",
        "article": "Article 36 - Supervision & investigations",
        "description": "No mention that SBS can investigate groups, initiate officio investigations, and request information from any related parties.",
        "keywords": ["investigation", "SBS authority", "request information", "officio"],
        "severity": "high",
        "points": 5
    },
    {
        "id": "A37-01",
        "article": "Article 37 - Internal audit responsibilities",
        "description": "Internal Audit unit not required to include compliance with this Regulation in its annual plan per Annex of the Internal Audit Regulation.",
        "keywords": ["internal audit", "annual plan", "Article 37", "Annex"],
        "severity": "medium",
        "points": 3
    },
    {
        "id": "A38-01",
        "article": "Article 38 - External audit scope",
        "description": "External audit firms must perform reviews in accordance with Annex I of the External Audit Regulation; document lacks confirmation of this scope.",
        "keywords": ["external audit", "Annex I", "Article 38", "audit scope"],
        "severity": "medium",
        "points": 3
    },

    # Transitional & final provisions (effective date, schedule)
    {
        "id": "AF-01",
        "article": "Article Final - Transitional timetable",
        "description": "No evidence of a timetable for adapting limits (staged reductions from 2025 to 2030); verify compliance calendar is in place.",
        "keywords": ["timetable", "transition", "2025", "2026", "2030", "schedule"],
        "severity": "medium",
        "points": 2
    },
    {
        "id": "AF-02",
        "article": "Article Final - Infringements",
        "description": "Missing procedures for notifying SBS and implementing remediation plans when exceeding Article 202/204 limits (quarterly reporting of remediation plan).",
        "keywords": ["infringement", "remediation plan", "quarterly report", "Article 6 (infringements)"],
        "severity": "high",
        "points": 5
    }
]

# --- End of rule list ---



# ===============================================================
# 6) Rule Detection Engine
# ===============================================================
@dataclass
class Violation:
    rule_id: str
    article: str
    excerpt: str
    severity: str
    note: str = ""


def find_violations(text: str, rules: List[Dict[str, Any]]) -> List[Violation]:
    normalized = unidecode.unidecode(text.lower())
    detected = []

    for rule in rules:
        matched = False

        for kw in rule.get("keywords", []):
            if kw.lower() in normalized:
                matched = True
                break

        if matched:
            excerpt = ""
            for kw in rule["keywords"]:
                idx = normalized.find(kw.lower())
                if idx >= 0:
                    start = max(0, idx - 100)
                    end = min(len(text), idx + len(kw) + 100)
                    excerpt = text[start:end].strip()
                    break

            detected.append(Violation(
                rule_id=rule["id"],
                article=rule["article"],
                severity=rule["severity"],
                excerpt=excerpt,
                note=rule["description"]
            ))
    return detected


# ===============================================================
# 7) Risk Score Engine (0â€“20)
# ===============================================================
def compute_risk_score(violations: List[Violation], rules: List[Dict[str, Any]]) -> Tuple[float, int]:
    points_map = {r["id"]: r.get("points", 1) for r in rules}
    total_points = sum(points_map.get(v.rule_id, 1) for v in violations)
    max_points = 20.0

    raw = min(total_points, max_points)
    risk_score = int(round((raw / max_points) * 20))

    return raw, risk_score
 



# ===============================================================
# 8) Gemini LLM Wrapper
# ===============================================================
LLM_MODEL = "gemini-2.5-flash-lite"

def call_gemini_chat(system_prompt: str, user_prompt: str, max_output_tokens: int = 1024) -> str:
    model = genai.GenerativeModel(LLM_MODEL)

    # NEW Gemini Format: list of dicts with "text"
    contents = [
        {"text": system_prompt},
        {"text": user_prompt}
    ]

    response = model.generate_content(
        contents,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": 0.0
        }
    )

    return response.text



# ===============================================================
# 8) Gemini Multi-Agent LLM Wrapper (Supervisor + Writer)
# ===============================================================

LLM_MODEL = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------
# Base function: clean unified wrapper
# ---------------------------------------------------------------
def _call_gemini(messages: list, max_output_tokens: int = 1024) -> str:
    """
    Internal unified Gemini call.
    Accepts a list of {"text": "..."} objects.
    """

    model = genai.GenerativeModel(LLM_MODEL)

    response = model.generate_content(
        messages,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": max_output_tokens
        }
    )

    return response.text


# ---------------------------------------------------------------
# Agent 1: Supervisor Agent
# ---------------------------------------------------------------
def supervisor_agent(system_prompt: str, user_prompt: str) -> str:
    """
    Agent specialized in regulatory analysis, validation,
    and structuring of findings.
    """

    messages = [
        {"text": f"[SUPERVISOR ROLE]\n{system_prompt}"},
        {"text": user_prompt}
    ]

    return _call_gemini(messages, max_output_tokens=2048)


# ---------------------------------------------------------------
# Agent 2: Writer Agent
# ---------------------------------------------------------------
def writer_agent(system_prompt: str, supervisor_output: str) -> str:
    """
    Agent specialized in redaction, narrative structure, and clarity.
    Receives the supervisor output and produces a refined final document.
    """

    messages = [
        {"text": f"[WRITER ROLE]\n{system_prompt}"},
        {"text": supervisor_output}
    ]

    return _call_gemini(messages, max_output_tokens=2048)


# ---------------------------------------------------------------
# High-level wrapper for document audit
# ---------------------------------------------------------------
def call_gemini_chat(system_prompt: str, user_prompt: str) -> str:
    """
    High-level call combining Supervisor + Writer agents.
    system_prompt â†’ instructions for both agents
    user_prompt â†’ violations + document content
    """

    # 1) Supervisor validates, organizes, checks structure
    supervisor_output = supervisor_agent(system_prompt, user_prompt)

    # 2) Writer improves clarity, drafting, and final presentation
    final_output = writer_agent(
        "Rewrite and improve this report with formal structure, "
        "professional tone, clarity, coherence and completeness.",
        supervisor_output
    )

    return final_output



# ===============================================================
# 9) Main AUDIT FUNCTION
# ===============================================================
@dataclass
class AuditResult:
    violations: List[Violation] = field(default_factory=list)
    raw_score: float = 0.0
    risk_score: int = 0
    secure_text: str = ""
    report: str = ""


def audit_document(file_path: str, rules=SAMPLE_RULES) -> AuditResult:
    result = AuditResult()

    # Load document
    raw_text = load_document(file_path)

    # Redact PII
    secure_text = redact_pii(raw_text)
    result.secure_text = secure_text

    # Detect violations
    violations = find_violations(secure_text, rules)
    result.violations = violations

    # Risk score
    raw, score = compute_risk_score(violations, rules)
    result.raw_score = raw
    result.risk_score = score

    # Build violation summary
    summary = "\n".join([
        f"- {v.rule_id} ({v.article}) | {v.severity} | Excerpt: {v.excerpt}"
        for v in violations
    ]) or "No violations detected."

    # LLM prompt
    system_prompt = (
        "You are a compliance auditor assistant specialized in Peruvian banking regulation. "
        "Given detected violations, provide concise observations (what's wrong), recommended remediation steps, and a professional structured audit report. "
        "Use active voice and keep recommendations actionable. If no violations, produce a short compliance confirmation."
    )

    user_prompt = f"""
Document (PII removed):
{secure_text[:5000]}

Detected Violations:
{summary}

Risk Score: {score}/20

Please:
1. Summarize weaknesses.
2. Provide recommendations.
3. Generate a structured audit report.
    """

    result.report = call_gemini_chat(system_prompt, user_prompt)

    return result


# ===============================================================
# 10) RUN AUDIT
# ===============================================================
FILE = "/kaggle/input/data-bbva/gestion-riesgo BBVA.pdf"  # Change to your file

audit = audit_document(FILE)

print("=== RISK SCORE ===")
print(audit.risk_score)

print("\n=== VIOLATIONS ===")
for v in audit.violations:
    print(v.rule_id, v.article, v.severity)

print("\n=== AUDIT REPORT ===")
print(audit.report[:5000])



# ===============================================================
# 11) SAVE OUTPUT AS DOWNLOADABLE TEXT FILE
# ===============================================================
output_path = "/kaggle/working/audit_report.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== RISK SCORE ===\n")
    f.write(str(audit.risk_score) + "\n\n")

    f.write("=== VIOLATIONS ===\n")
    for v in audit.violations:
        f.write(f"{v.rule_id} | {v.article} | {v.severity}\n")
    f.write("\n")

    f.write("=== FULL AUDIT REPORT ===\n")
    f.write(audit.report)

print(f"\nğŸ“„ File saved: {output_path}")
print("ğŸ‘‰ Download it from the right panel under 'Output files'.")

