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


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import os
import uuid
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext


print("âœ… ADK components imported successfully.")


"""
risk_agent.py

Enterprise ISO-27001 Risk-Assessment Agent - Prototype

Requirements:
    pip install pydantic

Run:
    python risk_agent.py

This file is a compact, framework-agnostic implementation of:
 - RootCoordinator (orchestrator)
 - AssetClassificationAgent
 - TechnicalVulnerabilityAgent
 - ClauseMappingAgent
 - RiskScoringAgent
 - RecommendationAgent
 - Tools (DB, MFA check, cloud config parser)
 - Example demo run
"""

from __future__ import annotations
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import logging
import uuid
import math

# ---------------------------
# Logging / Observability
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("risk-agent")

DB_NAME = "risk_assessment.db"

# ---------------------------
# Pydantic Schemas
# ---------------------------

class AssetBase(BaseModel):
    asset_id: str
    name: str
    asset_type: str  # "people" | "hardware" | "cloud" | "saas" | "digital" | "service"
    owner: Optional[str]
    description: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"mfa": True, "roles": ["dev"], "contains_pii": True}

class GapAssessment(BaseModel):
    project_id: str
    controls_implemented: List[str] = []
    controls_missing: List[str] = []
    notes: Optional[str] = None

class IdentifiedRisk(BaseModel):
    risk_id: str
    asset_id: str
    title: str
    description: str
    p: int  # privacy (0-5)
    c: int  # confidentiality (0-5)
    i: int  # integrity (0-5)
    a: int  # availability (0-5)
    likelihood: float  # 0-1
    impact: float  # 0-1
    score: float  # combined numeric
    iso_controls: List[str] = []
    category: str  # "technical" or "non-technical"
    recommendations: List[str] = []

class RiskReport(BaseModel):
    report_id: str
    project_id: str
    generated_at: datetime
    summary: str
    risks: List[IdentifiedRisk] = []

# ---------------------------
# Simple DB helpers
# ---------------------------

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS assets (
        asset_id TEXT PRIMARY KEY,
        name TEXT,
        asset_type TEXT,
        owner TEXT,
        description TEXT,
        metadata TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS gap_assessments (
        project_id TEXT PRIMARY KEY,
        controls_implemented TEXT,
        controls_missing TEXT,
        notes TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS risks (
        risk_id TEXT PRIMARY KEY,
        project_id TEXT,
        asset_id TEXT,
        title TEXT,
        payload TEXT,
        FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
    )""")
    conn.commit()
    conn.close()

def save_asset(asset: AssetBase):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO assets (asset_id, name, asset_type, owner, description, metadata)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (asset.asset_id, asset.name, asset.asset_type, asset.owner, asset.description, json.dumps(asset.metadata)))
    conn.commit()
    conn.close()

def save_gap_assessment(gap: GapAssessment):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO gap_assessments (project_id, controls_implemented, controls_missing, notes)
    VALUES (?, ?, ?, ?)
    """, (gap.project_id, json.dumps(gap.controls_implemented), json.dumps(gap.controls_missing), gap.notes))
    conn.commit()
    conn.close()

def save_risk(project_id: str, risk: IdentifiedRisk):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO risks (risk_id, project_id, asset_id, title, payload)
    VALUES (?, ?, ?, ?, ?)
    """, (risk.risk_id, project_id, risk.asset_id, risk.title, risk.json()))
    conn.commit()
    conn.close()

# ---------------------------
# ISO 27001 Clause Mapping (simple lookup)
# ---------------------------
ISO_ANNEX_A_MAP = {
    # simplified mapping: tag -> description
    "A.5.1": "Information security policies",
    "A.6.1": "Organization of information security",
    "A.7.1": "Human resource security",
    "A.8.1": "Asset management",
    "A.9.1": "Access control",
    "A.10.1": "Cryptography",
    "A.11.1": "Physical security",
    "A.12.1": "Operational procedures and responsibilities",
    "A.13.1": "Communications security",
    "A.14.1": "System acquisition, development and maintenance",
    "A.15.1": "Supplier relationships",
    "A.16.1": "Information security incident management",
    "A.17.1": "Information security aspects of business continuity management",
    "A.18.1": "Compliance"
}

def map_to_iso_controls(tags: List[str]) -> List[str]:
    # naive matching: if a tag contains keyword map to control
    mapped = set()
    for t in tags:
        t_low = t.lower()
        if "access" in t_low or "mfa" in t_low or "privilege" in t_low:
            mapped.add("A.9.1")
        if "asset" in t_low or "inventory" in t_low:
            mapped.add("A.8.1")
        if "cloud" in t_low or "aws" in t_low or "azure" in t_low:
            mapped.add("A.14.1")
        if "incident" in t_low or "breach" in t_low:
            mapped.add("A.16.1")
        if "backup" in t_low or "availability" in t_low:
            mapped.add("A.17.1")
        if "privacy" in t_low or "pii" in t_low:
            mapped.add("A.18.1")
    # fallback if nothing matched
    if not mapped:
        mapped.add("A.12.1")
    return sorted(mapped)

# ---------------------------
# Tools (deterministic helpers)
# ---------------------------

def check_mfa(metadata: Dict[str, Any]) -> bool:
    # metadata expected to contain 'auth' or 'mfa' flags
    if metadata.get("mfa") is True: 
        return True
    if metadata.get("auth_methods"):
        return "mfa" in [m.lower() for m in metadata.get("auth_methods", [])]
    return False

def estimate_pcia_from_metadata(asset: AssetBase) -> Dict[str,int]:
    # Heuristic: base PCIA scores 0-5
    meta = asset.metadata
    p = 5 if meta.get("contains_pii") else (2 if meta.get("contains_personal_data") else 0)
    c = 5 if meta.get("is_confidential") else (3 if meta.get("sensitive") else 1)
    i = 5 if meta.get("critical_for_integrity") else (2 if asset.asset_type in ("hardware","digital") else 1)
    a = 5 if meta.get("requires_high_availability") else (2 if asset.asset_type in ("cloud","saas") else 1)
    # clamp
    return {"p": int(min(5, max(0, p))), "c": int(min(5, max(0, c))), "i": int(min(5, max(0, i))), "a": int(min(5, max(0, a)))}

def likelihood_from_controls(pcia: Dict[str,int], gap: GapAssessment, asset: AssetBase) -> float:
    # Very simple model: more missing controls -> higher likelihood; assets with sensitive flags increase likelihood
    missing = len(gap.controls_missing) if gap and gap.controls_missing else 0
    sensitivity = (pcia["p"] + pcia["c"]) / 10.0  # 0-1
    base = 0.05 + 0.1 * missing
    bump = 0.2 * sensitivity
    # penalize if no MFA for accounts
    if asset.asset_type in ("people", "saas", "cloud") and not check_mfa(asset.metadata):
        bump += 0.15
    val = min(0.99, base + bump)
    return round(val, 3)

def impact_from_pcia(pcia: Dict[str,int]) -> float:
    # map PCIA (0-5 each) to 0-1 impact
    total = pcia["p"]*0.35 + pcia["c"]*0.3 + pcia["i"]*0.2 + pcia["a"]*0.15
    return round(min(1.0, total/5.0), 3)

def combine_score(likelihood: float, impact: float) -> float:
    # common risk score: likelihood * impact scaled 0-100
    return round(likelihood * impact * 100, 2)

def recommend_treatment(risk: IdentifiedRisk) -> List[str]:
    recs = []
    # generic recommendations guided by risk attributes
    iso_tags = risk.iso_controls
    if "A.9.1" in iso_tags:
        recs.append("Enforce least privilege; implement RBAC and review access logs.")
        recs.append("Require MFA for all accounts accessing sensitive resources.")
    if "A.8.1" in iso_tags:
        recs.append("Maintain canonical asset inventory and periodic reconciliation.")
    if "A.14.1" in iso_tags:
        recs.append("Enforce secure IaC templates; enable cloud-native logging and monitoring; restrict public endpoints.")
    if "A.16.1" in iso_tags:
        recs.append("Implement incident response playbook and perform tabletop exercises.")
    if not recs:
        recs.append("Investigate the finding and define a prioritized remediation plan according to impact.")
    # deduplicate and return
    final = []
    for r in recs:
        if r not in final:
            final.append(r)
    return final

# ---------------------------
# Agents (simple classes)
# ---------------------------

class BaseAgent:
    name: str
    def __init__(self, name: str):
        self.name = name

class AssetClassificationAgent(BaseAgent):
    def classify(self, raw_asset: Dict[str,Any]) -> AssetBase:
        # Normalize and create AssetBase
        aid = raw_asset.get("asset_id") or f"ASSET-{uuid.uuid4().hex[:8].upper()}"
        metadata = raw_asset.get("metadata", {})
        # basic normalization: set flags from metadata or from keys
        asset = AssetBase(
            asset_id=aid,
            name=raw_asset.get("name", "Unnamed Asset"),
            asset_type=raw_asset.get("asset_type", "digital"),
            owner=raw_asset.get("owner"),
            description=raw_asset.get("description"),
            metadata=metadata
        )
        logger.debug(f"{self.name} classified asset {asset.asset_id} as {asset.asset_type}")
        return asset

class TechnicalVulnerabilityAgent(BaseAgent):
    def analyze(self, asset: AssetBase, gap: GapAssessment) -> List[Dict[str,Any]]:
        findings = []
        # Example checks
        if asset.asset_type in ("people", "saas", "cloud"):
            if not check_mfa(asset.metadata):
                findings.append({"title": "Missing MFA", "tags": ["mfa","access","cloud"], "severity_hint": "high", "category":"technical",
                                 "desc": f"Asset '{asset.name}' lacks MFA on authentication methods."})
        if asset.asset_type == "hardware":
            if asset.metadata.get("last_patch_days", 999) > 90:
                findings.append({"title":"Unpatched system", "tags":["patching","integrity","availability"], "severity_hint":"medium",
                                 "category":"technical","desc": f"System '{asset.name}' not patched for {asset.metadata.get('last_patch_days')} days."})
        # cloud config example
        if asset.asset_type == "cloud":
            if asset.metadata.get("publicly_exposed", False):
                findings.append({"title":"Publicly exposed endpoint", "tags":["cloud","exposure","availability"], "severity_hint":"high",
                                 "category":"technical","desc": f"Cloud asset '{asset.name}' has public access allowed."})
        # privacy checks
        if asset.metadata.get("contains_pii", False):
            findings.append({"title":"Contains PII", "tags":["privacy","pii"], "severity_hint":"high","category":"non-technical",
                             "desc": f"Asset '{asset.name}' stores or processes personal data."})
        # gap assessment derived finding (if control missing)
        for missing in (gap.controls_missing or []):
            if "access control" in missing.lower():
                findings.append({"title":"Access Control Gap", "tags":["access","control"], "severity_hint":"medium", "category":"technical",
                                 "desc": f"Gap assessment lists missing control: {missing}"})
        return findings

class ClauseMappingAgent(BaseAgent):
    def map(self, finding_tags: List[str]) -> List[str]:
        return map_to_iso_controls(finding_tags)

class RiskScoringAgent(BaseAgent):
    def score(self, asset: AssetBase, finding: Dict[str,Any], gap: GapAssessment) -> IdentifiedRisk:
        pcia = estimate_pcia_from_metadata(asset)
        likelihood = likelihood_from_controls(pcia, gap, asset)
        impact = impact_from_pcia(pcia)
        score = combine_score(likelihood, impact)
        iso_controls = ClauseMappingAgent("clause-mapper").map(finding.get("tags", []))
        risk = IdentifiedRisk(
            risk_id=f"RISK-{uuid.uuid4().hex[:8].upper()}",
            asset_id=asset.asset_id,
            title=finding.get("title"),
            description=finding.get("desc", ""),
            p=pcia["p"], c=pcia["c"], i=pcia["i"], a=pcia["a"],
            likelihood=likelihood, impact=impact, score=score,
            iso_controls=iso_controls,
            category=finding.get("category","technical"),
            recommendations=[]
        )
        return risk

class RecommendationAgent(BaseAgent):
    def recommend(self, risk: IdentifiedRisk) -> List[str]:
        recs = recommend_treatment(risk)
        return recs

# ---------------------------
# Root Coordinator (orchestrator)
# ---------------------------

class RootCoordinator(BaseAgent):
    def __init__(self):
        super().__init__("root-coordinator")
        self.classifier = AssetClassificationAgent("asset-classifier")
        self.tech_agent = TechnicalVulnerabilityAgent("tech-agent")
        self.scorer = RiskScoringAgent("risk-scorer")
        self.recommender = RecommendationAgent("recommender")
        self.clause_mapper = ClauseMappingAgent("clause-mapper")

    def ingest_assets(self, raw_assets: List[Dict[str,Any]]):
        assets = []
        for raw in raw_assets:
            asset = self.classifier.classify(raw)
            save_asset(asset)
            assets.append(asset)
        return assets

    def ingest_gap(self, gap: GapAssessment):
        save_gap_assessment(gap)

    def analyze_project(self, project_id: str) -> RiskReport:
        # load assets for project (simple prototype: all assets are analyzed)
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT asset_id, name, asset_type, owner, description, metadata FROM assets")
        rows = cur.fetchall()
        conn.close()
        assets = []
        for r in rows:
            try:
                metadata = json.loads(r[5]) if r[5] else {}
            except:
                metadata = {}
            assets.append(AssetBase(asset_id=r[0], name=r[1], asset_type=r[2], owner=r[3], description=r[4], metadata=metadata))
        # load gap
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT controls_implemented, controls_missing, notes FROM gap_assessments WHERE project_id = ?", (project_id,))
        gap_row = cur.fetchone()
        conn.close()
        gap = GapAssessment(project_id=project_id, controls_implemented=[], controls_missing=[]) if not gap_row else GapAssessment(
            project_id=project_id,
            controls_implemented=json.loads(gap_row[0]) if gap_row[0] else [],
            controls_missing=json.loads(gap_row[1]) if gap_row[1] else [],
            notes=gap_row[2]
        )
        logger.info(f"Analyzing {len(assets)} assets for project {project_id} with {len(gap.controls_missing)} missing controls")
        report = RiskReport(report_id=f"REPORT-{uuid.uuid4().hex[:8].upper()}", project_id=project_id, generated_at=datetime.utcnow(), summary="", risks=[])
        # iterate assets
        for asset in assets:
            findings = self.tech_agent.analyze(asset, gap)
            for f in findings:
                risk = self.scorer.score(asset, f, gap)
                risk.recommendations = self.recommender.recommend(risk)
                # optionally enrich description with heuristics
                risk.description = f"{f.get('desc')} (severity_hint={f.get('severity_hint')})"
                report.risks.append(risk)
                save_risk(project_id, risk)
        # compose summary
        if not report.risks:
            report.summary = "No risks identified by automated checks. Manual review recommended."
        else:
            # compute top issues
            top = sorted(report.risks, key=lambda r: r.score, reverse=True)[:5]
            report.summary = f"Identified {len(report.risks)} risks. Top issue: {top[0].title} (score {top[0].score})."
        logger.info(f"Report generated with {len(report.risks)} risks")
        return report

# ---------------------------
# Small Demo Data & Runner
# ---------------------------

def demo_setup():
    # example raw asset inventory
    raw_assets = [
        {"asset_id":"ASSET-EMP-01","name":"Alice - Product Manager","asset_type":"people","owner":"alice", "metadata":{"contains_pii":True, "mfa":False, "roles":["product"], "is_confidential":True}},
        {"asset_id":"ASSET-LAP-01","name":"Dev Laptop - b.singh","asset_type":"hardware","owner":"b.singh","metadata":{"last_patch_days":120, "is_confidential":False}},
        {"asset_id":"ASSET-AWS-01","name":"AlphaTech Prod Cluster","asset_type":"cloud","owner":"infra.team","metadata":{"publicly_exposed":True, "contains_pii":False, "requires_high_availability":True, "mfa":False}},
        {"asset_id":"ASSET-SALES-FS","name":"Shared Sales Drive","asset_type":"digital","owner":"sales","metadata":{"contains_pii":True, "sensitive":True}}
    ]
    gap = GapAssessment(project_id="PRJ-ALPHA", controls_implemented=["A.5.1","A.8.1"], controls_missing=["Access control - MFA not enforced","Backup policy not formalized"], notes="Initial intake")
    return raw_assets, gap

def print_report(report: RiskReport):
    print("="*80)
    print(f"RISK REPORT: {report.report_id}  Project: {report.project_id}  Generated: {report.generated_at.isoformat()}")
    print("SUMMARY:", report.summary)
    print("-"*80)
    for r in sorted(report.risks, key=lambda x: x.score, reverse=True):
        print(f"RISK ID: {r.risk_id} | Asset: {r.asset_id} | Title: {r.title} | Score: {r.score}")
        print(f"  PCIA -> P:{r.p} C:{r.c} I:{r.i} A:{r.a}  Likelihood:{r.likelihood}  Impact:{r.impact}")
        print(f"  ISO Controls: {r.iso_controls}")
        print(f"  Recommendations:")
        for rec in r.recommendations:
            print(f"    - {rec}")
        print("-"*40)
    print("="*80)

# ---------------------------
# Main
# ---------------------------

def main():
    init_db()
    coordinator = RootCoordinator()
    raw_assets, gap = demo_setup()
    coordinator.ingest_assets(raw_assets)
    coordinator.ingest_gap(gap)
    report = coordinator.analyze_project("PRJ-ALPHA")
    print_report(report)

if __name__ == "__main__":
    main()





