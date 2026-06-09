# Cell 1: Install Dependencies
!pip install -q google-generativeai
!pip install -q pandas numpy matplotlib seaborn
!pip install -q pypdf nbformat


# Cell 2: Import Libraries
import os
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from io import BytesIO

# Google Generative AI
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# Data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# File processing
import nbformat
from pathlib import Path
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# Kaggle specific (using colab would require different imports and methods)
from kaggle_secrets import UserSecretsClient

print("âœ“ All Libraries Imported Successfully")


# Cell 3: Configure API Key
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ API Key Configured from Kaggle Secrets")
except Exception as e:
    print(f"âš  API Key Error: {str(e)}")
    print("ğŸ“Œ To fix: Go to Add-ons â†’ Secrets â†’ Add 'GOOGLE_API_KEY'")
    GOOGLE_API_KEY = None



# Cell 4: Agent Configuration
CONFIG = {
    "team": "YourTeamName",
    "model": "gemini-2.0-flash-001",
    "max_tokens": 2000,
    "temperature": 0.3,
    "version": "1.0.0",
    "max_file_size_mb": 10,
    "supported_formats": [".ipynb", ".pdf", ".csv", ".json", ".txt", ".py", ".md"]
}

print(f"\n{'='*60}")
print(f"{'PRIVACY GUARDIAN CONFIGURATION':^60}")
print(f"{'='*60}")
for k, v in CONFIG.items():
    print(f"{k:.<30} {v}")
print(f"{'='*60}\n")



# Cell 5: Automated File Reading & Parsing
class FileIngestionEngine:
    """Automated file reading and content extraction"""
    
    @staticmethod
    def read_file_auto(file_path: str) -> Dict[str, Any]:
        """
        Automatically detect file type and extract content
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict with content, metadata, and file info
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = path.suffix.lower()
        
        # Route to appropriate parser
        parsers = {
            '.ipynb': FileIngestionEngine._parse_notebook,
            '.pdf': FileIngestionEngine._parse_pdf,
            '.csv': FileIngestionEngine._parse_csv,
            '.json': FileIngestionEngine._parse_json,
            '.txt': FileIngestionEngine._parse_text,
            '.py': FileIngestionEngine._parse_text,
            '.md': FileIngestionEngine._parse_text
        }
        
        parser = parsers.get(file_ext, FileIngestionEngine._parse_text)
        return parser(path)
    
    @staticmethod
    def _parse_notebook(path: Path) -> Dict[str, Any]:
        """Parse Jupyter notebook"""
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        code_content = []
        markdown_content = []
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                code_content.append(cell.source)
            elif cell.cell_type == 'markdown':
                markdown_content.append(cell.source)
        
        return {
            "content": "\n\n".join(code_content + markdown_content),
            "code_content": "\n\n".join(code_content),
            "markdown_content": "\n\n".join(markdown_content),
            "metadata": {
                "code_cells": len(code_content),
                "markdown_cells": len(markdown_content),
                "total_cells": len(nb.cells)
            },
            "file_type": "notebook",
            "file_name": path.name,
            "file_size_kb": path.stat().st_size / 1024
        }
    
    @staticmethod
    def _parse_pdf(path: Path) -> Dict[str, Any]:
        """Parse PDF document"""
        reader = PdfReader(str(path))
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        
        content = "\n\n".join(text)
        return {
            "content": content,
            "metadata": {
                "pages": len(reader.pages),
                "word_count": len(content.split())
            },
            "file_type": "pdf",
            "file_name": path.name,
            "file_size_kb": path.stat().st_size / 1024
        }
    
    @staticmethod
    def _parse_csv(path: Path) -> Dict[str, Any]:
        """Parse CSV file"""
        df = pd.read_csv(path)
        content = df.to_string()
        
        return {
            "content": content,
            "metadata": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            },
            "file_type": "csv",
            "file_name": path.name,
            "file_size_kb": path.stat().st_size / 1024
        }
    
    @staticmethod
    def _parse_json(path: Path) -> Dict[str, Any]:
        """Parse JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = json.dumps(data, indent=2)
        return {
            "content": content,
            "metadata": {
                "keys": list(data.keys()) if isinstance(data, dict) else None,
                "is_list": isinstance(data, list),
                "size": len(data)
            },
            "file_type": "json",
            "file_name": path.name,
            "file_size_kb": path.stat().st_size / 1024
        }
    
    @staticmethod
    def _parse_text(path: Path) -> Dict[str, Any]:
        """Parse plain text files"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "metadata": {
                "lines": len(content.split('\n')),
                "word_count": len(content.split()),
                "char_count": len(content)
            },
            "file_type": path.suffix[1:] if path.suffix else "text",
            "file_name": path.name,
            "file_size_kb": path.stat().st_size / 1024
        }

print("âœ“ File Ingestion Engine Ready")



# Cell 6: Prompt Templates
PROMPT_TEMPLATES = {
    "pii_scan": """You are an expert privacy compliance officer. Analyze this {file_type} for PII.

Content (first 3000 chars):
{content}

Identify ALL instances of:
- Full names (first + last)
- Email addresses  
- Phone numbers (any format)
- Physical addresses
- SSN / Government IDs
- Credit card numbers
- IP addresses
- Any other personal identifiers

Return ONLY valid JSON:
{{
  "findings": [
    {{"type": "email", "value": "user@example.com", "location": "line 5", "confidence": 0.95, "context": "surrounding text..."}}
  ],
  "summary": "Found X PII instances: Y emails, Z phones...",
  "risk_level": "high",
  "pii_count": {{"email": 2, "phone": 1}}
}}""",

    "secret_scan": """You are a security expert specializing in credential detection. Analyze this {context} for exposed secrets.

Code/Config Content (first 3000 chars):
{content}

Identify ALL:
- API keys (OpenAI: sk-*, AWS: AKIA*, GCP: AIza*, GitHub: ghp_*, etc.)
- Access tokens & OAuth tokens
- Database URLs with credentials
- Private keys (-----BEGIN PRIVATE KEY-----)
- Passwords in variables
- JWT tokens
- Webhook URLs with secrets

For each finding assess severity (critical/high/medium/low) and potential impact.

Return ONLY valid JSON:
{{
  "secrets_found": [
    {{"type": "openai_api_key", "masked_value": "sk-****xyz", "severity": "critical", "line": 15, "context": "API_KEY = 'sk-...'"}},
  ],
  "total_critical": 2,
  "total_high": 1,
  "risk_score": 85,
  "immediate_actions": ["Rotate OpenAI key", "Use environment variables"]
}}""",

    "risk_assessment": """You are a CISO conducting risk assessment. Analyze these security findings.

Document Type: {document_type}
Sharing Context: {sharing_context}

Combined Scan Results:
{scan_results}

Generate comprehensive risk assessment:

Return ONLY valid JSON:
{{
  "overall_risk_score": 85,
  "severity_level": "critical",
  "top_risks": [
    {{"risk": "Exposed OpenAI API key in public notebook", "impact": "Unauthorized API usage, $1000s in charges", "priority": 1}},
    {{"risk": "5 email addresses visible", "impact": "GDPR violation, privacy breach", "priority": 2}}
  ],
  "remediation_plan": [
    "1. IMMEDIATE: Rotate all API keys within 1 hour",
    "2. SHORT-TERM: Move secrets to environment variables",
    "3. LONG-TERM: Implement pre-commit secret scanning"
  ],
  "compliance_impact": "Violates GDPR Article 5, SOC 2 Type II requirements",
  "estimated_exposure_time": "public for X days"
}}""",

    "remediation_plan": """You are a security remediation specialist. Create an actionable fix plan.

Security Findings:
{findings}

Urgency: {urgency_level}
Resources: {resource_constraints}

Create detailed remediation with:
1. Immediate actions (next 1 hour)
2. Short-term fixes (24 hours)
3. Long-term improvements (1 week)
4. Preventive measures
5. Time estimates and difficulty ratings

Return ONLY valid JSON:
{{
  "immediate_actions": [
    {{"action": "Rotate exposed API keys", "time": "15 min", "difficulty": "easy", "steps": ["Go to provider dashboard", "Regenerate key", "Update .env file"]}}
  ],
  "short_term": [...],
  "long_term": [...],
  "preventive_measures": ["Pre-commit hooks", "CI/CD scanning", "Secret management tool"],
  "total_estimated_time": "4 hours",
  "cost_estimate": "$0 (free tools available)"
}}""",

    "best_practices": """You are a security architect. Analyze code against industry best practices.

Industry: {industry}
Compliance Required: {compliance_requirements}

Code Sample (first 2000 chars):
{code_sample}

Evaluate:
1. Secrets management
2. Data handling practices  
3. Error handling security
4. Compliance with {compliance_requirements}

Return ONLY valid JSON with specific code improvements and examples:
{{
  "compliance_score": 45,
  "violations": [
    {{"standard": "GDPR Art. 5", "violation": "PII in logs without consent", "severity": "high", "article": "5.1.c"}}
  ],
  "best_practice_gaps": ["No environment variables", "Hardcoded credentials", "No input validation"],
  "code_improvements": [
    {{"issue": "Hardcoded API key", "current": "API_KEY='sk-123'", "improved": "API_KEY=os.getenv('OPENAI_KEY')", "benefit": "Credential isolation"}}
  ],
  "actionable_recommendations": [
    "Implement AWS Secrets Manager or HashiCorp Vault",
    "Add data classification labels to all PII fields",
    "Enable comprehensive audit logging"
  ]
}}"""
}

print("âœ“ Prompt Templates Loaded")



# Cell 7: Tool Functions with Templates
def scan_document_for_pii(file_content: str, file_type: str) -> str:
    """Scan document for PII using Gemini with structured prompt"""
    prompt = PROMPT_TEMPLATES["pii_scan"].format(
        file_type=file_type,
        content=file_content[:100000]
    )
    
    model = genai.GenerativeModel(CONFIG['model'])
    response = model.generate_content(prompt)
    return response.text


def detect_exposed_secrets(code_content: str, context: str = "general") -> str:
    """Detect exposed secrets using Gemini with structured prompt"""
    prompt = PROMPT_TEMPLATES["secret_scan"].format(
        context=context,
        content=code_content[:100000]
    )
    
    model = genai.GenerativeModel(CONFIG['model'])
    response = model.generate_content(prompt)
    return response.text


def generate_risk_assessment(scan_results: str, document_type: str, sharing_context: str) -> str:
    """Generate comprehensive risk assessment"""
    prompt = PROMPT_TEMPLATES["risk_assessment"].format(
        document_type=document_type,
        sharing_context=sharing_context,
        scan_results=scan_results[:4000]
    )
    
    model = genai.GenerativeModel(CONFIG['model'])
    response = model.generate_content(prompt)
    return response.text


def create_remediation_plan(findings: str, urgency_level: str, resource_constraints: str = "small team, limited budget") -> str:
    """Create actionable remediation plan"""
    prompt = PROMPT_TEMPLATES["remediation_plan"].format(
        findings=findings[:3000],
        urgency_level=urgency_level,
        resource_constraints=resource_constraints
    )
    
    model = genai.GenerativeModel(CONFIG['model'])
    response = model.generate_content(prompt)
    return response.text


def analyze_security_best_practices(code_sample: str, industry: str, compliance_requirements: str) -> str:
    """Analyze code against security best practices"""
    prompt = PROMPT_TEMPLATES["best_practices"].format(
        code_sample=code_sample[:2000],
        industry=industry,
        compliance_requirements=compliance_requirements
    )
    
    model = genai.GenerativeModel(CONFIG['model'])
    response = model.generate_content(prompt)
    return response.text


print("âœ“ 5 Tool Functions Defined")
print("  â€¢ scan_document_for_pii")
print("  â€¢ detect_exposed_secrets")
print("  â€¢ generate_risk_assessment")
print("  â€¢ create_remediation_plan")
print("  â€¢ analyze_security_best_practices")
print("\nâœ“ FileIngestionEngine ready for automated file reading")


print("\n" + "="*60)
print("FILE UPLOAD HELPER".center(60))
print("="*60)

def list_available_files(directory='.'):
    """List all scannable files in directory"""
    path = Path(directory)
    files = []
    
    for ext in CONFIG['supported_formats']:
        files.extend(list(path.glob(f'*{ext}')))
    
    return [str(f) for f in files]

# Show available files
available_files = list_available_files()

if available_files:
    print(f"\nğŸ“� Found {len(available_files)} scannable files:")
    for f in available_files[:10]:  # Show first 10
        print(f"   â€¢ {f}")
    if len(available_files) > 10:
        print(f"   ... and {len(available_files) - 10} more")
else:
    print("\nğŸ“� No files found in current directory.")
    print("\nğŸ’¡ To upload files in Kaggle:")
    print("   1. Click 'File' â†’ 'Upload' in the menu")
    print("   2. Or use: from google.colab import files; files.upload()")
    print("   3. Then run: agent.scan_file('your_file.ipynb')")

print("\nâœ“ File upload helper ready")


# Cell 8: Function Declarations for Gemini Tool Calling
function_declarations = [
    FunctionDeclaration(
        name="scan_document_for_pii",
        description="Scans document content for personally identifiable information (PII) including names, emails, phones, addresses, SSNs, and other personal data",
        parameters={
            "type": "object",
            "properties": {
                "file_content": {
                    "type": "string",
                    "description": "The text content of the file to scan"
                },
                "file_type": {
                    "type": "string",
                    "description": "Type of file being scanned (e.g., notebook, pdf, csv, json)"
                }
            },
            "required": ["file_content", "file_type"]
        }
    ),
    FunctionDeclaration(
        name="detect_exposed_secrets",
        description="Detects exposed API keys, tokens, credentials, and other secrets in code or configuration files",
        parameters={
            "type": "object",
            "properties": {
                "code_content": {
                    "type": "string",
                    "description": "Code or configuration content to analyze for secrets"
                },
                "context": {
                    "type": "string",
                    "description": "Context about the code (e.g., 'jupyter notebook', 'config file', 'python script')"
                }
            },
            "required": ["code_content"]
        }
    ),
    FunctionDeclaration(
        name="generate_risk_assessment",
        description="Generates comprehensive risk assessment with scoring based on security findings",
        parameters={
            "type": "object",
            "properties": {
                "scan_results": {
                    "type": "string",
                    "description": "Combined results from PII and secret scans"
                },
                "document_type": {
                    "type": "string",
                    "description": "Type of document being assessed"
                },
                "sharing_context": {
                    "type": "string",
                    "description": "Where/how the document will be shared (e.g., 'public Kaggle notebook', 'internal team', 'GitHub repo')"
                }
            },
            "required": ["scan_results", "document_type", "sharing_context"]
        }
    ),
    FunctionDeclaration(
        name="create_remediation_plan",
        description="Creates step-by-step remediation plan with timeline and priorities for fixing security issues",
        parameters={
            "type": "object",
            "properties": {
                "findings": {
                    "type": "string",
                    "description": "All security findings that need remediation"
                },
                "urgency_level": {
                    "type": "string",
                    "description": "How urgent fixes are needed (critical/high/medium/low)"
                },
                "resource_constraints": {
                    "type": "string",
                    "description": "Available resources - team size, timeline, budget constraints"
                }
            },
            "required": ["findings", "urgency_level"]
        }
    ),
    FunctionDeclaration(
        name="analyze_security_best_practices",
        description="Analyzes code against industry security best practices and compliance requirements",
        parameters={
            "type": "object",
            "properties": {
                "code_sample": {
                    "type": "string",
                    "description": "Code to analyze for security best practices"
                },
                "industry": {
                    "type": "string",
                    "description": "Industry context (e.g., fintech, healthcare, ecommerce, general)"
                },
                "compliance_requirements": {
                    "type": "string",
                    "description": "Relevant compliance standards (e.g., GDPR, HIPAA, PCI-DSS, SOC 2)"
                }
            },
            "required": ["code_sample", "industry", "compliance_requirements"]
        }
    )
]

# Create Tool object
tools = Tool(function_declarations=function_declarations)
print(f"âœ“ Function Declarations Created ({len(function_declarations)} tools)")



# Cell 9: Conversation Memory System
@dataclass
class ConversationMemory:
    """Manages conversation history and context for the agent"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    scan_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 20
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only recent messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def add_scan_result(self, file_name: str, scan_data: Dict[str, Any]):
        """Store scan results for comparison and history"""
        self.scan_history.append({
            "file_name": file_name,
            "timestamp": datetime.now().isoformat(),
            "risk_score": scan_data.get("risk_score", 0),
            "findings_count": scan_data.get("findings_count", 0),
            "severity": scan_data.get("severity", "unknown")
        })
    
    def get_context(self) -> str:
        """Get recent conversation context"""
        if not self.messages:
            return "No previous conversation."
        
        context = "Recent conversation:\n"
        for msg in self.messages[-5:]:
            content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            context += f"{msg['role']}: {content_preview}\n"
        return context
    
    def get_scan_summary(self) -> str:
        """Get summary of recent scans"""
        if not self.scan_history:
            return "No previous scans."
        
        recent_scans = self.scan_history[-5:]
        summary = f"Recent scans ({len(recent_scans)}):\n"
        for scan in recent_scans:
            summary += f"  â€¢ {scan['file_name']}: Risk {scan['risk_score']}/100 ({scan['severity']})\n"
        return summary
    
    def clear(self):
        """Clear all conversation history"""
        self.messages.clear()
        self.scan_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m['role'] == 'user'),
            "agent_messages": sum(1 for m in self.messages if m['role'] == 'agent'),
            "total_scans": len(self.scan_history),
            "avg_risk_score": sum(s['risk_score'] for s in self.scan_history) / len(self.scan_history) if self.scan_history else 0
        }

memory = ConversationMemory(max_history=20)
print(f"âœ“ Memory System Initialized (Max: {memory.max_history} messages)")


# Cell 10: Logging System for Observability
@dataclass
class AgentLogger:
    """Comprehensive logging system for agent operations"""
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def log(self, level: str, event: str, details: Dict[str, Any] = None):
        """Add a log entry"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details or {}
        })
    
    def info(self, event: str, **kwargs):
        """Log info level event"""
        self.log("INFO", event, kwargs)
    
    def error(self, event: str, **kwargs):
        """Log error level event"""
        self.log("ERROR", event, kwargs)
    
    def warning(self, event: str, **kwargs):
        """Log warning level event"""
        self.log("WARNING", event, kwargs)
    
    def security(self, event: str, **kwargs):
        """Log security-related event"""
        self.log("SECURITY", event, kwargs)
    
    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        """Get most recent log entries"""
        return self.logs[-count:]
    
    def get_logs_by_level(self, level: str) -> List[Dict]:
        """Get all logs of a specific level"""
        return [log for log in self.logs if log['level'] == level]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "total_logs": len(self.logs),
            "info_count": sum(1 for log in self.logs if log['level'] == 'INFO'),
            "error_count": sum(1 for log in self.logs if log['level'] == 'ERROR'),
            "warning_count": sum(1 for log in self.logs if log['level'] == 'WARNING'),
            "security_count": sum(1 for log in self.logs if log['level'] == 'SECURITY')
        }
    
    def export_logs(self, filename: str = "privacy_guardian_logs.json"):
        """Export logs to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"âœ“ Logs exported to {filename}")
    
    def print_summary(self):
        """Print log summary"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("LOGGING SUMMARY".center(60))
        print("="*60)
        for key, value in stats.items():
            print(f"{key:.<40} {value}")
        print("="*60 + "\n")

logger = AgentLogger()
logger.info("Logger initialized", system="Privacy Guardian")
print("âœ“ Logging System Ready")



# Cell 11: Performance Metrics Tracker
@dataclass
class PerformanceMetrics:
    """Track agent performance metrics"""
    scans_completed: int = 0
    tools_called: int = 0
    total_response_time: float = 0.0
    errors: int = 0
    pii_detected: int = 0
    secrets_detected: int = 0
    high_risk_findings: int = 0
    
    def record_scan(self, response_time: float, pii_count: int, secret_count: int, risk_level: str):
        """Record a completed scan"""
        self.scans_completed += 1
        self.total_response_time += response_time
        self.pii_detected += pii_count
        self.secrets_detected += secret_count
        if risk_level in ['critical', 'high']:
            self.high_risk_findings += 1
    
    def record_tool_call(self):
        """Record a tool function call"""
        self.tools_called += 1
    
    def record_error(self):
        """Record an error"""
        self.errors += 1
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        return self.total_response_time / self.scans_completed if self.scans_completed > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "scans_completed": self.scans_completed,
            "tools_called": self.tools_called,
            "avg_response_time": round(self.get_avg_response_time(), 2),
            "errors": self.errors,
            "total_pii_detected": self.pii_detected,
            "total_secrets_detected": self.secrets_detected,
            "high_risk_findings": self.high_risk_findings,
            "error_rate": round(self.errors / self.scans_completed * 100, 2) if self.scans_completed > 0 else 0.0
        }
    
    def print_dashboard(self):
        """Print metrics dashboard"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("PERFORMANCE METRICS DASHBOARD".center(60))
        print("="*60)
        print(f"{'Scans Completed':.<40} {summary['scans_completed']}")
        print(f"{'Tools Called':.<40} {summary['tools_called']}")
        print(f"{'Avg Response Time':.<40} {summary['avg_response_time']}s")
        print(f"{'Errors':.<40} {summary['errors']}")
        print(f"{'Error Rate':.<40} {summary['error_rate']}%")
        print("-"*60)
        print(f"{'Total PII Detected':.<40} {summary['total_pii_detected']}")
        print(f"{'Total Secrets Detected':.<40} {summary['total_secrets_detected']}")
        print(f"{'High Risk Findings':.<40} {summary['high_risk_findings']}")
        print("="*60 + "\n")

metrics = PerformanceMetrics()
print("âœ“ Performance Metrics Tracker Initialized")


print("\n" + "="*60)
print("PHASE 3 COMPLETE".center(60))
print("="*60)
print("âœ“ Function Declarations Created (5 tools)")
print("âœ“ Conversation Memory System Ready")
print("âœ“ Logging System Active")
print("âœ“ Performance Metrics Tracking Enabled")
print("="*60)


# Cell 12: Main Privacy Guardian Agent Class
class PrivacyGuardianAgent:
    """
    Main orchestrating agent with automated pipelines
    
    Features:
    - Automated file reading and parsing
    - One-command full security scan
    - Batch processing for multiple files
    - Intelligent agent routing
    - Comprehensive reporting
    """
    
    def __init__(self, config: Dict, tools: Tool, memory: ConversationMemory, 
                 logger: AgentLogger, metrics: PerformanceMetrics):
        """Initialize the Privacy Guardian Agent"""
        self.config = config
        self.tools = tools
        self.memory = memory
        self.logger = logger
        self.metrics = metrics
        self.file_engine = FileIngestionEngine()
        
        # Initialize Gemini model with tools
        self.model = genai.GenerativeModel(
            model_name=config['model'],
            tools=[tools]
        )
        
        self.logger.info("Agent initialized", 
                        model=config['model'], 
                        team=config['team'])
        
        print(f"âœ“ Privacy Guardian Agent Initialized")
        print(f"  Model: {config['model']}")
        print(f"  Team: {config['team']}")
        print(f"  Automation: Enabled")
    
    def _call_function(self, function_call) -> str:
        """Execute tool function and return result"""
        function_name = function_call.name
        function_args = dict(function_call.args)
        
        self.logger.info("Function called", 
                        function=function_name, 
                        args=str(function_args)[:100])
        self.metrics.record_tool_call()
        
        function_map = {
            "scan_document_for_pii": scan_document_for_pii,
            "detect_exposed_secrets": detect_exposed_secrets,
            "generate_risk_assessment": generate_risk_assessment,
            "create_remediation_plan": create_remediation_plan,
            "analyze_security_best_practices": analyze_security_best_practices
        }
        
        if function_name in function_map:
            try:
                result = function_map[function_name](**function_args)
                self.logger.security("Function executed successfully", 
                                   function=function_name)
                return result
            except Exception as e:
                self.logger.error("Function execution failed", 
                                function=function_name, 
                                error=str(e))
                self.metrics.record_error()
                return f"Error executing {function_name}: {str(e)}"
        else:
            self.logger.error("Unknown function called", function=function_name)
            return f"Unknown function: {function_name}"
    
    def run(self, user_query: str) -> str:
        """Main conversational interface"""
        start_time = time.time()
        
        try:
            self.logger.info("Query received", query=user_query[:200])
            self.memory.add_message("user", user_query)
            
            system_prompt = f"""You are DataSentry, an expert security agent for Team {self.config['team']}.

Mission: Protect users from data leaks, PII exposure, and security vulnerabilities.

Capabilities:
- Scan documents for PII (names, emails, phones, addresses, SSNs)
- Detect exposed API keys, tokens, and credentials
- Generate comprehensive risk assessments
- Create actionable remediation plans
- Analyze code against security best practices

Context:
{self.memory.get_context()}

Scan History:
{self.memory.get_scan_summary()}

Be thorough and actionable. When finding issues:
1. Explain what was found
2. Explain why it's risky
3. Provide immediate fix steps
4. Suggest prevention measures"""

            chat = self.model.start_chat()
            response = chat.send_message(f"{system_prompt}\n\nUser: {user_query}")
            
            # Handle function calls
            function_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
            
            if function_calls:
                function_responses = []
                for fc in function_calls:
                    result = self._call_function(fc)
                    function_responses.append(result)
                response = chat.send_message(function_responses)
            
            # Extract response
            try:
                response_text = response.text
            except:
                response_text = "Analysis complete."
            
            self.memory.add_message("agent", response_text)
            
            elapsed = time.time() - start_time
            self.logger.info("Query completed", response_time=f"{elapsed:.2f}s")
            
            return response_text
            
        except Exception as e:
            self.metrics.record_error()
            self.logger.error("Query failed", error=str(e))
            return f"Error: {str(e)}"
    
    def scan_file(self, file_path: str, sharing_context: str = "public Kaggle notebook") -> Dict[str, Any]:
        """
        ğŸš€ AUTOMATED PIPELINE: One-command full security scan of any file
        
        Args:
            file_path: Path to file
            sharing_context: Where file will be shared
            
        Returns:
            Complete scan results with all findings
        """
        start_time = time.time()
        
        try:
            # Step 1: Auto-detect and read file
            self.logger.security("Starting automated file scan", file=file_path)
            print(f"\nğŸ”� Scanning: {file_path}")
            print("="*60)
            
            file_data = self.file_engine.read_file_auto(file_path)
            print(f"âœ“ File parsed: {file_data['file_type']} ({file_data['file_size_kb']:.2f} KB)")
            
            # Step 2: Parallel PII + Secret scanning
            print("âœ“ Running PII detection...")
            pii_results = scan_document_for_pii(
                file_data['content'], 
                file_data['file_type']
            )
            
            # Focus on code content for secrets if notebook
            scan_content = file_data.get('code_content', file_data['content'])
            print("âœ“ Running secret scanner...")
            secret_results = detect_exposed_secrets(
                scan_content,
                f"{file_data['file_type']} file"
            )
            # Step 3: Risk assessment
            print("âœ“ Generating risk assessment...")
            combined = f"PII Findings:\n{pii_results}\n\nSecret Findings:\n{secret_results}"
            risk_assessment = generate_risk_assessment(
                combined,
                file_data['file_type'],
                sharing_context
            )
            
            # Step 4: Parse results
            try:
                # Try to parse JSON, handling potential markdown code blocks
                pii_clean = pii_results.strip()
                if '```json' in pii_clean:
                    pii_clean = pii_clean.split('```json')[1].split('```')[0].strip()
                elif '```' in pii_clean:
                    pii_clean = pii_clean.split('```')[1].split('```')[0].strip()
                
                pii_data = json.loads(pii_clean)
                pii_count = len(pii_data.get('findings', []))
            except (json.JSONDecodeError, IndexError) as e:
                self.logger.warning("PII JSON parse failed", error=str(e))
                pii_data = {}
                pii_count = 0
                # Fallback: count mentions in text
                if 'email' in pii_results.lower():
                    pii_count += pii_results.lower().count('email')
            
            try:
                secret_clean = secret_results.strip()
                if '```json' in secret_clean:
                    secret_clean = secret_clean.split('```json')[1].split('```')[0].strip()
                elif '```' in secret_clean:
                    secret_clean = secret_clean.split('```')[1].split('```')[0].strip()
                
                secret_data = json.loads(secret_clean)
                secret_count = len(secret_data.get('secrets_found', []))
            except (json.JSONDecodeError, IndexError) as e:
                self.logger.warning("Secret JSON parse failed", error=str(e))
                secret_data = {}
                secret_count = 0
                # Fallback: count mentions in text
                if 'api' in secret_results.lower() or 'key' in secret_results.lower():
                    secret_count += secret_results.lower().count('api_key') + secret_results.lower().count('aws')
            
            try:
                risk_clean = risk_assessment.strip()
                if '```json' in risk_clean:
                    risk_clean = risk_clean.split('```json')[1].split('```')[0].strip()
                elif '```' in risk_clean:
                    risk_clean = risk_clean.split('```')[1].split('```')[0].strip()
                
                risk_data = json.loads(risk_clean)
                risk_score = risk_data.get('overall_risk_score', 0)
                severity = risk_data.get('severity_level', 'unknown')
            except (json.JSONDecodeError, IndexError) as e:
                self.logger.warning("Risk JSON parse failed", error=str(e))
                risk_data = {}
                # Calculate fallback risk score
                risk_score = min((pii_count * 10) + (secret_count * 20), 100)
                severity = 'critical' if risk_score >= 70 else 'high' if risk_score >= 40 else 'medium' if risk_score >= 20 else 'low'
            
            # Step 5: Record metrics
            elapsed = time.time() - start_time
            self.metrics.record_scan(elapsed, pii_count, secret_count, severity)
            
            scan_result = {
                "risk_score": risk_score,
                "findings_count": pii_count + secret_count,
                "severity": severity
            }
            self.memory.add_scan_result(file_data['file_name'], scan_result)
            
            print(f"\nğŸ“Š SCAN COMPLETE ({elapsed:.2f}s)")
            print(f"Risk Score: {risk_score}/100 ({severity.upper()})")
            print(f"PII Found: {pii_count} | Secrets Found: {secret_count}")
            print("="*60 + "\n")
            
            self.logger.security("File scan completed",
                               file=file_path,
                               risk_score=risk_score,
                               findings=pii_count + secret_count)
            
            return {
                "file_info": file_data,
                "pii_findings": pii_data if pii_count else {},
                "secret_findings": secret_data if secret_count else {},
                "risk_assessment": risk_data,
                "summary": {
                    "risk_score": risk_score,
                    "severity": severity,
                    "pii_count": pii_count,
                    "secret_count": secret_count,
                    "scan_time": elapsed
                }
            }
            
        except Exception as e:
            self.metrics.record_error()
            self.logger.error("File scan failed", file=file_path, error=str(e))
            print(f"â�Œ Scan failed: {str(e)}")
            return {"error": str(e), "file": file_path}
    
    def batch_scan(self, file_paths: List[str], generate_report: bool = True) -> Dict[str, Any]:
        """
        ğŸš€ BATCH AUTOMATION: Scan multiple files and generate summary report
        
        Args:
            file_paths: List of file paths to scan
            generate_report: Whether to generate summary report
            
        Returns:
            Batch scan results with aggregated statistics
        """
        print(f"\nğŸ”� BATCH SCAN: Processing {len(file_paths)} files...")
        print("="*60 + "\n")
        
        results = []
        total_pii = 0
        total_secrets = 0
        high_risk_files = []
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{len(file_paths)}] {file_path}")
            try:
                result = self.scan_file(file_path)
                results.append(result)
                
                if 'summary' in result:
                    total_pii += result['summary'].get('pii_count', 0)
                    total_secrets += result['summary'].get('secret_count', 0)
                    
                    if result['summary'].get('severity') in ['critical', 'high']:
                        high_risk_files.append(file_path)
                        
            except Exception as e:
                print(f"  â�Œ Failed: {e}")
                results.append({"file": file_path, "error": str(e)})
        
        batch_summary = {
            "total_files": len(file_paths),
            "successful_scans": len([r for r in results if 'error' not in r]),
            "failed_scans": len([r for r in results if 'error' in r]),
            "total_pii_found": total_pii,
            "total_secrets_found": total_secrets,
            "high_risk_files": high_risk_files,
            "high_risk_count": len(high_risk_files),
            "results": results
        }
        
        if generate_report:
            self._print_batch_report(batch_summary)
        
        self.logger.security("Batch scan completed",
                           files=len(file_paths),
                           high_risk=len(high_risk_files))
        
        return batch_summary
    
    def _print_batch_report(self, summary: Dict[str, Any]):
        """Print formatted batch scan report"""
        print("\n" + "="*60)
        print("BATCH SCAN REPORT".center(60))
        print("="*60)
        print(f"Total Files Scanned: {summary['total_files']}")
        print(f"Successful: {summary['successful_scans']} | Failed: {summary['failed_scans']}")
        print(f"\nFindings:")
        print(f"  â€¢ Total PII Detected: {summary['total_pii_found']}")
        print(f"  â€¢ Total Secrets Found: {summary['total_secrets_found']}")
        print(f"  â€¢ High Risk Files: {summary['high_risk_count']}")
        
        if summary['high_risk_files']:
            print(f"\nâš ï¸�  High Risk Files:")
            for file in summary['high_risk_files']:
                print(f"    â€¢ {file}")
        
        print("="*60 + "\n")
    
    def generate_full_report(self, scan_result: Dict[str, Any], export_path: str = None) -> str:
        """
        Generate comprehensive security report
        
        Args:
            scan_result: Results from scan_file()
            export_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("DataSentry SECURITY REPORT".center(70))
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Team: {self.config['team']}")
        report_lines.append("")
        
        if 'file_info' in scan_result:
            fi = scan_result['file_info']
            report_lines.append("FILE INFORMATION")
            report_lines.append("-" * 70)
            report_lines.append(f"File: {fi['file_name']}")
            report_lines.append(f"Type: {fi['file_type']}")
            report_lines.append(f"Size: {fi['file_size_kb']:.2f} KB")
            report_lines.append("")
        
        if 'summary' in scan_result:
            s = scan_result['summary']
            report_lines.append("RISK ASSESSMENT")
            report_lines.append("-" * 70)
            report_lines.append(f"Overall Risk Score: {s['risk_score']}/100")
            report_lines.append(f"Severity Level: {s['severity'].upper()}")
            report_lines.append(f"PII Detected: {s['pii_count']}")
            report_lines.append(f"Secrets Found: {s['secret_count']}")
            report_lines.append(f"Scan Time: {s['scan_time']:.2f}s")
            report_lines.append("")
        
        if 'risk_assessment' in scan_result and scan_result['risk_assessment']:
            ra = scan_result['risk_assessment']
            if 'remediation_plan' in ra:
                report_lines.append("REMEDIATION PLAN")
                report_lines.append("-" * 70)
                for i, action in enumerate(ra['remediation_plan'][:5], 1):
                    report_lines.append(f"{i}. {action}")
                report_lines.append("")
        
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        if export_path:
            with open(export_path, 'w') as f:
                f.write(report)
            print(f"âœ“ Report exported to {export_path}")
        
        return report
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "agent_info": {
                "team": self.config['team'],
                "model": self.config['model'],
                "version": self.config['version']
            },
            "performance_metrics": self.metrics.get_summary(),
            "memory_stats": self.memory.get_stats(),
            "logger_stats": self.logger.get_stats()
        }
    
    def print_stats(self):
        """Print formatted statistics dashboard"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("DataSentry- STATISTICS DASHBOARD".center(70))
        print("="*70)
        
        print("\nğŸ¤– AGENT INFO")
        print("-"*70)
        for key, value in stats['agent_info'].items():
            print(f"{key:.<40} {value}")
        
        print("\nğŸ“Š PERFORMANCE METRICS")
        print("-"*70)
        for key, value in stats['performance_metrics'].items():
            print(f"{key:.<40} {value}")
        
        print("\nğŸ§  MEMORY STATS")
        print("-"*70)
        for key, value in stats['memory_stats'].items():
            print(f"{key:.<40} {value}")
        
        print("\nğŸ“� LOGGER STATS")
        print("-"*70)
        for key, value in stats['logger_stats'].items():
            print(f"{key:.<40} {value}")
        
        print("="*70 + "\n")
    
    def reset(self):
        """Reset agent memory"""
        self.memory.clear()
        self.logger.info("Agent reset")
        print("âœ“ Agent memory cleared")
    
    def export_session(self, filename: str = "privacy_guardian_session.json"):
        """Export complete session data"""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "stats": self.get_stats(),
            "conversation": self.memory.messages,
            "scan_history": self.memory.scan_history
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"âœ“ Session exported to {filename}")


# Cell 13: Initialize the Agent
if GOOGLE_API_KEY:
    agent = PrivacyGuardianAgent(
        config=CONFIG,
        tools=tools,
        memory=memory,
        logger=logger,
        metrics=metrics
    )
    print("\n" + "="*60)
    print("âœ“ PRIVACY GUARDIAN AGENT READY".center(60))
    print("="*60)
    print("\nğŸš€ AUTOMATION FEATURES:")
    print("  â€¢ agent.scan_file(path) - One-command full scan")
    print("  â€¢ agent.batch_scan([files]) - Process multiple files")
    print("  â€¢ agent.generate_full_report(result) - Export reports")
    print("  â€¢ agent.run('question') - Conversational queries")
    print("  â€¢ agent.print_stats() - View dashboard")
    print("="*60 + "\n")
else:
    agent = None
    print("âš  Configure API key to initialize agent")


print("\n" + "="*60)
print("PHASE 4 COMPLETE - AUTOMATION ENABLED".center(60))
print("="*60)
print("âœ“ Auto file reading (7 formats)")
print("âœ“ One-command full scan pipeline")
print("âœ“ Batch processing for multiple files")
print("âœ“ Automated report generation")
print("âœ“ Intelligent agent orchestration")
print("="*60)


# CELL-14: Automated Demonostration
print("="*70)
print("DataSentry - AUTOMATED DEMONSTRATION".center(70))
print("="*70)


result = agent.scan_file('/kaggle/input/vulnerable/vulnerable_data_pipeline.ipynb')


# Step 3: Display results
print("\nğŸ“Š Step 3: Security Scan Results")
print("="*70)

if 'summary' in result:
    s = result['summary']
    
    # Visual risk indicator
    if s['severity'] == 'critical':
        risk_icon = "ğŸ”´ CRITICAL"
    elif s['severity'] == 'high':
        risk_icon = "ğŸŸ  HIGH"
    elif s['severity'] == 'medium':
        risk_icon = "ğŸŸ¡ MEDIUM"
    else:
        risk_icon = "ğŸŸ¢ LOW"
    
    print(f"\nRisk Level: {risk_icon}")
    print(f"Risk Score: {s['risk_score']}/100")
    print(f"\nFindings:")
    print(f"  â€¢ PII Detected: {s['pii_count']} instances")
    print(f"  â€¢ Secrets Found: {s['secret_count']} instances")
    print(f"  â€¢ Scan Time: {s['scan_time']:.2f} seconds")
    
    # Show top recommendations
    if 'risk_assessment' in result and result['risk_assessment']:
        ra = result['risk_assessment']
        if 'remediation_plan' in ra and ra['remediation_plan']:
            print(f"\nğŸ”§ Top 3 Remediation Steps:")
            for i, action in enumerate(ra['remediation_plan'][:3], 1):
                print(f"   {i}. {action}")

print("\n" + "="*70)


# Step 4: Quick visualization 
if 'summary' in result and 'error' not in result:
    print("\nğŸ“ˆ Step 4: Generating security dashboard...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('DataSentry - Quick Security Report', fontsize=14, fontweight='bold')

    # Risk score gauge
    risk_score = result['summary']['risk_score']
    color = 'red' if risk_score >= 70 else 'orange' if risk_score >= 40 else 'yellow' if risk_score >= 20 else 'green'
    ax1.barh(['Risk Score'], [risk_score], color=color, height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Risk Score (0-100)')
    ax1.set_title(f'Overall Risk: {risk_score}/100')
    ax1.axvline(x=70, color='red', linestyle='--', alpha=0.3, label='Critical')
    ax1.axvline(x=40, color='orange', linestyle='--', alpha=0.3, label='High')
    ax1.legend()

    # Findings breakdown
    findings = {
        'PII': result['summary']['pii_count'],
        'Secrets': result['summary']['secret_count']
    }
    ax2.bar(findings.keys(), findings.values(), color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('Count')
    ax2.set_title('Security Findings')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('security_report.png', dpi=150, bbox_inches='tight')
    print("   âœ“ Dashboard saved: security_report.png")
    plt.show()
else:
    print("\nâš ï¸� Skipping visualization - scan had errors")


# Step 5: Export report 
if 'summary' in result and 'error' not in result:
    print("\nğŸ’¾ Step 5: Exporting reports...")
    report = agent.generate_full_report(result, export_path='security_report.txt')
    print("   âœ“ Full report saved: security_report.txt")

    agent.export_session('demo_session.json')
    print("   âœ“ Session data saved: demo_session.json")
else:
    print("\nâš ï¸� Skipping report export - scan had errors")


# Step 6: Show agent statistics
print("\nğŸ“Š Step 6: Agent Performance Statistics")
print("="*70)
stats = agent.get_stats()
perf = stats['performance_metrics']

print(f"Scans Completed: {perf['scans_completed']}")
print(f"Tools Called: {perf['tools_called']}")
print(f"Avg Response Time: {perf['avg_response_time']}s")
print(f"Total PII Detected: {perf['total_pii_detected']}")
print(f"Total Secrets Detected: {perf['total_secrets_detected']}")


# Demo complete message
print("\n" + "="*70)
print("âœ… DEMONSTRATION COMPLETE".center(70))
print("="*70)
print("""
What just happened (automatically):
1. Created vulnerable notebook
2. Ran full security scan (PII + Secrets + Risk)
3. Generated risk assessment
4. Created visualization dashboard
5. Exported detailed reports
6. Showed performance metrics

All with ONE command: agent.scan_file('filename')

ğŸš€ Try it yourself:
   result = agent.scan_file('your_notebook.ipynb')
   
ğŸ“Š For multiple files:
   agent.batch_scan(['file1.ipynb', 'file2.py'])
   
ğŸ’¬ Ask questions:
   agent.run("What should I fix first?")
""")
print("="*70)




