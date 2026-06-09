import os
import subprocess
import sys

def run_command(cmd, msg):
    print(f"â�³ {msg}...", end="\r")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"âœ… {msg} " + " " * 20) # Padding to clear line
    else:
        print(f"â�Œ {msg} (Failed)")
        print(f"Error: {result.stderr}")

print("ğŸš€ Starting Master Environment Setup...\n")

# 1ï¸�âƒ£ System Dependencies (Nmap + PDF Rendering Libraries)
# WeasyPrint needs libcairo, libpango, etc. to generate PDFs
sys_deps_cmd = (
    "apt-get update -qq && "
    "apt-get install -y nmap libcairo2 libpango-1.0-0 libpangocairo-1.0-0 "
    "libgdk-pixbuf2.0-0 libffi-dev libjpeg-dev libxml2-dev libxslt1-dev "
    "zlib1g-dev -qq"
)
run_command(sys_deps_cmd, "Installing System Deps (Nmap + PDF Drivers)")

# 2ï¸�âƒ£ Install GO 1.23 (Required for Nuclei)
run_command("rm -rf /usr/local/go", "Cleaning old Go")
run_command("wget -q https://go.dev/dl/go1.23.2.linux-amd64.tar.gz", "Downloading Go 1.23")
run_command("tar -C /usr/local -xzf go1.23.2.linux-amd64.tar.gz", "Extracting Go 1.23")

# Set PATH for this  so Python can find Go
os.environ["PATH"] += ":/usr/local/go/bin:/root/go/bin"
os.environ["GOPATH"] = "/root/go"

# 3ï¸�âƒ£ Install Security Tools (Nuclei & Amass)
run_command("/usr/local/go/bin/go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest", "Installing Nuclei (Latest)")
run_command("/usr/local/go/bin/go install github.com/owasp-amass/amass/v4/...@master", "Installing Amass (Master)")

# 4ï¸�âƒ£ Clone & Prepare Templates
run_command("rm -rf /root/nuclei-templates && git clone https://github.com/projectdiscovery/nuclei-templates.git /root/nuclei-templates", 
            "Cloning Nuclei Templates")

# 4.1 Create Minimal Template Set (For faster/safer scanning)
minimal_cmd = r"""
set -e
mkdir -p /root/nuclei-templates/minimal
# Copy safe/informational templates
# Note: We check if files exist before copying to avoid errors if repo structure changes
cp /root/nuclei-templates/http/exposures/configs/symfony-profiler.yaml /root/nuclei-templates/minimal/ 2>/dev/null || true
cp /root/nuclei-templates/http/exposures/configs/symfony-security-config.yaml /root/nuclei-templates/minimal/ 2>/dev/null || true
cp /root/nuclei-templates/http/exposures/configs/webpack-config.yaml /root/nuclei-templates/minimal/ 2>/dev/null || true
cp /root/nuclei-templates/http/exposures/configs/vite-config.yaml /root/nuclei-templates/minimal/ 2>/dev/null || true
"""
run_command(minimal_cmd, "Configuring 'Minimal' Template Set")

# 5ï¸�âƒ£ Install Python AI & Reporting Libraries
# We install google-adk, generativeai, and weasyprint (for PDFs)
run_command("pip install -q -U google-adk google-generativeai xmltodict weasyprint", "Installing Python AI & PDF Libs")

print("\nğŸ�‰ setup Complete!")
print(f"ğŸ“‚ Templates: /root/nuclei-templates/minimal")
print(f"ğŸ§  AI Agents: Ready (ADK Installed)")
print(f"ğŸ“„ PDF Gen: Ready (WeasyPrint Installed)")


import google.generativeai as genai
from google.adk import Agent
import weasyprint
import xmltodict

print("âœ… Imports successful! You are ready to code.")




import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )




import os, re, json, time, shutil, subprocess, textwrap, glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# -------------------------
# âš™ï¸� CONFIGURATION
# -------------------------

# [CRITICAL] KAGGLE SAFETY TOGGLE
# Set False for submission to avoid "Unauthorized Scanning" bans.
# Set True only if scanning YOUR OWN 'localhost' or authorized labs.
RUN_LIVE_SCAN = False # <-------- set False to run the live scan

# Target Scope (Only used if RUN_LIVE_SCAN is True)
TARGET_DOMAIN = "demo.testfire.net"   # Use a safe demo site
SAFE_MODE = True                      # Enforces strict rate limits
OUTPUT_DIR = "artifacts"              # Where reports/logs are saved

# Tool Configuration
NUCLEI_TEMPLATES_DIR = "/root/nuclei-templates"
NUCLEI_RATE_LIMIT = 50                # Slow down to prevent WAF blocking
TOP_N_FOR_REPORT = 5                  # Limit AI context to top 5 risks

# [DATA SOURCE] Offline Mode Input Folder
# If Live Scan is OFF, the pipeline looks here for uploaded JSON files.
RAW_DIR = "/kaggle/input/nuclei-scans" # <------- Offline input of the scaned file

os.makedirs(OUTPUT_DIR, exist_ok=True)


def html_escape(s: str) -> str:
    """Escape text so it can be safely inserted into HTML."""
    if s is None:
        return ""
    return (str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


class GeminiClient:
    """
    Simple wrapper to call Gemini.
    Prefers gemini-2.5-pro for deep reasoning, flash for speed.
    Auto-disables itself if API key or SDK are unavailable.
    """
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.enabled = bool(self.api_key)
        self.genai = None
        self._init()

    def _init(self):
        if not self.enabled:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except Exception:
            self.enabled = False
            self.genai = None

    def generate(self, prompt: str, model: str = "gemini-2.5-flash", temperature: float = 0.3) -> str:
        """Generate text from Gemini; returns empty string on failure."""
        if not self.enabled or self.genai is None:
            return ""
        try:
            m = self.genai.GenerativeModel(model)
            resp = m.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return (resp.text or "").strip()
        except Exception:
            return ""


def have(cmd: str) -> bool:
    """Checks if a specific tool (like 'nmap' or 'nuclei') is installed in PATH."""
    return shutil.which(cmd) is not None


def run(cmd: List[str], outfile: Optional[str] = None, timeout: int = 3600, env=None) -> str:
    """
    Executes a system command (subprocess) securely.
    - Captures STDOUT/STDERR.
    - Handles timeouts to prevent hanging agents.
    - Optionally writes output directly to a file.
    """
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if res.returncode != 0:
        print(res.stderr[:1200])  # Print first 1200 chars of error
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    out = res.stdout
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(out)
    return out


def jsonl_read(path: str) -> List[Dict]:
    """Reads JSON Lines (.jsonl) files, common in security tool outputs."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # Skip malformed lines quietly
                pass
    return rows


def json_write(obj, path: str):
    """Helper to dump Python objects to JSON files with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def require_target():
    """Safety check to ensure we don't scan empty targets in Live Mode."""
    if RUN_LIVE_SCAN and not TARGET_DOMAIN.strip():
        raise ValueError("Set TARGET_DOMAIN to a PERMISSIONED/LAB domain before running.")


@dataclass
class Finding:
    """
    The standard 'Currency' of this pipeline.
    Every tool's output is converted into this format so the AI can read it.
    """
    title: str            # Name of the vulnerability (e.g., "SQL Injection")
    severity: str         # low, medium, high, critical
    host: str             # The URL or IP affected
    evidence: str         # Proof (e.g., the payload used)
    template_id: str      # ID of the rule that found it
    timestamp: str        # When it was found
    cve_ids: List[str]    # Common Vulnerabilities and Exposures IDs
    cve_links: List[str]  # Links to NVD database
    cvss: Optional[float] # Numeric risk score (0.0 - 10.0)
    confidence: str       # How sure is the tool?
    risk_score: float     # Our custom calculated priority score
    notes: str            # AI or Logic generated remediation hints


# -------------------------
# ğŸ•µï¸� STEP 2 â€” ENV CHECK
# -------------------------
def ensure_environment():
    """
    Pre-flight check. Verifies that Nmap, Nuclei, and Amass are installed.
    If tools are missing in Live Mode, it warns the user.
    """
    print("\n[ENV] Checking tools...")
    missing = []
    for tool in ["amass", "nuclei"]:
        if RUN_LIVE_SCAN and not have(tool):
            missing.append(tool)

    if missing:
        print(f"[ENV] Missing tools: {missing}")
        print("[ENV] Install them in a separate Kaggle cell before running.")
    if RUN_LIVE_SCAN and not os.path.isdir(NUCLEI_TEMPLATES_DIR):
        print(f"[ENV] Missing {NUCLEI_TEMPLATES_DIR}. Clone it first.")
    print("[ENV] OK (or will skip missing steps).")


# -------------------------
# ğŸ¤– AGENT 1: DISCOVERY (The Scout)
# -------------------------
def discovery_agent() -> List[str]:
    """
    [LIVE MODE ONLY]
    Role: Reconnaissance.
    Task: Uses 'Amass' to perform passive subdomain enumeration.
          It finds hidden parts of the website (e.g., 'dev.target.com') without
          directly attacking them.
    Output: A list of assets (subdomains) saved to 'assets.txt'.
    """
    require_target()
    print("\n[DISCOVERY] Running Amass passive discovery...")
    assets = set()

    assets_path = os.path.join(OUTPUT_DIR, "assets.txt")

    # Amass passive scan
    if have("amass"):
        try:
            # Run amass in passive mode (-passive) to stay stealthy
            out = run(
                ["amass", "enum", "-passive", "-d", TARGET_DOMAIN],
                timeout=1200
            )
            for line in out.splitlines():
                line = line.strip()
                if line.endswith(TARGET_DOMAIN):
                    assets.add(line)
        except Exception as e:
            print("[DISCOVERY] Amass timeout/blocked. Falling back to domain only.")

    # Fallback: If reconnaissance fails, at least scan the main domain
    if not assets:
        assets.add(TARGET_DOMAIN)

    assets = sorted(assets)
    with open(assets_path, "w") as f:
        f.write("\n".join(assets))

    print(f"[DISCOVERY] Found {len(assets)} assets. Saved to {assets_path}")
    return assets


# -------------------------
# ğŸ¤– AGENT 2: SCANNER (The Hunter)
# -------------------------
def scanner_agent(assets: List[str]) -> str:
    """
    [LIVE MODE ONLY]
    Role: Vulnerability Scanning.
    Task: Takes the list of assets from the Discovery Agent and runs 'Nuclei'.
          It checks for misconfigurations, exposed panels, and known CVEs.
    Safety: Removes API keys from environment variables so they aren't leaked
            in the tool's logs.
    Output: A raw JSONL file containing technical findings.
    """
    require_target()
    print("\n[SCAN] Running Nuclei safe scan...")

    assets_path = os.path.join(OUTPUT_DIR, "assets.txt")
    scan_path = os.path.join(OUTPUT_DIR, "scan.jsonl")

    # We use all templates in the directory
    safe_template_paths = [f"{NUCLEI_TEMPLATES_DIR}/**/*.yaml"]

    cmd = [
        "nuclei",
        "-l", assets_path,
        "-t", ",".join(safe_template_paths),
        "-exclude-tags", "osint,dork,google",  # Don't hit Google APIs
        "-rate-limit", str(NUCLEI_RATE_LIMIT), # Stay under WAF radar
        "-rate-limit", "50",
        "-severity", "low,medium,high,critical",
        "-jsonl",
        "-o", scan_path
    ]

    # [SECURITY] Scrub sensitive env vars before passing to subprocess
    env = os.environ.copy()
    env.pop("GOOGLE_API_KEY", None)

    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print("STDOUT:\n", res.stdout[:1500])
        print("STDERR:\n", res.stderr[:1500])
        raise RuntimeError("Nuclei failed")

    print(f"[SCAN] Saved raw nuclei JSONL to {scan_path}")
    return scan_path


# -------------------------
# ğŸ¤– AGENT 2b: OFFLINE INGEST (The Alternative)
# -------------------------
def offline_ingest_agent(raw_dir: str) -> str:
    """
    [KAGGLE SAFE MODE]
    Role: Data Ingestion.
    Task: If Live Scanning is disabled, this agent looks for pre-uploaded
          scan files (Synthetic Data) in the Kaggle Input directory.
          This allows the pipeline to run legally without internet attacks.
    Output: Merges all found files into 'scan.jsonl'.
    """
    print("\n[INGEST] Offline mode: loading sample scan files...")
    scan_path = os.path.join(OUTPUT_DIR, "scan.jsonl")
    files = glob.glob(os.path.join(raw_dir, "**/*.*"), recursive=True)

    rows = []
    for f in files:
        lf = f.lower()
        try:
            if lf.endswith(".jsonl"):
                rows.extend(jsonl_read(f))
            elif lf.endswith(".json"):
                data = json.load(open(f, "r", encoding="utf-8"))
                if isinstance(data, list):
                    rows.extend(data)
        except Exception as e:
            print("Skipping:", f, "reason:", e)

    with open(scan_path, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")

    print(f"[INGEST] Loaded {len(rows)} raw records â†’ {scan_path}")
    return scan_path



# -------------------------
# ğŸ¤– AGENT 3: PARSER (The Translator)
# -------------------------
def parser_agent(scan_path: str) -> List[Finding]:
    """
    Role: Normalization.
    Task: Reads the messy JSON output from tools (Nuclei/Nmap) and maps it
          to our clean 'Finding' dataclass.
    Why:  The AI (Gemini) needs consistent field names to understand the data.
    Output: A list of structured 'Finding' objects saved to 'findings.json'.
    """
    print("\n[PARSE] Normalizing nuclei output...")
    rows = jsonl_read(scan_path)
    findings: List[Finding] = []

    for r in rows:
        # Extract fields safely using .get()
        info = r.get("info", {}) or {}
        title = info.get("name", "Unknown Finding")
        severity = (info.get("severity") or "info").lower()
        host = r.get("host") or r.get("matched-at") or "unknown"
        evidence = (
            r.get("matched-at") or
            r.get("extractor") or
            r.get("curl-command") or
            str(r)[:220]
        )
        template_id = r.get("template-id") or info.get("id") or "unknown"
        timestamp = r.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        findings.append(Finding(
            title=title,
            severity=severity,
            host=host,
            evidence=evidence,
            template_id=template_id,
            timestamp=timestamp,
            cve_ids=[],      # Will be filled by Enricher
            cve_links=[],
            cvss=None,
            confidence="low",
            risk_score=0.0,
            notes=""
        ))

    findings_path = os.path.join(OUTPUT_DIR, "findings.json")
    json_write([asdict(f) for f in findings], findings_path)
    print(f"[PARSE] {len(findings)} findings â†’ {findings_path}")
    return findings


# -------------------------
# ğŸ¤– AGENT 4: ENRICHER & SCORER (The Analyst)
# -------------------------
SEV_BASE = {"info": 0, "low": 2, "medium": 5, "high": 8, "critical": 10}

def enrich_and_score_agent(findings: List[Finding]) -> List[Finding]:
    """
    Role: Risk Assessment Logic.
    Task: 1. Extracts CVE IDs using Regex (e.g., CVE-2023-1234).
          2. Generates links to the National Vulnerability Database (NVD).
          3. Calculates a 'Risk Score' based on severity + confidence.
    Output: The same list of findings, but now populated with scores and links.
    """
    print("\n[ENRICH+SCORE] Adding CVE links + risk scoring...")
    cve_re = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)

    for f in findings:
        # 1. Find CVEs in the text
        cves = set()
        for text in [f.title, f.template_id, f.evidence]:
            for m in cve_re.findall(text or ""):
                cves.add(m.upper())
        f.cve_ids = sorted(cves)
        f.cve_links = [f"https://nvd.nist.gov/vuln/detail/{c}" for c in f.cve_ids]

        # 2. Determine Confidence
        if "cves/" in f.template_id or f.cve_ids:
            f.confidence = "high"
        elif f.severity in ("high", "critical"):
            f.confidence = "medium"
        else:
            f.confidence = "low"

        # 3. Calculate Math Score
        base = SEV_BASE.get(f.severity, 0)
        cve_bonus = 1.5 if f.cve_ids else 0
        conf_mult = {"low": 0.8, "medium": 1.0, "high": 1.2}[f.confidence]
        f.risk_score = round((base + cve_bonus) * conf_mult, 2)

        f.notes = "Known CVE detected; prioritize patching." if f.cve_ids else "No CVE tag; verify manually."

    scored_path = os.path.join(OUTPUT_DIR, "findings_scored.json")
    json_write([asdict(x) for x in findings], scored_path)
    print(f"[ENRICH+SCORE] Saved â†’ {scored_path}")
    return findings


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Web Application Security Assessment Report - {{SITE_NAME}}</title>

  <style>
    :root{
      --ink:#0f172a; --muted:#475569; --line:#e2e8f0; --soft:#f8fafc;
      --critical:#b91c1c; --high:#c2410c; --medium:#b45309; --low:#0f766e; --info:#1d4ed8;
    }
    *{box-sizing:border-box}
    body{
      margin:0; padding:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color:var(--ink); background:#fff; line-height:1.55;
    }
    .page{
      width: 210mm;
      min-height: 297mm;
      padding: 18mm 16mm;
      margin: 0 auto;
      border-bottom: 10px solid #fff;
      position: relative;
    }
    .page-break{ page-break-after: always; break-after: page; }

    header.report-header{
      display:flex; justify-content:space-between; align-items:flex-start;
      border-bottom:2px solid var(--ink); padding-bottom:10px; margin-bottom:14px;
    }
    .title h1{font-size:22px; margin:0 0 6px 0; letter-spacing:.2px;}
    .title p{margin:0; color:var(--muted); font-size:13px;}
    .doc-meta{
      font-size:12.5px; color:var(--muted); text-align:right;
    }
    .doc-meta div{margin-bottom:4px}
    .badge{
      display:inline-block; padding:3px 8px; border-radius:999px; font-weight:700; font-size:12px; letter-spacing:.3px;
      color:#fff; vertical-align:middle;
    }
    .b-critical{background:var(--critical)}
    .b-high{background:var(--high)}
    .b-medium{background:var(--medium)}
    .b-low{background:var(--low)}
    .b-info{background:var(--info)}

    h2{font-size:16.5px; margin:14px 0 8px}
    h3{font-size:14.5px; margin:12px 0 6px}
    p{margin:6px 0}
    ul{margin:6px 0 6px 18px}
    li{margin:3px 0}
    .muted{color:var(--muted)}

    .card{
      border:1px solid var(--line); background:var(--soft);
      border-radius:12px; padding:10px 12px; margin:8px 0;
    }
    table{
      width:100%; border-collapse:collapse; margin-top:6px; font-size:13.5px;
    }
    th, td{
      border:1px solid var(--line); padding:8px 9px; vertical-align:top;
    }
    th{background:#f1f5f9; text-align:left}

    .severity-row td:first-child{font-weight:700}
    .sev-critical{color:var(--critical)}
    .sev-high{color:var(--high)}
    .sev-medium{color:var(--medium)}
    .sev-low{color:var(--low)}

    .finding{
      border:1px solid var(--line); border-radius:14px; padding:12px; margin:10px 0;
    }
    .finding-header{
      display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:4px;
    }
    .finding-title{font-weight:800; font-size:14.8px;}
    .finding-meta{font-size:12.5px; color:var(--muted)}
    .poc{
      background:#0b1020; color:#e5e7eb; border-radius:10px; padding:10px; font-size:12.8px; overflow:auto;
      margin:6px 0;
    }
    .poc code{color:#e5e7eb}

    .grid-2{display:grid; grid-template-columns: 1fr 1fr; gap:10px;}
    .risk-matrix{
      display:grid; grid-template-columns: 70px repeat(3,1fr); border:1px solid var(--line); border-radius:10px; overflow:hidden;
      font-size:12.8px; margin-top:6px;
    }
    .risk-matrix div{border-right:1px solid var(--line); border-bottom:1px solid var(--line); padding:8px}
    .risk-matrix .head{background:#f1f5f9; font-weight:700}
    .risk-matrix .cell-critical{background:#fee2e2}
    .risk-matrix .cell-high{background:#ffedd5}
    .risk-matrix .cell-medium{background:#fef9c3}

    footer.page-footer{
      position:absolute; bottom:10mm; left:16mm; right:16mm;
      display:flex; justify-content:space-between; font-size:11px; color:var(--muted);
      border-top:1px dashed var(--line); padding-top:6px;
    }

    @media print{
      body{background:#fff}
      .page{border:none; margin:0; width:auto; min-height:auto;}
    }
  </style>
</head>

<body>

  <!-- =========================
       PAGE 1: EXEC SUMMARY
  ========================== -->
  <section class="page page-break">
    <header class="report-header">
      <div class="title">
        <h1>Web Application Security Assessment Report â€” {{SITE_NAME}}</h1>
        <p>Executive Summary & Scope</p>
      </div>
      <div class="doc-meta">
        <div><strong>Date:</strong> {{DATE}}</div>
        <div><strong>Author:</strong> {{AUTHOR}}</div>
        <div><strong>Version:</strong> {{VERSION}}</div>
      </div>
    </header>

    <h2>1. Executive Summary (Bottom Line)</h2>
    {{EXEC_SUMMARY_HTML}}

    <div class="card">
      <p style="margin-top:0">
        <strong>Overall Security Status:</strong>
        <span class="badge {{STATUS_BADGE_CLASS}}">{{STATUS_LABEL}}</span>
      </p>
      <p class="muted" style="margin-bottom:0">
        {{STATUS_LINE}}
      </p>
    </div>

    <h3>Business Impact</h3>
    {{BUSINESS_IMPACT_UL}}

    <h2>2. Scorecard / Visual Summary</h2>
    {{SCORECARD_TABLE}}

    <h2>3. Scope & Methodology</h2>
    <div class="grid-2">
      <div class="card">
        <h3 style="margin-top:0">Scope</h3>
        {{SCOPE_HTML}}
      </div>

      <div class="card">
        <h3 style="margin-top:0">Methodology & Tools</h3>
        {{METHODOLOGY_UL}}
      </div>
    </div>

    <footer class="page-footer">
      <div>{{SITE_NAME}} Security Assessment</div>
      <div>Page 1 of 3</div>
    </footer>
  </section>


  <!-- =========================
       PAGE 2: TECH FINDINGS
  ========================== -->
  <section class="page page-break">
    <header class="report-header">
      <div class="title">
        <h1>Detailed Technical Findings</h1>
        <p>Critical & High Risk Issues</p>
      </div>
      <div class="doc-meta">
        <div><strong>Target:</strong> {{TARGET_URL}}</div>
        <div><strong>Version:</strong> {{VERSION}}</div>
      </div>
    </header>

    {{TOP_FINDINGS_HTML}}

    <div class="card">
      <h3 style="margin-top:0">Other Medium / Low Observations</h3>
      {{OTHER_FINDINGS_UL}}
    </div>

    <footer class="page-footer">
      <div>{{SITE_NAME}} Security Assessment</div>
      <div>Page 2 of 3</div>
    </footer>
  </section>


  <!-- =========================
       PAGE 3: REMEDIATION
  ========================== -->
  <section class="page">
    <header class="report-header">
      <div class="title">
        <h1>Remediation & Conclusion</h1>
        <p>Fix Strategy, Risk Model, Closing Notes</p>
      </div>
      <div class="doc-meta">
        <div><strong>Target:</strong> {{TARGET_URL}}</div>
        <div><strong>Version:</strong> {{VERSION}}</div>
      </div>
    </header>

    <h2>1. Remediation Strategy (Fixes)</h2>
    {{REMEDIATION_HTML}}

    <div class="card">
      <h3 style="margin-top:0">General Best Practices (Quick Wins)</h3>
      {{BEST_PRACTICES_UL}}
    </div>

    <h2>2. Risk Matrix (Likelihood Ã— Impact)</h2>
    <p class="muted">
      Risks were rated by combining exploit likelihood (ease + exposure) and business impact (data/system/user harm).
    </p>

    <div class="risk-matrix">
      <div class="head">Impact â†“ / Likelihood â†’</div>
      <div class="head">Low</div>
      <div class="head">Medium</div>
      <div class="head">High</div>

      <div class="head">Low</div>
      <div>Low</div>
      <div>Low</div>
      <div>Medium</div>

      <div class="head">Medium</div>
      <div>Low</div>
      <div class="cell-medium">Medium</div>
      <div class="cell-high">High</div>

      <div class="head">High</div>
      <div>Medium</div>
      <div class="cell-high">High</div>
      <div class="cell-critical">Critical</div>
    </div>

    <h2>3. Conclusion</h2>
    {{CONCLUSION_HTML}}

    <h2>4. Disclaimer</h2>
    <p class="muted">
      This report reflects the security posture of the application during the stated testing window.
      It does not guarantee immunity against all future vulnerabilities or attacks. Security is an ongoing process.
    </p>

    <footer class="page-footer">
      <div>{{SITE_NAME}} Security Assessment</div>
      <div>Page 3 of 3</div>
    </footer>
  </section>

</body>
</html>
"""


def reporter_agent(findings: List[Finding]) -> str:
    """
    Multi-agent content generation + HTML injection.
    Agents:
      A) ExecutiveSummaryAgent (gemini-2.5-pro)
      B) ScorecardScopeAgent   (gemini-2.5-flash)
      C) FindingsWriterAgent   (gemini-2.5-pro)
      D) RemediationAgent      (gemini-2.5-pro)
    Output:
      artifacts/report.html
    """
    print("\n[REPORT] Generating executive + technical report (multi-agent)...")

    # ------------------------
    # 1) Prepare payloads
    # ------------------------
    top = sorted(findings, key=lambda x: x.risk_score, reverse=True)[:TOP_N_FOR_REPORT]
    rest = sorted(findings, key=lambda x: x.risk_score, reverse=True)[TOP_N_FOR_REPORT:]

    payload_top = [asdict(x) for x in top]
    payload_all = [asdict(x) for x in findings]

    # Severity counts
    sev_counts = { "critical":0, "high":0, "medium":0, "low":0, "info":0 }
    for f in findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1

    # Overall status label
    if sev_counts["critical"] > 0:
        status_label = "CRITICAL"
        status_class = "b-critical"
    elif sev_counts["high"] > 0:
        status_label = "AT RISK"
        status_class = "b-high"
    elif sev_counts["medium"] > 0:
        status_label = "NEEDS IMPROVEMENT"
        status_class = "b-medium"
    else:
        status_label = "GOOD"
        status_class = "b-low"

    status_line = (
        f"The assessment identified "
        f"<strong>{sev_counts['critical']} Critical</strong>, "
        f"<strong>{sev_counts['high']} High</strong>, "
        f"<strong>{sev_counts['medium']} Medium</strong>, and "
        f"<strong>{sev_counts['low']} Low</strong> issues."
    )

    site_name = TARGET_DOMAIN if RUN_LIVE_SCAN else "ExampleSite"
    target_url = f"https://{TARGET_DOMAIN}" if RUN_LIVE_SCAN else "https://example.com"

    # ------------------------
    # 2) Initialize Gemini
    # ------------------------
    llm = GeminiClient()

    # ------------------------
    # 3) Agent A: Executive summary
    # ------------------------
    exec_summary_html = ""
    business_impact_ul = ""

    if llm.enabled:
        prompt = f"""
You are ExecutiveSummaryAgent.
Write ONLY from the findings JSON below. Don't invent new vulns.

Findings (top risks):
{json.dumps(payload_top, indent=2)}

Return:
1) A tight executive summary as HTML paragraph(s) (<p>..</p>)
2) Business impact bullets as <ul><li>..</li></ul>
Keep it enterprise and non-technical.

Output format:
[EXEC_SUMMARY_HTML]
...
[/EXEC_SUMMARY_HTML]
[BUSINESS_IMPACT_UL]
...
[/BUSINESS_IMPACT_UL]
"""
        resp = llm.generate(prompt, model="gemini-2.5-pro", temperature=0.2)
        # crude tag extraction
        m1 = re.search(r"\[EXEC_SUMMARY_HTML\](.*?)\[/EXEC_SUMMARY_HTML\]", resp, re.S)
        m2 = re.search(r"\[BUSINESS_IMPACT_UL\](.*?)\[/BUSINESS_IMPACT_UL\]", resp, re.S)
        exec_summary_html = m1.group(1).strip() if m1 else ""
        business_impact_ul = m2.group(1).strip() if m2 else ""

    if not exec_summary_html:
        exec_summary_html = f"""
<p>
A targeted security assessment was conducted on <strong>{html_escape(site_name)}</strong> to identify
vulnerabilities that could impact confidentiality, integrity, or availability.
Testing combined automated scanning and manual verification focused on authentication,
input handling, and  management.
</p>
<p>
The objective was to discover exploitable weaknesses before they can be abused.
</p>
""".strip()

    if not business_impact_ul:
        business_impact_ul = """
<ul>
  <li><strong>Data exposure:</strong> attacks may leak sensitive user or business data.</li>
  <li><strong>Account takeover:</strong>  or auth weaknesses can enable hijacking.</li>
  <li><strong>Operational risk:</strong> exploitation can cause downtime and trust loss.</li>
</ul>
""".strip()

    # ------------------------
    # 4) Agent B: Scorecard + Scope + Methodology
    # ------------------------
    scorecard_table = ""
    scope_html = ""
    methodology_ul = ""

    if llm.enabled:
        prompt = f"""
You are ScorecardScopeAgent.
Use ONLY the JSON and severity counts. Don't invent.

Severity counts:
{json.dumps(sev_counts, indent=2)}

Top findings:
{json.dumps(payload_top, indent=2)}

Return:
A) A full HTML <table> for the scorecard with notes per severity
B) Scope block as HTML <p> lines (Target URL, period, env, out-of-scope)
C) Methodology as <ul><li>..</li></ul>

Output format:
[SCORECARD_TABLE]
...
[/SCORECARD_TABLE]
[SCOPE_HTML]
...
[/SCOPE_HTML]
[METHODOLOGY_UL]
...
[/METHODOLOGY_UL]
"""
        resp = llm.generate(prompt, model="gemini-2.5-flash", temperature=0.2)
        m1 = re.search(r"\[SCORECARD_TABLE\](.*?)\[/SCORECARD_TABLE\]", resp, re.S)
        m2 = re.search(r"\[SCOPE_HTML\](.*?)\[/SCOPE_HTML\]", resp, re.S)
        m3 = re.search(r"\[METHODOLOGY_UL\](.*?)\[/METHODOLOGY_UL\]", resp, re.S)

        scorecard_table = m1.group(1).strip() if m1 else ""
        scope_html = m2.group(1).strip() if m2 else ""
        methodology_ul = m3.group(1).strip() if m3 else ""

    if not scorecard_table:
        scorecard_table = f"""
<table>
  <thead>
    <tr><th>Severity</th><th>Count</th><th>Notes</th></tr>
  </thead>
  <tbody>
    <tr class="severity-row">
      <td class="sev-critical">Critical</td>
      <td>{sev_counts['critical']}</td>
      <td>Remote exploitation likely; direct data/system impact.</td>
    </tr>
    <tr class="severity-row">
      <td class="sev-high">High</td>
      <td>{sev_counts['high']}</td>
      <td>Exploitation feasible; significant business impact.</td>
    </tr>
    <tr class="severity-row">
      <td class="sev-medium">Medium</td>
      <td>{sev_counts['medium']}</td>
      <td>Requires conditions/user interaction; moderate impact.</td>
    </tr>
    <tr class="severity-row">
      <td class="sev-low">Low</td>
      <td>{sev_counts['low']}</td>
      <td>Minor impact or low likelihood.</td>
    </tr>
  </tbody>
</table>
""".strip()

    if not scope_html:
        scope_html = f"""
<p><strong>Target URL:</strong> {html_escape(target_url)}</p>
<p><strong>Testing Period:</strong> 20 Nov â€“ 22 Nov 2025</p>
<p><strong>Environment:</strong> Production (non-destructive tests only)</p>
<p><strong>Out of Scope:</strong> DoS/Stress testing, 3rd-party vendor systems</p>
""".strip()

    if not methodology_ul:
        methodology_ul = """
<ul>
  <li>Automated discovery and scan (Nuclei, optional Nmap)</li>
  <li>Manual validation where applicable</li>
  <li>OWASP Testing Guide aligned checks</li>
  <li>Risk rated by Likelihood Ã— Impact</li>
</ul>
""".strip()

    # ------------------------
    # 5) Agent C: Top findings HTML
    # ------------------------
    top_findings_html = ""

    if llm.enabled and payload_top:
        prompt = f"""
You are FindingsWriterAgent.
Write enterprise-grade technical findings in HTML.
Use ONLY the JSON. Do NOT invent endpoints or PoCs.

Findings JSON:
{json.dumps(payload_top, indent=2)}

For each finding, output a block matching:

<div class="finding">
  <div class="finding-header">
    <div class="finding-title">#) TITLE</div>
    <div class="badge b-SEVERITY">SEVERITY</div>
  </div>
  <div class="finding-meta">Affected Endpoint ...</div>
  <h3>Description</h3><p>...</p>
  <h3>Evidence / Proof of Concept (PoC)</h3>
  <div class="poc"><code>...</code></div>
  <h3>Impact</h3><ul><li>...</li></ul>
</div>

Return ONLY HTML.
"""
        top_findings_html = llm.generate(prompt, model="gemini-2.5-pro", temperature=0.25)

    if not top_findings_html:
        # deterministic fallback
        blocks = []
        for i, f in enumerate(top, 1):
            sev = f.severity.lower()
            badge = f"b-{sev if sev in ('critical','high','medium','low','info') else 'info'}"
            blocks.append(f"""
<div class="finding">
  <div class="finding-header">
    <div class="finding-title">{i}) {html_escape(f.title)}</div>
    <div class="badge {badge}">{html_escape(sev.upper())}</div>
  </div>
  <div class="finding-meta">Affected Host: <code>{html_escape(f.host)}</code> â€¢ Template: <code>{html_escape(f.template_id)}</code></div>
  <h3>Description</h3>
  <p>{html_escape(f.notes or "Automated finding; validate manually.")}</p>
  <h3>Evidence / Proof of Concept (PoC)</h3>
  <div class="poc"><code>{html_escape(f.evidence)}</code></div>
  <h3>Impact</h3>
  <ul>
    <li>Potential compromise depending on exploitability.</li>
    <li>May affect confidentiality, integrity, or availability.</li>
  </ul>
</div>
""")
        top_findings_html = "\n".join(blocks)

    # ------------------------
    # 6) Medium/Low list
    # ------------------------
    other_ul_items = []
    for f in rest[:12]:
        other_ul_items.append(
            f"<li><strong>{html_escape(f.title)}</strong> "
            f"(<code>{html_escape(f.severity)}</code>) â€” {html_escape(f.host)}</li>"
        )
    other_findings_ul = "<ul>" + "\n".join(other_ul_items) + "</ul>" if other_ul_items else "<ul><li>No additional findings.</li></ul>"

    # ------------------------
    # 7) Agent D: Remediation + Best practices + Conclusion
    # ------------------------
    remediation_html = ""
    best_practices_ul = ""
    conclusion_html = ""

    if llm.enabled:
        prompt = f"""
You are RemediationAgent.
Use ONLY findings JSON. Do NOT invent new vulns.

Findings JSON:
{json.dumps(payload_top, indent=2)}

Return:
A) Remediation blocks in HTML, one per finding, matching this style:

<div class="finding">
  <div class="finding-header">
    <div class="finding-title">Recommendation for Finding #: TITLE</div>
    <div class="badge b-SEVERITY">SEVERITY</div>
  </div>
  <ul>
    <li>Fix...</li>
  </ul>
  (optional) code snippet inside <div class="poc"><code>..</code></div>
</div>

B) General best practices as a <ul>
C) Conclusion as HTML <p>

Output format:
[REMEDIATION_HTML]...[/REMEDIATION_HTML]
[BEST_PRACTICES_UL]...[/BEST_PRACTICES_UL]
[CONCLUSION_HTML]...[/CONCLUSION_HTML]
"""
        resp = llm.generate(prompt, model="gemini-2.5-pro", temperature=0.25)
        m1 = re.search(r"\[REMEDIATION_HTML\](.*?)\[/REMEDIATION_HTML\]", resp, re.S)
        m2 = re.search(r"\[BEST_PRACTICES_UL\](.*?)\[/BEST_PRACTICES_UL\]", resp, re.S)
        m3 = re.search(r"\[CONCLUSION_HTML\](.*?)\[/CONCLUSION_HTML\]", resp, re.S)

        remediation_html = m1.group(1).strip() if m1 else ""
        best_practices_ul = m2.group(1).strip() if m2 else ""
        conclusion_html = m3.group(1).strip() if m3 else ""

    if not remediation_html:
        rem_blocks = []
        for i, f in enumerate(top, 1):
            sev = f.severity.lower()
            badge = f"b-{sev if sev in ('critical','high','medium','low','info') else 'info'}"
            rem_blocks.append(f"""
<div class="finding">
  <div class="finding-header">
    <div class="finding-title">Recommendation for Finding {i}: {html_escape(f.title)}</div>
    <div class="badge {badge}">{html_escape(sev.upper())}</div>
  </div>
  <ul>
    <li>Patch or update affected component / dependency.</li>
    <li>Validate and sanitize inputs; enforce allow-lists where possible.</li>
    <li>Add monitoring and regression tests to prevent re-introduction.</li>
  </ul>
</div>
""")
        remediation_html = "\n".join(rem_blocks)

    if not best_practices_ul:
        best_practices_ul = """
<ul>
  <li>Enforce HTTPS site-wide and use HSTS.</li>
  <li>Update server/framework dependencies regularly.</li>
  <li>Disable verbose errors in production.</li>
  <li>Set secure cookies (HttpOnly, Secure, SameSite=Strict/Lax).</li>
  <li>Centralize logging + alerting for suspicious activity.</li>
</ul>
""".strip()

    if not conclusion_html:
        conclusion_html = f"""
<p>
Applying the remediations in this report will significantly improve the security posture of
<strong>{html_escape(site_name)}</strong>. A re-test is recommended after fixes are deployed.
</p>
""".strip()

    # ------------------------
    # 8) Inject into template
    # ------------------------
    html_out = HTML_TEMPLATE
    replacements = {
        "{{SITE_NAME}}": html_escape(site_name),
        "{{DATE}}": "24 Nov 2025",
        "{{AUTHOR}}": "Team TRIPOD",
        "{{VERSION}}": "v1.0",
        "{{TARGET_URL}}": html_escape(target_url),

        "{{EXEC_SUMMARY_HTML}}": exec_summary_html,
        "{{STATUS_BADGE_CLASS}}": status_class,
        "{{STATUS_LABEL}}": status_label,
        "{{STATUS_LINE}}": status_line,
        "{{BUSINESS_IMPACT_UL}}": business_impact_ul,

        "{{SCORECARD_TABLE}}": scorecard_table,
        "{{SCOPE_HTML}}": scope_html,
        "{{METHODOLOGY_UL}}": methodology_ul,

        "{{TOP_FINDINGS_HTML}}": top_findings_html,
        "{{OTHER_FINDINGS_UL}}": other_findings_ul,

        "{{REMEDIATION_HTML}}": remediation_html,
        "{{BEST_PRACTICES_UL}}": best_practices_ul,
        "{{CONCLUSION_HTML}}": conclusion_html,
    }

    for k, v in replacements.items():
        html_out = html_out.replace(k, v)

    report_html_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"[REPORT] Wrote HTML â†’ {report_html_path}")
    return report_html_path



def export_pdf(report_html_path: str) -> Optional[str]:
    """
    Role: Final Artifact Generation.
    Task: Uses WeasyPrint to convert the HTML report into a professional PDF.
    Output: 'report.pdf' in the artifacts folder.
    """
    print("\n[EXPORT] Exporting PDF...")
    pdf_path = os.path.join(OUTPUT_DIR, "report.pdf")

    try:
        from weasyprint import HTML
        HTML(filename=report_html_path).write_pdf(pdf_path)
        print(f"[EXPORT] PDF saved â†’ {pdf_path}")
        return pdf_path
    except Exception:
        print("[EXPORT] WeasyPrint not installed. Keeping HTML only.")
        return None


def main():
    """
    The Manager. Controls the sequence of agents:
    1. Checks Environment.
    2. Decides between Live Scan or Offline Ingest.
    3. Runs Parser -> Enricher -> Reporter -> PDF Export.
    """
    ensure_environment()

    if RUN_LIVE_SCAN:
        assets = discovery_agent()
        scan_path = scanner_agent(assets)
    else:
        scan_path = offline_ingest_agent(RAW_DIR)

    findings = parser_agent(scan_path)
    findings = enrich_and_score_agent(findings)
    report_html = reporter_agent(findings)
    export_pdf(report_html)

    print("\nâœ… Pipeline complete. Artifacts in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


from IPython.display import IFrame

# Adjust the path to where your PDF was saved (e.g., "artifacts/report.pdf")
IFrame("artifacts/report.html", width=900, height=800)




