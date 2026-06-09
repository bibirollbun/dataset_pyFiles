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
import re
import json
import base64
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime

# DB libs (example)
import psycopg2
from psycopg2.extras import execute_values

# PDF text extraction libraries
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None  # Fallback - we will handle below

# Optional: OCR for scanned PDFs (pytesseract + pillow)
try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:
    convert_from_path = None
    pytesseract = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adk_medical_ingest")

# Path to your generated flowchart image uploaded earlier
DIAGRAM_URL = "/mnt/data/A_flowchart_in_the_image_illustrates_an_AI-driven_.png"

# -------------------------
# Domain dataclasses
# -------------------------
@dataclass
class ExtractedRecord:
    member_id: str
    patient_name: Optional[str]
    dob: Optional[str]
    encounter_date: Optional[str]
    diagnosis_codes: List[str]
    procedure_codes: List[str]
    lab_results: Dict[str, Any]
    raw_text: str
    source_pdf: str
    extracted_at: str = datetime.utcnow().isoformat()

# -------------------------
# PDF Extraction Component
# -------------------------
class PDFExtractor:
    """Extract text from a PDF. Uses pdfminer when available; falls back to OCR if needed."""
    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled and (convert_from_path is not None) and (pytesseract is not None)

    def extract(self, pdf_path: str) -> str:
        """Return the full textual content of the PDF."""
        logger.info("Extracting text from PDF: %s", pdf_path)

        text = ""
        # Try pdfminer first (works for text PDFs)
        if pdf_extract_text is not None:
            try:
                text = pdf_extract_text(pdf_path)
                # quick check if extraction succeeded
                if text and len(text.strip()) > 50:
                    logger.debug("pdfminer returned text length: %d", len(text))
                    return text
                logger.debug("pdfminer returned short text; will try OCR if enabled.")
            except Exception as e:
                logger.warning("pdfminer extraction failed: %s", str(e))
                text = ""

        # If pdfminer didn't give meaningful text and OCR is enabled, fallback to OCR
        if self.ocr_enabled:
            logger.info("Attempting OCR extraction via pdf2image + pytesseract")
            try:
                pages = convert_from_path(pdf_path, dpi=200)
                ocr_text_parts = []
                for i, page in enumerate(pages):
                    part = pytesseract.image_to_string(page)
                    ocr_text_parts.append(part)
                text = "\n".join(ocr_text_parts)
                logger.debug("OCR text length: %d", len(text))
            except Exception as e:
                logger.error("OCR extraction failed: %s", str(e))
                # final fallback: raise
                raise RuntimeError("PDF text extraction failed: " + str(e))
        else:
            raise RuntimeError("PDF text extraction returned empty text and OCR is disabled or not available.")

        return text

# -------------------------
# AI Extraction Component (ADK / LLM)
# -------------------------
class AIExtractor:
    """
    Use an LLM (via ADK or direct client) to parse the raw text into a structured record.
    This is a stubbed method that demonstrates the expected input/output contract.
    Replace the `call_llm_extract` method with your ADK tool integration or LLM client call.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 1024):
        self.model_name = model_name
        self.max_tokens = max_tokens
        # Example: hold API keys or ADK client
        self.api_key = os.getenv("LLM_API_KEY")

    def call_llm_extract(self, raw_text: str, prompt_instructions: str) -> Dict[str, Any]:
        """
        Replace with ADK tool call or direct LLM client invocation.
        For now, this is a dummy extraction that uses regex heuristics for demonstration.
        """
        logger.info("Calling LLM/extraction tool (stub).")
        # ----
        # In production: build a structured prompt + pass `raw_text` as file or tool input to ADK.
        # Example ADK pseudo-call:
        # response = adk.client.tools.run(
        #     tool_name="pdf_field_extractor",
        #     inputs={"file": file_url, "instructions": prompt_instructions}
        # )
        # parsed_json = response["structured_output"]
        # ----

        # Simple heuristic extraction (demonstration only)
        parsed = {
            "member_id": self._extract_member_id(raw_text),
            "patient_name": self._extract_patient_name(raw_text),
            "dob": self._extract_dob(raw_text),
            "encounter_date": self._extract_encounter_date(raw_text),
            "diagnosis_codes": self._extract_icd_codes(raw_text),
            "procedure_codes": self._extract_cpt_codes(raw_text),
            "lab_results": self._extract_lab_results(raw_text),
        }
        return parsed

    # ---------- Simple heuristic extractors (for demo only) ----------
    def _extract_member_id(self, r: str) -> str:
        m = re.search(r"Member(?: ID)?:\s*([A-Za-z0-9\-]+)", r, re.I)
        return m.group(1) if m else "UNKNOWN_MEMBER"

    def _extract_patient_name(self, r: str) -> Optional[str]:
        m = re.search(r"Patient Name:\s*([A-Z ,.'-]+)", r, re.I)
        return m.group(1).strip() if m else None

    def _extract_dob(self, r: str) -> Optional[str]:
        m = re.search(r"(?:DOB|Date of Birth)[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})", r, re.I)
        return m.group(1) if m else None

    def _extract_encounter_date(self, r: str) -> Optional[str]:
        m = re.search(r"(?:Encounter Date|Date of Service)[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})", r, re.I)
        return m.group(1) if m else None

    def _extract_icd_codes(self, r: str) -> List[str]:
        return re.findall(r"[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]*", r)[:5]  # naive sample

    def _extract_cpt_codes(self, r: str) -> List[str]:
        return re.findall(r"\b\d{5}\b", r)[:5]

    def _extract_lab_results(self, r: str) -> Dict[str, Any]:
        labs = {}
        for match in re.finditer(r"(A1C|HbA1c|Hemoglobin A1c)[:\s]*([0-9.]+)%", r, re.I):
            labs["A1C"] = match.group(2)
        return labs

    def extract_structured(self, raw_text: str, source_pdf: str) -> ExtractedRecord:
        prompt_instructions = (
            "Extract the following fields into JSON: member_id, patient_name, dob (MM/DD/YYYY), "
            "encounter_date, diagnosis_codes (list), procedure_codes (list), lab_results (map)."
            "Return only JSON."
        )
        parsed = self.call_llm_extract(raw_text, prompt_instructions)
        record = ExtractedRecord(
            member_id=parsed.get("member_id", "UNKNOWN"),
            patient_name=parsed.get("patient_name"),
            dob=parsed.get("dob"),
            encounter_date=parsed.get("encounter_date"),
            diagnosis_codes=parsed.get("diagnosis_codes", []),
            procedure_codes=parsed.get("procedure_codes", []),
            lab_results=parsed.get("lab_results", {}),
            raw_text=raw_text[:2000],  # store truncated raw_text
            source_pdf=source_pdf,
        )
        return record

# -------------------------
# Database Loader
# -------------------------
class DatabaseLoader:
    """Simple Postgres loader. Replace with SQLAlchemy or your DB client of choice."""
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _get_conn(self):
        return psycopg2.connect(self.dsn)

    def create_table_if_not_exists(self):
        sql = """
        CREATE TABLE IF NOT EXISTS extracted_records (
            id SERIAL PRIMARY KEY,
            member_id TEXT,
            patient_name TEXT,
            dob DATE,
            encounter_date DATE,
            diagnosis_codes JSONB,
            procedure_codes JSONB,
            lab_results JSONB,
            raw_text TEXT,
            source_pdf TEXT,
            extracted_at TIMESTAMP
        );
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def insert_record(self, rec: ExtractedRecord):
        sql = """
        INSERT INTO extracted_records
        (member_id, patient_name, dob, encounter_date, diagnosis_codes, procedure_codes, lab_results, raw_text, source_pdf, extracted_at)
        VALUES %s
        """
        tuple_values = [(
            rec.member_id,
            rec.patient_name,
            self._parse_date(rec.dob),
            self._parse_date(rec.encounter_date),
            json.dumps(rec.diagnosis_codes),
            json.dumps(rec.procedure_codes),
            json.dumps(rec.lab_results),
            rec.raw_text,
            rec.source_pdf,
            rec.extracted_at,
        )]
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, tuple_values)
            conn.commit()

    def _parse_date(self, s: Optional[str]):
        if not s:
            return None
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.date()
            except Exception:
                continue
        return None

# -------------------------
# HEDIS Engine Client (stub)
# -------------------------
class HEDISEngineClient:
    """
    Stub for HEDIS engine integration.
    Replace with the API calls to your measurement engine.
    """
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key

    def evaluate_record(self, rec: ExtractedRecord) -> Dict[str, Any]:
        """
        Send structured record to HEDIS engine and return measures + open gaps.
        Example return:
            {
                "member_id": "...",
                "open_gaps": [
                    {"measure": "HbA1c Poor Control", "reason": "No A1C in last 12 months"},
                    ...
                ]
            }
        """
        # In production, call the real service:
        # resp = requests.post(self.endpoint, json=payload, headers={"Authorization": f"Bearer {self.api_key}"})
        # return resp.json()

        # Demo logic (very naive)
        open_gaps = []
        # Example HEDIS rule: diabetic patients should have A1C in last 12 months
        if "A1C" not in rec.lab_results or float(rec.lab_results.get("A1C", 999)) > 9.0:
            open_gaps.append({
                "measure": "Diabetes - A1C Monitoring",
                "reason": "A1C missing or high"
            })
        return {"member_id": rec.member_id, "open_gaps": open_gaps}

# -------------------------
# Notifier - email / sms stub
# -------------------------
class Notifier:
    """Simple notifier with email and SMS methods (stubs)."""
    def __init__(self, smtp_config: Optional[Dict] = None, sms_config: Optional[Dict] = None):
        self.smtp = smtp_config
        self.sms = sms_config

    def notify_provider(self, provider_contact: Dict[str, str], message: str):
        # Replace with real email or portal integration
        logger.info("NOTIFY PROVIDER to %s: %s", provider_contact, message)

    def notify_member(self, member_contact: Dict[str, str], message: str):
        # Replace with Twilio, SNS, or SMS/email gateway
        logger.info("NOTIFY MEMBER to %s: %s", member_contact, message)

# -------------------------
# Orchestrator
# -------------------------
class AgentOrchestrator:
    def __init__(self, pdf_extractor: PDFExtractor, ai_extractor: AIExtractor,
                 db_loader: DatabaseLoader, hedis_client: HEDISEngineClient,
                 notifier: Notifier):
        self.pdf_extractor = pdf_extractor
        self.ai_extractor = ai_extractor
        self.db_loader = db_loader
        self.hedis_client = hedis_client
        self.notifier = notifier

    def process_pdf(self, pdf_path: str, provider_contact: Dict[str,str], member_contact: Dict[str,str]):
        logger.info("Starting pipeline for: %s", pdf_path)
        raw_text = self.pdf_extractor.extract(pdf_path)

        # AI extraction
        rec = self.ai_extractor.extract_structured(raw_text, source_pdf=pdf_path)
        logger.info("Extracted record for member_id=%s", rec.member_id)

        # Persist to DB
        self.db_loader.create_table_if_not_exists()
        self.db_loader.insert_record(rec)
        logger.info("Inserted record into DB for member %s", rec.member_id)

        # HEDIS evaluation
        hedis_result = self.hedis_client.evaluate_record(rec)
        logger.info("HEDIS result: %s", hedis_result)

        # If open gaps, notify
        if hedis_result.get("open_gaps"):
            for gap in hedis_result["open_gaps"]:
                # build a message
                msg = f"HEDIS Gap: {gap['measure']} - {gap['reason']} (Member: {rec.member_id})"
                self.notifier.notify_provider(provider_contact, msg)
                self.notifier.notify_member(member_contact, msg)
        else:
            logger.info("No open care gaps for member %s", rec.member_id)

# -------------------------
# Example usage (main)
# -------------------------
def main():
    # Example configuration - replace with secrets/real endpoints or env variables
    PG_DSN = os.getenv("PG_DSN", "dbname=hedis user=hedis_user password=hedis_pass host=localhost port=5432")
    HEDIS_ENDPOINT = os.getenv("HEDIS_ENDPOINT", "https://hedis.example/api/evaluate")
    LLM_API_KEY = os.getenv("LLM_API_KEY", None)

    # Local uploaded flowchart reference (provided by user)
    logger.info("Using flowchart asset at: %s", DIAGRAM_URL)

    pdf_extractor = PDFExtractor(ocr_enabled=True)
    ai_extractor = AIExtractor(model_name="adk-llm-model")
    db_loader = DatabaseLoader(dsn=PG_DSN)
    hedis_client = HEDISEngineClient(endpoint=HEDIS_ENDPOINT, api_key=os.getenv("HEDIS_API_KEY"))
    notifier = Notifier(smtp_config=None, sms_config=None)

    orchestrator = AgentOrchestrator(pdf_extractor, ai_extractor, db_loader, hedis_client, notifier)

    # Example provider/member contact dicts
    provider_contact = {"email": "provider@example.com", "name": "Dr. Smith"}
    member_contact = {"sms": "+15551234567", "email": "patient@example.com"}

    # Example PDF path - replace with the real incoming PDF path or URL you receive
    sample_pdf = os.getenv("SAMPLE_PDF_PATH", "example_medical_record.pdf")

    if not os.path.exists(sample_pdf):
        logger.warning("Sample PDF doesn't exist at %s — update SAMPLE_PDF_PATH env or provide the file.", sample_pdf)
    else:
        orchestrator.process_pdf(sample_pdf, provider_contact, member_contact)

if __name__ == "__main__":
    main()


