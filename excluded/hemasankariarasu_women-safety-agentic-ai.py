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
# agent.py
# agent.py
from google.adk.agents.llm_agent import Agent
from tools import (
    emergency_alert,
    store_evidence,
    generate_fir,
    fetch_legal_guide,
    record_mood,
)
import os
from dotenv import load_dotenv

load_dotenv()

root_agent = Agent(
    model="gemini-3-pro-preview",  # or any configured model in your env
    name="women_safety_agent",
    description="Helps women with safety steps, legal rights, emergency alerts, evidence storage and emotional support.",
    instruction=(
        "You are a safety-first assistant for women in distress. Use the provided tools "
        "for sending alerts, storing evidence, generating FIR drafts, providing legal guidance, "
        "and tracking mood. Be empathetic, preserve user privacy, and never encourage illegal evasion."
    ),
    tools=[
        emergency_alert,
        store_evidence,
        generate_fir,
        fetch_legal_guide,
        record_mood,
    ],
)



# tools.py
import os
from typing import Dict
from twilio.rest import Client
from evidence import EvidenceLocker
from fir import FIRGenerator
from legal import LegalGuide
from mood import MoodTracker

# Load env
TW_SID = os.getenv("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TW_FROM = os.getenv("TWILIO_FROM_NUMBER")
TRUSTED_CONTACTS = os.getenv("TRUSTED_CONTACTS", "")

# Twilio client
tw_client = Client(TW_SID, TW_TOKEN) if TW_SID else None

def emergency_alert(payload: Dict) -> Dict:
    """
    Tool signature used by ADK should accept structured args and return dict.
    payload example:
      {
        "message": "I'm in danger. Please help!",
        "location": "12.97,80.23",
        "user_name": "Meena"
      }
    """
    message = payload.get("message", "Emergency alert")
    location = payload.get("location")
    contacts = [c.strip() for c in TRUSTED_CONTACTS.split(",") if c.strip()]

    sent = []
    errors = []
    if not tw_client:
        return {"status": "error", "error": "Twilio not configured"}

    for contact in contacts:
        try:
            body = f"EMERGENCY: {message}\nName: {payload.get('user_name','Unknown')}"
            if location:
                body += f"\nLocation: https://maps.google.com/?q={location}"
            sms = tw_client.messages.create(
                body=body,
                from_=TW_FROM,
                to=contact
            )
            sent.append({"to": contact, "sid": sms.sid})
        except Exception as e:
            errors.append({"to": contact, "error": str(e)})

    return {"status": "success" if sent else "partial_failure", "sent": sent, "errors": errors}


def store_evidence(payload: Dict) -> Dict:
    """
    payload:
      {
        "filename": "audio_20251125.mp3",
        "bytes": <base64 or bytes>,
        "content_type": "audio/mpeg",
        "user_id": "user123"
      }
    """
    locker = EvidenceLocker()
    try:
        res = locker.save_and_encrypt(payload["filename"], payload["bytes"], payload.get("user_id"))
        return {"status": "success", "object_name": res}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def generate_fir(payload: Dict) -> Dict:
    """
    payload:
      {
        "user_details": {...},
        "incident": {...}
      }
    """
    generator = FIRGenerator()
    try:
        doc = generator.make_fir(payload["user_details"], payload["incident"])
        return {"status": "success", "fir_text": doc}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def fetch_legal_guide(payload: Dict) -> Dict:
    """
    payload:
       {"locale": "ta-IN", "topic": "domestic_violence"}
    """
    lg = LegalGuide()
    try:
        guide = lg.get_guide(payload.get("locale", "en-IN"), payload.get("topic"))
        return {"status": "success", "guide": guide}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def record_mood(payload: Dict) -> Dict:
    """
    payload:
      {"user_id": "u1", "mood": "anxious", "notes": "..."}
    """
    mt = MoodTracker()
    try:
        mt.record(payload["user_id"], payload["mood"], payload.get("notes"))
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}



# evidence.py
import os
from google.cloud import storage
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import uuid

GCS_BUCKET = os.getenv("GCS_BUCKET")
ENC_KEY = os.getenv("EVIDENCE_ENC_KEY")  # 32 bytes (256-bit) base64 or hex

def _get_key_bytes():
    # try base64 decode first
    try:
        return base64.b64decode(ENC_KEY)
    except Exception:
        # fallback: interpret as hex
        return bytes.fromhex(ENC_KEY)

class EvidenceLocker:
    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(GCS_BUCKET)

    def save_and_encrypt(self, filename: str, file_bytes: bytes, user_id: str = "anon"):
        key = _get_key_bytes()
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, file_bytes, None)
        # store nonce + ciphertext
        payload = nonce + ct
        object_name = f"{user_id}/{uuid.uuid4().hex}_{filename}.enc"
        blob = self.bucket.blob(object_name)
        blob.upload_from_string(payload)
        # Return object location (no decryption key returned here)
        return object_name

    def download_and_decrypt(self, object_name: str):
        blob = self.bucket.blob(object_name)
        payload = blob.download_as_bytes()
        nonce = payload[:12]
        ct = payload[12:]
        key = _get_key_bytes()
        aesgcm = AESGCM(key)
        plain = aesgcm.decrypt(nonce, ct, None)
        return plain



# legal.py
import json
import os

# Keep a curated JSON of legal guides per locale & topic.
# For prototype we load local JSON; in prod, store in Firestore or BigQuery.
DATA_FILE = os.path.join(os.path.dirname(__file__), "legal_guides.json")

class LegalGuide:
    def __init__(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                self.guides = json.load(f)
        else:
            # minimal fallback
            self.guides = {
                "en-IN": {
                    "domestic_violence": "Under Section ... contact the local women police station and file under Domestic Violence Act..."
                }
            }

    def get_guide(self, locale="en-IN", topic=None):
        locale_data = self.guides.get(locale, {})
        if topic:
            return locale_data.get(topic, "No guide found for this topic in this locale.")
        return locale_data



# mood.py
import os
from google.cloud import firestore
import datetime

PROJECT = os.getenv("FIRESTORE_PROJECT")

class MoodTracker:
    def __init__(self):
        self.client = firestore.Client(project=PROJECT)
        self.col = self.client.collection("mood_entries")

    def record(self, user_id: str, mood: str, notes: str = None):
        doc = {
            "user_id": user_id,
            "mood": mood,
            "notes": notes or "",
            "ts": datetime.datetime.utcnow()
        }
        self.col.add(doc)

    def summary(self, user_id: str, days: int = 30):
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        docs = self.col.where("user_id", "==", user_id).where("ts", ">", cutoff).stream()
        result = []
        for d in docs:
            data = d.to_dict()
            result.append(data)
        return result



adk create women_safety_agent
# copy the files above into the created directory (agent.py, tools.py, ...)



echo 'GOOGLE_API_KEY="GOOGLE_API_KEY"' > .env
# and add other required env vars



adk run women_safety_agent





