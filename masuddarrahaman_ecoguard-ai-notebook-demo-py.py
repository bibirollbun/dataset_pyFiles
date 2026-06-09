# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ============================================================
# CELL 1 — FULL SYSTEM (ERROR-PROOF + UNICODE-SAFE)
# ============================================================

!pip install fpdf --quiet

import io, time
import pandas as pd
from fpdf import FPDF
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
ADK_ENABLED = False  # Gemini disabled by default

print("Google API Key Loaded. ADK Mode:", ADK_ENABLED)


# ------------------------------------------------------------
# SANITIZER — Removes ALL unicode so PDF never breaks
# ------------------------------------------------------------
def sanitize(text):
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("–", "-")   # en dash
            .replace("—", "-")   # em dash
            .replace("−", "-")   # minus
            .replace("°", "deg") # degree
            .replace("₂", "2")   # subscript 2
            .encode("latin-1", "replace")
            .decode("latin-1")
    )


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def read_csv_bytes(b):
    try: return pd.read_csv(io.BytesIO(b))
    except: return pd.DataFrame()

EMISSION_FACTORS = {"electricity": 0.85, "waste": 1.2}

def compute_emissions(df):
    e = df.get("energy_kwh", pd.Series([0])).sum() * EMISSION_FACTORS["electricity"]
    w = df.get("waste_kg", pd.Series([0])).sum() * EMISSION_FACTORS["waste"]
    return {"energy": e, "waste": w, "total": e + w, "rows": len(df)}


# ------------------------------------------------------------
# MEMORY + LOGGING
# ------------------------------------------------------------
class Session:
    def __init__(self): self.data = {}
    def save(self, k, v): self.data[k] = v
    def load(self, k, default=None): return self.data.get(k, default)

class MemoryBank:
    def __init__(self): self.store = {}
    def add(self, key, value): self.store.setdefault(key, []).append({"ts": time.time(), "value": value})

class Observability:
    def __init__(self): self.logs = []
    def log(self, agent, msg): self.logs.append({"agent": agent, "msg": msg, "ts": time.time()})

session = Session()
memory = MemoryBank()
obs = Observability()


# ------------------------------------------------------------
# AGENTS
# ------------------------------------------------------------
class Agent:
    def __init__(self, name): self.name = name
    def run(self, payload): raise NotImplementedError


# Intake Agent
class IntakeAgent(Agent):
    def __init__(self): super().__init__("intake")
    def run(self, payload):
        obs.log(self.name, "Ingesting CSV")
        df = read_csv_bytes(payload["bytes"])
        df = df.rename(columns=str.lower)
        df["energy_kwh"] = pd.to_numeric(df.get("energy_kwh", 0), errors='coerce').fillna(0)
        df["waste_kg"] = pd.to_numeric(df.get("waste_kg", 0), errors='coerce').fillna(0)
        session.save("dataset", df)
        return df


# Carbon Agent
class CarbonAgent(Agent):
    def __init__(self): super().__init__("carbon")
    def run(self, df):
        obs.log(self.name, "Computing emissions")
        em = compute_emissions(df)
        session.save("emissions", em)
        memory.add("emissions", em)
        return em


# Recommendations Agent
class RecAgent(Agent):
    def __init__(self): super().__init__("rec")
    def run(self, em):

        obs.log(self.name, "Creating recommendations")

        # NO Unicode allowed here
        return sanitize(
            "1) Replace old bulbs with LED lighting (20% savings).\n"
            "2) Maintain appliances regularly (5-10% savings).\n"
            "3) Reduce idle machine usage (5% savings)."
        )


# Reporting Agent
class ReportAgent(Agent):
    def __init__(self): super().__init__("report")
    def run(self, payload):
        em = payload["emissions"]
        rec = sanitize(payload["recommend"])

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(0, 10, sanitize("EcoGuard-AI Sustainability Report"), ln=True)

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, sanitize(f"Energy CO2: {em['energy']}"), ln=True)
        pdf.cell(0, 8, sanitize(f"Waste CO2: {em['waste']}"), ln=True)
        pdf.cell(0, 8, sanitize(f"Total CO2: {em['total']}"), ln=True)

        pdf.ln(5)
        pdf.multi_cell(0, 8, sanitize("Recommendations:\n" + rec))

        out = "/tmp/ecoguard_report.pdf"
        pdf.output(out)
        return out


# Instantiate agents
A1, A2, A3, A4 = IntakeAgent(), CarbonAgent(), RecAgent(), ReportAgent()


# ------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------
def run_pipeline(csv_bytes):
    df = A1.run({"bytes": csv_bytes})
    em = A2.run(df)
    rec = A3.run(em)
    rpt = A4.run({"emissions": em, "recommend": rec})
    return df, em, rec, rpt


# Run test
TEST_CSV = b"date,energy_kwh,waste_kg\n2025-01-01,120,10\n2025-02-01,140,8"
df, em, rec, rpt = run_pipeline(TEST_CSV)

print("=== PIPELINE OK ===")
print(em)
print(rec)
print("PDF saved at:", rpt)



print("=== DATAFRAME ===")
print(df)

print("\n=== EMISSIONS ===")
print(em)



print("=== LOGS ===")
for l in obs.logs:
    print(l)



print("=== MEMORY ===")
print(memory.store)



print("PDF saved at:", rpt)



import shutil

shutil.copy("/tmp/ecoguard_report.pdf", "/kaggle/working/ecoguard_report.pdf")

print("PDF exported to Kaggle working folder:")
print("/kaggle/working/ecoguard_report.pdf")





