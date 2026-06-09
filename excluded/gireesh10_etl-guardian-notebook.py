# ============================================================
# ETL GUARDIAN - MULTI-AGENT ETL DEBUGGING CAPSTONE PROJECT
# ENTERPRISE TRACK
# ============================================================

import re, uuid, time, sqlite3, json, random, asyncio
from rich import print



conn = sqlite3.connect("memory.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    source TEXT,
    logs TEXT,
    created REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS fixes (
    id TEXT PRIMARY KEY,
    incident_id TEXT,
    patch TEXT,
    validated INTEGER,
    created REAL
)
""")

conn.commit()

def save_incident(id, source, logs):
    conn.execute("INSERT OR REPLACE INTO incidents VALUES (?,?,?,?)",
                 (id, source, logs, time.time()))
    conn.commit()

def get_incident(id):
    r = conn.execute("SELECT id,source,logs,created FROM incidents WHERE id=?",(id,)).fetchone()
    if not r: return None
    return {"id": r[0], "source": r[1], "logs": r[2], "created": r[3]}

def store_fix(fid, iid, patch, validated):
    conn.execute("INSERT INTO fixes VALUES (?,?,?,?,?)",
                 (fid, iid, patch, validated, time.time()))
    conn.commit()



class PlannerAgent:
    def __init__(self):
        self.log_analyzer = LogAnalyzerAgent()
        self.rca = RCAAgent()
        self.fix_gen = FixGeneratorAgent()
        self.validator = ValidatorAgent()

    async def handle_incident(self, incident_id):
        inc = get_incident(incident_id)

        la_task = asyncio.create_task(self.log_analyzer.analyze(inc))
        rca_task = asyncio.create_task(self.rca.analyze(inc))

        la_res = await la_task
        rca_res = await rca_task

        fix = self.fix_gen.generate(inc, la_res, rca_res)
        val = self.validator.validate(fix, inc)

        store_fix(fix["id"], inc["id"], fix["patch"], 1 if val["passed"] else 0)

        return {
            "incident": inc["id"],
            "fix_id": fix["id"],
            "patch": fix["patch"],
            "validation": val
        }



class LogAnalyzerAgent:
    async def analyze(self, inc):
        logs = inc["logs"]
        errors = re.findall(r"ERROR.*", logs)
        signature = str(uuid.uuid5(uuid.NAMESPACE_URL, errors[0] if errors else "noerror"))

        print("[yellow]Log Analyzer:[/yellow]", errors)
        return {"signature": signature, "errors": errors}



class RCAAgent:
    async def analyze(self, inc):
        logs = inc["logs"]

        if "Schema mismatch" in logs:
            cause = "Column type mismatch"
        elif "Partition not found" in logs:
            cause = "Missing partition"
        elif "Connection refused" in logs:
            cause = "Service unavailable"
        elif "CSV parse error" in logs:
            cause = "Malformed CSV row"
        else:
            cause = "Unknown"

        print("[cyan]RCA Agent:[/cyan] Cause:", cause)
        return {"root_cause": cause}



class FixGeneratorAgent:
    def generate(self, inc, la, rca):
        cause = rca["root_cause"]
        fid = str(uuid.uuid4())

        if cause == "Column type mismatch":
            patch = "ALTER TABLE users MODIFY age INT;"
        elif cause == "Missing partition":
            patch = "RUN FIX_PARTITION('/data/2025/11/18');"
        elif cause == "Service unavailable":
            patch = "RESTART metadata-store;"
        elif cause == "Malformed CSV row":
            patch = "DROP malformed row 123;"
        else:
            patch = "# No auto-fix available"

        print("[green]Fix Generator:[/green]", patch)
        return {"id": fid, "patch": patch}



class ValidatorAgent:
    def validate(self, fix, inc):
        passed = not fix["patch"].startswith("#")
        print("[magenta]Validator:[/magenta] Passed:", passed)
        return {"passed": passed}



planner = PlannerAgent()

async def run_demo(n=5):
    results = []
    for _ in range(n):
        iid = create_incident()
        print(f"\n[bold blue]Processing Incident:[/bold blue] {iid}")
        res = await planner.handle_incident(iid)
        results.append(res)
    return results

import nest_asyncio
nest_asyncio.apply()

results = await run_demo(5)
results



planner = PlannerAgent()

async def run_demo(n=5):
    results = []
    for _ in range(n):
        iid = create_incident()
        print(f"\n[bold blue]Processing Incident:[/bold blue] {iid}")
        res = await planner.handle_incident(iid)
        results.append(res)
    return results

results = asyncio.run(run_demo(5))
results



cur = conn.cursor()

total = cur.execute("SELECT COUNT(*) FROM fixes").fetchone()[0]
validated = cur.execute("SELECT COUNT(*) FROM fixes WHERE validated=1").fetchone()[0]

precision = validated / total if total else 0

print(f"[green]Total Fixes:[/green] {total}")
print(f"[green]Validated Fixes:[/green] {validated}")
print(f"[green]Fix Precision:[/green] {precision:.2f}")



print("\n[underline]Incidents Stored:[/underline]")
for row in conn.execute("SELECT * FROM incidents"):
    print(row)

print("\n[underline]Fixes Stored:[/underline]")
for row in conn.execute("SELECT * FROM fixes"):
    print(row)



print("""
===========================================
ETL GUARDIAN — EXECUTION COMPLETE
You can now publish this notebook.
===========================================
""")


