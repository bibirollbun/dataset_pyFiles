# Cell type: Code (bash mode or pip from Python)
!pip install --quiet openai python-dotenv matplotlib pandas



# Cell type: Code (python)
import json, os
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MEDS_FILE = os.path.join(DATA_DIR, "meds.json")

DEFAULT_MEDS = [
    {
        "id": "metoprolol_50",
        "name": "Metoprolol 50mg",
        "freq": "BID",
        "purpose": "Lower blood pressure",
        "long_description": "Metoprolol is a beta-blocker used to treat high blood pressure and heart-related issues."
    },
    {
        "id": "vitd_1000",
        "name": "Vitamin D 1000 IU",
        "freq": "QD",
        "purpose": "Bone health",
        "long_description": "Vitamin D helps the body absorb calcium and is important for bone health."
    },
    {
        "id": "amoxicillin_500",
        "name": "Amoxicillin 500mg",
        "freq": "TID",
        "purpose": "Antibiotic",
        "long_description": "Amoxicillin is an antibiotic used to treat bacterial infections."
    }
]

with open(MEDS_FILE, "w", encoding="utf-8") as f:
    json.dump(DEFAULT_MEDS, f, indent=2)

print("Wrote meds.json:", MEDS_FILE)



# Cell type: Code (python)
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("med_memory.sqlite3")

def ensure_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS doses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            med_id TEXT,
            scheduled_time TEXT,
            taken_time TEXT,
            status TEXT,
            note TEXT
        );
    ''')
    conn.commit()
    conn.close()

ensure_db()

class MemoryBank:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
    def log_scheduled(self, user_id, med_id, scheduled_time):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO doses (user_id, med_id, scheduled_time, status) VALUES (?,?,?,?)',
                  (user_id, med_id, scheduled_time, 'pending'))
        conn.commit()
        conn.close()
    def mark_taken(self, user_id, med_id, scheduled_time, taken_time=None):
        taken_time = taken_time or datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE doses SET taken_time=?, status="taken" WHERE user_id=? AND med_id=? AND scheduled_time=?',
                  (taken_time, user_id, med_id, scheduled_time))
        conn.commit()
        conn.close()
    def get_user_history(self, user_id, limit=50):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT med_id, scheduled_time, taken_time, status FROM doses WHERE user_id=? ORDER BY scheduled_time DESC LIMIT ?',
                  (user_id, limit))
        rows = c.fetchall()
        conn.close()
        return rows
    def adherence_rate(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM doses WHERE user_id=?', (user_id,))
        total = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM doses WHERE user_id=? AND status="taken"', (user_id,))
        taken = c.fetchone()[0]
        conn.close()
        if total == 0:
            return None
        return taken / total



# Cell type: Code (python)
import json, os
MEDS_FILE = os.path.join("data", "meds.json")

class MedDBTool:
    def __init__(self, meds_file=MEDS_FILE):
        with open(meds_file, 'r', encoding='utf-8') as f:
            meds = json.load(f)
        self._meds = {m['id']: m for m in meds}
    def get_med_by_id(self, med_id):
        return self._meds.get(med_id)
    def list_meds(self):
        return list(self._meds.values())
    def get_user_meds(self, user_profile):
        # user_profile expects dict with 'med_ids' list
        if isinstance(user_profile, dict) and user_profile.get('med_ids'):
            return [self._meds[mid] for mid in user_profile['med_ids'] if mid in self._meds]
        return list(self._meds.values())
        
med_db = MedDBTool()
print("Med ids:", list(med_db._meds.keys()))



# Cell type: Code (python)
from datetime import datetime, date, time as dtime

# SchedulerAgent
class SchedulerAgent:
    def __init__(self, med_db_tool, memory_bank, session=None):
        self.med_db = med_db_tool
        self.memory = memory_bank
        self.session = session or {}
    def _suggest_times(self, freq):
        if freq == 'QD': return [dtime(9,0)]
        if freq == 'BID': return [dtime(9,0), dtime(21,0)]
        if freq == 'TID': return [dtime(8,0), dtime(14,0), dtime(20,0)]
        return [dtime(9,0)]
    def create_daily_schedule(self, user_id, user_meds, on_date: date):
        schedule = []
        for med in user_meds:
            times = self._suggest_times(med.get('freq'))
            for t in times:
                scheduled_dt = datetime.combine(on_date, t)
                entry = {
                    'user_id': user_id,
                    'med_id': med.get('id'),
                    'med_name': med.get('name'),
                    'scheduled_time': scheduled_dt.isoformat(),
                    'status': 'pending'
                }
                schedule.append(entry)
                self.memory.log_scheduled(user_id, med.get('id'), scheduled_dt.isoformat())
        self.session['today_schedule'] = schedule
        return schedule

# ExplainerAgent (uses a very small deterministic fallback for Kaggle runtime)
class ExplainerAgent:
    def __init__(self, med_db_tool):
        self.med_db = med_db_tool
    def compact_context(self, long_description, max_chars=300):
        if not long_description: return ''
        return (long_description[:max_chars] + '...') if len(long_description)>max_chars else long_description
    def explain_med(self, med_id):
        med = self.med_db.get_med_by_id(med_id)
        if not med:
            return "Medication not found."
        compact = self.compact_context(med.get('long_description',''))
        # deterministic explanation template
        text = (f"- Purpose: {med.get('purpose')}.\n"
                f"- Common side effects: fatigue, nausea (examples).\n"
                f"- Safety tip: Consult your healthcare provider about interactions. Context: {compact}")
        return text

# NotifierAgent (mock)
class NotifierAgent:
    def __init__(self):
        pass
    def send_reminder(self, user_contact, med_name, scheduled_time):
        sent = {'status':'ok', 'to':user_contact, 'med':med_name, 'scheduled_time': scheduled_time}
        # For demo print to notebook output
        print("[MOCK NOTIFY] ->", sent)
        return sent

# create instances
memory = MemoryBank()
med_db = med_db  # from previous cell
scheduler = SchedulerAgent(med_db, memory, session={})
explainer = ExplainerAgent(med_db)
notifier = NotifierAgent()



# Cell type: Code (python)
from datetime import date

USER = {'id': 'demo_user_1', 'name': 'Demo User', 'med_ids': ['metoprolol_50', 'vitd_1000']}

# 1) Create today's schedule
user_meds = med_db.get_user_meds(USER)
sched = scheduler.create_daily_schedule(USER['id'], user_meds, date.today())
print("Created schedule entries:", len(sched))
for e in sched:
    print("-", e['med_name'], e['scheduled_time'], e['status'])

# 2) Ask explainer for the first med
first_med_id = sched[0]['med_id']
print("\nExplainer for", first_med_id)
print(explainer.explain_med(first_med_id))

# 3) Send mock reminder for first scheduled item
print("\nSending mock reminder...")
notifier.send_reminder('demo@example.com', sched[0]['med_name'], sched[0]['scheduled_time'])

# 4) Mark first dose as taken
memory.mark_taken(USER['id'], sched[0]['med_id'], sched[0]['scheduled_time'])
print("\nMarked taken for", sched[0]['med_name'])

# 5) Show adherence rate & history
rate = memory.adherence_rate(USER['id'])
print("\nAdherence rate:", (f"{rate*100:.1f}%" if rate is not None else "No doses yet"))

print("\nHistory (most recent):")
for row in memory.get_user_history(USER['id'], limit=20):
    print(row)



# Cell type: Code (python)
import pandas as pd
import matplotlib.pyplot as plt

rows = memory.get_user_history("demo_user_1", limit=200)
if not rows:
    print("No history to plot")
else:
    df = pd.DataFrame(rows, columns=['med_id', 'scheduled_time', 'taken_time', 'status'])
    df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    df['taken'] = df['status']=='taken'
    summary = df.groupby(df['scheduled_time'].dt.date)['taken'].mean().reset_index()
    plt.figure(figsize=(8,3))
    plt.plot(summary['scheduled_time'], summary['taken'])
    plt.ylim(0,1)
    plt.title('Daily adherence (fraction taken)')
    plt.xlabel('Date'); plt.ylabel('Adherence')
    plt.grid(True)
    plt.show()


