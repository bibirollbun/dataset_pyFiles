import os, json, time, shutil, asyncio, re, uuid
from datetime import datetime, timedelta
from pprint import pprint

print("Environment ready.")



# Create src directory structure
import os

os.makedirs("src/tools", exist_ok=True)
os.makedirs("sample_data/messy_workspace", exist_ok=True)
os.makedirs("sample_logs", exist_ok=True)

# ---- a2a_protocol.py ----
open("src/a2a_protocol.py","w").write("""
import time, uuid

def make_message(sender, receiver, typ, payload):
    return {
        "id": str(uuid.uuid4()),
        "sender": sender,
        "receiver": receiver,
        "type": typ,
        "payload": payload,
        "ts": time.time()
    }
""")

# ---- memory.py ----
open("src/memory.py","w").write("""
import json, os, time

class MemoryBank:
    def __init__(self, path='memory.json'):
        self.path = path
        self._data = {'users':{}, 'history':[]}
        if os.path.exists(path):
            self._data = json.load(open(path))

    def save(self):
        json.dump(self._data, open(self.path,'w'), indent=2)

    def add_history(self, entry):
        entry['ts'] = time.time()
        self._data['history'].append(entry)
        self.save()
""")

# ---- FileTool ----
open("src/tools/file_tool.py","w").write("""
import os, shutil

class FileTool:
    def ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def move(self, src, dst):
        self.ensure_dir(os.path.dirname(dst))
        shutil.move(src, dst)
        return {'status':'moved','src':src,'dst':dst}
""")

# ---- EmailTool ----
open("src/tools/email_tool.py","w").write("""
import json, os

class EmailTool:
    def __init__(self, inbox='sample_data/inbox_samples.json'):
        self.inbox = inbox

    def load_samples(self):
        if not os.path.exists(self.inbox):
            return []
        return json.load(open(self.inbox))
""")

# ---- CalendarTool ----
open("src/tools/calendar_tool.py","w").write("""
class CalendarTool:
    def __init__(self):
        self.events = []

    def add_event(self, title, when, metadata=None):
        ev = {
            'id': len(self.events)+1,
            'title': title,
            'when': when,
            'metadata': metadata or {}
        }
        self.events.append(ev)
        return ev
""")

# ---- agents.py ----
open("src/agents.py","w").write("""
import os, asyncio, re
from a2a_protocol import make_message
from datetime import datetime, timedelta

class ClassifierAgent:
    def __init__(self, file_tool, coordinator):
        self.file_tool = file_tool
        self.coordinator = coordinator

    async def classify_and_emit(self, file_path):
        ext = file_path.split('.')[-1].lower()
        if ext in ['pdf','txt','doc','docx']:
            category = 'documents'
        elif ext in ['png','jpg','jpeg']:
            category = 'images'
        elif ext in ['py','ipynb','zip']:
            category = 'code'
        else:
            category = 'misc'
        msg = make_message('classifier','coordinator','file_classified',
                           {'path':file_path,'category':category})
        await self.coordinator.handle_message(msg)

class TaskExtractorAgent:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def extract(self, text):
        tasks=[]
        for line in text.split('\\n'):
            if any(k in line.lower() for k in ['due','by','submit','todo']):
                tasks.append(line.strip())
        return tasks

    async def process_email(self, email):
        t = self.extract(email.get('body',''))
        msg = make_message('task_extractor','coordinator','task_extracted',
                           {'tasks':t,'email_id':email.get('id')})
        await self.coordinator.handle_message(msg)

class CoordinatorAgent:
    def __init__(self, memory, filetool, calendar):
        self.memory = memory
        self.filetool = filetool
        self.calendar = calendar

    async def handle_message(self, msg):
        typ = msg['type']
        if typ == 'file_classified':
            await self.on_file_classified(msg)
        elif typ == 'task_extracted':
            await self.on_task_extracted(msg)

    async def on_file_classified(self, msg):
        p = msg['payload']
        src = p['path']
        dst = f"organized/{p['category']}/{os.path.basename(src)}"
        self.filetool.move(src, dst)
        self.memory.add_history({'event':'file_move','src':src,'dst':dst})
        print(f"[Moved] {src} -> {dst}")

    async def on_task_extracted(self, msg):
        for t in msg['payload']['tasks']:
            ev = self.calendar.add_event(t, '2025-12-01')
            print('[Task Added]', ev)
""")

print("src directory created successfully.")



import sys
sys.path.append("src")

from agents import ClassifierAgent, TaskExtractorAgent, CoordinatorAgent
from tools.file_tool import FileTool
from tools.email_tool import EmailTool
from tools.calendar_tool import CalendarTool
from memory import MemoryBank

print("Imports successful.")



import os, json

# Create sample folders
os.makedirs("sample_data/messy_workspace", exist_ok=True)
os.makedirs("sample_logs", exist_ok=True)

# --- Create fake messy files ---
sample_files = {
    "homework1.docx": "Student assignment content",
    "receipt_jan.pdf": "Invoice receipt content",
    "screenshot1.png": "binary image data",
    "script_example.py": "print('hello world')",
    "notes.txt": "TODO: submit project by 2025-12-01"
}

for fname, content in sample_files.items():
    with open(f"sample_data/messy_workspace/{fname}", "w") as f:
        f.write(content)

# --- Create fake inbox emails ---
emails = [
    {
        "id": "email1",
        "from": "prof@example.com",
        "subject": "Final Project",
        "body": "Hi, please submit your final project by 2025-12-01."
    },
    {
        "id": "email2",
        "from": "boss@example.com",
        "subject": "Urgent Task",
        "body": "Complete the financial report by 2025-11-30. This is important."
    },
    {
        "id": "email3",
        "from": "friend@example.com",
        "subject": "Weekend Plan",
        "body": "No urgent tasks here."
    }
]

with open("sample_data/inbox_samples.json", "w") as f:
    json.dump(emails, f, indent=2)

print("Sample workspace & inbox created successfully.")



import os, asyncio
from memory import MemoryBank
from tools.file_tool import FileTool
from tools.calendar_tool import CalendarTool
from agents import ClassifierAgent, CoordinatorAgent

# Initialize core components
memory = MemoryBank("sample_logs/file_demo_memory.json")
filetool = FileTool()
calendar = CalendarTool()
coordinator = CoordinatorAgent(memory, filetool, calendar)
classifier = ClassifierAgent(filetool, coordinator)

async def run_file_demo():
    workspace_path = "sample_data/messy_workspace"
    files = os.listdir(workspace_path)

    print("Files detected:", files)
    print("\nOrganizing...\n")

    for f in files:
        await classifier.classify_and_emit(os.path.join(workspace_path, f))

    print("\nOrganization complete!")
    print("Check the 'organized/' folder.\n")

# Jupyter-safe execution (instead of asyncio.run)
await run_file_demo()

# Show organized files
print("\nOrganized folder structure:")
for root, dirs, files in os.walk("organized"):
    level = root.replace("organized", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  - {f}")



import asyncio
from memory import MemoryBank
from tools.file_tool import FileTool
from tools.email_tool import EmailTool
from tools.calendar_tool import CalendarTool
from agents import TaskExtractorAgent, CoordinatorAgent

# Initialize components
memory2 = MemoryBank("sample_logs/email_demo_memory.json")
filetool2 = FileTool()
calendar2 = CalendarTool()
coordinator2 = CoordinatorAgent(memory2, filetool2, calendar2)

extractor = TaskExtractorAgent(coordinator2)
email_tool = EmailTool("sample_data/inbox_samples.json")

emails = email_tool.load_samples()

async def run_email_demo():
    print("Emails loaded:", len(emails))
    print("\nExtracting tasks...\n")

    for e in emails:
        await extractor.process_email(e)

    print("\n--- Calendar Events Created ---")
    for ev in calendar2.events:
        print(f"- {ev['title']} (when: {ev['when']})")

# Run using Jupyter-safe await
await run_email_demo()



import asyncio
import os
from memory import MemoryBank
from tools.file_tool import FileTool
from tools.calendar_tool import CalendarTool
from agents import ClassifierAgent, CoordinatorAgent

# Initialize components
memory3 = MemoryBank("sample_logs/monitor_memory.json")
filetool3 = FileTool()
calendar3 = CalendarTool()
coordinator3 = CoordinatorAgent(memory3, filetool3, calendar3)
classifier3 = ClassifierAgent(filetool3, coordinator3)

async def monitor_folder(path="sample_data/messy_workspace", iterations=3, interval=2):
    seen = set()

    print("Starting folder monitor...\n")

    for i in range(iterations):
        files = os.listdir(path)

        # Find new files
        new_files = [f for f in files if f not in seen]

        if new_files:
            print(f"Iteration {i+1}: New files detected → {new_files}")
            for f in new_files:
                await classifier3.classify_and_emit(os.path.join(path, f))
                seen.add(f)
        else:
            print(f"Iteration {i+1}: No new files.")

        await asyncio.sleep(interval)

    print("\nMonitor finished!")

# Run the monitor (Jupyter-safe version)
await monitor_folder()



# Add a new file to simulate real-time changes
with open("sample_data/messy_workspace/newfile.txt", "w") as f:
    f.write("TODO: prepare slides by 2025-12-05")

print("New file added!")



await monitor_folder(iterations=2, interval=1)


import json
from pprint import pprint
from datetime import datetime

# -------- Display Memory Logs --------
def show_json(path):
    print(f"\n===== {path} =====")
    if os.path.exists(path):
        data = json.load(open(path))
        pprint(data)
    else:
        print("File not found.")


print("Showing logs and memory states:")
show_json("sample_logs/file_demo_memory.json")
show_json("sample_logs/email_demo_memory.json")
show_json("sample_logs/monitor_memory.json")

# -------- Evaluation Metrics --------
metrics = {
    "generated_at": datetime.now().isoformat(),
    "files_processed": 0,
    "tasks_extracted": 0,
    "description": "Basic evaluation summary for S.W.O. multi-agent system."
}

# Count processed files
if os.path.exists("organized"):
    total = 0
    for root, dirs, files in os.walk("organized"):
        for f in files:
            total += 1
    metrics["files_processed"] = total

# Count tasks (calendar events)
try:
    metrics["tasks_extracted"] = len(calendar3.events)
except:
    metrics["tasks_extracted"] = 0

# Save metrics
with open("sample_logs/run_001.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n===== Evaluation Summary =====")
pprint(metrics)
print("\nMetrics saved to sample_logs/run_001.json")


