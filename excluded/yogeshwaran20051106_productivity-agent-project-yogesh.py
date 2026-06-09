# Productivity Agent - Capstone Project (Yogesh)
# NOTE: Kaggle input() interaction doesn't work. Use local machine for full demo.

import os
import sqlite3
import time
import threading
from datetime import datetime, timedelta

# Optional LLM support
try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

DB_PATH = 'tasks.db'

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    description TEXT,
                    due TIMESTAMP,
                    priority INTEGER,
                    status TEXT
                )''')
    conn.commit()
    conn.close()

def add_task(title, description='', due=None, priority=3, status='pending'):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO tasks (title, description, due, priority, status) VALUES (?,?,?,?,?)',
              (title, description, due, priority, status))
    conn.commit()
    task_id = c.lastrowid
    conn.close()
    return task_id

def list_tasks(show_all=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if show_all:
        c.execute('SELECT id, title, description, due, priority, status FROM tasks ORDER BY priority ASC, due ASC')
    else:
        c.execute("SELECT id, title, description, due, priority, status FROM tasks WHERE status!='done' ORDER BY priority ASC, due ASC")
    rows = c.fetchall()
    conn.close()
    return rows

def update_task_status(task_id, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE tasks SET status=? WHERE id=?', (status, task_id))
    conn.commit()
    conn.close()

def format_tasks(rows):
    if not rows:
        return "No tasks found."
    lines = []
    for r in rows:
        tid, title, desc, due, pr, st = r
        due_val = due if due else "no-due"
        lines.append(f"[{tid}] {title} ({st}) priority={pr}, due={due_val}\n  {desc}")
    return "\n".join(lines)

def call_llm(prompt):
    if not OPENAI_AVAILABLE:
        return "(LLM not installed)"
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "(OPENAI_API_KEY not set)"
    openai.api_key = key
    resp = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.2
    )
    return resp.choices[0].text.strip()

def parse_and_execute(user_text):
    text = user_text.lower()

    if text.startswith("add"):
        parts = user_text.split("|")
        title = parts[0].replace("add", "").strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        tid = add_task(title, desc)
        return f"Added task id={tid}"

    if "list" in text:
        rows = list_tasks()
        return format_tasks(rows)

    if "done" in text or "complete" in text:
        import re
        m = re.search(r"(\d+)", text)
        if m:
            tid = int(m.group(1))
            update_task_status(tid, "done")
            return f"Task {tid} marked done"
        return "Specify a task id!"

    return "Unknown command!"

# Initialize DB
init_db()

# Add demo tasks for Kaggle output
add_task("Submit capstone draft", "Attach notebook and explanation", priority=1)
add_task("Buy groceries", "Milk, rice, vegetables", priority=3)

print("Demo task list:")
print(format_tasks(list_tasks()))

