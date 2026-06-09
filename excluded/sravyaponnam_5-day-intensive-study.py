#!/usr/bin/env python3
"""
five_day_agent.py
Simple CLI "agent" for managing a 5-day intensive study/training program.

Features:
- Prepopulated 5-day schedule (customizable)
- View day's tasks
- Mark tasks complete / add notes
- Show progress summary
- Save/load plan to JSON
- Export to CSV
"""

import json
import csv
import os
from datetime import datetime

DEFAULT_PLAN = {
    "title": "5-Day Intensive Program",
    "created": datetime.utcnow().isoformat(),
    "days": [
        {"day": 1, "name": "Introduction & Foundation", "tasks": [
            {"id": 1, "title": "Overview & goals", "done": False, "notes": ""},
            {"id": 2, "title": "Install tools / environment", "done": False, "notes": ""},
            {"id": 3, "title": "Core concept lecture", "done": False, "notes": ""}
        ]},
        {"day": 2, "name": "Deep dive - Part 1", "tasks": [
            {"id": 1, "title": "Hands-on lab 1", "done": False, "notes": ""},
            {"id": 2, "title": "Problem solving session", "done": False, "notes": ""},
            {"id": 3, "title": "Q&A and recap", "done": False, "notes": ""}
        ]},
        {"day": 3, "name": "Deep dive - Part 2", "tasks": [
            {"id": 1, "title": "Advanced concepts", "done": False, "notes": ""},
            {"id": 2, "title": "Project start (mini)", "done": False, "notes": ""},
            {"id": 3, "title": "Checkpoint review", "done": False, "notes": ""}
        ]},
        {"day": 4, "name": "Project & Practice", "tasks": [
            {"id": 1, "title": "Project work", "done": False, "notes": ""},
            {"id": 2, "title": "Peer review / feedback", "done": False, "notes": ""},
            {"id": 3, "title": "Optimization & polish", "done": False, "notes": ""}
        ]},
        {"day": 5, "name": "Wrap-up & Presentation", "tasks": [
            {"id": 1, "title": "Final presentations", "done": False, "notes": ""},
            {"id": 2, "title": "Retrospective", "done": False, "notes": ""},
            {"id": 3, "title": "Next steps & resources", "done": False, "notes": ""}
        ]}
    ]
}

SAVE_FILE = "five_day_plan.json"

def save_plan(plan, filename=SAVE_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"[saved] {filename}")

def load_plan(filename=SAVE_FILE):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return DEFAULT_PLAN.copy()

def show_day(plan, day_no):
    days = plan["days"]
    if not (1 <= day_no <= len(days)):
        print("Invalid day number.")
        return
    d = days[day_no-1]
    print(f"\nDay {d['day']}: {d['name']}")
    for t in d["tasks"]:
        status = "✅" if t["done"] else "❌"
        print(f"  [{t['id']}] {status} {t['title']}")
        if t.get("notes"):
            print(f"      notes: {t['notes']}")

def mark_task(plan, day_no, task_id, done=True):
    try:
        t = plan["days"][day_no-1]["tasks"][task_id-1]
    except Exception:
        print("Task not found.")
        return
    t["done"] = bool(done)
    print(f"Marked task {task_id} on Day {day_no} as {'done' if done else 'not done'}.")

def add_note(plan, day_no, task_id, note):
    try:
        t = plan["days"][day_no-1]["tasks"][task_id-1]
    except Exception:
        print("Task not found.")
        return
    prev = t.get("notes","")
    t["notes"] = (prev + "\n" + note).strip() if prev else note
    print("Note added.")

def progress_summary(plan):
    total = 0
    done = 0
    for d in plan["days"]:
        for t in d["tasks"]:
            total += 1
            if t.get("done"):
                done += 1
    pct = (done/total*100) if total else 0
    print(f"\nProgress: {done}/{total} tasks done ({pct:.1f}%).")

def export_csv(plan, filename="five_day_plan.csv"):
    rows = []
    for d in plan["days"]:
        for t in d["tasks"]:
            rows.append({
                "day": d["day"],
                "day_name": d["name"],
                "task_id": t["id"],
                "task_title": t["title"],
                "done": t["done"],
                "notes": t.get("notes","")
            })
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["day","day_name","task_id","task_title","done","notes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[exported] {filename}")

def print_menu():
    print("""
Available commands:
  show <day>              - show schedule for day 1..5
  mark <day> <task_id>    - mark task done
  unmark <day> <task_id>  - mark task not done
  note <day> <task_id>    - add a note (will ask for note text)
  summary                 - show progress summary
  save                    - save plan to JSON
  export                  - export plan to CSV
  reset                   - reset to default plan
  quit                    - exit the agent
  help                    - show this menu
""")

def cli_loop():
    plan = load_plan()
    print(f"Agent loaded: {plan.get('title')}")
    print_menu()
    while True:
        cmd = input("\n> ").strip()
        if not cmd:
            continue
        parts = cmd.split()
        if parts[0] == "show" and len(parts) >= 2:
            try:
                day_no = int(parts[1])
                show_day(plan, day_no)
            except ValueError:
                print("Day must be a number.")
        elif parts[0] == "mark" and len(parts) >= 3:
            try:
                day_no = int(parts[1]); task_id = int(parts[2])
                mark_task(plan, day_no, task_id, done=True)
            except ValueError:
                print("Day and task_id must be numbers.")
        elif parts[0] == "unmark" and len(parts) >= 3:
            try:
                day_no = int(parts[1]); task_id = int(parts[2])
                mark_task(plan, day_no, task_id, done=False)
            except ValueError:
                print("Day and task_id must be numbers.")
        elif parts[0] == "note" and len(parts) >= 3:
            try:
                day_no = int(parts[1]); task_id = int(parts[2])
                note = input("Enter note text: ").strip()
                if note:
                    add_note(plan, day_no, task_id, note)
                else:
                    print("Empty note, cancelled.")
            except ValueError:
                print("Day and task_id must be numbers.")
        elif parts[0] == "summary":
            progress_summary(plan)
        elif parts[0] == "save":
            save_plan(plan)
        elif parts[0] == "export":
            export_csv(plan)
        elif parts[0] == "reset":
            confirm = input("Reset to default plan? (y/N) ").lower()
            if confirm == "y":
                plan = DEFAULT_PLAN.copy()
                print("Plan reset.")
        elif parts[0] in ("quit","exit"):
            confirm = input("Save before exit? (Y/n) ").lower()
            if confirm in ("","y","yes"):
                save_plan(plan)
            print("Goodbye!")
            break
        elif parts[0] == "help":
            print_menu()
        else:
            print("Unknown command. Type 'help'.")

if __name__ == "__main__":
    try:
        cli_loop()
    except KeyboardInterrupt:
        print("\nInterrupted -- exiting. Bye.")


