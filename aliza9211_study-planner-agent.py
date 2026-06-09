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


# Cell: imports & utilities
from datetime import datetime, timedelta
import math
import json
import textwrap

# simple sentence splitter for summarizer
def split_sentences(text):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def clamp(x, a, b):
    return max(a, min(b, x))



# Cell: Tools (represent "tools" in ADK)
def tool_calculate_total_available_hours(daily_hours_dict):
    """Sum of hours available each day in the input dict."""
    return sum(daily_hours_dict.values())

def tool_allocate_hours_to_subjects(subjects, total_hours, mode='balanced'):
    """
    subjects: list of dicts [{'name':..., 'priority':1-5, 'difficulty':1-5, 'estimated_hours':... or None}]
    total_hours: float (hours to allocate)
    mode: 'balanced' or 'priority'
    Returns: dict mapping subject->hours
    """
    # base weight = priority * difficulty
    weights = []
    for s in subjects:
        p = clamp(s.get('priority', 3), 1, 5)
        d = clamp(s.get('difficulty', 3), 1, 5)
        est = s.get('estimated_hours', None)
        weights.append((s, p * d, est))
    # if some have estimated_hours, reserve those first
    reserved = 0
    allocation = {}
    for s,w,est in weights:
        if est:
            allocation[s['name']] = min(est, total_hours)  # reserve up to est
            reserved += allocation[s['name']]
    remaining = max(0, total_hours - reserved)
    # distribute remaining by weight
    total_weight = sum(w for s,w,est in weights if not est)
    if total_weight == 0:
        # equal distribution
        equal = remaining / max(1, sum(1 for s,w,est in weights if not est))
        for s,w,est in weights:
            if not est:
                allocation[s['name']] = equal
    else:
        for s,w,est in weights:
            if not est:
                allocation[s['name']] = remaining * (w / total_weight)
    # clamp small numbers
    for k in allocation:
        if allocation[k] < 0.25:
            allocation[k] = round(allocation[k], 2)
        else:
            allocation[k] = round(allocation[k], 2)
    return allocation

def tool_plan_daily_schedule(day_name, hours_available, subject_hours_alloc, max_block=2.0):
    """
    Create a simple daily schedule dividing subject_hours_alloc (dict) into study blocks not exceeding max_block hours.
    Returns a list of (start_time, end_time, subject, note).
    We'll use relative times (start at 09:00) — user can edit times later.
    """
    start_time = datetime(2000,1,1,9,0)  # dummy date, time at 9:00
    schedule = []
    cur = start_time
    # flatten subject_hours_alloc into blocks
    for subj, hrs in subject_hours_alloc.items():
        remaining = hrs
        while remaining > 0.01:
            block = min(max_block, remaining)
            end = cur + timedelta(hours=block)
            schedule.append({
                'day': day_name,
                'start': cur.strftime('%H:%M'),
                'end': end.strftime('%H:%M'),
                'subject': subj,
                'duration_hours': round(block,2),
                'note': ''
            })
            # add short break of 15 minutes after each block
            cur = end + timedelta(minutes=15)
            remaining -= block
    return schedule

def tool_summarize_notes(text, max_sentences=3):
    """
    Very simple extractive summarizer: score sentences by keyword overlap.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""
    # build keyword set from text (simple)
    words = [w.lower().strip('.,!?') for w in text.split()]
    freq = {}
    for w in words:
        if len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    # score sentences
    scores = []
    for s in sentences:
        s_words = [w.lower().strip('.,!?') for w in s.split()]
        score = sum(freq.get(w, 0) for w in s_words)
        scores.append((score, s))
    scores.sort(reverse=True, key=lambda x: x[0])
    chosen = [s for sc,s in scores[:max_sentences]]
    # preserve original order
    chosen_sorted = [s for s in sentences if s in chosen]
    return " ".join(chosen_sorted)



# Cell: Planner + Agent orchestration
class SmartStudyAgent:
    def __init__(self, name="SmartStudyAgent"):
        self.name = name
        # store history / memory for session
        self.history = []
    
    # High-level pipeline (workflow execution)
    def create_study_plan(self, user_input):
        """
        user_input: dict {
            'subjects': [{'name','priority','difficulty','estimated_hours' (optional)}...],
            'daily_hours': {'Mon':3, 'Tue':4, ...} OR single float default,
            'mode': 'balanced' or 'priority'
        }
        returns: result dict with daily & weekly plans and summaries
        """
        # 1. Tool call: calculate total hours
        if isinstance(user_input.get('daily_hours'), dict):
            total_hours = tool_calculate_total_available_hours(user_input['daily_hours'])
            daily_hours_dict = user_input['daily_hours']
        else:
            # if single float -> assume that per day for 7 days
            per_day = float(user_input.get('daily_hours', 2))
            daily_hours_dict = {d: per_day for d in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']}
            total_hours = tool_calculate_total_available_hours(daily_hours_dict)
        
        subjects = user_input.get('subjects', [])
        mode = user_input.get('mode', 'balanced')
        
        # 2. Tool call: allocate weekly hours among subjects
        allocated_weekly = tool_allocate_hours_to_subjects(subjects, total_hours, mode=mode)
        
        # 3. Reasoning: split weekly allocation into daily based on daily_hours proportions
        # compute day proportions
        day_values = [daily_hours_dict[d] for d in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']]
        total_day_values = sum(day_values) or 1
        day_fracs = {d: daily_hours_dict[d]/total_day_values for d in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']}
        
        # build per-day subject allocations
        daily_allocations = {d: {} for d in day_fracs}
        for subj, week_hours in allocated_weekly.items():
            for d, frac in day_fracs.items():
                daily_allocations[d][subj] = round(week_hours * frac, 2)
        
        # 4. Tool call: plan each day's schedule (blocks)
        schedules = {}
        for d in daily_allocations:
            schedules[d] = tool_plan_daily_schedule(d, daily_hours_dict.get(d, 0), daily_allocations[d], max_block=2.0)
        
        # Save to history
        result = {
            'total_hours': total_hours,
            'allocated_weekly': allocated_weekly,
            'daily_allocations': daily_allocations,
            'schedules': schedules,
            'notes_summary': {}
        }
        self.history.append({'input': user_input, 'result': result, 'timestamp': datetime.now().isoformat()})
        return result
    
    def summarize_notes_for_subject(self, subject_name, note_text, max_sentences=3):
        # Tool call: summarizer
        summ = tool_summarize_notes(note_text, max_sentences=max_sentences)
        # store
        self.history.append({'type': 'summary', 'subject': subject_name, 'summary': summ, 'timestamp': datetime.now().isoformat()})
        return summ



# Cell: Example usage
agent = SmartStudyAgent()

# Example user input
user_input = {
    'subjects': [
        {'name': 'Operating Systems', 'priority': 5, 'difficulty': 5},
        {'name': 'DBMS', 'priority': 4, 'difficulty': 4},
        {'name': 'Java', 'priority': 4, 'difficulty': 3},
        {'name': 'Maths 2', 'priority': 5, 'difficulty': 4}
    ],
    # specify daily hours as dict (Mon-Sun)
    'daily_hours': {'Mon':3, 'Tue':3, 'Wed':3, 'Thu':3, 'Fri':3, 'Sat':5, 'Sun':4},
    'mode': 'priority'
}

plan = agent.create_study_plan(user_input)

# show summary
print("Total weekly hours available:", plan['total_hours'])
print("\nWeekly allocation (hours per subject):")
for s,h in plan['allocated_weekly'].items():
    print(f" - {s}: {h} hrs")

print("\nDaily allocations (Mon example):")
import pprint
pprint.pprint(plan['daily_allocations']['Mon'])

print("\nMon schedule blocks:")
for block in plan['schedules']['Mon'][:10]:
    print(f"{block['start']} - {block['end']} | {block['subject']} | {block['duration_hours']} hrs")



# Cell: Summarizer demo
example_notes = """
Operating System is the core of the computer system. It manages hardware and provides services to applications.
Important topics: CPU scheduling, memory management, processes and threads, deadlock, file systems, and virtual memory.
CPU scheduling algorithms include FCFS, SJF, Round Robin, Priority Scheduling. Understanding context switch is essential.
Virtual memory uses paging; page replacement algorithms like FIFO, LRU, Optimal matter for performance.
Deadlock conditions: mutual exclusion, hold and wait, no preemption, circular wait. Avoidance vs prevention.
"""

summary = agent.summarize_notes_for_subject("Operating Systems", example_notes, max_sentences=3)
print("Summary:\n", summary)



# Cell: Export to markdown
def export_plan_to_markdown(plan, title="Smart Study Planner"):
    md = []
    md.append(f"# {title}\n")
    md.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append(f"**Total weekly hours available:** {plan['total_hours']}\n")
    md.append("## Weekly Allocation\n")
    for s,h in plan['allocated_weekly'].items():
        md.append(f"- **{s}**: {h} hrs\n")
    md.append("\n## Daily Allocations (sample: Monday)\n")
    for s,h in plan['daily_allocations']['Mon'].items():
        md.append(f"- {s}: {h} hrs\n")
    md.append("\n## Monday Schedule\n")
    for b in plan['schedules']['Mon']:
        md.append(f"- {b['start']} - {b['end']}: {b['subject']} ({b['duration_hours']} hrs)\n")
    return "\n".join(md)

md_text = export_plan_to_markdown(plan)
print(md_text[:1000])  # print first part
# If on Kaggle, you can save:
with open('study_plan.md','w',encoding='utf-8') as f:
    f.write(md_text)
print("Saved study_plan.md")


