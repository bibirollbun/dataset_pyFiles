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


import re
from datetime import datetime
from copy import deepcopy

case_memory_bank = []

class IngestAgent:
    def parse_input(self, input_text):
        events = []
        for line in input_text.strip().split('\n'):
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) user=(\w+)(?: ip=([\d\.]+))?(?: file=([\w\/\.]+))?", line)
            if match:
                timestamp, action, user, ip, file = match.groups()
                events.append({
                    "timestamp": timestamp,
                    "action": action,
                    "user": user,
                    "ip": ip if ip else '',
                    "file": file if file else ''
                })
        return events

class TimelineAgent:
    def construct_timeline(self, events):
        events_sorted = sorted(events, key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"))
        return events_sorted

ATTACK_ARCHETYPES = [
    {
        "name": "Insider Threat",
        "markers": ["FILE_DELETE", "FILE_ACCESS", "LOGIN"],
        "traits": ["Familiar with system", "Targeted resources", "Covers tracks"],
        "priority": 2
    },
    {
        "name": "Persistent Threat/APT",
        "markers": ["FAILED_LOGIN", "LOGIN", "FILE_ACCESS", "multiple sources"],
        "traits": ["Patient", "Stealthy", "Tries several avenues"],
        "priority": 3
    },
    {
        "name": "Opportunist",
        "markers": ["LOGIN", "FILE_ACCESS"],
        "traits": ["Quick", "Resource Seeker"],
        "priority": 1
    },
    {
        "name": "Hacktivist/Saboteur",
        "markers": ["FILE_DELETE"],
        "traits": ["Destructive", "Motivated by message", "Defaces or deletes"],
        "priority": 2
    }
]

class TechnicalAnalysisAgent:
    def analyze(self, timeline):
        anomalies = []
        login_ips = set()
        last_user = ""
        for ev in timeline:
            if ev['action'] == "LOGIN":
                if ev['ip'] and ev['ip'] in login_ips:
                    anomalies.append(f"Repeated IP login detected: {ev['ip']} for user {ev['user']}")
                login_ips.add(ev['ip'])
                if last_user == ev['user']:
                    anomalies.append(f"User {ev['user']} logged in again soon after previous session.")
                last_user = ev['user']
            if ev['action'] == "FAILED_LOGIN":
                anomalies.append(f"Failed login attempt for user {ev['user']} at {ev['timestamp']}")
            if ev['action'] == "FILE_DELETE":
                anomalies.append(f"File deletion detected: {ev['file']} by {ev['user']} at {ev['timestamp']}")
        return anomalies

class BehavioralAnalysisAgent:
    def analyze(self, timeline):
        behaviors = []
        failed_ratio = sum(ev['action']=="FAILED_LOGIN" for ev in timeline) / max(1, len(timeline))
        if failed_ratio > 0.4:
            behaviors.append("High proportion of failed logins - possible brute force.")
        files_accessed = [ev['file'] for ev in timeline if ev['action']=="FILE_ACCESS"]
        if files_accessed:
            behaviors.append(f"Files accessed: {', '.join([f for f in files_accessed if f])}")
        users = set(ev['user'] for ev in timeline)
        if len(users) > 1:
            behaviors.append("Multiple users involved — possible shared account or insider help.")
        return behaviors

class ProfilerAgent:
    def match_archetype(self, timeline):
        action_set = set(ev['action'] for ev in timeline)
        for profile in sorted(ATTACK_ARCHETYPES, key=lambda x: -x["priority"]):
            if any(marker in action_set for marker in profile["markers"]):
                return deepcopy(profile)
        return {"name": "Unknown", "markers": [], "traits": [], "priority": 0}

    def profile_adversary(self, timeline, tech_findings, behav_findings):
        archetype = self.match_archetype(timeline)
        profile = {
            "archetype": archetype["name"],
            "traits": archetype["traits"],
            "details": tech_findings + behav_findings,
        }
        profile["llm_narrative"] = (
            f"Based on the artifacts and behavior, this case likely fits the '{archetype['name']}' profile — "
            f"{', '.join(archetype['traits'])}. "
            f"Key findings: {'; '.join(tech_findings+behav_findings) if (tech_findings+behav_findings) else 'No significant anomalies.'}"
        )
        return profile

class SummarizerAgent:
    def summarize(self, timeline, profile):
        s = "# Investigation Report\n\n"
        s += "### Timeline\n"
        for ev in timeline:
            s += f"- {ev['timestamp']}: **{ev['action']}** by {ev['user']} "
            if ev["ip"]: s += f"(IP: {ev['ip']}) "
            if ev["file"]: s += f"on {ev['file']}"
            s += "\n"
        s += "\n### Adversary Profile\n"
        s += f"- Type: **{profile['archetype']}**\n"
        s += f"- Traits: {', '.join(profile['traits'])}\n"
        s += "\n### Analytical Commentary\n"
        for d in profile['details']:
            s += f"• {d}\n"
        s += f"\n---\n{profile['llm_narrative']}\n"
        return s

def run_case_analysis(case_log_text):
    ingest = IngestAgent()
    log_events = ingest.parse_input(case_log_text)
    timeline_agent = TimelineAgent()
    timeline = timeline_agent.construct_timeline(log_events)

    tech_agent = TechnicalAnalysisAgent()
    tech_findings = tech_agent.analyze(timeline)

    behav_agent = BehavioralAnalysisAgent()
    behav_findings = behav_agent.analyze(timeline)

    profiler = ProfilerAgent()
    profile = profiler.profile_adversary(timeline, tech_findings, behav_findings)

    summarizer = SummarizerAgent()
    report = summarizer.summarize(timeline, profile)
    return {
        "timeline": timeline,
        "profile": profile,
        "report": report,
    }


case3_log = """
2025-11-14 12:05:18 LOGIN user=susan ip=45.4.5.6
2025-11-14 12:06:25 FAILED_LOGIN user=susan ip=45.4.5.6
2025-11-14 12:07:00 FAILED_LOGIN user=susan ip=45.4.5.6
2025-11-14 12:09:10 LOGIN user=susan ip=45.4.5.6
2025-11-14 12:10:22 FILE_ACCESS user=susan file=/db/config.json
"""
new_case_log = case3_log

result = run_case_analysis(new_case_log)
print(result['report'])

case_data = {
    "log": new_case_log,
    "archetype": result["profile"]["archetype"],
    "traits": result["profile"]["traits"],
    "llm_narrative": result["profile"]["llm_narrative"]
}
case_memory_bank.append(case_data)


def compare_with_memory_bank(latest_case, memory_bank):
    print("\n# Memory Bank Comparison\n")
    if len(memory_bank) < 2:
        print("Not enough cases for comparison yet.")
        return

    last_case = memory_bank[-1]
    recurrences = []
    for p in memory_bank[:-1]:
        if p['archetype'] == last_case['archetype']:
            recurrences.append(p)
    if recurrences:
        print(f"Recurring archetype: {last_case['archetype']}, seen {len(recurrences)+1} times including current.")
        for i, case in enumerate(recurrences):
            print(f"Case {i+1}: Traits — {', '.join(case['traits'])}")
            print("Narrative excerpt:", case['llm_narrative'][:120], "...")
    else:
        print("No recurring attacker archetypes detected in prior cases.")

compare_with_memory_bank(result, case_memory_bank)


user_case_log = """
2025-11-16 10:00:02 LOGIN user=anil ip=1.2.3.4
2025-11-16 10:01:10 FAILED_LOGIN user=anil ip=1.2.3.4
2025-11-16 10:02:17 FILE_ACCESS user=anil file=/secrets/plan.docx
2025-11-16 10:03:21 FILE_DELETE user=anil file=/secrets/plan.docx
"""


class ProfilerAgent:
    def match_archetype(self, timeline):
        action_set = set(ev['action'] for ev in timeline)
        best_profile = {"name": "Unknown", "markers": [], "traits": [], "priority": 0}
        reason = "No matching pattern found."
        for profile in sorted(ATTACK_ARCHETYPES, key=lambda x: -x["priority"]):
            matched = [marker for marker in profile["markers"] if marker in action_set]
            if matched:
                reason = (
                    f"Matched actions: {', '.join(matched)} "
                    f"→ classified as '{profile['name']}' due to known behavior patterns."
                )
                best_profile = deepcopy(profile)
                break
        return best_profile, reason

    def profile_adversary(self, timeline, tech_findings, behav_findings):
        archetype, reason = self.match_archetype(timeline)
        profile = {
            "archetype": archetype["name"],
            "traits": archetype["traits"],
            "details": tech_findings + behav_findings,
            "why_this_profile": reason,  # <-- added explainability here
        }
        profile["llm_narrative"] = (
            f"Based on the artifacts and behavior, this case likely fits the '{archetype['name']}' profile — "
            f"{', '.join(archetype['traits'])}. "
            f"Key findings: {'; '.join(tech_findings+behav_findings) if (tech_findings+behav_findings) else 'No significant anomalies.'}"
        )
        return profile


s += f"\n### Explainability\n"
s += f"{profile.get('why_this_profile', '')}\n"


import matplotlib.pyplot as plt

def visualize_timeline(timeline):
    # Map actions to a color for visibility
    color_map = {
        'LOGIN': 'blue',
        'FAILED_LOGIN': 'orange',
        'FILE_ACCESS': 'green',
        'FILE_DELETE': 'red'
    }
    ys = []
    labels = []
    times = []
    for i, ev in enumerate(timeline):
        t = datetime.strptime(ev['timestamp'], "%Y-%m-%d %H:%M:%S")
        times.append(t)
        ys.append(i)
        action = ev["action"]
        labels.append(f"{ev['action']} ({ev['user']})")
    colors = [color_map.get(ev["action"], "gray") for ev in timeline]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(times, ys, c=colors, s=200, marker="o", edgecolors='k')
    for x, y, label in zip(times, ys, labels):
        ax.text(x, y + 0.15, label, ha='center', fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Timestamp")
    ax.set_title("Incident Timeline Visualization")
    plt.tight_layout()
    plt.grid(axis='x')
    plt.show()

# Example use after you get a result:
visualize_timeline(result["timeline"])

