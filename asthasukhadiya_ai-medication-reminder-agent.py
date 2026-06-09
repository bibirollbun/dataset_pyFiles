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


# Import necessary modules
from typing import Dict, List
from datetime import datetime, timedelta

# Simple in-memory memory bank and session service implementation for demo
class MemoryBank:
    def __init__(self):
        self.store = {}

    def get(self, session_id, key, default=None):
        return self.store.get(session_id, {}).get(key, default)

    def set(self, session_id, key, value):
        if session_id not in self.store:
            self.store[session_id] = {}
        self.store[session_id][key] = value

memory_bank = MemoryBank()

# Base Agent class
class Agent:
    def act(self, session_id: str, context: Dict):
        raise NotImplementedError("Agent action method must be implemented")

# Reminder Agent: sends medication reminders
class ReminderAgent(Agent):
    def act(self, session_id: str, context: Dict):
        profile = memory_bank.get(session_id, 'patient_profile')
        now = datetime.now()

        if not profile:
            return "No patient profile found."

        next_dose_time = profile.get('next_dose_time')
        medication = profile.get('medication')

        if not next_dose_time or not medication:
            return "Incomplete patient profile."

        # If current time is past next dose time, send reminder
        if now >= next_dose_time:
            self.send_reminder(session_id, medication)
            # Update next dose time to next schedule (e.g., 24 hrs after)
            new_time = next_dose_time + timedelta(hours=24)
            profile['next_dose_time'] = new_time
            memory_bank.set(session_id, 'patient_profile', profile)
            return f"Reminder sent to take {medication}."
        else:
            return "No reminder needed now."

    def send_reminder(self, session_id, medication):
        # Simulate sending reminder (e.g., via text-to-speech or notification)
        print(f"[ReminderAgent] Session {session_id}: Reminder -> Please take your medication: {medication}")

# Monitoring Agent: monitors medication adherence logs
class MonitoringAgent(Agent):
    def act(self, session_id: str, context: Dict):
        adherence_log = memory_bank.get(session_id, 'adherence_log', [])
        missed_doses = sum(1 for entry in adherence_log if entry.get('status') == 'missed')

        # If missed doses >= 3, escalate alert
        if missed_doses >= 3:
            notification_agent.act(session_id, context)
            return f"3 or more missed doses detected. Caregiver notified."
        else:
            return f"Missed doses: {missed_doses}. Monitoring normal."

# Notification Agent: alerts caregiver if issues detected
class NotificationAgent(Agent):
    def act(self, session_id: str, context: Dict):
        caregiver_contact = memory_bank.get(session_id, 'caregiver_contact')
        if not caregiver_contact:
            return "Caregiver contact not found."
        # Simulate sending alert to caregiver
        print(f"[NotificationAgent] Alert sent to caregiver at {caregiver_contact} for session {session_id}.")
        return "Caregiver alerted."

# Initialize agents
reminder_agent = ReminderAgent()
monitoring_agent = MonitoringAgent()
notification_agent = NotificationAgent()

# Example usage
if __name__ == "__main__":
    session_id = "patient_001"

    # Set patient profile and caregiver contact
    memory_bank.set(session_id, 'patient_profile', {
        'name': 'Alice',
        'medication': 'Blood Pressure Pill',
        'next_dose_time': datetime.now() - timedelta(minutes=5)  # Due 5 minutes ago
    })
    memory_bank.set(session_id, 'caregiver_contact', 'caregiver@example.com')

    # Initialize adherence log with some missed doses
    memory_bank.set(session_id, 'adherence_log', [
        {'date': '2025-11-25', 'status': 'taken'},
        {'date': '2025-11-26', 'status': 'missed'},
        {'date': '2025-11-27', 'status': 'missed'},
        {'date': '2025-11-28', 'status': 'missed'},
    ])

    # Run agents sequentially
    res1 = reminder_agent.act(session_id, {})
    print(res1)
    res2 = monitoring_agent.act(session_id, {})
    print(res2)

