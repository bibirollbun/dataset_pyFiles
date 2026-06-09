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


import threading

# Memory for voter and voting history
class MemoryBank:
    def __init__(self):
        self.voters = {}        # voter_id: status
        self.votes = {}         # voter_id: candidate
        self.fraud_reports = [] # list of fraud incidents

# Session state manager
class SessionManager:
    def __init__(self):
        self.sessions = {}      # voter_id: session_status

    def start_session(self, voter_id):
        self.sessions[voter_id] = 'active'
    def end_session(self, voter_id):
        self.sessions[voter_id] = 'ended'

# Registration agent (thread)
def registration_agent(voter_id, memory_bank, session_manager):
    # Tool: Identity verification
    if voter_id not in memory_bank.voters:
        memory_bank.voters[voter_id] = 'registered'
        session_manager.start_session(voter_id)
        return "Registration successful"
    else:
        return "Already registered"

# Voting agent (thread)
def voting_agent(voter_id, candidate, memory_bank, session_manager):
    # Tool: Vote database
    if (voter_id in memory_bank.voters and
        voter_id not in memory_bank.votes and
        session_manager.sessions.get(voter_id) == 'active'):
        memory_bank.votes[voter_id] = candidate
        session_manager.end_session(voter_id)
        return "Vote cast successfully"
    else:
        return "Cannot vote (check registration/session)"

# Fraud detection agent (thread)
def fraud_detection_agent(memory_bank):
    # Tool: Anomaly detection
    reports = []
    # Detect duplicate votes
    if len(memory_bank.votes) != len(set(memory_bank.votes.keys())):
        reports.append("Duplicate voting detected")
    # Detect multiple registrations (example)
    multiple_regs = [v for v in memory_bank.voters if list(memory_bank.voters.values()).count('registered') > 1]
    if multiple_regs:
        reports.append("Multiple registrations detected")
    memory_bank.fraud_reports.extend(reports)
    return reports

# Orchestrator, parallel execution
def voting_orchestrator(voter_id, candidate):
    memory_bank = MemoryBank()
    session_manager = SessionManager()
    threads = []

    # Step 1: Register voter
    t1 = threading.Thread(target=registration_agent, args=(voter_id, memory_bank, session_manager))
    threads.append(t1)
    t1.start()

    # Step 2: Vote
    t2 = threading.Thread(target=voting_agent, args=(voter_id, candidate, memory_bank, session_manager))
    threads.append(t2)
    t2.start()

    # Step 3: Detect fraud
    t3 = threading.Thread(target=fraud_detection_agent, args=(memory_bank,))
    threads.append(t3)
    t3.start()

    for t in threads:
        t.join()

    # Collect and print results
    print(f"Registered Voters: {memory_bank.voters}")
    print(f"Votes: {memory_bank.votes}")
    print(f"Fraud Reports: {memory_bank.fraud_reports}")

# Demo usage
voting_orchestrator("voter123", "Alice")

