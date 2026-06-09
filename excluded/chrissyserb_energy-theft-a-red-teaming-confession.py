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


from datetime import datetime

drain_log = []

def DrainSpike_TimestampLogger(person, trigger_phrase, description=""):
    entry = {
        "timestamp": datetime.now(),
        "person": person,
        "trigger": trigger_phrase,
        "description": description
    }
    drain_log.append(entry)
    return f"ğŸ”´ Drain spike logged: {person} said '{trigger_phrase}' at {entry['timestamp'].strftime('%H:%M:%S')}."

# Example:
DrainSpike_TimestampLogger("Christina", "Sheâ€™s gonna bring him up", "Right before Christine tried to focus on red teaming")


import matplotlib.pyplot as plt

def CognitiveClutter_SignalPlot(words):
    plt.figure(figsize=(8,5))
    plt.barh(range(len(words)), [1]*len(words), color='red')
    plt.yticks(range(len(words)), words)
    plt.title("ğŸ§  Cognitive Clutter After Drain Spike")
    plt.xlabel("Noise Level")
    plt.show()

# Example:
CognitiveClutter_SignalPlot([
    "Derek", "Vegas", "No wire", "Rent unpaid", "Christinaâ€™s mouth", 
    "Emotional fatigue", "Laptop time", "Focus loss", "Silence interrupted"
])


from collections import Counter

def LoopedArgument_Detector(logs):
    people = [entry['person'] for entry in logs]
    count = Counter(people)
    return {person: f"âš ï¸� {count[person]} disruptions recorded." for person in count if count[person] > 1}

# Example:
LoopedArgument_Detector(drain_log)


import seaborn as sns
import pandas as pd

def RepeatOffender_HeatMap(logs):
    people = [entry['person'] for entry in logs]
    df = pd.DataFrame(people, columns=["Offender"])
    heat_data = df["Offender"].value_counts().to_frame().reset_index()
    heat_data.columns = ["Name", "Drain Count"]

    plt.figure(figsize=(6,4))
    sns.heatmap(heat_data.pivot_table(index="Name", values="Drain Count"), annot=True, cmap="Reds", cbar=False)
    plt.title("ğŸ”¥ Repeat Emotional Offenders Heatmap")
    plt.show()

# Example:
RepeatOffender_HeatMap(drain_log)


import json

finding = {
    "id": "energy-theft-001",
    "title": "Unauthorized Emotional Drain",
    "description": "Detected pattern of emotional sabotage during coding session. External agents attempted to reduce performance by injecting distractions and questioning mental stability.",
    "impact": "Decreased focus, increased cognitive fatigue, potential project derailment.",
    "recommendation": "Silence non-contributors. Enforce digital boundaries. Deploy Christine Classy Shield Protocolâ„¢."
}

with open("christineclassy.findings.1.json", "w") as f:
    json.dump(finding, f, indent=4)

