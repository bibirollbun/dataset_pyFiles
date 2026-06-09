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


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd
import random
import datetime
import matplotlib.pyplot as plt

random.seed(42)
print("Zentrack AI Demo Notebook Initialized")



user_profile = {
    "name": "Demo User",
    "timezone": "Asia/Kolkata",
    "habits": [
        {"name": "Drink Water", "target": "8 glasses/day"},
        {"name": "Walking", "target": "5000 steps/day"},
        {"name": "Meditation", "target": "10 minutes/day"}
    ]
}

user_profile



schedule = []

start_date = datetime.date.today() - datetime.timedelta(days=13)
dates = [start_date + datetime.timedelta(days=i) for i in range(14)]

for d in dates:
    for habit in user_profile["habits"]:
        schedule.append({
            "date": d,
            "habit": habit["name"],
            "reminder_time": "09:00 AM"
        })

df_schedule = pd.DataFrame(schedule)
df_schedule.head()



history = []

actions = ["done", "skipped", "snoozed"]

for entry in schedule:
    user_action = random.choices(
        population=actions,
        weights=[0.6, 0.25, 0.15],
        k=1
    )[0]

    history.append({
        "date": entry["date"],
        "habit": entry["habit"],
        "status": user_action
    })

df_history = pd.DataFrame(history)
df_history.head()



summary = (
    df_history.assign(count=1)
    .pivot_table(index="date", columns="habit", values="count", aggfunc="sum")
)

completion_rate = (df_history["status"] == "done").mean() * 100

print(f"Weekly Completion Rate: {completion_rate:.2f}%")

df_summary = df_history.groupby(["date", "status"]).size().unstack(fill_value=0)
df_summary.tail()



completion_daily = df_history.groupby("date")["status"].apply(
    lambda x: (x == "done").mean() * 100
)

plt.figure(figsize=(12, 5))
completion_daily.plot(marker="o")
plt.title("Daily Completion Rate (%)")
plt.xlabel("Date")
plt.ylabel("Completion %")
plt.grid()
plt.show()



df_history.to_csv("habit_history.csv", index=False)
df_schedule.to_csv("habit_schedule.csv", index=False)

print("CSV files exported: habit_history.csv, habit_schedule.csv")



recommendations = [
    "Try drinking 1 glass of water immediately after waking up.",
    "Short evening walks improve mood and sleep quality.",
    "Meditating at the same time every day builds stronger habits."
]

random.choice(recommendations)



print("Zentrack AI Demo Completed Successfully ✔")



!git clone https://github.com/nivedithan29/zentrack-ai.git



%cd zentrack-ai
!python src/demo/run_simulation.py



import sys
sys.path.append("/kaggle/working/zentrack-ai")
sys.path.append("/kaggle/working/zentrack-ai/src")
print("Path fixed!")



%cd /kaggle/working/zentrack-ai
!python -m src.demo.run_simulation





