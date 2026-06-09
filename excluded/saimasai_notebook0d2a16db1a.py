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


from datetime import datetime, timedelta
import random
from IPython.display import display, HTML

# --- Topic Summaries ---
topic_summaries = {
    "vectors": "Vectors are quantities with magnitude and direction. Example: velocity, force.",
    "bonding": "Chemical bonding joins atoms to form molecules. Example: ionic, covalent bonds.",
    "respiration": "Respiration is the process by which cells produce energy using oxygen.",
    "physics": "Physics studies matter, energy, and their interactions.",
    "chemistry": "Chemistry studies substances, reactions, and molecular structures.",
    "biology": "Biology studies living organisms and life processes."
}

motivations = [
    "ğŸ’ª Keep going, youâ€™re improving every day!",
    "ğŸš€ Small steps today, big results tomorrow!",
    "ğŸŒŸ Focus now, succeed later!"
]

# --- Function to Generate Study Plan ---
def generate_study_plan(subjects, weak_topics, exam_date):
    try:
        datetime.strptime(exam_date, "%Y-%m-%d")
    except:
        display(HTML("<p style='color:red;'>â�Œ Please enter exam date in YYYY-MM-DD format.</p>"))
        return
    
    subjects_list = [s.strip() for s in subjects.split(',') if s.strip()]
    weak_list = [w.strip() for w in weak_topics.split(',') if w.strip()]
    
    if not subjects_list:
        display(HTML("<p style='color:red;'>â�Œ Please enter at least one subject.</p>"))
        return
    
    today = datetime.today()
    topic_pool = weak_list + [s for s in subjects_list if s not in weak_list]
    
    html = "<table style='width:100%; border-collapse: collapse; font-family: Arial;'>"
    html += "<tr style='background-color:#4CAF50; color:white;'><th>Date</th><th>Topic</th><th>Summary</th><th>Motivation</th></tr>"
    
    for i in range(7):
        day_date = today + timedelta(days=i)
        daily_topics = topic_pool[i*2:i*2+2] if len(topic_pool) >= (i+1)*2 else topic_pool[i*2:]
        if not daily_topics:
            continue
        for idx, t in enumerate(daily_topics):
            color = "red" if t.lower() in [w.lower() for w in weak_list] else "green"
            summary = topic_summaries.get(t.lower(), "Study key points for this topic.")
            motivation = random.choice(motivations)
            row_color = "#f9f9f9" if idx % 2 == 0 else "#ffffff"
            html += f"<tr style='background-color:{row_color};'>"
            html += f"<td style='border:1px solid black; padding:5px; text-align:center;'>{day_date.strftime('%d %b')}</td>"
            html += f"<td style='border:1px solid black; padding:5px; color:{color}; font-weight:bold;'>{t}</td>"
            html += f"<td style='border:1px solid black; padding:5px;'>{summary}</td>"
            html += f"<td style='border:1px solid black; padding:5px; text-align:center;'>{motivation}</td>"
            html += "</tr>"
    
    html += "</table>"
    display(HTML(html))

# --- Example Usage ---
subjects = "Physics, Chemistry, Biology"
weak_topics = "Vectors, Bonding, Respiration"
exam_date = "2025-12-01"

generate_study_plan(subjects, weak_topics, exam_date)


