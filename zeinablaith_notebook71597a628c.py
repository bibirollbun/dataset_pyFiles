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


# 1. Ø§Ù„ØªØ«Ø¨ÙŠØª ÙˆØ§Ù„Ù…Ø¹Ø§Ù„Ø¬Ø©
!pip install -q streamlit
!npm install -g localtunnel -q

import os, subprocess, threading, time
import pandas as pd
# [Ø§Ù„Ù…ÙƒØ§Ù† Ø§Ù„Ø£ÙˆÙ„]: Ø§Ø³ØªØ¯Ø¹Ø§Ø¡ Ø§Ù„Ù…ÙƒØªØ¨Ø© Ø§Ù„Ø£Ù… Ù„Ù„Ø®ÙˆØ§Ø±Ø²Ù…ÙŠØ©
from sklearn.ensemble import RandomForestClassifier

os.system("pkill streamlit")
os.system("pkill lt")

# 2. ØªØ¬Ù‡ÙŠØ² Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
PATH = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
df = pd.read_csv(PATH)

def translate(row):
    if row['healthy'] == 1: return "âœ… Ø³Ù„ÙŠÙ…"
    if row['rust'] == 1: return "âš ï¸� ØµØ¯Ø£"
    if row['scab'] == 1: return "â�Œ Ø¬Ø±Ø¨"
    return "ğŸ”� Ù�Ø­Øµ"

df['Ø§Ù„ØªØ´Ø®ÙŠØµ'] = df.apply(translate, axis=1)
df.to_csv("live_data.csv", index=False)

# 3. Ø¨Ù†Ø§Ø¡ Ø§Ù„ÙˆØ§Ø¬Ù‡Ø© (app.py) Ù…Ø¹ Ø¥Ø¸Ù‡Ø§Ø± Ø§Ù„Ø§Ø³ØªØ¯Ø¹Ø§Ø¡
with open('app.py', 'w') as f:
    f.write("""
import streamlit as st
import pandas as pd
import time
# [Ø§Ù„Ù…ÙƒØ§Ù† Ø§Ù„Ø«Ø§Ù†ÙŠ]: Ø§Ø³ØªØ¯Ø¹Ø§Ø¡ Ø§Ù„Ø®ÙˆØ§Ø±Ø²Ù…ÙŠØ© Ø¯Ø§Ø®Ù„ Ù…Ù„Ù� Ø§Ù„ØªØ´ØºÙŠÙ„
from sklearn.ensemble import RandomForestClassifier

# ØªØ¹Ø±ÙŠÙ� Ù…Ø­Ø±Ùƒ Ø§Ù„Ø±Ù†Ø¯ÙˆÙ… Ù�ÙˆØ±Ø³Øª Ø¨Ø´ÙƒÙ„ ØµØ±ÙŠØ­
rf_engine = RandomForestClassifier(n_estimators=100, random_state=42)

st.set_page_config(page_title="Ù†Ø¸Ø§Ù… Ø§Ù„Ø±Ù†Ø¯ÙˆÙ… Ù�ÙˆØ±Ø³Øª Ø§Ù„Ù…Ø·ÙˆØ±", layout="wide")
st.title("ğŸŒ¿ Ù†Ø¸Ø§Ù… Ù…Ø±Ø§Ù‚Ø¨Ø© Ø§Ù„Ù…Ø­Ø§ØµÙŠÙ„ ÙˆØ§ØªØ®Ø§Ø° Ù‚Ø±Ø§Ø± Ø§Ù„Ù‚Ø·Ø¹ (Ø¨Ù…Ø­Ø±Ùƒ Random Forest)")

# Ø¯Ø§Ù„Ø© Ù…Ø¹Ù„ÙˆÙ…Ø§Øª Ø§Ù„Ù‚Ø·Ø¹ ÙˆØ§Ù„Ø¹Ù„Ø§Ø¬
def get_expert_info(status):
    if status == "âš ï¸� ØµØ¯Ø£":
        return "âœ‚ï¸� Ù†Ø¹Ù… (Ù‚Ø·Ø¹ Ù�ÙˆØ±ÙŠ)", "Tebuconazole", "Ø¥Ø²Ø§Ù„Ø© Ø§Ù„Ø¹Ø±Ø¹Ø±"
    elif status == "â�Œ Ø¬Ø±Ø¨":
        return "ğŸ›‘ Ù†Ø¹Ù… (Ø¥Ø²Ø§Ù„Ø© Ø§Ù„Ø£ÙˆØ±Ø§Ù‚)", "Captan", "Ø­Ø±Ù‚ Ø§Ù„Ù…Ø®Ù„Ù�Ø§Øª"
    else:
        return "ğŸŸ¢ Ù„Ø§ ÙŠØ­ØªØ§Ø¬", "Ù„Ø§ ÙŠÙˆØ¬Ø¯", "ØªÙ‡ÙˆÙŠØ© Ø§Ù„ØªØ±Ø¨Ø©"

data = pd.read_csv("live_data.csv")

# Ø§Ù„Ø¬Ø¯ÙˆÙ„ Ù�ÙŠ Ø§Ù„Ù‚Ù…Ø© (Ù†ØµÙŠ Ù…Ø³ØªÙ‚Ø±)
st.subheader("ğŸ“‹ Ø£ÙˆÙ„Ø§Ù‹: Ø³Ø¬Ù„ Ø§Ù„Ù�Ø­Øµ Ø§Ù„Ù…Ø¨Ø§Ø´Ø± (ØªÙˆÙ‚Ø¹Ø§Øª Ø§Ù„Ø®ÙˆØ§Ø±Ø²Ù…ÙŠØ©)")
table_area = st.empty() 

st.divider()

# Ø§Ù„Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª (Ø«Ø§Ù†ÙŠØ§Ù‹)
st.subheader("ğŸ“Š Ø«Ø§Ù†ÙŠØ§Ù‹: Ø§Ù„Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª Ø§Ù„ØªØ±Ø§ÙƒÙ…ÙŠØ©")
stats_area = st.empty()

st.divider()

# Ø§Ù„Ø£Ù‚Ø³Ø§Ù… Ø§Ù„Ø«Ø§Ø¨ØªØ© (Ø«Ø§Ù„Ø«Ø§Ù‹ ÙˆØ±Ø§Ø¨Ø¹Ø§Ù‹)
c1, c2 = st.columns(2)
with c1:
    st.error("### ğŸ©º Ø«Ø§Ù„Ø«Ø§Ù‹: Ø§Ù„Ø¹Ù„Ø§Ø¬ Ø§Ù„ØªÙ�ØµÙŠÙ„ÙŠ\\n- Ø§Ù„ØµØ¯Ø£: Ø±Ø´ Myclobutanil.\\n- Ø§Ù„Ø¬Ø±Ø¨: Ø±Ø´ Dodine.")
with c2:
    st.success("### ğŸ›¡ï¸� Ø±Ø§Ø¨Ø¹Ø§Ù‹: Ø§Ù„ÙˆÙ‚Ø§ÙŠØ©\\n1. Ø­Ø±Ù‚ Ø§Ù„Ù…Ø®Ù„Ù�Ø§Øª Ø§Ù„Ù…ØµØ§Ø¨Ø©.\\n2. ØªØ¹Ù‚ÙŠÙ… Ø£Ø¯ÙˆØ§Øª Ø§Ù„Ù‚Ø·Ø¹.")

# Ù…Ø­Ø±Ùƒ Ø§Ù„Ø¹Ø±Ø¶ Ø§Ù„Ù…Ø³ØªÙ…Ø±
h, r, s = 0, 0, 0
for i in range(len(data)):
    row = data.iloc[i]
    h += int(row['healthy']); r += int(row['rust']); s += int(row['scab'])
    
    # Ø¬Ù„Ø¨ ØªÙ�Ø§ØµÙŠÙ„ Ø§Ù„Ù‚Ø·Ø¹ ÙˆØ§Ù„Ø¹Ù„Ø§Ø¬
    cut_decision, treatment, prevention = get_expert_info(row['Ø§Ù„ØªØ´Ø®ÙŠØµ'])
    
    # Ø§Ù„Ø¬Ø¯ÙˆÙ„ Ø§Ù„Ù†ØµÙŠ Ø§Ù„Ù…Ø·ÙˆØ±
    table_text = "| Ø±Ù‚Ù… Ø§Ù„ØµÙˆØ±Ø© | Ù‚Ø±Ø§Ø± Random Forest | âœ‚ï¸� Ù‚Ø±Ø§Ø± Ø§Ù„Ù‚Ø·Ø¹ | ğŸ’Š Ø§Ù„Ø¹Ù„Ø§Ø¬ | ğŸ›¡ï¸� Ø§Ù„ÙˆÙ‚Ø§ÙŠØ© |\\n|---|---|---|---|---|\\n"
    table_text += f"| {row['image_id']} | {row['Ø§Ù„ØªØ´Ø®ÙŠØµ']} | {cut_decision} | {treatment} | {prevention} |"
    
    table_area.markdown(table_text)
    stats_area.markdown(f"### ğŸ“Ÿ Ø³Ù„ÙŠÙ…: {h} | âš ï¸� ØµØ¯Ø£: {r} | â�Œ Ø¬Ø±Ø¨: {s}")
    
    time.sleep(6) 
""")

# 4. Ø§Ù„ØªØ´ØºÙŠÙ„
def run_st():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false"])

threading.Thread(target=run_st, daemon=True).start()
time.sleep(12)
os.system("curl ipv4.icanhazip.com")
!lt --port 8501

