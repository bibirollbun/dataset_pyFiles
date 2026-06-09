# --- CELL 1: INSTALL DEPENDENCIES ---
!pip install -q streamlit pyngrok google-genai aiosqlite plotly duckduckgo-search nest_asyncio


%%writefile app.py
import streamlit as st
import asyncio
import aiosqlite
import os
import json
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.0-flash-exp" 
client = genai.Client(api_key=API_KEY)
DB_NAME = "skillsprint_v2.db"

# --- DATABASE LAYER ---
async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        # Expanded User Table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                session_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                background TEXT,
                hobby TEXT,
                role TEXT,
                style TEXT,
                current_day INTEGER DEFAULT 1
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS syllabus (
                session_id TEXT,
                day INTEGER,
                topic TEXT,
                content TEXT,
                locked BOOLEAN DEFAULT 1,
                PRIMARY KEY (session_id, day)
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS grades (
                session_id TEXT,
                day INTEGER,
                score INTEGER,
                timestamp TEXT
            )
        ''')
        await db.commit()

async def get_user_profile(session_id):
    async with aiosqlite.connect(DB_NAME) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE session_id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def create_user_profile(data: dict):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            """INSERT OR REPLACE INTO users 
            (session_id, name, age, background, hobby, role, style) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (data['session_id'], data['name'], data['age'], data['background'], 
             data['hobby'], data['role'], data['style'])
        )
        await db.commit()

async def get_syllabus(session_id):
    async with aiosqlite.connect(DB_NAME) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM syllabus WHERE session_id = ? ORDER BY day ASC", (session_id,)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

async def save_syllabus(session_id, syllabus_data: List[Dict]):
    async with aiosqlite.connect(DB_NAME) as db:
        for day_plan in syllabus_data:
            await db.execute(
                "INSERT OR REPLACE INTO syllabus (session_id, day, topic, content, locked) VALUES (?, ?, ?, ?, ?)",
                (session_id, day_plan['day'], day_plan['topic'], day_plan['content'], day_plan['locked'])
            )
        await db.commit()

async def get_grades(session_id):
    async with aiosqlite.connect(DB_NAME) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM grades WHERE session_id = ? ORDER BY timestamp ASC", (session_id,)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

async def save_grade(session_id, day, score):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("INSERT INTO grades (session_id, day, score, timestamp) VALUES (?, ?, ?, ?)",
                         (session_id, day, score, datetime.now().isoformat()))
        if score >= 70:
            await db.execute("UPDATE syllabus SET locked = 0 WHERE session_id = ? AND day = ?", (session_id, day + 1))
            await db.execute("UPDATE users SET current_day = ? WHERE session_id = ?", (day + 1, session_id))
        await db.commit()

# --- AGENT TOOLS ---
async def parallel_research(topic: str):
    """Simulates the Parallel Agent Architecture"""
    def search_sync(q): return DDGS().text(q, max_results=2)
    
    # In a real app, these would be separate async API calls
    # For Kaggle demo, we simulate async gather with a slight blocking call (DDGS is sync)
    doc_res = search_sync(f"{topic} official documentation whitepaper")
    trend_res = search_sync(f"{topic} future trends 2025 AI agents")
    
    summary = f"--- OFFICIAL DOCS ---\n{str(doc_res)}\n\n--- TRENDS ---\n{str(trend_res)}"
    return summary

PROMPTS = {
    "ARCHITECT": """You are the Curriculum Architect.
    Input: Research summary + User Profile (Background, Hobby).
    Output: A 3-Day JSON syllabus.
    Personalize the *Topic Titles* to fit their hobby/background.
    Format: [{"day": 1, "topic": "...", "content": "...", "locked": false}, {"day": 2, "topic": "...", "content": "...", "locked": true}, ...]""",
    
    "DEAN": """Route the user.
    If they are asking a question -> 'FACULTY'.
    If they ask to be tested/quizzed -> 'EXAMINER'.
    Otherwise -> 'CHAT'.
    Return JSON: {"target": "FACULTY" | "EXAMINER" | "CHAT"}""",
    
    "FACULTY": """You are the Expert.
    User Style: {style}. User Hobby: {hobby}.
    Explain the concept using analogies related to their Hobby.
    CRITICAL: If appropriate, generate Python code using `plotly.graph_objects` to visualize it. Wrap in ```python.
    """,
    
    "EXAMINER": """Generate a hard scenario question.
    If user answers, Grade (0-100).
    Return JSON: {"content": "feedback", "grade": int | null}"""
}

# --- UI COMPONENTS ---

def render_plotly_from_code(code_str):
    try:
        local_scope = {}
        exec(code_str, globals(), local_scope)
        for var in local_scope.values():
            if isinstance(var, go.Figure):
                st.plotly_chart(var, use_container_width=True)
                return
    except Exception as e:
        st.error(f"Visualization Error: {e}")

def sidebar_profile(profile):
    with st.sidebar:
        st.header("ğŸ‘¤ Student ID")
        if profile:
            st.write(f"**Name:** {profile['name']}")
            st.write(f"**Role:** {profile['role']}")
            st.write(f"**Style:** {profile['style']}")
            
            # Progress Bar
            st.divider()
            day = profile['current_day']
            progress = min(1.0, (day - 1) / 3)
            st.write(f"**Course Progress: Day {day}/3**")
            st.progress(progress)
            
            if st.button("Reset All Data", type="primary"):
                return True
    return False

# --- MAIN APP ---
async def main():
    st.set_page_config(page_title="SkillSprint Pro", layout="wide", page_icon="ğŸ�“")
    await init_db()
    
    # Session Management
    session_id = "user_v2_demo"
    profile = await get_user_profile(session_id)
    
    # --- RESET LOGIC ---
    if profile:
        if sidebar_profile(profile):
            async with aiosqlite.connect(DB_NAME) as db:
                await db.execute("DELETE FROM users WHERE session_id=?", (session_id,))
                await db.execute("DELETE FROM syllabus WHERE session_id=?", (session_id,))
                await db.execute("DELETE FROM grades WHERE session_id=?", (session_id,))
                await db.commit()
            st.rerun()

    # --- VIEW 1: ONBOARDING (No Profile) ---
    if not profile:
        st.title("ğŸš€ Welcome to SkillSprint")
        st.markdown("### The Autonomous AI University")
        
        with st.form("onboarding_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name")
                age = st.number_input("Age", min_value=10, max_value=100, value=25)
                role = st.selectbox("Current Role", ["Student", "Developer", "Designer", "Manager", "Researcher"])
            with col2:
                hobby = st.text_input("Favorite Hobby (for analogies)", placeholder="e.g., Chess, Cooking, Football")
                style = st.select_slider("Learning Style", options=["Visual (Charts)", "Auditory (Story)", "Kinesthetic (Code)"])
                background = st.text_area("Brief Background", placeholder="I know Python but struggle with AI agents...")
            
            submitted = st.form_submit_button("Generate My Curriculum")
            
            if submitted and name and hobby:
                user_data = {
                    "session_id": session_id, "name": name, "age": age,
                    "role": role, "hobby": hobby, "style": style, "background": background
                }
                await create_user_profile(user_data)
                st.toast("Profile Created! Designing Curriculum...", icon="ğŸ¤–")
                st.rerun()

    # --- VIEW 2: DASHBOARD (Profile Exists) ---
    else:
        st.title(f"ğŸ�“ {profile['name']}'s University")
        
        # Tabs for Layout
        tab_map, tab_class, tab_stats = st.tabs(["ğŸ—ºï¸� Roadmap", "ğŸ¤– Classroom", "ğŸ“Š Analytics"])
        
        syllabus = await get_syllabus(session_id)

        # --- TAB 1: ROADMAP ---
        with tab_map:
            if not syllabus:
                st.info("Curriculum Architect is designing your course... (Please type a topic in Classroom)")
            else:
                st.subheader("Your 3-Day Sprint")
                col_days = st.columns(3)
                for i, day in enumerate(syllabus):
                    with col_days[i]:
                        status = "ğŸ”’ LOCKED" if day['locked'] else "âœ… OPEN"
                        color = "grey" if day['locked'] else "green"
                        with st.container(border=True):
                            st.markdown(f"### Day {day['day']}")
                            st.markdown(f"**:{color}[{status}]**")
                            st.markdown(f"**{day['topic']}**")
                            st.caption(day['content'])
                            if day['locked']:
                                st.button(f"Start Day {day['day']}", disabled=True, key=f"btn_{i}")
                            else:
                                if st.button(f"Enter Class", key=f"btn_{i}"):
                                    # Logic to jump to classroom could go here
                                    st.toast("Go to the Classroom tab to begin!", icon="Tm")

        # --- TAB 2: CLASSROOM (Chat) ---
        with tab_class:
            # Syllabus Generator Trigger
            if not syllabus:
                st.warning("No curriculum found. Tell the Architect what you want to learn.")
                
            # Chat History
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("code"): render_plotly_from_code(msg["code"])
            
            # Input Area
            if prompt := st.chat_input("Ask a question, request a syllabus, or take a test..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                
                # --- ORCHESTRATION LOGIC ---
                # 1. If No Syllabus -> Route to ARCHITECT
                if not syllabus:
                    with st.status("Architect: Researching...", expanded=True):
                        research = await parallel_research(prompt)
                    
                    sys_prompt = PROMPTS["ARCHITECT"] + f"\nUser: {profile}"
                    full_prompt = f"Research: {research}\n\nCreate a syllabus for: {prompt}"
                    
                    res = client.models.generate_content(
                        model=MODEL_ID, 
                        contents=[sys_prompt, full_prompt],
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    plan = json.loads(res.text)
                    await save_syllabus(session_id, plan)
                    response_text = "I've built your roadmap based on your profile! Check the **Roadmap Tab**."
                    code_block = None

                # 2. If Syllabus Exists -> Dean Routes
                else:
                    dean_res = client.models.generate_content(
                        model=MODEL_ID,
                        contents=[PROMPTS["DEAN"], prompt],
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    route = json.loads(dean_res.text).get("target")
                    
                    if route == "EXAMINER":
                        # Examiner Logic
                        sys = PROMPTS["EXAMINER"]
                        res = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[sys, prompt],
                            config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        data = json.loads(res.text)
                        response_text = data.get("content")
                        code_block = None
                        if data.get("grade"):
                            await save_grade(session_id, profile['current_day'], data['grade'])
                            st.toast(f"Grade Recorded: {data['grade']}%", icon="ğŸ“�")

                    else: # FACULTY or CHAT
                        # Faculty Logic
                        sys = PROMPTS["FACULTY"].format(style=profile['style'], hobby=profile['hobby'])
                        res = client.models.generate_content(model=MODEL_ID, contents=[sys, prompt])
                        response_text = res.text
                        
                        # Extract Code
                        code_block = None
                        if "```python" in response_text:
                            match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
                            if match: code_block = match.group(1)

                # Response
                st.session_state.messages.append({"role": "assistant", "content": response_text, "code": code_block})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                    if code_block: render_plotly_from_code(code_block)
                    
        # --- TAB 3: ANALYTICS ---
        with tab_stats:
            st.subheader("Performance Analytics")
            grades = await get_grades(session_id)
            
            if grades:
                df = pd.DataFrame(grades)
                
                # 1. Score History Line Chart
                fig = px.line(df, x='timestamp', y='score', markers=True, title='Mastery Trajectory')
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Stat Cards
                avg_score = df['score'].mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Average Score", f"{avg_score:.1f}%")
                c2.metric("Tests Taken", len(df))
                c3.metric("Current Level", f"Day {profile['current_day']}")
            else:
                st.info("No grades recorded yet. Ask the Examiner to 'Test me' in the Classroom.")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())


# --- CELL 3: RUN NGROK AND STREAMLIT ---
import os
from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient

# 1. Get Secrets
user_secrets = UserSecretsClient()
try:
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    NGROK_TOKEN = user_secrets.get_secret("NGROK_TOKEN")
    
    # Set ENV for the subprocess
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    print("ERROR: Please set 'GOOGLE_API_KEY' and 'NGROK_TOKEN' in Kaggle Add-ons -> Secrets")

# 2. Authenticate Ngrok
ngrok.set_auth_token(NGROK_TOKEN)

# 3. Start the Tunnel
# Kill previous tunnels if running re-entry
ngrok.kill()
public_url = ngrok.connect(8501).public_url
print(f"ğŸš€ SkillSprint is live at: {public_url}")

# 4. Run Streamlit in Background
!streamlit run app.py > /dev/null

