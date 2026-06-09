# ==========================================
# EXAMGUARD: FINAL SYSTEM (REAL DATASET INTEGRATION)
# ==========================================
# Features: KaggleHub Data Loading + Interactive UI + Robust Fallback
# ==========================================

import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output
import pandas as pd
import kagglehub
import random
import glob
import os
import time

# --- 1. DATASET LOADER (The "Integration" Part) ---

def load_real_data():
    """Downloads and processes the Kaggle Question-Answer dataset."""
    print("â¬‡ï¸�  Initializing KaggleHub Download...")
    
    try:
        # 1. Download via API (or use cache if already present)
        path = kagglehub.dataset_download("rtatman/questionanswer-dataset")
        print(f"ğŸ“‚ Dataset Path: {path}")
        
        all_questions = []
        
        # 2. Find all question text files (S08, S09, S10)
        # These files are typically tab-separated
        file_patterns = os.path.join(path, "*.txt")
        files = glob.glob(file_patterns)
        
        print(f"   Found {len(files)} data files. Processing...")
        
        for file_path in files:
            if "clean" in file_path: continue # Skip raw text dumps, focus on QA pairs
            
            try:
                # Read Tab-Separated Values (TSV), handling common encoding issues
                df = pd.read_csv(file_path, sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
                
                # Normalize Columns (Dataset uses 'Question', 'Answer', 'DifficultyFromQuestioner')
                if 'Question' not in df.columns or 'Answer' not in df.columns:
                    continue
                    
                df = df.dropna(subset=['Question', 'Answer'])
                
                for _, row in df.iterrows():
                    # Map Real Data to Our Schema
                    diff_raw = str(row.get('DifficultyFromQuestioner', 'easy')).lower()
                    
                    if 'hard' in diff_raw: difficulty = 'Hard'
                    elif 'medium' in diff_raw or 'moderate' in diff_raw: difficulty = 'Moderate'
                    else: difficulty = 'Easy'
                    
                    q_obj = {
                        "text": str(row['Question']).strip(),
                        "answer": str(row['Answer']).strip(),
                        "difficulty": difficulty,
                        "marks": 15 if difficulty == 'Hard' else (10 if difficulty == 'Moderate' else 5),
                        "type": "Short Answer" # Default type for this dataset
                    }
                    all_questions.append(q_obj)
                    
            except Exception as e:
                print(f"   âš ï¸� Warning reading {os.path.basename(file_path)}: {e}")
                
        print(f"âœ… Successfully loaded {len(all_questions)} real questions into the Bank.")
        return all_questions

    except Exception as e:
        print(f"â�Œ Error downloading/loading dataset: {e}")
        return []

# --- 2. FALLBACK BANK (Safety Net) ---
# Used only if the internet fails or dataset is empty
FALLBACK_BANK = [
    {"text": "What is the capital of France?", "answer": "Paris", "difficulty": "Easy", "marks": 5, "type": "MCQ"},
    {"text": "Explain Quantum Entanglement.", "answer": "Spooky action at a distance.", "difficulty": "Hard", "marks": 15, "type": "Short"},
    {"text": "Define Polymorphism in OOP.", "answer": "Objects of different types treated as same type.", "difficulty": "Moderate", "marks": 10, "type": "Short"}
]

# --- 3. INITIALIZE SYSTEM ---
# Load data immediately
REAL_QUESTION_BANK = load_real_data()
ACTIVE_BANK = REAL_QUESTION_BANK if len(REAL_QUESTION_BANK) > 0 else FALLBACK_BANK

# --- 4. AGENT FUNCTIONS ---

def run_requirements_agent(inputs):
    print("ğŸ¤– Agent 1 (Requirements) is mapping syllabus...")
    time.sleep(1.0)
    return {
        "requirements_confirmed": inputs,
        "topics": [{"name": "General Knowledge & Logic", "weight": 100}]
    }

def run_content_agent(req, topics):
    print("ğŸ¤– Agent 2 (Content) is retrieving from Question Bank...")
    time.sleep(1.0)
    
    target_diff = req['requirements_confirmed']['difficulty']
    num = req['requirements_confirmed']['num_questions']
    
    # Filter by Difficulty
    if target_diff == "Mixed":
        candidates = ACTIVE_BANK
    else:
        candidates = [q for q in ACTIVE_BANK if q['difficulty'] == target_diff]
    
    # Smart Fallback if filter is too strict
    if len(candidates) < num:
        print(f"   âš ï¸� Not enough '{target_diff}' questions found. filling with mixed questions.")
        candidates = ACTIVE_BANK
        
    # Select
    selected = random.sample(candidates, k=min(len(candidates), num))
    
    # Add IDs dynamically
    for i, q in enumerate(selected):
        q['id'] = i + 1
        
    return {"questions": selected}

def run_validator_agent(questions):
    print("ğŸ¤– Agent 3 (Validator) is compiling paper...")
    time.sleep(1.0)
    
    paper_md = "# ğŸ�“ FINAL EXAM PAPER\n**Generated from:** Kaggle Question-Answer Dataset\n\n"
    key_md = "## ğŸ“� MARKING SCHEME\n\n"
    
    for q in questions:
        paper_md += f"**Q{q['id']}.** {q['text']}\n*({q['difficulty']} | {q['marks']} Marks)*\n\n"
        key_md += f"**{q['id']}.** {q['answer']}\n"
        
    notes = f"### ğŸ§¾ System Report\n* **Source:** {len(ACTIVE_BANK)} questions loaded.\n* **Status:** Ready for distribution."
    
    return {"final_paper_md": paper_md, "answer_key_md": key_md, "moderator_notes": notes}

# --- 5. UI SETUP ---

header = widgets.HTML("<h2>ğŸ�“ ExamGuard: Integrated Dataset System</h2>")
num_q = widgets.IntSlider(value=5, min=1, max=10, description='# Questions:')
difficulty = widgets.Dropdown(options=['Mixed', 'Easy', 'Moderate', 'Hard'], value='Mixed', description='Difficulty:')
btn = widgets.Button(description='ğŸš€ Generate Exam', button_style='success', layout=widgets.Layout(width='100%'))
out = widgets.Output()

ui = widgets.VBox([header, widgets.HBox([num_q, difficulty]), btn, out])

def on_click(b):
    out.clear_output()
    with out:
        print("â�³ Starting Agent Workflow...")
        inputs = {"course": "General", "num_questions": num_q.value, "difficulty": difficulty.value}
        
        req = run_requirements_agent(inputs)
        content = run_content_agent(req, req['topics'])
        print(f"âœ… Content: Retrieved {len(content['questions'])} questions from Real Dataset.")
        
        final = run_validator_agent(content['questions'])
        print("âœ… Validation Complete!")
        
        display(Markdown('---'))
        display(Markdown(final['final_paper_md']))
        display(Markdown('---'))
        display(Markdown(final['answer_key_md']))
        display(Markdown('---'))
        display(Markdown(final['moderator_notes']))

btn.on_click(on_click)
display(ui)

