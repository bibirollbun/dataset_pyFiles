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


# Setup & imports
# Paste this into a code cell and run.

# If any package is missing on Kaggle, uncomment the pip installs.
# Kaggle typically has pandas & scikit-learn preinstalled.

# !pip install pandas scikit-learn

import json
import math
from datetime import datetime, timedelta
from textwrap import wrap
from typing import List, Dict, Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Imports ready. PathForge Agent initialized.")



# Seed skill ontology and resource index (small sample)
skill_ontology = {
    "Data Science": {
        "modules": [
            {"id": "ds_py", "title": "Python for Data Analysis", "outcomes": ["python", "pandas", "numpy"]},
            {"id": "ds_stats", "title": "Probability & Statistics", "outcomes": ["probability", "statistics"]},
            {"id": "ds_sql", "title": "SQL for Analytics", "outcomes": ["sql", "databases"]},
            {"id": "ds_ml", "title": "Supervised ML", "outcomes": ["regression", "classification", "sklearn"]},
            {"id": "ds_prod", "title": "Model Deployment & MLOps", "outcomes": ["api", "docker", "fastapi"]},
            {"id": "ds_capstone", "title": "Capstone Project", "outcomes": ["end-to-end project", "portfolio"]},
        ]
    },
    "Frontend Development": {
        "modules": [
            {"id": "fe_html", "title": "HTML & CSS Fundamentals", "outcomes": ["html", "css"]},
            {"id": "fe_js", "title": "Modern JavaScript (ES6+)", "outcomes": ["javascript", "dom", "es6"]},
            {"id": "fe_react", "title": "React & State Management", "outcomes": ["react", "hooks", "redux"]},
            {"id": "fe_build", "title": "Build Tools & Deployment", "outcomes": ["webpack", "netlify", "vite"]},
            {"id": "fe_project", "title": "Portfolio Project", "outcomes": ["portfolio", "deployed app"]}
        ]
    }
}

# Resource library: manually seeded small dataset (expandable)
resources = [
    {"id": "r1", "title": "Python for Everybody (Coursera)", "type": "course", "url": "https://coursera.org/python",
     "tags": ["python", "beginner"], "duration_hours": 40, "cost": "free", "level": "Beginner"},
    {"id": "r2", "title": "Kaggle: Pandas Microcourse", "type": "course", "url": "https://kaggle.com/pandas",
     "tags": ["pandas", "python", "hands-on"], "duration_hours": 8, "cost": "free", "level": "Beginner"},
    {"id": "r3", "title": "Statistical Learning (lecture notes)", "type": "book", "url": "https://example/stat",
     "tags": ["statistics", "probability"], "duration_hours": 20, "cost": "free", "level": "Intermediate"},
    {"id": "r4", "title": "Hands-On ML with Scikit-Learn, Keras, and TensorFlow (book)", "type": "book",
     "url": "https://example/hands-on-ml", "tags": ["ml", "sklearn", "deep-learning"], "duration_hours": 50,
     "cost": "paid", "level": "Intermediate"},
    {"id": "r5", "title": "SQL for Data Analysts (Mode)", "type": "course", "url": "https://mode.com/sql",
     "tags": ["sql", "databases"], "duration_hours": 12, "cost": "free", "level": "Beginner"},
    {"id": "r6", "title": "Deploy ML Models with FastAPI (tutorial)", "type": "tutorial", "url": "https://example/fastapi",
     "tags": ["fastapi", "api", "deployment"], "duration_hours": 6, "cost": "free", "level": "Intermediate"},
    {"id": "r7", "title": "Frontend: Modern JS (free video course)", "type": "course", "url": "https://example/js",
     "tags": ["javascript", "dom"], "duration_hours": 15, "cost": "free", "level": "Beginner"},
    {"id": "r8", "title": "React Official Tutorial", "type": "tutorial", "url": "https://reactjs.org/tutorial",
     "tags": ["react", "hooks"], "duration_hours": 10, "cost": "free", "level": "Beginner"},
]

# Convert resource list to DataFrame for easy display & export
resources_df = pd.DataFrame(resources)
display(resources_df)



from typing import List, Dict, Any
print("Typing imports loaded.")



# Intake function & example profile

def create_user_profile(name: str,
                        target_field: str,
                        current_level: str = "Beginner",
                        hours_per_week: int = 10,
                        deadline_weeks: int = 24,
                        preferred_format: List[str] = None,
                        budget: str = "low") -> Dict[str, Any]:
    return {
        "name": name,
        "target_field": target_field,
        "current_level": current_level,
        "hours_per_week": hours_per_week,
        "deadline_weeks": deadline_weeks,
        "preferred_format": preferred_format or ["hands-on", "video", "course"],
        "budget": budget
    }

# Example user
user = create_user_profile(name="Aisha", target_field="Data Science", hours_per_week=12, deadline_weeks=36, preferred_format=["hands-on", "project"])
user



# Matching engine - minimalist approach using TF-IDF over resource tags+title+type
# This is an MVP ranking: expand later with embeddings & quality metrics.

def resource_text_repr(res):
    # Create a short textual representation used for vector matching
    tags = " ".join(res.get("tags", []))
    return f"{res['title']} {res['type']} {tags} {res.get('level','')}"
    
# Build TF-IDF matrix for resources
corpus = [resource_text_repr(r) for r in resources]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(corpus)

def rank_resources_for_module(module: Dict[str, Any], user_profile: Dict[str, Any], top_k=5):
    # Compose query text from module outcomes + user preferences
    query_text = " ".join(module.get("outcomes", []))
    # Boost with user's format preference keywords
    query_text += " " + " ".join(user_profile.get("preferred_format", []))
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    
    scored = []
    for idx, score in enumerate(sims):
        r = resources[idx].copy()
        # Simple filters for budget & level: penalize heavy paid or mismatched level
        level_penalty = 0
        if user_profile["current_level"].lower() == "beginner" and r["level"].lower() == "intermediate":
            level_penalty += 0.05
        budget_penalty = 0
        if user_profile.get("budget","low")=="low" and r["cost"]=="paid":
            budget_penalty += 0.03
        final_score = float(score) - level_penalty - budget_penalty
        scored.append((final_score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# Test on one module
module_example = skill_ontology["Data Science"]["modules"][0]  # Python for Data Analysis
ranked = rank_resources_for_module(module_example, user, top_k=3)
ranked



# Roadmap generator (phased: Foundations -> Core -> Advanced -> Capstone)
# Strategy: map modules in ontology order into weeks based on user hours and estimated durations.

def estimate_module_hours(module_id):
    # Simple heuristic: deduce from module id or set default
    # Could be augmented by matching resources' durations.
    default_hours = {
        "foundations": 60,
        "core": 120,
        "advanced": 100,
        "capstone": 40
    }
    return 30  # MVP: 30 hours per module placeholder

def generate_roadmap(user_profile, ontology, resources_lookup, weeks_buffer=2):
    field = user_profile["target_field"]
    modules = ontology.get(field, {}).get("modules", [])
    roadmap = []
    # distribute modules across the user's deadline (weeks)
    total_weeks = user_profile["deadline_weeks"]
    # simple distribution: equal weeks per module
    if not modules:
        raise ValueError("No modules found for field: " + field)
    weeks_per_module = max(1, total_weeks // len(modules))
    start = datetime.utcnow().date()
    for i, m in enumerate(modules):
        start_week = start + timedelta(weeks=i*weeks_per_module)
        end_week = start_week + timedelta(weeks=weeks_per_module-1)
        # get top resources for module
        ranked = rank_resources_for_module(m, user_profile, top_k=3)
        roadmap.append({
            "module_id": m["id"],
            "module_title": m["title"],
            "start_week": str(start_week),
            "end_week": str(end_week),
            "top_resources": [r for score, r in ranked]
        })
    return roadmap

roadmap = generate_roadmap(user, skill_ontology, resources_df)
pd.DataFrame([
    {"module_id": r["module_id"], "title": r["module_title"], "start": r["start_week"], "end": r["end_week"],
     "top_resource_titles": "; ".join([t["title"] for t in r["top_resources"]])}
    for r in roadmap
])



# Export roadmap and ranked resources into CSV files for submission / download.

def roadmap_to_df(roadmap):
    rows = []
    for r in roadmap:
        for i, res in enumerate(r["top_resources"], start=1):
            rows.append({
                "module_id": r["module_id"],
                "module_title": r["module_title"],
                "start_week": r["start_week"],
                "end_week": r["end_week"],
                "rank": i,
                "resource_id": res["id"],
                "resource_title": res["title"],
                "resource_type": res["type"],
                "resource_url": res["url"],
                "resource_tags": ",".join(res["tags"]),
                "resource_duration_hours": res["duration_hours"],
                "resource_cost": res["cost"],
                "resource_level": res["level"]
            })
    return pd.DataFrame(rows)

roadmap_df = roadmap_to_df(roadmap)
roadmap_df.to_csv("pathforge_roadmap_resources.csv", index=False)
resources_df.to_csv("pathforge_resource_index.csv", index=False)
print("Saved: pathforge_roadmap_resources.csv and pathforge_resource_index.csv")
display(roadmap_df.head(50))



# Minimal interactive loop (text) to try other user profiles in notebook

def run_simulation_for_profile(profile):
    print(f"\n=== Generating roadmap for {profile['name']} â†’ {profile['target_field']} ===")
    rm = generate_roadmap(profile, skill_ontology, resources_df)
    df = roadmap_to_df(rm)
    display(df)
    print("Exported CSVs present in notebook file browser.")

# Example: try another profile quickly
user2 = create_user_profile("Ravi", "Frontend Development", hours_per_week=8, deadline_weeks=16, preferred_format=["video", "project"])
run_simulation_for_profile(user2)



# ================================
# FINAL OUTPUT CELL â€” PATHFORGE AGENT
# ================================

print("====================================")
print("        FINAL OUTPUT SUMMARY        ")
print("====================================\n")

# 1. USER PROFILE
print("=== 1) USER PROFILE ===")
try:
    print(user)
except:
    print("User profile not found. Make sure the create_user_profile cell has been run.")

print("\n====================================\n")

# 2. ROADMAP DATAFRAME
print("=== 2) ROADMAP (Full Table) ===")
try:
    display(roadmap_df)
except:
    print("Roadmap DataFrame not found. Run the roadmap generator cell first.")

print("\n====================================\n")

# 3. RESOURCE INDEX DATAFRAME
print("=== 3) RESOURCE INDEX ===")
try:
    display(resources_df)
except:
    print("Resources DataFrame not found. Run the resources cell first.")

print("\n====================================\n")

# 4. CSV PREVIEWS
print("=== 4) CSV OUTPUT PREVIEW ===")

import os
working_dir = "/kaggle/working"

csv_paths = [
    "pathforge_roadmap_resources.csv",
    "pathforge_resource_index.csv"
]

for csv_name in csv_paths:
    csv_file = os.path.join(working_dir, csv_name)
    print(f"\n--- Preview of {csv_name} ---")
    if os.path.exists(csv_file):
        try:
            preview = pd.read_csv(csv_file).head(10)
            display(preview)
        except Exception as e:
            print(f"Could not read {csv_name}: {e}")
    else:
        print(f"{csv_name} not found in /kaggle/working.")

print("\n====================================\n")

# 5. LIST FILES IN WORKING DIRECTORY
print("=== 5) FILES IN /kaggle/working ===\n")
!ls -lha /kaggle/working

print("\n====================================")
print("        END OF FINAL OUTPUT         ")
print("====================================")


