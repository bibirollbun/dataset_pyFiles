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


!pip install -q google-generativeai sentence-transformers faiss-cpu pandas requests python-dotenv



import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from typing import List, Dict

import google.generativeai as genai

# ðŸ”¥ Your working Gemini model
GEMINI_MODEL = "gemini-2.0-flash"

# ðŸ”¥ Paste your real key here
genai.configure(api_key="AIzaSyAtJp2oGI-JIGtLwKrIsS94m8H_Z9nvXMA")



def generate_from_gemini(prompt, max_tokens=100):
    try:
        # your Gemini API call here
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return None





def local_caption_fallback(title):
    caption = f"{title} â€” quick, tasty and perfect for your next meal! ðŸ”¥"
    hashtags = "#EasyRecipe #Paneer #IndianFood #CookingShorts #VegDelight"
    cta = "Try it out today!"
    return f"CAPTION: {caption}\nHASHTAGS: {hashtags}\nCTA: {cta}"



csv_text = """title,cuisine,time_min,ingredients,steps,tags
Masala Egg Rice,Indian,20,"rice,eggs,onion,tomato,green chili,oil,spices","1. Cook rice. 2. Saute onion/tomato/spices. 3. Scramble eggs. 4. Mix rice and masala","quick;easy;egg"
Pasta Aglio e Olio,Italian,15,"pasta,garlic,chili flakes,olive oil,parsley","1. Boil pasta. 2. Saute garlic in oil. 3. Toss pasta with oil and parsley","vegetarian;quick"
Paneer Tikka Wrap,Indian,30,"paneer,yogurt,spices,wraps,pepper,onion","1. Marinate paneer. 2. Grill. 3. Assemble wrap","vegetarian;travel-friendly"
"""
open("recipes.csv","w").write(csv_text)
RECIPES = pd.read_csv("recipes.csv")
RECIPES



embed_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = (RECIPES['title'] + " | " + RECIPES['ingredients'] + " | " + RECIPES['tags']).tolist()
embeddings = embed_model.encode(texts, convert_to_numpy=True)

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

meta = RECIPES.to_dict(orient="records")

print("FAISS index built:", index.ntotal)



def faiss_search(query, k=1):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(meta):
            results.append(meta[idx])
    return results



def local_script_fallback(title):
    return f"""
INTRO:
Quick and tasty {title} youâ€™ll love!

SHOT 1:
Show ingredients on screen.

SHOT 2:
Cook paneer until golden and spicy.

SHOT 3:
Wrap it up and serve hot.

CTA:
Save & follow for more quick recipes!
"""


def generate_script(title, context):
    prompt = f"""Write a short 60-second cooking video script for:
Recipe: {title}
Context: {context}

Include:
- Intro
- 3 shots (actions + voiceover)
- Closing CTA
"""
    out = generate_from_gemini(prompt, max_tokens=200)
    return out or local_script_fallback(title)



def generate_caption_or_fallback(title, context):
    prompt = f"""Write IG caption for {title}, max 125 chars, 5 hashtags, 1 CTA."""
    out = generate_from_gemini(prompt, max_tokens=60)
    return out or local_caption_fallback(title)



def local_shoot_plan_template(recipe):
    return f"""
SHOT 1 â€“ INGREDIENTS
- Show all ingredients for {recipe['title']}
- Overhead flat-lay shot

SHOT 2 â€“ COOKING
- Pan cooking or grilling paneer
- Close-up sizzling shots

SHOT 3 â€“ FINAL PLATE
- Assemble and serve
- Final beauty shot + smile
"""


def generate_shoot_plan(title, recipe):
    prompt = f"""Plan 3-shot filming checklist for {title}."""
    out = generate_from_gemini(prompt, max_tokens=120)
    return out or local_shoot_plan_template(recipe)



def run_query(query):
    print("=== Query ===")
    print(query)

    results = faiss_search(query)
    recipe = results[0]

    print("\n--- RECIPE ---")
    print(recipe)

    ctx = f"{recipe['ingredients']} | {recipe['steps']} | {recipe['tags']}"

    print("\n--- SCRIPT ---")
    print(generate_script(recipe['title'], ctx))

    print("\n--- CAPTION ---")
    print(generate_caption_or_fallback(recipe['title'], ctx))

    print("\n--- SHOOT PLAN ---")
    print(generate_shoot_plan(recipe['title']))



def run_query(query):
    print("=== Query ===")
    print(query)

    results = faiss_search(query)
    recipe = results[0]

    print("\n--- RECIPE ---")
    print(recipe)

    ctx = f"{recipe['ingredients']} | {recipe['steps']} | {recipe['tags']}"

    print("\n--- SCRIPT ---")
    print(generate_script(recipe['title'], ctx))

    print("\n--- CAPTION ---")
    print(generate_caption_or_fallback(recipe['title'], ctx))

    print("\n--- SHOOT PLAN ---")
    print(generate_shoot_plan(recipe['title'], recipe))




run_query("Make a quick paneer recipe for a 60-second Instagram video")


