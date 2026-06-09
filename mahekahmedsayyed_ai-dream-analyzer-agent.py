!pip install google-generativeai supabase python-dotenv


from getpass import getpass

GEMINI_API_KEY = getpass("Enter your Gemini API Key: ")
SUPABASE_URL = getpass("Enter your Supabase URL: ") 
SUPABASE_KEY = getpass("Enter your Supabase API Key: ")

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)


from supabase import create_client, Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_dream_input():
    print("Write your dream:")
    dream = input()
    return dream.strip()

# Clean dream text
def clean_dream(text):
    return " ".join(text.split())


import json

def extract_symbols(dream_text):
    prompt = f"""
    You are a Dream Symbol Extraction Agent.

    Extract symbols ONLY from this dream.

    Dream: {dream_text}

    Return STRICT JSON ONLY.
    No explanation.
    No extra text.
    No commentary.
    No backticks.

    The ONLY valid output:
    {{
      "people": [],
      "objects": [],
      "animals": [],
      "places": [],
      "actions": [],
      "other_symbols": []
    }}
    """

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)

    # HARD clean: remove code fences and random text
    cleaned = response.text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # If JSON is NOT first, extract only the JSON part
    if "{" in cleaned:
        cleaned = cleaned[cleaned.index("{") : ]
    if "}" in cleaned:
        cleaned = cleaned[: cleaned.rindex("}") + 1]

    return cleaned

def parse_json_output(text):
    try:
        return json.loads(text)
    except:
        print("JSON error â€” printing raw output:")
        print(text)
        return None




def detect_emotions(dream_text):
    prompt = f"""
    You are an Emotion Detection Agent.

    Analyze the dream and detect:
    - dominant emotion
    - secondary emotions
    - intensity (1â€“10 scale)

    Dream: {dream_text}

    Return JSON only:
    {{
      "dominant": "",
      "secondary": [],
      "intensity": 0
    }}
    """

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    output = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(output)



def interpret_dream(dream_text, symbols, emotions):
    prompt = f"""
    You are a Dream Interpretation Agent.

    Use:
    - Dream content
    - Extracted symbols: {symbols}
    - Detected emotions: {emotions}

    Provide:
    1) Two short interpretations
    2) One self-reflection question

    Format:
    Interpretation 1: ...
    Interpretation 2: ...
    Reflection Question: ...
    """

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()



def create_embedding(text):
    result = genai.embed_content(model="models/text-embedding-004", content=text)
    return result["embedding"]



def save_to_supabase(dream_text, symbols, emotions, interpretation, embedding):
    data = {
        "dream_text": dream_text,
        "symbols": symbols,
        "emotions": emotions,
        "interpretation": interpretation,
        "embedding": embedding
    }

    response = supabase.table("dreams").insert(data).execute()
    return response



def run_full_pipeline():
    dream = get_dream_input()
    cleaned = clean_dream(dream)

    # 1. Extract symbols
    symbols = parse_json_output(extract_symbols(cleaned))

    # 2. Detect emotions
    emotions = detect_emotions(cleaned)

    # 3. Interpretation
    interpretation = interpret_dream(cleaned, symbols, emotions)

    # 4. Embedding
    embedding = create_embedding(cleaned)

    # 5. Save to Supabase
    save_to_supabase(cleaned, symbols, emotions, interpretation, embedding)

    print("\nğŸ�‰ DREAM ANALYZED & SAVED SUCCESSFULLY!")
    print("Symbols:", symbols)
    print("Emotions:", emotions)
    print("Interpretation:", interpretation)


run_full_pipeline()


def embed_query(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result["embedding"]



def fetch_similar_dreams(query_embedding, match_count=3):
    response = supabase.rpc(
        "match_dreams",
        {
            "query_embedding": query_embedding,
            "match_count": match_count
        }
    ).execute()

    return response.data



def build_context(dream_rows):
    context = ""
    for d in dream_rows:
        context += f"""
Dream: {d['dream_text']}
Symbols: {d['symbols']}
Emotions: {d['emotions']}
Interpretation: {d['interpretation']}
---
"""
    return context



def chat_about_dream(query):
    query_emb = embed_query(query)
    dream_rows = fetch_similar_dreams(query_emb, 3)
    context = build_context(dream_rows)

    prompt = f"""
    You are a Dream Chat Agent.

    User question: {query}
    Dream memories:
    {context}

    Give:
    - one meaning
    - one emotional insight
    - one reflection question
    """

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text




# Chat Loop 
def chat_loop():
    print("Dream Chat Agent is ready! Type 'exit' to stop.\n")
    
    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break

        answer = chat_about_dream(query)
        print("\nAI:", answer, "\n")

chat_loop()




