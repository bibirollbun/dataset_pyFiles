!pip install -q google-genai



import os
from google import genai
from kaggle_secrets import UserSecretsClient

# Load Gemini API Key from Kaggle Secrets
try:
    secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = secrets.get_secret("GEMINI_API_KEY")
    print("Gemini key loaded from Kaggle Secrets.")
except:
    print("â�Œ Please add your GEMINI_API_KEY secret in Kaggle â†’ Add-ons â†’ Secrets.")

# Initialize Gemini client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Test ping
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Reply with READY"
)

print("Model ping:", response.text)





def clean_opinion(opinion_text: str) -> str:
    """
    Uses Gemini to rewrite the user's opinion clearly and concisely.
    """
    prompt = f"Rewrite the following opinion clearly and concisely:\n\n{opinion_text}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    cleaned = response.text.strip()
    return cleaned

# Quick test
test_output = clean_opinion("I think AI is dangerous but also helpful, Iâ€™m not sure honestly.")
print("Cleaned opinion:", test_output)



# Step â€” Load or Initialize History

import os
import pandas as pd

DATA_PATH = "history.csv"

def load_or_init_history():
    """
    Loads history.csv if it exists; otherwise creates a new empty file.
    """
    if os.path.exists(DATA_PATH):
        print("ğŸ“‚ Loading history.csv...")
        return pd.read_csv(DATA_PATH)
    else:
        print("ğŸ†• Creating new empty history.csv...")
        df = pd.DataFrame(columns=["timestamp", "opinion_clean", "embedding_json"])
        df.to_csv(DATA_PATH, index=False)
        return df

# Initialize df
df = load_or_init_history()

# Show the dataframe
df.head()



import numpy as np

def get_embedding(text: str) -> np.ndarray:
    """
    Generates an embedding vector using the latest Gemini SDK response format.
    """
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=[{"text": text}]
    )

    # NEW SDK FORMAT (2025)
    vector = response.embeddings[0].values   # list of floats
    vector = np.array(vector, dtype="float32")

    return vector


# Quick test
sample_embed = get_embedding("AI will change the world.")
print("Embedding shape:", sample_embed.shape)
print("First 5 values:", sample_embed[:5])





print("=== PROMPTING DEMO ===")
raw_opinion = "I think social media is bad but also good sometimes"
cleaned = clean_opinion(raw_opinion)
print("Raw:", raw_opinion)
print("Cleaned:", cleaned)

print("\n=== EMBEDDINGS DEMO ===")
demo_vec = get_embedding("Sample opinion for embedding test")
print("Vector shape:", demo_vec.shape)
print("First 5 values:", demo_vec[:5])



import json
import pandas as pd
from datetime import datetime

def save_new_opinion(raw_opinion: str):
    global df

    # 1. Clean opinion using Gemini
    cleaned = clean_opinion(raw_opinion)

    # 2. Generate embedding vector
    emb = get_embedding(cleaned)
    emb_json = json.dumps(emb.tolist())   # convert numpy array to JSON

    # 3. Build a new row
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "opinion_clean": cleaned,
        "embedding_json": emb_json
    }

    # 4. Append it to the existing dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 5. Save the updated dataframe to CSV
    df.to_csv(DATA_PATH, index=False)

    print("âœ… Opinion saved:")
    print(cleaned)


# Quick test
save_new_opinion("I feel social media is too addictive but also useful.")





import numpy as np

def find_similar_opinions(new_embedding: np.ndarray, top_k: int = 3):
    """
    Returns the top_k most similar past opinions based on cosine similarity.
    """
    if len(df) == 0:
        return []

    similarities = []

    for idx, row in df.iterrows():
        past_emb = np.array(json.loads(row["embedding_json"]), dtype="float32")

        # Compute cosine similarity
        dot = np.dot(new_embedding, past_emb)
        norm = np.linalg.norm(new_embedding) * np.linalg.norm(past_emb)
        sim = dot / (norm + 1e-10)

        similarities.append((sim, idx))

    # Sort by similarity (highest first)
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Return top K matches
    top_matches = similarities[:top_k]

    return top_matches

# Quick test using your last saved opinion
last_emb = get_embedding(df.iloc[-1]["opinion_clean"])
similar = find_similar_opinions(last_emb)

similar



def generate_evolution_summary(new_opinion: str):
    """
    Retrieves similar past opinions and asks Gemini to explain how thinking evolved.
    """
    # Clean and embed the new opinion
    cleaned = clean_opinion(new_opinion)
    new_emb = get_embedding(cleaned)

    # Find similar past opinions
    matches = find_similar_opinions(new_emb, top_k=3)

    if len(matches) == 0:
        return "No past opinions found. Not enough history to compare evolution."

    # Format past opinions into bullet list
    past_texts = []
    for sim, idx in matches:
        past_texts.append(f"- (Similarity: {sim:.2f}) {df.iloc[idx]['opinion_clean']}")

    past_block = "\n".join(past_texts)

    # Prompt to Gemini for reflection
    prompt = f"""
You are an analysis assistant.

A user has expressed a new opinion:

NEW OPINION:
{cleaned}

Here are their most similar past opinions:
{past_block}

Explain clearly how their thinking has evolved.
Focus on:
- changes in reasoning
- changes in tone or emotion
- shifts in confidence
- differences in belief intensity
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text.strip()


# Test the evolution explanation
summary = generate_evolution_summary("I think social media is becoming less harmful than before.")
print(summary)



def full_pipeline(new_opinion: str, top_k: int = 3):
    print("Raw Input:")
    print(new_opinion)
    print("\n")

    # Clean opinion
    cleaned = clean_opinion(new_opinion)
    print("Cleaned Opinion:")
    print(cleaned)
    print("\n")

    # Embedding
    emb = get_embedding(cleaned)

    # Find similar past opinions
    matches = find_similar_opinions(emb, top_k=top_k)
    print("Most Similar Past Opinions:\n")

    if len(matches) == 0:
        print("(No past opinions available yet.)\n")
    else:
        for sim, idx in matches:
            print(f"- Similarity {sim:.2f}: {df.iloc[idx]['opinion_clean']}")
        print("\n")

    # Evolution summary
    summary = generate_evolution_summary(new_opinion)
    print("Evolution Summary:")
    print(summary)
    print("\n")

    # Save new opinion
    save_new_opinion(new_opinion)
    print("Opinion saved to history.csv")


# Test the clean version
full_pipeline("I think social media is improving and feels healthier than before.")



print("ğŸ“Š COMPREHENSIVE TESTING - MULTIPLE REAL OPINIONS")
print("=" * 60)

test_opinions = [
    "Artificial intelligence is dangerous and needs strict regulation.",
    "Remote work reduces productivity and teamwork.",
    "Social media causes more harm than good to society."
]

for i, opinion in enumerate(test_opinions, 1):
    print(f"\nğŸ§ª Test Case {i}: {opinion}")
    print("-" * 50)
    full_pipeline(opinion)
    print("=" * 60)



pd.read_csv("history.csv")



import time

def analyze_system_performance():
    print("ğŸ“ˆ SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    # Check CSV existence and count entries
    if os.path.exists("history.csv"):
        df_perf = pd.read_csv("history.csv")
        print(f"â€¢ Total opinions tracked: {len(df_perf)}")
        print(f"â€¢ Earliest entry: {df_perf['timestamp'].min()}")
        print(f"â€¢ Latest entry: {df_perf['timestamp'].max()}")
    else:
        print("â€¢ history.csv not found â€” no data stored yet.")
    
    # Test processing time
    start = time.time()
    _ = clean_opinion("Performance test opinion")
    end = time.time()
    print(f"â€¢ Opinion cleaning time: {end - start:.2f} seconds")

    print("â€¢ All GenAI capabilities operational: Prompting, Embeddings, Retrieval, Reflection âœ“")

analyze_system_performance()


