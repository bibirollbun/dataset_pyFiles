import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("ğŸ”‘ API Key Loaded Successfully!")
except Exception as e:
    print("â�Œ Add `GOOGLE_API_KEY` to Kaggle Secrets.", e)



from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from google.genai import Client

client = Client(api_key=os.environ["GOOGLE_API_KEY"])

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

kb_docs = [
    "Walking 30 minutes daily improves cardiovascular health",
    "High protein vegetarian meals increase metabolism",
    "Consistent sleep improves cognition and hormonal balance",
    "Strength training 3 times weekly supports fat loss",
]

embeddings = embedding_model.encode(kb_docs)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def search_kb(query):
    vec = embedding_model.encode([query])
    D, I = index.search(vec, 2)
    return "\n".join([kb_docs[i] for i in I[0]])



shared_memory = {"profile": {}, "history": []}

def intake_agent_fn(message):
    prompt = f"Extract health profile key/value JSON only: {message}"
    response = client.responses.generate(model="gemini-1.5-flash", input=prompt)
    try:
        shared_memory["profile"].update(json.loads(response.output_text))
    except:
        pass
    return f"Profile Updated: {shared_memory['profile']}"

def planner_agent_fn(message):
    prompt = f"Create today's wellness plan based on profile: {shared_memory['profile']}"
    response = client.responses.generate(model="gemini-1.5-flash", input=prompt)
    return response.output_text

def rag_agent_fn(message):
    context = search_kb(message)
    prompt = f"Use this science knowledge:\n{context}\n\nQ:{message}"
    response = client.responses.generate(model="gemini-1.5-flash", input=prompt)
    return response.output_text

def coach_agent_fn(message):
    prompt = f"Provide empathy + motivation + next steps:\n{message}"
    response = client.responses.generate(model="gemini-1.5-flash", input=prompt)
    return response.output_text



def orchestrator(query):
    shared_memory["history"].append(query)

    if any(x in query.lower() for x in ["age", "diet", "goal", "weight", "vegetarian"]):
        return intake_agent_fn(query)
    if any(x in query.lower() for x in ["plan", "schedule", "routine"]):
        return planner_agent_fn(query)
    if any(x in query.lower() for x in ["feel", "motivation", "tired", "anxiety"]):
        return coach_agent_fn(query)
    return rag_agent_fn(query)



print("ğŸ‘¤ Intake:", orchestrator("I am 29, vegetarian, lightly active, want more energy"))
print("\nğŸ“š Info:", orchestrator("Suggest breakfast options for more energy"))
print("\nğŸ“… Plan:", orchestrator("Create my wellness schedule for today"))
print("\nâ�¤ï¸� Coach:", orchestrator("I feel tired and lazy today"))


