## 2. Install & Import Libraries

!pip install --upgrade --no-cache-dir protobuf==4.25.3
!pip install  googleapis-common-protos
!pip install  -q google-genai
!pip install  sentence-transformers faiss-cpu pypdf


from google.genai import Client, types
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np
import textwrap



from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")


  # in Kaggle: use environment or secrets
client = Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"



from google.genai import Client, types

client = Client(api_key=GEMINI_API_KEY)

def ask_gemini(prompt):
    res = client.models.generate_content(
        model="gemini-2.5-flash",   # Supported by API key
        contents=prompt
    )
    return res.text



from google.genai import Client, types

client = Client(api_key=GEMINI_API_KEY)

def ask_gemini(prompt: str, system_instruction: str = None):
    full_prompt = ""
    if system_instruction:
        full_prompt += system_instruction + "\n\n"
    full_prompt += prompt

    contents = [
        types.Content(role="user", parts=[types.Part(text=full_prompt)])
    ]

    res = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.4,
        )
    )
    return res.candidates[0].content.parts[0].text



pdf_path = "/kaggle/input/project/archiflow_ai_project_notes.pdf"
pdf_path



def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)

raw_text = extract_text_from_pdf(pdf_path)
print(raw_text[:2000])  # preview



def chunk_text(text, max_tokens=400, overlap=50):
    # Simple char-based chunking, good enough for this project
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

chunks = chunk_text(raw_text, max_tokens=300, overlap=50)
len(chunks)



embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("Index size:", index.ntotal)



def retrieve_relevant_chunks(query: str, k: int = 5):
    query_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k)
    selected_chunks = [chunks[i] for i in indices[0]]
    return selected_chunks



RAG_SYSTEM_PROMPT = """
You are StudyBuddy RAG Agent.
You must ONLY answer using the provided context from the student's PDF notes/syllabus.
If the answer is not in the context, say you don't know.

Always:
- Explain in simple terms.
- Refer to context logically, but don't hallucinate details.
"""



def answer_with_rag(user_query: str) -> str:
    context_chunks = retrieve_relevant_chunks(user_query, k=5)
    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
Context from student's PDF:
{context_text}

---
User question: {user_query}

Using ONLY the context above, answer the question clearly.
If the answer is missing, say: "I couldn't find this in your notes."
"""

    return ask_gemini(prompt, system_instruction=RAG_SYSTEM_PROMPT)



answer_with_rag("What is MVP Scope")



from datetime import date, timedelta

TASKS = []

def add_task(title: str, est_hours: float = 1.0, due_day_offset: int = None):
    """Add a study task. due_day_offset = days from today (optional)."""
    due_date = date.today() + timedelta(days=due_day_offset) if due_day_offset is not None else None
    TASKS.append({
        "title": title,
        "est_hours": est_hours,
        "due_date": due_date
    })

def list_tasks():
    return TASKS



def generate_study_plan(num_days: int, hours_per_day: float = 2.0):
    """
    Very simple planner: assign tasks in order across days respecting daily hour budget.
    """
    plan = { (date.today() + timedelta(days=i)).isoformat(): [] for i in range(num_days) }

    remaining_hours = {
        day: hours_per_day for day in plan.keys()
    }

    for task in TASKS:
        needed = task["est_hours"]
        for day in plan.keys():
            if needed <= 0:
                break
            if remaining_hours[day] <= 0:
                continue
            assign_hours = min(remaining_hours[day], needed)
            plan[day].append({"title": task["title"], "hours": float(assign_hours)})
            remaining_hours[day] -= assign_hours
            needed -= assign_hours

    return plan



TASK_SYSTEM_PROMPT = """
You are StudyBuddy Planner Agent.
Your job is to turn raw syllabus/notes text into a list of atomic study tasks.

Rules:
- Each task should be small (1â€“2 hours).
- Include the topic name and type (e.g., "Read", "Revise", "Solve problems").
- Return output as bullet points.
"""

def extract_tasks_from_pdf(sample_chunk_count: int = 5):
    # take a subset of chunks to avoid flooding
    sample_text = "\n\n".join(chunks[:sample_chunk_count])
    prompt = f"""
Here is a sample of the student's syllabus/notes:

{sample_text}

From this, list concrete study tasks the student should complete.
"""
    response = ask_gemini(prompt, system_instruction=TASK_SYSTEM_PROMPT)
    return response



def pretty_study_plan(plan_dict, preferences_text: str = ""):
    # Convert plan dict to readable text and let Gemini polish it
    raw_plan_lines = []
    for day, tasks in plan_dict.items():
        raw_plan_lines.append(f"Day {day}:")
        if not tasks:
            raw_plan_lines.append("  - Free / buffer")
        else:
            for t in tasks:
                raw_plan_lines.append(f"  - {t['title']} (~{t['hours']}h)")
    raw_plan_text = "\n".join(raw_plan_lines)

    prompt = f"""
User preferences: {preferences_text}

Raw plan:
{raw_plan_text}

Rewrite this as a friendly, structured 7-day study plan with headings and bullet points.
"""
    return ask_gemini(prompt, system_instruction="You are a helpful study planning assistant.")



INTENT_SYSTEM_PROMPT = """
You are an intent classifier for the StudyBuddy Agent.
You must classify the user's message into one of:
- PDF_QA  -> asking about concepts from the notes/PDF
- PLAN    -> asking for schedule, plan, tasks, or time management
- BOTH    -> if the user wants both understanding and planning
Return exactly one word: PDF_QA, PLAN, or BOTH.
"""

def classify_intent(message: str) -> str:
    res = ask_gemini(message, system_instruction=INTENT_SYSTEM_PROMPT)
    res_clean = res.strip().upper()
    if "BOTH" in res_clean:
        return "BOTH"
    if "PLAN" in res_clean:
        return "PLAN"
    return "PDF_QA"



def studybuddy_agent(message: str, num_days: int = 7, hours_per_day: float = 2.0, preferences: str = ""):
    intent = classify_intent(message)
    print(f"[DEBUG] Intent: {intent}")

    responses = []

    if intent in ("PDF_QA", "BOTH"):
        answer = answer_with_rag(message)
        responses.append("ðŸ“˜ **Answer from your notes:**\n" + answer)

    if intent in ("PLAN", "BOTH"):
        # for demo, we assume tasks already exist in TASKS (you can also auto-fill from extract_tasks_from_pdf)
        plan = generate_study_plan(num_days=num_days, hours_per_day=hours_per_day)
        pretty = pretty_study_plan(plan, preferences_text=preferences)
        responses.append("ðŸ“… **Personalized Study Plan:**\n" + pretty)

    return "\n\n---\n\n".join(responses)



# 1. Ask a pure PDF question
print(studybuddy_agent("Explain MVP."))

# 2. Ask for planning
print(studybuddy_agent(
    "I have an exam in 10 days, please create a plan to cover all topics.",
    num_days=10,
    hours_per_day=3.0,
    preferences="I prefer studying in the evening and want revisions before the exam."
))

# 3. Combined request
print(studybuddy_agent(
    "Explain overfitting from my notes and also give me a plan to revise it over the next 3 days.",
    num_days=3,
    hours_per_day=1.5
))


