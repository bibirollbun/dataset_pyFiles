# Kaggle notebook cell - run these commands (one cell)
# 1) Prevent tokenizers parallelism warning at runtime:
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("TOKENIZERS_PARALLELISM set to false")

# 2) Install protobuf compatibility and required libraries but try to avoid upgrading huge system packages.
# We install protobuf first (fixes MessageFactory errors), then required libs with --no-deps to avoid changing system deps.
!pip install --upgrade --force-reinstall "protobuf==3.20.3" || true

# Install minimal needed libs WITHOUT touching other deps.
# Use --no-deps to reduce the chance pip upgrades other system packages that cause conflicts.
!pip install --upgrade --force-reinstall sentence-transformers faiss-cpu transformers gradio PyPDF2 openai --no-deps || true

# 3) IMPORTANT: Restart the kernel after these operations (Kernel -> Restart).
print("If commands ran, now Restart the Kernel/Runtime (very important).")



# === PolicyAssist ===


# 0) Install dependencies (run once)
!pip install -q faiss-cpu tiktoken datasets transformers sentence-transformers gradio PyPDF2

# 1) Imports & Config
import os, json, textwrap, uuid, time, logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict, deque

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from transformers import pipeline
import gradio as gr

# 1a) Logging + metrics
logging.basicConfig(
    filename="policyassist.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
metrics = defaultdict(int)

def log_event(name, **kwargs):
    metrics[f"count_{name}"] += 1
    logging.info(f"EVENT {name} | {kwargs}")

# 2) No cloud LLMs (fully local)
use_openai = False
print("Using OpenAI embeddings/LLM? ", use_openai)

# 3) Create sample docs (5 files) in ./docs if they don't exist
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

sample_docs = {
    "hr_policy.txt": """
HR Policy - Paid Leave & Sick Leave

1. Paid Time Off (PTO): Full-time employees are eligible for 20 days of paid time off per calendar year.
2. Sick Leave: Employees receive 10 paid sick days per year. Sick leave may be used for personal illness or care of immediate family.
3. Maternity/Paternity Leave: Maternity leave is 16 weeks for eligible employees; paternity leave is 2 weeks.
4. Procedure: To apply for leave, submit a leave request via the HR portal at least 2 days in advance when possible. For emergencies, contact HR directly.
""",
    "benefits.txt": """
Benefits Guide

1. Health Insurance: Full-time employees are eligible for health coverage starting from their date of hire. Spouse and dependents can be enrolled during open enrollment or within 30 days of a qualifying event.
2. Retirement: Company matches up to 5% of employee contributions to the retirement plan.
3. Enrollment: To enroll, complete the Benefits Enrollment Form and submit to benefits@company.example within 30 days of hire or qualifying event.
""",
    "it_assets.txt": """
IT Assets & Hardware Replacement Policy

1. Hardware Requests: Employees can request laptops, monitors, and peripherals via the IT ticketing system.
2. Replacement: For damaged or failing hardware, submit a replacement request with photos and justification. IT will evaluate within 3 business days.
3. Approval Flow: Manager approval is required for replacement requests. For urgent replacement (e.g. device not functioning), IT will provide a temporary loaner.
""",
    "remote_policy.txt": """
Remote Work Policy

1. Short-term Remote Work: Employees may request remote work for up to 10 days with manager approval.
2. Long-term Remote Work (>30 days): Requires manager approval and HR sign-off; business justification must be provided.
3. Process: Submit remote work request form through internal portal; include dates and reason.
""",
    "code_of_conduct.txt": """
Code of Conduct & Reporting

1. Harassment: The company has zero tolerance for harassment. Any person experiencing or witnessing harassment should report it to HR immediately.
2. Reporting Procedure: Reports can be made via the HR portal or to hr@company.example. Confidentiality will be maintained where possible.
3. Investigation: The company will investigate allegations and may take disciplinary action up to termination.
"""
}

for fname, body in sample_docs.items():
    p = DOCS_DIR / fname
    if not p.exists() or p.stat().st_size == 0:
        p.write_text(body.strip(), encoding="utf-8")
print("Sample docs present in ./docs")

# 4) Document loading + PDF helper
def load_text_from_pdf(path):
    reader = PdfReader(path)
    text = []
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)

def load_documents():
    docs = []
    for p in sorted(DOCS_DIR.iterdir()):
        if p.suffix.lower() == ".pdf":
            text = load_text_from_pdf(p)
        else:
            text = p.read_text(encoding="utf-8")
        docs.append({"path": str(p.name), "text": text})
    return docs

# 5) Chunking (simple word-based with overlap)
def chunk_text(text, chunk_size=250, overlap=50):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
        i += max(chunk_size - overlap, 1)
    return chunks

# 6) Embeddings: SentenceTransformers (local)
# Using a small fast model for RAG-style retrieval. [web:13]
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts_sbert(texts):
    return sbert_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# 7) Build FAISS index helper
def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# 8) Ingest docs => chunks => embeddings => FAISS index
def ingest_docs_and_build_index():
    docs = load_documents()
    chunks = []
    metadatas = []
    for doc in docs:
        for c in chunk_text(doc["text"], chunk_size=250, overlap=50):
            chunks.append(c)
            metadatas.append({"source": doc["path"]})
    embeddings = embed_texts_sbert(chunks).astype("float32")
    index = build_faiss_index(embeddings)
    log_event("engest_docs", num_chunks=len(chunks), use_openai=False)
    return index, embeddings, chunks, metadatas

index, embeddings, chunks, metadatas = ingest_docs_and_build_index()
print(f"Ingested {len(chunks)} chunks into FAISS index. (use_openai={use_openai})")

# 9) Retrieval function
def retrieve(query, k=4):
    q_emb = embed_texts_sbert([query]).astype("float32")
    D, I = index.search(q_emb, k)
    results = []
    for j, i in enumerate(I[0]):
        if i < 0 or i >= len(chunks):
            continue
        results.append(
            {
                "text": chunks[i],
                "meta": metadatas[i],
                "score": float(D[0][j]),
            }
        )
    log_event("retrieve", query=query, returned=len(results))
    return results

# 10) Local LLM (summarizer) for answers
# Uses DistilBART CNN summarization as a lightweight answer generator. [web:16]
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

SYSTEM_PROMPT = (
    "You are PolicyAssist, an enterprise assistant. "
    "Use the provided context passages to answer employee questions concisely and precisely. "
    "Always mention the source file name in the answer when relevant. "
    "If the answer is not in the documents, say "
    "\"I couldn't find a definitive answer in the documents\" and suggest asking HR or checking the portal. "
    "Keep answers short (2-5 sentences)."
)

def generate_answer_with_local_llm(query, retrieved):
    # Merge retrieved chunks + question, then summarize.
    context = "\n\n".join([f"Source: {r['meta']['source']}\n{r['text']}" for r in retrieved])
    combined = (SYSTEM_PROMPT + "\n\n" + context + "\n\nQuestion: " + query)[:3000]
    try:
        out = summarizer(combined, max_length=180, min_length=40, do_sample=False)
        ans = out[0]["summary_text"]
    except Exception as e:
        ans = "I couldn't generate an answer using the local model. Please check the HR/IT portal."
    log_event("local_summarizer", q_len=len(query.split()), n_retrieved=len(retrieved))
    return ans

# 11) Answer wrapper
def answer_with_llm(query, retrieved, user_context=None):
    # Simple wrapper that could be extended to inject user_context.
    ans = generate_answer_with_local_llm(query, retrieved)
    low_confidence = any(
        kw in ans.lower()
        for kw in [
            "couldn't find a definitive answer",
            "could not find a definitive answer",
            "not in the documents",
            "cannot find",
        ]
    )
    log_event("prompt_built", prompt_len=len(query.split()))
    return {"answer": ans, "low_confidence": low_confidence}

# 12) Sessions & Memory
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}  # session_id -> dict

    def create(self):
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"history": deque(maxlen=40), "created": time.time()}
        return sid

    def get(self, sid):
        return self.sessions.get(sid)

    def add_message(self, sid, role, text):
        s = self.sessions.setdefault(
            sid, {"history": deque(maxlen=40), "created": time.time()}
        )
        s["history"].append({"role": role, "text": text, "ts": time.time()})

class MemoryBank:
    def __init__(self):
        self.mem = defaultdict(dict)

    def write(self, user_id, key, value):
        self.mem[user_id][key] = value

    def read(self, user_id, key):
        return self.mem[user_id].get(key)

    def summary(self, user_id):
        return self.mem[user_id]

session_service = InMemorySessionService()
memory_bank = MemoryBank()
memory_bank.write("user_demo", "role", "Software Engineer")
memory_bank.write("user_demo", "location", "Bengaluru")

# 13) Tools
def faiss_retrieval_tool(query, k=4):
    retrieved = retrieve(query, k=k)
    log_event("faiss_retrieval_tool", q_len=len(query.split()), returned=len(retrieved))
    return retrieved

def mock_directory_tool(name_query):
    q = name_query.lower()
    if "hr" in q:
        return {"name": "HR Team", "email": "hr@company.example", "phone": "+91-80-0000-1111"}
    if "it" in q:
        return {"name": "IT Support", "email": "it@company.example", "phone": "+91-80-0000-2222"}
    return {"message": "No directory match"}

# 14) PolicyAnswerer agent
def policy_answerer_agent(query, retrieved, user_context=None):
    res = answer_with_llm(query, retrieved, user_context=user_context)
    log_event("policy_answerer", low_confidence=res["low_confidence"])
    return res

# 15) Orchestrator (multi-step behavior)
def orchestrator(session_id, user_id, query):
    # Save user message
    session_service.add_message(session_id, "user", query)

    # 1) Retrieval
    retrieved = faiss_retrieval_tool(query, k=6)

    # 2) User context from memory
    user_ctx = memory_bank.summary(user_id)

    # 3) Call policy answerer
    res = policy_answerer_agent(query, retrieved, user_context=user_ctx)

    # 4) Clarifier loop on low confidence
    if res["low_confidence"]:
        clarifier = (
            "I couldn't find a definitive answer in the documents. "
            "Do you mean (A) for full-time employees or (B) for contractors? Reply A or B."
        )
        session_service.add_message(session_id, "agent", clarifier)
        log_event("clarifier_issued", session=session_id)
        return {"type": "clarify", "content": clarifier}

    # 5) Normal answer path
    session_service.add_message(session_id, "agent", res["answer"])
    log_event("orchestrator_answered", session=session_id)
    return {
        "type": "answer",
        "content": res["answer"],
        "sources": [r["meta"]["source"] for r in retrieved[:3]],
    }

# 16) Evaluation harness
test_cases = [
    {
        "q": "How many sick days per year do employees have?",
        "expected_source": "hr_policy",
    },
    {
        "q": "How can I replace a laptop?",
        "expected_source": "it_assets",
    },
    {
        "q": "Who approves long term remote work >30 days?",
        "expected_source": "remote_policy",
    },
]

def evaluate(test_cases, k=4):
    results = []
    for tc in test_cases:
        retrieved = faiss_retrieval_tool(tc["q"], k=k)
        relevant = sum(
            1 for r in retrieved if tc["expected_source"] in r["meta"]["source"]
        )
        p_at_k = relevant / (len(retrieved) if retrieved else 1)
        ans_res = policy_answerer_agent(tc["q"], retrieved, user_context={})
        results.append(
            {
                "q": tc["q"],
                "p@k": p_at_k,
                "low_confidence": ans_res["low_confidence"],
                "answer": ans_res["answer"],
            }
        )
        log_event("eval_case", query=tc["q"], p_at_k=p_at_k)
    return results

# 17) Export metrics helper
def export_metrics(path="metrics.json"):
    with open(path, "w") as f:
        json.dump({"metrics": dict(metrics)}, f, indent=2)
    logging.info("exported metrics")
    return path

# 18) Gradio UI
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# PolicyAssist — Enterprise Q&A\n"
            "Multi-agent demo: Retriever tool, Local LLM agent, Clarifier loop, "
            "Sessions & Memory, Observability."
        )

        with gr.Row():
            create_btn = gr.Button("Create Session")
            sid_out = gr.Textbox(label="Session ID", interactive=False)
            user_id_in = gr.Textbox(
                label="User ID (for Memory)", value="user_demo"
            )

        with gr.Row():
            inp = gr.Textbox(
                label="Ask a policy question",
                placeholder="e.g., 'How many sick leaves do employees get?'",
            )
            ask_btn = gr.Button("Ask")

        out = gr.Textbox(label="Agent Response", lines=6)
        sources_out = gr.Textbox(
            label="Top Sources (retrieval)", interactive=False
        )

        eval_btn = gr.Button("Run Evaluation (test cases)")
        eval_out = gr.JSON(label="Evaluation Results")

        metrics_btn = gr.Button("Export Metrics & Logs")
        metrics_link = gr.Textbox(
            label="Exported Metrics Path", interactive=False
        )

        # Create session
        def on_create_session():
            sid = session_service.create()
            return sid

        create_btn.click(on_create_session, inputs=None, outputs=sid_out)

        # Ask handler
        def on_ask(session_id, user_id, question):
            if not session_id:
                return "Please create a session first (click Create Session).", ""
            res = orchestrator(session_id, user_id, question)
            if res["type"] == "clarify":
                return res["content"], ""
            else:
                sources = res.get("sources", [])
                sources_txt = "\n".join([f"- {s}" for s in sources])
                return res["content"], sources_txt

        ask_btn.click(
            on_ask, inputs=[sid_out, user_id_in, inp], outputs=[out, sources_out]
        )

        # Run evaluation
        def on_eval():
            return evaluate(test_cases)

        eval_btn.click(on_eval, inputs=None, outputs=eval_out)

        # Export metrics
        def on_export_metrics():
            return export_metrics()

        metrics_btn.click(
            on_export_metrics, inputs=None, outputs=metrics_link
        )

    return demo

demo = create_gradio_interface()

print("Launching Gradio demo... (set share=False if you don't want a public link)")
demo.launch(share=True, server_port=7860)

# === End of Notebook Script ===


