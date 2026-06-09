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


# Cell 1 â€” Notebook header & helpers
from pathlib import Path
import json, sys, time

NOTEBOOK_TITLE = "MicroPro - Simplified Capstone (FAISS-free)"
print(NOTEBOOK_TITLE)
def safe_print(x): 
    try:
        print(x)
    except Exception:
        sys.stdout.write(str(x)+"\n")


# Cell 2 â€” Knowledge base (documents)
# Replace/add your microprocessor notes here or upload files and read them into this list.

documents = [
    "8085 is an 8-bit microprocessor developed by Intel. It has registers A, B, C, D, E, H, L and a flags register.",
    "8085 interrupts: TRAP (highest), RST7.5, RST6.5, RST5.5, INTR (vectored/non-vectored).",
    "8259 PIC: Programmable Interrupt Controller; can be cascaded master-slave to expand interrupt lines.",
    "8254 PIT: Programmable Interval Timer used for generating precise time delays and frequencies.",
    "Example assembly: To add two 8-bit numbers at memory 4000H and 4001H store result at 4002H, load, add and store sequence is required."
]

# if you uploaded files, you can read them like:
# p = Path("data/microprocessor_notes.md")
# if p.exists(): documents.append(p.read_text())
safe_print(f"Loaded {len(documents)} documents.")


# Cell 3 â€” Simple search with scoring (no external libs)
import re
def simple_search(query, docs, top_k=5):
    q_words = re.findall(r"\w+", query.lower())
    scores = []
    for doc in docs:
        doc_words = re.findall(r"\w+", doc.lower())
        # score = number of query tokens that appear in doc
        match_count = sum(1 for w in q_words if w in doc_words)
        # boost for longer context matches
        score = match_count + 0.1 * len(set(q_words).intersection(set(doc_words)))
        scores.append((score, doc))
    # sort by score desc and filter zero
    results = [d for s,d in sorted(scores, key=lambda x: x[0], reverse=True) if s>0]
    if not results:
        return ["No relevant info found in the local knowledge base."]
    return results[:top_k]

# quick test
safe_print(simple_search("what are 8085 interrupts", documents))


# Cell 4 â€” Mock LLM generator (deterministic, no API)
def mock_generate_answer(context_list, query):
    # Combine top contexts and produce a short answer
    context_text = "\n\n".join(context_list[:5])
    answer = f"Question: {query}\n\nSummary from notes:\n{context_text}\n\nAnswer (brief):\n"
    # simple heuristic: if "interrupt" present, craft an interrupt-style answer
    ql = query.lower()
    if "interrupt" in ql or "interrupts" in ql:
        answer += ("8085 hardware interrupts include TRAP (highest priority), RST7.5, RST6.5, RST5.5 and INTR. "
                   "TRAP is non-maskable and highest priority.")
    elif "8259" in ql or "pic" in ql:
        answer += "8259 PIC is a programmable interrupt controller; it can be used in master-slave cascaded mode to expand interrupts."
    elif "8254" in ql or "pit" in ql:
        answer += "8254 is a programmable interval timer used for time delays and frequency generation in embedded systems."
    else:
        # fallback: summarize first context sentence-ish
        first = context_text.split(".")[0]
        answer += (first + ". If more detail is needed, add more notes or upload reference files.")
    return answer

# quick test
safe_print(mock_generate_answer(simple_search("explain 8259", documents), "Explain 8259 PIC"))


# Cell 5 â€” Optional external LLM (OpenAI) â€” COMMENTED by default
# If you want better answers, uncomment and set OPENAI_API_KEY as an environment variable or paste it below.
"""
try:
    import openai
    OPENAI_API_KEY = None  # or put your key string here (not recommended in public notebooks)
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
    elif "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        safe_print("OpenAI API key not provided; using mock generator.")

    def openai_generate(context_list, query):
        context_text = "\n\n".join(context_list[:5])
        prompt = f"Use the context to answer the question concisely.\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change if needed
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return resp['choices'][0]['message']['content'].strip()
except Exception as e:
    safe_print("OpenAI integration not enabled or failed:", e)
    def openai_generate(context_list, query):
        return mock_generate_answer(context_list, query)
"""
# End optional block
safe_print("External LLM block is optional and currently commented out.")


# Cell 6 â€” Agent function (uses simple_search + mock_generate or external LLM if enabled)
def micropro_agent(query, top_k=5, use_external_llm=False):
    hits = simple_search(query, documents, top_k=top_k)
    # If external LLM available and desired, call openai_generate here (must be enabled in Cell 5)
    if use_external_llm:
        try:
            # openai_generate defined only if you enabled the optional block
            ans = openai_generate(hits, query)
        except Exception:
            ans = mock_generate_answer(hits, query)
    else:
        ans = mock_generate_answer(hits, query)
    # return answer + compact citations
    citations = "\n\n---\nCitations (local documents):\n" + "\n\n".join(hits[:top_k])
    return ans + citations

# quick test
safe_print(micropro_agent("How many interrupts does 8085 have?"))


# Cell 7 â€” Add and list documents
def add_document(text):
    documents.append(text.strip())
    return f"Document added. Total docs: {len(documents)}"

def list_documents(limit=10):
    return documents[:limit]

# Example usage (uncomment to add)
# add_document("New note: 8086 has segment registers CS, DS, SS, ES and general registers AX, BX, CX, DX.")
# safe_print(list_documents())


# Cell 8 â€” Simple evaluation helpers
import pandas as pd

def evaluate(questions):
    rows = []
    for q in questions:
        ans = micropro_agent(q)
        # short length and whether 'No relevant' phrase present
        ok = 0 if "No relevant info found" in ans else 1
        rows.append({"question": q, "short_ok": ok, "answer_preview": ans[:250]})
    return pd.DataFrame(rows)

# Example evaluation
sample_qs = [
    "Explain the 8085 interrupt structure.",
    "Describe 8254 PIT.",
    "How to add two 8-bit numbers at addresses 4000H and 4001H?"
]
df_eval = evaluate(sample_qs)
df_eval


# Cell 9 â€” UI (tries Gradio; falls back to simple input())
def run_console():
    try:
        while True:
            q = input("\nAsk MicroPro (type 'exit' to quit): ").strip()
            if q.lower() in ("exit","quit"):
                break
            print("\n" + micropro_agent(q) + "\n")
    except Exception as e:
        safe_print("Console UI stopped:", e)

# Try to use gradio if installed
try:
    import gradio as gr
    def gradio_fn(q):
        return micropro_agent(q)
    demo = gr.Interface(fn=gradio_fn, inputs=gr.Textbox(lines=2), outputs=gr.Textbox(lines=12),
                       title="MicroPro - Microprocessor Study Agent (Simple)")
    safe_print("Gradio detected â€” run demo.launch() to start an interface (in Colab, set share=True if needed).")
    # To launch in notebook: demo.launch(share=False)
except Exception:
    safe_print("Gradio not available â€” use run_console() to interact.")
    # run_console()  # Uncomment to run interactive console UI in environments that support input()


# Cell 10 â€” Auto-generate a small README file for Kaggle submission
readme = f"""
{NOTEBOOK_TITLE}

What this notebook does:
- Simple Retrieval (keyword) + generation (mock or external LLM)
- No FAISS required (runs in any Colab/Kaggle without special installs)
- Add your own notes to `documents` or upload files and append them

How to run:
- Run cells 1 -> 10 in order
- Use micropro_agent(query) to get answers
- Optional: enable OpenAI block in Cell 5 and set API key to use real LLMs

Author: Harsh Kumar Singh
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
Path("README_for_submission.md").write_text(readme.strip())
safe_print("README_for_submission.md written to notebook directory.")

