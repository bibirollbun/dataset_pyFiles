!pip install PyPDF2 beautifulsoup4 requests --quiet
print("Dependencies installed.")


import os
import json
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from datetime import datetime
print("All required packages installed.")


MEMORY_BANK = {"assignments": {}}


def log_event(agent, event, details=""):
    print({
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "event": event,
        "details": details
    })


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    log_event("NotesAgent", "PDF_EXTRACT", path)
    return text


def chunk_text(text, chunk_size=1200):
    words = text.split()
    chunks, current = [], []
    count = 0

    for w in words:
        current.append(w)
        count += len(w)
        if count > chunk_size:
            chunks.append(" ".join(current))
            current, count = [], 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def fake_llm_summarize(chunk):
    log_event("NotesAgent", "SUMMARIZE_CHUNK", f"{len(chunk)} chars")
    return "Summary: " + chunk[:200] + "..."


def summarize_pdf(pdf_path):
    raw = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw)

    summaries = []
    for c in chunks:
        summaries.append(fake_llm_summarize(c))

    final_summary = "\n\n".join(summaries)
    log_event("NotesAgent", "FINAL_SUMMARY", f"{len(summaries)} chunks processed")
    return final_summary


def simple_web_search(query):
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    res = requests.post(url, data=params)

    soup = BeautifulSoup(res.text, "html.parser")
    results = []

    for a in soup.select(".result__a")[:5]:
        results.append({"title": a.text, "link": a["href"]})

    log_event("SearchAgent", "SEARCH", query)
    return results


with open("eduflow_demo_state.json", "w") as f:
    json.dump(MEMORY_BANK, f, indent=2)

log_event("System", "STATE_SAVED", "eduflow_demo_state.json created")


timetable = {
    "2025-11-25": [
        {"subject": "Maths", "time": "9:00 AM"},
        {"subject": "Physics", "time": "11:00 AM"},
    ]
}

plan = create_daily_plan("emil", "2025-11-25", timetable)
plan


summary = summarize_pdf("/kaggle/input/sample-notes-pdf/Module 4-Part 1.pdf")
print(summary)


simple_web_search("introduction to cyber security notes")

