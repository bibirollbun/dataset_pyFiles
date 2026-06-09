!pip install beautifulsoup4 requests matplotlib pandas


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import time
import random
import statistics
from datetime import datetime
# show plots inline (Kaggle)
%matplotlib inline

# Simple observability logger
def log(agent, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{agent}] {msg}")



# Long-term memory (in-memory dict for demo; replace with DB if needed)
MEMORY = {
    "queries": [],       # stores past queries
    "summaries": {},     # query -> summary
    "analytics": {}      # query -> analytics dict
}

def memory_store(key, value):
    MEMORY[key] = value

def memory_append_query(q):
    MEMORY["queries"].append({"query": q, "time": datetime.now().isoformat()})



# AGENT: Query Understanding (context engineering)
def query_agent(user_query):
    log("QueryAgent", f"Received query: {user_query}")
    keywords = [w.lower() for w in user_query.split() if len(w)>2]
    context = {
        "user_query": user_query,
        "keywords": keywords,
        "timestamp": datetime.now().isoformat()
    }
    memory_append_query(user_query)
    log("QueryAgent", f"Extracted keywords: {keywords}")
    return context

# TOOL AGENT: Search Agent (Wikipedia scraping; acts as external tool)
def search_wikipedia(keyword):
    # tries exact page, otherwise falls back to first paragraph of search result
    try:
        url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            paras = soup.select("p")
            text = " ".join([p.get_text(strip=True) for p in paras[:4]])
            return text
        else:
            return None
    except Exception as e:
        log("SearchAgent", f"Error for {keyword}: {e}")
        return None

# AGENT: Parallel Search Agent (runs multiple searches in parallel)
def parallel_search_agent(keywords, max_workers=4):
    log("SearchAgent", f"Searching for keywords: {keywords}")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_to_kw = {exe.submit(search_wikipedia, kw): kw for kw in keywords}
        for fut in concurrent.futures.as_completed(future_to_kw):
            kw = future_to_kw[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = None
            results[kw] = res
            log("SearchAgent", f"Completed: {kw} len={len(res) if res else 0}")
    return results

# AGENT: Skill Extraction Agent (rules + simple keyword matching)
SKILL_MAP = {
    "data": ["Python", "Pandas", "SQL", "Machine Learning", "Statistics"],
    "ai": ["Python", "TensorFlow", "PyTorch", "LLMs", "Data Engineering"],
    "cloud": ["AWS", "GCP", "Azure", "Kubernetes", "Docker"],
    "security": ["Networking", "PenTesting", "SIEM", "Cryptography"]
}

def skill_extraction_agent(context_text):
    log("SkillAgent", "Extracting skills from context")
    txt = (context_text or "").lower()
    skills = set()
    for k, v in SKILL_MAP.items():
        if k in txt:
            skills.update(v)
    if not skills:
        # fallback: pick a few common tech skills
        skills.update(["Communication", "Problem Solving", "Teamwork"])
    skills_list = list(skills)
    log("SkillAgent", f"Detected skills: {skills_list}")
    return skills_list

# AGENT: Salary/Market Data Agent (toy dataset + plotting)
def salary_agent():
    # demo salary data (replace with real dataset when available)
    df = pd.DataFrame({
        "Country": ["US","India","UK","Canada","Germany"],
        "AvgSalaryUSD": [120000, 25000, 80000, 95000, 90000]
    })
    log("SalaryAgent", "Prepared salary sample dataset")
    return df

# AGENT: Demand Forecast Agent (simple synthetic forecasting for demo)
def demand_forecast_agent(years=range(2023,2031)):
    # create a synthetic demand time series with some randomness
    base = [random.randint(60,85) + int(2*(y-2023)) for y in years]  # gentle upward trend
    df = pd.DataFrame({"Year": list(years), "DemandScore": base})
    log("DemandAgent", f"Created forecast for years {list(years)}")
    return df

# AGENT: Summarizer Agent (very simple extractive summarizer)
def summarizer_agent(search_results):
    log("Summarizer", "Building summary from search results")
    joined = " ".join([r for r in search_results.values() if r])
    if not joined:
        return None
    sentences = joined.split(". ")
    # pick top N sentences heuristically
    summary = ". ".join(sentences[:6]).strip()
    log("Summarizer", f"Summary length: {len(summary)} chars")
    return summary



def supervisor_flow(user_query):
    # 1. Query understanding
    ctx = query_agent(user_query)

    # 2. Parallel search (Tool usage)
    keywords = ctx["keywords"][:6] or [ctx["user_query"]]
    search_results = parallel_search_agent(keywords)

    # 3. Summarize
    summary = summarizer_agent(search_results) or "No consolidated summary found."
    MEMORY["summaries"][user_query] = summary

    # 4. Skills extraction
    skills = skill_extraction_agent(summary)

    # 5. Salary analysis & plotting
    salary_df = salary_agent()
    # plot salary
    plt.figure(figsize=(6,3))
    plt.bar(salary_df["Country"], salary_df["AvgSalaryUSD"])
    plt.title(f"Average Salary (sample) — {user_query}")
    plt.ylabel("USD")
    plt.show()

    # 6. Demand forecast
    demand_df = demand_forecast_agent()
    plt.figure(figsize=(6,3))
    plt.plot(demand_df["Year"], demand_df["DemandScore"], marker="o")
    plt.title("Demand Forecast (synthetic)")
    plt.xlabel("Year")
    plt.ylabel("Demand Score")
    plt.grid(True)
    plt.show()

    # 7. Analytics & evaluation: basic scoring
    avg_salary = salary_df["AvgSalaryUSD"].mean()
    demand_score = demand_df["DemandScore"].mean()
    score = (avg_salary/1000) * (demand_score/100)  # toy combined score
    analytics = {
        "avg_salary": int(avg_salary),
        "avg_demand_score": float(round(demand_score,2)),
        "composite_score": float(round(score,2)),
        "skills": skills
    }
    MEMORY["analytics"][user_query] = analytics
    log("Supervisor", f"Analytics computed: {analytics}")

    # 8. Final report (structured)
    report = {
        "query": user_query,
        "summary": summary,
        "skills": skills,
        "salary_df": salary_df,
        "demand_df": demand_df,
        "analytics": analytics,
        "timestamp": datetime.now().isoformat()
    }
    return report


# Run end-to-end demo
user_query = "AI Engineer"
log("User", f"Running multi-agent workflow for: {user_query}")
report = supervisor_flow(user_query)

# print structured final report
print("\n\n===== FINAL REPORT =====")
print(f"Query: {report['query']}")
print(f"\nSummary (excerpt):\n{(report['summary'] or 'No info available')[:800]}")
print(f"\nDetected Skills: {report['skills']}")
print(f"\nAnalytics: {report['analytics']}")
print("\n===== END =====")


