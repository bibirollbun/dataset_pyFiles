import os
import asyncio
import json
import httpx
from datetime import datetime
from openai import OpenAI
import uuid
import requests
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

OpenAI_key = user_secrets.get_secret("OpenAI_key")

SERVER_ADDRESS = "https://historictext.org"

LLM_REASONING_URL =     f"{SERVER_ADDRESS}/llm_reasoning"
VECTOR_SEARCH_URL =     f"{SERVER_ADDRESS}/vector_search"
ENTITY_SEARCH_URL =     f"{SERVER_ADDRESS}/entity_search"
HYBRID_SEARCH_URL =     f"{SERVER_ADDRESS}/entity_hybrid"
GENERAL_KNOWLEDGE_URL = f"{SERVER_ADDRESS}/general_knowledge"
WEB_SEARCH_URL =        f"{SERVER_ADDRESS}/web_search"
MAX_ITERATIONS = 10


long_timeout = httpx.Timeout(
    connect=60.0,   
    read=300.0,     
    write=300.0,    
    pool=60.0       
)

async def call_fastapi_async(url: str, payload: dict = None, files: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=long_timeout) as client:
        try:
            if files is not None:
                response = await client.post(url, files=files, timeout=60)
            else:
                response = await client.post(url, json=payload, timeout=60)
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        except Exception as e:
            print("Error calling FastAPI endpoint:", e)
            return {"error": str(e)}


def get_endpoint_and_payload(action, context):
    action_type = action['type']
    query = action.get('query', '')
    if action_type == "vector_search":
        return VECTOR_SEARCH_URL, {"query": query, "k": 128, "bge_threshold": 0.2, "semantic_threshold": 0.25 }
    elif action_type == "entity_hybrid":
        return HYBRID_SEARCH_URL, {"query": query, "k": 128, "bge_threshold": 0.2, "semantic_threshold": 0.25 }
    elif action_type == "entity_search":
        if isinstance(query, str):
            entities = [e.strip() for e in query.split(",")]
        return ENTITY_SEARCH_URL, {"entities": entities, "mode": "substring"}
    elif action_type == "general_knowledge":
        return GENERAL_KNOWLEDGE_URL, {"query": query}
    elif action_type == "web_search":
        return WEB_SEARCH_URL, {"query": query, "search_context_size": "medium"}
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def dedup_log(log: list[dict], limit: int = 16) -> list[dict]:
    reversed_log = list(reversed(log))
    aggregated: dict[tuple[str, str], dict] = {}
    for item in reversed_log:
        key = (item["action"], item["query"])
        if key not in aggregated:
            aggregated[key] = item.copy()
        else:
            aggregated[key]["result_count"] += item["result_count"]
    unique_recent = list(reversed(list(aggregated.values())))[:limit]
    return unique_recent


def dedup_supporting_evidence(evidences: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for ev in evidences:
        
        key = (
            ev.get("source"),
            ev.get("value", "").strip(),
            json.dumps(ev.get("details", {}), sort_keys=True)
        )
        if key not in seen:
            unique.append(ev)
            seen.add(key)
    return unique


def pretty_evidence(ev, maxlen=75):
    if isinstance(ev, dict):
        src = ev.get('source', '?')
        val = ev.get('value', '')[:maxlen]
        details = ev.get('details', {}) #[:maxlen]
        meta = ev.get('meta', {})
        if src == "entity_search":
            count = details.get('count')
            mode = meta.get('mode')
            return f"[entity_search] {val} — {count} matches (mode: {mode})"
        year = details.get('year')
        doc = details.get('doc_name')
        return f"[{src}] {val} ({year}, {doc})"
    return f"[BAD TYPE: {type(ev)}] {str(ev)[:maxlen]}"


async def agent_loop(user_query):
    context = []
    reasoning_log = []
    previous_hypotheses = []
    supporting_evidence = []
    agent_thoughts: str = ""
    active_question = user_query

    for iteration in range(MAX_ITERATIONS):
        reasoning_request = {
            "user_query": user_query,
            "active_question": active_question,
            "agent_thoughts": agent_thoughts,
            "context": context,
            "previous_hypotheses": previous_hypotheses,
            "supporting_evidence": supporting_evidence,
            "reasoning_log": dedup_log(reasoning_log, 16),
            "iteration": iteration,
        }

        reasoning = await call_fastapi_async(LLM_REASONING_URL, reasoning_request)
        new_evidence = reasoning.get("new_facts", [])
        supporting_evidence.extend(new_evidence)
        supporting_evidence[:] = dedup_supporting_evidence(supporting_evidence)  

        print(f"ITERATION:   {iteration}")
        print(f"→ Actions: {[a['type'] for a in reasoning['actions']]}")
        print(f"→ Hypothesis: {reasoning.get('hypothesis')}")
        print(f"→ Memory: {reasoning.get('agent_thoughts')}")
        print(f"→ Supporting evidence new: {len(reasoning.get('new_facts', []))}")
        print(f"→ Confidence: {reasoning.get('confidence')}")
        print(f"→ Active question: {reasoning.get('active_question')}")
        print(f"→ Total supporting_evidences : {len(supporting_evidence)}")

        all_new_evidence = []
        for action in reasoning["actions"]:
            endpoint_url, payload = get_endpoint_and_payload(action, context)
            print(f"  [{action['type']}] Query: {action['query']}")
            try:
                evidence = await call_fastapi_async(endpoint_url, payload)
            except Exception as e:
                print(f"    !!! Endpoint error: {e}")
                evidence = []
            print(f"    ↳ {len(evidence)} evidence found")
            for ev in evidence:
                print("    ", pretty_evidence(ev))

            reasoning_log.append({
                "iteration": iteration,
                "action": action['type'],
                "query": action['query'],
                "result_count": len(evidence),
            })
            all_new_evidence.extend(evidence)
        
        context.extend(all_new_evidence)
        active_question = reasoning.get("active_question", active_question)
        previous_hypotheses = reasoning.get("previous_hypotheses", previous_hypotheses)
        agent_thoughts = reasoning.get("agent_thoughts", agent_thoughts)
        if reasoning.get("finalize"):
            print("\n=== FINALIZED ===")
            print(f"Final hypothesis: {reasoning.get('hypothesis')}")
            print(f"Supporting evidence: {len(reasoning.get('new_facts', []))}")
            break

    print("\n--- Reasoning complete ---\nFull log:")
    for step in reasoning_log:
        print(f"{step['iteration']}: [{step['action']}] {step['query']} ({step['result_count']})")
    print("\nContext size:", len(context))

    print("\n--- Supporting Evidence ---")
    for xxx in supporting_evidence:
        print(f"Supp: {xxx}")


    
    output = {
        "user_query": user_query,
        "final_hypothesis": reasoning.get("hypothesis"),
        "supporting_evidence": supporting_evidence,
        "reasoning_log": reasoning_log,
        "context": context,
        "agent_thoughts": agent_thoughts,
        "previous_hypotheses": previous_hypotheses,
    }
    #dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = "agent_run.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n=== Saved agent reasoning to: {fname} ===\n")


user_query = "Find all information and facts related to the object or place named Bararoá."
await agent_loop(user_query)


with open("/kaggle/working/agent_run.json", "r", encoding="utf-8") as f:
    data = json.load(f)
evidence_list = data["supporting_evidence"]
evidence_text_blocks = []
for item in evidence_list:
    evidence_text_blocks.append(f"Supp: {item}")
evidence_text = "\n".join(evidence_text_blocks)


class OpenAIClient:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", timeout=900, retries=3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _post(self, endpoint, data, idempotency_key=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key or str(uuid.uuid4()),
        }
        last_exception = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exception = e
        raise last_exception

    def chat_completion(self, **kwargs):
        return self._post("chat/completions", kwargs)

    def embedding(self, **kwargs):
        return self._post("embeddings", kwargs)

    def responses(self, **kwargs):
        return self._post("responses", kwargs)

llm_client = OpenAIClient(api_key=OpenAI_key)


system_prompt = """
You are a historical research assistant. Your task is to read the list of supporting evidence blocks related to the user's query, analyze and summarize all facts and clues regarding the geographical location of the object or place named in the question.

Carefully extract and organize all factual details such as coordinates, distances, directions, river names, historical events, and variant spellings.
Convert the evidence into a clear, human-readable summary for a non-expert.
If multiple sources mention similar facts (e.g. coordinates or locations), group and clarify them.
Explain the meaning of coordinate systems or conversions if present.
Include a short paragraph with the most probable modern location or area, if the data allows.

Never repeat the evidence blocks verbatim — always write an integrated narrative.
If there is some uncertainty or discrepancy, briefly mention it.

Your answer should be informative, concise, and accessible to a general audience.
"""

response = llm_client.chat_completion(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": evidence_text}
    ],
    max_tokens=600,
    temperature=0.2,
)
print(response["choices"][0]["message"]["content"])

