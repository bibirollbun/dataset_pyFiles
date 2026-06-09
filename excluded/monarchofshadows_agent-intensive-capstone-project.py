import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent as LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


import re
import requests
from bs4 import BeautifulSoup
def extract_claim_from_url(url: str) -> dict:
    """
    Fetch a web page and extract a single main factual claim.

    Args:
        url (str): Public URL of a news article / post.

    Returns:
        dict: {
          "success": bool,
          "raw_text": str,
          "candidate_claim": str
        }
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return {
            "success": False,
            "raw_text": "",
            "candidate_claim": f"Error while fetching URL: {e}"
        }

    soup = BeautifulSoup(resp.text, "html.parser")

    # Very crude: title + first <h1>/<h2> + first long <p>
    title = soup.title.string if soup.title else ""
    headings = " ".join(
        h.get_text(" ", strip=True)
        for h in soup.find_all(["h1", "h2"])[:2]
    )
    first_para = ""
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text.split()) > 12:
            first_para = text
            break

    raw = " ".join([title, headings, first_para]).strip()
    raw = re.sub(r"\s+", " ", raw)

    if not raw:
        return {
            "success": False,
            "raw_text": "",
            "candidate_claim": "Could not extract any meaningful text from page."
        }

    # Let the LLM refine it, but we still give a best guess
    candidate = raw[:400]

    return {
        "success": True,
        "raw_text": raw,
        "candidate_claim": candidate
    }



def similar_claims_lookup(normalized_claim: str) -> dict:
    """
    Look up previously seen claims that are similar to the input claim.

    Args:
        normalized_claim (str): Canonical English claim sentence.

    Returns:
        dict: {
          "matches": [
            {
              "claim_id": str,
              "normalized_claim": str,
              "verdict": str,
              "confidence": int,
              "similarity": float,
              "evidence_summary": str,
              "timestamp": str
            }, ...
          ]
        }
    """
    matches = memory_store.search_similar_claims(normalized_claim)
    return {"matches": matches}


def save_verdict_to_memory(
    normalized_claim: str,
    verdict: str,
    confidence: int,
    evidence_summary: str
) -> dict:
    """
    Persist the final verdict for future similar-claim retrieval.

    Args:
        normalized_claim (str): Canonical English claim.
        verdict (str): One of true/mostly_true/misleading/false/unverifiable.
        confidence (int): 0-100 confidence score.
        evidence_summary (str): Short bullet summary of key evidence.

    Returns:
        dict: {"success": bool, "claim_id": str}
    """
    cid = memory_store.add_claim(
        normalized_claim=normalized_claim,
        verdict=verdict,
        confidence=confidence,
        evidence_summary=evidence_summary,
    )
    return {"success": True, "claim_id": cid}


from dataclasses import dataclass, asdict
# from typing import List, Dict
import uuid
import time
import math

@dataclass
class ClaimRecord:
    claim_id: str
    normalized_claim: str
    verdict: str
    confidence: int
    evidence_summary: str
    created_at: float

class MemoryStore:
    """
    Toy in-memory store for claims.

    In a real deployment youâ€™d back this with a DB or vector store.
    """

    def __init__(self) -> None:
        self._claims: List[ClaimRecord] = []

    def add_claim(
        self,
        normalized_claim: str,
        verdict: str,
        confidence: int,
        evidence_summary: str,
    ) -> str:
        cid = str(uuid.uuid4())
        rec = ClaimRecord(
            claim_id=cid,
            normalized_claim=normalized_claim.strip(),
            verdict=verdict,
            confidence=int(confidence),
            evidence_summary=evidence_summary.strip(),
            created_at=time.time(),
        )
        self._claims.append(rec)
        return cid

    def search_similar_claims(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Extremely crude "similarity": token-level Jaccard.

        Youâ€™re free to replace this with real embeddings for extra points.
        """
        q_tokens = set(query.lower().split())
        scored: list[tuple[float, ClaimRecord]] = []

        for rec in self._claims:
            r_tokens = set(rec.normalized_claim.lower().split())
            if not r_tokens:
                continue
            inter = len(q_tokens & r_tokens)
            union = len(q_tokens | r_tokens)
            sim = inter / union if union else 0.0
            if sim > 0.25:  # cheap filter
                scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict] = []
        for sim, rec in scored[:top_k]:
            d = asdict(rec)
            d["similarity"] = float(round(sim, 3))
            out.append(d)
        return out

memory_store = MemoryStore()



MODEL_NAME = "gemini-2.5-flash-lite" 
APP_NAME = "fake_news_reality_court"

claim_normalizer = LlmAgent(
    name="claim_normalizer",
    model=MODEL_NAME,
    description="Extracts and normalizes the main factual claim from user input.",
    instruction="""
You are the Claim Extraction & Normalization agent.

Input will be either:
- a raw text claim, OR
- a URL starting with http/https.

Your job:
1. If input looks like a URL, call the tool `extract_claim_from_url` to fetch page content.
2. From the raw text (either user text or fetched content), identify the single main factual claim.
3. Rewrite it as ONE short English sentence that could be labeled true/false/misleading/unverifiable.
4. Remove opinions, rhetorical wording, clickbait, and side-notes.
5. Return STRICT JSON with keys:
   - normalized_claim: str
   - claim_type: "text" | "url_extracted"
   - language: ISO code of ORIGINAL language, e.g., "en", "hi".
6. Do NOT try to check whether the claim is true.

ALWAYS answer ONLY with JSON, no explanation.
""",
    tools=[extract_claim_from_url],   # custom FunctionTool, NO built-in tools here
    output_key="normalized_claim_json",
)


similar_case_retriever = LlmAgent(
    name="similar_case_retriever",
    model=MODEL_NAME,
    description="Looks up similar previously-checked claims from long-term memory.",
    instruction="""
You are the Similar-Case Retriever agent.

Context:
- A previous agent stored its JSON output in state under key `normalized_claim_json`.
- That JSON has key `normalized_claim`.

Your tasks:
1. Read the `normalized_claim_json` from state.
2. Extract the `normalized_claim` field.
3. Call the tool `similar_claims_lookup` with this normalized claim.
4. The tool returns {"matches": [...]} containing past cases.

You must output STRICT JSON:
{
  "normalized_claim": "...",
  "similar_cases": [ ... ]
}

Do NOT invent matches. Only use what the tool returns.
Do NOT do any fact-checking here.
""",
    tools=[similar_claims_lookup],
    output_key="similar_cases_json",
)


prosecutor_agent = LlmAgent(
    name="prosecutor_agent",
    model=MODEL_NAME,
    description="Builds strongest possible case that the claim is false or misleading.",
    instruction="""
You are the Prosecutor in a fake-news court.

State contains:
- `normalized_claim_json`
- `similar_cases_json`

Your tasks:
1. Read the normalized claim from `normalized_claim_json.normalized_claim`.
2. Read any similar past cases from `similar_cases_json.similar_cases`.
3. Use the `google_search` tool to find information that suggests the claim is FALSE, MISLEADING, OUTDATED, or LACKS EVIDENCE.
   Use multiple queries:
   - exact claim in quotes
   - claim + "fact check"
   - claim + "hoax", "myth", or "debunked"
4. For each relevant search result, decide if it:
   - directly contradicts the claim,
   - shows it's taken out of context,
   - or shows it's an old claim no longer true.
5. Optionally reuse similar past cases that had verdict "false" or "misleading".

Scoring credibility:
- Without extra tools, YOU rate domain credibility:
  high: major news orgs, gov sites, academic, big fact-checkers;
  medium: regional news, known blogs;
  low: random blogs, forums, unknown sites.
Return credibility_score from 0.0 (very low) to 1.0 (very high).

You must output STRICT JSON:
{
  "side": "prosecution",
  "normalized_claim": "...",
  "arguments": [
    {
      "claim_fragment": "...",
      "stance": "against",
      "reasoning": "...",
      "evidence": [
        {
          "url": "...",
          "title": "...",
          "snippet": "...",
          "domain": "...",
          "credibility_score": 0.0,
          "published_date": "YYYY-MM-DD or null"
        }
      ]
    }
  ]
}

Do NOT declare a final verdict.
""",
    tools=[google_search],   # built-in tool; no custom tools here
    output_key="prosecution_json",
)


defender_agent = LlmAgent(
    name="defender_agent",
    model=MODEL_NAME,
    description="Builds strongest possible case that the claim is true or mostly true.",
    instruction="""
You are the Defender in a fake-news court.

State contains:
- `normalized_claim_json`
- `similar_cases_json`

Your tasks:
1. Read the normalized claim.
2. Use the `google_search` tool to find information that SUPPORTS the claim.
   Queries:
   - exact claim
   - claim + "official site", "press release", "data", "report".
3. Prefer:
   - official gov, institutional or academic sources,
   - multiple independent high-cred sources,
   - recent information.

Score domain credibility as in the Prosecutor instructions.

You must output STRICT JSON:
{
  "side": "defense",
  "normalized_claim": "...",
  "arguments": [
    {
      "claim_fragment": "...",
      "stance": "for",
      "reasoning": "...",
      "evidence": [
        {
          "url": "...",
          "title": "...",
          "snippet": "...",
          "domain": "...",
          "credibility_score": 0.0,
          "published_date": "YYYY-MM-DD or null"
        }
      ]
    }
  ]
}

Do NOT declare a final verdict.
""",
    tools=[google_search],
    output_key="defense_json",
)


evidence_auditor = LlmAgent(
    name="evidence_auditor",
    model=MODEL_NAME,
    description="Cleans and critiques prosecution and defense evidence.",
    instruction="""
You are the Evidence Auditor.

State contains:
- `prosecution_json`
- `defense_json`

These are JSON strings from the Prosecutor and Defender.

Your tasks:
1. Parse both JSON structures.
2. Remove clearly useless entries:
   - duplicate URLs,
   - obviously unrelated pages,
   - evidence with credibility_score < 0.2 unless it quotes high-cred sources.
3. Identify obvious gaps or weaknesses on each side.

You MUST return STRICT JSON:
{
  "cleaned_prosecution": {...},
  "cleaned_defense": {...},
  "issues": [
    "string describing an audit issue",
    ...
  ]
}

Do NOT make up new evidence. Only prune/annotate.
Do NOT issue any verdict.
""",
    tools=[],   # NO tools â€“ pure reasoning
    output_key="audited_evidence_json",
)


judge_agent = LlmAgent(
    name="judge_agent",
    model= MODEL_NAME,
    description="Final arbiter that assigns verdict + confidence.",
    instruction="""
You are the Judge in "Reality Court".

State contains:
- `normalized_claim_json`
- `similar_cases_json`
- `audited_evidence_json`

Possible verdict labels:
- "true"
- "mostly_true"
- "misleading"
- "false"
- "unverifiable"

Your tasks:
1. Parse all three JSON blobs from state.
2. Weigh strength of evidence:
   - number & quality of high-cred sources on each side,
   - presence of major fact-checkers,
   - recency,
   - whether evidence directly addresses the claim or only loosely.
3. Use similar past cases only as weak prior info, never as authority.
4. Choose ONE verdict label.
5. Assign `confidence` from 0â€“100:
   - 90â€“100: strong multi-source agreement,
   - 60â€“89: decent but not overwhelming,
   - 40â€“59: mixed/weak,
   - < 40: basically guesswork â†’ usually "unverifiable".
6. Create a short bullet-point rationale referencing the evidence.

Return STRICT JSON:
{
  "normalized_claim": "...",
  "verdict": "true|mostly_true|misleading|false|unverifiable",
  "confidence": 0-100,
  "rationale": [
    "bullet 1",
    "bullet 2"
  ],
  "key_evidence": [
    {
      "url": "...",
      "title": "...",
      "domain": "...",
      "stance": "for|against",
      "short_reason": "why this evidence matters"
    }
  ]
}

Do NOT talk about agents or tools in the rationale.
""",
    tools=[],   # pure LLM
    output_key="judge_json",
)


verdict_persister = LlmAgent(
    name="verdict_persister",
    model=MODEL_NAME,
    description="Persists judge verdict into long-term memory.",
    instruction="""
You are the Verdict Persister.

State contains:
- `normalized_claim_json`
- `judge_json`

Your tasks:
1. Parse both JSON blobs.
2. Extract:
   - normalized_claim
   - verdict
   - confidence
   - short textual evidence summary (e.g., concatenate rationale bullets).
3. Call the `save_verdict_to_memory` tool with those values.
4. Return STRICT JSON:
{
  "claim_id": "...",
  "normalized_claim": "...",
  "verdict": "...",
  "confidence": 0-100
}

Do NOT change the verdict or confidence.
""",
    tools=[save_verdict_to_memory],
    output_key="persistence_json",
)



output_formatter = LlmAgent(
    name="output_formatter",
    model=MODEL_NAME,
    description="Formats judge JSON into a user-friendly court-style answer.",
    instruction="""
You are the Output Formatter.

State contains:
- `normalized_claim_json`
- `judge_json`
- `persistence_json`

Your job is ONLY formatting. Do NOT change verdict/confidence.

Produce a concise, user-facing answer with this structure:

1. "Claim:" <normalized claim>
2. "Verdict:" <label> (confidence XX/100)
3. Bullet list "Why:" with 3â€“6 bullets from rationale.
4. "Key evidence:" list, each item like:
   - [domain] title â€“ short_reason (URL)

Keep it short and readable.

Return plain text (no JSON).
""",
    tools=[],
    # This is the final visible response; no output_key needed.
)

# -------------------------------------------------------------------
# WORKFLOW: parallel + sequential
# -------------------------------------------------------------------

# Evidence gathering in parallel
evidence_parallel = ParallelAgent(
    name="evidence_parallel",
    sub_agents=[prosecutor_agent, defender_agent],
)

# Main pipeline
root_agent = SequentialAgent(
    name="fake_news_reality_court",
    sub_agents=[
        claim_normalizer,
        similar_case_retriever,
        evidence_parallel,
        evidence_auditor,
        judge_agent,
        verdict_persister,
        output_formatter,
    ],
)


USER_ID = "user1"

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

await run_session(
    runner,
    ["Humans use only 10% of their brain"],
    "session1"
)




