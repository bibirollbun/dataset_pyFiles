import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


!pip install --quiet google-genai


# Core imports for ADK and GenAI
from typing import Dict, Any
import asyncio, time, re

# ADK framework imports
from google.adk.agents import Agent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext

# GenAI client + types
from google.genai import types
from google import genai

print("ADK and GenAI imports OK")


# Create GenAI client using the API key
API_KEY = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)

class LLMHelper:
    """Simple helper that wraps genai.Client with retry logic."""

    def __init__(self, client, model_name: str = "models/gemini-2.0-flash"):
        self.client = client
        self.model_name = model_name

    def _extract_text(self, resp):
        if getattr(resp, "text", None):
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            return getattr(resp.candidates[0], "text", "")
        return ""

    def generate_with_retries(
        self,
        prompt: str,
        max_output_tokens: int = 500,
        temperature: float = 0.0,
        attempts: int = 5,
        exp_base: float = 7.0,
        initial_delay: float = 1.0,
        retry_on_statuses=(429, 500, 503, 504),
    ) -> str:

        last_exc = None

        for attempt in range(1, attempts + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"max_output_tokens": max_output_tokens, "temperature": temperature},
                )
                text = self._extract_text(resp)
                if text and text.strip():
                    return text
                last_exc = RuntimeError("Empty response")

            except Exception as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(e, "status", None)

                if status is None:
                    # fallback: extract status from error text
                    m = re.search(r"\b(4\d{2}|5\d{2})\b", str(e))
                    status = int(m.group()) if m else None

                if status is not None and status not in retry_on_statuses:
                    raise

            # retry backoff
            if attempt < attempts:
                wait = initial_delay * (exp_base ** (attempt - 1))
                wait = min(wait, 60.0)
                import random
                sleep_time = wait + random.uniform(-wait * 0.1, wait * 0.1)
                print(f"[Retry {attempt}] Waiting {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)
                continue

            # all retries failed
            raise last_exc

# Create the helper instance
llm_helper = LLMHelper(client)
print("LLM helper ready (Gemini Flash).")



# Prompt builders 

def meal_plan_prompt(days: int, diet: str, budget: int) -> str:
    """Return a strict plain-text prompt to generate a readable meal plan."""
    template = "\n".join(
        [
            f"DAY {d}:\n"
            "Breakfast:\n"
            "Lunch:\n"
            "Dinner:\n"
            for d in range(1, days + 1)
        ]
    )
    return f"""Generate a clean, human-readable {days}-day {diet} meal plan.
Budget: {budget} INR/day

Output EXACTLY like this â€” plain text, no quotes, no parentheses, no code/text markers:

{template}

Rules:
- Use simple plain text only.
- DO NOT add quotes, parentheses, Python-style strings, or tuple formatting.
- DO NOT wrap output in backticks or code fences.
"""

def shopping_prompt(plan_text: str) -> str:
    """Return a strict plain-text prompt to generate a grouped shopping list."""
    return f"""Generate a clean, human-readable grouped shopping list for ONE PERSON.

Meal Plan:
{plan_text}

Format EXACTLY like this â€” plain text:

GRAINS & PULSES:
- item â€” qty
- item â€” qty

VEGETABLES:
- item â€” qty

FRUITS:
- item â€” qty

DAIRY / PROTEIN:
- item â€” qty

OTHER:
- item â€” qty

Rules:
- DO NOT use bold/markdown symbols.
- Use '-' for bullets only.
- DO NOT wrap output in quotes or parentheses.
"""

def cost_prompt(plan_text: str) -> str:
    """Return a strict plain-text prompt to generate a cost estimate."""
    return f"""Estimate cost for ONE PERSON. Output MUST be clean plain text.

Meal Plan:
{plan_text}

Format EXACTLY like this:

Daily Cost (INR): <number>
Total Cost (INR): <number>

CALCULATION:
- item: qty @ â‚¹unit_price/unit = â‚¹cost
- item: qty @ â‚¹unit_price/unit = â‚¹cost

Rules:
- DO NOT add explanations or extra text.
- DO NOT use markdown or wrap output in quotes/parentheses.
"""

def nutrition_prompt(plan_text: str) -> str:
    """Return a strict plain-text prompt to generate calories+protein per meal/day."""
    return f"""Generate clean calorie + protein breakdown for each meal and day.

Meal Plan:
{plan_text}

Format EXACTLY like this â€” plain text:

DAY 1:
- Breakfast: Calories: <kcal>, Protein: <g>
- Lunch: Calories: <kcal>, Protein: <g>
- Dinner: Calories: <kcal>, Protein: <g>
TOTAL Day 1: Calories: <kcal>, Protein: <g>

Repeat for all days.

Rules:
- NO quotes, NO parentheses, NO JSON, NO extra commentary.
- Output plain, readable text only.
"""



# Cell 6 â€” ADK-compatible agent classes
from typing import Dict, Any
import asyncio, re

class PlannerAgent(Agent):
    def __init__(self, name: str = "planner"):
        super().__init__(name=name)

    async def handle(self, query: str, tool_context: ToolContext = None) -> Dict[str, Any]:
        q = query.lower()

        
        m_days = re.search(r"(\d+)\s*(day|days|d)", q)
        if not m_days:
            m_days = re.search(r"\b(\d+)\b", q)
        days = int(m_days.group(1)) if m_days else 3  

        
        m_budget = re.search(r"(\d+)\s*(rs|inr|â‚¹|rupees)", q)
        if not m_budget:
            m_budget = re.search(r"budget\s*:?(\d+)", q)
        budget = int(m_budget.group(1)) if m_budget else 500  

        # --- DIET LOGIC ---
        if ("non veg" in q) or ("non-veg" in q) or ("non vegetarian" in q):
            diet = "non-vegetarian"
        elif ("mixed" in q) or ("both" in q) or ("veg and non" in q):
            diet = "mixed"
        elif ("veg" in q or "vegetarian" in q):
            diet = "vegetarian"
        else:
            diet = "omnivore"

        
        prompt = meal_plan_prompt(days=days, diet=diet, budget=budget)
        text = await asyncio.to_thread(llm_helper.generate_with_retries, prompt, 1500, 0.0)

        return {
            "ok": True,
            "text": text,
            "plan": text,   # stored for follow-up agents
            "meta": {"days": days, "budget": budget, "diet": diet}
        }


class ShoppingAgent(Agent):
    def __init__(self, name: str = "shopping"):
        super().__init__(name=name)

    async def handle(self, query: str, tool_context: ToolContext = None) -> Dict[str, Any]:
        # Get last_plan from context ONLY
        plan = tool_context.get("last_plan") if isinstance(tool_context, dict) else None
        if not plan:
            return {"ok": False, "error": "No plan found. Generate a plan first."}

        prompt = shopping_prompt(plan)
        text = await asyncio.to_thread(llm_helper.generate_with_retries, prompt, 700, 0.0)
        return {"ok": True, "text": text}


class CostAgent(Agent):
    def __init__(self, name: str = "cost"):
        super().__init__(name=name)

    async def handle(self, query: str, tool_context: ToolContext = None) -> Dict[str, Any]:
        # Get last_plan from context ONLY
        plan = tool_context.get("last_plan") if isinstance(tool_context, dict) else None
        if not plan:
            return {"ok": False, "error": "No plan found. Generate a plan first."}

        prompt = cost_prompt(plan)
        text = await asyncio.to_thread(llm_helper.generate_with_retries, prompt, 400, 0.0)
        return {"ok": True, "text": text}


class NutritionAgent(Agent):
    def __init__(self, name: str = "nutrition"):
        super().__init__(name=name)

    async def handle(self, query: str, tool_context: ToolContext = None) -> Dict[str, Any]:
        # Get last_plan from context ONLY
        plan = tool_context.get("last_plan") if isinstance(tool_context, dict) else None
        if not plan:
            return {"ok": False, "error": "No plan found. Generate a plan first."}

        prompt = nutrition_prompt(plan)
        text = await asyncio.to_thread(llm_helper.generate_with_retries, prompt, 900, 0.0)
        return {"ok": True, "text": text}



#: AutoOrchestrator
from typing import Dict, Any
import asyncio, time, re

class AutoOrchestrator(Agent):
    def __init__(self, agents: Dict[str, Agent], name: str = "orchestra"):
        super().__init__(name=name)
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "context", {})
        object.__setattr__(self, "_timeout", 30)

   
    def _intent(self, q: str) -> str:
        q = q.lower()

        # Shopping
        if any(k in q for k in ["shopping", "grocery", "ingredients", "list"]):
            return "shopping"

        # Cost / Pricing
        if any(k in q for k in ["cost", "price", "estimate", "â‚¹", "rs", "total"]):
            return "cost"

        # Nutrition
        if any(k in q for k in ["nutrition", "calorie", "protein", "macro"]):
            return "nutrition"

        # Explicit plan request
        if re.search(r"\b\d+\s*(day|days|d)\b", q) or "meal plan" in q:
            return "planner"

       
        return "planner"

    # SAFE AGENT EXEC 
    async def _safe_call(self, name: str, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.agents.get(name)
        if agent is None:
            return {"ok": False, "error": f"agent '{name}' missing"}
        try:
            return await asyncio.wait_for(agent.handle(query, tool_context=ctx), timeout=self._timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"{name} timeout"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    
    async def handle(self, query: str, tool_context: Dict[str, Any] = None) -> Dict[str, Any]:
        ctx = self.context
        intent = self._intent(query)

        # If follow-up agent needs plan but there is none
        if intent in ("shopping", "cost", "nutrition") and not ctx.get("last_plan"):
            planner_res = await self._safe_call("planner", query, ctx)
            if planner_res.get("ok"):
                ctx["last_plan"] = planner_res.get("plan")
                ctx.update(planner_res.get("meta", {}))
            else:
                return {"ok": False, "error": "Auto-plan failed: " + planner_res.get("error", "unknown")}

        
        res = await self._safe_call(intent, query, ctx)

        # If planner ran, save plan
        if intent == "planner" and res.get("ok"):
            ctx["last_plan"] = res.get("plan")
            ctx.update(res.get("meta", {}))

        return res



agents = {
    "planner": PlannerAgent(name="planner"),
    "shopping": ShoppingAgent(name="shopping"),
    "cost": CostAgent(name="cost"),
    "nutrition": NutritionAgent(name="nutrition"),
}

orchestra = AutoOrchestrator(agents=agents)
print("AutoOrchestrator ready.")



# Demo 

queries = [
    "Create a 5-day mixed meal plan with budget 1000 INR/day",
    "Give me a shopping list for the last plan",
    "Estimate cost for the last plan",
    "Analyze nutrition for the last plan"
]

for q in queries:
    print("\n" + "="*40)
    print("QUERY:", q)
    print("="*40 + "\n")

    out = await orchestra.handle(q, tool_context=orchestra.context)

    if isinstance(out, dict):
        if out.get("ok"):
            print(out.get("text", ""))
        else:
            print("ERROR:", out.get("error"))
    else:
        print(str(out))

    print("\n" + "-"*40 + "\n")





