# CODE
# 1) Install dependencies (run this cell first)
# - google-genai (Gemini SDK) for interacting with Gemini
# - google-adk (if available) -- we will show ADK concepts in-notebook
# - transformers fallback (optional)
# - pandas for nice tables

!pip install --quiet google-genai google-adk pandas
# Note: If `google-adk` is not available in Kaggle environment at time of running,
# the notebook implements a small ADK-like orchestration in pure python.




# CODE
# 2) Imports & lightweight logger
import os, ast, logging, json
from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codedoc-agent")



import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Load API key from Kaggle secrets
user_secrets = UserSecretsClient()
os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")

def make_genai_client():
    """
    Configure Gemini client using GOOGLE_API_KEY (Kaggle secret).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Add it in Kaggle Secrets.")
    genai.configure(api_key=api_key)
    return genai

def genai_generate_text(prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512):
    """
    Generate text using Gemini (new SDK).
    """
    genai_client = make_genai_client()
    model_client = genai_client.GenerativeModel(model)
    response = model_client.generate_content(prompt)
    return response.text



# CODE
# 4) AST parser (custom tool)
def parse_code_with_ast(code: str) -> Dict[str, Any]:
    """
    Extract simple structure from Python function using ast.
    Returns dict: name, args, num_lines, returns_present, docstring, top_comments
    """
    result = {"error": None}
    try:
        tree = ast.parse(code)
    except Exception as e:
        return {"error": f"parse_error: {e}"}
    func_node = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break
    if func_node is None:
        # fallback: return top-level summary
        lines = [ln for ln in code.splitlines() if ln.strip()]
        return {"name": None, "args": [], "num_lines": len(lines), "has_returns": False, "docstring": None}
    args = [a.arg for a in func_node.args.args]
    num_body_stmts = len(func_node.body)
    has_return = any(isinstance(n, ast.Return) for n in ast.walk(func_node))
    doc = ast.get_docstring(func_node)
    # capture comments in the code (simple heuristic)
    comments = [line.strip() for line in code.splitlines() if line.strip().startswith("#")]
    return {"name": func_node.name, "args": args, "num_body_stmts": num_body_stmts, "has_return": has_return, "docstring": doc, "comments": comments}



# CODE
# 5) Define simple agent classes
@dataclass
class ParserAgent:
    name: str = "ParserAgent"
    def run(self, code: str) -> Dict[str, Any]:
        logger.info("[%s] parsing code...", self.name)
        parsed = parse_code_with_ast(code)
        return {"parsed": parsed, "code": code}

@dataclass
class DocstringAgent:
    name: str = "DocstringAgent"
    model: str = "gemini-2.5-flash"  # change if you have access to larger model

    def run(self, parsed_bundle: Dict[str, Any]) -> Dict[str, Any]:
        code = parsed_bundle["code"]
        parsed = parsed_bundle["parsed"]
        logger.info("[%s] generating docstring with Gemini model=%s...", self.name, self.model)

        # Construct prompt for Gemini: ask for docstring + 1-sentence summary + assumptions if any
        prompt = f"""
You are an assistant that writes clear Python function docstrings suitable for learners.
Given the function code below, produce:
1) A proper triple-quoted docstring (describe purpose, args with types/meaning if clear, return).
2) A one-sentence summary.
3) Any assumptions you made.

Code:

Parsed info:
{json.dumps(parsed)}

Return as JSON with keys: docstring, summary, assumptions.
"""
        # Call Gemini
        gen_text = genai_generate_text(prompt, model=self.model, max_output_tokens=512)
        # Try cleaning: often the model will return text; we keep it raw
        return {"docstring_raw": gen_text, "parsed": parsed, "code": code}

@dataclass
class EvaluatorAgent:
    name: str = "EvaluatorAgent"
    model: str = "gemini-2.5-flash"  # optional usage for a second opinion

    def run(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate docstring clarity with simple heuristics and optional LLM-based scoring.
        - Heuristic checks: presence of 'Args'/'Returns', length.
        - If Gemini is accessible, ask Gemini for a 1-5 clarity score and suggestion.
        """
        doc = bundle.get("docstring_raw", "")
        code = bundle.get("code", "")
        parsed = bundle.get("parsed", {})
        # Heuristic
        has_args = ("Args" in doc) or ("Parameters" in doc) or ("Returns" in doc) or ("return" in doc.lower())
        length = len(doc.split())
        heur_score = 3
        if has_args and length > 30:
            heur_score = 5
        elif has_args and length > 12:
            heur_score = 4
        elif length < 10:
            heur_score = 2
        else:
            heur_score = 3

        # Optional Gemini eval (wrapped in try to avoid hard failure)
        gemini_eval = None
        try:
            prompt = f"Rate the clarity of this docstring (1-5 integer) and give one brief suggestion.\n\nCode:\n```\n{code}\n```\nDocstring:\n```\n{doc}\n```\nReturn output as JSON: {{\"score\":<int>, \"suggestion\":\"...\"}}"
            gen_text = genai_generate_text(prompt, model=self.model, max_output_tokens=120)
            gemini_eval = gen_text
        except Exception as e:
            logger.warning("Gemini eval failed: %s", e)
            gemini_eval = None

        return {"heuristic_score": heur_score, "gemini_eval_raw": gemini_eval}




# CODE
# 6) Simple Session store
class SimpleSessionStore:
    def __init__(self):
        self.events = []  # each event is a dict with code, docstring, eval, timestamp if desired
    def append(self, event: Dict[str, Any]):
        self.events.append(event)
    def to_df(self):
        return pd.DataFrame(self.events)

session_store = SimpleSessionStore()



# CODE
# 7) Pipeline orchestration function
parser = ParserAgent()
doc_agent = DocstringAgent()
eval_agent = EvaluatorAgent()

def run_pipeline(code_snippet: str, model_for_doc: str = "gemini-2.5-flash"):
    # Parser
    p_out = parser.run(code_snippet)
    # Docstring agent (use given model)
    doc_agent.model = model_for_doc
    d_out = doc_agent.run(p_out)
    # Evaluate
    bundle = {**p_out, **d_out}
    e_out = eval_agent.run(bundle)
    # Aggregate
    result = {**bundle, **e_out}
    # Store in session
    session_store.append({"code": code_snippet, "docstring": d_out.get("docstring_raw", ""), "eval_heur": e_out.get("heuristic_score"), "eval_gemini": e_out.get("gemini_eval_raw")})
    return result

# Examples
examples = [
    # Messy/obfuscated variable names
    """
def var_7(var_21, var_28):
    if var_21>0:
        var_28 = var_28 * 2
        return var_28
    else:
        return None
""",
    # Missing docstring but descriptive names
    """
def compute_average(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)
""",
    # Slightly more complex with comments
    """
def merge_lists(a, b):
    # append b into a if not present
    for item in b:
        if item not in a:
            a.append(item)
    return a
"""
]

results = []
for code in examples:
    print("\n--- Running example ---")
    res = run_pipeline(code, model_for_doc="gemini-2.5-flash")
    results.append(res)
    print("Docstring (raw):\n", res.get("docstring_raw")[:800])
    print("Heuristic score:", res.get("heuristic_score"))
    print("Gemini eval (raw):\n", (res.get("gemini_eval_raw") or "")[:400])



# CODE
# 8) Show session events as a table
df = session_store.to_df()
# shorten long text for display
def short(x, n=200):
    if x is None: return ""
    s = str(x)
    return s if len(s)<=n else s[:n]+"..."
df_display = df.copy()
df_display["docstring_short"] = df_display["docstring"].apply(lambda x: short(x, 240))
df_display["eval_gemini_short"] = df_display["eval_gemini"].apply(lambda x: short(x, 240))
display(df_display[["code","docstring_short","eval_heur","eval_gemini_short"]])





