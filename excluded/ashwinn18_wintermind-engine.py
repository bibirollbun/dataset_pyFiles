import os, json, tempfile, traceback, textwrap, datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Try to access Kaggle Secrets safely
GOOGLE_API_KEY_SECRET_LABEL = "GOOGLE_API_KEY"  # change if your secret uses another label

def get_google_api_key_from_kaggle_secrets(label: str = GOOGLE_API_KEY_SECRET_LABEL) -> Optional[str]:
    """
    Attempts to fetch a secret named `label` from Kaggle Secrets.
    Falls back to environment variable if not available.
    Returns None if no key found.
    """
    key = None
    try:
        # Kaggle environment provides kaggle_secrets
        from kaggle_secrets import UserSecretsClient
        try:
            key = UserSecretsClient().get_secret(label)
        except Exception:
            # secret missing or not set for this notebook
            key = None
    except Exception:
        # kaggle_secrets not available (not running on Kaggle)
        key = None

    # fallback to env var
    if not key:
        key = os.environ.get("GOOGLE_API_KEY")
    return key

# retrieve key (may be None)
GOOGLE_API_KEY = get_google_api_key_from_kaggle_secrets()

# If ace_tools is available in Kaggle environment use it to display DataFrames nicely
try:
    from ace_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None

# ---------------- WebTools (safe; will not error offline) ----------------
class WebTools:
    def __init__(self, api_key: Optional[str] = None):
        # prefer key passed in, else use env/secret
        self.key = api_key if api_key is not None else GOOGLE_API_KEY
        self.has_key = bool(self.key)

    def web_search(self, query: str, max_results: int = 3):
        """
        Uses Google Custom Search JSON API if key present and internet is available.
        Otherwise returns deterministic mock results.
        """
        if not self.has_key:
            # deterministic mock result so behavior is the same across runs
            return [{"title": f"Mock result for '{query}' (no key)", "link":"https://example.com/mock", "snippet":"Mock result because GOOGLE_API_KEY is not set."}]
        try:
            import requests
            # If you want to enable real search you must also set SEARCH_CX env var or secret.
            cx = os.environ.get("SEARCH_CX")
            if not cx:
                # no cx configured -> fallback
                return [{"title": f"Mock result for '{query}' (no CX)", "link":"https://example.com/mock", "snippet":"Missing SEARCH_CX; set SEARCH_CX env var or Kaggle secret."}]
            params = {"key": self.key, "cx": cx, "q": query, "num": min(max_results, 10)}
            r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            results = []
            for it in items[:max_results]:
                results.append({"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet", "")})
            if not results:
                return [{"title": f"No results for '{query}'", "link":"https://example.com/empty", "snippet":"No items returned from API"}]
            return results
        except Exception as e:
            # safe fallback — never raise
            return [{"title": f"Fallback for '{query}'", "link":"https://example.com/fallback", "snippet": f"error: {repr(e)}"}]

    def fetch_url_text(self, url: str) -> str:
        try:
            import requests
            resp = requests.get(url, timeout=6); resp.raise_for_status()
            return resp.text[:2000]
        except Exception as e:
            return f"[FAILED TO FETCH {url} — {repr(e)}]"

# ---------------- Agents ----------------
@dataclass
class AgentResponse:
    success: bool
    result: Any = None
    log: List[str] = field(default_factory=list)
    error: Optional[str] = None

class CodeAssistantAgent:
    def __init__(self, webtools: WebTools):
        self.web = webtools; self.log=[]
    def generate_analysis_code(self, task: str, data_var: str = "df") -> str:
        code = textwrap.dedent(f"""
        def _generated_analysis({data_var}):
            out = {{}}
            out['shape'] = {data_var}.shape
            out['columns'] = list({data_var}.columns)
            try:
                out['describe'] = {data_var}.describe().to_dict()
            except Exception:
                out['describe'] = "describe_failed"
            num = {data_var}.select_dtypes(include=['number']).corr().abs()
            if num.shape[0] > 0:
                corr_sums = num.sum().sort_values(ascending=False)
                out['top_correlated_columns'] = corr_sums.head(5).to_dict()
            else:
                out['top_correlated_columns'] = {{}}
            return out
        _generated_analysis(df)
        """)
        self.log.append(f"Generated code for: {task}")
        return code
    def execute_code(self, code: str, exec_globals: Optional[dict] = None, exec_locals: Optional[dict] = None) -> AgentResponse:
        g = {"__builtins__": __builtins__} if exec_globals is None else exec_globals
        l = {} if exec_locals is None else exec_locals
        try:
            exec(code, g, l)
            result = None
            if "_generated_analysis" in g:
                try:
                    result = g["_generated_analysis"](g.get("df", l.get("df")))
                except Exception:
                    result = None
            if result is None:
                result = l.get("result", g.get("result", None))
            return AgentResponse(success=True, result=result, log=self.log.copy())
        except Exception as e:
            tb = traceback.format_exc()
            return AgentResponse(success=False, error=str(e), log=self.log.copy()+[tb])

class DataAnalystAgent:
    def __init__(self, webtools: WebTools):
        self.web = webtools; self.log=[]
    def load_data(self, path: Optional[str] = None) -> AgentResponse:
        try:
            if path and os.path.exists(path):
                df = pd.read_csv(path)
                self.log.append(f"Loaded CSV from {path} shape={df.shape}")
            else:
                rng = np.random.default_rng(seed=42)
                n = 500
                df = pd.DataFrame({
                    "date": pd.date_range(end=pd.Timestamp.today(), periods=n).astype(str),
                    "product_id": rng.integers(1000, 1010, size=n),
                    "units_sold": rng.integers(0, 50, size=n),
                    "unit_price": np.round(rng.uniform(100, 2000, size=n), 2),
                    "region": rng.choice(["north","south","east","west"], size=n)
                })
                df["revenue"] = df["units_sold"] * df["unit_price"]
                self.log.append(f"Generated synthetic data shape={df.shape}")
            return AgentResponse(success=True, result=df, log=self.log.copy())
        except Exception as e:
            return AgentResponse(success=False, error=str(e), log=self.log.copy())
    def basic_summary(self, df: pd.DataFrame) -> AgentResponse:
        try:
            summary = {"shape": df.shape, "columns": list(df.columns), "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(), "head": df.head(5)}
            self.log.append("Computed basic summary")
            return AgentResponse(success=True, result=summary, log=self.log.copy())
        except Exception as e:
            return AgentResponse(success=False, error=str(e), log=self.log.copy())
    def plot_revenue_over_time(self, df: pd.DataFrame, save_path: Optional[str] = None) -> AgentResponse:
        try:
            df2 = df.copy(); df2["date"] = pd.to_datetime(df2["date"])
            daily = df2.groupby(df2["date"].dt.date)["revenue"].sum().reset_index()
            plt.figure(figsize=(10,4)); plt.plot(daily["date"], daily["revenue"]); plt.title("Daily revenue"); plt.xlabel("Date"); plt.ylabel("Revenue"); plt.tight_layout()
            if save_path: plt.savefig(save_path, bbox_inches="tight")
            plt.show()
            self.log.append("Plotted revenue")
            return AgentResponse(success=True, result={"plot_saved": bool(save_path)}, log=self.log.copy())
        except Exception as e:
            return AgentResponse(success=False, error=str(e), log=self.log.copy())

# ---------------- New: Class Teacher Agents (Days 1-3) ----------------
class ClassTeacherAgent:
    """
    Generic teacher agent that provides lesson materials, exercises, and references for a day.
    """
    def __init__(self, day_number: int, title: str, topics: List[str], exercises: List[str]):
        self.day = day_number
        self.title = title
        self.topics = topics
        self.exercises = exercises
        self.log = []

    def teach(self) -> Dict[str, Any]:
        """
        Returns a structured lesson for the day.
        """
        lesson = {
            "day": self.day,
            "title": self.title,
            "topics": self.topics,
            "exercises": self.exercises,
            "notes": self._generate_notes()
        }
        self.log.append(f"Taught day {self.day}: {self.title}")
        return lesson

    def _generate_notes(self) -> str:
        # Concrete short notes for the lesson
        lines = [f"Day {self.day} - {self.title}", ""]
        lines += [f"- {t}" for t in self.topics]
        lines += ["", "Exercises:"]
        for i, ex in enumerate(self.exercises, 1):
            lines.append(f"{i}. {ex}")
        return "\n".join(lines)

# Create three specific class teacher agents for Days 1-3 of the 5-day agent course
teacher_day1 = ClassTeacherAgent(
    day_number=1,
    title="Agents Foundations & Prompting",
    topics=[
        "Definition of an AI Agent",
        "Agent vs Model vs Tool",
        "Prompt engineering basics and intent decomposition",
        "Designing task breakdowns for agents"
    ],
    exercises=[
        "Decompose the task 'generate a marketing plan' into 6 agent tasks.",
        "Write 3 prompt templates for a Planner agent."
    ]
)

teacher_day2 = ClassTeacherAgent(
    day_number=2,
    title="Tools & Tooling for Agents",
    topics=[
        "Tool interfaces and tool safety",
        "Connecting tools: web search, calculators, file stores",
        "Retries, timeouts, and tool fallback strategies"
    ],
    exercises=[
        "Design a small tool spec (input/output) for a 'price fetcher' tool.",
        "Implement a safe retry logic pseudo-code for tool calls."
    ]
)

teacher_day3 = ClassTeacherAgent(
    day_number=3,
    title="Context, Memory & Multi-Agent Coordination",
    topics=[
        "Short-term vs long-term memory in agents",
        "State management and session stitching",
        "Design patterns for multi-agent orchestration"
    ],
    exercises=[
        "Design a memory schema for storing user preferences across sessions.",
        "Sketch an orchestrator flow for 3 cooperating agents."
    ]
)

# ---------------- Orchestrator (extended to include teaching) ----------------
class SimpleOrchestrator:
    def __init__(self, code_agent: CodeAssistantAgent, data_agent: DataAnalystAgent, webtools: WebTools, teachers: List[ClassTeacherAgent]=None):
        self.code_agent = code_agent; self.data_agent = data_agent; self.web = webtools
        self.teachers = teachers or []
        self.log = []

    def run_goal(self, goal: str, data_path: Optional[str] = None, include_lessons: bool = True) -> Dict[str, Any]:
        out = {"goal": goal, "timestamp": str(datetime.datetime.utcnow()), "steps": []}
        self.log.append(f"Start goal: {goal}")

        # 1. Optionally run lessons (Days 1-3)
        lessons = []
        if include_lessons and self.teachers:
            for t in self.teachers:
                lesson = t.teach()
                lessons.append(lesson)
                out["steps"].append({"step": f"lesson_day_{t.day}", "ok": True, "title": t.title})
        out["lessons"] = lessons

        # 2. load data
        load_resp = self.data_agent.load_data(data_path)
        out["steps"].append({"step": "load_data", "ok": load_resp.success})
        if not load_resp.success:
            out["error"] = load_resp.error
            return out
        df = load_resp.result

        # show data if ace_tools available
        if display_dataframe_to_user is not None:
            try:
                display_dataframe_to_user("Capstone Sample Data", df.head(100))
            except Exception:
                print(df.head())

        # 3. basic summary
        summ = self.data_agent.basic_summary(df)
        out["steps"].append({"step": "basic_summary", "ok": summ.success, "result_shape": summ.result.get("shape") if summ.success else None})
        if summ.success:
            out["summary"] = {"columns": summ.result["columns"], "dtypes": summ.result["dtypes"]}

        # 4. plot
        tmp_plot = os.path.join(tempfile.gettempdir(), "capstone_revenue_plot.png")
        plot_resp = self.data_agent.plot_revenue_over_time(df, save_path=tmp_plot)
        out["steps"].append({"step": "plot_revenue", "ok": plot_resp.success, "plot_path": tmp_plot if plot_resp.success else None})

        # 5. CodeAgent generates code and executes
        task = "Provide descriptive statistics and top correlated numeric columns"
        code = self.code_agent.generate_analysis_code(task=task, data_var="df")
        exec_globals = {"df": df, "__builtins__": __builtins__}
        exec_resp = self.code_agent.execute_code(code, exec_globals=exec_globals)
        out["steps"].append({"step": "code_execution", "ok": exec_resp.success})
        if exec_resp.success:
            out["analysis"] = exec_resp.result

        # 6. Persist outputs (lessons, summary, analysis, report)
        out_dir = os.path.join(tempfile.gettempdir(), "agents_capstone_outputs")
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf8") as f:
            json.dump({"goal": goal, "summary": out.get("summary"), "analysis": out.get("analysis"), "lessons": out.get("lessons")}, f, indent=2, default=str)
        out["summary_path"] = summary_path

        # Write markdown report (includes lessons)
        md_lines = [
            "# Agents Capstone Report",
            f"**Goal:** {goal}",
            f"**Generated at (UTC):** {out['timestamp']}",
            "",
            "## Lessons (Days 1-3)",
            ""
        ]
        for lesson in lessons:
            md_lines.append(f"### Day {lesson['day']}: {lesson['title']}")
            md_lines.append("")
            md_lines.append(lesson["notes"])
            md_lines.append("")

        md_lines += ["## Data Summary", ""]
        for c in out.get("summary", {}).get("columns", []):
            md_lines.append(f"- {c}")
        md_lines.append("")
        md_lines.append("## Analysis")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(out.get("analysis"), indent=2, default=str))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append(f"![plot]({tmp_plot})")
        md_text = "\n".join(md_lines)
        md_path = os.path.join(out_dir, "report_with_lessons.md")
        with open(md_path, "w", encoding="utf8") as f:
            f.write(md_text)
        out["report_path"] = md_path

        out["ok"] = True
        self.log.append("Completed orchestrator run")
        return out

# ---------------- Demo Run (edit DATA_PATH if you want real data) ----------------
def demo_run(data_path: Optional[str] = None, include_lessons: bool = True):
    print("=== Agents Intensive Demo Run (with Days 1-3 Teacher Agents) ===")
    webtools = WebTools(api_key=GOOGLE_API_KEY)
    code_agent = CodeAssistantAgent(webtools)
    data_agent = DataAnalystAgent(webtools)
    # pass teacher agents list
    teachers = [teacher_day1, teacher_day2, teacher_day3]
    orch = SimpleOrchestrator(code_agent, data_agent, webtools, teachers=teachers)

    GOAL = "Analyze sample sales data and produce a short summary with lessons for Days 1-3."
    result = orch.run_goal(GOAL, data_path=data_path, include_lessons=include_lessons)

    print("\n--- RESULT SUMMARY ---")
    keys_to_print = {k: v for k, v in result.items() if k not in ("analysis",)}
    print(json.dumps(keys_to_print, indent=2, default=str))
    if "analysis" in result and result["analysis"] is not None:
        print("\n--- ANALYSIS (top-level keys) ---")
        for k in result["analysis"].keys():
            print("-", k)
    print("\nReport saved to:", result.get("report_path"))
    print("Summary JSON saved to:", result.get("summary_path"))
    return result

# Run demo (leave DATA_PATH = None to use synthetic data)
DATA_PATH = None
result = demo_run(data_path=DATA_PATH, include_lessons=True)

# Print produced file locations and existence
out_dir = os.path.join(tempfile.gettempdir(), "agents_capstone_outputs")
print("\nFiles produced in:", out_dir)
for fname in ("report_with_lessons.md", "summary.json", "capstone_revenue_plot.png"):
    p = os.path.join(out_dir, fname)
    print(f"- {fname}: {p} (exists={os.path.exists(p)})")


