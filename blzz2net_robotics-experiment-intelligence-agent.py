import pandas as pd
import numpy as np
from pathlib import Path
!ls /kaggle/input



from pathlib import Path

DATA_DIR = Path("/kaggle/input/synthetic-tb3-experiments-for-agent-analysis")

!ls "$DATA_DIR"



import pandas as pd
import numpy as np
from pathlib import Path

def load_run(file_path: Path) -> pd.DataFrame:
    """Load a single experiment CSV and validate columns."""
    df = pd.read_csv(file_path)
    required_cols = ["timestamp", "x", "y", "x_gt", "y_gt", "v_linear", "v_angular", "cpu", "memory"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")
    return df

def parse_config_from_name(filename: str) -> dict:
    """
    Parse config info from filename pattern:
    algorithma_gmappingon_5min.csv
    """
    base = Path(filename).stem.lower()  
    parts = base.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected filename pattern: {filename}")

    algo_raw = parts[0]   # 'algorithma'
    gmap_raw = parts[1]   # 'gmappingon'
    dur_raw  = parts[2]   # '5min'

    algo_letter = algo_raw.replace("algorithm", "").upper()  # 'A' or 'B'
    gmapping    = "ON" if "on" in gmap_raw else "OFF"
    duration    = int(dur_raw.replace("min", "").strip())

    return {
        "algorithm": f"Algorithm{algo_letter}",
        "gmapping": gmapping,
        "duration_min": duration,
    }

def compute_localization_metrics(df: pd.DataFrame) -> dict:
    """Compute localization + resource metrics for one run."""
    dx = df["x"] - df["x_gt"]
    dy = df["y"] - df["y_gt"]
    err = np.sqrt(dx**2 + dy**2)

    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    max_err = float(np.max(err))

    cpu_mean = float(df["cpu"].mean())
    mem_mean = float(df["memory"].mean())

    return {
        "rmse_error": rmse,
        "mae_error": mae,
        "max_error": max_err,
        "cpu_mean": cpu_mean,
        "memory_mean": mem_mean,
    }

def summarize_all_runs(data_dir: Path) -> pd.DataFrame:
    """Scan CSVs in data_dir and build a summary table with config + metrics."""
    rows = []
    for csv_path in data_dir.glob("*.csv"):
        name_lower = csv_path.name.lower()
        # Only process files that look like our experiment logs
        if not name_lower.startswith("algorithma_") and not name_lower.startswith("algorithmb_"):
            continue

        df = load_run(csv_path)
        cfg = parse_config_from_name(csv_path.name)
        metrics = compute_localization_metrics(df)

        row = {
            "file": csv_path.name,
            **cfg,
            **metrics,
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError(f"No valid experiment CSVs found in {data_dir}")

    return summary_df.sort_values(["algorithm", "gmapping", "duration_min"])



summary_df = summarize_all_runs(DATA_DIR)
summary_df



class MemoryStore:
    """Very simple in-notebook memory for past queries & results."""
    def __init__(self):
        self.history = []

    def add_entry(self, question: str, result: dict):
        self.history.append({"question": question, "result": result})

    def get_history(self):
        return self.history


class DataAnalysisAgent:
    def __init__(self, summary_df: pd.DataFrame):
        self.summary_df = summary_df

    def get_min_rmse(self):
        idx = self.summary_df["rmse_error"].idxmin()
        return self.summary_df.loc[idx].to_dict()

    def filter_by_gmapping(self, state: str):
        return self.summary_df[self.summary_df["gmapping"] == state].copy()

    def filter_by_duration(self, minutes: int):
        return self.summary_df[self.summary_df["duration_min"] == minutes].copy()


class ExplanationAgent:
    def explain_best_config(self, best_row: dict) -> str:
        return (
            f"The best configuration by RMSE is {best_row['algorithm']} with "
            f"Gmapping {best_row['gmapping']} for {best_row['duration_min']} minutes.\n"
            f"RMSE = {best_row['rmse_error']:.4f} m, "
            f"MAE = {best_row['mae_error']:.4f} m, "
            f"CPU ≈ {best_row['cpu_mean']:.1f}%, "
            f"Memory ≈ {best_row['memory_mean']:.1f} MB."
        )

    def explain_gmapping_effect(self, df_on: pd.DataFrame, df_off: pd.DataFrame, duration: int) -> str:
        mean_on = df_on["rmse_error"].mean()
        mean_off = df_off["rmse_error"].mean()
        diff = mean_off - mean_on
        direction = "lower" if diff > 0 else "higher"
        return (
            f"For {duration}-minute runs, Gmapping ON yields average RMSE of {mean_on:.4f} m, "
            f"while Gmapping OFF yields {mean_off:.4f} m. "
            f"That means Gmapping ON has {abs(diff):.4f} m {direction} error on average."
        )


class OrchestratorAgent:
    def __init__(self, summary_df: pd.DataFrame):
        self.memory = MemoryStore()
        self.analysis_agent = DataAnalysisAgent(summary_df)
        self.explainer = ExplanationAgent()

    def answer(self, question: str) -> str:
        question_lower = question.lower()

        if "minimum localization error" in question_lower or "lowest rmse" in question_lower:
            best = self.analysis_agent.get_min_rmse()
            text = self.explainer.explain_best_config(best)
            self.memory.add_entry(question, {"best_config": best})
            return text

        if "gmapping" in question_lower and "10 min" in question_lower or "10-minute" in question_lower:
            df_on = self.analysis_agent.filter_by_gmapping("ON")
            df_on = df_on[df_on["duration_min"] == 10]
            df_off = self.analysis_agent.filter_by_gmapping("OFF")
            df_off = df_off[df_off["duration_min"] == 10]
            text = self.explainer.explain_gmapping_effect(df_on, df_off, duration=10)
            self.memory.add_entry(question, {"gmapping_on": df_on.to_dict(), "gmapping_off": df_off.to_dict()})
            return text

        return "I understand your question, but this prototype currently supports only a small set of query types (best configuration, Gmapping vs non-Gmapping)."



agent = OrchestratorAgent(summary_df)

print(agent.answer("Which configuration has the minimum localization error?"))
print()
print(agent.answer("How does Gmapping affect localization for 10-minute runs?"))



class MemoryStore:
    """Simple in-notebook memory for past queries and results."""
    def __init__(self):
        self.history = []

    def add_entry(self, question: str, result: dict):
        self.history.append({"question": question, "result": result})

    def get_history(self):
        return self.history


class DataAnalysisAgent:
    def __init__(self, summary_df: pd.DataFrame):
        self.summary_df = summary_df

    def get_min_rmse(self) -> dict:
        idx = self.summary_df["rmse_error"].idxmin()
        return self.summary_df.loc[idx].to_dict()

    def filter_by_gmapping_and_duration(self, gmapping_state: str, duration_min: int) -> pd.DataFrame:
        return self.summary_df[
            (self.summary_df["gmapping"] == gmapping_state) &
            (self.summary_df["duration_min"] == duration_min)
        ].copy()


class ExplanationAgent:
    def explain_best_config(self, best_row: dict) -> str:
        return (
            f"The best configuration by RMSE is {best_row['algorithm']} "
            f"with Gmapping {best_row['gmapping']} for {int(best_row['duration_min'])} minutes.\n"
            f"RMSE = {best_row['rmse_error']:.4f} m, "
            f"MAE = {best_row['mae_error']:.4f} m, "
            f"max error = {best_row['max_error']:.4f} m,\n"
            f"CPU ≈ {best_row['cpu_mean']:.1f}%, "
            f"Memory ≈ {best_row['memory_mean']:.1f} MB."
        )

    def explain_gmapping_effect(self, df_on: pd.DataFrame, df_off: pd.DataFrame, duration: int) -> str:
        if df_on.empty or df_off.empty:
            return f"No matching runs found for {duration}-minute duration."

        mean_on = df_on["rmse_error"].mean()
        mean_off = df_off["rmse_error"].mean()
        diff = mean_off - mean_on
        if diff > 0:
            direction = "lower"
        elif diff < 0:
            direction = "higher"
        else:
            direction = "the same"

        return (
            f"For {duration}-minute runs:\n"
            f"- Gmapping ON:  average RMSE = {mean_on:.4f} m\n"
            f"- Gmapping OFF: average RMSE = {mean_off:.4f} m\n\n"
            f"On average, Gmapping ON shows {abs(diff):.4f} m {direction} error compared to Gmapping OFF."
        )


class OrchestratorAgent:
    """
    Very simple natural-language router.
    In a production deployment, this could be powered by Gemini to interpret
    arbitrary questions and decide which tools/agents to call.
    """
    def __init__(self, summary_df: pd.DataFrame):
        self.memory = MemoryStore()
        self.analysis_agent = DataAnalysisAgent(summary_df)
        self.explainer = ExplanationAgent()

    def answer(self, question: str) -> str:
        q = question.lower()

        # Query 1: best / minimum localization error
        if "minimum localization error" in q or "lowest rmse" in q or "best configuration" in q:
            best = self.analysis_agent.get_min_rmse()
            text = self.explainer.explain_best_config(best)
            self.memory.add_entry(question, {"best_config": best})
            return text

        # Query 2: gmapping effect for 10-minute runs
        if "gmapping" in q and "10" in q:
            df_on = self.analysis_agent.filter_by_gmapping_and_duration("ON", 10)
            df_off = self.analysis_agent.filter_by_gmapping_and_duration("OFF", 10)
            text = self.explainer.explain_gmapping_effect(df_on, df_off, duration=10)
            self.memory.add_entry(question, {"gmapping_on": df_on.to_dict(), "gmapping_off": df_off.to_dict()})
            return text

        # Fallback
        return (
            "I understand your question, but this prototype currently supports only a small set of "
            "query types, such as:\n"
            "- 'Which configuration has the minimum localization error?'\n"
            "- 'How does Gmapping affect localization for 10-minute runs?'"
        )



agent = OrchestratorAgent(summary_df)

print("Q1: Which configuration has the minimum localization error?\n")
print(agent.answer("Which configuration has the minimum localization error?"))

print("\n" + "="*80 + "\n")

print("Q2: How does Gmapping affect localization for 10-minute runs?\n")
print(agent.answer("How does Gmapping affect localization for 10-minute runs?"))


