%%writefile compare_submissions.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
import time
import importlib.util
import copy
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Load user function from code path
def load_user_function(code_path: str):
    spec = importlib.util.spec_from_file_location("user", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "p"):
        raise AttributeError("no function 'p'")
    return mod.p

# Measure one task: return (success, time)
def measure_one(task_id: int, code_path: str, task_dir: str) -> tuple[bool, float]:
    start = time.time()
    try:
        data = json.load(open(os.path.join(task_dir, f"task{task_id:03d}.json")))
        examples = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
        p = load_user_function(code_path)
        for ex in examples:
            inp = copy.deepcopy(ex["input"])
            out = p(inp)
            out_str = json.dumps(out).replace("true","1").replace("false","0")
            if re.search(r"[^0-9,\[\]\s\.]", out_str):
                raise ValueError("Invalid output")
            user_grid = np.array(json.loads(out_str))
            if not np.array_equal(user_grid, np.array(ex["output"])):
                raise ValueError("Mismatch")
    except Exception:
        elapsed = time.time() - start
        return False, elapsed
    elapsed = time.time() - start
    return True, elapsed

# Measure one submission: summary + task details
def measure_submission(sub_dir: Path, task_dir: str) -> Dict:
    py_files = sorted(sub_dir.glob("task*.py"))
    if not py_files:
        raise ValueError(f"{sub_dir} has no .py files.")
    total_bytes = sum(f.stat().st_size for f in py_files)

    task_details: List[Dict] = []
    ok_cnt = 0
    for tid in range(1, 401):
        py_path = sub_dir / f"task{tid:03d}.py"
        detail = {"task": tid}
        if not py_path.exists():
            detail.update({"time": 0.0, "bytes": 0})
            task_details.append(detail)
            continue
        detail["bytes"] = py_path.stat().st_size
        try:
            ok, elapsed = measure_one(tid, str(py_path), task_dir)
            detail.update({"time": elapsed})
            if ok:
                ok_cnt += 1
        except Exception:
            detail.update({"time": 0.0, "bytes": detail["bytes"]})
        task_details.append(detail)

    times = [t["time"] for t in task_details if t["time"] > 0]
    total_time = sum(times) if times else 0.0
    avg_time = np.mean(times) if times else 0.0
    max_time = max(times) if times else 0.0
    min_time = min(times) if times else 0.0

    return {
        "files": len(py_files),
        "bytes": total_bytes,
        "total_sec": round(total_time, 3),
        "avg_sec": round(avg_time, 3),
        "max_sec": round(max_time, 3),
        "min_sec": round(min_time, 3),
        "correct": ok_cnt,
        "missing": 400 - len(py_files),
        "task_details": task_details,
    }

# Main function
def main():
    parser = argparse.ArgumentParser(description="Compare submissions: code size, time (per task/total)")
    parser.add_argument("folders", nargs="+", help="Submission folders")
    parser.add_argument("--task_dir", default="/kaggle/input/google-code-golf-2025/", help="task*.json dir")
    args = parser.parse_args()

    results: List[Dict] = []
    for i, p in enumerate(args.folders):
        sub_dir = Path(p).resolve()
        if not sub_dir.is_dir():
            print(f"Warning: {p} is not a directory. Skipping.")
            continue
        print(f"Measuring: {sub_dir.name} ...")
        res = measure_submission(sub_dir, args.task_dir)
        res["name"] = f"Sub {i+1} ({sub_dir.name})"
        results.append(res)

    if not results:
        print("No submissions measured.")
        return

    # Summary DF
    df_summary = pd.DataFrame(results)
    df_summary = df_summary[["name", "files", "missing", "bytes", "total_sec", "avg_sec", "max_sec", "min_sec", "correct"]]
    df_summary = df_summary.sort_values(["bytes", "total_sec", "correct"], ascending=[True, True, False]).reset_index(drop=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    def highlight_best(s):
        if s.name in ["bytes", "total_sec", "avg_sec", "max_sec", "min_sec"]:
            return ["background-color: #d4edda" if v == s.min() else "" for v in s]
        if s.name == "correct":
            return ["background-color: #d4edda" if v == s.max() else "" for v in s]
        return [""] * len(s)

    styled_summary = (
        df_summary.style.format({
            "bytes": "{:,}",
            "total_sec": "{:.3f}",
            "avg_sec": "{:.3f}",
            "max_sec": "{:.3f}",
            "min_sec": "{:.3f}",
        })
        .apply(highlight_best)
        .set_caption("Submission Summary Comparison")
    )

    # Console output
    print("\n" + "="*80)
    print(" " * 30 + "SUBMISSION COMPARISON TABLE")
    print("="*80)
    print(df_summary.to_markdown(index=False, tablefmt="grid"))
    print("="*80)

    # Per-task comparison DF (side-by-side: task, bytes_1, time_1, bytes_2, time_2, ...)
    if results:
        df_tasks = pd.DataFrame(results[0]["task_details"])[["task"]]
        for res in results:
            sub_name = res["name"]
            temp_df = pd.DataFrame(res["task_details"])
            df_tasks[f"{sub_name}_bytes"] = temp_df["bytes"]
            df_tasks[f"{sub_name}_time"] = temp_df["time"].round(3)

    styled_tasks = df_tasks.style.set_caption("Per-Task Comparison (Side-by-Side)")

    # HTML content
    html_content = f"""
    <html><head><meta charset="utf-8"><style>
    table {{font-family: monospace; border-collapse: collapse; width: 100%;}}
    th, td {{border: 1px solid #ccc; padding: 8px; text-align: center;}}
    th {{background-color: #f2f2f2;}}
    </style></head><body>
    <h2>Submission Summary Comparison</h2>
    {styled_summary.to_html()}
    <h2>Per-Task Comparison (Side-by-Side)</h2>
    {styled_tasks.to_html()}
    </body></html>
    """

    html_path = "submission_comparison.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nHTML report saved: {html_path}")

if __name__ == "__main__":
    main()


!python compare_submissions.py \
    /kaggle/input/code-golf-sample \
    /kaggle/input/code-golf-sample \
    /kaggle/input/code-golf-sample


from IPython.display import HTML, display

html_content = "/kaggle/working/submission_comparison.html"
display(HTML(html_content))

