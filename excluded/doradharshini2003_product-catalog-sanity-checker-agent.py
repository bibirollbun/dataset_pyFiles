import pandas as pd
from io import StringIO



session_state = {
    "logs": [],
    "last_df_summary": None,
    "last_report": None
}

def log_event(event_type, details):
    session_state["logs"].append({"event": event_type, "details": details})

def load_csv_from_text(csv_text: str) -> pd.DataFrame:
    log_event("tool_load_csv", "Loading CSV from text")
    return pd.read_csv(StringIO(csv_text))

def check_rules(df: pd.DataFrame):
    log_event("tool_check_rules", "Checking rules on dataframe")
    issues = []
    for idx, row in df.iterrows():
        row_issues = []
        if pd.isna(row.get("name")) or row.get("name") == "":
            row_issues.append("Missing name")
        if pd.isna(row.get("category")) or row.get("category") == "":
            row_issues.append("Missing category")
        price = row.get("price")
        stock = row.get("stock")
        if pd.notna(price) and price <= 0:
            row_issues.append("Non‑positive price")
        if pd.notna(stock) and (stock < 0 or stock > 10000):
            row_issues.append("Suspicious stock")
        if row_issues:
            issues.append({"row_index": idx, "product_id": row.get("product_id"), "issues": row_issues})
    return issues

def compute_quality_score(issues):
    log_event("tool_score", "Computing quality score")
    base = 100
    penalty = 2 * len(issues)
    return max(0, base - penalty)



def data_scan_agent(df: pd.DataFrame):
    log_event("agent_data_scan", "Scanning for nulls and negatives")
    scan = {
        "row_count": len(df),
        "null_counts": df.isna().sum().to_dict()
    }
    return scan

def rules_agent(df: pd.DataFrame, scan_results):
    log_event("agent_rules", "Applying business rules")
    return check_rules(df)

def report_agent(scan_results, rules_results, score):
    log_event("agent_report", "Generating text report (no Gemini)")
    lines = []
    lines.append(f"Rows analysed: {scan_results['row_count']}")
    lines.append(f"Null values per column: {scan_results['null_counts']}")
    lines.append(f"Total rows with rule violations: {len(rules_results)}")
    lines.append(f"Overall catalog quality score: {score}/100")
    if rules_results:
        lines.append("\nKey issues:")
        for r in rules_results[:10]:
            lines.append(f"- product_id={r['product_id']} (row {r['row_index']}): {', '.join(r['issues'])}")
    else:
        lines.append("No rule violations found.")
    return "\n".join(lines)





def orchestrator_agent(df: pd.DataFrame):
    log_event("agent_orchestrator", "Starting pipeline")
    scan_results = data_scan_agent(df)
    rules_results = rules_agent(df, scan_results)
    score = compute_quality_score(rules_results)
    report = report_agent(scan_results, rules_results, score)

    session_state["last_df_summary"] = scan_results
    session_state["last_report"] = report
    log_event("agent_orchestrator", "Pipeline finished")
    return report



sample_csv_text = """product_id,name,price,stock,category
1,Phone X,699,50,Electronics
2,,0,10,Electronics
3,Laptop Y,1200,-5,Electronics
4,Shoes Z,49,20000,Fashion
5,Book A,10,5,
"""

df = load_csv_from_text(sample_csv_text)
report = orchestrator_agent(df)

print("=== GENERATED REPORT ===")
print(report)

print("\n=== LOGS ===")
for log in session_state["logs"]:
    print(log)


