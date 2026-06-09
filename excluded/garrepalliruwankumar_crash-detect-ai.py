# ----------------------------------------------------------
# CrashDetect AI - Final Submission Code (No Arguments Needed)
# ----------------------------------------------------------

import re
from collections import Counter
import pandas as pd


# ----------------------------------------------------------
# Load Logs
# ----------------------------------------------------------
def load_logs(path="logs.txt"):
    try:
        with open(path, "r") as f:
            return f.readlines()
    except FileNotFoundError:
        print("logs.txt not found! Creating a sample file...")
        with open("logs.txt", "w") as f:
            f.write("INFO: Sample log created.\n")
        return ["INFO: Sample log created.\n"]


# ----------------------------------------------------------
# Classify Log Lines
# ----------------------------------------------------------
def classify(line):
    line_u = line.upper()
    if "ERROR" in line_u:
        return "ERROR"
    elif "WARNING" in line_u:
        return "WARNING"
    return "INFO"


# ----------------------------------------------------------
# Timestamp Extraction
# ----------------------------------------------------------
timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

def extract_timestamp(line):
    match = re.search(timestamp_pattern, line)
    return match.group(0) if match else None


# ----------------------------------------------------------
# Root Cause Suggestions (Offline Rule-Based)
# ----------------------------------------------------------
patterns = {
    "connection refused": "Check database/network connectivity.",
    "timeout": "High system load or slow service response.",
    "not found": "Missing file or resource. Verify paths.",
    "permission": "Check file or system permissions.",
    "crash": "Process stopped unexpectedly. Review logs before crash."
}

def suggest_fix(line):
    lower = line.lower()
    for key in patterns:
        if key in lower:
            return patterns[key]
    return "No automatic suggestion available."


# ----------------------------------------------------------
# Analyze Log Lines
# ----------------------------------------------------------
def analyze(logs):
    errors, warnings, info = [], [], []

    for line in logs:
        category = classify(line)

        entry = {
            "timestamp": extract_timestamp(line),
            "log": line.strip(),
            "suggestion": suggest_fix(line)
        }

        if category == "ERROR":
            errors.append(entry)
        elif category == "WARNING":
            warnings.append(entry)
        else:
            info.append(entry)

    return errors, warnings, info


# ----------------------------------------------------------
# Recurring Errors
# ----------------------------------------------------------
def recurring_errors(errors):
    items = [e["log"] for e in errors]
    return Counter(items).most_common()


# ----------------------------------------------------------
# Create Submission File (TXT + PARQUET)
# ----------------------------------------------------------
def create_submission(errors, warnings, info, recurring):
    lines = []

    lines.append("=== CrashDetect AI Report ===")
    lines.append("")
    lines.append(f"Total Errors: {len(errors)}")
    lines.append(f"Total Warnings: {len(warnings)}")
    lines.append(f"Total Info Messages: {len(info)}")
    lines.append("")

    lines.append("=== Recurring Errors ===")
    if recurring:
        for text, count in recurring:
            lines.append(f"{text}  --> {count} times")
    else:
        lines.append("No recurring errors found.")
    lines.append("")

    lines.append("=== Prevention Tips ===")
    lines.append("- Monitor logs frequently for repeating issues.")
    lines.append("- Address root causes instead of temporary fixes.")
    lines.append("- Review system performance during peak load times.")
    lines.append("- Validate configuration before deployment.")
    lines.append("")

    # Save submission.txt
    with open("submission.txt", "w") as f:
        f.write("\n".join(lines))

    # Save submission.parquet
    df = pd.DataFrame({
        "errors": [len(errors)],
        "warnings": [len(warnings)],
        "info": [len(info)]
    })
    df.to_parquet("submission.parquet")

    print("âœ… Submission files created:")
    print(" - submission.txt")
    print(" - submission.parquet")


# ----------------------------------------------------------
# MAIN EXECUTION (No Arguments Needed)
# ----------------------------------------------------------
if __name__ == "__main__":
    print("CrashDetect AI Running...\n")

    logs = load_logs()
    errors, warnings, info = analyze(logs)
    recurring = recurring_errors(errors)

    create_submission(errors, warnings, info, recurring)

    print("\nðŸŽ‰ Completed! Your submission files are ready.")

