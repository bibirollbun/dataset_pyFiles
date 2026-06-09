import json
import os
from datetime import datetime

# Path to the example JSON file in Kaggle dataset
input_path = "/kaggle/input/openai-gpt-oss-20b-red-teaming/example-harmony-findings.json"

# Load the example JSON
with open(input_path, "r") as f:
    example_data = json.load(f)

# Display top-level keys (optional, for guidance)
print("Top-level keys in the example JSON:", example_data.keys())

# Function to create a new findings JSON
def make_findings_json(example, issue_title, observed_behavior, desired_behavior, severity="1", breadth="1"):
    new_finding = example.copy()
    
    # Update issue-specific details
    new_finding["issue_title"] = issue_title
    new_finding["issue_summary"]["observed_behavior"] = observed_behavior
    new_finding["issue_summary"]["desired_behavior"] = desired_behavior
    new_finding["issue_summary"]["self_assessed_severity"] = severity
    new_finding["issue_summary"]["self_assessed_breadth"] = breadth
    
    # Example: you can update the harmony response walkthroughs if needed
    new_finding["harmony_response_walkthroughs"] = [
        "<|start|>system<|message|>You are ChatGPT, a helpful assistant.<|end|>"
        "<|start|>user<|message|>REPLACE_WITH_YOUR_PROMPT<|end|>"
        "<|start|>assistant<|message|>REPLACE_WITH_YOUR_MODEL_OUTPUT<|end|>"
    ]
    
    return new_finding

# Create a new finding
my_finding = make_findings_json(
    example=example_data,
    issue_title="REPLACE: Example Issue Title",
    observed_behavior="REPLACE: What the model currently does",
    desired_behavior="REPLACE: What the model should do",
    severity="1",
    breadth="1"
)

# Save the new findings JSON
timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
output_path = f"/kaggle/working/myteam.findings.{timestamp}.json"

with open(output_path, "w") as f:
    json.dump(my_finding, f, indent=4)

print(f"New findings JSON saved at: {output_path}")


