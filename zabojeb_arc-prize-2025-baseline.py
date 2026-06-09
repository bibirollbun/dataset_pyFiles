import json
import os


def ask_llm(task):
    """
    Placeholder LLM solver for an ARC task.

    Args:
        task (dict): A dictionary with keys:
            - 'train': list of {'input': grid, 'output': grid}
            - 'test': list of {'input': grid}
    Returns:
        List[dict]: A list of predictions for each test input, where each dict has:
            - 'attempt_1': predicted grid
            - 'attempt_2': predicted grid
    """

    # For baseline, simply echo the test input as both attempts
    predictions = []
    for test_pair in task.get('test', []):
        inp = test_pair['input']
        predictions.append({
            'attempt_1': inp,
            'attempt_2': inp
        })

    return predictions


# Path to the test challenges file (adjust if needed)
input_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
output_path = '/kaggle/working/submission.json'

# Load tasks
with open(input_path, 'r') as f:
    tasks = json.load(f)

# Build submission dict
submission = {}
for task_id, task in tasks.items():
    submission[task_id] = ask_llm(task)

# Write submission
with open(output_path, 'w') as f:
    json.dump(submission, f)

print(f"Submission saved to {output_path}")

