import json
import numpy as np
from pathlib import Path


# Define paths
data_path = Path('/kaggle/input/arc-prize-2025')
test_challenges_path = data_path / 'arc-agi_test_challenges.json'


def simple_baseline_prediction(train_examples, test_input):
    """
    Simple baseline: try common patterns from ARC
    1. Copy input to output
    2. Try patterns from training examples
    """
    # Strategy 1: Just return the input (identity function)
    attempt_1 = test_input

    # Strategy 2: Try to apply transformation from first training example
    if train_examples and len(train_examples) > 0:
        first_train = train_examples[0]
        train_input = first_train['input']
        train_output = first_train['output']

        # If same shape, try copying the output pattern
        if len(train_input) == len(test_input) and len(train_input[0]) == len(test_input[0]):
            attempt_2 = train_output
        else:
            # Otherwise, try to scale or repeat
            out_h = len(train_output)
            out_w = len(train_output[0]) if out_h > 0 else 0
            attempt_2 = [[0 for _ in range(out_w)] for _ in range(out_h)] if out_h > 0 else test_input
    else:
        attempt_2 = test_input

    return attempt_1, attempt_2


# Load test challenges
with open(test_challenges_path, 'r') as f:
    test_challenges = json.load(f)

print(f"Loaded {len(test_challenges)} test tasks")


# Create submission dictionary
submission = {}

# Generate predictions for each task
for task_id, task_data in test_challenges.items():
    train_examples = task_data['train']
    test_examples = task_data['test']

    task_predictions = []

    # For each test case in the task
    for test_case in test_examples:
        test_input = test_case['input']

        # Generate two attempts
        attempt_1, attempt_2 = simple_baseline_prediction(train_examples, test_input)

        task_predictions.append({
            'attempt_1': attempt_1,
            'attempt_2': attempt_2
        })

    # Update submission
    submission[task_id] = task_predictions

print(f"Generated predictions for {len(submission)} tasks")


# Save submission
with open('submission.json', 'w') as f:
    json.dump(submission, f)

print("Submission saved to submission.json")
print(f"File size: {Path('submission.json').stat().st_size / 1024:.1f} KB")


# Validate submission format
print("\nValidating submission format...")
valid = True
for task_id, predictions in submission.items():
    if not isinstance(predictions, list):
        print(f"ERROR: {task_id} predictions is not a list")
        valid = False
    for i, pred in enumerate(predictions):
        if 'attempt_1' not in pred or 'attempt_2' not in pred:
            print(f"ERROR: {task_id} test case {i} missing attempts")
            valid = False

if valid:
    print("✓ Validation passed!")
else:
    print("✗ Validation failed!")

