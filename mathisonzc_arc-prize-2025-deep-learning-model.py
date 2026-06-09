import json
import numpy as np
from collections import Counter
import copy


def predict_with_pattern_analysis(train_examples, test_input):
    """
    Pattern-based prediction using training examples
    """
    # Strategy 1: Check if it's a copy task
    is_identity = True
    for ex in train_examples:
        if not ex['input'] or not ex['output']:
            is_identity = False
            break
        if len(ex['input']) != len(ex['output']):
            is_identity = False
            break
        if len(ex['input'][0]) != len(ex['output'][0]):
            is_identity = False
            break
        if ex['input'] != ex['output']:
            is_identity = False
            break

    if is_identity and len(train_examples) > 0:
        return test_input

    # Strategy 2: Check for consistent output size
    output_sizes = []
    for ex in train_examples:
        if ex['output']:
            h = len(ex['output'])
            w = len(ex['output'][0]) if ex['output'] else 0
            output_sizes.append((h, w))

    if output_sizes and len(set(output_sizes)) == 1:
        # All outputs have the same size
        if len(train_examples) > 0:
            return train_examples[0]['output']

    # Strategy 3: Check for scaling patterns
    scale_factors = []
    for ex in train_examples:
        if ex['input'] and ex['output']:
            in_h, in_w = len(ex['input']), len(ex['input'][0])
            out_h, out_w = len(ex['output']), len(ex['output'][0])
            if in_h > 0 and in_w > 0:
                scale_factors.append((out_h / in_h, out_w / in_w))

    if scale_factors and len(set(scale_factors)) == 1:
        # Consistent scaling
        scale_h, scale_w = scale_factors[0]
        test_h, test_w = len(test_input), len(test_input[0]) if test_input else 0
        out_h = int(test_h * scale_h)
        out_w = int(test_w * scale_w)

        # Create scaled output
        output = []
        for i in range(out_h):
            row = []
            for j in range(out_w):
                src_i = int(i / scale_h)
                src_j = int(j / scale_w)
                if src_i < test_h and src_j < test_w:
                    row.append(test_input[src_i][src_j])
                else:
                    row.append(0)
            output.append(row)
        return output

    # Default: return input
    return test_input


# Load test challenges
test_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'

print("Loading test challenges...")
with open(test_challenges_path, 'r') as f:
    test_challenges = json.load(f)

print(f"Loaded {len(test_challenges)} test tasks")


# Generate predictions
predictions = {}

print("\nGenerating predictions...")
for idx, (task_id, task_data) in enumerate(test_challenges.items(), 1):
    if idx % 50 == 0:
        print(f"Progress: {idx}/{len(test_challenges)}")

    train_examples = task_data['train']
    test_examples = task_data['test']

    task_predictions = []

    for test_case in test_examples:
        test_input = test_case['input']

        # Attempt 1: Pattern-based analysis
        try:
            attempt_1 = predict_with_pattern_analysis(train_examples, test_input)
        except Exception:
            attempt_1 = test_input

        # Attempt 2: Simple heuristic
        try:
            if train_examples:
                first_train = train_examples[0]
                in_h = len(first_train['input'])
                in_w = len(first_train['input'][0]) if first_train['input'] else 0
                test_h = len(test_input)
                test_w = len(test_input[0]) if test_input else 0

                if in_h == test_h and in_w == test_w:
                    attempt_2 = first_train['output']
                else:
                    attempt_2 = test_input
            else:
                attempt_2 = test_input
        except Exception:
            attempt_2 = test_input

        task_predictions.append({
            'attempt_1': attempt_1,
            'attempt_2': attempt_2
        })

    predictions[task_id] = task_predictions

print(f"\nGenerated predictions for {len(predictions)} tasks")


# Save submission
with open('submission.json', 'w') as f:
    json.dump(predictions, f)

print("Submission saved to submission.json")

