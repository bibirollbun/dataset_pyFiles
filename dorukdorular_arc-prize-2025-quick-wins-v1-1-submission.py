import json
import sys
import os

# Add framework
sys.path.insert(0, '/kaggle/input/arc-solver-framework-v1')

from improved_solver import solve_with_two_attempts

print('Framework loaded!')


# Load data
DATA_PATH = '/kaggle/input/arc-prize-2025/'

with open(os.path.join(DATA_PATH, 'arc-agi_test_challenges.json'), 'r') as f:
    test_challenges = json.load(f)

with open(os.path.join(DATA_PATH, 'arc-agi_training_challenges.json'), 'r') as f:
    training_challenges = json.load(f)

print(f'Test tasks: {len(test_challenges)}')
print(f'Training tasks: {len(training_challenges)}')


# Generate predictions
submission = {}

for task_id, task_data in test_challenges.items():
    training_examples = None
    if task_id in training_challenges:
        training_examples = training_challenges[task_id].get('train', [])
    
    predictions = []
    for test_example in task_data['test']:
        attempt_1, attempt_2 = solve_with_two_attempts(
            task_id,
            test_example['input'],
            training_examples
        )
        predictions.append({
            'attempt_1': attempt_1,
            'attempt_2': attempt_2
        })
    
    submission[task_id] = predictions

print(f'Generated predictions for {len(submission)} tasks')


# Save submission
with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)

print('Submission saved to /kaggle/working/submission.json')
print(f'File size: {os.path.getsize("/kaggle/working/submission.json")} bytes')

