import json
import numpy as np
import time
import os


# ğŸ”� Load ARC dataset - handles both test and evaluation sets
base_path = "/kaggle/input/arc-prize-2025"

print("Available ARC dataset files:")
if os.path.exists(base_path):
    for f in os.listdir(base_path):
        print(f" - {f}")

# Try test challenges first (for competition), then evaluation
challenges = None
data_source = None

try:
    test_path = f"{base_path}/arc-agi_test_challenges.json"
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            challenges = json.load(f)
        data_source = "test_challenges (COMPETITION MODE)"
    else:
        eval_path = f"{base_path}/arc-agi_evaluation_challenges.json"
        with open(eval_path, 'r') as f:
            challenges = json.load(f)
        data_source = "evaluation_challenges (DEVELOPMENT MODE)"
except Exception as e:
    print(f"Error loading Kaggle data: {e}")
    # Fallback for local testing
    challenges = {
        "sample": {
            "train": [{"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]}],
            "test": [{"input": [[0,0,1],[1,0,0],[0,1,1]]}]
        }
    }
    data_source = "fallback sample"

print(f"\nğŸ“Š Loaded {len(challenges)} tasks from {data_source}")


def qrix_solver(train_pairs, test_input):
    """Victory36 qRIX solver with enhanced pattern recognition"""
    try:
        test_array = np.array(test_input)
        
        if not train_pairs or len(train_pairs) == 0:
            return test_input
        
        # Use first training example for pattern detection
        input_train = np.array(train_pairs[0]["input"])
        output_train = np.array(train_pairs[0]["output"])
        
        # Strategy 1: Same size transformations
        if input_train.shape == output_train.shape:
            # Binary inversion pattern
            unique_vals = np.unique(np.concatenate([input_train.flatten(), output_train.flatten()]))
            if len(unique_vals) <= 2 and np.array_equal(input_train, 1 - output_train):
                result = (1 - test_array)
                return np.clip(result, 0, 9).tolist()
            
            # Border filling pattern
            if np.sum(output_train) > np.sum(input_train):
                result = np.copy(test_array)
                if test_array.shape[0] >= 2 and test_array.shape[1] >= 2:
                    result[0, :] = 1; result[-1, :] = 1
                    result[:, 0] = 1; result[:, -1] = 1
                return result.tolist()
        
        # Strategy 2: Size scaling patterns
        elif output_train.shape != input_train.shape:
            if input_train.shape[0] > 0 and input_train.shape[1] > 0:
                scale_h = test_array.shape[0] / input_train.shape[0]
                scale_w = test_array.shape[1] / input_train.shape[1]
                result = np.zeros_like(test_array)
                for i in range(min(output_train.shape[0], result.shape[0])):
                    for j in range(min(output_train.shape[1], result.shape[1])):
                        new_i, new_j = int(i * scale_h), int(j * scale_w)
                        if new_i < result.shape[0] and new_j < result.shape[1]:
                            result[new_i, new_j] = output_train[i, j]
                return result.tolist()
        
        return test_input
    except Exception as e:
        print(f"qRIX solver error on task: {e}")
        return test_input

print("âœ… qRIX solver loaded successfully")


# ğŸš€ Generate ALL predictions
print("ğŸ�¯ Victory36 qRIX - Generating Predictions")
print("=" * 60)
print(f"ğŸš€ Processing ALL {len(challenges)} tasks...")

results = {}
start_time = time.time()

# Process ALL tasks (no subset limit for full 400)
for i, (task_id, task_data) in enumerate(challenges.items()):
    if i % 100 == 0 and i > 0:
        print(f"   ... processed {i} tasks")
    
    task_results = []
    for test_case in task_data["test"]:
        prediction = qrix_solver(task_data["train"], test_case["input"])
        task_results.append({
            "attempt_1": prediction,
            "attempt_2": prediction  # Deterministic solver
        })
    results[task_id] = task_results

# Build complete submission
submission = {
    "_metadata": {
        "submission": "Victory36 qRIX",
        "team": "Victory36 Labs / AI Publishing International LLP",
        "contact": "pr@coaching2100.com",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tasks_processed": len(challenges),
        "license": "CC BY 4.0",
        "citation": "Chollet, Francois, et al. 'ARC Prize 2025.' Kaggle, 2025",
        "processing_time": f"{time.time() - start_time:.3f}s"
    },
    **results
}

# Save submission
with open('submission.json', 'w') as f:
    json.dump(submission, f)

print(f"âœ… Generated {len(challenges)} predictions in {time.time() - start_time:.3f}s")
print(f"ğŸ“� Saved to: submission.json")


# ğŸ”� Validation Cell
print("ğŸ”� VICTORY36 qRIX - Submission Validation")
print("=" * 60)

try:
    with open("submission.json", "r") as f:
        submission = json.load(f)

    # Task keys
    task_keys = [k for k in submission.keys() if not k.startswith("_")]
    num_tasks = len(task_keys)

    # Task Count
    print(f"ğŸ“Š Task Count: {num_tasks} tasks")
    expected = 400 if "test_challenges" in data_source else len(challenges)
    print("   Status:", "âœ… PERFECT" if num_tasks == expected else f"âš ï¸� {num_tasks}/{expected}")

    # Structure Check (sample 5 tasks)
    invalid_structs = []
    for task_id in list(task_keys)[:5]:
        entries = submission[task_id]
        if not isinstance(entries, list) or len(entries) == 0:
            invalid_structs.append(f"{task_id}: not a list")
            continue
        if not all(isinstance(e, dict) and "attempt_1" in e and "attempt_2" in e for e in entries):
            invalid_structs.append(f"{task_id}: missing attempt keys")

    print(f"\nğŸ�—ï¸� Structure Check: {'âœ… PERFECT' if not invalid_structs else 'â�Œ Issues found'}")
    if invalid_structs:
        for err in invalid_structs[:3]:
            print("   -", err)

    # Metadata Check
    required_fields = ["submission", "team", "contact", "timestamp", "tasks_processed", "license", "citation"]
    missing_meta = [f for f in required_fields if f not in submission["_metadata"]]

    print(f"\nğŸ“‹ Metadata Check: {'âœ… COMPLETE' if not missing_meta else 'â�Œ Missing'}")
    if missing_meta:
        print("   Missing fields:", missing_meta)

    # Final Status
    all_good = not invalid_structs and num_tasks == expected and not missing_meta
    print(f"\nğŸ“ˆ FINAL STATUS: {'ğŸ�‰ SUBMISSION READY' if all_good else 'âš ï¸� NEEDS REVIEW'}")
    
    if all_good:
        print(f"\nğŸ�† Victory36 qRIX submission validated: {num_tasks} tasks processed successfully!")
    
except Exception as e:
    print(f"â�Œ Validation failed: {e}")

