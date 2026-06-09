import json
import os
import random

def _shape_from_attempt(proto_attempt):
    """Return (H, W) from a sample attempt grid."""
    H = len(proto_attempt)
    W = len(proto_attempt[0]) if H > 0 else 0
    return H, W
def _zeros(H, W, fill=0): return [[int(fill) for _ in range(W)] for __ in range(H)]
def _random_grid(H, W, rng): return [[rng.randint(0, 9) for _ in range(W)] for __ in range(H)]

def build_submission_from_sample(sample_json_path, out_path="submission.json", seed=0):
    rng = random.Random(seed)
    with open(sample_json_path, "r") as f:
        sample = json.load(f)

    submission = {}
    total_tests = 0

    for task_id, sample_entries in sample.items():
        outputs = []
        for i, entry in enumerate(sample_entries):
            proto = entry.get("attempt_1", [])
            H, W = _shape_from_attempt(proto)
            a1 = _zeros(H, W, fill=0)       
            a2 = _random_grid(H, W, rng)
            outputs.append({"attempt_1": a1, "attempt_2": a2})
            total_tests += 1
        submission[task_id] = outputs

    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"Wrote {out_path} with {len(submission)} tasks and {total_tests} test grids "
          f"(shapes copied from sample_submission).")

if __name__ == "__main__":
    sample_path = "/kaggle/input/arc-prize-2025/sample_submission.json"
    if not os.path.exists(sample_path):
        raise FileNotFoundError("sample_submission.json not found. Make sure the ARC Prize 2025 dataset "
                                "is attached to the Notebook.")
    build_submission_from_sample(sample_path, "submission.json", seed=0)

