import json
import numpy as np
import matplotlib.pyplot as plt


with open("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json") as f:\
    tests = json.load(f)


submission = {
    tid: [
        {
            "attempt_1": [[0]*3 for _ in range(3)],
            "attempt_2": [[0]*3 for _ in range(3)]
        } for _ in t
    ]
    for tid, t in tests.items()
}


with open("submission.json", "w") as f:
    json.dump(submission, f)




