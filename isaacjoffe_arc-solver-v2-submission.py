!pip install /kaggle/input/nengo-spa-package/nengo-4.1.0-py3-none-any.whl --no-index
!pip install /kaggle/input/nengo-spa-package/nengo_spa-2.0.0-py3-none-any.whl --no-index
!pip install /kaggle/input/interrupting-cow-package-3/interruptingcow-0.8-py3-none-any.whl --no-index


import sys
sys.path.insert(1, "/kaggle/input/arc-solver-v3")
from kaggle_solver import ObjObjSolver as MySolver

import json
with open("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json", "r") as f:
    test_tasks = json.load(f)

import interruptingcow
MAX_SOLVING_TIME = 500
class SolvingTimeoutException(Exception): pass

submission = {}
task_no = 0
for task_id in list(test_tasks.keys()):
    print("\n" + "=" * 79)
    print(f"Attempting Task #{task_no}: {task_id}")
    task = test_tasks[task_id]
    task_no += 1
    print("=" * 79 + "\n")

    try:
        with interruptingcow.timeout(MAX_SOLVING_TIME, exception=SolvingTimeoutException):
            answers = MySolver(task).solve_task()
            submission[task_id] = [{"attempt_1": answer.get_data().astype(int).tolist(), "attempt_2": [[0]]} for answer in answers]

    except SolvingTimeoutException:
        print(f"Solving Timeout: Could not solve after {MAX_SOLVING_TIME} seconds.")
        submission[task_id] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(len(task["test"]))]

    except Exception as e:
        print(f"Other Error: \"{e}\"")
        submission[task_id] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(len(task["test"]))]

with open("submission.json", "w") as f:
    json.dump(submission, f, indent=4)

