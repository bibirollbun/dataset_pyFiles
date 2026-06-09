import json
import zipfile

with open("/kaggle/input/arc-agi-code-golf-216-solutions/solutions.json") as f:
    solutions = json.load(f)

with zipfile.ZipFile("submission.zip", 'w') as zipf:
    for i in range(1, 401):
        code = solutions.get(str(i), "p=lambda g:g")
        zipf.writestr(f"task{i:03}.py", code)

print(f"{len(solutions)} problems solved.")

