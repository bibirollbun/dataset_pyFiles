# Test imports from dataset
! pip install --no-index --find-links /kaggle/input/localdependencies python-constraint2
! pip install --no-index --find-links /kaggle/input/localdependencies ordered-set



import sys
sys.path.insert(0, "/kaggle/input/apg-framework/abstract-port-graphs/")


import json
from pathlib import Path
from synthesis.kaggle_entrypoint import solve_private_set
output_dict = solve_private_set("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json", 1)
Path("submission.json").write_text(json.dumps(output_dict, indent=2))

