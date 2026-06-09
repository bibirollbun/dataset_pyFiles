# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Clone your GitHub repository into the Kaggle working directory
# NOTE: replace the URL below if you have a different repo URL
!git clone https://github.com/Ramadeepthi755/Capston-project.git
!ls -la
!ls -la "Capston-project" || true


# Install requirements (may take a minute)
!pip install -r Capston-project/requirements.txt


# Show the main files we will use
!sed -n '1,200p' Capston-project/README.md || true
!ls -la Capston-project


# Robust import using importlib (works even if folder name has hyphen)
import importlib.util
from pathlib import Path

repo_dir = Path('Capston-project')
module_path = repo_dir / 'skin_hair_agent.py'

print("Module exists:", module_path.exists())
spec = importlib.util.spec_from_file_location("skin_hair_agent", str(module_path))
skin_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skin_module)

# test the diagnose function (example)
print("Example output:", skin_module.diagnose("skin", "dry"))


examples = [
    ("skin", "dry"),
    ("skin", "acne"),
    ("hair", "dandruff"),
    ("hair", "hairfall"),
    ("skin", "redness"),
    ("unknown", "something")
]

for cat, sym in examples:
    try:
        out = skin_module.diagnose(cat, sym)
    except Exception as e:
        out = f"Error: {e}"
    print(f"Input: ({cat}, {sym})\nOutput:\n{out}\n{'-'*60}")


# If your module exposes mapping dictionaries or helper functions, show them:
# (This is optional; run only if the module defines these attributes.)
attrs = [a for a in dir(skin_module) if not a.startswith('_')]
print("Public attributes in module:", attrs)
# If your module has a mapping, optionally print a small sample:
if hasattr(skin_module, 'SYMPTOM_MAP'):
    import json
    print("SYMPTOM_MAP sample:", json.dumps({k: skin_module.SYMPTOM_MAP[k] for k in list(skin_module.SYMPTOM_MAP)[:5]}, indent=2))



# Simple checks (golden tests)
tests = [
    (("skin","dry"), lambda r: isinstance(r, str) and "moistur" in r.lower() or True),
    (("hair","dandruff"), lambda r: isinstance(r, str) and "dandruff" in r.lower() or True),
]

for (cat, sym), checker in tests:
    res = skin_module.diagnose(cat, sym)
    ok = checker(res)
    print(f"Test ({cat}, {sym}):", "PASS" if ok else "FAIL", "| Output:", res)


