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


import os, zipfile

# Create 400 tiny task files
os.makedirs("submission", exist_ok=True)
code = "def p(g):return[g[::-1]for g in g]"

for i in range(1,401):
    with open(f"submission/task{i:03}.py","w") as f:
        f.write(code)

# Create submission.zip
with zipfile.ZipFile("submission.zip","w") as z:
    for i in range(1,401):
        z.write(f"submission/task{i:03}.py",arcname=f"task{i:03}.py")

print("✅ submission.zip created with 400 tasks!")


