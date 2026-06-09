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


import pandas as pd

# ===============================
# LOAD FINAL SUBMISSIONS
# ===============================
sub_A = pd.read_csv("/kaggle/input/top-4-model-blending-approach/submission_A.csv")  # best meta + isotonic + TTA
sub_B = pd.read_csv("/kaggle/input/top-4-model-blending-approach/submission_B.csv")  # TE sweep + stack
sub_C = pd.read_csv("/kaggle/input/top-4-model-blending-approach/submission_C.csv")  # alternate meta
sub_D = pd.read_csv("/kaggle/input/top-4-model-blending-approach/submission_D.csv")  # Optuna-tuned model

# ===============================
# SAFETY CHECKS
# ===============================
assert sub_A["id"].equals(sub_B["id"])
assert sub_A["id"].equals(sub_C["id"])
assert sub_A["id"].equals(sub_D["id"])

# ===============================
# FINAL WEIGHTS (POST-OPTUNA ADJUSTED)
# ===============================
wA, wB, wC, wD = 0.42, 0.25, 0.13, 0.20

# ===============================
# BLEND
# ===============================
final_pred = (
    wA * sub_A["diagnosed_diabetes"] +
    wB * sub_B["diagnosed_diabetes"] +
    wC * sub_C["diagnosed_diabetes"] +
    wD * sub_D["diagnosed_diabetes"]
)

final_submission = pd.DataFrame({
    "id": sub_A["id"],
    "diagnosed_diabetes": final_pred
})

final_submission.to_csv("submission.csv", index=False)
print("✅ submission_final.csv written")
print(final_submission.head())


