import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub1 = pd.read_csv("/kaggle/input/ps-s5e8-lightgb-model-add-original-dataset/submission.csv")  # 0.97541
sub2 = pd.read_csv("/kaggle/input/ps-s5e8-xgboost-fe/submission.csv") # 0.97703


sub1.head()


sub2.head()


r1 = sub1['y']
r2 = sub2['y']

r = 0.45
sub = r * r1 + (1 - r) * r2


# r1 = sub1['y'].rank(method='average') / (len(sub1)+1)
# r2 = sub2['y'].rank(method='average') / (len(sub2)+1)

# r = 0.3
# sub = r * r1 + (1 - r) * r2


submission = pd.DataFrame({"id": test_df["id"], "y": sub})
submission.to_csv("submission.csv", index=False)
submission.head()

