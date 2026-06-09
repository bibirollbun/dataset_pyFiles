import pandas as pd
submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
print(submission.shape)
submission.head(10)


import numpy as np


submission["ADHD_Outcome"] = np.random.choice([0, 1], size=(len(submission),), p=[0.1, 0.9])
submission["Sex_F"] = np.random.choice([0, 1], size=(len(submission),), p=[0.1, 0.9])


submission.to_csv("submission002.csv", index=False)

