import pandas as pd
import numpy as np


df1 = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


df1.to_csv("submission.csv", index=False)

