


import pandas as pd

df = pd.read_csv(f"/kaggle/input/waveform-inversion/sample_submission.csv")
df.to_csv("submission.csv", index = None)

