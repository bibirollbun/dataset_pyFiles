import pandas as pd
file = pd.read_csv("/kaggle/input/787filed/submission - 2025-12-31T142128.964.csv")
file.to_csv('submission.csv', index=False)

