import pandas as pd
import numpy as np

data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
data.describe()



num_entries = 200000

#ID column
ids = np.arange(1, num_entries + 1)

# Generate Price column with random values rounded to 6 decimal places
prices = np.round(np.random.uniform(81.300000, 81.900000, num_entries), 6)


submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

submission["Price"] = prices
submission.to_csv('submission.csv', index=False)




