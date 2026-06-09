import kagglehub
import pandas as pd

from pathlib import Path

data_path = Path(kagglehub.competition_download("jigsaw-agile-community-rules"))

test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

submission_df = pd.DataFrame({"row_id": test_df["row_id"], "rule_violation": 0.5})

submission_df.to_csv("submission.csv")

submission_df.head()

