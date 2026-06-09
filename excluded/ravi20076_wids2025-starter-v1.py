


%%time 

import pandas as pd, numpy as np
import lightgbm as lgb, xgboost as xgb, catboost as cb
from gc import collect
from tqdm.notebook import tqdm

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier 


train  = pd.read_csv(f"/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv", index_col = "participant_id")
test   = pd.read_csv(f"/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv", index_col = "participant_id")
ytrain = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")
sub_fl = pd.read_excel(f"/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")


display(sub_fl.head(10))


%%time 

model = \
VotingClassifier(
    [("LGBM1C", 
      lgb.LGBMClassifier(
        num_iter = 300, 
        metric = "auc", 
        verbosity = -1, 
        random_state = 42, 
        max_depth = 6, 
        learning_rate = 0.03,
      )
     ),
    ],
    voting = "soft",
    n_jobs = -1,
)

model.fit(train, ytrain["ADHD_Outcome"])
test_preds = model.predict_proba(test)[:,1]

sub_fl["ADHD_Outcome"] = np.uint8(np.round(test_preds, 0))
sub_fl.to_csv(f"submission.csv", index = None)

!head submission.csv

