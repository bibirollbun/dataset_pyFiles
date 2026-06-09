import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip3 install flexml


from flexml import Classification
import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')


train_df.head()


exp = Classification(
    train_df,
    target_col='y',
    encoding_method_map= {
        # Encode binary columns via label encoding (0 - 1)
        'default': 'label_encoder',
        'housing': 'label_encoder',
        'loan': 'label_encoder',
        'default': 'label_encoder',
    },
    ordinal_encode_map={ # ordinal encoding for months, 0 to 11
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    }
)


exp.start_experiment(experiment_size="quick", eval_metric="ROC-AUC")
# 'quick' mode only runs most-used and fastest models, 'wide' runs all available models


probs = exp.predict_proba(test_df)


pred_df = test_df.reset_index()[['id']]
pred_df['y'] = probs[:, 1]
pred_df


pred_df.to_csv("submission.csv", index=False)

