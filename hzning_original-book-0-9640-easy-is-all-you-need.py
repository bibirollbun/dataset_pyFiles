import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_df.head()


y_train = train_df["y"]
train_id = train_df["id"]
train = train_df.drop(columns=['id'])
train = train.drop(columns=['y'])
test_ID = test_df["id"]
test = test_df.drop(columns=['id'])


all_data = pd.concat((train, test)).reset_index(drop=True)


all_data = pd.get_dummies(all_data, columns=['job','marital','education','default','housing','loan','month','contact','poutcome'])


X_train = all_data[:train_id.shape[0]]
X_test = all_data[train_id.shape[0]:]


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# 1. Data Preparation
# --------------------------------------------------
X, y    = X_train, y_train
class_0 = y.sum()
class_1 = len(y) - class_0
scale_pos_weight = class_1 / class_0

# --------------------------------------------------
# 2. Define Three Base Models
# --------------------------------------------------
xgb = XGBClassifier(
    max_depth=4,
    learning_rate=0.01,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    class_weights=[scale_pos_weight, 1],
    random_seed=42,
    verbose=0
)

lgbm = LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: scale_pos_weight, 1: 1},
    random_state=42
)

# --------------------------------------------------
# 3. Build a Soft-Voting Ensemble
# --------------------------------------------------
ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('cat', cat), ('lgbm', lgbm)],
    voting='soft'
)

# --------------------------------------------------
# 4. Train
# --------------------------------------------------
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
ensemble.fit(X_train_part, y_train_part)

# --------------------------------------------------
# 5. Search for the Optimal Threshold on the Validation Set
# --------------------------------------------------
val_probs = ensemble.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_acc = 0
for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)
    acc = (preds == y_val).mean()
    if acc > best_acc:
        best_acc, best_threshold = acc, threshold
print(f'Optimal threshold on validation set: {best_threshold:.2f}, Accuracy: {best_acc:.4f}')

# --------------------------------------------------
# 6. Generate Probabilities on the Test Set
# --------------------------------------------------
test_probs = ensemble.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test_ID,
    'Personality': test_probs  # output raw probability decimals
})



submission.to_csv('submission.csv', index=False)
print("Submission file generated with predicted probabilities")

