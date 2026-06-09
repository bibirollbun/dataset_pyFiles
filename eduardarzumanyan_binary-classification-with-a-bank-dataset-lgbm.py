import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',index_col = 'id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',index_col = 'id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


object_cols = train_df.select_dtypes(include="object").columns

le = LabelEncoder()

for col in object_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


def add_interaction_features(df):
    df_new = df.copy()
    features = [col for col in df.columns if col != 'y']
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            col1 = features[i]
            col2 = features[j]
            new_col_name = f"{col1}_x_{col2}"
            df_new[new_col_name] = df[col1] * df[col2]
    return df_new

train_df = add_interaction_features(train_df)
test_df = add_interaction_features(test_df)


X = train_df.drop(["y"], axis=1)
y = train_df["y"]
X_test = test_df


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
y_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"-----------------")
    print(f"******{fold + 1}/{n_splits}*******")
    print(f"-----------------")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        random_state=42,
        verbosity=-1,
        n_estimators=25000,
        learning_rate=0.06,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        num_leaves=100,
        max_depth=10,
        max_bin=4523,
        reg_alpha=0.79,
        reg_lambda=3,
        n_jobs = -1
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=500)
        ]
    )
    
    y_probs += model.predict_proba(X_test)[:, 1] / n_splits


sample_submission['y'] = y_probs
sample_submission.to_csv('submission.csv',index=False)




