# Upgrade Torch to 2.6.0+CUDA 12.4
!pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Confirm torch version
import torch
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)

import torch
print(torch.__version__)
print(torch.version.cuda)
!nvidia-smi


!pip install warpgbm --no-build-isolation


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]  
    # Pairwise interactions
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_triplet_interactions(df, features):
    df_new = df.copy()
    # Triplet interactions
    for f1, f2, f3 in itertools.combinations(features, 3):
        df_new[f"{f1}_x_{f2}_x_{f3}"] = df_new[f1] * df_new[f2] * df_new[f3]
    return df_new

def add_interaction_features(df, features):
    df_new = df.copy()
    
    # Pairwise interactions (existing part)
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)

    # Triplet interactions (new part)
    for f1, f2, f3 in itertools.combinations(features, 3):
        # Adding operations for triplet combinations
        df_new[f"{f1}_plus_{f2}_plus_{f3}"] = df_new[f1] + df_new[f2] + df_new[f3]
        df_new[f"{f1}_minus_{f2}_minus_{f3}"] = df_new[f1] - df_new[f2] - df_new[f3]
        df_new[f"{f2}_minus_{f1}_minus_{f3}"] = df_new[f2] - df_new[f1] - df_new[f3]
        df_new[f"{f3}_minus_{f1}_minus_{f2}"] = df_new[f3] - df_new[f1] - df_new[f2]
        df_new[f"{f1}_div_{f2}_div_{f3}"] = df_new[f1] / (df_new[f2] * df_new[f3] + 1e-5)
        df_new[f"{f2}_div_{f1}_div_{f3}"] = df_new[f2] / (df_new[f1] * df_new[f3] + 1e-5)
        df_new[f"{f3}_div_{f1}_div_{f2}"] = df_new[f3] / (df_new[f1] * df_new[f2] + 1e-5)

    return df_new


def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

train = add_interaction_features(train, numerical_features)
test = add_interaction_features(test, numerical_features)

train = add_triplet_interactions(train, numerical_features)  # Adding triplet interactions
test = add_triplet_interactions(test, numerical_features)  # Adding triplet interactions

train = add_statistical_features(train, numerical_features)
test = add_statistical_features(test, numerical_features)

# Encoding categorical columns
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_train = poly.fit_transform(train[numerical_features])
poly_test = poly.transform(test[numerical_features])
poly_feature_names = poly.get_feature_names_out(numerical_features)

poly_train_df = pd.DataFrame(poly_train, columns=poly_feature_names)
poly_test_df = pd.DataFrame(poly_test, columns=poly_feature_names)

train = pd.concat([train.reset_index(drop=True), poly_train_df], axis=1)
test = pd.concat([test.reset_index(drop=True), poly_test_df], axis=1)

# Prepare features and target
X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  
X_test = test.drop(columns=['id'])

X.loc[:, ~X.columns.duplicated()]
X_test.loc[:, X.columns ]

FEATURES = X.columns.tolist()



X.shape


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import time
from warpgbm import WarpGBM

FOLDS = 5
TOP_K =150
RANDOM_SEED = 42

# Assume: X, y, X_test already defined
# X = train.drop(columns=["target"])
# y = train["target"]
# X_test = test.copy()

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))
rmsles = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}/{FOLDS}")

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # â”€â”€ Per-Fold Feature Selection â”€â”€
    # corrs = x_train.corrwith(y_train).abs().sort_values(ascending=False)
    # top_features = corrs.head(TOP_K).index.tolist()

    x_train = x_train.values#[top_features].astype(np.float32).values
    x_valid = x_valid.values#[top_features].astype(np.float32).values
    x_test = X_test.values#[top_features].astype(np.float32).values

    model = WarpGBM(
        max_depth=12,
        num_bins=127,
        n_estimators=5000,
        learning_rate=0.005,
        verbosity=False,
        colsample_bytree=0.5
    )

    start = time.time()
    model.fit(
        x_train, 
        y_train.values,
        X_eval=x_valid,
        y_eval=y_valid.values,
        eval_every_n_trees=20,
        early_stopping_rounds=10,
        # eval_metric="corr"
    )
    oof_pred = model.predict(x_valid)
    test_pred = model.predict(x_test)
    end = time.time()

    oof[valid_idx] = oof_pred
    preds += test_pred / FOLDS

    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
    rmsles.append(rmsle)

    print(f"RMSLE: {rmsle:.4f} | Time: {end - start:.1f}s")

# â”€â”€â”€ Final Results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
mean_rmsle = np.mean(rmsles)
std_rmsle = np.std(rmsles)
print(f"\nMean RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")



# model = WarpGBM(
#     max_depth=10,
#     num_bins=127,
#     n_estimators=590,
#     learning_rate=0.02,
#     verbosity=False,
#     colsample_bytree=0.7
# )

# model.fit(
#     X.values, 
#     y.values,
# )

# preds_w_retrain = ( preds +  model.predict(X_test.values) ) / 2


submission['Calories'] = np.expm1( preds )
submission.to_csv('submission.csv', index=False)

print("\nSubmission Head:")
print(submission.head())



