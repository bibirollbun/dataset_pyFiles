import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

def preprocess_data(df, is_train=True):
    X = df.drop(columns=["id"])
    if is_train:
        y = X.pop("rainfall")
        return X, y
    return X

# Preprocess data
X_train, y_train = preprocess_data(train_df)
X_test = preprocess_data(test_df, is_train=False)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on test data
y_test_pred = model.predict(X_test_scaled)

# Save predictions
submission = pd.DataFrame({"id": test_df["id"], "rainfall": y_test_pred})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

def preprocess_data(df, is_train=True):
    X = df.drop(columns=["id"])
    if is_train:
        y = X.pop("rainfall")
        return X, y
    return X

# Preprocess data
X_train, y_train = preprocess_data(train_df)
X_test = preprocess_data(test_df, is_train=False)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on test data
y_test_pred = model.predict(X_test_scaled)

# Save predictions
submission = pd.DataFrame({"id": test_df["id"], "rainfall": y_test_pred})
submission.to_csv("submission2.csv", index=False)
print("Predictions saved to submission2.csv")


import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

def preprocess_data(df, is_train=True):
    X = df.drop(columns=["id"])
    if is_train:
        y = X.pop("rainfall")
        return X, y
    return X

# Preprocess data
X_train, y_train = preprocess_data(train_df)
X_test = preprocess_data(test_df, is_train=False)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Train XGBoost Classifier
model = xgb.XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train_split, y_train_split)

# Validate model
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")

# Make predictions on test data
y_test_pred = model.predict(X_test_scaled)

# Save predictions
submission = pd.DataFrame({"id": test_df["id"], "rainfall": y_test_pred})
submission.to_csv("submission3.csv", index=False)
print("Predictions saved to submission3.csv")


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

def preprocess_data(df, is_train=True):
    X = df.drop(columns=["id"])
    if is_train:
        y = X.pop("rainfall")
        return X, y
    return X

# Preprocess data
X_train, y_train = preprocess_data(train_df)
X_test = preprocess_data(test_df, is_train=False)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Engineering - Adding Interaction Terms
X_train["feature_sum"] = X_train.sum(axis=1)
X_test["feature_sum"] = X_test.sum(axis=1)
X_train["feature_mean"] = X_train.mean(axis=1)
X_test["feature_mean"] = X_test.mean(axis=1)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Define base models
xgb_model = xgb.XGBClassifier(n_estimators=1000, max_depth=12, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=1000, max_depth=12, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, random_state=42)

# Stacking Model
stacked_model = StackingClassifier(
    estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
    final_estimator=LogisticRegression(),
    passthrough=True
)

# Train stacked model
stacked_model.fit(X_train_split, y_train_split)

# Validate model
y_val_pred = stacked_model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")

# Make predictions on test data
y_test_pred = stacked_model.predict(X_test_scaled)

# Save predictions
submission = pd.DataFrame({"id": test_df["id"], "rainfall": y_test_pred})
submission.to_csv("submission4.csv", index=False)
print("Predictions saved to submission4.csv")


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import optuna

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

def preprocess_data(df, is_train=True):
    X = df.drop(columns=["id"])
    if is_train:
        y = X.pop("rainfall")
        return X, y
    return X

# Preprocess data
X_train, y_train = preprocess_data(train_df)
X_test = preprocess_data(test_df, is_train=False)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Engineering - Adding Polynomial Features
X_train["feature_sum"] = X_train.sum(axis=1)
X_test["feature_sum"] = X_test.sum(axis=1)
X_train["feature_mean"] = X_train.mean(axis=1)
X_test["feature_mean"] = X_test.mean(axis=1)

# Handle Imbalanced Data
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Standardize numerical features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Define base models
xgb_model = xgb.XGBClassifier(n_estimators=2000, max_depth=15, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=2000, max_depth=15, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=42)
cat_model = cb.CatBoostClassifier(iterations=2000, depth=12, learning_rate=0.02, verbose=0, random_state=42)

# Stacking Model
stacked_model = StackingClassifier(
    estimators=[("xgb", xgb_model), ("lgb", lgb_model), ("cat", cat_model)],
    final_estimator=LogisticRegression(),
    passthrough=True
)

# Train stacked model
stacked_model.fit(X_train_split, y_train_split)

# Validate model
y_val_pred = stacked_model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")

# Make predictions on test data
y_test_pred = stacked_model.predict(X_test_scaled)

# Save predictions
submission = pd.DataFrame({"id": test_df["id"], "rainfall": y_test_pred})
submission.to_csv("submission5.csv", index=False)
print("Predictions saved to submission5.csv")


import pandas as pd, numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
print("Train shape", train.shape )
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print("Test shape:", test.shape )
test.head()


RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier  #


# WEIGHTS TO ADJUST IMPORTANCE OF FEATURES DURING KNN
WGT = {'day': 24, 'pressure': 1, 'maxtemp': 1, 'temparature': 1, 'mintemp': 1, 'dewpoint': 1, 'humidity': 1, 
       'cloud': 1, 'sunshine': 1, 'winddirection': 1, 'windspeed': 1}


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=777)
    
oof_knn = np.zeros(len(train))
pred_knn = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rainfall"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rainfall"]
    x_test = test[FEATURES].copy()

    for c in FEATURES:
        m = x_train[c].mean()
        s = x_train[c].std()
        x_train[c] = WGT[c] * (x_train[c]-m)/s
        x_valid[c] = WGT[c] * (x_valid[c]-m)/s
        x_test[c] = WGT[c] * (x_test[c]-m)/s
        x_test[c] = x_test[c].fillna(0)
        x_train[c] = x_train[c].fillna(0)

    model = KNeighborsClassifier(n_neighbors=201, p=1)
    model.fit(x_train.values, y_train.values)

    # INFER OOF
    oof_knn[test_index] = model.predict_proba(x_valid.values)[:,1]
    # INFER TEST
    pred_knn += model.predict_proba(x_test.values)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_knn /= FOLDS


from sklearn.metrics import roc_auc_score
true = train.rainfall.values
m = roc_auc_score(true, oof_knn)
print(f"KNN CV Score AUC = {m:.3f}")


print("Best Public Notebook achieves LB = 0.954!")
best_public = pd.read_csv("/kaggle/input/submission95427/submission95427.csv")
display( best_public.head() )
best_public = best_public.rainfall.values


from scipy.stats import rankdata

print("Ensemble achieves LB = 0.961! Hooray!")
sub = pd.read_csv("/kaggle/input/submission-ensemble/submission_ensemble.csv")
sub.rainfall = -0.25 * rankdata( pred_knn ) + 1.25 * rankdata( best_public )
sub.rainfall = rankdata( sub.rainfall ) / len(sub)
print( sub.shape )
sub.to_csv(f"submission_ensemble.csv",index=False)
sub.head()




