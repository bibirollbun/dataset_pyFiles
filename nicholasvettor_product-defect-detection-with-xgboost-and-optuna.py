import pandas as pd
from xgboost import XGBClassifier, DMatrix
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


#import train / test data

train_data = pd.read_csv("/kaggle/input/product-defect-detection/train_data.csv")
test_data = pd.read_csv("/kaggle/input/product-defect-detection/test_data.csv")

print(train_data.head())


# data processing

# fill blank values
train_data.fillna(train_data.select_dtypes(include=[np.number]).mean(), inplace=True)
test_data.fillna(test_data.select_dtypes(include=[np.number]).mean(), inplace=True)

# split dataset
X = train_data.drop(columns=['Defective'])
y = train_data['Defective']

# setup one hot encoding with a column transformer
categorical_features = ["ProductionLine"]
encoder = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough" # only change our selected column, then ignore the rest
)

# apply one hot encoding to train / test
X = encoder.fit_transform(X)
test_data = encoder.transform(test_data)

# standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardize train data
test_data = scaler.transform(test_data)  # Standardize test data

# data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# imputer - fills in blank rows
imputer = SimpleImputer(strategy="mean")  # You can also use "median" or "most_frequent"
X = imputer.fit_transform(X)  # Apply to training data
test_features = imputer.transform(test_data)  # Apply to test data


# optuna - hyperparameter tuner
!pip install optuna


# cross val scoring (for later)
def combined_scorer(y_true, y_pred):
    # set arrays back to numpy for scoring
    y_true = cp.asnumpy(y_true)
    y_pred = cp.asnumpy(y_pred)

    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return (recall + f1) / 2

custom_scorer = make_scorer(combined_scorer)


# optuna objective
import optuna
import cupy as cp

# migrate the data to the gpu to improve speed
X_train_gpu = cp.asarray(X_train)
y_train_gpu = cp.asarray(y_train)

X_val_gpu = cp.asarray(X_val)
y_val_gpu = cp.asarray(y_val)

# create training function
def objective(trial):
    # parameters
    max_depth = trial.suggest_int('max_depth', 3, 20)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 10, 150)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)

    min_child_weight = trial.suggest_float("min_child_weight", 1, 10)
    gamma = trial.suggest_float("gamma", 0, 5)

    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)

    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0)

    num_negatives, num_positives = np.bincount(y_train)
    default_scale_pos_weight = num_negatives / num_positives
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.1, 10.0)

    # model
    model = XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate, 
        subsample=subsample, 
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True, 
        tree_method="hist",
        device="cuda"
    )

    # convert cupy array to np array
    X_train_np = X_train_gpu.get()
    y_train_np = y_train_gpu.get()

    # cross val model scoring
    score = cross_val_score(model, X_train_np, y_train_np, cv=5, scoring=custom_scorer)

    return score.mean()

# start optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000) # 500 | 1000 is overkill but i am 2nd by lke 0.16 so this could give results

# print best results
print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


# create new model based on optuna results
tuned_model = XGBClassifier(**trial.params, enable_categorical=True, device="cuda")
tuned_model.fit(X_train, y_train)

# print scoring of new model
y_pred = tuned_model.predict(X_val)
recall = recall_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
f1_score = f1_score(y_val, y_pred)

print(f"Validation Recall Score: {recall:.4f}")
print(f"Validation Accuracy Score: {accuracy:.4f}")
print(f"Validation F1 Score: {f1_score:.4f}")


# test on test_data
preds = tuned_model.predict(test_features)
print(preds)


# submit final prediction to competition
ID_data = pd.read_csv("/kaggle/input/product-defect-detection/test_data.csv")["ProductID"]
print(ID_data)

# submit predictions
submission = pd.DataFrame({
    "ProductID": ID_data,
    "Defective": preds
})
submission.to_csv('submission.csv', index=False)

print("scores submitted")

