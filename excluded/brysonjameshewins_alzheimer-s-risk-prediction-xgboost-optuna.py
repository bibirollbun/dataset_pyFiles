#import pandas
import pandas as pd

# import data
train = pd.read_csv("/kaggle/input/alzheimers-disease-risk-prediction-eu-business/train.csv")
test = pd.read_csv("/kaggle/input/alzheimers-disease-risk-prediction-eu-business/test.csv")

# drop useless column - almost every single value is confidential? (also its the only categorical variable)
train = train.drop("DoctorInCharge", axis=1)
test = test.drop("DoctorInCharge", axis=1)

# print sample data
print(train.head())


# data exploration
import seaborn as sns
import matplotlib.pyplot as plt

# get rid of seaborn warnings (amazing practices)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')

# split columns
X = train.drop("Diagnosis", axis=1)
y = train["Diagnosis"]

# get numerical columns
num_cols=X.select_dtypes(include='float64').columns.tolist()

# fancy code to create correct-sized plots
num_plots = len(num_cols)  # Number of numerical columns
rows = (num_plots // 3) + (num_plots % 3 > 0)  # Arrange in a grid (3 columns per row)

# lower figure size for plots to ensure 0 overlap
fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))  # Adjust figure size for compact layou

# flatten axes
axes = axes.flatten()

# create histograms for column
for i, col in enumerate(num_cols):
    sns.histplot(x=train[col], kde=True, hue=y, ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='both', labelsize=8)

# show graphs
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# heatmap to show correlations
corr_matrix = X.drop(columns=X.select_dtypes(include='float64')).corr()

# graph heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, fmt='.2f', linewidths=0.5)

# title
plt.title("Correlation Matrix for Alzheimers", fontsize=16)
plt.show()


# data processing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare standardize
scaler = StandardScaler()
scaler.fit(X)

# standardize data
X = scaler.transform(X)
test = scaler.transform(test)

# split train / val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# hyperparameter tuning
!pip install optuna

# import base libraries
import optuna
import numpy as np

# import specific elements
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

# cross val scoring (for later)
def combined_scorer(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return (accuracy + f1) / 2

custom_scorer = make_scorer(combined_scorer)

# trial function
def objective(trial):
    # define parameters
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

    # create model
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

    # fit model
    model.fit(X_train, y_train)

    # I could add GPU acceleration to this function to get rid of that annoying error, but it adds like 0.1% performance for like 20 minutes of work
    
    # run custom scoring function
    score = cross_val_score(model, X_train, y_train, cv=5, scoring=custom_scorer)

    # return mean of scores
    return score.mean()

# define study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# print study results
print(f"Best trial: {study.best_trial}")
print(f"Best parameters: \n {study.best_trial.params}")


# scoring
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

# create new tuned model
tuned_model = XGBClassifier(**study.best_trial.params, device="cuda")
tuned_model.fit(X_train, y_train)

# get scores
y_pred = tuned_model.predict(X_val)
recall = recall_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
f1_score = f1_score(y_val, y_pred)

# print scores
print(f"Validation Recall Score: {recall:.4f}")
print(f"Validation Accuracy Score: {accuracy:.4f}")
print(f"Validation F1 Score: {f1_score:.4f}")


# get final predictions
preds = tuned_model.predict(test)
print(preds)


# submit final prediction to competition
ID_data = pd.read_csv("/kaggle/input/alzheimers-disease-risk-prediction-eu-business/test.csv")["PatientID"] # get new copy of patientIDs
print(ID_data)

# submit predictions
submission = pd.DataFrame({
    "PatientID": ID_data,
    "Diagnosis": preds
})
submission.to_csv('submission.csv', index=False)

# print sample data
print(submission.head())
print(submission.tail())

print("scores submitted")

