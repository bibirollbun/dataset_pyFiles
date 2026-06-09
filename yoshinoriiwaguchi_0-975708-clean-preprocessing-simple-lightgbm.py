# ----------------------------
# 1. Imports
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler


# ----------------------------
# 2. Data Loading 
# ----------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


# ----------------------------
#  3. Preprocessing & Feature Engineering
# ----------------------------
def preprocess(df):
    # Fill missing values for social fear-related columns
    # Prioritize "Drained_after_socializing", then fallback to "Stage_fear"
    df["Social_Fear_Filled"] = df["Drained_after_socializing"].fillna(df["Stage_fear"])
    df["Social_Fear_Filled"] = df["Social_Fear_Filled"].fillna("Unknown")

    # Convert text-based fear info into a numeric score
    df["Social_Fear_Score"] = df["Social_Fear_Filled"].map({"Yes": 1, "No": 0, "Unknown": 0.5})

    # Drop the original columns after transformation
    df.drop(columns=["Stage_fear", "Drained_after_socializing", "Social_Fear_Filled"], inplace=True)

    # Fill missing values in numeric columns with their column-wise mean
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())

    # Create a feature that represents overall social activity level
    df["Social_Activity_Score"] = df["Going_outside"] + df["Post_frequency"] - df["Time_spent_Alone"]

    # Calculate total social engagement by summing key interaction metrics
    df["Social_Engagement_Score"] = (
        df["Social_event_attendance"] + df["Friends_circle_size"] + df["Post_frequency"]
    )

    # Normalize engagement score by time spent alone (+1 to avoid division by zero)
    df["Score_ratio"] = df["Social_Engagement_Score"] / (df["Time_spent_Alone"] + 1)

    return df



# Run preprocessing on both training and test datasets
train = preprocess(train)
test = preprocess(test)

# Encode target labels: Introvert = 0, Extrovert = 1
label_map = {'Introvert': 0, 'Extrovert': 1}
y = train["Personality"].map(label_map)

# Drop target and ID columns to get feature matrix
X = train.drop(columns=["Personality", "id"])
X_test = test.drop(columns=["id"])


# ----------------------------
# 4. Hyperparameter Tuning with Optuna
# ----------------------------

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    params = {
        "objective": "binary",                        # Binary classification
        "metric": "binary_logloss",                   # Evaluation metric
        "boosting_type": "gbdt",                      # Gradient boosting decision tree
        "verbosity": -1,                              # Suppress LightGBM logs
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),   # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True), # L2 regularization
        "random_state": 42,
    }

    # Initialize the model with the trial parameters
    model = lgb.LGBMClassifier(**params)

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

    # Return the average CV accuracy
    return score

# Create and run the Optuna study
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30)

# Display the best parameters and best score
best_params = study.best_params
print("âœ… Best Params:", best_params)
print(f"âœ… CV Accuracy: {study.best_value:.6f}")



# ----------------------------
# 5. Final Training & Prediction
# ----------------------------

# Train the LightGBM model on the full training data using the best parameters
lgb_model = lgb.LGBMClassifier(**best_params)
lgb_model.fit(X, y)

# Predict the probability of "Extrovert" on the test data
lgb_pred_test = lgb_model.predict_proba(X_test)[:, 1]



# ----------------------------
# 6. Submission File Creation
# ----------------------------

# Create submission file using predicted probabilities
submission = sample_submission.copy()
submission["Personality"] = np.where(lgb_pred_test > 0.5, "Extrovert", "Introvert")
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv has been saved.")


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

yes_no_map = {'Yes': 1, 'No': 0}
df_mapped = df.replace(yes_no_map)

numeric_df = df_mapped.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Figure 1: Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()


total_ratio = df["Personality"].value_counts(normalize=True)

null_mask = df["Stage_fear"].isnull() & df["Drained_after_socializing"].isnull()
null_ratio = df[null_mask]["Personality"].value_counts(normalize=True)

compare_df = pd.DataFrame({
    "All Data": total_ratio,
    "Both Missing": null_ratio
}).T.fillna(0) 

compare_df.plot(kind='bar', stacked=True, figsize=(6, 4), color=["skyblue", "salmon"])
plt.title("Figure 2: Introvert vs Extrovert Ratio - All Data vs Both Missing")
plt.ylabel("Ratio")
plt.xticks(rotation=0)
plt.legend(title="Personality")
plt.tight_layout()
plt.show()



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

yes_no_map = {'Yes': 1, 'No': 0}
df = df.replace(yes_no_map)
df = df.infer_objects(copy=False) 

df["Social_Fear_Filled"] = df["Drained_after_socializing"].fillna(df["Stage_fear"])
df["Social_Fear_Filled"] = df["Social_Fear_Filled"].fillna("Unknown")
df["Social_Fear_Score"] = df["Social_Fear_Filled"].map({"Yes": 1, "No": 0, "Unknown": 0.5})

df["Social_Activity_Score"] = df["Going_outside"] + df["Post_frequency"] - df["Time_spent_Alone"]
df["Social_Engagement_Score"] = (
    df["Social_event_attendance"] + df["Friends_circle_size"] + df["Post_frequency"]
)
df["Score_ratio"] = df["Social_Engagement_Score"] / (df["Time_spent_Alone"] + 1)

sns.set(style="whitegrid")

# 1. Social_Activity_Score
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Social_Activity_Score", hue="Personality", kde=True, element="step", palette="Set2")
plt.title("Figure 3: Distribution of Social_Activity_Score by Personality")
plt.xlabel("Social_Activity_Score")
plt.tight_layout()
plt.show()

# 2. Social_Engagement_Score
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Social_Engagement_Score", hue="Personality", kde=True, element="step", palette="Set2")
plt.title("Figure 4: Distribution of Social_Engagement_Score by Personality")
plt.xlabel("Social_Engagement_Score")
plt.tight_layout()
plt.show()

# 3. Score_ratio
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Score_ratio", hue="Personality", kde=True, element="step", palette="Set2")
plt.title("Figure 5: Distribution of Score_ratio by Personality")
plt.xlabel("Score_ratio")
plt.tight_layout()
plt.show()


