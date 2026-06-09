# ğŸ“˜ Rainfall Prediction with TabPFN & LightGBM
# Kaggle Playground Series S5E3

# Goal: Predict whether it will rain tomorrow (classification)
# Metric: ROC AUC
# Models: TabPFN, LightGBM, Blending

# Competition: https://www.kaggle.com/competitions/playground-series-s5e3


# ğŸ§° Setup & Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
!pip install tabpfn --quiet
from tabpfn import TabPFNClassifier

import warnings
warnings.filterwarnings("ignore")

# Utility functions
def evaluate_model(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"ROC AUC: {auc:.5f}")
    return auc

def create_submission_file(y_pred, sample_path, output_path):
    submission = pd.read_csv(sample_path)
    submission["rainfall"] = y_pred
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")



# ğŸ“¥ Load Data

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Check for missing values
print("\nMissing values:")
print(train.isnull().sum())


# Class balance
print("\nClass distribution:")
print(train["rainfall"].value_counts(normalize=True))


# Correlation with target (numerical only)
numeric_cols = train.select_dtypes(include=np.number).columns
corr_matrix = train[numeric_cols].corr()
corr_with_target = corr_matrix["rainfall"].sort_values(ascending=False)
print("\nCorrelation with rainfall:")
print(corr_with_target)


# Visualize top correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_target.values[1:10], y=corr_with_target.index[1:10])
plt.title("Top correlations with rainfall")
plt.tight_layout()
plt.show()


# We found that 'cloud', 'humidity', and 'sunshine' showed the strongest correlations with rainfall.
# This early insight will guide some of our later feature engineering and help interpret model behavior.


# ğŸ¤– Baseline: TabPFN Classifier

# Prepare data
X_tabpfn = train.drop(columns=["id", "rainfall"])
y_tabpfn = train["rainfall"]

# Split for evaluation
X_train_tabpfn, X_val_tabpfn, y_train_tabpfn, y_val_tabpfn = train_test_split(
    X_tabpfn, y_tabpfn, test_size=0.2, stratify=y_tabpfn, random_state=42
)


# Train TabPFN
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model_tabpfn = TabPFNClassifier(device=device)
model_tabpfn.fit(X_train_tabpfn.to_numpy(), y_train_tabpfn.to_numpy())


# Predict probabilities
y_val_tabpfn_proba = model_tabpfn.predict_proba(X_val_tabpfn.to_numpy())[:, 1]
evaluate_model(y_val_tabpfn, y_val_tabpfn_proba)

# TabPFN performs surprisingly well out of the box.
# With almost no feature engineering, it was able to achieve a validation AUC > 0.87.
# We'll compare this later against LightGBM baselines.

#When we tried submitting the TabPFN outputs: Private Score =  0.90236, Public Score = 0.86323. Not bad at all given that we one-shotted the submission in a few minutes with no preprocessing


# ğŸ§¹ Preprocessing for LightGBM



# Drop ID column
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


# Target Variable Distribution
train['rainfall'].value_counts(normalize=True)
train['rainfall'].hist()


# Encode categorical features (e.g., winddirection)
le = LabelEncoder()
train["winddirection"] = le.fit_transform(train["winddirection"])
test["winddirection"] = le.fit_transform(test["winddirection"])


# Create feature set and label
X = train.drop(columns=["rainfall"])
y = train["rainfall"]


# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train set:", X_train.shape)
print("Validation set:", X_val.shape)



# ğŸŒ³ LightGBM Baseline

model_lgb = lgb.LGBMClassifier(random_state=42)
model_lgb.fit(X_train, y_train)

# Predict
y_val_lgb_proba = model_lgb.predict_proba(X_val)[:, 1]
evaluate_model(y_val, y_val_lgb_proba)

# Our LightGBM baseline is simple but strong.
# With default hyperparameters, it performs decently â€” although TabPFN outperforms it on the validation set.
# This motivates us to explore tuning and feature engineering next.

#Private Score: 0.87813, Public Score: 0.81415


# ğŸ§ª Feature Engineering & Selection

#1. Interaction Features: Some variables make sense in relation to each other (e.g. temp_range = maxtemp - mintemp, humidity_x_cloud = humidity * cloud)

train["temp_range"] = train["maxtemp"] - train["mintemp"]
test["temp_range"] = test["maxtemp"] - test["mintemp"]

#2. Polynomial Features: Some patterns might not be linear, let's try squaring them to let the model capture more curvature
train["humidity_squared"] = train["humidity"] ** 2
test["humidity_squared"] = test["humidity"] ** 2

#3. Binning and Bucketing: LightGBM is a tree-based model, perhaps thresholding some data points would help it find a pattern
train["wind_bin"] =pd.cut(train["windspeed"], bins=[0, 10, 20, 40, 100], labels=["low", "med", "high", "extreme"])
test["wind_bin"] =pd.cut(test["windspeed"], bins=[0, 10, 20, 40, 100], labels=["low", "med", "high", "extreme"])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["wind_bin_enc"] = le.fit_transform(train["wind_bin"].astype(str))
test["wind_bin_enc"] = le.transform(test["wind_bin"].astype(str))

#4. Time Features: day values go from 1-365, probably representing day of year. We want the model to capture seasonal patterns
train["day_sin"] = np.sin(2 * np.pi * train["day"] / 365)
train["day_cos"] = np.cos(2 * np.pi * train["day"] / 365)

test["day_sin"] = np.sin(2 * np.pi * test["day"] / 365)
test["day_cos"] = np.cos(2 * np.pi * test["day"] / 365)

train.drop(columns=["wind_bin"], inplace = True)
test.drop(columns=["wind_bin"], inplace = True)


# Checking correlations and model contribution

#Now let's test the correlations of the new features
corr_matrix = train.corr(numeric_only=True)
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True)
plt.title("Full Feature Correlation Heatmap")
plt.show()



# Now let's try the model with our new features

# First, let's build a clean feature list based on these learnings
# Baseline features = only original columns
baseline_features = [
    'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
    'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'
]

# Engineered features = original + new
engineered_features = baseline_features + [
    'temp_range', 'humidity_squared', 'wind_bin_enc'
]


# Then we try out LightGBM with our new features included

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_and_evaluate(features, label='rainfall'):
    X = train[features]
    y = train[label]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    return model, auc

model_base, auc_base = train_and_evaluate(baseline_features)
model_eng, auc_eng = train_and_evaluate(engineered_features)

#Report
print(f"Baseline ROC AUC: {auc_base:.5f}")
print(f"Engineered ROC AUC: {auc_eng:.5f}")

#Baseline ROC AUC: 0.85850
#Engineered ROC AUC: 0.85777


# We then check the feature importance, to validate the impact of our new features

importances = model_eng.feature_importances_
importance_df = pd.DataFrame({
    'feature': engineered_features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df, palette='mako')
plt.title("Engineered Model - LightGBM Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# I was pleasently surprised that one of our engineering features (temp_range) made it to the top 5


# Then we shortlist a group of features based on the importance list

selected_features = [
    'windspeed', 'sunshine', 'pressure', 'cloud', 'temp_range',
    'dewpoint', 'mintemp', 'maxtemp', 'humidity'
]


# Running again

model_base, auc_base = train_and_evaluate(baseline_features)
model_eng, auc_eng = train_and_evaluate(engineered_features)
model_selected, auc_selected = train_and_evaluate(selected_features)

#Report
print(f"Baseline ROC AUC: {auc_base:.5f}")
print(f"Engineered ROC AUC: {auc_eng:.5f}")
print(f"Selected ROC AUC: {auc_selected:.5f}")

#Baseline ROC AUC: 0.85850
#Engineered ROC AUC: 0.85777
#Selected ROC AUC: 0.86386

# We can now see that our selected features are outperforming the baseline feature set, and the full engineered feature set


#We tried submitting the output from our Selected Features list. 
#Private Score: 0.88287, Public Score: 0.81013. Better than plain LightGBM, but still not as good as TabPFN!


# âš™ï¸� Now let's try hyperparameter tuning for LightGBM

from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

param_grid = {
    'num_leaves': [15, 31, 50, 100],
    'max_depth': [-1, 3, 5, 10],
    'min_child_samples': [5, 10, 20, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500]
}


#Use the selected features from before
X = train[selected_features]
y= train['rainfall']

#Setup the model
model = LGBMClassifier(random_state=42)

#Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=25,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

#Fit
random_search.fit(X,y)

#Best model and score
print("Best AUC:", random_search.best_score_)
print("Best Params:", random_search.best_params_)

#Best AUC: 0.8889562289562289
#Best Params: {'num_leaves': 50, 'n_estimators': 300, 'min_child_samples': 50, 'max_depth': 3, 'learning_rate': 0.01}



#Retraining the best model on full training dataset

best_params = random_search.best_params_

model_tuned = LGBMClassifier(**best_params, random_state=42)

model_tuned.fit(X, y)

X_test = test[selected_features]

# Predict probabilities
y_test_pred = model_tuned.predict_proba(X_test)[:, 1]  # Prob of class 1

#Private Score 	0.89993	Public Score 0.83360 -> We can see an improvement on the previous LightGBM score (Feature Engineering w/o )


#Now let's try blending TabPFN and LightGBM

#sub_tabpfn = pd.read_csv("submission_tabpfn.csv")  # from TabPFN
#sub_lgb = pd.read_csv("submission_lgb_tuned.csv")  # from tuned LightGBM

#assert all(sub_tabpfn["id"] == sub_lgb["id"])

#blended_preds = 0.6 * sub_tabpfn["rainfall"] + 0.4 * sub_lgb["rainfall"]


#Now checking the differences between both sets
#diff = (sub_tabpfn["rainfall"] - sub_lgb["rainfall"]).abs()
#print(diff.describe())

# And checking where they disagree on predictions
#tabpfn_bin = (sub_tabpfn["rainfall"] > 0.5).astype(int)
#lgb_bin = (sub_lgb["rainfall"] > 0.5).astype(int)
#print((tabpfn_bin != lgb_bin).mean())  # percentage of disagreement

#Private Score: 0.90349, Public Score 0.85518

