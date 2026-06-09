import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

import warnings, os, gc, sys, math, json, random, itertools

from scipy import stats
from scipy.stats import ks_2samp


warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
sns.set_palette("crest")
pd.set_option("display.max_columns", 100)


train = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')
test = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')
train_enc = train.replace({"RiskLevel": {0:"Low Risk", 1:"Mid Risk", 2:"High Risk"}})
##encoded switches the ints to categories


def quick_overview(df, name="train"):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train, "train")
quick_overview(test , "test")

print(f"Duplicate rows (train): {train.duplicated().sum()}  |  (test): {test.duplicated().sum()}")



fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=train_enc, x="RiskLevel", ax=ax)
ax.set_title("Risk Level Balance")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x()+.35, p.get_height()+20), ha="center")

plt.ylim(0, 400)
plt.show()

print(train_enc["RiskLevel"].value_counts(normalize=True).rename("proportion"))


train.isnull().sum() 



def plot_kde(data, name, columns=None, figsize=(8, 4), fill=True, max_density=None):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    columns = data.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=figsize)
    for col in columns:
        sns.kdeplot(data[col], label=col, linewidth=2,clip=(0, None),linestyle="-.")
        
    if max_density is not None:
        plt.ylim(0, max_density)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


for col in test.drop(columns=['Id','Usage']).columns.tolist():
    combo = pd.DataFrame({
        'Train' : train[col],
        'Test' : test[col]
    })
    plot_kde(combo,col)


outlier_summary = {}
for col in train.drop(columns=['Id','Usage']):
    z = np.abs(stats.zscore(train[col]))
    outlier_summary[col] = (z>3).sum()   # 3-Ïƒ rule

pd.Series(outlier_summary, name="#outliers (>3Ïƒ)").sort_values(ascending=False).to_frame().style.bar()


#Outlier Mask for Blood glucose
bg = train["Blood glucose"]
z    = np.abs(stats.zscore(bg, nan_policy="omit"))
outlier_mask = (z > 3)

outliers= train_enc.loc[outlier_mask, ["Blood glucose", "RiskLevel"]]
base_counts   = train["RiskLevel"].value_counts()
outlier_counts = outliers["RiskLevel"].value_counts()

fig, ax = plt.subplots(figsize=(4,3))
sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax)
ax.set_title("RiskLevel among 3Ïƒ Blood Glucose outliers")
ax.set_ylabel("count")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", (p.get_x()+0.3, p.get_height()+30))

plt.show()

# Proportion print-out
print("Outlier group distribution")
display(outlier_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))

print("Comparison with overall training distribution")
display(base_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))


#Outlier Mask for Blood glucose
bt = train["BodyTemp"]
z    = np.abs(stats.zscore(bt, nan_policy="omit"))
outlier_mask = (z > 3)

outliers= train_enc.loc[outlier_mask, ["BodyTemp", "RiskLevel"]]
base_counts   = train["RiskLevel"].value_counts()
outlier_counts = outliers["RiskLevel"].value_counts()

fig, ax = plt.subplots(figsize=(4,3))
sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax)
ax.set_title("Risk Level among 3Ïƒ BodyTemp outliers")
ax.set_ylabel("count")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", (p.get_x()+0.3, p.get_height()+30))

plt.show()

# Proportion print-out
print("Outlier group distribution")
display(outlier_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))

print("Comparison with overall training distribution")
display(base_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))


# Numeric vs target
num_cols = train_enc.drop(columns=['Id','Usage']).select_dtypes(include='number').columns.tolist()
fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x="RiskLevel", y=col, data=train_enc, ax=axes[i])
    axes[i].set_title(f"{col} by Risk Level")
plt.tight_layout()
plt.show()


corr = train.drop(columns=['Id','Usage']).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8) 
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation")
plt.show()


target_corr = train.drop(columns=['Id','Usage']).corr()["RiskLevel"].drop(
    "RiskLevel").sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))


!wget -q https://raw.githubusercontent.com/Sanjidx090/Work/refs/heads/main/mlolympiadbd2025/train.csv
!wget -q https://raw.githubusercontent.com/Sanjidx090/Work/refs/heads/main/mlolympiadbd2025/test.csv
!wget -q https://raw.githubusercontent.com/Sanjidx090/Work/refs/heads/main/mlolympiadbd2025/sample_submission.csv



# new try shall we?



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
# Load the datasets directly from the notebook's environment
try:
    train_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')
except FileNotFoundError:
    # Fallback for local execution if files are in the same directory
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')


# --- 2. Feature Engineering ---
# This function creates more medically relevant interaction features
def create_features(df):
    """Creates new features based on domain knowledge."""
    df_copy = df.copy()
    # Pulse Pressure: The difference between systolic and diastolic pressure.
    df_copy['PulsePressure'] = df_copy['SystolicBP'] - df_copy['DiastolicBP']
    
    # Mean Arterial Pressure (MAP): A better indicator of blood perfusion.
    df_copy['MeanArterialPressure'] = df_copy['DiastolicBP'] + (df_copy['PulsePressure'] / 3)
    
    # Interaction between age and blood pressure, as risk often increases with both.
    df_copy['Age_Systolic_Interaction'] = df_copy['Age'] * df_copy['SystolicBP']
    
    return df_copy

train_featured = create_features(train_df)
test_featured = create_features(test_df)

# --- 3. Model Training and Prediction ---
# Define features (X) and target (y)
features = [
    'Age', 'SystolicBP', 'DiastolicBP', 'Blood glucose', 'BodyTemp', 'HeartRate',
    'PulsePressure', 'MeanArterialPressure', 'Age_Systolic_Interaction'
]
X_train = train_featured[features]
y_train = train_featured['RiskLevel']
X_test = test_featured[features]

# Use Stratified K-Fold for robust cross-validation, which is good for classification
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize LightGBM Classifier - a powerful and efficient gradient boosting model
lgbm = LGBMClassifier(random_state=42)

# --- 4. Cross-Validation Score ---
# Calculate the CV score to see the improvement
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    lgbm.fit(X_train_fold, y_train_fold)
    preds = lgbm.predict(X_val_fold)
    score = np.mean(y_val_fold == preds)
    cv_scores.append(score)

print(f"ğŸš€ New LightGBM CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"ğŸ“‰ Previous Notebook Score: 0.8052")


# --- 5. Create Submission File ---
# Train the model on the full training data
lgbm.fit(X_train, y_train)

# Make predictions on the test data
predictions = lgbm.predict(X_test)

# Create the submission DataFrame
submission_df = pd.DataFrame({'Id': test_df['Id'], 'RiskLevel': predictions})

# Save the submission file
submission_df.to_csv('submission(Hell)yeah.csv', index=False)

print("\nâœ… Submission file 'submission.csv' has been created successfully!")


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')
except FileNotFoundError:
    # Fallback for local execution if files are in the same directory
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

# --- 2. Your Winning Feature Engineering ---
# This feature set is the key, so we're keeping it.
def create_features_original(df):
    df_enhanced = df.copy()
    df_enhanced['BP_Category'] = pd.cut(df['SystolicBP'], bins=[0, 120, 130, 140, 200], labels=[0, 1, 2, 3])
    df_enhanced['Health_Score'] = (df['SystolicBP'] - 120) + (df['DiastolicBP'] - 80) + (df['Blood glucose'] - 7)
    df_enhanced['HR_Zone'] = pd.cut(df['HeartRate'], bins=[0, 60, 80, 100, 200], labels=[0, 1, 2, 3])
    df_enhanced['Temp_Abnormal'] = ((df['BodyTemp'] < 98.0) | (df['BodyTemp'] > 99.5)).astype(int)
    df_enhanced['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3])
    
    for feature in ['BP_Category', 'HR_Zone', 'Age_Group']:
        df_enhanced[feature] = df_enhanced[feature].astype('category').cat.codes
        
    return df_enhanced

# --- 3. Prepare Data ---
X_train = create_features_original(train_df.drop(columns=['RiskLevel', 'Id', 'Usage']))
y_train = train_df['RiskLevel']
X_test = create_features_original(test_df.drop(columns=['Id', 'Usage']))

# Ensure columns match perfectly between train and test
X_test = X_test[X_train.columns]


# --- 4. Hyperparameter Tuning with Class Weighting ---
print("--- Searching for the best RandomForest parameters with Class Weighting ---")

# Define the parameter grid to search.
# We include 'class_weight' to let the grid search decide if balancing helps.
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None], # The key addition
    'max_features': ['sqrt', 'log2']
}

# Use Stratified K-Fold for reliable cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1, # Use all available CPU cores
    verbose=1
)

# Run the search
grid_search.fit(X_train, y_train)

print(f"\nğŸ�† Your previous best score: 0.87128")
print(f"ğŸ�¯ Best CV Score from Grid Search: {grid_search.best_score_:.5f}")
print("ğŸ”� Best Parameters Found:")
print(grid_search.best_params_)

# --- 5. Create Final Submission with the Best Model ---
# The best model is already trained on the full dataset by GridSearchCV
best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

submission_df = pd.DataFrame({'Id': test_df['Id'], 'RiskLevel': predictions})
submission_df.to_csv('submission_tuned_weighted_rf.csv', index=False)

print("\nâœ… Submission file 'submission_tuned_weighted_rf.csv' created.")
print("\nPrediction distribution from the tuned model:")
print(submission_df['RiskLevel'].value_counts().sort_index())


import pandas as pd
file_path = "/kaggle/working/submission_tuned_weighted_rf.csv"
df = pd.read_csv(file_path)

# Inspect columns, dtypes, and unique values
summary = []
for col in df.columns:
    uniques = pd.Series(df[col].unique())
    # Truncate long unique lists for readability
    uniques_display = uniques.head(10).tolist()
    summary.append({
        "column": col,
        "dtype": str(df[col].dtype),
        "n_unique": df[col].nunique(),
        "sample_uniques": uniques_display
    })

summary_df = pd.DataFrame(summary)

# Identify columns that look like class labels with {0,1,2}
candidate_cols = []
for col in df.columns:
    try:
        vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
        if vals.issubset({0,1,2}) and len(vals) > 0:
            candidate_cols.append(col)
    except Exception:
        pass

# If there is a column literally named 'RiskLevel', prefer that one;
# otherwise, if exactly one candidate, use it; else don't change anything yet.
target_col = None
if "RiskLevel" in df.columns:
    target_col = "RiskLevel"
elif len(candidate_cols) == 1:
    target_col = candidate_cols[0]

result_info = {"target_col": target_col, "candidate_cols": candidate_cols}

# If we have a target, compute value counts before/afterfine
before_counts = None
after_counts = None
output_path = None
if target_col is not None:
    before_counts = df[target_col].value_counts(dropna=False).sort_index()
    mapping = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
    # Ensure numeric for mapping
    numeric_series = pd.to_numeric(df[target_col], errors="coerce")
    mapped = numeric_series.map(mapping)
    df[target_col] = mapped
    after_counts = df[target_col].value_counts(dropna=False)
    output_path = "/kaggle/working/submission_tuned_weighted_rf.csv"
    df.to_csv(output_path, index=False)


#Jjsajsakdkqdlqkdkqwleknqnqlfwknqlfqlfqlknfqlkflqkfnlfqlf

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')
except FileNotFoundError:
    # Fallback for local execution
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

# --- 2. Advanced and Original Feature Engineering ---
# Using the successful feature set from our last attempt
def create_new_features(train_df, test_df):
    from sklearn.ensemble import IsolationForest
    combined_df = pd.concat([train_df.drop('RiskLevel', axis=1), test_df], ignore_index=True)

    # User's original features
    combined_df['BP_Category'] = pd.cut(combined_df['SystolicBP'], bins=[0, 120, 130, 140, 200], labels=[0, 1, 2, 3])
    combined_df['Health_Score'] = (combined_df['SystolicBP'] - 120) + (combined_df['DiastolicBP'] - 80) + (combined_df['Blood glucose'] - 7)
    combined_df['HR_Zone'] = pd.cut(combined_df['HeartRate'], bins=[0, 60, 80, 100, 200], labels=[0, 1, 2, 3])
    combined_df['Temp_Abnormal'] = ((combined_df['BodyTemp'] < 98.0) | (combined_df['BodyTemp'] > 99.5)).astype(int)
    combined_df['Age_Group'] = pd.cut(combined_df['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3])

    for feature in ['BP_Category', 'HR_Zone', 'Age_Group']:
        combined_df[feature] = combined_df[feature].astype('category').cat.codes
        
    # Advanced features
    combined_df['Shock_Index'] = combined_df['HeartRate'] / combined_df['SystolicBP']
    
    iso_features = ['Age', 'SystolicBP', 'Blood glucose', 'HeartRate']
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(train_df[iso_features])
    combined_df['Anomaly_Score'] = -iso.decision_function(combined_df[iso_features])

    final_features = combined_df.drop(columns=['Id', 'Usage', 'Age', 'SystolicBP', 'HeartRate'])
    
    X_train_final = final_features.iloc[:len(train_df)]
    X_test_final = final_features.iloc[len(train_df):]
    
    return X_train_final, X_test_final

# --- 3. Prepare Data ---
X_train, X_test = create_new_features(train_df, test_df)
y_train = train_df['RiskLevel']

# --- 4. 5-Fold Training and Ensembling ---
print("--- Training 5 models and ensembling their predictions ---")

# Use Stratified K-Fold with 5 splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = []
oof_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"--- Fold {fold+1}/5 ---")
    
    # Split data for this fold
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_fold, y_train_fold)

    # Validate and score
    val_preds = model.predict(X_val_fold)
    oof_preds[val_idx] = val_preds
    fold_score = accuracy_score(y_val_fold, val_preds)
    oof_scores.append(fold_score)
    print(f"Fold {fold+1} Accuracy: {fold_score:.5f}")

    # Predict on the test set and store the predictions
    test_fold_preds = model.predict(X_test)
    test_preds.append(test_fold_preds)

print(f"\nğŸ“ˆ Overall CV Score: {np.mean(oof_scores):.5f} (+/- {np.std(oof_scores):.4f})\n")


# --- 5. Create Final Submission via Majority Vote ---
# Convert list of prediction arrays into a 2D numpy array
test_preds_array = np.array(test_preds)

# Use scipy.stats.mode to find the most common prediction for each row (test sample)
# This is the majority vote
final_predictions, _ = mode(test_preds_array, axis=0)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'RiskLevel': final_predictions})
submission_df.to_csv('submission_ensembled_folds.csv', index=False)

print("âœ… Submission file 'submission_ensembled_folds.csv' created.")
print("\nPrediction distribution from the ensembled model:")
print(submission_df['RiskLevel'].value_counts().sort_index())


import pandas as pd
file_path = "/kaggle/working/submission_ensembled_folds.csv"
df = pd.read_csv(file_path)

# Inspect columns, dtypes, and unique values
summary = []
for col in df.columns:
    uniques = pd.Series(df[col].unique())
    # Truncate long unique lists for readability
    uniques_display = uniques.head(10).tolist()
    summary.append({
        "column": col,
        "dtype": str(df[col].dtype),
        "n_unique": df[col].nunique(),
        "sample_uniques": uniques_display
    })

summary_df = pd.DataFrame(summary)

# Identify columns that look like class labels with {0,1,2}
candidate_cols = []
for col in df.columns:
    try:
        vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
        if vals.issubset({0,1,2}) and len(vals) > 0:
            candidate_cols.append(col)
    except Exception:
        pass

# If there is a column literally named 'RiskLevel', prefer that one;
# otherwise, if exactly one candidate, use it; else don't change anything yet.
target_col = None
if "RiskLevel" in df.columns:
    target_col = "RiskLevel"
elif len(candidate_cols) == 1:
    target_col = candidate_cols[0]

result_info = {"target_col": target_col, "candidate_cols": candidate_cols}

# If we have a target, compute value counts before/afterfine
before_counts = None
after_counts = None
output_path = None
if target_col is not None:
    before_counts = df[target_col].value_counts(dropna=False).sort_index()
    mapping = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
    # Ensure numeric for mapping
    numeric_series = pd.to_numeric(df[target_col], errors="coerce")
    mapped = numeric_series.map(mapping)
    df[target_col] = mapped
    after_counts = df[target_col].value_counts(dropna=False)
    output_path = "/kaggle/working/submission.csv"
    df.to_csv(output_path, index=False)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import subprocess

# --- 1. Fetch Data ---
# This uses the wget command from your notebook to download the data.
print("Attempting to download data...")
try:
    subprocess.run(["wget", "-q", "https://raw.githubusercontent.com/Sanjidx090/Work/refs/heads/main/mlolympiadbd2025/train.csv"], check=True)
    train_df = pd.read_csv('train.csv')
    print("train.csv downloaded successfully.")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Could not download the data. Please ensure 'train.csv' is available.")
    # Create a sample dataframe if download fails, to ensure the code runs for demonstration
    train_data = {'RiskLevel': np.random.choice([0, 1, 2], 811, p=[0.40, 0.33, 0.27])}
    train_df = pd.DataFrame(train_data)
    print("Using sample data for demonstration.")


# --- 2. Prepare Data for Splitting ---
X = train_df.drop('RiskLevel', axis=1, errors='ignore')
y = train_df['RiskLevel']

# --- 3. Create 15-Fold Split and Get Distributions ---
n_splits = 15
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_distributions = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Check the distribution in the validation set for each fold
    y_val_fold = y.iloc[val_idx]
    dist = y_val_fold.value_counts().sort_index()
    dist.name = f"Fold {fold+1}"
    fold_distributions.append(dist)

# Create a DataFrame from the list of distributions
dist_df = pd.concat(fold_distributions, axis=1).T
dist_df.columns = ['Low Risk (0)', 'Mid Risk (1)', 'High Risk (2)']

# --- 4. Create the Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 8))

dist_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

ax.set_title('Distribution of Risk Classes Across 15 Folds', fontsize=16, fontweight='bold')
ax.set_xlabel('Cross-Validation Fold', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.legend(title='Risk Level')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the plot
plt.show()

