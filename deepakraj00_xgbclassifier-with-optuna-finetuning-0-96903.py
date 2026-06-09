import pandas as pd 
print(f"pandas version: {pd.__version__}")

import matplotlib
print(f"matplotlib version: {matplotlib.__version__}")

import numpy as np
print(f"NumPy version: {np.__version__}")

import scipy as sp
print(f"SciPy version: {sp.__version__}") 

import IPython
from IPython import display
print(f"IPython version: {IPython.__version__}") 

import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

#misc libraries
import random
import time
import os

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)


print(os.listdir("../input/playground-series-s5e8"))



#Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
import optuna

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score , train_test_split,cross_validate
from sklearn.metrics import make_scorer, f1_score , roc_auc_score,classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve


#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


data_raw = pd.read_csv('../input/playground-series-s5e8/train.csv')
data_raw = data_raw.set_index("id")
data_raw['education'] = data_raw['education'].replace("unknown", np.nan)

data_test  = pd.read_csv('../input/playground-series-s5e8/test.csv')
data_test = data_test.set_index("id")
data_test['education'] = data_test['education'].replace("unknown", np.nan)

data_cleaner = [data_raw, data_test]

print (data_raw.info()) # Shape is (7_50_000,18)
data_raw.head()


data_raw.describe(include='all')


print('Train columns with null values:\n', data_raw.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_test.isnull().sum())
print("-"*10)


###COMPLETING: complete or delete missing values in dataset
# Delete High Collinearity Features
# drop_column = ['id']
# ata_raw.drop(drop_column, axis=1, inplace = True)

# Preprocessing for numerical data
# numerical_transformer = SimpleImputer(strategy='median')


###CREATE: Feature Engineering for train and test/validation dataset

data_raw['total_contact']= data_raw['campaign'] + data_raw['previous']
data_raw['recent_contact_flag'] = [1 if x < 30 else 0 for x in data_raw['pdays']]
data_raw["pdays"] = [0 if x==-1 else x for x in data_raw["pdays"]]
data_raw['contact_intensity'] = data_raw['total_contact'] / (data_raw['pdays']+1)


# Make a copy named unnecesary_df
unnecesary_df = data_raw.copy()

# Compute debt exposure directly
unnecesary_df['debt_exposure_ratio'] = (
    unnecesary_df['loan'].eq('yes').astype(int) +
    unnecesary_df['housing'].eq('yes').astype(int) +
    unnecesary_df['default'].eq('yes').astype(int)
) / unnecesary_df['balance'].replace(0, pd.NA)

# Replace NaN with 0
unnecesary_df['debt_exposure_ratio'] = unnecesary_df['debt_exposure_ratio'].fillna(0)

data_raw['debt_exposure_ratio']=unnecesary_df['debt_exposure_ratio'].copy()



data_raw.head()


# job is ohe
# marital is ohe
# education is ordinal encoding and impute unknown
# default is binary encoding
# housing is binary encoding
# loan is binary encoding
# contact is ohe
# month is Cyclic Encoding (sine/cosine)
# poutcome is ohe


from sklearn.preprocessing import FunctionTransformer

def iqr_clip(X):
    X = pd.DataFrame(X)
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return X.clip(lower=lower, upper=upper, axis=1)

iqr_transformer = FunctionTransformer(iqr_clip, validate=False)



# Columns
ohe_cols = ["job", "marital", "contact", "poutcome", "month"]
ord_cols = ["education"]
bin_cols = ["default", "housing", "loan"]
num_cols = [x for x in data_raw.columns if data_raw[x].dtype in ['float64','int64'] and x != "y"]

# One-hot for categorical
categorical_transformer_ohe = OneHotEncoder(handle_unknown='ignore')

# Ordinal encoding for education
categorical_transformer_ord = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(categories=[['primary','secondary','tertiary']], handle_unknown='use_encoded_value', unknown_value=-1))
])

# Binary columns
binary_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Numerical columns
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('iqr_clip', iqr_transformer),
    ('scaler', StandardScaler())
])

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('ohe', categorical_transformer_ohe, ohe_cols),
    ('ord', categorical_transformer_ord, ord_cols),
    ('bin', binary_transformer, bin_cols),
    ('num', numerical_transformer, num_cols)
])



data_raw.describe()


data_raw.describe(include='object')


# Discrete Variable Correlation by Subscription(Proportions of target per category) 
categorical_cols = [x for x in data_raw.columns if data_raw[x].dtype not in ['float64','int64']]

for col in categorical_cols:
    # Proportion of target per category (y=1)
    proportions = (
        data_raw.groupby(col)["y"]
        .mean()
        .reset_index()
        .rename(columns={"y": "Subscribed"})
    )

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14,4))

    # Left: proportion of y=1 per category
    sns.barplot(data=proportions, x=col, y='Subscribed', ax=axes[0],
                order=proportions.sort_values('Subscribed', ascending=False)[col])
    axes[0].set_title(f'Proportion of Subscribed by {col}')
    axes[0].tick_params(axis='x', rotation=45)

    # Right: count of samples per category
    sns.countplot(data=data_raw, x=col, ax=axes[1], 
                  order=data_raw[col].value_counts().index)
    axes[1].set_title(f'Counts per {col}')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    print('-'*10, '\n')


import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = [x for x in data_raw.columns if data_raw[x].dtype in ['float64','int64'] and x != "y"]

for col in numerical_cols:
    
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # Histogram â†’ distribution
    sns.histplot(data_raw[col], kde=True, ax=axes[0],bins=30, color='skyblue') 
    axes[0].set_title(f'Distribution of {col}')

    # Boxplot â†’ outliers
    sns.boxplot(x=data_raw[col], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()
    
    print('-'*10, '\n')



import itertools
import math
import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = [x for x in data_raw.columns if data_raw[x].dtype in ['float64','int64'] and x != "y"]

pairs = list(itertools.combinations(numerical_cols, 2))
n_pairs = len(pairs)

# Choose grid size (rows x cols)
n_cols = 3  # adjust as needed
n_rows = math.ceil(n_pairs / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
axes = axes.flatten()  # flatten for easy iteration

for i, (col1, col2) in enumerate(pairs):
    corr = data_raw[col1].corr(data_raw[col2])
    sns.scatterplot(data=data_raw, x=col1, y=col2, color='skyblue', ax=axes[i])
    axes[i].set_title(f'{col1} vs {col2}\nr={corr:.2f}')


plt.tight_layout()
plt.show()



numerical_cols = [x for x in data_raw.columns if data_raw[x].dtype in ['float64','int64'] and x != "y"]

for col in numerical_cols:
    plt.figure(figsize=(8,4))
    sns.violinplot(x='y', y=col, data=data_raw,hue="y")
    plt.title(f'Violin plot of {col} by Target (y)')
    plt.xlabel('Target (y)')
    plt.ylabel(col)
    plt.show()
    print('-'*10, '\n')



numerical_cols = [x for x in data_raw.columns if data_raw[x].dtype in ['float64','int64']]

# Compute correlation matrix
corr_matrix = data_raw[numerical_cols].corr()

# Create a polished heatmap
plt.figure(figsize=(14,12))
colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr_matrix,
    cmap=colormap,
    square=True,
    annot=True,
    fmt=".2f",
    linewidths=0.1,
    linecolor='white',
    cbar_kws={'shrink':0.9},
    annot_kws={'fontsize':12},
    vmax=1.0
)

plt.title('Correlation Heatmap of Numerical Features', y=1.05, size=15)
plt.show()



y_raw = data_raw["y"].copy()
X_raw = data_raw.drop(["y"],axis=1).copy()


subset_size = 50000
X_subset, _, y_subset, _ = train_test_split(
    X_raw, y_raw,
    train_size=subset_size,
    stratify=y_raw,
    random_state=42
)


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(n_jobs=-1),
    
    #GLM
    linear_model.LogisticRegression(max_iter=1000),
    
    #Navies Bayes
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    
    XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric='logloss')    
    ]


# Stratified 5-fold cross validation (preserves class ratio, important for imbalance)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train F1 Mean', 'MLA Val F1 Mean', 'MLA Val F1 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = y_raw


# Define F1 scorer for binary classification
f1_scorer = make_scorer(f1_score, average='binary')

# Index through MLA and save performance to table
row_index = 0
for alg in MLA:

    # Set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    # Create pipeline: preprocessing + classifier
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', alg)
    ])
    
    # Cross-validate pipeline on stratified subset using F1 scoring
    cv_results = cross_validate(pipe, X_subset, y_subset, cv=cv,
                                scoring=f1_scorer,
                                return_train_score=True, n_jobs=-1)  # n_jobs=-1 speeds up where supported

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train F1 Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Val F1 Mean'] = cv_results['test_score'].mean()   
    # If this is a non-biased random sample, then +/-3 standard deviations (std) from the mean captures 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Val F1 3*STD'] = cv_results['test_score'].std()*3   # worst-case estimate
    
    row_index += 1

# Print and sort table by validation F1-score
MLA_compare.sort_values(by=['MLA Val F1 Mean'], ascending=False, inplace=True)
MLA_compare



# y_raw = your true labels
# Naive baseline: predict all 0 (majority class)
y_pred_baseline = np.zeros_like(y_raw)  # predict negative for all
baseline_f1 = f1_score(y_raw, y_pred_baseline)
print("Baseline F1-score:", baseline_f1)


# ROC-AUC
# For ROC-AUC, we need predicted probabilities; all zeros = 0 probability for positive class
y_pred_prob_baseline = np.zeros_like(y_raw)
baseline_roc_auc = roc_auc_score(y_raw, y_pred_prob_baseline)
print("Baseline ROC-AUC:", baseline_roc_auc)


# ---------------------------
#  Optuna Objective Function
# ---------------------------
def objective(trial):

    # Model hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 500, 3000)   # larger range for boosting rounds
    max_depth = trial.suggest_int("max_depth", 3, 15)             # allow deeper trees
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True) 
    subsample = trial.suggest_float("subsample", 0.6, 1.0)        # slightly narrower for stability
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 50)  # higher upper bound for imbalanced datasets
    
    # Regularization hyperparameters
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)        # L1
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)      # L2
    gamma = trial.suggest_float("gamma", 0.0, 10.0)                # Split reduction
    
    # Class imbalance handling
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 20.0)

    # Additional boosting params
    max_bin = trial.suggest_int("max_bin", 128, 512)               # controls histogram binning (GPU)
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    sampling_method = trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"])
    
    # Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_scores = []

    for train_idx, val_idx in cv.split(X_raw, y_raw):
        X_tr, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_val = y_raw.iloc[train_idx], y_raw.iloc[val_idx]

        # Pipeline with preprocessor
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                gamma=gamma,
                scale_pos_weight=scale_pos_weight,
                max_bin=max_bin,
                grow_policy=grow_policy,
                sampling_method=sampling_method,
                use_label_encoder=False,
                eval_metric="auc",
                random_state=42,
                tree_method="gpu_hist",   # GPU optimized
                gpu_id=0
            ))
        ])

        # Fit on training fold
        pipe.fit(X_tr, y_tr)

        # Predict on validation fold
        y_val_proba = pipe.predict_proba(X_val)[:, 1]
        roc_scores.append(roc_auc_score(y_val, y_val_proba))

    # Average metrics across folds
    return np.mean(roc_scores)

# ---------------------------
#  Run Optuna Study
# ---------------------------
study = optuna.create_study(direction="maximize", study_name="xgb_roc")
study.optimize(objective, n_trials=100, timeout=3600, show_progress_bar=True)

best_trial = study.best_trial
print("Best Value:", best_trial.value)
print("Best Params:", best_trial.params)



df = study.trials_dataframe()
df = df[['number', 'value', 'params_n_estimators', 'params_max_depth', 
         'params_learning_rate', 'params_subsample', 'params_colsample_bytree']]
df


# ---------------------------
# Step 1: Split final validation set
# ---------------------------
subset_size = 50000  # use if dataset is huge, else use full data
X_final_val, X_train, y_final_val, y_train = train_test_split(
    X_raw, y_raw,
    train_size=subset_size,
    stratify=y_raw,
    random_state=42
)

# ---------------------------
# Step 2: Transform data and train final model with early stopping
# ---------------------------
best_params = best_trial.params

# Preprocess the data first
X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_final_val)

final_clf = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    tree_method='gpu_hist',  # GPU if available
    gpu_id=0
)

# Fit with early stopping
final_clf.fit(
    X_train_trans, y_train,
    eval_set=[(X_val_trans, y_final_val)],
    early_stopping_rounds=10,
    verbose=False
)

# ---------------------------
# Step 3: Predict on validation/test set
# ---------------------------
y_pred = final_clf.predict(X_val_trans)
y_proba = final_clf.predict_proba(X_val_trans)[:,1]

# ---------------------------
# Step 4: Metrics
# ---------------------------
print("Classification Report:\n")
print(classification_report(y_final_val, y_pred))

print("Confusion Matrix:\n")
cm = confusion_matrix(y_final_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

f1_final = f1_score(y_final_val, y_pred, average='macro')
roc_final = roc_auc_score(y_final_val, y_proba)
print(f"F1 (macro): {f1_final:.4f}")
print(f"ROC-AUC: {roc_final:.4f}")

# ---------------------------
# Step 5: ROC Curve
# ---------------------------
fpr, tpr, _ = roc_curve(y_final_val, y_proba)
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_final:.4f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ---------------------------
# Step 6: Precision-Recall Curve
# ---------------------------
precision, recall, _ = precision_recall_curve(y_final_val, y_proba)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()



import joblib

final_pipe_full = Pipeline([
    ('preprocessor', preprocessor),  # reuse the same preprocessor
    ('classifier', XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        tree_method='gpu_hist',  # GPU if available
        gpu_id=0
    ))
])

# Fit on entire dataset
final_pipe_full.fit(X_raw, y_raw)

# ---------------------------
# Step 8: Save pipeline for competition/test set
# ---------------------------
joblib.dump(final_pipe_full, "final_model_pipeline.pkl")
print("Pipeline saved as 'final_model_pipeline.pkl'")



###CREATE: Feature Engineering for train and test/validation dataset

data_test['total_contact']= data_test['campaign'] + data_test['previous']
data_test['recent_contact_flag'] = [1 if x < 30 else 0 for x in data_test['pdays']]
data_test["pdays"] = [0 if x==-1 else x for x in data_test["pdays"]]
data_test['contact_intensity'] = data_test['total_contact'] / (data_test['pdays']+1)

# Make a copy named unnecesary_df
unnecesary_df = data_test.copy()

# Compute debt exposure directly
unnecesary_df['debt_exposure_ratio'] = (
    unnecesary_df['loan'].eq('yes').astype(int) +
    unnecesary_df['housing'].eq('yes').astype(int) +
    unnecesary_df['default'].eq('yes').astype(int)
) / unnecesary_df['balance'].replace(0, pd.NA)

# Replace NaN with 0
unnecesary_df['debt_exposure_ratio'] = unnecesary_df['debt_exposure_ratio'].fillna(0)

data_test['debt_exposure_ratio']=unnecesary_df['debt_exposure_ratio'].copy()



# ---------------------------
# Load and predict on test set
# ---------------------------
loaded_pipe = joblib.load("final_model_pipeline.pkl")
y_test_pred = loaded_pipe.predict(data_test)       # X_test is competition/test set
y_test_proba = loaded_pipe.predict_proba(data_test)[:,1]



# ---------------------------
# Prepare submission dataframe
# ---------------------------
submission = pd.DataFrame({
    "id": data_test.index,  # use the index as id
    "y": y_test_proba
})

# ---------------------------
# Save submission file
# ---------------------------
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")

