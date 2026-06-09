import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




## IMPORT DATASETS ##

train= pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test= pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col="id")

## Basic Inspection of the data ##
print("-----Shape of train data-----")
print(train.shape)
print("\n-----Shape of test data-----")
print(test.shape)
print("\n-----Data types of train dataset-----")
train.info()
print("\n-----First 5 rows of train data-----")
train.head(5)



print(train.isna().sum())
train.drop_duplicates(inplace=True)




# Target distribution
y_distribution = train['y'].value_counts(normalize=True)

# colors
custom_colors = ['#7e7f9a', '#f3de8a']

# floating effect
explode = [0.05] * len(y_distribution)

plt.figure(figsize=(6, 6))
wedges, texts, autotexts = plt.pie(
    y_distribution,
    labels=y_distribution.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=custom_colors,
    explode=explode,
    wedgeprops={
        'width': 0.5,   # hollow effect
        'antialiased': True
    },
    textprops={'fontsize': 12, 'weight': 'bold', 'color': '#ffffff'}
)

# Style the percentages 
for autotext in autotexts:
    autotext.set_color('#ffffff')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

plt.title('Distribution of the Target Variable', fontsize=14, fontweight='bold', color='#333333')
plt.axis('equal')
plt.tight_layout()
plt.show()




import math

numerics = train.select_dtypes(include=['number'])

numerics_filtered = [col for col in numerics if col != 'y']

sns.set_theme(style="whitegrid", font="Arial", font_scale=1.1)
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.edgecolor": "#cccccc",
})




# Number of plots
n_cols = 3
n_plots = len(numerics_filtered)
n_rows = math.ceil(n_plots / n_cols)

# subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()  # Flatten in case of multiple rows

# Plot each histogram
for i, col in enumerate(numerics_filtered):
    sns.histplot(
    train[col],
    kde=True,
    color="#7e7f9a",          
    edgecolor="white",
    bins=30,
    ax=axes[i],
    line_kws={"color": "#f3de8a", "linewidth": 2}  
)
    axes[i].set_title(f"Distribution of {col}", color="#333333")
    axes[i].set_xlabel(col, color="#333333")
    axes[i].set_ylabel("Frequency", color="#333333")



# empty subplots? remove
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()





# style and font 
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

#  categorical columns
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

n_cols = 3
n_plots = len(cat_cols)
n_rows = math.ceil(n_plots / n_cols)

# subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    # Count values for ordering and heights
    order = train[col].value_counts().index

    counts = train[col].value_counts().reindex(order).values
    
    # Draw countplot
    sns.countplot(
        data=train, x=col, order=order,
        color="#7e7f9a",
        ax=axes[i]
    )

    axes[i].set_title(f"{col}")
    axes[i].set_xlabel('')
    axes[i].tick_params(axis="x", rotation=50, labelsize=10)
    axes[i].set_ylabel("Count")


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()




from IPython.display import display

for col in cat_cols:
    # Count values for each category
    counts = train[col].value_counts()
    
    # DataFrame for nicer display
    df = pd.DataFrame({
        'Category': counts.index,
        'Count': counts.values
    })
    
    print(f"Variable: {col}")
    display(df)





def plot_percentage_stacked_bar_grid(train, custom_colors):
    cat_cols = train.select_dtypes(include=['object']).columns
    n = len(cat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols  # ensures enough rows even if n isn't exactly divisible

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 12), sharey=True)
    axes = axes.flatten()  # Flatten to iterate easily

    for ax, cat_col in zip(axes, cat_cols):
        ct = pd.crosstab(train[cat_col], train['y'])
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        ct_norm.plot(kind='bar', stacked=True, ax=ax, color=custom_colors)
        ax.set_title(f'By {cat_col.capitalize()}')
        ax.set_xlabel(cat_col.capitalize())
        ax.set_ylabel('Proportion')
        ax.tick_params(axis='x', rotation=45)


    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_percentage_stacked_bar_grid(train, custom_colors)




# style and fonts
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# remove out 'y' from numerics 
numerics_filtered = [col for col in numerics if col != 'y']

# sample of the data for efficiency
sampled_train = train.sample(20000, random_state=1)

#  layout setup
n_cols = 3
n_plots = len(numerics_filtered)
n_rows = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

# plot each violin plot
for i, col in enumerate(numerics_filtered):
    sns.violinplot(x='y', y=col, data=sampled_train, ax=axes[i], palette=['#7e7f9a', '#f3de8a'])
    axes[i].set_title(f"{col} vs y")
    axes[i].set_xlabel("y")
    axes[i].set_ylabel(col)


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()





# Select numeric columns
numerics = train.select_dtypes(include=['number']).columns

# Correlation matrix
corr = train[numerics].corr()

# Mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Clean theme
sns.set_theme(style="whitegrid")
sns.set(font_scale=0.85)

# Figure size
n = corr.shape[0]
fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(6, n * 0.45)))

# palette
cmap = sns.light_palette("seagreen", as_cmap=True)  # or "forestgreen"

# Heatmap
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    annot_kws={"size": 8, "weight": "bold"},
    cmap=cmap, vmin=0, vmax=1,  # Only positive scale
    linewidths=0.4, linecolor="white",
    square=True, cbar_kws={"shrink": 0.7, "label": "Correlation"}
)

# label readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

# Title
ax.set_title("Feature Correlation Matrix", fontsize=12, weight="bold", pad=15)

plt.tight_layout()
plt.show()



def prep_features(df):
    df = df.copy()

    # -------------------------------
    # Binary encoding for yes/no features
    # -------------------------------
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # -------------------------------
    # Handle 'pdays'
    # -------------------------------
    # Create a flag for whether client was previously contacted
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    # Fill -1 with a high number so tree can still split effectively
    df['pdays_filled'] = df['pdays'].where(df['pdays'] != -1, df['pdays'].max() + 1)
    df['pdays_bin'] = pd.cut(
        df['pdays_filled'],
        bins=[-1, 7, 30, 180, np.inf],
        labels=['<week', '<month', '<6months', 'never']
    ).astype('category').cat.codes

    # -------------------------------
    # Derived ratio features
    # -------------------------------
    # Average call duration per campaign attempt
    df['avg_duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    # Ratio of current campaign contacts to previous contacts
    df['campaign_to_previous_ratio'] = df['campaign'] / (df['previous'] + 1)

    # -------------------------------
    # Cyclical encoding for months
    # -------------------------------
    month_num = df['month'].map({
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    })
    df['month_sin'] = np.sin(2 * np.pi * month_num / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_num / 12)

    # -------------------------------
    # Balance
    # -------------------------------
    df['log_balance'] = np.log1p(df["balance"])
    df['balance_bin'] = pd.cut(
        df['balance'],
        bins=[-np.inf, 0, 500, 2000, np.inf],
        labels=['neg_debt', 'low', 'medium', 'high'])
    df['balance_above_mean'] = (df['balance'] > df['balance'].mean()).astype(int)
    df['is_negative_balance'] = (df['balance'] < 0).astype(int)

    # -------------------------------
    # Combined financial obligations
    # -------------------------------
    df["financial_obligations"] = ((df["default"] == 1) |
                                   (df["housing"] == 1) |
                                   (df["loan"] == 1)).astype(int) 

    # -------------------------------
    # High-value customer feature
    # -------------------------------
    df["is_high_value_customer"] = (
        (df["balance"] > 5000) &
        (df["education"].isin(["tertiary"])) &
        (df["housing"] == 0) &  # no housing loan
        (df["loan"] == 0)       # no personal loan
    ).astype(int)


    # -------------------------------
    # Duration
    # -------------------------------
    df['log_duration'] = np.log1p(df["duration"])
    # Short: <2min, Medium: 2â€“6min, Long: >6min
    df['duration_cat'] = pd.cut(
        df['duration'],
        bins=[-np.inf, 120, 360, np.inf],
        labels=['short', 'medium', 'long']
    )
    df['duration_per_call'] = df['duration'] / (df['campaign'] + 1)

    # -------------------------------
    # Encoding categorical bins for tree-based models
    # -------------------------------
    # For tree-based models, label encoding is enough, no need for one-hot
    for col in ['balance_bin', 'duration_cat']:
        df[col] = df[col].astype('category').cat.codes

    return df

train_fe = prep_features(train)
test_fe = prep_features(test)



from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)


# -------------------------------
# Function to evaluate the modesl
# -------------------------------
def evaluate_model(model, X_valid, y_valid, threshold=0.5, model_name='Model'):
    """
    Evaluate a binary classifier:
    - Plots ROC curve and confusion matrix side by side
    - Computes and prints main metrics in a table
    """
    # Get predicted probabilities
    y_proba = model.predict_proba(X_valid)[:, 1]

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_valid, y_proba)
    roc_auc = auc(fpr, tpr)

    # Convert probabilities to class predictions
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_valid, y_pred)

    # Side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    axes[0].plot(fpr, tpr, color='#4e79a7', lw=2, label=f'{model_name} ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, label='Random Guess')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title(f'ROC Curve â€“ {model_name}', fontsize=14, fontweight='bold', color='#333333')
    axes[0].legend(loc='lower right')
    axes[0].grid(False)
    axes[0].set_facecolor('white')

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=axes[1], cmap=plt.cm.Blues, colorbar=False)
    axes[1].set_title(f'Confusion Matrix â€“ {model_name}', fontsize=14, fontweight='bold', color='#333333')
    axes[1].grid(False)
    axes[1].set_facecolor('white')

    plt.tight_layout()
    plt.show()

    # Compute metrics
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)

    # Print metrics table
    metrics_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    })
    metrics_table['Value'] = metrics_table['Value'].apply(lambda x: f'{x:.3f}')

    print(metrics_table)
    return metrics_table




from sklearn.model_selection import train_test_split

# -------------------------------------
# Preparation of training and test data
# -------------------------------------

# Define features and target
features = train_fe.drop(columns='y').columns.tolist()
X = train_fe[features]
y = train_fe['y']

# Split train/validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Categorical columns
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# Mapping labels for later use
y_train_num = y_train.map({'no': 0, 'yes': 1}) if y_train.dtype == 'O' else y_train
y_valid_num = y_valid.map({'no': 0, 'yes': 1}) if y_valid.dtype == 'O' else y_valid




print(y_train.mean(), y_valid.mean())


from catboost import CatBoostClassifier, Pool

# CatBoost Pool objects
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

model_catboost = CatBoostClassifier(
    iterations=1000, #tried 400,600
    learning_rate=0.1,#tried 0.05
    depth=10,# tried 6, 8
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100
)

model_catboost.fit(train_pool, eval_set=valid_pool)



metrics_catboost = evaluate_model(model_catboost, valid_pool, y_valid, threshold=0.5, model_name='CatBoost')



#---------------
# Submission
#---------------
features = test_fe.columns.tolist()
cat_features = test_fe.select_dtypes(include=['object', 'category']).columns.tolist()

# Pool for test data 
test_pool = Pool(test_fe[features], cat_features=cat_features)

# Prediction of  probabilities and dataframe for submission
y_pred_proba = model_catboost.predict_proba(test_pool)[:, 1]
submission_catboost = pd.DataFrame({
    "id": test_fe.index, 
    "y": y_pred_proba
})

submission_catboost.to_csv("submission_catboost.csv", index=False)
submission_catboost



from lightgbm import LGBMClassifier
import lightgbm as lgb

for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')


model_lgb = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)

model_lgb.fit(X_train, y_train)

metrics_lgbm = evaluate_model(model_lgb, X_valid, y_valid, threshold=0.5, model_name='LightGBM')



#---------------
# Submission
#---------------

for col in cat_features:
    test_fe[col] = test_fe[col].astype('category')

# Prepare the test set
y_proba_lgb = model_lgb.predict_proba(test_fe)[:, 1]


submission_lgb = pd.DataFrame({
    "id": test_fe.index,  
    "y": y_proba_lgb
})
submission_lgb.to_csv('submission_lgb.csv', index=False)
submission_lgb


from xgboost import XGBClassifier


# Initialize classifier
model_xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    enable_categorical=True,
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=10, 
    min_child_weight=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=50
)

# Fit the model
model_xgb.fit(
    X_train, y_train_num,
    eval_set=[(X_train, y_train_num), (X_valid, y_valid_num)],
    verbose=100
)

# Best iteration & score
print("Best ROC AUC:", model_xgb.best_score)
print("Best iteration:", model_xgb.best_iteration)



metrics_xgboost = evaluate_model(model_xgb, X_valid, y_valid, threshold=0.5, model_name='XGBoost')



#---------------
# Submission
#---------------
for col in cat_features:
    test_fe[col] = test_fe[col].astype('category')

# Make predictions on the test set using the fitted XGBClassifier
y_proba_xgb = model_xgb.predict_proba(test_fe)[:, 1]

# Prepare submission
submission_xgb = pd.DataFrame({
    "id": test_fe.index,
    "y": y_proba_xgb
})

# Save to CSV
submission_xgb.to_csv('/kaggle/working/submission_xgb.csv', index=False)
submission_xgb


from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Make sure categorical features are set to 'category' dtype
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')


model_lgb = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    n_estimators=1000,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, valid_idx in cv.split(X, y):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=0)  # 0 to suppress verbose
        ]
    )

    
    y_pred = model_lgb.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc)

print("AUC per fold:", auc_scores)
print("Mean AUC:", np.mean(auc_scores))
print("Std AUC:", np.std(auc_scores))



# Make sure categorical features are set to 'category' dtype
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')


model_lgb.fit(X, y) # training on the full data set before predictions for submission




#---------------
# Submission
#---------------

for col in cat_features:
    test_fe[col] = test_fe[col].astype('category')

# Prepare the test set
y_proba_lgb = model_lgb.predict_proba(test_fe)[:, 1]


submission_lgb = pd.DataFrame({
    "id": test_fe.index,  
    "y": y_proba_lgb
})
submission_lgb.to_csv('submission_lgb_final.csv', index=False)
submission_lgb

