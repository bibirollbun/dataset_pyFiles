import pandas as pd
import numpy as np
import os 
import time 
import math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.io as pio
import plotly.subplots as sp
from scipy.stats import skew, kurtosis, zscore
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency
import plotly.figure_factory as ff  

warnings.filterwarnings('ignore')
sns.set(style='darkgrid')
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')


train.head()


train.info()


train.describe().round(2)


print("Duplicated Rows:",train.duplicated().sum())
print("-"*30)
print("Number of Rows:",train.shape[0])
print("-"*30)
print("Number of Columns:",train.shape[1])


train.isnull().sum()


print("Numeric Col Names",train.select_dtypes(include=['number']).columns)


print("Categorical Col Names",train.select_dtypes(include=['object']).columns)


color_palette = ['#f2f0f7', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']


num_col = [
    'annual_income', 'debt_to_income_ratio', 'credit_score',
    'loan_amount', 'interest_rate'
]

cat_col = [
    'gender', 'marital_status', 'education_level', 'employment_status',
    'loan_purpose', 'grade_subgrade'
]

target_col = 'loan_paid_back'

FEATURES = num_col + cat_col
CATS = cat_col


for col in cat_col:
    print(f"Unique categories in '{col}' column: {train[col].unique()}")
    print("<--- --- --- --- --- --- --- --- --- --->\n")


print("\n===== Skewness & Kurtosis =====")
for col in ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'loan_paid_back']:
    print(f"{col:25s} | Skewness: {train[col].skew():.4f} | Kurtosis: {train[col].kurtosis():.4f}")


for col in cat_col:
    # Crosstab
    ct = pd.crosstab(train[col], train['loan_paid_back'])
    ct_perc = (ct.T / ct.sum(axis=1)).T * 100
    print(f"\n===== Crosstab: {col} vs Loan Paid Back =====")
    print(ct_perc.round(2))

    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi-Square Test for {col}: Ï‡Â²={chi2:.2f}, p-value={p:.4f}")


class_dist = train['loan_paid_back'].value_counts(normalize=True)
print("\n===== Loan Paid Back Distribution =====")
print(class_dist.round(3))

imbalance_ratio = class_dist.min() / class_dist.max()
print(f"\nClass Imbalance Ratio (minority/majority): {imbalance_ratio:.3f}")


gender_count = train['gender'].value_counts().reset_index()
gender_count.columns = ['gender', 'Count']

fig = px.bar(
    gender_count,
    x='gender',               
    y='Count',                
    color='gender',
    color_discrete_sequence=px.colors.sequential.Purp,
    title="Gender Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


marital_count = train['marital_status'].value_counts().reset_index()
marital_count.columns = ['marital_status', 'Count']

fig = px.bar(
    marital_count,
    x='marital_status',
    y='Count',
    color='marital_status',
    color_discrete_sequence=px.colors.sequential.Purp,
    title="Marital Status Distribution",
    text='Count'  
)

fig.update_layout(width=600, height=400)
fig.show()


education_count = train['education_level'].value_counts().reset_index()
education_count.columns = ['education_level', 'Count']

fig = px.bar(
    education_count,
    x='education_level',
    y='Count',
    color='education_level',
    color_discrete_sequence=px.colors.sequential.Purp,
    title="Education Level Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


employment_count = train['employment_status'].value_counts().reset_index()
employment_count.columns = ['employment_status', 'Count']

fig = px.bar(
    employment_count,
    x='employment_status',
    y='Count',
    color='employment_status',
    color_discrete_sequence=px.colors.sequential.Purp,
    title="Employment Status Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


loan_purpose_count = train['loan_purpose'].value_counts().reset_index()
loan_purpose_count.columns = ['loan_purpose', 'Count']

fig = px.bar(
    loan_purpose_count,
    x='loan_purpose',
    y='Count',
    color='loan_purpose',
    color_discrete_sequence=px.colors.sequential.Purp,
    title="Loan Purpose Distribution",
    text='Count'
)

fig.update_layout(width=700, height=400)
fig.show()


top10 = train['grade_subgrade'].value_counts().head(10)

fig = px.bar(
    top10,
    x=top10.index,
    y=top10.values,
    text=top10.values,  
    title='Top 10 Frequent Grade_Subgrade Categories',
    color=top10.values,
    color_continuous_scale=px.colors.sequential.Purp
)

fig.update_layout(
    width=700, 
    height=400,
    xaxis_title='Grade_Subgrade',
    yaxis_title='Count'
)
fig.show()


fig, axes = plt.subplots(nrows=len(num_col), ncols=1, figsize=(8, 18))

for i, col in enumerate(num_col):
    sns.histplot(
        train[col],
        bins=50,
        kde=True,
        ax=axes[i],
        color=color_palette[i % len(color_palette)],
        edgecolor='black',
        linewidth=0.5
    )
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


loan_status_counts = train['loan_paid_back'].value_counts().reset_index()
loan_status_counts.columns = ['loan_paid_back', 'count']

fig = px.pie(
    loan_status_counts, 
    values='count', 
    names='loan_paid_back', 
    title='Target Distribution',
    color='loan_paid_back', 
    color_discrete_sequence=px.colors.sequential.Purp
)

fig.update_layout(
    width=500,
    height=400
)

fig.show()


plt.figure(figsize=(6, 5))
sns.boxplot(x='loan_paid_back', y='debt_to_income_ratio', data=train, palette="YlGnBu")
plt.title("Debt-to-Income Ratio vs Loan Paid Back")
plt.xlabel("Loan Paid Back")
plt.ylabel("Debt-to-Income Ratio")
plt.show()


corr_matrix = train[num_col].corr()
print(corr_matrix)


plt.figure(figsize=(6, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
    cbar=True, square=True, linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()


fig = px.bar(
    train.groupby('loan_paid_back', as_index=False)['annual_income'].mean().round(3),
    x='loan_paid_back',
    y='annual_income',
    color='annual_income',
    color_continuous_scale=px.colors.sequential.Purp,
    title='Average Annual Income by Loan Paid Back'
)

fig.update_layout(
    width=500,
    height=400,
    xaxis_title='Loan Paid Back (0 = No, 1 = Yes)',
    yaxis_title='Average Annual Income',
    coloraxis_showscale=False
)

fig.show()


print("===== Mean Comparison by Loan Paid Back =====")
for col in num_col:
    means = train.groupby('loan_paid_back')[col].mean().round(2)
    print(f"{col:25s}: {means.to_dict()}")

for col in num_col:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.boxplot(data=train, x='loan_paid_back', y=col, ax=axes[0])
    axes[0].set_title(f'{col} vs Loan Paid Back â€” Boxplot')
    sns.violinplot(data=train, x='loan_paid_back', y=col, ax=axes[1])
    axes[1].set_title(f'{col} vs Loan Paid Back â€” Violinplot')
    plt.tight_layout()
    plt.show()


fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Interest Rate vs Credit Score", "Interest Rate vs Loan Amount"),
    horizontal_spacing=0.08
)

scatter1 = px.scatter(
    train,
    x='credit_score',
    y='interest_rate',
    color='loan_paid_back',
    color_continuous_scale=px.colors.sequential.Purp,
    opacity=0.5
).data[0]  # extract trace

scatter2 = px.scatter(
    train,
    x='loan_amount',
    y='interest_rate',
    color='loan_paid_back',
    color_continuous_scale=px.colors.sequential.Purp,
    opacity=0.5
).data[0]

fig.add_trace(scatter1, row=1, col=1)
fig.add_trace(scatter2, row=1, col=2)

fig.update_layout(
    title_text="Interest Rate Relationships with Key Predictors",
    width=850,
    height=450,
    coloraxis=dict(colorscale=px.colors.sequential.Purp),
    showlegend=True
)

fig.update_xaxes(title_text="Credit Score", row=1, col=1)
fig.update_yaxes(title_text="Interest Rate (%)", row=1, col=1)
fig.update_xaxes(title_text="Loan Amount", row=1, col=2)
fig.update_yaxes(title_text="Interest Rate (%)", row=1, col=2)

fig.show()


print("Creating engineered features...")
cat_cols = train.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    if col in orig.columns:
        orig[col] = orig[col].astype('category')


train['income_to_loan_ratio'] = train['annual_income'] / (train['loan_amount'] + 1e-5)
train['income_to_interest_ratio'] = train['annual_income'] / (train['interest_rate'] + 1e-5)
train['loan_burden_index'] = train['loan_amount'] / (train['annual_income'] + 1e-5)
train['risk_score'] = (train['debt_to_income_ratio'] * train['interest_rate']) / (train['credit_score'] + 1e-5)
train['credit_utilization_flag'] = (train['debt_to_income_ratio'] > 0.3).astype(int)
train['is_high_credit'] = (train['credit_score'] > 700).astype(int)


test['income_to_loan_ratio'] = test['annual_income'] / (test['loan_amount'] + 1e-5)
test['income_to_interest_ratio'] = test['annual_income'] / (test['interest_rate'] + 1e-5)
test['loan_burden_index'] = test['loan_amount'] / (test['annual_income'] + 1e-5)
test['risk_score'] = (test['debt_to_income_ratio'] * test['interest_rate']) / (test['credit_score'] + 1e-5)
test['credit_utilization_flag'] = (test['debt_to_income_ratio'] > 0.3).astype(int)
test['is_high_credit'] = (test['credit_score'] > 700).astype(int)


CATS = ['gender', 'marital_status', 'education_level', 
        'employment_status', 'loan_purpose', 'grade_subgrade']


TARGET = 'loan_paid_back'


BASE = [col for col in train.columns if col not in ['id', TARGET]]
ORIG = []

for col in BASE:
    # Check if column exists in orig to avoid errors
    if col in orig.columns:
        # MEAN encoding from Original Data
        mean_map = orig.groupby(col)[TARGET].mean()
        new_mean_col_name = f"orig_mean_{col}"
        mean_map.name = new_mean_col_name
        
        train = train.merge(mean_map, on=col, how='left')
        test = test.merge(mean_map, on=col, how='left')
        ORIG.append(new_mean_col_name)

        # COUNT encoding from Original Data
        new_count_col_name = f"orig_count_{col}"
        count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
        
        train = train.merge(count_map, on=col, how='left')
        test = test.merge(count_map, on=col, how='left')
        ORIG.append(new_count_col_name)

print(f'{len(ORIG)} Orig Features Created!!')
FEATURES = BASE + ORIG
print(f'{len(FEATURES)} Total Features.')


train.head()


train = train.drop(columns=["id"])
test = test.drop(columns=["id"])


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


for col in CATS:
    # Only convert if present in features
    if col in X.columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')


def objective(trial):
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": False,
        "class_weight": {0: 1, 1: scale_pos_weight},
        "n_estimators": trial.suggest_int("n_estimators", 300, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "verbose": -1
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc"
        )

        preds = model.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(y_va, preds))

    return np.mean(auc_scores)


#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=30, timeout=3600)


#best_params = study.best_params
#best_auc = study.best_value

#print(best_params)
#print(best_auc)


N_SPLITS = 10


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# LightGBM Parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 31,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu'  
}

# ==========================================
# 5. Training Loop
# ==========================================
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

print("\nStarting LightGBM Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**lgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=1000)
        ]
    )

    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'--- Fold {fold}/{N_SPLITS} AUC: {fold_score:.5f} ---')
    
    # Test predictions
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

# ==========================================
# 6. Evaluation & Visualization
# ==========================================
overall_auc = roc_auc_score(y, oof_preds)
print(f'\n==================================')
print(f'Overall OOF AUC: {overall_auc:.5f}')
print(f'==================================')

plt.figure(figsize=(12, 5))

# Plot 1: ROC Curve
plt.subplot(1, 2, 1)
fpr, tpr, thresholds = roc_curve(y, oof_preds)
plt.plot(fpr, tpr, color='blue', label=f'Overall AUC = {overall_auc:.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (LGBM OOF)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Plot 2: Distribution
plt.subplot(1, 2, 2)
sns.histplot(test_preds, bins=50, kde=True, color='purple')
plt.title('Distribution of Test Predictions (LGBM)')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


feature_importance_df = pd.DataFrame()
feature_importance_df['feature'] = X.columns
feature_importance_df['importance'] = 0

feature_imp = pd.DataFrame({
    'Value': model.feature_importances_,
    'Feature': X.columns
})

feature_imp = feature_imp.sort_values(by="Value", ascending=False)

print("\nTop 20 Features:")
print(feature_imp.head(20))


plt.figure(figsize=(8, 8))
sns.barplot(x="Value", y="Feature", data=feature_imp.head(20), palette="viridis")
plt.title('LightGBM Feature Importance (Top 20 - Last Fold)')
plt.xlabel('Importance Score (Split)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


submission = pd.DataFrame({
    'id': submission.id,  
    'prediction': test_preds
})
submission.to_csv('submission.csv', index=False)
submission.head()


plt.figure(figsize=(5, 4))
plt.hist(submission['prediction'], bins=30)
plt.tight_layout()
plt.show()


submission.describe().round(3)

