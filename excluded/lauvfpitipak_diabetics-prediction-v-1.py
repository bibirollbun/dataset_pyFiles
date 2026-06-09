import pandas as pd
import numpy as np
import time 
import math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.subplots as sp
from scipy.stats import skew, kurtosis, zscore
from scipy.stats import chi2_contingency
import plotly.figure_factory as ff  
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier 

warnings.filterwarnings('ignore')
sns.set(style='darkgrid')
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="sqlalchemy")
warnings.filterwarnings("ignore", category=DeprecationWarning)


print("Loading Dataset....")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)

train.head(3)


TARGET = 'diagnosed_diabetes'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = train.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]
print(f'{len(BASE)} Base Features:{BASE}')


train.isnull().sum()


train.info()


train.describe().round(2).T


print("Duplicated Rows:",train.duplicated().sum())
print("-"*30)
print("Number of Rows:",train.shape[0])
print("-"*30)
print("Number of Columns:",train.shape[1])


print("Numeric Col Names",train.select_dtypes(include=['number']).columns)
print("-"*30)
print("Categorical Col Names",train.select_dtypes(include=['object']).columns)


num_col = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides']

bool_col = ['family_history_diabetes', 'hypertension_history',
       'cardiovascular_history']

cat_col = ['gender', 'ethnicity', 'education_level', 'income_level',
       'smoking_status', 'employment_status']

target_col = 'diagnosed_diabetes'


color_palette = [
    "#371EA3","#4E3DA5","#655CAC",
    "#7B7CB7","#929BC6","#A8BBD8",
    "#BFD7E7","#D5EAF1","#E9F7F8"
]


print("\n===== Grouped by Diabetes Diagnosis =====")
print(train.groupby(target_col)[['age', 'bmi', 'cholesterol_total']].mean().round(3))


print("\n===== Skewness & Kurtosis ====")
for col in num_col:
    print(f"{col:25s} | Skewness: {train[col].skew():.4f} | Kurtosis: {train[col].kurtosis():.4f}")


for col in cat_col:
    # Crosstab
    ct = pd.crosstab(train[col], train['diagnosed_diabetes'])
    ct_perc = (ct.T / ct.sum(axis=1)).T * 100
    print(f"\n===== Crosstab: {col} vs Diagnosed Diabetics =====")
    print(ct_perc.round(2))

    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi-Square Test for {col}: χ²={chi2:.2f}, p-value={p:.4f}")


class_dist = train['diagnosed_diabetes'].value_counts(normalize=True)
print("\n===== Diagnosed Diabetes Distribution =====")
print(class_dist.round(3))

imbalance_ratio = class_dist.min() / class_dist.max()
print(f"\nClass Imbalance Ratio (minority/majority): {imbalance_ratio:.3f}")


diab_count = train['diagnosed_diabetes'].value_counts().reset_index()
diab_count.columns = ['diagnosed_diabetes', 'Count']

fig = px.pie(
    diab_count,
    names='diagnosed_diabetes',
    values='Count',
    color='diagnosed_diabetes',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Diagnosed Diabetes Distribution"
)

fig.update_layout(width=600, height=400)

fig.show()


gender_count = train['gender'].value_counts().reset_index()
gender_count.columns = ['gender', 'Count']

fig = px.bar(
    gender_count,
    x='gender',               
    y='Count',                
    color='gender',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Gender Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


ethnicity_count = train['ethnicity'].value_counts().reset_index()
ethnicity_count.columns = ['ethnicity', 'Count']

fig = px.bar(
    ethnicity_count,
    x='ethnicity',
    y='Count',
    color='ethnicity',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Ethnicity Distribution",
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
    color_discrete_sequence=px.colors.sequential.Agsunset,
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
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Employment Status Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


smoking_count = train['smoking_status'].value_counts().reset_index()
smoking_count.columns = ['smoking_status', 'Count']

fig = px.bar(
    smoking_count,
    x='smoking_status',
    y='Count',
    color='smoking_status',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Smoking Status Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


income_count = train['income_level'].value_counts().reset_index()
income_count.columns = ['income_level', 'Count']

fig = px.bar(
    income_count,
    x='income_level',
    y='Count',
    color='income_level',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Income Level Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


plt.figure(figsize=(15, 50))
for i, col in enumerate(num_col):
    plt.subplot(len(num_col) // 2 + 1, 2, i + 1)
    sns.boxplot(
        data=train,
        x=target_col,
        y=col
    )
    plt.title(f"{col} vs {target_col}")
    plt.xlabel(target_col)
    plt.ylabel(col)

plt.tight_layout()
plt.show()


n_features = len(num_col)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.ravel()

for idx, feature in enumerate(num_col):
    sns.histplot(
        data=train,
        x=feature,
        hue=target_col,
        kde=True,
        stat="density",
        common_norm=False,
        palette=color_palette[:2],
        ax=axes[idx]
    )
    axes[idx].set_title(feature)

for i in range(n_features, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Train Dataset
sns.countplot(data=train, x=TARGET, ax=ax[0], palette='viridis')
ax[0].set_title(f'Train: {TARGET} Distribution')
for p in ax[0].patches:
    ax[0].annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Original Dataset (if target exists)
if TARGET in orig.columns:
    sns.countplot(data=orig, x=TARGET, ax=ax[1], palette='viridis')
    ax[1].set_title(f'Original: {TARGET} Distribution')
    for p in ax[1].patches:
        ax[1].annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()


df_plot = pd.concat([
    train[NUMS].assign(Source='Train'),
    test[NUMS].assign(Source='Test'),
    orig[NUMS].assign(Source='Original')
])

n_cols = 3
n_rows = (len(NUMS) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(NUMS):
    sns.kdeplot(data=df_plot, x=col, hue='Source', ax=axes[i], 
                fill=True, common_norm=False, warn_singular=False)
    axes[i].set_title(col)

for i in range(len(NUMS), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

del df_plot


corr_mat = train[num_col].corr()

plt.figure(figsize=(14,12))
sns.heatmap(
    corr_mat,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)
plt.title("Correlation Matrix (Numerical Variables)")
plt.tight_layout()
plt.show()


X = train.drop(columns=[target_col, 'id'])
y = train[target_col]
X_test = test.drop(columns=['id'])


cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

for col in cat_cols:
    X[col] = X[col].cat.codes
    X_test[col] = X_test[col].cat.codes

cat_idx = [X.columns.get_loc(col) for col in cat_cols]


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


results = {}


cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_state=42,
    task_type="GPU",
    devices="0",
    verbose=100
)

cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_cols,
    early_stopping_rounds=50
)

results["CatBoost"] = roc_auc_score(
    y_valid, cat_model.predict_proba(X_valid)[:, 1]
)
test_pred_cat = cat_model.predict_proba(X_test)[:, 1]


lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    device="gpu",
    random_state=42,
    verbose=-1
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    categorical_feature=cat_cols
)

results["LightGBM"] = roc_auc_score(
    y_valid, lgb_model.predict_proba(X_valid)[:, 1]
)

test_pred_lgb = lgb_model.predict_proba(X_test)[:, 1]


xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="auc",
    early_stopping_rounds=50,
    enable_categorical=True,
    tree_method="hist",
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=100
)

results["XGBoost"] = roc_auc_score(
    y_valid, xgb_model.predict_proba(X_valid)[:, 1]
)

test_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]


hgb_model = HistGradientBoostingClassifier(
    max_iter=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    categorical_features=cat_idx,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50
)

hgb_model.fit(X_train, y_train)

results["HistGB"] = roc_auc_score(
    y_valid, hgb_model.predict_proba(X_valid)[:, 1]
)

test_pred_hgb = hgb_model.predict_proba(X_test)[:, 1]


final_preds = (
    test_pred_cat +
    test_pred_lgb +
    test_pred_xgb +
    test_pred_hgb
) / 4

plt.figure()
plt.hist(final_preds, bins=50)
plt.title("Ensemble Test Prediction Histogram")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.show()

print("Ensemble mean prediction:", final_preds.mean())
print(results)


submission = pd.DataFrame({
    "id": test["id"],
    "target": final_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()

