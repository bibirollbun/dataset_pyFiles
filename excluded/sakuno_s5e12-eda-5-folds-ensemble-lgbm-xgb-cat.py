import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_train = df_train.drop("id", axis=1)


print("- The train set's shape is",df_train.shape[0], "rows and", df_train.shape[1], "columns.")
print("- The test set's shape is",df_test.shape[0], "rows and", df_test.shape[1], "columns.")


df_train.info()


pd.set_option('display.max_columns', None)
df_train.head()


print(f"- There are {df_train.isna().sum().sum()} missing values in train set.")
print(f"- There are {df_test.isna().sum().sum()} missing values in test set.")


print(f"- There are {df_train.duplicated().sum()} duplicates in train set.")
print(f"- There are {df_test.duplicated().sum()} duplicates in test set.")


numerical_features = ['age','physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides']


categorical_features = ['alcohol_consumption_per_week', 'gender', 'ethnicity', 'education_level', 'income_level', 
                        'smoking_status', 'employment_status', 'family_history_diabetes','hypertension_history', 'cardiovascular_history', ]


counts = df_train['diagnosed_diabetes'].value_counts().sort_index()
labels = ['No Diabetes\n(0)', 'Diagnosed Diabetes\n(1)']

plt.figure(figsize=(9, 9))
plt.pie(counts, 
        labels=labels,
        colors=['gray', 'red'],
        autopct=lambda pct: f'{pct:.2f}%\n({int(pct/100*len(df_train)):,} patients)',
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        explode=(0, 0.12))

plt.title('Distribution of Diagnosed Diabetes', 
          fontsize=20, fontweight='bold', pad=30, color='#2E2E2E')
plt.axis('equal')
plt.show()


df_train.describe()


for column in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(data=df_train, x=column, ax=axes[0], color='red')
    mean_value = df_train[column].mean()
    median_value = df_train[column].median()
    axes[0].axvline(mean_value, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    axes[0].axvline(median_value, color='yellow', linestyle='-.', linewidth=2, label=f'Median: {median_value:.2f}')
    axes[0].set_title(f'Histogram of {column}')
    axes[0].legend()

    sns.boxplot(data=df_train, x=column, ax=axes[1], color='red')
    axes[1].set_title(f'Boxplot of {column}')

    plt.tight_layout()
plt.show()


from matplotlib.colors import LinearSegmentedColormap
import numpy as np

n_categories = 9
colors = ['red', 'white']
cmap = LinearSegmentedColormap.from_list("red_bright", colors)
auto_red_gradient = [cmap(i) for i in np.linspace(0.3, 1, n_categories)]

sns.set_palette(auto_red_gradient)

for column in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_train, x=column, 
                  order=df_train[column].value_counts().index,
                  palette=auto_red_gradient[:df_train[column].nunique()])
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for column in categorical_features:
    order = df_train[column].value_counts().index
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_train, x=column, order=order, palette="Reds_r")
    plt.title(f'Distribution of {column}')
plt.show()


from IPython.display import display, Markdown

total_rows = len(df_train)

for col in numerical_features:
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_train[(df_train[col] < lower_bound) | (df_train[col] > upper_bound)]
    n_outliers = len(outliers)
    outlier_pct = 100 * n_outliers / total_rows
    
    display(Markdown(f"**{col}**:  "
                     f"Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}."))
    
    display(Markdown(f"Number of outliers â†’ **{n_outliers:,}** "
                     f"({outlier_pct:.2f}% of data).  \n"))


for column in numerical_features:
    plt.figure(figsize=(12, 6))
    
    sns.kdeplot(
        data=df_train,
        x=column,
        hue='diagnosed_diabetes',
        fill=True,
        palette=['red', 'gray'],
        alpha=0.5,
        linewidth=2,
        common_norm=False)
    
    plt.title(f'{column} Distribution by Diabetes Status (KDE)', 
              fontsize=16)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(title='Diabetes', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.show()


for column in categorical_features:
    order = df_train[column].value_counts().index
    sns.catplot(
        data=df_train,
        x=column,
        hue='diagnosed_diabetes',
        kind='count',
        order=order,
        palette=['red','gray'],
        height=5,
        aspect=2
    )
    plt.title(f'Distribution of {column} by Diabetes Status')
    plt.xticks(rotation=45)
    plt.show()


def create_features(df):
    df = df.copy()
    
    # Log transforms
    df['triglycerides'] = np.log1p(df['triglycerides'])
    cap = df['physical_activity_minutes_per_week'].quantile(0.99)
    df['physical_activity_minutes_per_week'] = np.log1p(df['physical_activity_minutes_per_week'].clip(upper=cap))
    
    # Clinical features
    df['bmi_cat'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,35,40,100],
                           labels=[0,1,2,3,4,5]).astype('category')
    df['age_group'] = pd.cut(df['age'], bins=[0,30,40,50,60,70,100],
                             labels=[0,1,2,3,4,5]).astype('category')
    df['hypertension'] = ((df['systolic_bp']>=140) | (df['diastolic_bp']>=90)).astype(int)
    df['age_x_bmi'] = df['age'] * df['bmi']
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # Convert ALL object columns to category (LGBM + CatBoost will love them)
    for col in categorical_features:
        df[col] = df[col].astype('category')
    
    return df

df_train = create_features(df_train)
df_test  = create_features(df_test)

X = df_train.drop(['diagnosed_diabetes'], axis=1)
y = df_train['diagnosed_diabetes']
X_test = df_test.drop('id', axis=1)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cat_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
aucs = []

# Get categorical column names for CatBoost
cat_features = X.select_dtypes(include='category').columns.tolist()
cat_feature_indices = [X.columns.get_loc(col) for col in cat_features]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # === LightGBM ===
    print("Training LightGBM")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=128,
        max_depth=-1,
        colsample_bytree=0.7,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=(y==0).sum() / (y==1).sum()
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    # === XGBoost ===
    print("Training XGBoost")
    X_train_xgb = X_train.copy()
    X_val_xgb   = X_val.copy()
    X_test_xgb  = X_test.copy()
    
    for col in cat_features:
        X_train_xgb[col] = X_train_xgb[col].cat.codes
        X_val_xgb[col]   = X_val_xgb[col].cat.codes
        X_test_xgb[col]  = X_test_xgb[col].cat.codes

    xgb_model = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=8,
        colsample_bytree=0.7,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=(y==0).sum() / (y==1).sum(),
        verbosity=0
    )
    xgb_model.fit(
        X_train_xgb, y_train,
        eval_set=[(X_val_xgb, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )

    # === CatBoost ===
    print("Training CatBoost")
    cat_model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=500,
        early_stopping_rounds=100,
        scale_pos_weight=(y==0).sum() / (y==1).sum(),
        cat_features=cat_feature_indices
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    # === OOF & Test Predictions ===
    lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    xgb_oof[val_idx] = xgb_model.predict_proba(X_val_xgb)[:, 1]
    cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]

    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / n_splits
    xgb_preds += xgb_model.predict_proba(X_test_xgb)[:, 1] / n_splits
    cat_preds += cat_model.predict_proba(X_test)[:, 1] / n_splits

    # Fold AUC  
    fold_auc = roc_auc_score(y_val, (lgb_oof[val_idx] + xgb_oof[val_idx] + cat_oof[val_idx]) / 3)
    aucs.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")

print(f"\n{'='*50}")
print(f"Mean CV AUC: {np.mean(aucs):.6f} Â± {np.std(aucs):.6f}")
print(f"Final OOF AUC: {roc_auc_score(y, (lgb_oof + xgb_oof + cat_oof)/3):.6f}")
print(f"{'='*50}")


final_pred = (lgb_preds + xgb_preds + cat_preds) / 3


submission = pd.DataFrame({
    'id': df_test.id,  # fixed: df_test['id'], not df_test.id
    'diagnosed_diabetes': final_pred
})
submission.to_csv('submission_lgb_xgb_cat_ensemble.csv', index=False)
print("Submission saved: submission_lgb_xgb_cat_ensemble.csv")

