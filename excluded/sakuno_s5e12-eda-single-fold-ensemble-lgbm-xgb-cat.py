import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import xgboost as xgb
import catboost as cb

from sklearn.metrics import roc_curve, auc, roc_auc_score


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
        textprops={'fontsize': 8, 'fontweight': 'bold'},
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
    
    df['triglycerides'] = np.log1p(df['triglycerides'])
    cap = df['physical_activity_minutes_per_week'].quantile(0.99)
    df['physical_activity_minutes_per_week'] = np.log1p(df['physical_activity_minutes_per_week'].clip(upper=cap))
    
    df['bmi_cat'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,35,40,100],
                           labels=[0,1,2,3,4,5]).astype('category')
    df['age_group'] = pd.cut(df['age'], bins=[0,30,40,50,60,70,100],
                             labels=[0,1,2,3,4,5]).astype('category')
    df['hypertension'] = ((df['systolic_bp']>=140) | (df['diastolic_bp']>=90)).astype(int)
    df['age_x_bmi'] = df['age'] * df['bmi']
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    for col in categorical_features:
        df[col] = df[col].astype('category')
    return df

df_train = create_features(df_train)
df_test  = create_features(df_test)


X = df_train.drop(['diagnosed_diabetes'], axis=1)
y = df_train['diagnosed_diabetes']
X_test = df_test.drop('id', axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)


lgb_model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=90,
                           colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1,
                           random_state=42, n_jobs=-1, verbose=-1)


lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])


lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]


fpr, tpr, thresholds = roc_curve(y_val, lgb_val_pred)

auc_score = auc(fpr, tpr)
print("Validation AUC score is:", auc_score)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.plot([0, 0, 1], [0, 1, 1], color='lightgreen', linestyle='--', linewidth=2, label='Perfect model')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LGBM model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


xgb_model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=7,
                              min_child_weight=10, subsample=0.85, colsample_bytree=0.7,
                              reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                              random_state=42, tree_method='hist', enable_categorical=True, verbosity=0)


xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]


fpr, tpr, thresholds = roc_curve(y_val, xgb_val_pred)

auc_score = auc(fpr, tpr)
print("Validation AUC score is:", auc_score)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.plot([0, 0, 1], [0, 1, 1], color='lightgreen', linestyle='--', linewidth=2, label='Perfect model')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


cats = X_train.select_dtypes(include='category').columns.tolist()
cat_model = cb.CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    cat_features=cats,
    random_seed=42,
    verbose=0)


cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)


cat_val_pred = cat_model.predict_proba(X_val)[:, 1]


fpr, tpr, thresholds = roc_curve(y_val, cat_val_pred)

auc_score = auc(fpr, tpr)
print("Validation AUC score is:", auc_score)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.plot([0, 0, 1], [0, 1, 1], color='lightgreen', linestyle='--', linewidth=2, label='Perfect model')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


best_auc = 0
best_w1, best_w2 = 0, 0

for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        blend = w1 * xgb_val_pred + w2 * lgb_val_pred + (1 - w1 - w2) * cat_val_pred
        auc_i = roc_auc_score(y_val, blend)
        if auc_i > best_auc:
            best_auc = auc_i
            best_w1, best_w2 = w1, w2


print(f"Best: XGB {best_w1:.2f} + LGBM {best_w2:.2f} + Cat {1-best_w1-best_w2:.2f} â†’ {best_auc:.6f}")


lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]
xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]


final_pred = best_w1 * xgb_test_pred + best_w2 * lgb_test_pred + (1 - best_w1 - best_w2) * cat_test_pred


submission = pd.DataFrame({
    'id': df_test.id,
    'diagnosed_diabetes': final_pred
})
submission.to_csv('submission.csv', index=False)

