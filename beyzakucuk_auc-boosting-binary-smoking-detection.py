import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot')
sns.set_context('notebook', font_scale=1.2)
sns.set_style('whitegrid')
sns.set_palette('husl')

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.display.float_format = '{:.3f}'.format


train = pd.read_csv("/kaggle/input/binary-smoke-detector/train.csv")
test = pd.read_csv("/kaggle/input/binary-smoke-detector/test.csv")


df = train.copy()
df_test = test.copy()

print("ğŸ§¾ Train Data Info:")
print(df.info())

print("\nğŸ§¾ Test Data Info:")
print(df_test.info())


df.head()


df_test.head()


print("Train Data Shape:", df.shape)
print("Test Data Shape:", df_test.shape)
print("\nTrain Columns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDublicated Values:\n", df.duplicated().sum())
print("\nTarget Distribution:\n", df['smoking'].value_counts(normalize=True))


def plot_target_distribution(df):
    plt.figure(figsize=(10,6))
    sns.countplot(x='smoking', data=df)
    plt.title('Smoking Status Distribution')
    plt.xlabel('Smoking Status (0: Non-smoker, 1: Smoker)')
    plt.ylabel('Count')
    plt.show()

plot_target_distribution(df)


df = df.drop(columns="id", axis=1)
df.columns


df_test = df_test.drop(columns="id", axis=1)
df_test.columns


def plot_numerical_distributions(df):
    num_cols = df.select_dtypes(include='number').columns.drop('smoking', errors='ignore')
    if len(num_cols) > 10:
        corr_with_target = df[num_cols].corrwith(df['smoking']).abs().sort_values(ascending=False)
        num_cols = corr_with_target[:10].index.tolist()

    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue='smoking', bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution', fontsize=12)
        axes[i].set_xlabel('')

    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle('Numerical Features Distributions', y=1.05)
    plt.show()

plot_numerical_distributions(df)


def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', pad=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

plot_correlation_matrix(df)

correlation_matrix = df.corr()
smoking_correlation = correlation_matrix['smoking'].sort_values(ascending=False)
smoking_correlation_abs = smoking_correlation.abs().sort_values(ascending=False)

print("\nFeature Correlations with 'smoking' (Sorted by Absolute Value):\n")
print(smoking_correlation_abs)


def analyze_outliers(df, return_details=True):
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_stats = []

    for col in numeric_cols:
        if col in ['id', 'smoking']: 
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_percent = (len(outliers) / len(df)) * 100

        outlier_stats.append({
            'Variable': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outlier_Count': len(outliers),
            'Outlier_Percent': outlier_percent
        })

    outlier_report = pd.DataFrame(outlier_stats)
    outlier_report = outlier_report.sort_values('Outlier_Percent', ascending=False)

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        x='Outlier_Percent',
        y='Variable',
        data=outlier_report,
        palette='Set2',
        edgecolor='black'
    )

    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.5, 
                p.get_y() + p.get_height()/2., 
                f'{width:.1f}%', 
                ha='left', 
                va='center',
                fontsize=10)

    plt.title('Outlier Analysis by Variable (1.5 IQR Method)', pad=20, fontsize=14)
    plt.xlabel('Percentage of Outliers (%)', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    plt.axvline(x=5, color='red', linestyle='--', alpha=0.5)
    plt.text(5.2, len(outlier_report)-1, '5% Threshold', color='red')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    if return_details:
        return outlier_report.set_index('Variable')
    return None

print("=== TRAIN DATA OUTLIER ANALYSIS ===")
train_outlier_report = analyze_outliers(df, return_details=True)

print("\n=== TEST DATA OUTLIER ANALYSIS ===")
test_outlier_report = analyze_outliers(df_test, return_details=True)

print("\nTrain Data Outlier Summary:")
print(train_outlier_report[['Outlier_Count', 'Outlier_Percent']].sort_values('Outlier_Percent', ascending=False))

print("\nTest Data Outlier Summary:")
print(test_outlier_report[['Outlier_Count', 'Outlier_Percent']].sort_values('Outlier_Percent', ascending=False))


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns

    plt.figure(figsize=(14,8))
    sns.boxplot(data=df[numeric_cols].drop(columns='smoking', errors='ignore'), orient='h', palette='Set2')
    plt.title('Pre-Outlier Distributions')
    plt.show()

    for col in numeric_cols:
        if col == 'smoking': continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        df[col] = df[col].clip(lower, upper)

    plt.figure(figsize=(14,8))
    sns.boxplot(data=df[numeric_cols].drop(columns='smoking', errors='ignore'), orient='h', palette='Set2')
    plt.title('Post-Outlier Distributions')
    plt.show()

    return df

df = handle_outliers(df)
df_test = handle_outliers(df_test)


def enhanced_feature_engineering(df):
    df = df.copy()

    df['BMI'] = df['weight(kg)'] / ((df['height(cm)']/100)**2)
    df['Waist_Height_Ratio'] = df['waist(cm)'] / df['height(cm)']
    df['Body_Fat_Percentage'] = (1.20 * df['BMI']) + (0.23 * df['age']) - 5.4
    df['Non_HDL_Chol'] = df['Cholesterol'] - df['HDL']
    df['Chol/HDL'] = df['Cholesterol'] / df['HDL']
    df['ALT/AST'] = np.where(df['AST'] != 0, df['ALT'] / df['AST'], 0)
    df['log_triglyceride'] = np.log1p(df['triglyceride'])
    df['log_Gtp'] = np.log1p(df['Gtp'])
    df['log_ALT'] = np.log1p(df['ALT'])
    df['pulse_pressure'] = df['systolic'] - df['relaxation']
    df['hemoglobin_creatinine'] = df['hemoglobin'] * df['serum creatinine']

    df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=False)
    df['BMI_bin'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=False)
    df['Gtp_bin'] = pd.qcut(df['Gtp'], q=4, labels=False, duplicates='drop')

    df['age_hemoglobin'] = df['age'] * df['hemoglobin']
    df['weight_height'] = df['weight(kg)'] / df['height(cm)']
    df['AST_ALT_diff'] = df['AST'] - df['ALT']

    df['Urine_protein_high'] = (df['Urine protein'] > 1).astype(int)

    return df

df = enhanced_feature_engineering(df)
df_test = enhanced_feature_engineering(df_test)


print("\nFirst 5 rows of Training Data After Feature Engineering:")
print(df.head())
print("\nFirst 5 rows of Test Data After Feature Engineering:")
print(df_test.head())


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif

X = df.drop(columns=['smoking'])
y = df['smoking']

# ANOVA F-score
f_selector = SelectKBest(score_func=f_classif, k='all')
f_selector.fit(X, y)

anova_scores = pd.DataFrame({
    'Feature': X.columns,
    'F_Score': f_selector.scores_
}).sort_values(by='F_Score', ascending=False)

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

# Merge and display
feature_scores = pd.merge(anova_scores, mi_df, on='Feature')
feature_scores = feature_scores.sort_values(by='F_Score', ascending=False)

plt.figure(figsize=(16, 10))
sns.barplot(x='F_Score', y='Feature', data=feature_scores, palette='Set2')
plt.title('ANOVA F-score by Feature')
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 10))
sns.barplot(x='MI_Score', y='Feature', data=feature_scores.sort_values('MI_Score', ascending=False), palette='Set2')
plt.title('Mutual Information Score by Feature')
plt.tight_layout()
plt.show()

top_features = feature_scores.sort_values(by='MI_Score', ascending=False).head(20)['Feature'].tolist()


X = df.drop(['id', 'smoking'], axis=1, errors='ignore')
y = df['smoking']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

numerical_features = X_train.select_dtypes(include=np.number).columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

print("Train features shape      :", X_train.shape)
print("Train target shape        :", y_train.shape)
print("Validation features shape :", X_val.shape)
print("Validation target shape   :", y_val.shape)


log_reg_params = {
    'max_iter': 2000, 
    'solver': 'saga',  
    'penalty': 'elasticnet',  
    'l1_ratio': 0.5, 
    'class_weight': 'balanced',  
    'random_state': 101,
    'n_jobs': -1
}

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_features), 
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

models = [
    ("Logistic Regression", LogisticRegression(**log_reg_params)),
    ("KNN", KNeighborsClassifier(n_jobs=-1)),
    ("SVM", SVC(probability=True, random_state=101, class_weight='balanced')),
    ("Decision Tree", DecisionTreeClassifier(random_state=101, class_weight='balanced')),
    ("Random Forest", RandomForestClassifier(random_state=101, class_weight='balanced', n_jobs=-1)),
    ("AdaBoost", AdaBoostClassifier(random_state=101)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=101)),
    ("XGBoost", XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=101,
        n_jobs=-1
    )),
    ("CatBoost", CatBoostClassifier(verbose=0, random_state=101)),
    ("LightGBM", LGBMClassifier(random_state=101, n_jobs=-1))
]

results_list = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    for name, model in models:
        try:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            scores = cross_validate(
                pipeline, X_train, y_train,
                cv=cv,
                scoring="roc_auc",
                return_train_score=True,
                n_jobs=1 
            )

            results_list.append({
                "Model": name,
                "Train ROC AUC Mean": scores['train_score'].mean(),
                "Train ROC AUC Std": scores['train_score'].std(),
                "Test ROC AUC Mean": scores['test_score'].mean(),
                "Test ROC AUC Std": scores['test_score'].std(),
            })
            
        except Exception as e:
            print(f"{name} An error occurred in the model: {str(e)}")

results = pd.DataFrame(results_list)
results.sort_values(by='Test ROC AUC Mean', ascending=False, inplace=True)

print("\nModel Comparison (Cross-Validation ROC AUC Scores):")
print(results)


results


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'depth': trial.suggest_int("depth", 4, 10),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1, 10),
        'iterations': trial.suggest_int("iterations", 100, 1000),
        'random_state': 42,
        'verbose': 0,
        'eval_metric': 'AUC'
    }

    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_score = study.best_value
print("ğŸ”§ Best Parameters:", best_params)
print("ğŸ�¯ Best ROC AUC Score:", best_score)


cat = CatBoostClassifier(**best_params, random_state=42, verbose=0)
lgb = LGBMClassifier(random_state=42, n_jobs=-1)
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

stack_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("stacking", StackingClassifier(
        estimators=[
            ('cat', cat),
            ('lgb', lgb),
            ('logreg', log_reg)
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
        passthrough=True
    ))
])

stack_pipe.fit(X_train, y_train)


y_val_pred = stack_pipe.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"ğŸ“ˆ Stacking ROC AUC on Validation: {roc_auc:.5f}")


fpr, tpr, _ = roc_curve(y_val, y_val_pred)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Stacking (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Stacking')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


import joblib

joblib.dump(stack_pipe, 'stacking_final_model.pkl')
print("ğŸ“¦ Model saved.")


X_test_final = df_test.drop(columns=['id'], errors='ignore')
test_preds = stack_pipe.predict(X_test_final)

submission = pd.DataFrame({
    'id': test['id'],
    'smoking': test_preds
})
submission.to_csv("stacking_submission.csv", index=False)
print("âœ… Submission file saved: stacking_submission.csv")

