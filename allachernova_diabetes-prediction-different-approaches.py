import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

X_test = test_df.drop('id', axis=1)


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"numeric_features: {numeric_features}")
print(f"categorical_features: {categorical_features}")


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]


counts = train_df['diagnosed_diabetes'].value_counts(normalize=True) * 100

plt.figure(figsize=(6, 4))
ax = train_df['diagnosed_diabetes'].value_counts().plot(
    kind='bar',
    color=['#6baed6', '#fd8d3c']
)

plt.title('Target class distribution')
plt.xlabel('Diagnosed diabetes')
plt.ylabel('Number of samples')
plt.xticks([0, 1], ['No diabetes', 'Has diabetes'], rotation=0)

for i, v in enumerate(train_df['diagnosed_diabetes'].value_counts()):
    ax.text(i, v + 500, f"{counts.iloc[i]:.1f}%", ha='center')

plt.tight_layout()
plt.show()



train_encoded = train_df.copy()
for col in categorical_features:
    train_encoded[col] = pd.factorize(train_df[col])[0]

all_features = numeric_features + categorical_features + ['diagnosed_diabetes']
corr_matrix = train_encoded[all_features].corr()


target_corr = corr_matrix['diagnosed_diabetes'].sort_values(ascending=False)

print("Top 10 features by correlation with the target variable:")
print(target_corr.abs().sort_values(ascending=False).head(11))

plt.figure(figsize=(20, 16))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, 
            mask=mask,
            cmap='coolwarm', 
            center=0,
            square=True, 
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=False)

plt.title('Correlation map', fontsize=20, pad=20)
plt.tight_layout()
plt.show()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


X_train.describe()


model.fit(X_train, y_train)


val_preds = model.predict_proba(X_val)[:, 1]
val_score = roc_auc_score(y_val, val_preds)
print(f"\nROC-AUC on validation: {val_score:.4f}")


logreg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    ))
])

logreg.fit(X_train, y_train)
val_preds = logreg.predict_proba(X_val)[:, 1]
print("LogReg ROC-AUC:", roc_auc_score(y_val, val_preds))



cat_features = categorical_features


cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    task_type='GPU', 
    verbose=200,
    auto_class_weights='Balanced'
)

cat_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)

val_preds = cat_model.predict_proba(X_val)[:, 1]
print("CatBoost ROC-AUC:", roc_auc_score(y_val, val_preds))


final_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    auto_class_weights='Balanced',
    task_type='GPU',
    verbose=200
)

final_model.fit(
    X, y,                 
    cat_features=cat_features
)



test_preds = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})

submission.to_csv('cat_boost_full.csv', index=False)



cat_probe = CatBoostClassifier(
    iterations=6000,         
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    auto_class_weights='Balanced',
    task_type='GPU',
    verbose=100
)

cat_probe.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)



final_model = CatBoostClassifier(
    iterations=3300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    auto_class_weights='Balanced',
    task_type='GPU',
    verbose=200
)

final_model.fit(
    X, y,
    cat_features=cat_features
)



lgbm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        device='gpu'
    ))
])

lgbm.fit(X_train, y_train)
val_preds = lgbm.predict_proba(X_val)[:, 1]
print("LGBM ROC-AUC:", roc_auc_score(y_val, val_preds))



test_preds = lgbm.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})
submission.to_csv('lgbm_submission.csv', index=False)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    logreg, X, y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print("CV ROC-AUC:", scores.mean(), "Â±", scores.std())



explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val)



final_preds = (
    0.4 * logreg.predict_proba(X_val)[:, 1] +
    0.6 * cat_model.predict_proba(X_val)[:, 1]
)

roc_auc_score(y_val, final_preds)


