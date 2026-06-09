import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
test  = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')


### PLOTS
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 7)


print(train.shape, test.shape)


display(train.head())


display(train.info())


target_col = 'target'


print(train[target_col].value_counts())


print(train[target_col].value_counts(normalize=True))


sns.countplot(x=target_col, data=train, palette='Set2')
plt.title('Target Distribution')
plt.show()


def count_neg_values(df):
    return (df == -1).sum().sort_values(ascending=False)


missing_train = count_neg_values(train)
missing_test = count_neg_values(test)


print("Top-10 columns with -1 (train):")
display(missing_train.head(10))


print("Top-10 columns with -1 (test):")
display(missing_test.head(10))


plt.figure(figsize=(10, 6))
ax = missing_train.head(15).plot(kind='bar', color='salmon')
plt.title('Missing Values (-1) per Feature (Train Dataset)')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


X = train.drop(columns=["id", "target"])
y = train["target"]
X_test = test.drop(columns=["id"])


### Some cat_cols are numerics
cat_cols = [c for c in X.columns if '_cat' in c]
for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)
num_cols = [c for c in X.select_dtypes(include=np.number).columns if c not in cat_cols]
bin_cols = [c for c in X.columns if '_bin' in c]


print(f'Numerical: {len(num_cols)} | Binary: {len(bin_cols)} | Categorical: {len(cat_cols)}')


columns_df = pd.DataFrame({
    'Category': ['Numerical', 'Binary', 'Categorical'],
    'Count': [len(num_cols), len(bin_cols), len(cat_cols)],
    'Columns': [num_cols, bin_cols, cat_cols]
})


display(columns_df)


sns.catplot(
    data=train[num_cols[:4] + ['target']].melt(id_vars='target'),
    x='target', y='value', col='variable',
    kind='boxen', col_wrap=2, sharey=False, height=4
)
plt.suptitle('Boxen plots of numeric features vs target', y=1.02)
plt.show()


corr_df = train[num_cols + bin_cols + [target_col]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.abs(), cmap='YlGnBu', square=True)
plt.title('Absolute Correlations (numeric & binary)')
plt.show()


### REMOVE -1 for use NAN for better understanding THEN use imputer in pipelines


for col in num_cols:
    X[col] = X[col].replace(-1, np.nan)
    X_test[col] = X_test[col].replace(-1, np.nan)


for col in cat_cols:
    X[col] = X[col].replace('-1', "missing")
    X_test[col] = X_test[col].replace('-1', "missing")


# -----------------
#### IMPORT SKLEARN
# -----------------
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
### MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------
# Normalized GINI for calculate previous to send
# ----------------------------------------------
def gini_normalized(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return 2 * auc - 1


numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)


model_logistic_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('logreg', LogisticRegression(
            penalty='l2',
            C=0.1,
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ))
    ])


# model_logistic_pipeline.fit(X, y)


# train_preds = model_logistic_pipeline.predict_proba(X)[:,1]


# gini_train_approx = gini_normalized(y, train_preds)
# print("Gini aproximado sobre todo el train set:", gini_train_approx)


# y_pred = (train_preds >= 0.5).astype(int)
# roc_auc = roc_auc_score(y, train_preds)
# precision = precision_score(y, y_pred)
# recall = recall_score(y, y_pred)
# f1 = f1_score(y, y_pred)
# print(f"ROC-AUC: {roc_auc:.5f}")
# print(f"Precision: {precision:.5f}")
# print(f"Recall: {recall:.5f}")
# print(f"F1-Score: {f1:.5f}")


# test_preds = model_logistic_pipeline.predict_proba(X_test)[:,1]


# submission = pd.DataFrame({"id": test["id"], "target": test_preds})
# submission.count()


# submission.to_csv("submission_log.csv", index=False)
# print("Archivo submission_log.csv generado.")


rf_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

rf_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

rf_preprocessor = ColumnTransformer(
    transformers=[
        ('num', rf_numeric_transformer, num_cols),
        ('cat', rf_categorical_transformer, cat_cols)
    ],
    remainder='drop'
)


model_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)


pipeline_rf = Pipeline(steps=[
    ('preprocessor', rf_preprocessor),
    ('model', model_rf)
])


# pipeline_rf.fit(X, y)


# y_prob_rf = pipeline_rf.predict_proba(X)[:, 1]
# y_pred_rf = (y_prob_rf >= 0.5).astype(int)


# gini_rf_train_approx = gini_normalized(y, y_prob_rf)
# print("Random Forest Gini aproximado sobre todo el train set:", gini_rf_train_approx)


# test_preds_rf = pipeline_rf.predict_proba(X_test)[:,1]


# submission_rf = pd.DataFrame({"id": test["id"], "target": test_preds_rf})
# submission_rf.count()


# submission_rf.to_csv("submission_rf_solved.csv", index=False)
# print("Archivo submission_rf_solved.csv generado.")


from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
from sklearn.metrics import make_scorer


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"Numerical features: {numerical_features[:5]}...")
print(f"Categorical features: {categorical_features[:5]}...")


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)


model_lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)


model_lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model_lgbm)
])


gini_scorer = make_scorer(gini_normalized, greater_is_better=True)


cv_scores = cross_val_score(model_lgbm_pipeline, X, y, cv=skf, scoring=gini_scorer)


cv_scores


print(f"Mean GINI: {cv_scores.mean():.4f}")
print(f"Std GINI:  {cv_scores.std():.4f}")
print(f"Min GINI:  {cv_scores.min():.4f}")
print(f"Max GINI:  {cv_scores.max():.4f}")
print(f"Range:     {cv_scores.max() - cv_scores.min():.4f}")


# print(f"Light GBM CV: {cv_scores.mean():.4f}")
# print(f"LogisticRegression: {gini_train_approx:.4f}")
# print(f"Diff: {cv_scores.mean() - gini_train_approx:.4f}")


for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{skf.get_n_splits()}")
X_train, X_val = X.iloc[train_index], X.iloc[val_index]
y_train, y_val = y.iloc[train_index], y.iloc[val_index]


model_lgbm_pipeline.fit(X_train, y_train)


y_prob_lgbm = model_lgbm_pipeline.predict_proba(X_val)[:, 1]
y_pred_lgbm = (y_prob_lgbm >= 0.5).astype(int)


roc_auc = roc_auc_score(y_val, y_prob_lgbm)
gini = gini_normalized(y_val, y_prob_lgbm)
precision = precision_score(y_val, y_pred_lgbm)
recall = recall_score(y_val, y_pred_lgbm)
f1 = f1_score(y_val, y_pred_lgbm)


print(f"  ROC-AUC: {roc_auc:.5f}")
print(f"  Gini: {gini:.5f}")
print(f"  Precision: {precision:.5f}")
print(f"  Recall: {recall:.5f}")
print(f"  F1-Score: {f1:.5f}")


test_preds_lgbm = model_lgbm_pipeline.predict_proba(X_test)[:, 1]


submission_lgbm = pd.DataFrame({"id": test["id"], "target": test_preds_lgbm})
submission_lgbm.to_csv("submission_lgbm.csv", index=False)
print("Archivo submission_lgbm.csv generado.")


submission_lgbm.count()


from sklearn.model_selection import RandomizedSearchCV


param_grid = {
    'classifier__n_estimators': [300, 500, 800],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [-1, 5, 10],
    'classifier__min_child_samples': [10, 20, 50],
    'classifier__num_leaves': [31, 63, 127]
}


random_search = RandomizedSearchCV(
    estimator=model_lgbm_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring=gini_scorer,
    cv=skf,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X, y)


best_params = grid_search.best_params_


print("Best parameters:", best_params)
print("Best CV score (Gini):", random_search.best_score_)


final_model = model_lgbm_pipeline.set_params(**best_params)
final_model.fit(X, y)


test_preds_lgbm_final = final_model.predict_proba(X_test)[:, 1]


submission_lgbm = pd.DataFrame({"id": test["id"], "target": test_preds_lgbm_final})
submission_lgbm.to_csv("submission.csv", index=False)
print("Archivo submission_lgbm.csv generado con hiperparámetros optimizados.")

