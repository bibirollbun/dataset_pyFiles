# Data analysis, processing, and manipulation
import numpy as np   # Math
import pandas as pd  # Data processing; CSV I/O

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Tests
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score

# Hyperparameter Tuning
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict

# Pipeline
from sklearn.pipeline import make_pipeline

# Caching
import os
from joblib import Memory

# Warnings
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
validation_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

df.info()
print("")


print(f"Missing data summary (Out of {len(df)} rows)")
pd.DataFrame({
  'feature': df.columns,
  'null_count': df.isna().sum().values,
  'null_percentage': (df.isna().mean().values * 100).round(2)
}).sort_values('null_percentage', ascending=False)


null_df = df.drop(['id', 'Personality'], axis=1)
for col in null_df.columns:
    # Create a null indicator
    null_df[f"null_{col}"] = df[col].isna().astype(int)

null_corr = null_df[[col for col in null_df.columns if col.startswith("null_")]].corr()
sns.heatmap(null_corr, annot=True, cmap="coolwarm")
plt.show()


mcar_df = pd.DataFrame(columns=['missing', 'feature', 'p-value', 'is_significant'])

for missing_col in null_df.columns:
    if not missing_col.startswith('null'): continue
    for feature_col in null_df.columns:
        if feature_col.startswith('null') or feature_col == missing_col.replace("null_", ""): continue
        non_missing_df = null_df.copy().dropna(subset=feature_col)
        feature_col_name = feature_col
        if non_missing_df[feature_col].dtypes != 'object':
            try:
                non_missing_df[f"{feature_col}_bin"] = pd.qcut(non_missing_df[feature_col],
                                                      q=4,
                                                      labels=['low', 'low-medium', 'medium', 'high'],
                                                      duplicates='drop'
                                                     )
                feature_col_name = f"{feature_col}_bin"
            except: continue
        contingency_table = pd.crosstab(non_missing_df[missing_col], non_missing_df[feature_col_name])
        stat, p, dof, exp = chi2_contingency(contingency_table)
        mcar_df = pd.concat([mcar_df, pd.DataFrame({
            'missing': [missing_col],
            'feature': [feature_col],
            'p-value': [p],
            'is_significant': [p < 0.05]
        })], ignore_index=True)

mcar_len = len(mcar_df)
mcar_significant_count = mcar_df['is_significant'].sum()
mcar_significant_percent = round(100 * (mcar_significant_count / mcar_len), 2)
print(f"MCAR Test\t{mcar_significant_count}/{mcar_len} ({mcar_significant_percent}%) tests are significant")
mcar_df.head()



mar_df = pd.DataFrame(columns=['missing', 'feature', 'coefficient', 'p-value', 'is_significant'])

for feature_col in null_df.columns:
    if feature_col.startswith("null_"): continue
    for missing_col in null_df.columns:
        if not missing_col.startswith("null_") or missing_col.replace("null_", "") == feature_col: continue
        non_missing_df = null_df.copy().dropna(subset=feature_col)
        if non_missing_df[feature_col].dtypes == 'object':
            non_missing_df[feature_col] = non_missing_df[feature_col].apply(lambda x: int(x == 'Yes'))
        res = smf.logit(f"{missing_col} ~ {feature_col}", data=non_missing_df).fit(disp=0)
        mar_df = pd.concat([mar_df, pd.DataFrame({
            'missing': [missing_col],
            'feature': [feature_col],
            'coefficient': [res.params[feature_col]],
            'p-value': [res.pvalues[feature_col]],
            'is_significant': [res.pvalues[feature_col] < 0.05]
        })], ignore_index=True)

mar_len = len(mar_df)
mar_significant_count = mar_df['is_significant'].sum()
mar_significant_percent = round(100 * (mar_significant_count / mar_len), 2)
print(f"MAR Test\t{mar_significant_count}/{mar_len} ({mar_significant_percent}%) tests are significant")
mar_df.head()


# Data preparation for subsequent plotting
dropped_df = df.drop(['id', 'Personality'], axis=1).dropna()
num_df = dropped_df[[col for col in dropped_df.columns if dropped_df[col].dtypes != 'object']]
cat_df = dropped_df[[col for col in dropped_df.columns if dropped_df[col].dtypes == 'object']]


# Visual analysis
plt.figure(figsize=(10,4))
plt.title("Numerical feature distribution")
plt.boxplot(num_df, labels=num_df.columns)
plt.tight_layout()
plt.show()

# Descriptive Analysis
num_df.describe().drop('count').map(lambda x: round(x, 2))


melted_cat_df = pd.DataFrame(cat_df).melt(value_vars=cat_df.columns)
plt.title("Categorical feature distribution")
sns.countplot(melted_cat_df, x='variable', hue='value')
plt.tight_layout()
plt.show()

descriptive_cat_df = pd.DataFrame(
  columns=[*[f"{col}-{val}" for col in cat_df.columns for val in ['No', 'Yes']]],
  index=['count', 'proportion']
)

descriptive_cat_df.columns = pd.MultiIndex.from_tuples([tuple(c.split("-")) for c in descriptive_cat_df.columns])

for statistic in ['count', 'proportion']:
  for col in cat_df.columns:
    for val in ['Yes', 'No']:
      count = cat_df[col].value_counts()[val]
      descriptive_cat_df.loc[statistic, (col, val)] = count if statistic == 'count' else round(100 * count/len(cat_df[col]), 2)
descriptive_cat_df.loc['count', ('Stage_fear', 'Yes')]
descriptive_cat_df


imbalance_df = df.value_counts('Personality').to_frame().reset_index()
imbalance_df['proportion'] = imbalance_df['count'].apply(lambda x: f"{100 * (x / len(df)):.2f}%")
imbalance_df


corr_df = df.copy().drop(['id', 'Personality'], axis=1)
for col in [cat_col for cat_col in corr_df.columns if corr_df[cat_col].dtypes == 'object']:
  corr_df[col] = corr_df[col].apply(lambda x: 1 if x == 'Yes' else 0)
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm")
plt.show()


clean_df = df.copy()
for col in [cat_col for cat_col in clean_df.columns if clean_df[cat_col].dtypes == 'object']:
  clean_df[col] = clean_df[col].apply(lambda x: 1 if x in ['Yes', 'Introvert'] else 0)

clean_df = clean_df.rename(columns={'Personality': 'Is_introvert'})
clean_df.head()


X = clean_df.drop(['id', 'Is_introvert'], axis=1)
y = clean_df['Is_introvert']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9, stratify=y)


imputer = IterativeImputer(random_state=9)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


scaler = RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


kf = StratifiedKFold(shuffle=True, random_state=9)
estimator = RandomForestClassifier(random_state=9)
selector = RFECV(estimator, scoring='f1', cv=kf)
selector.fit(X_train_scaled, y_train)
pd.DataFrame({
  'feature': [feature for feature in X.columns],
  'rank': selector.ranking_,
  'selected': selector.support_
})


pca = PCA(random_state=9).fit(X_train_scaled)
evr = pca.explained_variance_ratio_
cumvar = evr.cumsum()

plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o')
plt.xlabel("Number of components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

pca = PCA(n_components=4, random_state=9).fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


baseline_model = LogisticRegression(random_state=9)
baseline_cv_og = cross_val_score(baseline_model, X_train_scaled, y_train, cv=kf, scoring='f1')
baseline_cv_pca = cross_val_score(baseline_model, X_train_pca, y_train, cv=kf, scoring='f1')
baseline_f1_og = baseline_cv_og.mean()
baseline_f1_pca = baseline_cv_pca.mean()
print("Baseline model CV Results\n")
print("Original Features")
for i, result in enumerate(baseline_cv_og):
  print(f"\tFold {i + 1}: {result * 100:.2f}%")
print("PCA Components")
for i, result in enumerate(baseline_cv_pca):
  print(f"\tFold {i + 1}: {result * 100:.2f}%")
print(f"\nMean F1-Score (Original): {baseline_f1_og* 100:.4f}%")
print(f"Mean F1-Score (PCA): {baseline_f1_pca* 100:.4f}%")


def benchmark(model, compare_f1=None):
  model_name = model.__class__.__name__
  cv_og = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='f1')
  cv_pca = cross_val_score(model, X_train_pca, y_train, cv=kf, scoring='f1')
  f1_og = cv_og.mean()
  f1_pca = cv_pca.mean()
  diff_og = f"{'+' if f1_og > baseline_f1_og else '' }{100 * (f1_og - baseline_f1_og):.4f}%"
  diff_pca = f"{'+' if f1_pca > baseline_f1_pca else '' }{100 * (f1_pca - baseline_f1_pca):.4f}%"

  print(f"{model_name} CV Results\n")
  print("Original Features")
  for i, result in enumerate(cv_og):
    print(f"\tFold {i + 1}: {result * 100:.2f}%")
  print("PCA Components")
  for i, result in enumerate(cv_pca):
    print(f"\tFold {i + 1}: {result * 100:.2f}%")
  print(f"\nMean F1-Score (Original): {f1_og * 100:.4f}%")
  print(f"Mean F1-Score (PCA): {f1_pca * 100:.4f}%")
  print(f"\nBaseline Diff (Original): {diff_og}")
  print(f"Baseline Diff (PCA): {diff_pca}")

  if compare_f1:
    default_diff_og = f"{'+' if f1_og > compare_f1['og'] else ''}{100 * (f1_og - compare_f1['og']):.4f}%"
    default_diff_pca = f"{'+' if f1_pca > compare_f1['pca'] else ''}{100 * (f1_pca - compare_f1['pca']):.4f}%"
    print(f"\nDefault Diff (Original): {default_diff_og}")
    print(f"Default Diff (PCA): {default_diff_pca}\n")

  f1 = {'og': f1_og, 'pca': f1_pca}
  diff = {'og': diff_og, 'pca': diff_pca}
  return f1, diff


svc_f1, svc_diff = benchmark(SVC(random_state=9))


knn_f1, knn_diff = benchmark(KNeighborsClassifier())


rf_f1, rf_diff = benchmark(RandomForestClassifier(class_weight='balanced', random_state=9))


xgb_f1, xgb_diff = benchmark(XGBClassifier(random_state=9))


models = ['SVC', 'KNN', 'RF', 'XGB']
f1_scores = [svc_f1, knn_f1, rf_f1, xgb_f1]
diffs = [svc_diff, knn_diff, rf_diff, xgb_diff]

pd.DataFrame([
    {
        'model': model,
        'f1_og': f"{100 * f1['og']:.4f}%",
        'f1_pca': f"{100 * f1['pca']:.4f}%",
        'diff_og': diff['og'],
        'diff_pca': diff['pca'],
        'best_f1': f"{100 * (f1['og'] if f1['og'] > f1['pca'] else f1['pca']):.4f}%",
        'best_feature': 'original' if f1['og'] > f1['pca'] else 'pca'
    } for model, f1, diff in zip(models, f1_scores, diffs)
]).sort_values('best_f1', ascending=False)




svc_param_grid = {
  'kernel': ['rbf'],
  'C': [1, 1.25, 1.5],
  'gamma': [1, 1.25, 1.5],
}

svc_cv = GridSearchCV(SVC(probability=True, random_state=9),
                      svc_param_grid,
                      cv=kf,
                      n_jobs=-1,
                      refit=True,
                      scoring='f1')
svc_cv.fit(X_train_pca, y_train)
tuned_svc = svc_cv.best_estimator_
tuned_svc_f1, tuned_svc_diff = benchmark(tuned_svc, knn_f1)
print(svc_cv.best_params_)


knn_param_grid = {
  'n_neighbors': [3, 5, 7, 9],
  'metric': ['minkowski', 'euclidean'],
  'weights': ['uniform', 'distance']
}

knn_cv = GridSearchCV(KNeighborsClassifier(),
                      knn_param_grid,
                      cv=kf,
                      n_jobs=-1,
                      refit=True,
                      scoring='f1')
knn_cv.fit(X_train_pca, y_train)
tuned_knn = knn_cv.best_estimator_
tuned_knn_f1, tuned_knn_diff = benchmark(tuned_knn, knn_f1)
print(knn_cv.best_params_)


rf_param_grid = {
  'n_estimators': [175, 200],
  'max_depth': [6, 8],
  'min_samples_split': [2, 5],
  'min_samples_leaf': [1, 3],
}

rf_cv = GridSearchCV(RandomForestClassifier(random_state=9),
                     rf_param_grid,
                     cv=kf,
                     n_jobs=-1,
                     refit=True,
                     scoring='f1')
rf_cv.fit(X_train_pca, y_train)
tuned_rf = rf_cv.best_estimator_
tuned_rf_f1, tuned_rf_diff = benchmark(tuned_rf, rf_f1)
print(rf_cv.best_params_)


xgb_param_grid = {
  'n_estimators': [150, 185, 200],
  'learning_rate': [0.05, 0.1, 0.15],
  'max_depth': [3, 5],
  'min_child_weight': [1, 3],
  'subsample': [0.7, 0.5, 0.6],
  'colsample_bytree': [0.6, 0.5, 0.4],
}

xgb_cv = GridSearchCV(XGBClassifier(eval_metric='logloss',
                                    n_jobs=-1,
                                    random_state=9),
                      xgb_param_grid,
                      cv=kf,
                      n_jobs=-1,
                      refit=True,
                      scoring='f1')
xgb_cv.fit(X_train_pca, y_train)
tuned_xgb = xgb_cv.best_estimator_
tuned_xgb_f1, tuned_xgb_diff = benchmark(tuned_xgb, xgb_f1)
print(xgb_cv.best_params_)


svc_pred = cross_val_predict(tuned_svc, X_train_pca, y_train, cv=kf)
knn_pred = cross_val_predict(tuned_knn, X_train_pca, y_train, cv=kf)
rf_pred = cross_val_predict(tuned_rf, X_train_pca, y_train, cv=kf)
xgb_pred = cross_val_predict(tuned_xgb, X_train_pca, y_train, cv=kf)

errors_df = pd.DataFrame({
    'svc': svc_pred != y_train,
    'knn': knn_pred != y_train,
    'rf': rf_pred != y_train,
    'xgb': xgb_pred != y_train
})

sns.heatmap(errors_df.corr(), cmap='coolwarm', annot=True)
plt.show()
errors_df.sum(axis=1).value_counts()


meta_estimators = [('svc', tuned_svc),
                   ('knn', tuned_knn),
                   ('rf', tuned_rf),
                   ('xgb', tuned_xgb)]

meta = StackingClassifier(estimators=meta_estimators,
                          final_estimator=RidgeClassifier(random_state=9),
                          stack_method='predict_proba',
                          n_jobs=-1,
                          cv=kf)
meta_f1, meta_diff = benchmark(meta)
meta.fit(X_train_pca, y_train)


model_names = ['SVC', 'KNN', 'RF', 'XGB', 'Meta']
tuned_models = [tuned_svc, tuned_knn, tuned_rf, tuned_xgb, meta]

pd.DataFrame([
    {
        'model': model_name,
        'train_f1': f1_score(y_train, model.predict(X_train_pca)),
        'test_f1': f1_score(y_test, model.predict(X_test_pca))
    } for model_name, model in zip(model_names, tuned_models)
]).sort_values('test_f1', ascending=False)


pipeline = make_pipeline(imputer, scaler, pca, tuned_svc)
pipeline.fit(X, y)

X_validation = validation_df.drop('id', axis=1)
for col in [cat_col for cat_col in X_validation.columns if X_validation[cat_col].dtypes == 'object']:
  X_validation[col] = X_validation[col].apply(lambda x: 1 if x == 'Yes' else 0)

submission_df = validation_df[['id']].copy()
submission_df['Personality'] = pipeline.predict(X_validation)
submission_df['Personality'] = submission_df['Personality'].apply(lambda x: 'Introvert' if x == 1 else 'Extrovert')
submission_df.to_csv('submission.csv', index=False)

