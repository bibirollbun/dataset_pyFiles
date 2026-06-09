!pip install ISLP


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from ISLP import confusion_table
from ISLP.models import (ModelSpec as MS, summarize, contrast)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)


# get directory for data 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')



print("Train shape:", train.shape)
print("=========================================")
print("Test  shape:", test.shape)
print("=========================================")
print("\nTrain info:")
train.info()


train. head(10)


# set variable types and check for missing values and 0s 
cat_vars = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

num_vars = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

target = ['NObeyesdad']

train[cat_vars] = train[cat_vars].astype('category')
test[cat_vars] = test[cat_vars].astype('category')
train[target] = train[target].astype('category')




# calulate BMI to add to varibles 

for df in (train, test):
    df['BMI'] = df['Weight'] / (df['Height']**2)

# Register BMI as numeric
if 'BMI' not in num_vars:
    num_vars.append('BMI')


# Numeric summary
print("\nNumeric summary (train):")
print(train[num_vars].describe())



# Missingness checks
print("\nMissing values per column (train):")
print(train.isna().sum())
print("\nMissing values per column (test):")
print(test.isna().sum())


# Target distribution
print("\nTarget distribution (counts):")
print(train[target].value_counts())
print("\nTarget distribution (proportions):")
print(train[target].value_counts(normalize=True))

print("====================================================")
# Bar chart of target distribution
plt.figure(figsize=(10,4))
(train[target].value_counts(normalize=True)
     .sort_index()
     .plot(kind='bar'))
plt.title("Class Proportions: NObeyesdad")
plt.ylabel("Proportion")
plt.xlabel("Class")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Correlations among numeric features
corr = train[num_vars].corr()
plt.figure(figsize=(7,6))
im = plt.imshow(corr, interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(num_vars)), num_vars, rotation=45, ha='right')
plt.yticks(range(len(num_vars)), num_vars)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()


# Class-wise numeric means
group_means = train.groupby(target)[num_vars].mean()
print("\nClass-wise means of numeric features:")
display(group_means)


# Univariate histograms
fig, axes = plt.subplots(3, 3, figsize=(14,10))
axes = axes.ravel()
for i, col in enumerate(num_vars):
    axes[i].hist(train[col].values, bins=30)
    axes[i].set_title(col)
plt.tight_layout()
plt.show()


# Build explicit contrasts for each categorical (treatment coding)
encoded_cats = [contrast(v, 'drop') for v in cat_vars]

# 3) Assemble ModelSpec terms (categorical contrasts + numeric features)
all_terms = encoded_cats + num_vars

# 4) Fit the design on TRAIN+TEST predictors to capture ALL category levels
design = MS(all_terms, intercept=True)
predictor_cols = cat_vars + num_vars
combined_predictors = pd.concat([train[predictor_cols], test[predictor_cols]], ignore_index=True)
_ = design.fit(combined_predictors)

# 5) Transform TRAIN and TEST -> fully numeric design matrices
X_full      = design.transform(train[predictor_cols]) 
X_test_full = design.transform(test[predictor_cols])
y_full      = train[target]

# 6) Train/Valid split (stratified) using row indices so matrices stay aligned
from sklearn.model_selection import train_test_split
idx_train, idx_valid = train_test_split(
    np.arange(train.shape[0]),
    test_size=0.20,
    stratify=y_full,
    random_state=42
)
X_train = X_full.iloc[idx_train].copy()
X_valid = X_full.iloc[idx_valid].copy()
y_train = y_full.iloc[idx_train].copy()
y_valid = y_full.iloc[idx_valid].copy()

# 7) Drop ISLP 'intercept' 
if 'intercept' in X_train.columns:
    X_train_no_int = X_train.drop(columns=['intercept'])
    X_valid_no_int = X_valid.drop(columns=['intercept'])
    X_full_no_int  = X_full.drop(columns=['intercept'])
    X_test_no_int  = X_test_full.drop(columns=['intercept'])
else:
    X_train_no_int = X_train
    X_valid_no_int = X_valid
    X_full_no_int  = X_full
    X_test_no_int  = X_test_full



# standardize for Logit and SVM 
logit_clf = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('model', LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=2000, n_jobs=None, random_state=42
    ))
])

lda_clf = LDA(store_covariance=True, solver='svd')  
nb_clf  = GaussianNB()                             
svm_clf = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('model', SVC(kernel='rbf', C=3.0, gamma='scale', probability=True, random_state=42))
])

models = {
    'logit_multinomial': logit_clf,
    'lda': lda_clf,
    'naive_bayes': nb_clf,
    'svm_rbf': svm_clf
}

# Add shrinkage LDA variant (uses Ledoit–Wolf)
models['lda_shrink'] = LDA(solver='lsqr', shrinkage='auto')


results = []

def get_estimator(m):
    # works for Pipeline or bare estimator
    return m.named_steps['model'] if hasattr(m, 'named_steps') else m

# ensure 1D string labels 
y_train_1d = y_train.astype(str).to_numpy()
y_valid_1d = y_valid.astype(str).to_numpy()
y_full_1d  = y_full.astype(str).to_numpy()

for name, mdl in models.items():
    print(f"\n=== Fitting: {name} ===")

    # Fit
    mdl.fit(X_train_no_int, y_train_1d)

    # Predict labels
    y_pred = mdl.predict(X_valid_no_int)
    acc = accuracy_score(y_valid_1d, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # ISLP confusion table (rows=Predicted, cols=Truth)
    print("\nConfusion Table (rows=Predicted, cols=Truth):")
    C = confusion_table(y_pred, y_valid_1d)
    display(C)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_valid_1d, y_pred, digits=3))

    # Macro AUC (One-vs-Rest)
    macro_auc = np.nan
    est = get_estimator(mdl)
    if hasattr(est, "predict_proba"):
        y_proba = mdl.predict_proba(X_valid_no_int)
        classes_est = np.array([str(c) for c in est.classes_])  # order matches proba columns
        y_valid_bin = label_binarize(y_valid_1d, classes=classes_est)
        try:
            macro_auc = roc_auc_score(y_valid_bin, y_proba, average='macro')
            print(f"Macro ROC AUC (OvR): {macro_auc:.4f}")
        except Exception as e:
            print("AUC not computed:", e)

    # 5-fold CV accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(mdl, X_full_no_int, y_full_1d, cv=cv, scoring='accuracy')
    print(f"5-Fold CV Accuracy: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    results.append({
        'model': name,
        'val_accuracy': acc,
        'cv_mean_acc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'macro_auc_ovr': macro_auc
    })

results_df = pd.DataFrame(results).sort_values(by='val_accuracy', ascending=False)
print("\n=== Model Comparison (Validation) ===")
display(results_df)


# Compact SVM (RBF) hyperparameter sweep
def to_1d_str(y):
    # If DataFrame, require a single column then squeeze
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y has shape {y.shape}; expected a single column.")
        y = y.iloc[:, 0]
    # Convert to numpy and flatten if needed
    arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.reshape(-1)
    return arr.astype(str)

y_train_1d = to_1d_str(y_train)
y_valid_1d = to_1d_str(y_valid)
y_full_1d  = to_1d_str(y_full)
# Pipeline: scale -> SVC(probabilities enabled)
svm_pipe = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('model', SVC(kernel='rbf', probability=True, random_state=42))
])

# Compact grid
param_grid = {
    'model__C':     [0.5, 1, 2, 3, 5],
    'model__gamma': ['scale', 0.03, 0.1, 0.3]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_gs = GridSearchCV(
    estimator=svm_pipe,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)

# Fit on training split
svm_gs.fit(X_train_no_int, y_train_1d)
print("Best params:", svm_gs.best_params_)
print("Best CV mean accuracy:", svm_gs.best_score_)

# Validate
y_pred = svm_gs.predict(X_valid_no_int)
val_acc = accuracy_score(y_valid_1d, y_pred)
print(f"\nValidation Accuracy (tuned SVM): {val_acc:.4f}")

print("\nConfusion Table (rows=Predicted, cols=Truth):")
display(confusion_table(y_pred, y_valid_1d))

print("\nClassification Report:")
print(classification_report(y_valid_1d, y_pred, digits=3))

# Macro AUC (OvR)
est = svm_gs.best_estimator_.named_steps['model']
classes_est = np.array([str(c) for c in est.classes_])
y_proba = svm_gs.predict_proba(X_valid_no_int)
y_valid_bin = label_binarize(y_valid_1d, classes=classes_est)
macro_auc = roc_auc_score(y_valid_bin, y_proba, average='macro')
print(f"Macro ROC AUC (OvR): {macro_auc:.4f}")

# CV results heatmap (mean accuracy)
cvres = pd.DataFrame(svm_gs.cv_results_)
heat = cvres.pivot(index='param_model__C', columns='param_model__gamma', values='mean_test_score')
plt.figure(figsize=(6,4))
plt.imshow(heat.values, aspect='auto')
plt.xticks(range(heat.shape[1]), heat.columns.astype(str), rotation=45, ha='right')
plt.yticks(range(heat.shape[0]), heat.index.astype(str))
plt.title('SVM Grid CV Mean Accuracy')
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.tight_layout()
plt.show()




#  helper: ensure 1-D string labels
def to_1d_str(y):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(str)

y_train_1d = to_1d_str(y_train)
y_valid_1d = to_1d_str(y_valid)
y_full_1d  = to_1d_str(y_full)

# scoring helper
def eval_model(name, mdl):
    mdl.fit(X_train_no_int, y_train_1d)
    y_pred = mdl.predict(X_valid_no_int)
    acc = accuracy_score(y_valid_1d, y_pred)

    print(f"\n=== {name} ===")
    print(f"Validation Accuracy: {acc:.4f}")

    # ISLP confusion table (rows=Predicted, cols=Truth)
    print("\nConfusion Table (rows=Predicted, cols=Truth):")
    display(confusion_table(y_pred, y_valid_1d))

    print("\nClassification Report:")
    print(classification_report(y_valid_1d, y_pred, digits=3))

    # Macro AUC (OvR) if proba available
    macro_auc = np.nan
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X_valid_no_int)
        classes_est = np.array([str(c) for c in mdl.classes_])
        y_val_bin = label_binarize(y_valid_1d, classes=classes_est)
        macro_auc = roc_auc_score(y_val_bin, proba, average='macro')
        print(f"Macro ROC AUC (OvR): {macro_auc:.4f}")

    return acc, macro_auc, y_pred


v = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt_base = DecisionTreeClassifier(random_state=42)
dt_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 6, 12],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
}
dt_gs = GridSearchCV(dt_base, dt_grid, scoring='accuracy', cv=cv, n_jobs=-1, refit=True, verbose=1)
dt_gs.fit(X_train_no_int, y_train_1d)
dt_best = dt_gs.best_estimator_
print("Decision Tree best params:", dt_gs.best_params_)

dt_acc, dt_auc, _ = eval_model("Decision Tree (tuned)", dt_best)

# Feature importances 
dt_imp = pd.Series(dt_best.feature_importances_, index=X_train_no_int.columns).sort_values(ascending=False).head(20)
print("\nDecision Tree — Top 20 Feature Importances:")
display(dt_imp)

#  plot a shallow view of the tree
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(dt_best, max_depth=3, feature_names=X_train_no_int.columns, class_names=dt_best.classes_, filled=True, fontsize=6)
plt.tight_layout()
fig.savefig("/kaggle/working/tree_preview.png", dpi=150)
plt.close(fig)




bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=200,
    oob_score=True,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bag_acc, bag_auc, _ = eval_model("Bagging (200 trees, OOB enabled)", bag)
if hasattr(bag, "oob_score_"):
    print(f"OOB Score: {bag.oob_score_:.4f}")



rf = RandomForestClassifier(
    n_estimators=400,
    max_features='sqrt',
    min_samples_leaf=1,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf_acc, rf_auc, _ = eval_model("Random Forest (400, sqrt)", rf)
print(f"OOB Score: {rf.oob_score_:.4f}")

rf_imp = pd.Series(rf.feature_importances_, index=X_train_no_int.columns).sort_values(ascending=False).head(20)
print("\nRandom Forest — Top 20 Feature Importances:")
display(rf_imp)




gb_grid = {
    'n_estimators': [200, 400],
    'learning_rate': [0.05, 0.1],
    'max_depth': [1, 2]
}
gb_gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_grid, scoring='accuracy', cv=cv, n_jobs=-1, refit=True, verbose=1
)
gb_gs.fit(X_train_no_int, y_train_1d)
gb_best = gb_gs.best_estimator_
print("Gradient Boosting best params:", gb_gs.best_params_)

gb_acc, gb_auc, _ = eval_model("Gradient Boosting (tuned)", gb_best)

gb_imp = pd.Series(gb_best.feature_importances_, index=X_train_no_int.columns).sort_values(ascending=False).head(20)
print("\nGradient Boosting — Top 20 Feature Importances:")
display(gb_imp)


results_tree = pd.DataFrame({
    'model': ['DecisionTree', 'Bagging', 'RandomForest', 'GradientBoosting'],
    'val_accuracy': [dt_acc, bag_acc, rf_acc, gb_acc],
    'macro_auc_ovr': [dt_auc, bag_auc, rf_auc, gb_auc]
}).sort_values('val_accuracy', ascending=False)
print("\n=== Validation Summary (Tree-Based Models) ===")
display(results_tree)


os.makedirs("/kaggle/working", exist_ok=True)

# Models and their validation metrics from Section 8
model_objs = {
    'DecisionTree': dt_best,          # tuned GridSearchCV best estimator
    'Bagging': bag,                   # fitted spec
    'RandomForest': rf,               # fitted spec
    'GradientBoosting': gb_best       # tuned GridSearchCV best estimator
}
val_scores = {
    'DecisionTree': dt_acc,
    'Bagging': bag_acc,
    'RandomForest': rf_acc,
    'GradientBoosting': gb_acc
}
auc_scores = {
    'DecisionTree': dt_auc,
    'Bagging': bag_auc,
    'RandomForest': rf_auc,
    'GradientBoosting': gb_auc
}

# Select best by validation accuracy; break ties with macro AUC
best_name = max(val_scores, key=lambda k: (val_scores[k], np.nan_to_num(auc_scores[k], nan=-1.0)))
best_model = model_objs[best_name]

print(f"Best model: {best_name} "
      f"(val acc={val_scores[best_name]:.4f}, macro AUC={auc_scores[best_name]:.4f})")

# Fit BEST model on the FULL training set
best_model.fit(X_full_no_int, y_full_1d)

# Predict test set and save SINGLE Kaggle-ready file
test_pred = best_model.predict(X_test_no_int)
submission = pd.DataFrame({'id': test['id'], 'NObeyesdad': test_pred})
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Saved single submission:", out_path)




