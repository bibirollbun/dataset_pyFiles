import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics

seed = 48

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')


tr_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
ts_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
sb_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

or_00 = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')[tr_00.columns.tolist()]

target = 'loan_paid_back'

tr_00.head(5)


or_00.head(2)


bins = [300, 580, 670, 740, 800, 850]
labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

for df in [tr_00, ts_00, or_00]:
    # df['credit_score_group'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=False)
    # df['loan_to_income'] = np.divide(df['loan_amount'], df['annual_income'])
    # df['annual_payment'] = df['loan_amount']*df['interest_rate']/100
    # df['annual(payemnt_to_income)'] = np.divide(df['annual_payment'], df['annual_income'])
    df['grade'] = df['grade_subgrade'].apply(lambda x: x[0])
    df.drop(columns='grade_subgrade', inplace=True)

tr_00.head()


or_00.info()


cat_feats = ts_00.select_dtypes(exclude='number').columns.tolist()
num_feats = ts_00.select_dtypes(include='number').columns.tolist()


for df in [tr_00, ts_00, or_00]:
    df[cat_feats] = df[cat_feats].astype('category')

tr_00.info()


X = tr_00.copy()
X_or = or_00.copy()

y = X.pop(target)
y_or = X_or.pop(target)


from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

models = {
'cat_model' : CatBoostClassifier(
    iterations=35000,
    learning_rate=0.03,                # Controls step size; lower for fine-tuning
    depth=3,                           # Tree depth; balances bias-variance
    l2_leaf_reg=0.8,                   # L2 regularization on leaf values
    random_strength=0.5,              # Adds noise to tree splits for robustness
    bagging_temperature=0,          # Controls sampling randomness (0 = deterministic)
    border_count=4000,                  # Number of splits for numerical features
    grow_policy='SymmetricTree',       # Alternatives: 'Depthwise', 'Lossguide'
    # boosting_type='Ordered',            # Alternatives: 'Plain' (for small datasets)
    eval_metric='AUC',
    early_stopping_rounds=500,
    eval_fraction=0.15,
    verbose=500,
    random_seed=seed,                    # Ensures reproducibility
    use_best_model=True,               # Retain best iteration
    # task_type='GPU',                   # Use 'GPU' if available for speed
    od_type='Iter',                    # Overfitting detector type
    # od_wait=50, 
),

'xgb_model': XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1,
    objective='binary:logistic',
    eval_metric='auc',
    # early_stopping_rounds=50,
    tree_method='hist',
    random_state=seed,
    verbosity=1,
    enable_categorical=True
)

}


model = models['cat_model']

model.fit(X, y, cat_features=cat_feats)


n_splits = 4
scorer = 'roc_auc_ovo'

spliter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

test_pred_proba = pd.DataFrame()

plt.figure(figsize=(8,24))      
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), 1):
    # print(15*'--' + f'Training fold {f} of {n}' + 15*'--')
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]
    # Fit the model and predict_proba on validation
    clf = model.fit(X_tr, y_tr, cat_features=cat_feats)
    preds = clf.predict_proba(X_va)[:, 1]
    # Get the acu scores
    score = metrics.roc_auc_score(y_va, preds)
    print('•••> Fold_{} auc: {:.6f} ✓\n'.format(f, score))
    # Predit proba on test data
    test_pred_proba[f'y_test_proba_fold_{f}'] = clf.predict_proba(ts_00)[:, 1]
    
    # Plot the roc_curve of the models predictions
    plt.subplot(10, 2, f)  
    tpr, fpr, _  = metrics.roc_curve(y_va, preds)
    plt.plot(tpr, fpr, label='auc = {:.5f}'.format(score))
    plt.plot([0, 1], [0, 1], color='maroon')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title(f'roc_curves for fold_{f}', 
              color='maroon', fontsize=11, weight='bold')
plt.tight_layout(pad=2, h_pad=2, w_pad=2)


ts_proba = model.predict_proba(ts_00)[:, 1]


ax = pd.Series(ts_proba).plot.hist(
    bins=50, figsize=(8, 3), color='orange',
    title='Distribution of Predicted Test Probabilities')
plt.xlabel('Predicted Probalities');


sb_00[target] = ts_proba

sb_00.to_csv('submission.csv', index=False)

