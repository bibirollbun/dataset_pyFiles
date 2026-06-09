import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics



tr_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
ts_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
sb_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

target = 'loan_paid_back'

tr_00.head(5)


for df in [tr_00, ts_00]:
    df['grade'] = df['grade_subgrade'].apply(lambda x: x[0])

tr_00.head()


cat_feats = ts_00.select_dtypes(exclude='number').columns
num_feats = ts_00.select_dtypes(include='number').columns

count_cat_feats = ['gender', 'marital_status', 'loan_purpose']
order_cat_feats = ['education_level', 'employment_statu', 'grade_subgrade', 'grade']


education_level_dico = {
    'High School': 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3,
    "Other": -1
}

employment_status_dico = {
    'Employed': 5,
    'Unemployed': 1,
    'Self-employed': 4,
    'Retired': 3,
    'Student': 2
}

grade_dico = {
    'F': 1,
    'E': 2,
    'D': 3,
    'C': 4,
    'B': 5,
    'A': 1
}

grade_subgrade_list = tr_00['grade_subgrade'].sort_values().unique()
grade_subgrade_dico = {k: v for v, k in enumerate(grade_subgrade_list, start=1)}

for df in [tr_00, ts_00]:
    df['education_level'] = df['education_level'].map(education_level_dico)
    df['employment_status'] = df['employment_status'].map(employment_status_dico)
    df['grade'] = df['grade'].map(grade_dico)
    df['grade_subgrade'] = df['grade_subgrade'].map(grade_subgrade_dico)

    # for cat_feat in count_cat_feats:
    #     # Add category count feature
    #     cat_feat_count = tr_00[cat_feat].value_counts()
    #     df[f'{cat_feat}_count'] = df[cat_feat].map(cat_feat_count)
    #     # Drop original categorical column
    #     df = df.drop(columns=[cat_feat])

tr_00


# gender_dico = {
#     'Female': 1,
#     'Male': -1,
#     'Other': 0,
# }

# marital_status_dico = {
#     'Single': 0,
#     'Married': 1,
#     'Divorsed': -1,
#     'Widowed': -2
# }

# education_level_dico = {
#     'High School': 0,
#     "Bachelor's": 1,
#     "Master's": 2,
#     "PhD": 3,
#     "Other": -1
# }

# employment_status_dico = {
#     'Employed': 5,
#     'Unemployed': 1,
#     'Self-employed': 4,
#     'Retired': 3,
#     'Student': 2
# }

# grade_dico = {
#     'F': 1,
#     'E': 2,
#     'D': 3,
#     'C': 4,
#     'B': 5,
#     'A': 1
# }

# grade_subgrade_count = tr_00['grade_subgrade'].value_counts()
# loan_purpose_count = tr_00['loan_purpose'].value_counts()

# # Numerize cat_features
# def preprocessor(df):
#     df = df.copy()
#     df['grade'] = df['grade_subgrade'].apply(lambda x: str(x)[0])
#     df['grade'] = df['grade'].map(grade_dico)
#     df['gender'] = df['gender'].map(gender_dico)
#     df['marital_status'] = df['marital_status'].map(marital_status_dico)
#     df['education_level'] = df['education_level'].map(education_level_dico)
#     df['employment_status'] = df['employment_status'].map(employment_status_dico)
#     df['grade_subgrade'] = df['grade_subgrade'].map(grade_subgrade_count)
#     df['loan_purpose'] = df['loan_purpose'].map(loan_purpose_count)
#     # df = df.drop(columns=['grade_subgrade'])

#     return df


X = tr_00.copy()
y = X.pop(target)


from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

models = {
'cat_model' : CatBoostClassifier(
    iterations=25000,
    learning_rate=0.02,                # Controls step size; lower for fine-tuning
    depth=3,                           # Tree depth; balances bias-variance
    l2_leaf_reg=0.8,                   # L2 regularization on leaf values
    random_strength=0.5,              # Adds noise to tree splits for robustness
    bagging_temperature=0,          # Controls sampling randomness (0 = deterministic)
    border_count=3000,                  # Number of splits for numerical features
    grow_policy='SymmetricTree',       # Alternatives: 'Depthwise', 'Lossguide'
    # boosting_type='Ordered',            # Alternatives: 'Plain' (for small datasets)
    eval_metric='AUC',
    early_stopping_rounds=250,
    eval_fraction=0.1,
    verbose=500,
    random_seed=42,                    # Ensures reproducibility
    use_best_model=True,               # Retain best iteration
    task_type='GPU',                   # Use 'GPU' if available for speed
    od_type='Iter',                    # Overfitting detector type
    # od_wait=50, 
),

'xgb_model ': XGBClassifier(
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
    random_state=42,
    verbosity=1
)

}


model = models['cat_model']
model.fit(X, y, cat_features=count_cat_feats)


seed = 42
n_splits = 4
scorer = 'roc_auc_ovo'

spliter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

test_pred_proba = pd.DataFrame()

plt.figure(figsize=(9,36))      
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), 1):
    # print(15*'--' + f'Training fold {f} of {n}' + 15*'--')
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]
    # Fit the model and predict_proba on validation
    clf = model.fit(X_tr, y_tr, cat_features=count_cat_feats)
    preds = clf.predict_proba(X_va)[:, 1]
    # Get the acu scores
    score = metrics.roc_auc_score(y_va, preds)
    print('Fold_{} ==> auc: {:.6f}'.format(f, score))
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
    plt.title(f'roc_curves for fold_{f} on {len(preds)} candidates', 
              color='maroon', fontsize=11, weight='bold')
plt.tight_layout(pad=2, h_pad=3, w_pad=3)
# display(test_pred_proba)


ts_proba = model.predict_proba(ts_00)[:, 1]

sb_00[target] = ts_proba

sb_00.head()


sb_00.to_csv('submission.csv', index=False)

