import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import iqr

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.base import clone

from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool



seed=65


tr_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
ts_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
sb_00 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

or_00 = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')[tr_00.columns.tolist()]

target = 'loan_paid_back'

tr_00.head(5)


for df in [tr_00, ts_00, or_00]:
    duplicate_count = df.duplicated().sum()
    print(duplicate_count)


cat_feats = ts_00.select_dtypes(exclude='number').columns
num_feats = ts_00.select_dtypes(include='number').columns


tr_00[target].value_counts().plot.pie(labels=['1', '0'], 
                                             autopct='%1.1f%%', \
                                             explode=[0.05, 0.05],
                                             colors=['blue', 'red'], 
                                             radius=1.2,
                                             wedgeprops={'width': 0.7},
                                             title='Target Proportions'
                                             )
plt.ylabel('')


for feat in num_feats:    
    fig = px.histogram(tr_00,
        x=feat,
        color=target,
        marginal="box", 
        barmode="group",
        nbins=50,
        height=400,
        width=600,
        title=f"{feat} Distribution")
    fig.show()


# Define function to handle outliers
def remove_outliers(df):
    for col in ts_00.select_dtypes(include='number').columns:
        IQR = iqr(df[col])  # calculate the interquartile range
        df[col] = np.clip(df[col], 
                          (np.quantile(df[col], 0.25) - 1.51*IQR), 
                          (np.quantile(df[col], 0.75) + 1.51*IQR))  # clip the outliers in the range (25, 75)quantile -or+ 1.5 IQ
    return df


for df in [tr_00, ts_00, or_00]:
    # df['gender_educLevel'] = df['gender'] + df['education_level']
    # df['gender_loanPurpose'] = df['gender'] + df['loan_purpose']
    # df['educLevel_grade'] = df['education_level'] + df['grade_subgrade']
    # df['gender_marital'] = df['gender'] + df['marital_status']
    # df['educlevel_emplStatus'] = df['gender'] + df['employment_status']
    # df['emplStatus_marital'] = df['employment_status'] + df['marital_status']
    remove_outliers(df)
    
    # df['emplStatus_loanPurpose'] = df['employment_status'] + df['loan_purpose']


tr_00.head()


cat_feats = ts_00.select_dtypes(exclude='number').columns
num_feats = ts_00.select_dtypes(include='number').columns


tr_00.describe(include='number').T


tr_00.describe(exclude='number')


count_1 = tr_00.groupby(['gender', 'marital_status', target])['loan_purpose'].count().mean()

count_1


for cat_feat in cat_feats:
    print(f'{cat_feat}: {tr_00[cat_feat].nunique()} uniques')


for cat_feat in cat_feats:
    print('\n'+20*'++')
    print(f'{cat_feat}: {tr_00[cat_feat].value_counts()} uniques')


def preprocessor_count(df, cat_feats=cat_feats, num_feats=num_feats, reference_df=or_00):
    
    """
    Preprocesses a DataFrame by:
    - Normalizing numeric features by category-wise mean
    - Adding category count features
    - Dropping original categorical columns

    Parameters:
    - df: DataFrame to process
    - cat_feats: list of categorical feature names
    - num_feats: list of numeric feature names
    - reference_df: optional reference DataFrame for category counts (default: df itself)

    Returns:
    - Transformed DataFrame
    """
    df = df.copy()
    ref_df = reference_df if reference_df is not None else df

    for cat_feat in cat_feats:
        # # Normalize numeric features by category-wise mean
        # df['mean_income'] = df.groupby(cat_feat)['annual_income'].transform('mean')
        # df['mean_loanAmount'] = df.groupby(cat_feat)['loan_amount'].transform('mean')
        # df[num_feats] = df[num_feats] - category_means

        # Add category count feature
        cat_feat_count = ref_df[cat_feat].value_counts()/ref_df.shape[0]
        df[f'{cat_feat}_count'] = df[cat_feat].map(cat_feat_count)
        # Drop original categorical column
        df = df.drop(columns=[cat_feat])

    return df


tr_01 = preprocessor_count(tr_00)
or_01 = preprocessor_count(or_00)
ts_01 = preprocessor_count(ts_00)

tr_01.head()


ts_01.head()


or_neg = or_01[or_01[target]==1]
or_neg


include_original_data = False

if include_original_data:
    train = pd.concat([tr_01, or_neg], ignore_index=True)
    print('The original data is combined to the train data!')
else:
    train = tr_01
    print('The original data is not being used!')


# prep the train sets
X = tr_01.copy()
y = X.pop(target)

# prep the original sets
X_or = or_01.copy()
y_or = X_or.pop(target)


# model = CatBoostClassifier(
#     iterations=30000,
#     learning_rate=0.024,                # Controls step size; lower for fine-tuning
#     depth=3,                            # Tree depth; balances bias-variance
#     l2_leaf_reg=5,                      # L2 regularization on leaf values
#     random_strength=2.5,               # Adds noise to tree splits for robustness
#     bagging_temperature=5.0,            # Controls sampling randomness (0 = deterministic)
#     border_count=450,                   # Number of splits for numerical features
#     grow_policy='Depthwise',            # Alternatives: 'Depthwise', 'Lossguide', 'SymmetricTree'
#     boosting_type='Plain',             # Alternatives: 'Ordered' (for small datasets)
#     eval_metric='AUC',
#     early_stopping_rounds=500,
#     eval_fraction=0.2,
#     verbose=500,
#     random_seed=seed,                   # Ensures reproducibility
#     use_best_model=True,                # Retain best iteration
#     # task_type='GPU',
#     # Use 'GPU' if available for speed
#     od_type='Iter',                     # Overfitting detector type
#     # od_wait=50, 
# )


model = CatBoostClassifier(
    iterations=30000,
    learning_rate=0.025,                # Controls step size; lower for fine-tuning
    depth=3,                           # Tree depth; balances bias-variance
    l2_leaf_reg=2.0,                   # L2 regularization on leaf values
    # random_strength=1.5,              # Adds noise to tree splits for robustness
    # bagging_temperature=1.0,          # Controls sampling randomness (0 = deterministic)
    border_count=4000,                 # Number of splits for numerical features
    grow_policy='Depthwise',           # Alternatives: 'Depthwise', 'Lossguide', 'SymmetricTree'
    # boosting_type='Plain',            # Alternatives: 'Ordered' (for small datasets)
    eval_metric='AUC',
    early_stopping_rounds=500,
    eval_fraction=0.2,
    verbose=500,
    random_seed=seed,                  # Ensures reproducibility
    use_best_model=True,               # Retain best iteration
    # task_type='GPU',
    # Use 'GPU' if available for speed
    od_type='Iter',                    # Overfitting detector type
    # od_wait=50, 
)


model.fit(X, y)


from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.base import clone
import matplotlib.pyplot as plt
import pandas as pd

n_splits = 4
spliter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Store out-of-fold predictions
oof_preds = []
oof_true = []

plt.figure(figsize=(6, 5))      
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]

    # Clone the model before fitting
    clf = clone(model)
    clf.fit(X_tr, y_tr)

    preds = clf.predict_proba(X_va)[:, 1]

    # Save for overall ROC
    oof_preds.extend(preds)
    oof_true.extend(y_va)

    # Per-fold AUC
    score = metrics.roc_auc_score(y_va, preds)
    print(f'â€¢â€¢â€¢> Fold_{f} AUC: {score:.6f} âœ…\n')

    # Per-fold ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_va, preds)
    plt.plot(fpr, tpr, label=f'Fold {f} AUC  = {score:.5f}')

# Overall ROC curve
overall_auc = metrics.roc_auc_score(oof_true, oof_preds)
fpr, tpr, _ = metrics.roc_curve(oof_true, oof_preds)
plt.plot(fpr, tpr, color='black', linewidth=2,
         label=f'Overall AUC = {overall_auc:.5f}')

# Diagonal baseline
plt.plot([0, 1], [0, 1], color='maroon', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('ROC Curves of All Folds + Overall', 
          color='maroon', fontsize=11, weight='bold')
plt.tight_layout()
plt.show()



ts_proba = model.predict_proba(ts_01)[:, 1]

sb_00[target] = ts_proba

sb_00.head(10)


plt.subplot(121)
sb_00[target].plot.hist(bins=25, color='green', 
                        figsize=(10, 4), 
                        title='Hist of pred_proba in test set')
plt.xlabel('Predicted Proba')

plt.subplot(122)
threshold = 0.7
(sb_00[target] > threshold).value_counts().plot.pie(labels=['1', '0'], 
                                             autopct='%1.1f%%', \
                                             explode=[0.05, 0.05],
                                             colors=['blue', 'red'], 
                                             radius=1.2,
                                             wedgeprops={'width': 0.7},
                                             title=f'Target for {threshold} threshold'
                                             )
plt.ylabel('')

# Change background color of the plot area
plt.gca().set_facecolor('lightgray')
# Change background color of the entire figure (optional)
plt.gcf().set_facecolor('lightgray')

plt.show()


sb_00.to_csv('submission.csv', index=False)

print('ğŸ�� The file is ready for submission! ğŸ��')


for feat in num_feats:    
    fig = px.histogram(tr_00,
        x=feat,
        color=target,
        marginal="box", 
        barmode="group",
        nbins=50,
        height=400,
        width=600,
        title=f"{feat} Distribution")
    fig.show()

