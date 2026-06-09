import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import iqr

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
import category_encoders as ce

target_colors = ['#ffcc33', '#f6eabe']

import warnings
warnings.filterwarnings('ignore')

seed = 48

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')


tr_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
ts_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
sb_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

or_00 = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')[tr_00.columns.tolist()]

target = 'diagnosed_diabetes'

tr_00.head(3)


or_00.head(3)


tr_00.sample(5).T


num_feats = [feat for feat in ts_00.select_dtypes(include='number').columns.tolist() if ts_00[feat].nunique()>2]
cat_feats = ts_00.select_dtypes(exclude='number').columns.tolist()


# Define function to handle outliers
def remove_outliers(df):
    df = df.copy()
    for col in num_feats:
        if df[col].nunique()>20:
            IQR = iqr(df[col])  # calculate the interquartile range
            df[col] = np.clip(df[col], 
                              (np.quantile(df[col], 0.25) - 1.51*IQR), 
                              (np.quantile(df[col], 0.75) + 1.51*IQR)
                             ) # clip the outliers in the range (25, 75)quantile -or+ 1.5 IQ
    return df

# Remove outliers from the various datasets
tr_01 = remove_outliers(tr_00)
ts_01 = remove_outliers(ts_00)
or_01 = remove_outliers(or_00)


run_feat_eng = False
cross_combine_cat_feats = False


# Your mapping dictionaries
gender__dico = {'Male': -1, "Female": 1, 'Other': 0}
education_level__dico = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
smoking_status__dico = {'Never': 0, 'Former': 1, 'Current': 2}  
income_level__dico = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
employment_status__dico = {'Unemployed': 0, 'Student': 1, 'Employed': 2, 'Retired': 3}

# Collect mappings in one place keyed by column name
cat_mappings = {
    'gender': gender__dico,
    'education_level': education_level__dico,
    'smoking_status': smoking_status__dico,    # be careful with name consistency
    'income_level': income_level__dico,
    'employment_status': employment_status__dico,
}

# Apply to each dataframe
for df in [tr_01, ts_01, or_01]:
    for col, mapping in cat_mappings.items():
        # Replace with numeric codes; keep any unknowns as-is
        df[col] = df[col].replace(mapping)
        # If you are sure everything maps to integers, you can use int instead.
        df[col] = df[col].astype('Int64')  # nullable integer dtype
    
    if run_feat_eng:
        # create new feature
        for df in [tr_01, ts_01, or_01]:
            df['diasto_to_systo'] = np.divide(df['diastolic_bp'], df['systolic_bp'])
            df['hdl_cholesterol_to_total'] = np.divide(df['hdl_cholesterol'], df['cholesterol_total'])
            df['ldl_cholesterol_to_total'] = np.divide(df['ldl_cholesterol'], df['cholesterol_total'])
            df['ldl_to_hld'] = np.divide(df['ldl_cholesterol'], df['hdl_cholesterol'])
            df['waistHip_to_bmi'] = np.divide(df['waist_to_hip_ratio'], df['bmi'])
            df['heartRate_to_bmi'] = np.divide(df['heart_rate'], df['bmi'])
            df['triglycerides_to_cholesterol'] = np.divide(df['triglycerides'], df['cholesterol_total'])
            df['histories'] = df['family_history_diabetes'] + df['hypertension_history'] + df['cardiovascular_history']
            df['cholesterol_*_systo'] = df['cholesterol_total']*df['systolic_bp']/100
            # df['bmi_class'] = pd.cut(df['bmi'], [0, 18.5, 25, 30, 35, 40, 100], labels=[1, 2, 3, 4, 5, 6]).astype('int')
            # df['triglycerides_class'] = pd.cut(df['triglycerides'], [0, 150, 199, 1000], labels=[1, 2, 3]).astype('int')
            # df['age_group'] = pd.cut(df['age'], [0, 18, 25, 35, 45, 60, 100], labels=[1, 2, 3, 4, 5, 6]).astype('int')
            # df.drop(columns=['bmi', 'age'], inplace=True)
            if cross_combine_cat_feats:
                # Generate unique pairs without repetition
                for feat_1, feat_2 in combinations(cat_and_bool_feat, 2):
                    # Concatenate as strings to create interaction feature
                    df[f'{feat_1}_{feat_2}'] = df[feat_1].astype(str) + '_' + df[feat_2].astype(str)
                else:
                    pass
                
        else:
            pass
    
tr_01.head(3)


cat_feats = ts_01.select_dtypes(exclude='number').columns.tolist()
num_feats = ts_01.select_dtypes(include='number').columns.tolist()

for df in [tr_01, ts_01, or_01]:
    df[cat_feats] = df[cat_feats].astype('category')

tr_00.info()


X = tr_01.copy()
X_or = or_01.copy()

y = X.pop(target)
y_or = X_or.pop(target)


from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

models = {
'cat_model' : CatBoostClassifier(
    iterations=6000,
    learning_rate=0.02,                # Controls step size; lower for fine-tuning
    depth=3,                           # Tree depth; balances bias-variance
    l2_leaf_reg=0.8,                   # L2 regularization on leaf values
    random_strength=0.5,               # Adds noise to tree splits for robustness
    bagging_temperature=0,             # Controls sampling randomness (0 = deterministic)
    border_count=200,                  # Number of splits for numerical features
    grow_policy='SymmetricTree',       # Alternatives: 'Depthwise', 'Lossguide'
    boosting_type='Ordered',           # Alternatives: 'Plain' (for small datasets)
    eval_metric='AUC',
    early_stopping_rounds=100,
    eval_fraction=0.2,
    verbose=200,
    random_seed=seed,                  # Ensures reproducibility
    use_best_model=True,               # Retain best iteration
    # task_type='GPU',                   # Use 'GPU' if available for speed
    od_type='Iter',                    # Overfitting detector type
),

'xgb_model': XGBClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    random_state=seed,
    verbosity=0,
    enable_categorical=True
)
}

# The preprocessor used to handle cat_features for xgb and voting classifiers
preprocessor = ColumnTransformer(
    transformers=[
        ('encoder', ce.CatBoostEncoder(), cat_feats)
    ],  
    remainder= 'passthrough'
)


# CatBoosClassifier
cat_clf = models['cat_model']

# CatBoostClassifier-preprocessor pipe
cat_clf_ = make_pipeline(preprocessor, cat_clf)

# XGBClassifier-preprocessor pipe
# xgb_clf = Pipeline([('prep', preprocessor), ('model', models['xgb_model'])])

xgb_clf = models['xgb_model']


def run_oof_analysis(model, X, y, n_splits):
    # if not run:
    #     # return None, None
    
    spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_pred_proba = pd.DataFrame()
    scores = []

    plt.figure(figsize=(6.6, 7 * n_splits))

    for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), 1):
        print("=" * 33, f" Fold {f} of {n_splits} ", "=" * 34)

        X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
        y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]

        clf = clone(model)

        # Fit depending on model type
        model_name = clf.__class__.__name__
        if model_name == "CatBoostClassifier":
            clf.fit(
                X_tr, y_tr, 
                cat_features=cat_feats, 
                # verbose=1
            )
        elif model_name == 'XGBClassifier':
            # clf = Pipeline([('prep', preprocessor), ('model', model)])
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=50, 
                verbose=0
            )
        else:
            clf.fit(
                X_tr, y_tr
            )

        # Predict probabilities
        preds = clf.predict_proba(X_va)[:, 1]
        score = metrics.roc_auc_score(y_va, preds)
        scores.append(score)

        print(" " * 53, f"•••> Fold {f} AUC: {score:.6f} ✅\n")

        # # Store test predictions
        test_pred_proba[f'test_proba_0{f}_{model_name}'] = clf.predict_proba(ts_01)[:, 1]
        
        # Plot ROC curve
        fpr, tpr, _ = metrics.roc_curve(y_va, preds)
        plt.subplot(10, 2, f)
        plt.plot(fpr, tpr, label=f'AUC = {score:.6f}', color='navy')
        plt.plot([0, 1], [0, 1], color='maroon', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title(f'ROC Curve Fold {f}', color='maroon', fontsize=10, weight='bold')
        plt.suptitle(f'OOF with {model_name[:3]}_clf', fontsize=12, weight='bold')
    plt.tight_layout(pad=2, h_pad=2, w_pad=2)

    # Add mean prediction across folds
    test_pred_proba['average_proba'] = test_pred_proba.mean(axis=1)
    display(test_pred_proba.head(5))


run_oof_analysis(cat_clf, X, y, n_splits=4)


run_oof_analysis(xgb_clf, X, y, n_splits=4)


# define the weights
cat_weight = 1
xgb_weight = 9

# Buil the VotingClassifier
vot_clf = VotingClassifier(
    estimators=[('cat', cat_clf_), ('xgb', xgb_clf)],
    voting='soft',
    weights=[cat_weight, xgb_weight] # Let's have a higher weight for xgboost since it performed better
)


run_oof_analysis(vot_clf, X, y, n_splits=4)


final_model = vot_clf

final_model.fit(X, y)


ts_proba = final_model.predict_proba(ts_01)[:, 1]


sb_00[target] = ts_proba

threshold = 0.54
median_val = sb_00[target].median()

plt.subplot(121)
sb_00[target].plot.hist(bins=30, color='grey', 
                        figsize=(10, 4), edgecolor='lightgrey', 
                        title='Hist of predicted_proba in test set')
plt.xlabel('Predicted Proba')
plt.axvline(x=median_val, color='#d9004c', linestyle='--', linewidth=2)
plt.text(median_val-0.2, plt.ylim()[1]*0.93, f'Median = {median_val:.2f}',
         color='#d9004c', ha='center', va='bottom', fontsize=12)

plt.subplot(122)
(sb_00[target] > threshold).value_counts().plot.pie(labels=['1', '0'], 
                                             autopct='%1.1f%%',
                                             startangle=90,
                                             explode=[0.01, 0.02],
                                             colors=target_colors,
                                             radius=1.2,
                                             wedgeprops={'width': 0.7},
                                             title=f'Distribution of proba: threshold of {threshold}'
                                             )

plt.ylabel('')
plt.show()


sb_00.to_csv('submission.csv', index=False)

print('The submission file is ready!')

