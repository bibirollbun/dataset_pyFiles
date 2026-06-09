import sklearn
sklearn.__version__


!pip install hillclimbers


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import seaborn as sns
import tqdm

from hillclimbers import climb_hill, partial


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_original = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col=0)



oof_pred_orig_20k = pd.read_csv('/kaggle/input/02-original-data-plus-last-20k-preprocess-in-kf/10_oof_pred_original_20k.csv')
oof_pred_all = pd.read_csv('/kaggle/input/08-use-original-and-all-train/08_oof_pred_all_train.csv')

test_pred_orig_20k = pd.read_csv('/kaggle/input/02-original-data-plus-last-20k-preprocess-in-kf/10_test_pred_original_20k.csv')
test_pred_all = pd.read_csv('/kaggle/input/08-use-original-and-all-train/08_test_pred_all_train.csv')


target_col = 'physical_activity_minutes_per_week'
window_size = 1000
rolling_mean = df_train[target_col].rolling(window=window_size).mean()

threshold = 88
cutoff_mask = rolling_mean > threshold

# Get the first ID that satisfies the condition
cutoff_id = rolling_mean[cutoff_mask].index.min()
print(cutoff_id, int(0.678 * 1e6))

sep = cutoff_id


CATEGORICAL = [
    'gender', 'ethnicity', 'family_history_diabetes',
    'hypertension_history', 'cardiovascular_history',
]

ORDINAL = [
    'education_level','income_level','smoking_status','employment_status'
]

ORDINAL_MAP  = {
    'education_level': ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
    'income_level': ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
    'smoking_status': ['Never', 'Former', 'Current'],
    'employment_status': ['Unemployed', 'Student', 'Employed', 'Retired'],
}

NUMERICAL = [
    'age',
    'alcohol_consumption_per_week','physical_activity_minutes_per_week',
    'diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi',
    'waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
    'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides'
]

CUSTOM_TRANSFORM = []
TARGETS = ['diagnosed_diabetes']


def create_feat_transformer(random_state, CATEGORICAL, NUMERICAL, ORDINAL, ORDINAL_MAP):
    return ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('num', StandardScaler(), NUMERICAL),
            ('cat',
                 # "numerical" features are still quite low cardinality
                 Pipeline(steps=[
                     ('encoder', TargetEncoder(target_type='binary', cv=25, random_state=random_state)),
                     ('scaler', StandardScaler()),
                 ]),
                 CATEGORICAL + NUMERICAL
            ),
            ('ord',
                 Pipeline(steps=[
                     ('o_enc',
                          OrdinalEncoder(
                             handle_unknown='error',
                             categories=[
                                 ORDINAL_MAP[c] for c in ORDINAL
                             ]
                          )
                     ),
                     # this step does destroy the ordinal encoding, but I found it works better
                     ('t_encoder', TargetEncoder(target_type='binary', cv=25, random_state=random_state)),
                     ('scaler', StandardScaler()),
                 ]),
                 ORDINAL
            ),
            ('pca',
                 Pipeline(steps=[
                     ('pca', PCA(n_components=3)),
                     ('scaler', StandardScaler())
                 ]),
                 NUMERICAL
            )
        ]
    )


ft = create_feat_transformer(42, CATEGORICAL, NUMERICAL, ORDINAL, ORDINAL_MAP)

ys = df_train[TARGETS].iloc[sep:].values.ravel()
Xs = ft.fit_transform(df_train.iloc[sep:][NUMERICAL + CATEGORICAL + ORDINAL], ys)


oof_pred_df = pd.DataFrame({
    'LR_20k': oof_pred_orig_20k['LR'].iloc[df_original.shape[0]:].values,
    'XGB_20k': oof_pred_orig_20k['XGB'].iloc[df_original.shape[0]:].values,
    'LR_all': oof_pred_all['LR'].iloc[sep + df_original.shape[0]:].values,
    'XGB_all': oof_pred_all['XGB'].iloc[sep + df_original.shape[0]:].values,
})

test_pred_df = pd.DataFrame({
    'LR_20k': test_pred_orig_20k['LR'],
    'XGB_20k': test_pred_orig_20k['XGB'],
    'LR_all': test_pred_all['LR'],
    'XGB_all': test_pred_all['XGB'],    
})

df_climb = pd.DataFrame(
    # The train columns aren't actually used by hillclimbers
    np.hstack([Xs, ys.reshape(-1, 1)]),
    columns=list(ft.get_feature_names_out()) + TARGETS
)


def hc_score(ys_true, ys_pred):
    return roc_auc_score(ys_true, ys_pred)


test_preds, oof_preds_ensemble = climb_hill(
     train=df_climb,
     oof_pred_df=oof_pred_df,
     test_pred_df=test_pred_df,
     target="diagnosed_diabetes",
     objective="maximize",
     eval_metric=partial(hc_score),
     negative_weights=False,
     precision=0.001,
     plot_hill=True,
     plot_hist=True,
    return_oof_preds=True
)


roc_auc_score(ys, oof_preds_ensemble)


# Create submission
sub = pd.DataFrame({
    'id': df_test.index,
    'diagnosed_diabetes': test_preds
})

sub.to_csv('submission_ensemble.csv', index=False)


# Prepare stacking input features (validation predictions)
stack_Xs = np.column_stack((
    oof_pred_orig_20k['LR'].iloc[df_original.shape[0]:].values,
    oof_pred_orig_20k['XGB'].iloc[df_original.shape[0]:].values,
    oof_pred_all['LR'].iloc[sep + df_original.shape[0]:].values,
    oof_pred_all['XGB'].iloc[sep + df_original.shape[0]:].values,
))
stack_ys = ys

# Train logistic regression meta-model
meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(stack_Xs, stack_ys)

# Predict on validation set using meta-model
stack_preds = meta_clf.predict_proba(stack_Xs)[:, 1]

# Predict on test set using meta-model
stack_test_Xs = np.column_stack((
    test_pred_orig_20k['LR'],
    test_pred_orig_20k['XGB'],
    test_pred_all['LR'],
    test_pred_all['XGB'],
))
stack_test_pred = meta_clf.predict_proba(stack_test_Xs)[:, 1]

print("Stacking Ensemble AUC:", roc_auc_score(stack_ys, stack_preds))


roc_auc_score(ys, stack_preds)


# Create submission
sub = pd.DataFrame({
    'id': df_test.index,
    'diagnosed_diabetes': test_preds
})

sub.to_csv('submission_lr_stack.csv', index=False)

