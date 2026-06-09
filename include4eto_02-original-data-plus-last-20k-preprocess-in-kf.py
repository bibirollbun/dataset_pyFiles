import sklearn
sklearn.__version__


!pip install hillclimbers scikit-learn==1.6.1


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


target_col = 'physical_activity_minutes_per_week'
window_size = 1000
rolling_mean = df_train[target_col].rolling(window=window_size).mean()

threshold = 88
cutoff_mask = rolling_mean > threshold

# Get the first ID that satisfies the condition
cutoff_id = rolling_mean[cutoff_mask].index.min()
print(cutoff_id, int(0.678 * 1e6))

sep = cutoff_id


print(f"Pre-shift positive rate: {(df_train.iloc[:sep]['diagnosed_diabetes'] == 1).mean():.6f}")
print(f"Post-shift positive rate: {(df_train.iloc[sep:]['diagnosed_diabetes'] == 1).mean():.6f}")
print(f"Overall positive rate: {(df_train['diagnosed_diabetes'] == 1).mean():.6f}")


corr_train_last_20k = df_train.iloc[sep:].corr(numeric_only=True)['diagnosed_diabetes'].sort_values()
corr_orig = df_original.corr(numeric_only=True)['diagnosed_diabetes'].sort_values()[corr_train_last_20k.index]
corr_all_train = df_train.corr(numeric_only=True)['diagnosed_diabetes'].sort_values()
corr_diff_last_20k = np.abs(np.abs(corr_train_last_20k) - np.abs(corr_orig))
corr_diff_all = np.abs(np.abs(corr_all_train) - np.abs(corr_orig))


corr_all_train.abs().sort_values(ascending=False)
# Family history, physical activity, age are our top predictors
# Family history if by far the biggest


corr_diff_last_20k[corr_all_train.abs().sort_values(ascending=False).index]


corr_diff_all[corr_all_train.abs().sort_values(ascending=False).index]


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

col = 'physical_activity_minutes_per_week'
df_train[col].rolling(500).mean().plot(color='red', label='train', ax=ax[0])
df_test[col].rolling(500).mean().plot(color='blue', label='test', ax=ax[0])

ax[0].set_xlim(0.65 * 1e6, 0.725 * 1e6)
ax[0].vlines(sep, 75, 100)
ax[0].legend()

sns.kdeplot(df_train[col], label='train', ax=ax[1])
sns.kdeplot(df_test[col], label='test', ax=ax[1])
ax[1].legend()
ax[1].grid()


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


from sklearn.preprocessing import FunctionTransformer


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


# last 20k 50x more important
# in reality since the dataset size is different, this is ~10x more important
SAMPLE_WEIGHT = 750.0
# SAMPLE_WEIGHT = 5236.8421052631575

sample_weights = pd.Series(
    # index=df_train.index,
    data=np.hstack([
        np.ones((df_original.shape[0],)) * 1.0,
        np.ones((df_train.shape[0] - sep,)) * SAMPLE_WEIGHT
    ])
)


Xs_orig = df_original[NUMERICAL + CATEGORICAL + ORDINAL]
ys_orig = df_original[TARGETS]

Xs = df_train[NUMERICAL + CATEGORICAL + ORDINAL].iloc[sep:]
ys = df_train[TARGETS].iloc[sep:]

Xs_train_all = pd.concat([Xs_orig, Xs])
Xs_test = df_test.loc[:, NUMERICAL + CATEGORICAL + ORDINAL]


ys_reduced = pd.concat([ys_orig, ys]).to_numpy()


# we want to always sample even classes AND some of the last 20k
# Create a stratification variable that combines target and data source
ys_sample = np.hstack([
    ys_orig.to_numpy().flatten() * 2,  # original data: class 0 -> 0, class 1 -> 2
    ys.to_numpy().flatten() * 2 + 1    # last 20k data: class 0 -> 1, class 1 -> 3
])
# This creates 4 groups: [orig_class0, last20k_class0, orig_class1, last20k_class1]


FOLDS = 25
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1301)

oof_pred_lr = np.zeros(Xs_train_all .shape[0])
test_preds_lr = np.zeros((Xs_test.shape[0], FOLDS))
fold_roc_auc_lr = []
lr_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(Xs_train_all, ys_sample), start=1):
    print(f"Training fold {fold} ...")

    X_train_ = Xs_train_all.iloc[train_idx,:]
    X_val_   = Xs_train_all.iloc[val_idx,:]
    y_train = ys_reduced[train_idx,:]
    y_val   = ys_reduced[val_idx,:]
    s_wei     = sample_weights[train_idx]
    s_wei_val = sample_weights[val_idx]

    # random numbers I generated so I don't use range(5)
    # doesn't matter what they are so long as they're consistent btwn. runs
    seeds = [14562, 3691540]
    restart_auc = []
    
    for seed in seeds:
        preprocess_orig = create_feat_transformer(seed, CATEGORICAL, NUMERICAL, ORDINAL, ORDINAL_MAP)    
    
        X_train = preprocess_orig.fit_transform(X_train_, y_train.reshape(-1,))
        X_val = preprocess_orig.transform(X_val_)
        Xs_test_p = preprocess_orig.transform(Xs_test)
    
    
        lr_model = LogisticRegression(max_iter=1000, verbose=1, solver='lbfgs', class_weight='balanced', random_state=seed)
        lr_model.fit(X_train, y_train.ravel(), sample_weight=s_wei)
        lr_models.append(lr_model)
    
        # compute OOF predictions/test predictions
        lr_val_pred = lr_model.predict_proba(X_val)[:, 1]
        lr_test_pred = lr_model.predict_proba(Xs_test_p)[:, 1]
    
        lr_models.append(lr_model)
    
        roc_auc_lr = roc_auc_score(y_val, lr_val_pred, sample_weight=s_wei_val)
        
        oof_pred_lr[val_idx] += lr_val_pred
        test_preds_lr[:, fold - 1] += lr_test_pred
        restart_auc.append(roc_auc_lr)

        print(f'seed {seed}, auc {roc_auc_lr}')
    
    print(f"Fold {fold} ROC AUC (lr): {roc_auc_lr:.8f}")

    oof_pred_lr[val_idx] /= len(seeds)
    test_preds_lr[:, fold - 1] /= len(seeds)
    fold_roc_auc_lr.append(np.mean(restart_auc))



importance = np.mean([np.abs(lr_model.coef_[0]) for lr_model in lr_models], axis=0)
# feature_names = Xs_train.columns  # or use your feature names
feature_names = preprocess_orig.get_feature_names_out()

# Sort by importance
indices = np.argsort(importance)
sorted_features = feature_names[indices]
sorted_importance = importance[indices]

# Plot horizontally (features on y-axis for better readability)
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_importance)), sorted_importance)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('LR Feature Importance (Sorted)')
plt.tight_layout()
plt.show()


roc_auc_score(ys_reduced, oof_pred_lr, sample_weight=sample_weights)


def create_feat_transformer(random_state, CATEGORICAL, NUMERICAL, ORDINAL, ORDINAL_MAP):
    return ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('num', StandardScaler(), NUMERICAL),
            ('cat',
                 # "numerical" features are still quite low cardinality
                 Pipeline(steps=[
                     ('encoder', TargetEncoder(target_type='binary', cv=50, smooth='auto', random_state=random_state)),
                     # ('scaler', StandardScaler()),
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
                     ('t_encoder', TargetEncoder(target_type='binary', cv=50, smooth='auto', random_state=random_state)),
                     # ('scaler', StandardScaler()),
                 ]),
                 ORDINAL
            ),
            # ('pca',
            #      Pipeline(steps=[
            #          ('pca', PCA(n_components=3)),
            #          ('scaler', StandardScaler())
            #      ]),
            #      NUMERICAL
            # )
        ]
    )


FOLDS = 25
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1301)

oof_pred_xgb = np.zeros(Xs_train_all.shape[0])
test_preds_xgb = np.zeros((Xs_test.shape[0], FOLDS))
fold_roc_auc_xgb = []
xgb_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(Xs_train_all.to_numpy(), ys_sample), start=1):
    print(f"Training fold {fold} ...")

    X_train_ = Xs_train_all.iloc[train_idx,:]
    X_val_   = Xs_train_all.iloc[val_idx,:]
    y_train = ys_reduced[train_idx,:]
    y_val   = ys_reduced[val_idx,:]
    s_wei     = sample_weights[train_idx]
    s_wei_val = sample_weights[val_idx]

    seeds = [14562, 3691540]
    restart_auc = []
    
    for seed in seeds:
        preprocess_orig = create_feat_transformer(seed, CATEGORICAL, NUMERICAL, ORDINAL, ORDINAL_MAP)
        X_train = preprocess_orig.fit_transform(X_train_, y_train.reshape(-1,))
        X_val = preprocess_orig.transform(X_val_)
        Xs_test_p = preprocess_orig.transform(Xs_test)
    
    
        xgb_model = xgboost.XGBClassifier(
            **{'n_estimators': 15000,
                 'learning_rate': 0.02,
                 'max_depth': 3,
                 'subsample': 0.7,
                 'colsample_bytree': 0.7,
                 # 'min_child_weight': 10,
                 # 'reg_lambda': 1.5,
                 'tree_method': 'hist',
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'random_state': seed,
                 'n_jobs': -1,
                 'early_stopping_rounds':100,
                  'scale_pos_weight': (y_train.shape[0] - np.sum(y_train)) / np.sum(y_train),
        })
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=s_wei,
            sample_weight_eval_set=[s_wei_val],
            verbose=100
        )
        # compute OOF predictions/test predictions
        xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_test_pred = xgb_model.predict_proba(Xs_test_p)[:, 1]
    
        xgb_models.append(xgb_model)
        roc_auc_xgb = roc_auc_score(y_val, xgb_val_pred, sample_weight=s_wei_val)
    
        oof_pred_xgb[val_idx] += xgb_val_pred
        test_preds_xgb[:, fold - 1] += xgb_test_pred
        restart_auc.append(roc_auc_xgb)

        print(f'seed {seed}, auc {roc_auc_xgb}')
    
    print(f"Fold {fold} ROC AUC (lr): {roc_auc_xgb:.8f}")

    oof_pred_xgb[val_idx] /= len(seeds)
    test_preds_xgb[:, fold - 1] /= len(seeds)
    fold_roc_auc_xgb.append(np.mean(restart_auc))


roc_auc_score(ys_reduced[df_original.shape[0]:], oof_pred_xgb[df_original.shape[0]:])


importance = np.mean([np.abs(xgb_model.feature_importances_) for xgb_model in xgb_models], axis=0)
# feature_names = Xs_train.columns  # or use your feature names
feature_names = preprocess_orig.get_feature_names_out()

# Sort by importance
indices = np.argsort(importance)
sorted_features = feature_names[indices]
sorted_importance = importance[indices]

# Plot horizontally (features on y-axis for better readability)
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_importance)), sorted_importance)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGB Feature Importance (train-20k)')
plt.tight_layout()
plt.show()



preprocess_all = create_feat_transformer(42, CATEGORICAL, NUMERICAL, ORDINAL, ORDINAL_MAP)
X_train_all_p = preprocess_all.fit_transform(Xs_train_all, ys_reduced.reshape(-1,))
Xs_test_all_p = preprocess_all.transform(Xs_test)



oof_pred_df = pd.DataFrame({
    "LR": oof_pred_lr,
    "XGB": oof_pred_xgb
})

test_pred_df = pd.DataFrame({
    "LR": test_preds_lr.mean(axis=1),
    "XGB": test_preds_xgb.mean(axis=1)
})

df_climb = pd.DataFrame(
    # The train columns aren't actually used by hillclimbers
    np.hstack([X_train_all_p, ys_reduced]),
    columns=list(preprocess_orig.get_feature_names_out()) + TARGETS
)


# save oof and test preds
oof_pred_df.to_csv('10_oof_pred_original_20k.csv', index=False)
test_pred_df.to_csv('10_test_pred_original_20k.csv', index=False)


def hc_score(ys_true, ys_pred):
    return roc_auc_score(ys_true.iloc[df_original.shape[0]:], ys_pred.iloc[df_original.shape[0]:])#, sample_weight=sample_weights)


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


roc_auc_score(ys_reduced[df_original.shape[0]:], oof_preds_ensemble[df_original.shape[0]:])


# CV: 0.7026434196778887, LB: 0.70666 - (only XGB + LR/ 50.0)
# CV: 0.7027469943184947, LB: 0.70671 - 100.0
# CV: 0.7029174247103394, LB: 0.70688 - 500.0
# CV: 0.7031292741534567, LB: 0.70690 - 750.0
# CV: 0.7032562325497485, LB: 0.70683 - 5236.8421052631575
# the coefs don't explode with larger weights, though


# Create submission
sub = pd.DataFrame({
    'id': df_test.index,
    'diagnosed_diabetes': test_preds
})

sub.to_csv('submission.csv', index=False)


1

