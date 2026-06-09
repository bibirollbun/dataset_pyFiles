import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier, 
                              StackingClassifier, BaggingClassifier,VotingClassifier)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv

warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col = 'id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')
sample_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_df.info()


train_df.head()


train_df.isnull().sum()


test_df.info()


test_df.head()


test_df.isnull().sum()


train_copy = train_df.copy()
test_copy = test_df.copy()


train_copy = train_copy.bfill()
test_copy = test_copy.bfill()


train_copy.duplicated().sum()


test_copy.duplicated().sum()


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Pie chart
train_copy['rainfall'].value_counts().plot.pie(
    autopct='%.2f%%',
    colors=['lightblue', 'grey'],
    explode=[0.1, 0.1], 
    radius=1.2,
    ax=axes[0]
)
axes[0].set_ylabel('')
    
# Bar chart
sns_ax = sns.countplot(train_copy, x='rainfall', palette = ['lightblue', 'grey'], ax= axes[1])
for label in sns_ax.containers:
    sns_ax.bar_label(label, fontsize = 10)

fig.suptitle('Rainfall Distributions')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


fig, axes  = plt.subplots(5,2, figsize=(8,12))
axes = axes.flatten()

for f, feat in enumerate(test_copy.columns[1:], start = 0) :
    sns.lineplot(train_copy, x='day', y=feat, hue='rainfall', palette=['lightblue', 'gray'], ax = axes[f])

    if f != 0:
        legend = axes[f].get_legend()
        if legend is not None:
            legend.remove()

    axes[f].set_title(f'{feat} over days', fontsize=12)


fig.tight_layout()

plt.show()


sns.pairplot(train_copy, hue='rainfall', palette=['lightblue', 'grey']);


def pre_processing(df):
    df['humidity_previous_day'] = df['humidity'].shift(1).fillna(0)
    df['humidity_change_overnight'] = df['humidity'] - df['humidity_previous_day']
    df['pressure_previous_day'] = df['pressure'].shift(1).fillna(0)
    df['pressure_change_overnight'] = df['pressure'] - df['pressure_previous_day']
    df['dew_humidity'] = df['dewpoint']*df['humidity']
    df['temp_gap'] = df['maxtemp'] - df['mintemp']
    df['wind_speeddirection'] = df['windspeed']*df['winddirection']
    df['cloud_windspeed'] = df['cloud']*df['windspeed']
    df['cloud_to_humidity'] = df['cloud']/df['humidity']
    df['temp_to_humidity'] = df['cloud']/df['humidity']
    df['temp_to_sunshine'] = df['sunshine']/df['temparature']
    df['month'] = pd.cut(df['day'], bins=12, labels=range(1, 13)).astype('int')
    df['temp_previous_day'] = df['temparature'].shift(1).fillna(0)
    df['temp_change_overnight'] = df['temparature'] - df['temp_previous_day']
    df['expected_day'] = df.index%365 + 1
    df = df.drop(columns=['maxtemp', 'mintemp', 'day'])
    X = df.copy()
    try:
        y = X.pop('rainfall')
        return X, y
    except:
        pass
        return X


X_train_full, y_train_full = pre_processing(train_copy)

# Prepare the test set
X_test = pre_processing(test_copy)
X_test.sample(5)


X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)


[data.shape for data in [X_train, X_val, y_train, y_val]]


features_trans = make_column_transformer(
    (StandardScaler(), X_train_full.select_dtypes('number').columns.tolist()),
    (OneHotEncoder(), X_train_full.select_dtypes(exclude='number').columns.tolist()),
    remainder='drop', 
    sparse_threshold=0)


lgb_params = { 'n_estimators': 130, 
               'max_depth': 2, 
               'num_leaves': 250, 
               'feature_fraction': 0.7557630763041963, 
               'bagging_fraction': 0.604841116587687, 
               'bagging_freq': 3, 
               'min_child_samples': 72}

cat_params = {'iterations': 270, 
              'learning_rate': 0.16211513182629325, 
              'objective': 'CrossEntropy', 
              'colsample_bylevel': 0.8069812365614417, 
              'random_strength': 0.2312285611887174, 
              'depth': 4, 
              'boosting_type': 'Ordered', 
              'bootstrap_type': 'Bayesian', 
              'bagging_temperature': 0.6887719860711248}

xgb_params = { 'n_estimators': 190, 
               'learning_rate': 0.017792963423540194, 
               'max_depth': 6, 
               'subsample': 0.2579692108675591, 
               'colsample_bytree': 0.2487767930540334, 
               'min_child_weight': 4}


Models = [
    ('lgb',make_pipeline(features_trans, LGBMClassifier(**lgb_params,verbose=-1))),
    ('cat',make_pipeline(features_trans, CatBoostClassifier(**cat_params, verbose=False))),
    ('xgb',make_pipeline(features_trans,XGBClassifier(**xgb_params))),
    ('hgb', make_pipeline(features_trans, HistGradientBoostingClassifier())),
    ('rfc', make_pipeline(features_trans, RandomForestClassifier())),
    ('etc', make_pipeline(features_trans, ExtraTreesClassifier()))
] 

scores = [] 
models = [] 
my_cv_2 = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in Models:
    
    # Cross validation
    cv_score = cross_val_score(model, 
                               X=X_train_full, 
                               y=y_train_full, 
                               cv=my_cv_2,
                               scoring='roc_auc'
                              )
    
    scores.append(cv_score) 
    models.append(model_name) 
    scores_df = pd.DataFrame(scores, 
                             columns=['cv1', 'cv2', 'cv3', 'cv4', 'cv5'], 
                             index=models) 

scores_df['avg_score'] = scores_df.mean(axis=1)
scores_df['std_score'] = scores_df.std(axis=1)
scores_df = scores_df.sort_values(by='avg_score', ascending=False)


display(
        (scores_df.style.background_gradient(cmap='YlGn', axis=0)
         .format('{:.5f}')
         .set_properties(**{'font-size': '12pt', 'weight': 'bold'}))
       )


plt.figure(figsize=(6,3))
sns.lineplot(scores_df.iloc[:, :-1].T, palette=['green', 'steelblue', 'maroon'], marker='o')
plt.ylabel('Scores')
plt.legend(loc='right', bbox_to_anchor=(1.25, 0.7),
          fancybox=True, shadow=True, ncol=1)
plt.show()


estimators = [
    ('cat', make_pipeline(features_trans, CatBoostClassifier())),
    ('lgb',make_pipeline(features_trans, LGBMClassifier(**lgb_params,verbose=-1))),
    ('cat_b',make_pipeline(features_trans, CatBoostClassifier(**cat_params, verbose=False))),
    ('xgb',make_pipeline(features_trans,XGBClassifier(**xgb_params))),
]

stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LGBMClassifier(**lgb_params, verbose=-1),
            n_jobs=-1
    )

stacking_model


%%time

kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
y_true, y_hat, y_hat_proba = list(), list(), list()
X, y = X_train_full, y_train_full

for f, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    stacking_model.fit(X_train, y_train)

    preds = stacking_model.predict(X_test)
    pred_proba = stacking_model.predict_proba(X_test)[:,1]

    y_true.extend(y_test)
    y_hat.extend(preds)
    y_hat_proba.extend(pred_proba)

oof_accu = accuracy_score(y_true, y_hat)
oof_auc = roc_auc_score(y_true, y_hat_proba)

print('oof_score: {:.5f}\nauc_score: {:.5f}'.format(oof_accu, oof_auc))


estimators = [
    ('rfc', RandomForestClassifier()),
    ('etc', ExtraTreesClassifier()),
    ('hgb', HistGradientBoostingClassifier()),
    ('cat', CatBoostClassifier()),
    ('lgb', LGBMClassifier(**lgb_params,verbose=-1)),
    ('cat_b', CatBoostClassifier(**cat_params, verbose=False)),
    ('xgb', XGBClassifier(**xgb_params)),
]


stacking_model = make_pipeline(features_trans,
    StackingClassifier(
            estimators=estimators,
            final_estimator=LGBMClassifier(**lgb_params, verbose=-1),
            n_jobs=-1
    ))

stacking_model


%%time

kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
y_true, y_hat, y_hat_proba = list(), list(), list()
X, y = X_train_full, y_train_full

for f, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    stacking_model.fit(X_train, y_train)

    preds = stacking_model.predict(X_test)
    pred_proba = stacking_model.predict_proba(X_test)[:,1]

    y_true.extend(y_test)
    y_hat.extend(preds)
    y_hat_proba.extend(pred_proba)

oof_accu = accuracy_score(y_true, y_hat)
oof_auc = roc_auc_score(y_true, y_hat_proba)

print('oof_score: {:.5f}\nauc_score: {:.5f}'.format(oof_accu, oof_auc))


stacking_model.fit(X_train_full, y_train_full)


preds = stacking_model.predict_proba(test_copy)[:,1]


sample_df['rainfall'] = preds

sample_df.to_csv('submission.csv', index=False)

