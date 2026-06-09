from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv').drop(columns='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


df_train.info()


df_test.info()


df_train.y.value_counts()


categorical_columns = df_train.select_dtypes(include='object').columns
numerical_columns   = df_train.select_dtypes(exclude='object').columns


for cat_col in categorical_columns:
    # Determine the same order of categories for both y-values

    order = sorted(df_train[cat_col].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['blue', 'orange']
    for i, y_val in enumerate(sorted(df_train['y'].unique())):
        data = df_train[df_train['y'] == y_val][cat_col]
        counts = data.value_counts().reindex(order, fill_value=0)
        axes[i].bar(order, counts, color=colors[i])
        axes[i].set_title(f"{cat_col} (y = {y_val})")
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


for num_col in numerical_columns:
    if num_col == 'y':
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['blue', 'orange']
    for i, y_val in enumerate(sorted(df_train['y'].unique())):
        sns.violinplot(data=df_train[df_train['y'] == y_val], y=num_col, ax=axes[i], color=colors[i])
        axes[i].set_title(f"{num_col} (y = {y_val})")

    plt.tight_layout()
    plt.show()


df_train.loc[:, categorical_columns].nunique()


categorical_transformer = Pipeline(
    steps=[
        ('onehotenc', OneHotEncoder(handle_unknown='ignore'))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_columns)
    ],
    remainder='passthrough',
)


X = df_train.drop(columns='y')
y = df_train.y
X_test = df_test.drop(columns='id')


model_xgb = XGBClassifier(
    eval_metric='auc',
    device='cuda',
    tree_method='gpu_hist',
    n_gpus=1,
    verbosity=2,
)

my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model_xgb)
    ]
)

param_grid = {
    'model__n_estimators': [7500],
    'model__max_depth': [17],
    'model__learning_rate': [0.01],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8],
    'model__max_leaves': [125, 150],    
}

grid_search = GridSearchCV(
    estimator = my_pipeline,
    param_grid = param_grid,
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
    scoring = 'roc_auc',
    verbose=3,
    n_jobs=1,
    return_train_score=True,
)

grid_search.fit(
    X,
    y
)


xgb_best_params = {k: v for k, v in grid_search.best_estimator_.get_params().items()  if k.startswith('model__')}
xgb_best_params


xgb_submission = XGBClassifier(**xgb_best_params)

my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_submission)
    ]
)

my_pipeline.fit(X, y)


preds = my_pipeline.predict_proba(X_test)


df_submission['y'] = 1 - preds


df_submission.to_csv('submission.csv', index=False)

