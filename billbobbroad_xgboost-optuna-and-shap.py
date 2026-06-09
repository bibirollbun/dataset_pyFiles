%pip install xgboost
%pip install pandas
%pip install scikit-learn
%pip install optuna
%pip install optuna-integration[xgboost]
%pip install shap


import xgboost
import pandas as pd
import os
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv(os.sep.join(['','kaggle','input','playground-series-s5e2',"train.csv"]))
df_test = pd.read_csv(os.sep.join(['','kaggle','input','playground-series-s5e2',"test.csv"]))


def cast_cols(df):
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].astype('category')
    return df

df_train = cast_cols(df_train)
df_test = cast_cols(df_test)


df_train.head()


df_train.select_dtypes("category").describe()


df_test.select_dtypes("category").describe()


df_train.describe(include=np.number)


df_test.describe(include=np.number)


feats = df_train.drop(columns=['id', 'Price']).columns
cat_feats = df_train.drop(columns=['id', 'Price']).select_dtypes('category').columns
num_feats = df_train.drop(columns=['id', 'Price']).select_dtypes('number').columns
print(cat_feats)
print(num_feats)


import matplotlib.pyplot as plt


plt.hist(df_train['Price'])


num_bins = 25

df_train0 = df_train.copy()

for col in feats:
    print(col)
    # calculate averages, standard errors, and counts
    if col in cat_feats:
        df_train0[col] = df_train0[col].cat.add_categories('Missing')
        df_train0 = df_train0.fillna({col: 'Missing'})
        df_plot = df_train0 \
            .groupby(col, observed=False) \
            .agg({'Price': ['mean', 'sem', 'count']})
    elif col in num_feats:
        if len(df_train[col].unique()) > num_bins:
            df_plot = df_train \
                .groupby(pd.cut(df_train[col], num_bins), observed=False) \
                .agg({'Price': ['mean', 'sem', 'count']})
            # get midpoint to plot by
            df_plot.index = pd.IntervalIndex(df_plot.index).mid
        else:
            df_plot = df_train \
                .groupby(col, observed=False) \
                .agg({'Price': ['mean', 'sem', 'count']})
        if df_train[col].isna().any():
            df_missing = df_train.loc[df_train[col].isna()] \
                .agg({'Price': ['mean', 'sem', 'count']})
    # plot
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.errorbar(
        df_plot.index, 
        df_plot['Price']['mean'], 
        yerr=df_plot['Price']['sem'],
        capsize=5
    )
    if col in num_feats and df_train[col].isna().any():
        ax.axhline(
            df_missing['Price']['mean'], linestyle='--', label='missing'
        )
        ax.axhline(
            df_missing['Price']['mean']-df_missing['Price']['sem'], linestyle=':', label='missing'
        )
        ax.axhline(
            df_missing['Price']['mean']+df_missing['Price']['sem'], linestyle=':', label='missing'
        )
    ax2.bar(
        df_plot.index, 
        df_plot['Price']['count'], 
        alpha=0.3
    )
    ax.set_xlabel(col)
    ax.set_ylabel('Price')
    ax2.set_ylabel('Count')
    plt.show()


def objective(trial):
    X_train = df_train.drop(columns=['id', 'Price'])
    y_train = df_train['Price']
    dtrain = xgboost.DMatrix(X_train, label=y_train, enable_categorical=True)
    
    param = {
        "tree_method": 'hist',
        "device": "cuda",
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "eta": trial.suggest_float("eta", 0.001, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    }
    
    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
    history = xgboost.cv(
        param, 
        dtrain,
        nfold=5,
        callbacks=[pruning_callback],
        verbose_eval=False,
        early_stopping_rounds=100,
        num_boost_round=5000
        # early_stopping_rounds=trial.suggest_int("early_stopping_rounds", 5, 1000, log=True),
        # num_boost_round=trial.suggest_int("num_boost_round", 100, 5000, log=True)
    )
    mean_rmse = history['test-rmse-mean'].values[-1]
    return mean_rmse


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=1000)


# best_param = study.best_params


# best_param


best_param = {
    "tree_method": 'hist',
    "device": "cuda",
    'lambda': 0.04645724641904437,
    'alpha': 0.43409173387974664,
    'max_depth': 3,
    'eta': 0.09992595785405518,
    'gamma': 2.4148048908398682e-08
    # 'early_stopping_rounds': 6,
    # 'num_boost_round': 4461
}
# early_stopping_rounds=best_param.pop('early_stopping_rounds')
# num_boost_round=best_param.pop('num_boost_round')
early_stopping_rounds=100
num_boost_round=5000


df_train1, df_eval = train_test_split(df_train, test_size=0.2)
dtrain = xgboost.DMatrix(
    df_train1.drop(columns=['id', 'Price']), 
    label=df_train1['Price'],
    enable_categorical=True
)
deval = xgboost.DMatrix(
    df_eval.drop(columns=['id', 'Price']), 
    label=df_eval['Price'],
    enable_categorical=True
)
dtest = xgboost.DMatrix(
    df_test.drop(columns=['id']), 
    enable_categorical=True
)


bst = xgboost.train(
    best_param, 
    dtrain,
    verbose_eval=True,
    evals=[(deval, 'validation')],
    early_stopping_rounds=early_stopping_rounds,
    num_boost_round=num_boost_round
)


preds = bst.predict(dtest)


df_out = pd.DataFrame({'id': df_test['id'], 'Price': preds})


df_out.head(10)


df_out.to_csv('submission.csv', index=False)


import shap


from sklearn.preprocessing import LabelEncoder

cat_cols = df_train.select_dtypes('category').columns
df_eval1 = df_eval.copy()
label_encoders = {cat_col: LabelEncoder() for cat_col in cat_cols}
for col, le in label_encoders.items():
    le.fit(df_train[col])
    df_eval1[col] = le.transform(df_eval1[col])
x_np = df_eval[bst.feature_names].to_numpy()
x_np_le = df_eval1[bst.feature_names].to_numpy()


# Get shap values
shap_and_base_values = bst.predict(deval, pred_contribs=True)

# Organize data
shap_values = shap_and_base_values[:, :-1]
base_values = shap_and_base_values[:, -1]

# Create an Explanation
explanation: shap.Explanation = shap.Explanation(
    values=shap_values, 
    base_values=base_values, 
    feature_names=bst.feature_names,
    data=x_np_le,
    display_data=x_np
)


shap.plots.beeswarm(explanation)


for feat in bst.feature_names:
    shap.plots.scatter(explanation[:, feat], color=explanation)




