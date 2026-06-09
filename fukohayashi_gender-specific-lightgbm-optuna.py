import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna

import logging
logging.getLogger('optuna').setLevel(logging.ERROR)

import warnings
warnings.simplefilter("ignore")
pd.set_option('display.max_colwidth', None)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv' )


train_df


y_col = 'Calories'


train_df.columns


train_df.describe()


train_df.info()


test_df.info()


numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'Sex']

n_cols = 2
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

plt.figure(figsize=(12, 4 * n_rows))

for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    
    sns.kdeplot(data=train_df, x=col, hue='Sex', fill=True, common_norm=False, alpha=0.3)
    
    plt.title(f'Distribution of {col} by Sex')
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


numeric_df = train_df.select_dtypes(include='number')

sns.pairplot(numeric_df)
plt.suptitle("Scatter Plot Matrix", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# numeric_female_df = train_df.query('Sex == "female"').select_dtypes(include='number')

# sns.pairplot(numeric_female_df)
# plt.suptitle("Scatter Plot Matrix(female)", fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()


# numeric_male_df = train_df.query('Sex == "male"').select_dtypes(include='number')

# sns.pairplot(numeric_male_df)
# plt.suptitle("Scatter Plot Matrix(male)", fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()



bins = range(0, 101, 10)
labels = [i for i in range(0, 100, 10)]
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)

target_vars = list(train_df.drop(['id', 'Age', 'Sex', 'AgeGroup'],axis = 1).columns)

agg_df = train_df.groupby(['Sex', 'AgeGroup'])[target_vars].median().reset_index()

pivot_df = agg_df.pivot_table(index=['Sex', 'AgeGroup'], values=target_vars)

styled = pivot_df.style.background_gradient(cmap="Blues", axis=0)
train_df = train_df.drop('AgeGroup', axis=1)

styled



def df_pre(df):
    le = LabelEncoder()
    # df['Sex'] = le.fit_transform(df['Sex'])
    df['duration_cross_body_temp'] = df['Duration'] * df['Body_Temp']
    df['duration_cross_weight'] = df['Duration'] * df['Weight']

    # height_bins = range(120, 211, 10) 
    # height_labels = [i for i in range(120, 210, 10)]
    # df['HeightGroup'] = pd.cut(df['Height'], bins=height_bins, labels=height_labels, right=False)
    
    # median_weights = df.groupby('HeightGroup')['Weight'].median()
    
    # df['Weight_Median_by_HeightGroup'] = df['HeightGroup'].map(median_weights).astype(float)
    # df['Weight_Median_Diff'] = df['Weight'] / df['Weight_Median_by_HeightGroup']

    bins = range(0, 101, 10)
    labels = [i for i in range(0, 100, 10)]
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    median_rate = df.groupby('AgeGroup')['Heart_Rate'].median()
    
    df['Rate_Median_by_AgeGroup'] = df['AgeGroup'].map(median_rate).astype(float)
    df['HeartRate_Median_Diff'] = df['Heart_Rate'] / df['Rate_Median_by_AgeGroup']

    df['HeartRate_per_min'] = df['Heart_Rate'] / df['Duration']
    
    df = df.drop(['Rate_Median_by_AgeGroup', 'AgeGroup'], axis = 1)
    return df


def rmsle_score(y, pred):
    y = np.expm1(y)
    pred = np.expm1(pred)
    y = np.maximum(0, y)
    pred = np.maximum(0, pred)
    return np.sqrt(mean_squared_log_error(y, pred))



def get_model(best_params):
    return lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        num_leaves=best_params['lgb_num_leaves'],
        learning_rate=best_params['lgb_learning_rate'],
        feature_fraction=best_params['lgb_feature_fraction'],
        random_seed=seed,
        verbose=-1
    )


seed = 42
n_split = 10


train_df = df_pre(train_df) 
test_df = df_pre(test_df)


test_id = test_df["id"]


all_predictions = []


from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
import optuna
import shap

for sex_value in ['female', 'male']: 

    print(f"\n--- Processing for Sex = {sex_value} ---")

    train_sex = train_df[train_df['Sex'] == sex_value].copy()
    test_sex = test_df[test_df['Sex'] == sex_value].copy()

    y = train_sex[y_col]
    x = train_sex.drop([y_col, 'id', 'Sex'], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=seed)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    def try_model(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 50),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
            'random_seed': seed,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(x_train, y_train_log)
        pred_log = model.predict(x_val)
        return rmsle_score(y_val_log, pred_log)

    study = optuna.create_study(direction='minimize')
    study.optimize(try_model, n_trials=100, n_jobs=-1)

    best_params = study.best_trial.params

    model_shap = get_model(best_params)
    model_shap.fit(x, np.log1p(y))
    explainer = shap.TreeExplainer(model_shap)
    shap_values = explainer.shap_values(x)
    shap.summary_plot(shap_values, x)

    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    scores = []
    models = []
    for train_idx, val_idx in kf.split(x):
        x_tr, x_vl = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
        y_tr_log, y_vl_log = np.log1p(y_tr), np.log1p(y_vl)

        model = get_model(best_params)
        model.fit(x_tr, y_tr_log)
        pred_log = model.predict(x_vl)
        score = rmsle_score(y_vl_log, pred_log)

        scores.append(score)
        models.append(model)

    print(f"Sex = {sex_value} | CV RMSLE: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")

    test_x = test_sex.drop(['id', 'Sex'], axis=1)
    preds = [np.expm1(model.predict(test_x)) for model in models]
    preds_mean = np.mean(preds, axis=0)

    df_pred = pd.DataFrame({
        'id': test_sex['id'],
        y_col: preds_mean
    })

    all_predictions.append(df_pred)




submission = pd.concat(all_predictions).sort_values('id')
submission.to_csv('submission.csv', index=False)


submission










