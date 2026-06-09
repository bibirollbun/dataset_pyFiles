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


def df_pre(df):
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['duration_cross_body_temp'] = df['Duration'] * df['Body_Temp']
    
    return df


train_df = df_pre(train_df)
x = train_df.drop([y_col, 'id'], axis = 1)
y = train_df[y_col]


seed = 42
n_split = 10


from sklearn.metrics import mean_squared_log_error
def rmsle_score(y, pred):
    y = np.expm1(y)
    pred = np.expm1(pred)
 
    y = np.maximum(0, y)
    pred = np.maximum(0, pred)
    
    return np.sqrt(mean_squared_log_error(y, pred))


from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

x_train, x_object, y_train, y_object = train_test_split(x, y, test_size=0.3, random_state=seed)
y_train = np.log1p(y_train)
y_object = np.log1p(y_object)


def try_model(trial):

    lgb_params = {
        'objective': 'regression',  
        'metric': 'rmse',  
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 50),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
        'random_seed': seed,  
        'verbose': -1
    }

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(x_train, y_train)

    y_pred = lgb_model.predict(x_object)
    

    score = rmsle_score(y_object, y_pred)
    return score


study = optuna.create_study(direction='minimize') 
study.optimize(try_model, n_trials=100, n_jobs=-1)

best_params = study.best_trial.params


def get_model(best_params):

    import lightgbm as lgb
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': best_params['lgb_num_leaves'],
        'learning_rate': best_params['lgb_learning_rate'],
        'feature_fraction': best_params['lgb_feature_fraction'],
        'random_seed': 42,
        'verbose': -1
    }

    lgb_model = lgb.LGBMRegressor(**lgb_params)

    return lgb_model


model_shap = get_model(best_params)
model_shap.fit(x, np.log1p(y))

import shap

explainer_lgb = shap.TreeExplainer(model_shap)
shap_values_lgb = explainer_lgb.shap_values(x)

shap_values_mean = np.mean(shap_values_lgb, axis=0)

shap.summary_plot(shap_values_lgb, x)



kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
scores = []
models = []
for train_index, val_index in kf.split(x):
    X_train, X_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)

    model = get_model(best_params)
    model.fit(X_train, y_train)
    
    y_pred_cv = model.predict(X_val)
    score = rmsle_score(y_val, y_pred_cv)

    scores.append(score)
    models.append(model)

print(f'Cross-validated RMSLE score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}')


score_df = pd.DataFrame(data = scores, columns = ["score"])
score_df


pred_id = test_df["id"]
test = test_df.drop('id', axis = 1)
test = df_pre(test)
submit_score = []

for num, model in enumerate(models):
    pred = model.predict(test)
    submit_score.append(pred)

test_pred = np.mean(np.expm1(submit_score), axis=0)


submission = pd.DataFrame({
    'id': pred_id,
    y_col: test_pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission










