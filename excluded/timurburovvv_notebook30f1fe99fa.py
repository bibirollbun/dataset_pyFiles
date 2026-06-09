pip install catboost


pip install association_metrics



#импорт библиотек

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import association_metrics as am
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import association_metrics as am
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_train.csv")


train_df.head()


train_df.describe()


plt.figure(figsize=(8, 6))
correlation_matrix = train_df.select_dtypes(include=['float64', 'int64']).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Кор матрица')
plt.tight_layout()
plt.show()


train_df.dropna(axis=0, inplace=True)


y_train = train_df['Rating']
X_train = train_df.drop('Rating', axis=1)


X_train


def clean(df):
    _df = df.copy()
    _df['Cocoa Percent_numeric'] = _df['Cocoa Percent'].str.rstrip('%').astype(float)
    _df['Broad Bean Origin_cat'] = _df['Broad Bean Origin'].str.split(',', expand=True)[0].replace('\xa0', np.nan)
    _df['Bean Type_cat'] = _df['Bean Type'].str.split(r'[ ,(]', expand=True)[0].replace('\xa0', np.nan)
    _df['Specific Bean Origin_cat'] = _df['Specific Bean Origin'].str.split(',', expand=True)[0].replace('\xa0', np.nan)
    _df = _df.drop(['REF', 'Cocoa Percent', 'Bean Type', 'Broad Bean Origin', 'Specific Bean Origin'], axis=1)
    return _df

X_train_cleaned = clean(X_train)


discretizer = KBinsDiscretizer(n_bins=11, encode='ordinal', strategy='kmeans', random_state=42)
discretizer.fit(X_train_cleaned[['Cocoa Percent_numeric']])

X_train_cleaned['Cocoa Percent_numeric_binned'] = discretizer.transform(X_train_cleaned[['Cocoa Percent_numeric']])
X_train_cleaned.drop('Cocoa Percent_numeric', axis=1, inplace=True)


X_train_cleaned['Cocoa Percent_numeric_binned'].value_counts()


XC = X_train_cleaned.apply(lambda x: x.astype("category") if x.dtype == "object" else x)

cramersv = am.CramersV(XC)
cramersv.fit()


X_train_cleaned.drop('Broad Bean Origin_cat', axis=1, inplace=True)


X_train_cleaned['Bean Type_cat'] = X_train_cleaned.groupby(['Specific Bean Origin_cat'])['Bean Type_cat'].transform(
    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)
)

X_train_cleaned['Bean Type_cat'] = X_train_cleaned['Bean Type_cat'].fillna('UNKNOWN')


X_train_cleaned.info()


X_train_cleaned


cat_features = [0, 2, 3, 4]


cat_features = [0, 2, 3, 4]
model = CatBoostRegressor(cat_features=cat_features)
model.fit(X_train_cleaned, y_train)


pip install optuna


import optuna
from sklearn.metrics import mean_squared_error
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'depth': trial.suggest_int('depth', 2, 5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
        'cat_features': cat_features,
        'random_state': 42
    }
    model = CatBoostRegressor(**params,thread_count=-1)
    model.fit(X_train_cleaned, y_train, use_best_model=True)

    preds = model.predict(X_train_cleaned)

    rmse = mean_squared_error(y_train, preds, )
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10,n_jobs=-1,timeout=600)  

print('Лучшее значение RMSE: {}'.format(study.best_value))
print('Лучшие параметры: ')
for key, value in study.best_params.items():
    print('    {}: {}'.format(key, value))

best_params = study.best_params
best_params['cat_features'] = cat_features
best_params['task_type'] = 'CPU'
best_params['random_state'] = 42
best_params['verbose'] = True

model = CatBoostRegressor(**best_params,thread_count=28)
model.fit(X_train_cleaned, y_train)


model.score(X_train_cleaned, y_train)


test_df = pd.read_csv("/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_test_new.csv")


test_df.info()


X_test_cleaned = clean(test_df)
X_test_cleaned['Cocoa Percent_numeric_binned'] = discretizer.transform(X_test_cleaned[['Cocoa Percent_numeric']])
X_test_cleaned.drop(['Broad Bean Origin_cat', 'Cocoa Percent_numeric'], axis=1, inplace=True)


X_test_cleaned['Bean Type_cat'] = X_test_cleaned.groupby(['Specific Bean Origin_cat'])['Bean Type_cat'].transform(
    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)
)

X_test_cleaned['Bean Type_cat'] = X_test_cleaned['Bean Type_cat'].fillna('UNKNOWN')


X_test_cleaned.info()


pred = model.predict(X_test_cleaned)


test_df['id'] = np.arange(len(test_df))
test_df['Rating'] = pred

test_df[['id','Rating']].to_csv("submission.csv", index=False)




