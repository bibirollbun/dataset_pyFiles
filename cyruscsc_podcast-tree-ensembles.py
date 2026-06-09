SEED = 42
MIN_EPI_LEN = 0
MAX_EPI_LEN = 150
MIN_NUM_ADS = 0
MAX_NUM_ADS = 5


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def check_health(df):
    nans = df.isna().sum().sum()
    dups = df.duplicated().sum()

    return nans, dups

def remove_outliers(df, col, min_val, max_val):
    mask = (df[col] >= min_val) & (df[col] <= max_val)

    return df[mask]

def clip_outliers(df, col, min_val, max_val):
    df_res = df.copy()
    df_res[col] = df_res[col].apply(lambda x: np.clip(x, min_val, max_val))
    
    return df_res

def impute_values(df, num_strategy='median', cat_strategy='most_frequent'):
    df_res = df.copy()
    num_cols = df_res.select_dtypes(include=['number']).columns
    cat_cols = df_res.select_dtypes(exclude=['number']).columns
    
    if len(num_cols):
        num_imputer = SimpleImputer(strategy=num_strategy)
        df_res[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    if len(cat_cols):
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        df_res[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df_res


import pandas as pd

df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_train = remove_outliers(df_train, 'Episode_Length_minutes', MIN_EPI_LEN, MAX_EPI_LEN)
df_train = remove_outliers(df_train, 'Number_of_Ads', MIN_NUM_ADS, MAX_NUM_ADS)
df_train = impute_values(df_train)

nans, dups = check_health(df_train)
print(f"nan values: {nans}, duplicated rows: {dups}")

df_train.info()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, figsize=(20, 2)):
    n = len(df.select_dtypes(include=['number']).columns)
    
    fig, axis = plt.subplots(1, n, figsize=figsize)
    df.hist(ax=axis, edgecolor='black', grid=False)

def plot_correlations(df, figsize=(4, 4)):
    df_num = df.select_dtypes(include=['number'])

    plt.figure(figsize=figsize)
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.show()


plot_histograms(df_train)
plot_correlations(df_train)


import pandas as pd

def one_hot_encode(df, cols):
    df_res = pd.get_dummies(df, columns=cols)

    return df_res

def target_encode(df, cols, target, smoothing=2):
    df_res = df.copy()
    global_mean = df_res[target].mean()
    encodings = {}

    for col in cols: 
        agg = df.groupby(col)[target].agg(['mean', 'count'])
        counts = agg['count']
        means = agg['mean']

        weight = counts / (counts + smoothing)
        encoding = weight * means + (1 - weight) * global_mean

        df_res[col] = df_res[col].map(encoding)

        encodings[col] = encoding
        
    return df_res, encodings, global_mean

def apply_target_encodings(df, cols, encodings, default_value):
    df_res = df.copy()

    for col in cols:
        df_res[col] = df_res[col].map(encodings[col]).fillna(default_value)

    return df_res


df_train = one_hot_encode(df_train, ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])
df_train, target_encodings, target_global_mean = target_encode(df_train, ['Podcast_Name', 'Episode_Title'], 'Listening_Time_minutes')

df_train.info()


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def split_x_y(df, target, ignore):
    X = df.drop([target] + ignore, axis=1)
    y = df[target]

    return X, y

def train_models(X, y, models, random_state=SEED):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {
            'model': model,
            'rmse': rmse,
        }

        print(f"{name} - rmse: {results[name]['rmse']:.4f}")

    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    
    return best_model_name, results

def optimize_model(X, y, model):
    model.fit(X, y)
    
    return model


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'RFR': RandomForestRegressor(random_state=SEED),
    'GBR': GradientBoostingRegressor(random_state=SEED),
}

X, y = split_x_y(df_train, 'Listening_Time_minutes', ['id'])
best_model_name, results = train_models(X, y, models)
final_model = optimize_model(X, y, models[best_model_name])


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def tune_hyperparameters(X, y, model, param_grid, random_state=SEED):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X, y)

    best_rmse = np.sqrt(-grid_search.best_score_)
    
    print(f"best parameters: {grid_search.best_params_}")
    print(f"best rmse: {best_rmse:.4f}")
    
    return grid_search


param_grids = {
    'RFR': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GBR': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# tuned_model = tune_hyperparameters(X, y, models[best_model_name], param_grids[best_model_name])
# final_model = tuned_model.best_estimator_


import pandas as pd

# data preparation

df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_test = clip_outliers(df_test, 'Episode_Length_minutes', MIN_EPI_LEN, MAX_EPI_LEN)
df_test = clip_outliers(df_test, 'Number_of_Ads', MIN_NUM_ADS, MAX_NUM_ADS)
df_test = impute_values(df_test)

# feature engineering

df_test = one_hot_encode(df_test, ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])
df_test = apply_target_encodings(df_test, ['Podcast_Name', 'Episode_Title'], target_encodings, target_global_mean)

# prediction

pred = final_model.predict(df_test.drop('id', axis=1))
df_test['Listening_Time_minutes'] = pred

df_test['id'] = df_test['id'].astype(int)
df_test[['id', 'Listening_Time_minutes']].to_csv('submission.csv', index=False)




