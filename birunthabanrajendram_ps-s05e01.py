# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

import warnings
warnings.filterwarnings("ignore")


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.shape


train.info()


counts = train.groupby(["country","store","product"])["id"].count().rename('num_rows').reset_index()
display(counts)
counts_val_counts = counts["num_rows"].value_counts().rename("Count").reset_index().rename(columns={"index": "length"})
display(counts_val_counts)


train['num_sold'].isna().sum()


train = train.dropna().reset_index(drop=True)


counts = train.groupby(["country","store","product"])["num_sold"].count().rename("num_rows")
missing_data = counts.loc[counts != 2557]
missing_data_df = missing_data.reset_index()
missing_data_df["num_missing_rows"] = 2557 - missing_data_df["num_rows"]
missing_data_df


# train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')
# test['date'] = pd.to_datetime(test['date'], format='%Y-%m-%d')

# train['day_of_week'] = pd.to_datetime(train['date']).dt.dayofweek
# train['month'] = pd.to_datetime(train['date']).dt.month
# train['year'] = pd.to_datetime(train['date']).dt.year
# train['is_weekend'] = train['day_of_week'].isin([5, 6])

# test['day_of_week'] = pd.to_datetime(test['date']).dt.dayofweek
# test['month'] = pd.to_datetime(test['date']).dt.month
# test['year'] = pd.to_datetime(test['date']).dt.year
# test['is_weekend'] = test['day_of_week'].isin([5, 6])

print(min(train['date']), max(train['date']))

def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter.astype('float64')
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('float64')
    df['hour'] = df['date'].dt.hour.astype('float64')
    df['minute'] = df['date'].dt.minute.astype('float64')
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['sine_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    df['sine_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sine_year'] = np.sin(2 * np.pi * df['year']/7)
    df['cos_year'] = np.cos(2 * np.pi * df['year']/7)
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7

feature_engineering(train)
feature_engineering(test)


pd.set_option("display.max_columns", None)
# train = train.set_index('date')


train.head(10)


train.isna().sum()


agg_df = train.groupby(['date', 'store', 'product']).agg({'num_sold' : 'sum'}).reset_index().sort_values(['date', 'store', 'product'])
agg_df


stores = train['store'].unique()

for store in stores:
    store_data = train[train['store'] == store]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=store_data, x='product', y='num_sold', palette='viridis')
    plt.title(f'Sales Distribution for {store}', fontsize=14)
    plt.xlabel('Product', fontsize=12)
    plt.ylabel('Number Sold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


products = train['product'].unique()

for product in products:
    product_data = train[train['product'] == product]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=product_data, x='store', y='num_sold', palette='viridis')
    plt.title(f'Sales Distribution for {product}', fontsize=14)
    plt.xlabel('Store', fontsize=12)
    plt.ylabel('Number Sold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


daily_sales = train.groupby('date')['num_sold'].sum()
daily_sales.plot(figsize=(12, 6), title='Daily Sales Trends')


weekly_sales = train.groupby('day_of_week')['num_sold'].mean()
weekly_sales.plot(kind='bar', title='Average Sales by Day of the Week')


monthly_sales = train.groupby('month')['num_sold'].mean()
monthly_sales.plot(kind='bar', title='Average Sales by Month')


yearly_sales = train.groupby('year')['num_sold'].mean()
yearly_sales.plot(figsize=(12, 6), title='Yearly Sales Trends')


weekend_sales = train.groupby('is_weekend')['num_sold'].mean()
weekend_sales.plot(kind='bar', title='Weekend vs Weekday Sales')


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(daily_sales, model='multiplicative', period=365)
result.plot()


total_sales = agg_df.pivot(index='date',columns=['store', 'product'], values='num_sold')
total_sales.head()


for col in total_sales.columns:
    plt.plot(total_sales[col])
    plt.title(col)
    plt.show()


stores = total_sales.columns.get_level_values(0).unique()
products = total_sales.columns.get_level_values(1).unique()

fig, axes = plt.subplots(nrows=len(products), ncols=len(stores), figsize=(18, 20))

if len(stores) == 1:
    axes = [axes]
if len(products) == 1:
    axes = [axes]

for i, store in enumerate(stores):
    for j, product in enumerate(products):
        ax = axes[j][i]
        ax.plot(total_sales.index, total_sales[(store, product)], label=f"{product}")
        ax.set_title(f"{store} - {product}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Products Sold")
        ax.legend()

plt.tight_layout()
plt.show()


def missing_data(input_data):
    total = input_data.isnull().sum()
    percent = (input_data.isnull().sum()/input_data.isnull().count()*100)
    table = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    types = []
    for col in input_data.columns: 
        dtype = str(input_data[col].dtype)
        types.append(dtype)
    table["Types"] = types
    return(pd.DataFrame(table))

missing_data(total_sales)


train[train['num_sold'].isna()][['country', 'store', 'product']].value_counts()


# train = train.dropna().reset_index(drop=True)


plt.figure(figsize=(24, 6))
sns.lineplot(x=train['date'].dt.year, y=train['num_sold'], hue=train['country'], estimator='sum')
plt.title('Sales Trends by Country (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.grid()
plt.show()


agg_df_by_country = train.groupby(['date', 'country', 'product']).agg({'num_sold' : 'sum'}).reset_index().sort_values(['date', 'country', 'product'])
total_sales_by_country = agg_df_by_country.pivot(index='date',columns=['country', 'product'], values='num_sold')
total_sales_by_country.head()


# train = train.drop(columns=['id', 'date'])
lencoder = LabelEncoder()

for col in ['country', 'store', 'product']:
    train[col] = lencoder.fit_transform(train[col])
    test[col] = lencoder.fit_transform(test[col])

train['num_sold'] = np.log1p(train['num_sold'])

train.head(10)
# test.head(10)


train = train.drop(columns=['date'], axis=1)
test = test.drop(columns=['date'], axis=1)

X = train.drop(columns=['num_sold'])
y = train['num_sold']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBRegressor(
    n_estimators=3231,
    learning_rate=0.05895359669164567,
    max_depth=7,
    min_child_weight=4,
    subsample=0.8319649088461181,
    colsample_bytree=0.7107337151097438,
    gamma=0.0019772108405958213,
    reg_alpha=0.5384785820890761,
    reg_lambda=0.7912823880613118,
    objective='reg:squarederror',  # Correct regression objective
    random_state=42,
    n_jobs=-1,
    tree_method='gpu_hist',  # Enable GPU for histograms
    predictor='gpu_predictor',  # Enable GPU for prediction
    eval_metric='mape'  # Set evaluation metric to MAPE
)

catboost_model = CatBoostRegressor(
    n_estimators=1891,
    learning_rate=0.06761514972690001,
    depth=8,
    min_data_in_leaf=54,
    l2_leaf_reg=5.567375613813537,
    bagging_temperature=0.15478395184586632,
    random_strength=0.9462614107298501,
    loss_function='MAPE',
    eval_metric='MAPE',
    random_state=42,
    early_stopping_rounds=50  # Specify the GPU device (use '0' for the first GPU, or '1' for the second GPU, etc.)
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=3770,
    learning_rate=0.05038034487788465,
    max_depth=14,
    reg_alpha=0.20732364284443197,
    reg_lambda=0.004223724135505332,
    min_child_samples=29,
    colsample_bytree=0.6601202363535343,
    subsample=0.5597689123597346,
    objective='regression',  
    metric='mape',  
    n_jobs=-1,
    device='gpu'  # Enable GPU for LightGBM
)

meta_model = LinearRegression()


stacking_model = StackingRegressor(
    estimators=[('xgb1', xgb_model), ('xgb2', lgb_model), ('catboost', catboost_model)],
    final_estimator=meta_model,
    n_jobs=-1
)

stacking_model.fit(X, y)


submission_ids = test['id'] 
predictions = stacking_model.predict(test)

predictions = np.expm1(predictions)
# train.head(10)
# test.head(10)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})

submission.to_csv('submission5.csv', index=False)
print("File Saved!")
print(submission.head())


# # Prepare data for training
# X = train.drop(columns=['id', 'date', 'num_sold'])
# X = pd.get_dummies(X, drop_first=True)
# y = np.log1p(train['num_sold'])

# X_test = test.drop(columns=['id', 'date'])
# X_test = pd.get_dummies(X_test, drop_first=True)
# X_test = X_test.reindex(columns=X.columns, fill_value=0)

# from sklearn.model_selection import KFold
# from catboost import CatBoostRegressor

# # Initialize variables
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# train_predictions = np.zeros(len(train))
# mape_scores = []
# test_predictions_list = []

# # CatBoost Hyperparameters
# params = {
#     'iterations': 1000,             # Number of trees to build
#     'learning_rate': 0.1,           # Learning rate
#     'depth': 6,                     # Depth of the tree
#     'loss_function': 'MAPE',        # Loss function for regression
#     'cat_features': [],             # List of categorical features (if applicable)
#     'random_seed': 42,              # Random seed for reproducibility
#     'verbose': 200,                 # Print information every 200 iterations
# }


# # Train and validate model using 5-fold cross-validation
# for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # Train CatBoostRegressor
#     model = CatBoostRegressor(**params) #RandomForestRegressor(random_state=42, n_jobs=-1) #ExtraTreesRegressor(random_state=42, n_jobs=-1)  # #
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_val_pred = model.predict(X_val)
#     train_predictions[val_idx] = y_val_pred

#     # Calculate MAPE
#     mape = mean_absolute_percentage_error(y_val, y_val_pred)
#     mape_scores.append(mape)
#     print(f"Fold {fold}: MAPE = {mape:.4f}")

#     # Predict on test data for this fold
#     test_pred_fold = model.predict(X_test)
#     test_predictions_list.append(test_pred_fold)

# # Average test predictions across folds
# test_predictions_avg = np.mean(test_predictions_list, axis=0)
# test_predictions_avg = np.expm1(test_predictions_avg) 

# # Print training MAPE score
# print(f"Training MAPE score (5-fold average): {np.mean(mape_scores):.4f}")


from sklearn.base import BaseEstimator, RegressorMixin
import contextlib, io
import ydf; ydf.verbose(2)
from ydf import GradientBoostedTreesLearner

def YDFRegressor(learner_class):

    class YDFXRegressor(BaseEstimator, RegressorMixin):

        def __init__(self, params={}):
            self.params = params

        def fit(self, X, y):
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            target = y.name
            params = self.params.copy()
            params['label'] = target
            params['task'] = ydf.Task.REGRESSION
            X = pd.concat([X, y], axis=1)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.model = learner_class(**params).train(X)
            return self

        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                return self.model.predict(X)

    return YDFXRegressor


def build_model():
    
    x_input_cats = layers.Input(shape=(len(cat_features),))
    embs = []
    for j in range(len(cat_features)):
        e = layers.Embedding(t.cat_features_card[j], int(np.ceil(np.sqrt(t.cat_features_card[j]))))
        x = e(x_input_cats[:,j])
        x = layers.Flatten()(x)
        embs.append(x)
        
    x_input_nums = layers.Input(shape=(len(t.num_features),))
    
    x = layers.Concatenate(axis=-1)(embs+[x_input_nums]) 
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1)(x)
    
    model = keras.Model(inputs=[x_input_cats,x_input_nums], outputs=x)
    return model


# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor
# from sklearn.linear_model import RidgeClassifier, LogisticRegression, LinearRegression, Ridge, BayesianRidge
# from xgboost import DMatrix, XGBClassifier, XGBRegressor
# from lightgbm import log_evaluation, early_stopping, LGBMClassifier, LGBMRegressor, Dataset
# from catboost import CatBoostClassifier, CatBoostRegressor, Pool


# models = {
#     'NN': [_,
#            False],
#     'CAT': [CatBoostRegressor(**{'verbose': 0,
#                                  'random_state': 42,
#                                  'early_stopping_rounds': 200,
#                                  'eval_metric': "RMSE",
#                                  'n_estimators' : 2000,
#                                  'depth': 3,
#                                  'min_data_in_leaf': 96,
#                                  'l2_leaf_reg': 8.972890275248485,
#                                  'bagging_temperature': 0.18658249870341914, 
#                                  'random_strength': 0.14106593468982453,
#                                  'learning_rate': 0.01,
#                                 }),
#             False],
#     'CAT3': [CatBoostRegressor(**{'verbose': 0,
#                                   'random_state': 42,
#                                   'early_stopping_rounds': 200,
#                                   'eval_metric': "MAPE",
#                                   'n_estimators' : 2000,
#                                   'learning_rate': 0.01,
#                                   'depth': 8,
#                                   'min_data_in_leaf': 99,
#                                   'l2_leaf_reg': 7.7324870113971125, 
#                                   'bagging_temperature': 0.003232535109945575, 
#                                   'random_strength': 0.12145610701952099,
#                                  }),
#              False],
#     'CAT5': [CatBoostRegressor(**{'depth': 7,
#                                   'min_data_in_leaf': 59,
#                                   'l2_leaf_reg': 6.485681470975604, 
#                                   'bagging_temperature': 0.728613892125684,
#                                   'random_strength': 0.3565990691132947,
#                                   'verbose': 0,
#                                   'random_state': 42,
#                                   'early_stopping_rounds': 200,
#                                   'eval_metric': "MAPE",
#                                   'n_estimators' : 2000,
#                                   'learning_rate': 0.01,
#                                   "task_type": "GPU",
#                                  }),
#              False],
#     'CAT6': [CatBoostRegressor(**{'depth': 10,
#                                   'min_data_in_leaf': 67,
#                                   'l2_leaf_reg': 0.010658988402410939,
#                                   'bagging_temperature': 0.7381549501573549,
#                                   'random_strength': 0.10057316762567874,
#                                   'verbose': 0,
#                                   'random_state': 42,
#                                   'early_stopping_rounds': 200,
#                                   'eval_metric': "MAPE",
#                                   'n_estimators' : 2000,
#                                   'learning_rate': 0.01,
#                                   'bootstrap_type': 'Poisson',
#                                   "task_type": "GPU",
#                                  }),
#              False],
#     'XGB2': [XGBRegressor(**{'tree_method': 'hist',
#                              'n_estimators': 2000,
#                              'objective': 'reg:squarederror',
#                              'random_state': 42,
#                              'enable_categorical': True,
#                              'verbosity': 0,
#                              'early_stopping_rounds': 200,
#                              'eval_metric': 'rmse',
#                              'booster': 'gbtree', 
#                              'max_depth': 3,
#                              'min_child_weight': 16,
#                              'subsample': 0.8172380854733758, 
#                              'reg_alpha': 0.2734696712123178, 
#                              'reg_lambda': 0.5865768393479154,
#                              'colsample_bytree': 0.9766164536195251,
#                              'n_jobs': -1,
#                              'learning_rate': 0.01,
#                              'n_jobs': -1
#                             }),
#             False],
#     'XGB3': [XGBRegressor(**{'tree_method': 'hist',
#                              'n_estimators': 2000,
#                              'learning_rate': 0.01,
#                              'objective': 'reg:squarederror',
#                              'random_state': 42,
#                              'enable_categorical': True,
#                              'verbosity': 0,
#                              'early_stopping_rounds': 200,
#                              'eval_metric': 'mape',
#                              'booster': 'gbtree',
#                              'max_depth': 3,
#                              'min_child_weight': 12,
#                              'subsample': 0.7720667996291699, 
#                              'reg_alpha': 0.07869714859026081, 
#                              'reg_lambda': 0.9577219578640989, 
#                              'colsample_bytree': 0.9728085969282255, 
#                              'n_jobs': -1
#                            }),
#         False],
#     'XGB4': [XGBRegressor(**{'booster': 'gbtree',
#                              'max_depth': 3,
#                              'min_child_weight': 12,
#                              'subsample': 0.800221370346261,
#                              'reg_alpha': 0.4571249607822852,
#                              'reg_lambda': 0.6572354640280187,
#                              'colsample_bytree': 0.9982441671154363,
#                              'n_jobs': -1,
#                              'tree_method': 'hist',
#                              'n_estimators': 3000,
#                              'learning_rate': 0.01,
#                              'objective': 'reg:squarederror',
#                              'random_state': 42,
#                              'enable_categorical': True,
#                              'verbosity': 0,
#                              'early_stopping_rounds': 200,
#                              'eval_metric': 'mape',
#                              'booster': 'gbtree',
#                              "device": "cuda",
#                             }),
#              False],
#     'XGB5': [XGBRegressor(**{'booster': 'gbtree',
#                              'max_depth': 3,
#                              'min_child_weight': 19,
#                              'subsample': 0.8065343833518619,
#                              'reg_alpha': 0.3577049940509907,
#                              'reg_lambda': 0.8560297700871249,
#                              'colsample_bytree': 0.9866141987520272,
#                              'objective': 'reg:squarederror',
#                              'n_jobs': -1,
#                              'tree_method': 'hist',
#                              'n_estimators': 3000,
#                              'learning_rate': 0.01,
#                              'random_state': 42,
#                              'enable_categorical': True,
#                              'verbosity': 0,
#                              'early_stopping_rounds': 200,
#                              'eval_metric': 'mape', 
#                              "device": "cuda",
#                              }),
#              False],
#     'LGBM2': [LGBMRegressor(**{'random_state': 42,
#                                'early_stopping_round': 200,
#                                'verbose': -1,
#                                'boosting_type': 'gbdt',
#                                'eval_metric': 'rmse',
#                                'objective': 'regression_l2',
#                                'n_estimators': 5000,
#                                'max_depth': 13, 
#                                'num_leaves': 891, 
#                                'min_child_samples': 16,
#                                'min_child_weight': 11,
#                                'colsample_bytree': 0.48639630433139497,
#                                'reg_alpha': 0.45496760242817474,
#                                'reg_lambda': 0.9669296995303693,
#                                'learning_rate': 0.01
#                               }),
#              False],
#     'LGBM3': [LGBMRegressor(**{'random_state': 42,
#                                'early_stopping_round': 200,
#                                'verbose': -1,
#                                'boosting_type': 'gbdt',
#                                'eval_metric': 'rmse',
#                                'objective': 'regression_l2',
#                                'n_estimators': 2000,
#                                'max_depth': 6, 
#                                'num_leaves': 328,
#                                'min_child_samples': 10,
#                                'min_child_weight': 16,
#                                'colsample_bytree': 0.4893394195489041,
#                                'reg_alpha': 0.18334253987924942,
#                                'reg_lambda': 0.8328414321738785,
#                                'learning_rate': 0.01
#                               }),
#              False],
#     'LGBM4': [LGBMRegressor(**{'objective': 'regression_l2',
#                                'metric': 'mape', 
#                                'max_depth': 12, 
#                                'num_leaves': 878,
#                                'min_child_samples': 29,
#                                'min_child_weight': 14,
#                                'colsample_bytree': 0.49788260207319734, 
#                                'reg_alpha': 0.4747476308475839, 
#                                'reg_lambda': 0.6960820486441526,
#                                'random_state': 42,
#                                'early_stopping_round': 200,
#                                'verbose': -1,
#                                'boosting_type': 'gbdt',
#                                'eval_metric': 'mape',
#                                'objective': 'regression_l2',
#                                'n_estimators': 3000,
#                                'learning_rate': 0.01,
#                                }),
#               False],
#     'LGBM5': [LGBMRegressor(**{'objective': 'regression_l2',
#                                'metric': 'mape', 
#                                'max_depth': 7,
#                                'num_leaves': 123, 
#                                'min_child_samples': 21,
#                                'min_child_weight': 24,
#                                'colsample_bytree': 0.3641261996760593, 
#                                'reg_alpha': 0.03632800166349373, 
#                                'reg_lambda': 0.5287861861476272,
#                                'random_state': 42,
#                                'early_stopping_round': 200,
#                                'verbose': -1,
#                                'boosting_type': 'gbdt',
#                                'n_estimators': 3000,
#                                'learning_rate': 0.01,
#                                }),
#               False],
#     'LGBM6': [LGBMRegressor(**{'objective': 'regression_l2',
#                                'metric': 'mape',
#                                'max_depth': 6,
#                                'num_leaves': 502,
#                                'min_child_samples': 23,
#                                'min_child_weight': 18, 
#                                'colsample_bytree': 0.4714820876493163, 
#                                'reg_alpha': 0.054972003081022576, 
#                                'reg_lambda': 0.5774608955362155,
#                                'random_state': 42,
#                                'early_stopping_round': 200,
#                                'verbose': -1,
#                                'boosting_type': 'goss',
#                                'n_estimators': 3000,
#                                'learning_rate': 0.01,
#                               }),
#              False],
#     'LGBM7': [LGBMRegressor(**{'objective': 'regression_l2', 
#                                'metric': 'mape',
#                                'max_depth': 14,
#                                'num_leaves': 279,
#                                'min_child_samples': 7,
#                                'min_child_weight': 24, 
#                                'colsample_bytree': 0.43218993309765835,
#                                'reg_alpha': 0.42757392987472964,
#                                'reg_lambda': 0.9039762787446107,
#                                'random_state': 42,
#                                'early_stopping_round': 200,
#                                'verbose': -1,
#                                'boosting_type': 'goss',
#                                'n_estimators': 3000,
#                                'learning_rate': 0.01,
#                                }),
#               False],
#     'Ridge': [Ridge(tol=1e-2, max_iter=1000000,
#                     random_state=42),
#               False],
#     'BRidge': [BayesianRidge(tol=1e-2, n_iter=1000000),
#               False],
#     'LR': [LinearRegression(),
#               False],
#     'HGB': [HistGradientBoostingRegressor(**{'max_depth': 4,
#                                              'loss': 'squared_error',
#                                              'l2_regularization': 0.014082438341668873,
#                                              'min_samples_leaf': 39,
#                                              'max_leaf_nodes': 25,
#                                              'learning_rate': 0.01,
#                                              'max_iter': 2000,
#                                              'random_state': 42,
#                                              'early_stopping': 200,
#                                             }),
#               False],
#     'HGB2': [HistGradientBoostingRegressor(**{'max_depth': 4,
#                                               'loss': 'squared_error',
#                                               'l2_regularization': 1.0294569289519551e-05,
#                                               'min_samples_leaf': 12, 
#                                               'max_leaf_nodes': 26,
#                                               'learning_rate': 0.01,
#                                               'max_iter': 2000,
#                                               'random_state': 42,
#                                               'early_stopping': 200,
#                                              }),
#              False],
#     'HGB3': [HistGradientBoostingRegressor(**{'max_depth': 13, 
#                                               'loss': 'squared_error',
#                                               'l2_regularization': 0.05253480068908677,
#                                               'min_samples_leaf': 19,
#                                               'max_leaf_nodes': 40,
#                                               'learning_rate': 0.01,
#                                               'max_iter': 3000,
#                                               'random_state': 42,
#                                               'early_stopping': 200,
#                                              }),
#              False],
#     'HGB4': [HistGradientBoostingRegressor(**{'max_depth': 4, 
#                                               'loss': 'squared_error', 
#                                               'l2_regularization': 1.3248236291502028e-09,
#                                               'min_samples_leaf': 39,
#                                               'max_leaf_nodes': 29,
#                                               'learning_rate': 0.01,
#                                               'max_iter': 3000,
#                                               'random_state': 42,
#                                               'early_stopping': 200,
#                                              }),
#              False],
#     'YDF': [YDFRegressor(GradientBoostedTreesLearner)({'num_trees': 1000,
#                                                        'max_depth': 13,
#                                                        }),
#             False]
# }


# model = Model(X, X_enc, y, test, test_enc, models)
# TEST_preds = model.train()


# test['predicted_num_sold'] = test_predictions_avg
# print(test[['id', 'predicted_num_sold']])

# test[['id', 'predicted_num_sold']].to_csv('submission.csv', index=False)

