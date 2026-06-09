

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score

import eli5
from eli5.sklearn import PermutationImportance
     


path='/kaggle/input/playground-series-s5e2/'
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


target = 'Price'
num_feat = train.select_dtypes(np.number).columns
cat_feat = train.select_dtypes(exclude=np.number).columns

train.head()


train.shape


train.columns.values


train[target].hist(bins=200)


train[target].describe()


train['Brand'].unique()


train.groupby('Brand')[target].mean()


train.groupby('Brand')[target].agg(np.mean).plot(kind='bar')


train['Material'].unique()


train.groupby('Material')[target].mean()


def group_and_barplot (feat_groupby, feat_agg = 'Price', agg_funcs = [np.mean, np.median, np.size], feat_sort = 'mean', top = 50, subplots = True):
    return (
        train
        .groupby(feat_groupby)[feat_agg]
        .agg(agg_funcs)
        .sort_values(by=feat_sort, ascending=False)
        .head(top)   
).plot(kind='bar', figsize=(5, 5), subplots = subplots)


group_and_barplot('Brand', feat_sort='size');


group_and_barplot('Material', feat_sort='size');


'''to_plot =[ 'Size', 'Compartments',
       'Laptop Compartment', 'Waterproof', 'Style', 'Color',
       'Weight Capacity (kg)']'''


'''for i in to_plot:
    group_and_barplot(i, feat_sort='size');'''


train.select_dtypes(np.number).columns #Najpierw sprawdzamy, które kolumny zawierają wartości numeryczne.



feats = ['id', 'Compartments', 'Weight Capacity (kg)']
X = train[feats].values
y = train['Price'].values

model = DummyRegressor()
model.fit(X, y)
y_pred = model.predict(X)

mae(y, y_pred)


train.head()


train.info()


num_feat = train.select_dtypes(np.number).columns
cat_feat = train.select_dtypes(exclude=np.number).columns


for feat in cat_feat:
    train[feat]=train[feat].fillna("other")
    test[feat]=test[feat].fillna('other')


test.info()


train['Weight Capacity (kg)'] = train.groupby(["Brand","Material","Size","Compartments","Laptop Compartment"])['Weight Capacity (kg)'].transform(lambda x: x.fillna(x.median()))
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median())
train['Weight Capacity (kg)'].isna().sum()




test['Weight Capacity (kg)'] = test.groupby(["Brand","Material","Size","Compartments","Laptop Compartment"])['Weight Capacity (kg)'].transform(lambda x: x.fillna(x.median()))
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].median())
test['Weight Capacity (kg)'].isna().sum()


def transform_categoricals(df_, categorical_cols):
    return pd.get_dummies(df_, columns=categorical_cols)


def factorize(df):
    SUFFIX_CAT = '__cat' #oznaczamy zmienne kategorialne
    for feat in df.columns:
        if isinstance (df[feat][0], list): continue #jeżeli wartość jest listą to nic nie rób, bez tego funkcja zgłasza błąd
  
        factorized_values = df[feat].factorize()[0]
        if SUFFIX_CAT in feat: #jeżeli nazwa kolumny zawiera już __cat
            df[feat] = factorized_values #przypisz tę samą wartość
        else: 
            df[feat+SUFFIX_CAT] = factorized_values # w przeciwnym przypadku dodaj __cat
    return df
     


y_train = train[target]
df_train = train.drop([target], axis=1)
df_all = pd.concat([train, test], axis=0)


for c in cat_feat:
    print(c, "n unique:",df_all[c].nunique())


df_all = transform_categoricals(df_all, cat_feat)
df_all.columns


df_all = df_all.drop('id', axis = 1)


from sklearn.model_selection import KFold
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

# Assuming df_all, y_train, and df_train are already defined
X_train = df_all[:df_train.shape[0]]
X_test = df_all[df_train.shape[0]:]

# Prepare arrays to store out-of-fold predictions and test set predictions
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = LGBMRegressor()
    model.fit(X_tr, y_tr)
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_lgbm = test_preds


from xgboost import XGBRegressor
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_XGB = test_preds



from catboost import CatBoostRegressor 


oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])


for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        subsample=0.8,
        colsample_bylevel=0.8,
        loss_function='RMSE',
        random_seed=42,
        eval_metric='RMSE',
        verbose=100,
        early_stopping_rounds=100
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        #cat_features=categorical_features,
        use_best_model=True
    )
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_CATB = test_preds


df_train


train_factor = factorize(train)



train_factor.columns
cat_cols = ['Brand__cat', 'Material__cat', 'Size__cat',
       'Compartments__cat', 'Laptop Compartment__cat', 'Waterproof__cat',
       'Style__cat', 'Color__cat', 'Weight Capacity (kg)__cat']



X = train[cat_cols].values
y = train['Price'].values

model = DecisionTreeRegressor (max_depth=5)
scores = cross_val_score(model, X, y, scoring = 'neg_mean_absolute_error')
np.mean(scores)


m = DecisionTreeRegressor(max_depth=5)  #Tworzymy nowy model, żeby poznać ważność cech
m.fit(X, y)

imp = PermutationImportance(m, random_state=0).fit(X, y)
eli5.show_weights (imp, feature_names=cat_cols)


y_pred = (y_pred_XGB + y_pred_lgbm+ y_pred_CATB)/3

ssub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
ssub['Price'] = y_pred
ssub.to_csv('submission.csv', index = False)


ssub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


ssub




