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


import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import  TransformerMixin
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import random
from shapely.wkt import loads
from sklearn.metrics import mean_squared_error
from cuml.preprocessing import TargetEncoder


df1 = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df2 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test  =  pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df = pd.concat([df1, df2], axis='rows')


df.head()


df.isna().sum()


data_mapping={
    "Size":{
        "Small":2,
        "Medium":3,
        "Large":4,
        "Unknow":1
    }
}
materials_cost = {
    "Leather": 22,  # En euros par unité (exemple basé sur une paire de chaussures)
    "Canvas": 5,    # Estimation approximative
    "Nylon": 3,     # Coût moyen inférieur au cuir et à la toile
    "Polyester": 2,  # Généralement le moins cher
    "Unknow":1
}
brands_popularity = {
    "Nike": 1,
    "Adidas": 2,
    "Puma": 3,
    "Under Armour": 4,
    "Jansport": 5,
    "Unknow": 6
}


fill_na_mapping={
    "Brand":"Unknow",
    "Material":"Unknow",
    "Size":"Unknow",
    "Laptop Compartment":"Unknow",
    "Waterproof":"Unknow",
    "Style":"Unknow",
    "Color":"Unknow",
    "Weight Capacity (kg)":'Mean'
}


def fill_na_value(data):
    df = data.copy()
    df["nan_count"] = df.isna().sum(axis=1)
    for col in fill_na_mapping:
        if fill_na_mapping[col] ==  'Mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif fill_na_mapping[col]=="Mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(fill_na_mapping[col], inplace=True)
    return df
def mapping_data(data):
    df  = data.copy()
    df['Size'] = df['Size'].apply(lambda x : data_mapping['Size'][x])
    df['Brand'] = df['Brand'].apply(lambda x : brands_popularity[x])
    return df

def create_feature(data):
    df = data.copy()
    df['Compartments_Size'] = df["Compartments"]/(df['Size']+1)
    df["Weight Capacity (kg)_compartments"]= df['Weight Capacity (kg)']/(df['Compartments']+1)
    df['Compartments_Size_water_proof'] = df['Compartments_Size']* df['Waterproof'].apply(lambda x : 0 if x=='No'else 1)
    df['Weight Capacity (kg)_Size'] = df['Weight Capacity (kg)']/(df['Size']+1)
    df['Brand_Weight Capacity (kg)_Size'] = df["Brand"] * df['Weight Capacity (kg)']
    df['Brand_material'] = df['Brand'].astype(str) + df['Material'].astype(str)
    df['Brand_style'] = df['Brand'].astype(str) + df['Style'].astype(str)
    df['materials_cost'] =  df['Material'].apply(lambda x: materials_cost[x] if x in materials_cost else 0)
    df['materials_cost_brand'] = df['materials_cost']* df['Brand']
    df['Style_Color'] =  df['Style']+  df['Color']
    df['Brand_Color'] = df['Brand'].astype(str)+df['Color']
    df['Color_Material'] = df['Color']+  df['Material']
    df.drop(['materials_cost', 'Brand'], axis='columns',inplace=True)
    return df


df =  fill_na_value(df)
test  =  fill_na_value(test)


df =  mapping_data(df)
test =  mapping_data(test)


df = create_feature(df)
test =  create_feature(test)


df.to_csv('preproced_train.csv', index=False)
test.to_csv('preproced_test.csv', index=False)


df = pd.read_csv('preproced_train.csv')


df.head()


# numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])

# num_cols = 3  # Number of columns in the grid
# num_rows = (len(numerical_cols) + num_cols - 1) // num_cols

# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
# axes = axes.flatten()

# for i, col in enumerate(numerical_cols):
#     sns.histplot(df[col], ax=axes[i], kde=True)
#     axes[i].set_title(col)

#     ax_box = axes[i].inset_axes([0.2, -0.3, 0.6, 0.2])  # [x, y, width, height]
#     sns.boxplot(x=df[col], ax=ax_box, orient='h')
#     ax_box.set(xlabel='')
# for j in range(len(numerical_cols), len(axes)):
#     fig.delaxes(axes[j])

# plt.show()



df_cat = df.select_dtypes(exclude=np.number)


# num_cols = 3  # Number of columns in the grid
# num_rows = (len(df_cat.columns) + num_cols - 1) // num_cols

# # Create the subplots
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
# axes = axes.flatten()
# palette = sns.color_palette("Set2", len(df_cat.iloc[:, 0].value_counts()))

# for i, col in enumerate(df_cat.columns):
#     df_cat[col].value_counts().plot(kind='bar', ax=axes[i], color=palette)
#     axes[i].set_title(col)
#     axes[i].tick_params(axis='x', rotation=45, labelsize=8)
# for j in range(len(df_cat.columns), len(axes)):
#     fig.delaxes(axes[j])

# plt.show()



df = pd.read_csv('preproced_train.csv')


df.head()


class Custom_TargetEncoder(TransformerMixin):
    def __init__(self, except_col=[], cols=[], startegy="mean"):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []

    def fit(self, df, y=None):
        numerical_cols = df.select_dtypes(exclude=[np.number]).columns
        final_col =  numerical_cols.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        self.scaler = TargetEncoder(target_type="continuous",smooth="auto", shuffle=False).fit(df[self.col], y)
        return self
    
    def transform(self, data, y=None):
        df =data.copy()
        scaler_data =  self.scaler.transform(df[self.col])
        scaler_data_df = pd.DataFrame(scaler_data, columns=['target_encode_'+ col for col in self.col], index=df.index)
        others_cols  =  df.columns
        return pd.concat([scaler_data_df, df[others_cols]], axis='columns')

class Custom_Scaler(TransformerMixin):
    def __init__(self, except_col=[], cols=[], strategy="MinMax"):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []
        self.strategy = strategy

    def fit(self, df, y=None):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        final_col =  numerical_cols.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        self.scaler = MinMaxScaler().fit(df[self.col]) if self.strategy=="MinMax" else StandardScaler().fit(df[self.col])
        return self
    
    def transform(self, data, y=None):
        df =data.copy()
        scaler_data =  self.scaler.transform(df[self.col])
        scaler_data_df = pd.DataFrame(scaler_data, columns=self.col, index=df.index)
        others_cols  =  df.columns.difference(self.col)
        return pd.concat([scaler_data_df, df[others_cols]], axis='columns')

class CustomOneHotEncoder(TransformerMixin):
    def __init__(self, except_col=[], cols=[]):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []

    def fit(self, data):
        df =  data.copy()
        cat_col =  df.select_dtypes(exclude=[np.number]).columns
        final_col =  cat_col.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist'), self.col)
            ],
            remainder='passthrough'  # To keep other columns unchanged
        )
        self.preprocessor =  preprocessor
        self.preprocessor.fit(df[self.col])
        return self

    def transform(self, data):
        df =  data.copy()
        final_data_encoded =  self.preprocessor.transform(df[self.col])
        feature_names = (self.preprocessor
                        .named_transformers_['cat']
                        .get_feature_names_out(self.col))
        final_data_encoded_df = pd.DataFrame(final_data_encoded.toarray() if type(final_data_encoded)!=np.ndarray else final_data_encoded, columns=feature_names, index=df.index)
        others_col =  df.columns.difference(self.col)
        final_df  = pd.concat([df[others_col], final_data_encoded_df], axis='columns')
        return final_df

class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self, except_col=[]):
        self.except_col = except_col
        self.label_encoders = defaultdict(LabelEncoder)

    def fit(self,X , y=None):
        df  = X.copy()
        cat_col =  df.select_dtypes(exclude=[np.number]).columns
        final_col =  cat_col.difference(self.except_col)
        self.columns = final_col
        for col in self.columns:
            self.label_encoders[col]
            self.label_encoders[col].fit(df[col])
        return self

    def transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = X_copy[col].apply(lambda s: '<unknown>' if s not in self.label_encoders[col].classes_ else s)
            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, '<unknown>')
            X_copy[col] = self.label_encoders[col].transform(X_copy[col])
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = self.label_encoders[col].inverse_transform(X_copy[col])
        return X_copy


pipe = Pipeline([('label_encoding', MultiColumnLabelEncoder()), ('scaler', Custom_Scaler(cols=['Weight Capacity (kg)', 'Weight Capacity (kg)_compartments', 'Weight Capacity (kg)_Size', 'Brand_Weight Capacity (kg)_Size'], strategy='SDT'))])


transform_data  =  pipe.fit_transform(df)


X = transform_data.drop(['id', 'Price'], axis='columns')
y =transform_data['Price']


feateurs =  ['Weight Capacity (kg)']


encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')


encoder.fit(X[feateurs], y)


encoder.transform(X[feateurs])


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

feature_importances =[]
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",   
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "verbose":-1,
        "device":'gpu',
        "gpu_platform_id":0,  # Change if needed
        "gpu_device_id":0,  # First GPU (change for another)
    }
    from sklearn.model_selection import KFold
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
        for col in feateurs:
            encoder.fit(X_train[col], y_train)
            X_train[f"encoder{col}"] =  encoder.transform(X_train[col])
            X_test[f"encoder{col}"] =  encoder.transform(X_test[col])
        model = LGBMRegressor(**params)
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        
        # Calcul du score
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        print(score)
        scores.append(score)
        return scores
        feature_importances.append(model.feature_importances_)
    # Afficher les résultats
    print(f"Scores pour chaque fold : {scores}")
    print(f"Score moyen : {np.mean(scores):.4f}±{np.std(scores)}")
    return np.mean(scores) + np.std(scores)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print(study.best_value)
print(study.best_params)


LGBM_params  ={'learning_rate': 0.020616953181593577, 'n_estimators': 808, 'max_depth': 14, 'num_leaves': 27, 'min_child_samples': 97, 'subsample': 0.5008333169558032, 'colsample_bytree': 0.7282420581390141, 'reg_alpha': 0.15829550223358652, 'reg_lambda': 0.7256380159097973,
        "device":'gpu',
        "gpu_platform_id":0,  # Change if needed
        "gpu_device_id":0,  # First GPU (change for another)
              }


test  = pd.read_csv('preproced_test.csv')


test_transform  =  pipe.transform(test)


test_transform.drop(['id'], axis='columns', inplace=True)


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import StackingRegressor
skf = KFold(n_splits=5, shuffle=True, random_state=42)
prediction=[]
scores=[]
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
    for col in feateurs:
        encoder.fit(X_train[col], y_train)
        X_train[f"encoder{col}"] =  encoder.transform(X_train[col])
        X_test[f"encoder{col}"] =  encoder.transform(X_test[col])
        test_transform[f"encoder{col}"] =  encoder.transform(test_transform[col])
    model = LGBMRegressor(**LGBM_params)
    # Entraînement
    model.fit(X_train, y_train)
    #Prédiction
    y_pred1 = model.predict(X_test)

    # Calcul du score
    score1 =  np.sqrt(mean_squared_error(y_test, y_pred1))
    print(f"score model 1 : {score1}")
    scores.append(score1)
    prediction.append(model.predict(test_transform))
# Afficher les résultats
print(f"Scores pour chaque fold model 1 : {scores}")
print(f"Score moyen model 2: {np.mean(scores):.4f}±{np.std(scores)}")


model.feature_importances_


columns  = pd.DataFrame({"features" : model.feature_importances_, "columns": list(X.columns)+[f"encoder{col}" for col in feateurs]})


columns.sort_values('features')


submission  = pd.DataFrame([], columns=['id', 'Price'])
submission.id  =  test.id
submission['Price']  =np.mean((prediction), axis=0)


submission.head()


submission.to_csv('submission.csv', index=False)

