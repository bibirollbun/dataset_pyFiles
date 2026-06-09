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


%load_ext cudf.pandas
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


from sklearnex import patch_sklearn, config_context
patch_sklearn()


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
    # df['Brand']= df.groupby(['Style', 'Material','Size']).Brand.transform(lambda x : x.fillna(x.mode()[0]))
    # df['Size']= df.groupby(['Style', 'Material','Brand']).Size.transform(lambda x : x.fillna(x.mode()[0]))
    # df['Material']= df.groupby(['Style', 'Size','Brand']).Material.transform(lambda x : x.fillna(x.mode()[0]))
    # df['Style'] = df.groupby(['Material', 'Size','Brand']).Style.transform(lambda x : x.fillna(x.mode()[0]))
    # df['Weight Capacity (kg)'] = df.groupby(['Material', 'Size','Brand', 'Style'])['Weight Capacity (kg)'].transform(lambda x : x.fillna(x.mean()))
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
    df['Brand_Weight Capacity (kg)'] = df["Brand"] * df['Weight Capacity (kg)']
    df['Brand_material'] = df['Brand'].astype(str) + df['Material'].astype(str)
    df['Brand_style'] = df['Brand'].astype(str) + df['Style'].astype(str)
    df['materials_cost'] =  df['Material'].apply(lambda x: materials_cost[x] if x in materials_cost else 0)
    df['materials_cost_brand'] = df['materials_cost']* df['Brand']
    df['Style_Color'] =  df['Style']+  df['Color']
    df['Brand_Color'] = df['Brand'].astype(str)+df['Color']
    df['Color_Material'] = df['Color']+  df['Material']
    df['Brand_Weight Capacity (kg)_Size'] = df["Brand"] * df['Weight Capacity (kg)']*df['Size']
    df['Compartments__Size'] = df["Compartments"]*(df['Size'])
    df['Brand_water_proof']=df['Brand'] * df['Waterproof'].apply(lambda x : 0 if x=='No'else 1)
    df['material_cost_water_proof'] = df['materials_cost'] * df['Waterproof'].apply(lambda x : 0 if x=='No'else 1)
    df['Weight Capacity (kg)_material_cost'] = df['materials_cost']  * df['Weight Capacity (kg)']
    df['Weight Capacity (kg)_water_proof'] = df['Weight Capacity (kg)'] * df['Waterproof'].apply(lambda x : 0 if x=='No'else 1)
    df.drop(['materials_cost'], axis='columns',inplace=True)
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


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import KFold
class Custom_TargetEncoder(TransformerMixin):
    def __init__(self, except_col=[], cols=[], startegy="mean"):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []

    def fit(self, df, y=None):
        numerical_cols = df.select_dtypes(exclude=[np.number]).columns
        final_col =  numerical_cols.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        self.scaler = TargetEncoder(target_type="auto",smooth="auto", shuffle=False).fit(df[self.col], y)
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
        if self.strategy=="RBT":
            self.scaler = RobustScaler().fit(df[self.col]) 
        elif self.strategy=="STD" :
            self.scaler = StandardScaler().fit(df[self.col])
        else :
            self.scaler =  MinMaxScaler().fit(df[self.col]) 
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

    def fit(self, data, y=None):
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

    def transform(self, data,  y=None):
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
class CustomAggregationEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, groupby_col, agg_funcs, n_splits=5, random_state=None):
        self.groupby_col = groupby_col
        self.agg_funcs = agg_funcs
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold_aggregations_ = {}
        self.global_agg_ = None  # Stocker les moyennes globales

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X doit être un pandas.DataFrame.")
        
        self.kfold_aggregations_ = {}
        self.global_agg_ = {}  # Stocker les valeurs moyennes globales
        X = X.copy()

        for col, funcs in self.agg_funcs.items():
            for func in funcs:
                self.kfold_aggregations_[f"{col}_{func}"] = np.zeros(len(X))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

            # Agrégation KFold
            agg_values = X_train.groupby(self.groupby_col).agg(self.agg_funcs)
            agg_values.columns = [f"{col}_{func}" for col, funcs in self.agg_funcs.items() for func in funcs]
            agg_values.reset_index(inplace=True)

            # Merge et affectation
            X_val_merged = X_val.merge(agg_values, on=self.groupby_col, how='left')

            for col, funcs in self.agg_funcs.items():
                for func in funcs:
                    col_name = f"{col}_{func}"
                    if col_name in X_val_merged.columns:
                        self.kfold_aggregations_[col_name][val_idx] = X_val_merged[col_name].values
                    else:
                        raise KeyError(f"Colonne manquante après merge : {col_name}")

        # Stocker les moyennes globales pour transformer les nouveaux df
        self.global_agg_ = X.groupby(self.groupby_col).agg(self.agg_funcs)
        self.global_agg_.columns = [f"{self.groupby_col}_{func}" for col, funcs in self.agg_funcs.items() for func in funcs]
        self.global_agg_.reset_index(inplace=True)

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X doit être un pandas.DataFrame.")
        if self.global_agg_ is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant `transform`.")

        X_transformed = X.copy()

        # Merge avec les moyennes entraînées
        X_transformed = X_transformed.merge(self.global_agg_, on=self.groupby_col, how='left')

        # Remplir les valeurs manquantes avec la moyenne globale
        for col, funcs in self.agg_funcs.items():
            for func in funcs:
                col_name = f"{self.groupby_col}_{func}"
                global_mean = self.global_agg_[col_name].mean()
                X_transformed[col_name].fillna(global_mean, inplace=True)

        return X_transformed


('scaler', Custom_Scaler(cols=['Weight Capacity (kg)', 'Weight Capacity (kg)_compartments', 'Weight Capacity (kg)_Size', 'Brand_Weight Capacity (kg)_Size'], strategy='SDT'))


pipe = Pipeline([('label_encoding', MultiColumnLabelEncoder()), ('scaler', Custom_Scaler(cols=['Weight Capacity (kg)'], strategy='SDT'))])


transform_data  =  pipe.fit_transform(df)


X = transform_data.drop(['id', 'Price'], axis='columns')
y =transform_data['Price']


X.head()


X.columns


feateurs =  ['Weight Capacity (kg)', 'Weight Capacity (kg)_compartments', 'Weight Capacity (kg)_Size', 'Brand_Weight Capacity (kg)_Size']


STATS = ["mean","std","count","nunique","median","min","max","skew"]


agg_funcs ={
    "Price":STATS
}


# import optuna
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, accuracy_score

# feature_importances =[]
# def objective(trial):
#     params = {
#         "objective": "regression",
#         "metric": "rmse",
#         "boosting_type": "gbdt",   
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "num_leaves": trial.suggest_int("num_leaves", 20, 300),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
#         "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
#         "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
#         "verbose":-1,
#         "device":'gpu',
#         "gpu_platform_id":0,  # Change if needed
#         "gpu_device_id":0,  # First GPU (change for another)
#     }
#     from sklearn.model_selection import KFold
#     skf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#         encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean', seed=42)
#         for col in feateurs:
#             encoder.fit(X_train[col], y_train)
#             X_train[f"encoder{col}"] =  encoder.transform(X_train[col])
#             X_test[f"encoder{col}"] =  encoder.transform(X_test[col])
#         model = LGBMRegressor(**params)
#         # Entraînement
#         model.fit(X_train, y_train)
        
#         # Prédiction
#         y_pred = model.predict(X_test)
        
#         # Calcul du score
#         score = np.sqrt(mean_squared_error(y_test, y_pred))
#         print(score)
#         scores.append(score)
#         feature_importances.append(model.feature_importances_)
#         return score
#     # Afficher les résultats
#     print(f"Scores pour chaque fold : {scores}")
#     print(f"Score moyen : {np.mean(scores):.4f}±{np.std(scores)}")
#     return np.mean(scores) + np.std(scores)


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=10)
# print(study.best_value)
# print(study.best_params)


STATS = ["mean","std","count","nunique","median","min","max","skew"]


from sklearn.model_selection import KFold
def create_grouping_features(train,y_train, test, col, STATS=STATS):
    df  = train.copy()
    df['Price'] = y_train
    skf=KFold(n_splits=7, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(df):
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        temp = X_train.groupby(col)['Price'].agg(STATS)
        temp.columns  = [f"{col}_{s}" for s in STATS]
        X_train =  X_train.merge(temp, on=col, how='left')
        X_test = X_test.merge(temp, on=col, how='left')
        test = test.merge(temp, on=col, how='left')
        X_train = pd.concat([X_train,X_test])
        
        return X_train.drop('Price', axis='columns'), test


X.isna().sum()


X.max()


# import optuna

# def objective(trial):
#     params = {
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "lambda": trial.suggest_float("lambda", 0, 10),
#         "alpha": trial.suggest_float("alpha", 0, 10),
#         "devise":"cuda",
#         "tree_method":"hist"
#     }
#     from sklearn.model_selection import KFold
#     skf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#         encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean', seed=42)
#         for col in feateurs:
#             encoder.fit(X_train[col], y_train)
#             X_train[f"encoder{col}"] =  encoder.transform(X_train[col])
#             X_test[f"encoder{col}"] =  encoder.transform(X_test[col])
#         model = XGBRegressor(**params)
#         # Entraînement
#         model.fit(X_train, y_train)
                
#         # Prédiction
#         y_pred = model.predict(X_test)
        
#         # Calcul du score
#         score =np.sqrt(mean_squared_error(y_test, y_pred))
#         print(score)
#         return score
#         scores.append(score)
#     # Afficher les résultats
#     print(f"Scores pour chaque fold : {scores}")
#     print(f"Score moyen : {np.mean(scores):.4f}±{np.std(scores)}")
#     return np.mean(scores)+np.std(scores)


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=10)
# print(study.best_params)


cat_col =['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof',
       'Style', 'Color']


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        "task_type":"GPU",
        "verbose":0,
    }
    from sklearn.model_selection import TimeSeriesSplit
    tscv  = TimeSeriesSplit(n_splits= 5)
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
        model = CatBoostRegressor(**params)
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        # Calcul du score
        score = np.sqrt(mean_squared_error((y_test), (y_pred)))
        print(score)
        scores.append(score)
        feature_importances.append(model.feature_importances_)
        return score
        # Afficher les résultats
    print(f"Scores pour chaque fold : {scores}")
    print(f"Score moyen : {np.mean(scores):.4f}±{np.std(scores)}")
    return np.mean(scores) + np.std(scores)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print(study.best_params)


LGBM_params  ={'learning_rate': 0.02584498375236791, 'n_estimators': 623, 'max_depth': 12, 'num_leaves': 173, 'min_child_samples': 19, 'subsample': 0.5794348061092931, 'colsample_bytree': 0.6369914290828614, 'reg_alpha': 1.204601676692202, 'reg_lambda': 0.17291968913702077,"verbose":-1,
        "device":'gpu',
        "gpu_platform_id":0,  # Change if needed
        "gpu_device_id":0,  # First GPU (change for another)
              }
cat_params = {'iterations': 2157, 'learning_rate': 0.0768685087781287, 'depth': 5, 'l2_leaf_reg': 4.951585660297677, 'bagging_temperature': 0.9741076152604521,"task_type":"GPU","verbose":0 }


LGBM_params2  = {'learning_rate': 0.0165807949718273, 'n_estimators': 973, 'max_depth': 12, 'num_leaves': 83, 'min_child_samples': 70, 'subsample': 0.8677049647497916, 'colsample_bytree': 0.6503592461323768, 'reg_alpha': 0.011068989344831987, 'reg_lambda': 7.802715305217392,
        "device":'gpu',
        "gpu_platform_id":0,  # Change if needed
        "gpu_device_id":0,  # First GPU (change for another)
                }
XGB_params= {'max_depth': 5, 'learning_rate': 0.0994029949992508, 'n_estimators': 211, 'min_child_weight': 2.937598242197473, 'gamma': 4.391890056871294, 'subsample': 0.9322123252167951, 'colsample_bytree': 0.659280455272804, 'lambda': 0.3415615592380161, 'alpha': 6.076428545746548, "devise":"cuda", "tree_method":"hist"}
cat_params = {'iterations': 2001, 'learning_rate': 0.030704446269783442, 'depth': 6, 'l2_leaf_reg': 4.45626314753691, 'bagging_temperature': 0.9519166686882953, "task_type":"GPU",
        "verbose":0, }


test  = pd.read_csv('preproced_test.csv')


test_transform  =  pipe.transform(test)


test_transform.drop(['id'], axis='columns', inplace=True)


[col for col in test_transform.columns if col not in X.columns]


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor
skf = KFold(n_splits=5, shuffle=True, random_state=42)
prediction=[]
scores=[]
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
for col in feateurs:
        encoder.fit(X_train[col], y_train)
        X_train[f"encoder{col}"] =  encoder.transform(X_train[col])
        X_test[f"encoder{col}"] =  encoder.transform(X_test[col])
        test_transform[f"encoder{col}"] =  encoder.transform(test_transform[col])
model = LGBMRegressor(**LGBM_params2)
# model =  StackingRegressor(estimators=[('LGBM',LGBMRegressor(**LGBM_params2)),
                                          # ('XGB', XGBRegressor(**XGB_params))], final_estimator=Ridge(alpha=0.1))
    # Entraînement
model.fit(X_train, y_train)
    #Prédiction
    # y_pred1 = model.predict(X_test)

    # Calcul du score
    # score1 =  np.sqrt(mean_squared_error(y_test, y_pred1))
    # print(f"score model 1 : {score1}")
    # scores.append(score1)
prediction.append(model.predict(test_transform))
# Afficher les résultats
# print(f"Scores pour chaque fold model 1 : {scores}")
# print(f"Score moyen model 2: {np.mean(scores):.4f}±{np.std(scores)}")





model.feature_importances_


columns  = pd.DataFrame({"features" : model.feature_importances_, "columns": list(X.columns)+[f"encoder{col}" for col in feateurs]})


columns.sort_values('features')


submission  = pd.DataFrame([], columns=['id', 'Price'])
submission.id  =  test.id
submission['Price']  =np.mean((prediction), axis=0)


submission.head()


submission.to_csv('submission.csv', index=False)

