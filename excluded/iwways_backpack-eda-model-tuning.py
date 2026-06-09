import numpy as np 
import pandas as pd 
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import optuna


X_train = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
X_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_subm = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


X_train


X_train.info()


X_train.isna().sum()


X_test.isna().sum()


X_train.describe(include='all')


fig, axs = plt.subplots(4,3, figsize=(20,20))
numerical_features = X_train.select_dtypes(exclude='object').columns
r = 0
for feat in numerical_features:
    c = 0
    while c<3:
        if c == 0:
            sns.kdeplot(ax=axs[r, c], data=X_train, x=feat)
            axs[r,c].set_title(f'Density Plot of {feat}')
            axs[r,c].set(xlabel=None)
        elif c == 1:
            sns.violinplot(ax=axs[r,c], data=X_train, x=feat)
            axs[r,c].set_title(f'Violin Plot of {feat}')
            axs[r,c].set(xlabel=None)
        elif c == 2:
            sns.boxplot(ax=axs[r,c], data=X_train, x=feat)
            axs[r,c].set_title(f'Box Plot of {feat}')
            axs[r,c].set(xlabel=None)
        c += 1
    r += 1


fig, axs = plt.subplots(4,2, figsize=(20,30))
cat_features = X_train.select_dtypes(include='object').columns
r = 0
c = 0
for feat in cat_features:
    if c == 2:
        c = 0
        r += 1
    counts = sns.countplot(ax=axs[r,c], data=X_train, x=feat)
    for p in counts.patches:
        counts.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    
    axs[r,c].set_title(f'{feat}')
    axs[r,c].set(xlabel=None)

  
    c += 1


sns.heatmap(X_train[numerical_features].corr(), annot=True)


fig, axs = plt.subplots(4,2, figsize=(20,30))
cat_features = X_train.select_dtypes(include='object').columns
r = 0
c = 0
for feat in cat_features:
    if c == 2:
        c = 0
        r += 1
    counts = sns.boxplot(ax=axs[r,c], data=X_train, x=feat, y='Price')
    
    axs[r,c].set_title(f'{feat}')
    axs[r,c].set(xlabel=None)

  
    c += 1


grouped_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


for c in grouped_df.select_dtypes(include='object').columns:
    print(grouped_df.groupby(c)['Price'].mean())
    print()


for c in grouped_df.select_dtypes(exclude='object').columns:
    print(grouped_df.groupby(c)['Price'].mean())
    print()


X_train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
X_train_old = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')

X_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



X_train = pd.concat([X_train_old, X_train_extra], axis=0, ignore_index=True)


X_train




# import numpy as np
# import pandas as pd

# def add_is_missing_row(df, col):
#     """Adds a binary column indicating missing values."""
#     df[f'{col}_is_missing'] = df[col].isnull().astype(int)
#     return df

# # Define imputation strategies
# categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
# numerical_features = ["Weight Capacity (kg)"]

# # Fill categorical missing values with mode (most frequent value)
# for col in categorical_features:
#     X_train = add_is_missing_row(X_train, col)
#     mode_value = X_train[col].mode()
#     if not mode_value.empty:
#         X_train[col].fillna(mode_value[0], inplace=True)

#     X_test = add_is_missing_row(X_test, col)
#     mode_value_test = X_test[col].mode()
#     if not mode_value_test.empty:
#         X_test[col].fillna(mode_value_test[0], inplace=True)

# # Fill numerical missing values with median
# for col in numerical_features:
#     X_train = add_is_missing_row(X_train, col)
#     X_train[col].fillna(X_train[col].median(), inplace=True)

#     X_test = add_is_missing_row(X_test, col)
#     X_test[col].fillna(X_test[col].median(), inplace=True)



# num_cols = X_train.select_dtypes(exclude=['object', 'datetime', 'bool']).columns.tolist()
# num_cols.remove('Premium Amount')
# cat_cols = X_train.select_dtypes(include='object').columns.tolist()

# scaler = StandardScaler()

# X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
# X_test[num_cols] = scaler.transform(X_test[num_cols])


def perform_feature_engineering(df):
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Num'] = df['Size'].map(size_mapping)
    df['Compartments_per_Size'] = df['Compartments'] / df['Size_Num']    
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments'] 
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 'Large' else 0)

    df['Brand_Material'] = df['Brand']+'_'+df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Brand_Laptop_Compartment'] = df['Laptop Compartment'].apply(str) + '_' + df['Brand'].apply(str)
    df['Brand_Style'] = df['Brand']+ '_' + df['Style']
    df['Brand_Color'] = df['Brand'] + '_' + df['Color']
    df['Material_Size'] = df['Material'] + '_' + df['Size']


    df['Weight Capacity (lb)'] = df['Weight Capacity (kg)']/0.45359237
    
    return df



X_train_fe = perform_feature_engineering(X_train.copy())
X_test_fe = perform_feature_engineering(X_test.copy())


y_train = X_train_fe.Price
X_train_fe.drop('Price', axis=1, inplace=True)


from cuml.preprocessing import TargetEncoder
X_train_encoded_df = X_train_fe.copy()
X_test_encoded_df = X_test_fe.copy()


categorical_columns = X_train_encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_columns += ['Weight Capacity (kg)', 'Compartments']
TE = TargetEncoder(n_folds=25, smooth=30, split_method='random', stat='mean')
for c in categorical_columns:
    print(c)
    X_train_encoded_df[f"{c}"] = TE.fit_transform(X_train_encoded_df[c], y_train)
    X_test_encoded_df[f"{c}"] = TE.transform(X_test_encoded_df[c])
# s = np.sqrt(np.mean( (train.Price-train.pred)**2.0 ) )
# print(f"Validation RSME using Target Encode Weight Capacity = {s}")


# y_train = np.log1p(y_train)
log = False


import numpy as np

def add_features(df):
    # Ratio features
    df['Weight_Ratio_Material'] = df['Weight Capacity (kg)'] / df['Material']
    df['Weight_Ratio_Brand'] = df['Weight Capacity (kg)'] / df['Brand']
    df['Weight_Ratio_Size'] = df['Weight Capacity (kg)'] / df['Size']
    df['Weight_Ratio_Style'] = df['Weight Capacity (kg)'] / df['Style']
    df['Weight_Ratio_Color'] = df['Weight Capacity (kg)'] / df['Color']
    
    df['Weight_Brand_Material_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Material']
    df['Weight_Brand_Size_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Size']
    df['Weight_Brand_Style_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Style']
    df['Weight_Brand_Color_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Color']

    # Sum features
    df['Weight_Plus_Size'] = df['Weight Capacity (kg)'] + df['Size']
    df['Brand_Material_Sum'] = df['Brand'] + df['Material']

    # Difference features
    df['Weight_Minus_Size'] = df['Weight Capacity (kg)'] - df['Size']
    df['Brand_Material_Diff'] = df['Brand'] - df['Material']

    # Product features
    df['Weight_Size_Product'] = df['Weight Capacity (kg)'] * df['Size']
    df['Brand_Material_Product'] = df['Brand'] * df['Material']

    # Power & Root features
    df['Weight_Squared'] = df['Weight Capacity (kg)'] ** 2
    df['Weight_Sqrt'] = np.sqrt(df['Weight Capacity (kg)'])

    # Log features (handling zero values with log1p)
    df['Log_Weight'] = np.log1p(df['Weight Capacity (kg)'])
    df['Log_Size'] = np.log1p(df['Size'])

    return df

X_train_encoded_df = add_features(X_train_encoded_df)
X_test_encoded_df = add_features(X_test_encoded_df)


X_train_encoded_df


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 


def encode_features(X_train, X_test):
    combined_df = pd.concat([X_train, X_test], axis=0)
    categorical_columns = combined_df.select_dtypes(include=['object', 'category']).columns
    numerical_columns = combined_df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    print(numerical_columns)


    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough' 
    )
    
    preprocessor.fit(combined_df)
    
    X_train_nan_encoded = preprocessor.transform(X_train)
    X_test_nan_encoded = preprocessor.transform(X_test)
    
    encoded_feature_names = preprocessor.get_feature_names_out()
    X_train_encoded_df = pd.DataFrame(X_train_nan_encoded, columns=encoded_feature_names)
    X_test_encoded_df = pd.DataFrame(X_test_nan_encoded, columns=encoded_feature_names)

    return X_train_encoded_df, X_test_encoded_df


# X_train_encoded_df, X_test_encoded_df = encode_features(X_train_fe, X_test_fe)


from sklearn.model_selection import train_test_split


x_train, x_val, y_train_model, y_val = train_test_split(X_train_encoded_df, y_train, test_size=0.15, random_state=42)


def rmse(y_true, y_pred):
    return np.sqrt(np.sum(((y_true-y_pred)**2)/len(y_true)))



scores = {'model': [],
          'model_name':[], 
         'rmse': []
         }

def add_scores_rmse(model, model_name, y_true, y_pred):
    error = rmse(y_true, y_pred)
    scores['model'].append(model)
    scores['model_name'].append(model_name)
    scores['rmse'].append(error)
    print(f'RMSE for {model_name}: {error}')
    


from sklearn.dummy import DummyRegressor


dummy = DummyRegressor()
dummy.fit(x_train, y_train_model)


scores


add_scores_rmse(dummy, 'DummyRegressor', y_val, dummy.predict(x_val))


from sklearn.model_selection import cross_val_score


from cuml.ensemble import RandomForestRegressor





def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'max_features':trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }

    rf_model = RandomForestRegressor(**params)
    
    return -1*cross_val_score(rf_model, x_train, y_train_model, n_jobs=-1, cv=3, scoring='neg_root_mean_squared_error').mean()


# study = optuna.create_study(direction='minimize')
# study.optimize(rf_objective, n_trials=100)
# trial = study.best_trial
# param = trial.params


param = {'n_estimators': 259, 'max_depth': 7, 'max_features': 'sqrt'}


rf = RandomForestRegressor(**param)
rf.fit(x_train, y_train_model)


add_scores_rmse(rf, 'RandomForestRegressor', y_val, rf.predict(x_val))


from xgboost import XGBRegressor


def objective_xgb(trial):
    # params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 20, 500),
    #     'max_depth': int(trial.suggest_float('max_depth', 1, 100, log=True)),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    #     'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    #     'gamma': trial.suggest_float('gamma', 0, 2),
    #     'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
    #     'subsample': trial.suggest_float('subsample', 0, 1)
    # }

    params = {        
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=100),
            "max_depth":trial.suggest_int("max_depth", 1, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True), 
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0),
            "verbosity": 0, 
            "device": 'gpu', 
            "objective": 'reg:squarederror', 
            "tree_method": 'gpu_hist'
    }    
    
    xgb_model = XGBRegressor(**params)

    return -1*cross_val_score(xgb_model, x_train, y_train_model, n_jobs=-1, cv=5, scoring='neg_root_mean_squared_error').mean()

# study = optuna.create_study(direction='minimize')
# study.optimize(objective_xgb, n_trials=100)
# trial = study.best_trial
# param = trial.params


# param = {'n_estimators': 350, 'max_depth': 3, 'min_child_weight': 4, 'learning_rate': 0.08594446209873564, 'subsample': 0.8398992680035604, 'colsample_bytree': 0.9529221921014095, 'reg_alpha': 1.1463897658647415, 'reg_lambda': 6.708469240337878, 'gamma': 0.010410503602230198}
param = {'n_estimators': 616, 'max_depth': 7, 'min_child_weight': 7, 'learning_rate': 0.014817074030209496, 'subsample': 0.9279147512217129, 'colsample_bytree': 0.7512937797375905, 'reg_alpha': 7.54010884966121, 'reg_lambda': 6.202964607347674, 'gamma': 0.2552227561152478}

#param = trial.params
#param = {'n_estimators': 450, 'max_depth': 4, 'min_child_weight': 7, 'learning_rate': 0.010230533167302816, 'subsample': 0.7977355633601261, 'colsample_bytree': 0.1526355211804084, 'reg_alpha': 3.016256372030049, 'reg_lambda': 9.491053950347876, 'gamma': 0.7}
#param = {'n_estimators': 450, 'max_depth': 1, 'min_child_weight': 2, 'learning_rate': 0.05106717951268276, 'subsample': 0.8179078683056303, 'colsample_bytree': 0.5995719977594407, 'reg_alpha': 3.4566350070836216, 'reg_lambda': 7.86626093277777, 'gamma': 0.4063751228507485}

#param = {'n_estimators': 650, 'max_depth': 3, 'min_child_weight': 2, 'learning_rate': 0.08264629037419823, 'subsample': 0.9036414678597458, 'colsample_bytree': 0.36719528828306813, 'reg_alpha': 7.544418716508265, 'reg_lambda': 1.6293063491243416, 'gamma': 0.44748881496157156}


xgb = XGBRegressor(device='gpu', **param)
xgb.fit(x_train, y_train_model)


add_scores_rmse(xgb, 'XGBRegressor', y_val, xgb.predict(x_val))


pd.DataFrame({
    'id': sample_subm.id,
    'Price': xgb.predict(X_test_encoded_df)
}).to_csv('xgb_s.csv', index=False)


from catboost import CatBoostRegressor


def objective_cat(trial):
    # params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 20, 500),
    #     'max_depth': int(trial.suggest_float('max_depth', 1, 100, log=True)),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    #     'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    #     'gamma': trial.suggest_float('gamma', 0, 2),
    #     'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
    #     'subsample': trial.suggest_float('subsample', 0, 1)
    # }

    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000), 
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),  
        'depth': trial.suggest_int('depth', 1, 16),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),  
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'task_type': 'GPU',
        'verbose': 0
    }

    
    cat_model = CatBoostRegressor(**params)

    return -1*cross_val_score(cat_model, x_train, y_train_model, n_jobs=1, cv=5, scoring='neg_root_mean_squared_error').mean()

# study = optuna.create_study(direction='minimize')
# study.optimize(objective_cat, n_trials=100)
# trial = study.best_trial



# param = trial.params
#param = {'iterations': 757, 'early_stopping_rounds': 111, 'depth': 1, 'l2_leaf_reg': 2.51131199440927, 'learning_rate': 0.060630663020744394}
#param = {'iterations': 846, 'early_stopping_rounds': 193, 'depth': 3, 'l2_leaf_reg': 5.378616501408868, 'learning_rate': 0.09056944139570554}
param = {'iterations': 794, 'early_stopping_rounds': 73, 'depth': 7, 'l2_leaf_reg': 6.1286565088818605, 'learning_rate': 0.03321832033936305}
# param = {'depth': 6, 'learning_rate': 0.025187969747436947, 'iterations': 636, 'l2_leaf_reg': 5.910298555905153, 'bagging_temperature': 0.06752939348809489}
# param = {'iterations': 950, 'early_stopping_rounds': 11, 'depth': 4, 'l2_leaf_reg': 0.31610651106415355, 'learning_rate': 0.05932089640991059}


cat_model = CatBoostRegressor(task_type='GPU', verbose=100, **param)
cat_model.fit(x_train, y_train_model)


add_scores_rmse(cat_model, 'CatRegressor', y_val, cat_model.predict(x_val))


pd.DataFrame({
    'id': sample_subm.id,
    'Price': cat_model.predict(X_test_encoded_df)
}).to_csv('cat_s.csv', index=False)


from lightgbm import LGBMRegressor


def objective_light(trial):
    # params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 20, 500),
    #     'max_depth': int(trial.suggest_float('max_depth', 1, 100, log=True)),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    #     'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    #     'gamma': trial.suggest_float('gamma', 0, 2),
    #     'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
    #     'subsample': trial.suggest_float('subsample', 0, 1)
    # }

    params = {        
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=100),
            "max_depth":trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 7, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True), 
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.),
            "num_leaves": trial.suggest_int("num_leaves", 2^4+1, 2^10+1),
            'verbosity':0,
            'device': 'gpu'
    }    
    
    light_model = LGBMRegressor(**params)

    return -1*cross_val_score(light_model, x_train, y_train_model, n_jobs=-1, cv=3, scoring='neg_root_mean_squared_error').mean()

# study = optuna.create_study(direction='minimize')
# study.optimize(objective_light, n_trials=100)
# trial = study.best_trial


param = {'n_estimators': 954, 'max_depth': 6, 'min_child_weight': 15, 'learning_rate': 0.06870001729454589, 'subsample': 0.9406706665875935, 'colsample_bytree': 0.9495136656130558, 'reg_alpha': 1.9843307956041611, 'reg_lambda': 8.98799996278166, 'num_leaves': 8}

# param = {'n_estimators': 650, 'max_depth': 9, 'min_child_weight': 11, 'learning_rate': 0.06102876868870081, 'subsample': 0.700224397841362, 'colsample_bytree': 0.914550576518922, 'reg_alpha': 4.406561720528696, 'reg_lambda': 9.525905666639105, 'num_leaves': 8}
#param = {'n_estimators': 550, 'max_depth': 7, 'min_child_weight': 14, 'learning_rate': 0.09919135392165795, 'subsample': 0.9787568177167378, 'colsample_bytree': 0.13138520881272656, 'reg_alpha': 0.5208941568169729, 'reg_lambda': 1.22117449823864, 'num_leaves': 8}
#param = {'n_estimators': 950, 'max_depth': 6, 'min_child_weight': 8, 'learning_rate': 0.010420837349282069, 'subsample': 0.7668565054850575, 'colsample_bytree': 0.10035808296187733, 'reg_alpha': 9.064791632557105, 'reg_lambda': 4.768416974088945, 'num_leaves': 9}


light_model = LGBMRegressor(device='gpu', **param)
light_model.fit(x_train, y_train_model)



add_scores_rmse(light_model, 'LightGBMRegressor', y_val, light_model.predict(x_val))


pd.DataFrame({
    'id': sample_subm.id,
    'Price': light_model.predict(X_test_encoded_df)
}).to_csv('light_s.csv', index=False)


from sklearn.ensemble import HistGradientBoostingRegressor as hgbc


def objective_hgbc(trial):
    # params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 20, 500),
    #     'max_depth': int(trial.suggest_float('max_depth', 1, 100, log=True)),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    #     'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    #     'gamma': trial.suggest_float('gamma', 0, 2),
    #     'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
    #     'subsample': trial.suggest_float('subsample', 0, 1)
    # }

    params = {
    'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10), 
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
    'max_iter': trial.suggest_int('max_iter', 100, 1000),  
    'max_depth': trial.suggest_int('max_depth', 3, 50), 
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20), 
    'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100), 
    'max_bins': trial.suggest_int('max_bins', 64, 255),  
    'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),  
    'verbose': 0
    }
    hgbc_model = hgbc(**params)
    
    return -1*cross_val_score(hgbc_model, x_train, y_train_model, n_jobs=1, cv=3, scoring='neg_root_mean_squared_error').mean()

# study = optuna.create_study(direction='minimize')
# study.optimize(objective_hgbc, n_trials=100)
# trial = study.best_trial

# print('Accuracy: {}'.format(trial.value))

# print("Best hyperparameters: {}".format(trial.params)) 
# param = trial.params

# param = {'l2_regularization': 1.274073919941976, 'learning_rate': 0.0475945745057485, 'max_iter': 388, 'max_depth': 18, 'min_samples_leaf': 2, 'max_leaf_nodes': 12, 'max_bins': 238, 'early_stopping': False}


param = {'l2_regularization': 1.6370446213950829, 'learning_rate': 0.018857597792595007, 'max_iter': 498, 'max_depth': 49, 'min_samples_leaf': 20, 'max_leaf_nodes': 80, 'max_bins': 229, 'early_stopping': False}


hgbc_model = hgbc(**param)
hgbc_model.fit(x_train, y_train_model)


add_scores_rmse(hgbc_model, 'HistGradientBoosting', y_val, hgbc_model.predict(x_val))





from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge



estimators = [
    ('lgb', light_model),
    ('xgb', xgb),
     ('cat', cat_model),
    ('hgbc', hgbc_model),
    ('rf', rf)
]
reg = StackingRegressor(estimators=estimators, final_estimator=Ridge(tol=1e-2, max_iter=1000000,alpha=0.09983))

reg.fit(x_train, y_train_model)


add_scores_rmse(reg, 'Stacking', np.array(y_val), np.array(reg.predict(np.array(x_val))))


pd.DataFrame({
    'id': sample_subm.id,
    'Price': reg.predict(np.array(X_test_encoded_df))
}).to_csv('stacking_s.csv', index=False)


from sklearn.ensemble import VotingRegressor



softvoting = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb),('lgb', light_model),('cat', cat_model), ('hgbc', hgbc_model)])


# softvoting.fit(x_train, y_train_model)
# add_scores_rmse(softvoting, 'Voting', np.array(y_val), np.array(softvoting.predict(x_val)))


X_train


for i in range(len(scores['model'])):
    model = scores['model'][i]
    pred = model.predict(X_test_encoded_df)
    print(f'RMSE for {scores["model_name"][i]}: {scores["rmse"][i]}')
    submission = pd.DataFrame({
        "id": X_test['id'],
        'Price': pred
    })
    submission.to_csv(f'submission_{scores["model_name"][i]}.csv', index=False)





