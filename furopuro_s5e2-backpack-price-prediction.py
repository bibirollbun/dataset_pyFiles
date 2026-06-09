import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


tab_train = pd.read_csv( "/kaggle/input/playground-series-s5e2/train.csv" )
tab_train_extra = pd.read_csv( "/kaggle/input/playground-series-s5e2/training_extra.csv" )
tab_test = pd.read_csv( "/kaggle/input/playground-series-s5e2/test.csv" )
tab_sub  = pd.read_csv( "/kaggle/input/playground-series-s5e2/sample_submission.csv" )





all_data = [
    #tab_train, 
    pd.concat([tab_train, tab_train_extra], ignore_index=True),
    tab_test]



[i.dtypes for i in all_data ]


[i.columns for i in all_data ]


object_columnns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color' ]


for data in all_data:
 [ print( i , data[i].unique()) for i in object_columnns ] 
 print("/"*30)



import matplotlib.pyplot as plt
import seaborn as sns
data = all_data[0]['Price'].hist( bins=55)


 #all_data[0] = all_data[0][ all_data[0]['Price'] < 149 ]



#all_data_imp[0]["Size"]

#import matplotlib.pyplot as plt
#import seaborn as sns

for col_o in object_columnns:
    break
    plt.figure(figsize=(5, 2))
    sns.boxplot(x= col_o, y='Price', data=all_data_imp[0]) # data=expensive_data) 
    plt.xticks(rotation=45)
    plt.show()



for data in all_data:
    print( f"num of columns:" , data.shape[0])
    [ print( i ,sum(data[i].isna())) for i in data.columns ]
    print("-"*30)


#all_data[0] =  all_data[0].sample(n=30000, random_state=42)
#all_data[0] =  all_data[0].sample(n=500, random_state=42)

all_data[0].shape


all_data_imp = []
for df in all_data:
    for col in object_columnns:
        temp_ = df.loc[:,col].value_counts()
        impute_string = temp_.index[0]
        #impute_string = "none"        
        df.loc[:,col] = df.loc[:,col].fillna(impute_string)

    
    median_A = df['Weight Capacity (kg)'].median()
    df.loc[:,'Weight Capacity (kg)'] = df.loc[:,'Weight Capacity (kg)'].fillna(median_A, inplace=False)
    
    all_data_imp.append(df)



# Impute Missing Values in Object Columns with 'None'
# 
#obj_cols = train.select_dtypes(include=['object']).columns
#
#train[obj_cols] = train[obj_cols].fillna('None')
#test[obj_cols] = test[obj_cols].fillna('None')


all_data_imp[0].head(6)


# mapping 
mapping_dic = {
 "Laptop Compartment"  : {'Yes': 1, 'No': 0 },
 "Waterproof"  : {'Yes': 1, 'No': 0 },
 "Size"  : {'Small': 1, 'Medium': 2, 'Large': 3} }

# for nans
nan_placeholder = -1

# Apply the mapping
for df in all_data_imp:
    for col, mapping in mapping_dic.items():
        df.loc[:,col] = df.loc[:,col].map(mapping)
        df[col] = df[col].fillna(nan_placeholder)
        df[col] = df[col].astype(float) 





from sklearn.preprocessing import OneHotEncoder
columns_to_encode = ['Brand', 'Material', 'Color',"Style" ]
encoder = OneHotEncoder(sparse_output=False)

all_data_t = []
for df in all_data_imp:
    encoded_data = encoder.fit_transform(df.loc[:,columns_to_encode])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns_to_encode)) 
    df_temp = df.drop(columns_to_encode, axis=1)
    df_new = pd.concat([df_temp.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    all_data_t.append( df_new)



df_new.dtypes


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


y = all_data_t[0]['Price']
X = all_data_t[0].copy().drop(columns=['Price','id'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define predictors
rf_m = RandomForestRegressor(random_state=42)  # Random Forest Regression
gb_m = GradientBoostingRegressor(random_state=42)  # Gradient Boosting Regression
ab_m = AdaBoostRegressor(random_state=42)  # AdaBoost Regression
#mlp_m = MLPRegressor(random_state=42, hidden_layer_sizes=(100,), max_iter=5000)  # MLP Regression
XGBR_m = XGBRegressor(random_state=42)
    #device="cuda",
    #max_depth=5,
    #n_estimators=2000,
    #learning_rate=0.015,
    

# train some models
model_dic = {
    #"RandomForest": rf_m,
    "GradientBoosting": gb_m,
    "AdaBoost": ab_m,
    # "MLPRegressor": mlp_m,
    "XGBRegressor" : XGBR_m
}





from sklearn.model_selection import GridSearchCV #rf and others
from sklearn.model_selection import RandomizedSearchCV #GradientBoostingRegressor

param_grid_dic = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01,0.025, 0.05],
        "max_depth": [3, 4, 5],
        "min_samples_split": [5, 7 ,10, 12],
        "min_samples_leaf": [ 2, 3, 4],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2"]
    },
    "AdaBoost": {
        "n_estimators": [25, 50, 75],
        "learning_rate": [0.25, 0.5, 0.75],
        "loss": ["linear", "square", "exponential"]
    },
    "MLPRegressor": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50), (50, 50, 50)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"]
    },
    "XGBRegressor": {
        "n_estimators": [75,100,125],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.025, 0.05],
        "subsample": [0.9, 1.0]
    }
}







perform_gridsearch = True
cv_in = 5

if perform_gridsearch:    
    best_param_dic = {}
    for model_name in model_dic.keys():
        print( model_name )
        model = model_dic[ model_name ]
        param_grid = param_grid_dic[ model_name ]
        grid_search = GridSearchCV(model,
                                   param_grid,
                                   cv=cv_in,
                                   n_jobs=-1,
                                   verbose=2)
        grid_search.fit(X_train, y_train)
        best_param_dic[ model_name ] = grid_search.best_params_

else:
    best_param_dic = { 
        'RandomForest': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 300},
        'GradientBoosting': {'learning_rate': 0.01, 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 4,
                             'min_samples_split': 10, 'n_estimators': 100, 'subsample': 1.0},
        'AdaBoost': {'learning_rate': 0.5, 'loss': 'linear', 'n_estimators': 50},
        'XGBRegressor': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
    }
    
    # best_param_dic = {'RandomForest': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}, 'GradientBoosting': {'learning_rate': 0.01, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.9}, 'AdaBoost': {'learning_rate': 0.5, 'loss': 'exponential', 'n_estimators': 200}, 'XGBRegressor': {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 100, 'subsample': 0.9}}
    # best_param_dic = {'RandomForest': {'max_depth': None,  'max_features': 'sqrt',  'min_samples_leaf': 2,  'min_samples_split': 10,  'n_estimators': 300}, 'GradientBoosting': {'learning_rate': 0.01,  'max_depth': 3,  'max_features': 'sqrt',  'min_samples_leaf': 4,  'min_samples_split': 2,  'n_estimators': 100,  'subsample': 1.0}, 'AdaBoost': {'learning_rate': 0.5, 'loss': 'linear', 'n_estimators': 100}, 'MLPRegressor': {'activation': 'tanh',  'alpha': 0.001,  'hidden_layer_sizes': (100,),  'learning_rate': 'constant',  'solver': 'sgd'}}


print( best_param_dic )


#best_rf = RandomForestRegressor(**best_params, random_state=42)
#model_dic["Random_forest_tuned"] = best_rf

# train some models
model_dic_tuned = {
    # "RandomForest": RandomForestRegressor(**best_param_dic["RandomForest"] , random_state=42) ,
    "GradientBoosting": GradientBoostingRegressor(**best_param_dic["GradientBoosting"] , random_state=42),
    "AdaBoost": AdaBoostRegressor(**best_param_dic["AdaBoost"] , random_state=42),
     "XGBRegressor" : XGBRegressor(**best_param_dic["XGBRegressor"] , random_state=42)
    # "MLPRegressor": mlp_m
}




x_pred_dic = {}
y_pred_dic = {}

for name, model in model_dic_tuned.items():
    print(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    x_pred = model.predict(X_train)
    x_pred_dic[name] =  x_pred 
    y_pred_dic[name] =  y_pred




outcome_dic = {}
outcome_dic["test"] = {}
outcome_dic["train"] = {}

for name, y_pred in y_pred_dic.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse )
    r2 = r2_score(y_test, y_pred)
    outcome_dic["test"][name] = {"mse" : mse ,"r2": r2 , "rmse" : rmse }


for name, x_pred in x_pred_dic.items():
    mse = mean_squared_error(y_train, x_pred)
    r2 = r2_score(y_train , x_pred)
    rmse = np.sqrt(mse )
    outcome_dic["train"][name] = {"mse" : mse ,"r2": r2, "rmse" : rmse }




pd.DataFrame(outcome_dic["train"])


pd.DataFrame(outcome_dic["test"])


use_esemble = False


if use_esemble:
    sub_dic = {}
    for name, model in model_dic_tuned.items():
        x_sub = all_data_t[1].copy().drop(columns=['id'])
        y_sub = model.predict(x_sub)
        sub_dic[ name ] = y_sub
    df = pd.DataFrame( sub_dic )
    df_mean = df.mean(axis=1).to_frame(name="Price")
    data_sub = pd.DataFrame({
    'id': all_data_t[1]["id"],
    'Price':df_mean.squeeze().to_numpy() })
    print( data_sub )



selected_model  = 'GradientBoosting'

if not use_esemble:
    model = model_dic_tuned[selected_model]
    x_sub = all_data_t[1].copy().drop(columns=['id'])
    y_sub = model.predict(x_sub)
    data_sub = pd.DataFrame({
    'id': all_data_t[1]["id"],
    'Price':y_sub })
    print( data_sub )

    


print( tab_sub.head(3) )
print( tab_sub.shape)


print( data_sub.head(3) )
print( data_sub.shape )


data_sub.to_csv('submission.csv', index=False)

