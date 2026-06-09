import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder
import sklearn

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(24)

sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
training_extra_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train_df = pd.concat([train_df, training_extra_df])


print("train shape: ", train_df.shape)
print("test shape: ", test_df.shape)


orig_df = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
print("Original data shape", orig_df.shape )


orig_df = orig_df.groupby("Weight Capacity (kg)").Price.mean()
orig_df.name = "orig_Price"
orig_df.head()


train_df = train_df.merge(orig_df, on="Weight Capacity (kg)", how="left")
test_df = test_df.merge(orig_df, on="Weight Capacity (kg)", how="left")


orig_df = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
orig_df = orig_df.loc[(orig_df["Weight Capacity (kg)"]>5)&(orig_df["Weight Capacity (kg)"]<30)]
orig_df.columns = [f"orig_{c}" for c in orig_df.columns]
train_df = train_df.merge(orig_df.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
train_df = train_df.drop("id",axis=1)
test_df = test_df.merge(orig_df.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")


train_df = train_df.replace({
    "Compartments": {
        1.0: "one",
        2.0: "two",
        3.0: "three",
        4.0: "four",
        5.0: "five",
        6.0: "six",
        7.0: "seven",
        8.0: "eight",
        9.0: "nine",
        10.0: "ten"
    },
    "orig_Compartments": {
        1.0: "one",
        2.0: "two",
        3.0: "three",
        4.0: "four",
        5.0: "five",
        6.0: "six",
        7.0: "seven",
        8.0: "eight",
        9.0: "nine",
        10.0: "ten"
    },
})

test_df = test_df.replace({
    "Compartments": {
        1.0: "one",
        2.0: "two",
        3.0: "three",
        4.0: "four",
        5.0: "five",
        6.0: "six",
        7.0: "seven",
        8.0: "eight",
        9.0: "nine",
        10.0: "ten"
    },
    "orig_Compartments": {
        1.0: "one",
        2.0: "two",
        3.0: "three",
        4.0: "four",
        5.0: "five",
        6.0: "six",
        7.0: "seven",
        8.0: "eight",
        9.0: "nine",
        10.0: "ten"
    },
})


train_df = train_df.fillna({
    "Brand": "undefined",
    "Material": "undefined",
    "Size": "undefined",
    "Compartments": "undefined",
    "Laptop Compartment": "undefined",
    "Waterproof": "undefined",
    "Style": "undefined",
    "Color": "undefined",
    "orig_Brand": "undefined",
    "orig_Material": "undefined",
    "orig_Size": "undefined",
    "orig_Compartments": "undefined",
    "orig_Laptop Compartment": "undefined",
    "orig_Waterproof": "undefined",
    "orig_Style": "undefined",
    "orig_Color": "undefined",
})

test_df = test_df.fillna({
    "Brand": "undefined",
    "Material": "undefined",
    "Size": "undefined",
    "Compartments": "undefined",
    "Laptop Compartment": "undefined",
    "Waterproof": "undefined",
    "Style": "undefined",
    "Color": "undefined",
    "orig_Brand": "undefined",
    "orig_Material": "undefined",
    "orig_Size": "undefined",
    "orig_Compartments": "undefined",
    "orig_Laptop Compartment": "undefined",
    "orig_Waterproof": "undefined",
    "orig_Style": "undefined",
    "orig_Color": "undefined",
})


train_df = train_df.drop(columns=["orig_Weight Capacity (kg)"])
test_df = test_df.drop(columns=["orig_Weight Capacity (kg)"])


test_df["Price"] = 0


kf = KFold(n_splits=15, shuffle=True, random_state=42)
kf = list(kf.split(train_df.values))


import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, log_evaluation
import gc


train_df["Weight Capacity (kg)"] = train_df["Weight Capacity (kg)"].fillna(train_df["Weight Capacity (kg)"].mean())
train_df["orig_Price"] = train_df["orig_Price"].fillna(train_df["orig_Price"].mean())
test_df["Weight Capacity (kg)"] = test_df["Weight Capacity (kg)"].fillna(test_df["Weight Capacity (kg)"].mean())
test_df["orig_Price"] = test_df["orig_Price"].fillna(test_df["orig_Price"].mean())


gc.collect()


brand_map = {
    'Jansport': 4,
    'Under Armour': 3,
    'Nike': 2,
    'Adidas': 1,
    'Puma': 0,
    'undefined': 5
}
material_map = {
    'Leather': 3,
    'Canvas': 2,
    'Nylon': 1,
    'Polyester': 0,
    'undefined': 4
}
size_map = {
    'Medium': 2,
    'Small': 1,
    'Large': 0,
    'undefined': 3
}
compartments_map = {
    "one": 0,
    "two": 1,
    "three": 2,
    "four": 3,
    "five": 4,
    "six": 5,
    "seven": 6,
    "eight": 7,
    "nine": 8,
    "ten": 9,
    "undefined": 10
}
lapcomp_map = {
    'Yes':1 ,
    'No': 0,
    'undefined': 2
}
waterproof_map = {
    'Yes': 1,
    'No': 0,
    'undefined': 2
}
style_map = {
    'Tote': 2,
    'Messenger': 1,
    'Backpack': 0,
    'undefined': 3
}
color_map = {
    'Black': 5,
    'Green': 4,
    'Red': 3,
    'Blue': 2,
    'Gray': 1,
    'Pink': 0,
    'undefined': 6
}
train_df = train_df.replace({
    'Brand': brand_map, 
    'Material': material_map,
    'Size': size_map,
    'Compartments': compartments_map,
    'Laptop Compartment': lapcomp_map,
    'Waterproof': waterproof_map,
    'Style': style_map,
    'Color': color_map,
    'orig_Brand': brand_map, 
    'orig_Material': material_map,
    'orig_Size': size_map,
    'orig_Compartments': compartments_map,
    'orig_Laptop Compartment': lapcomp_map,
    'orig_Waterproof': waterproof_map,
    'orig_Style': style_map,
    'orig_Color': color_map
})
test_df = test_df.replace({
    'Brand': brand_map, 
    'Material': material_map,
    'Size': size_map,
    'Compartments': compartments_map,
    'Laptop Compartment': lapcomp_map,
    'Waterproof': waterproof_map,
    'Style': style_map,
    'Color': color_map,
    'orig_Brand': brand_map, 
    'orig_Material': material_map,
    'orig_Size': size_map,
    'orig_Compartments': compartments_map,
    'orig_Laptop Compartment': lapcomp_map,
    'orig_Waterproof': waterproof_map,
    'orig_Style': style_map,
    'orig_Color': color_map
})


COMBO = []
for i in train_df.drop(columns=
                       [
                           'Price', 
                           "Weight Capacity (kg)", 
                           "orig_Price",
                           'orig_Brand',
                           'orig_Material',
                           'orig_Size',
                           'orig_Compartments',
                           'orig_Laptop Compartment',
                           'orig_Waterproof',
                           'orig_Style',
                           'orig_Color'
                       ]
                      ).columns.values:
    train_df[f"{i}_wc"] = train_df[i].values*100 + train_df['Weight Capacity (kg)'].values
    test_df[f"{i}_wc"] = test_df[i].values*100 + test_df['Weight Capacity (kg)'].values
    COMBO.append(f"{i}_wc")


val_rmse_all_folds = []
test_prediction = []
ensemble_weight = []
STATS2 = ["mean","std","median"]
STATS = ["mean","std","count","nunique","median","min","max","skew"]
cat_features = train_df.select_dtypes(include=['int']).columns.tolist()
features = train_df.drop(columns=['Price']).columns

for fold, (train_idx, val_idx) in enumerate(kf):
    train_data = train_df.iloc[train_idx].reset_index(drop=True).astype('float').copy()
    val_data = train_df.iloc[val_idx].reset_index(drop=True).astype('float').copy()
    test_data = test_df.reset_index(drop=True).astype('float').copy()
    
    print(f"Fold {fold}")
    
    kf2 = KFold(n_splits=10, shuffle=True, random_state=42)
    kf2 = list(kf2.split(train_data.values))
    print("Encoding 1")
    for j, (train_idx2, val_idx2) in enumerate(tqdm(kf2)):
        train_data2 = train_data.loc[train_idx2,features.values.tolist()+['Price']].copy()
        val_data2 = train_data.loc[val_idx2, features.values.tolist()].copy()

        tmp = train_data2.groupby("Weight Capacity (kg)").Price.agg(STATS)
        tmp.columns = [f"TE_wc_{s}" for s in STATS]
        val_data2 = val_data2.merge(tmp, on="Weight Capacity (kg)", how="left")
        for c in tmp.columns:
            train_data.loc[val_idx2,c] = val_data2[c].values
        del tmp
        gc.collect()

        for col in COMBO:
            tmp = train_data2.groupby(col).Price.agg(STATS2)
            tmp.columns = [f"TE_{col}_{s}" for s in STATS2]
            val_data2 = val_data2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                train_data.loc[val_idx2,c] = val_data2[c].values
            del tmp
            gc.collect()
            
    tmp = train_data.groupby("Weight Capacity (kg)").Price.agg(STATS)
    tmp.columns = [f"TE_wc_{s}" for s in STATS]
    val_data = val_data.merge(tmp, on="Weight Capacity (kg)", how="left")
    test_data = test_data.merge(tmp, on="Weight Capacity (kg)", how="left")

    for col in COMBO:
        tmp = train_data.groupby(col).Price.agg(STATS2)
        tmp.columns = [f"TE_{col}_{s}" for s in STATS2]
        val_data = val_data.merge(tmp, on=col, how="left")
        test_data = test_data.merge(tmp, on=col, how="left")

    num_features = train_data.select_dtypes(include=['float']).columns.tolist()
    train_data[num_features] = train_data[num_features].fillna(train_data[num_features].median())
    val_data[num_features] = val_data[num_features].fillna(val_data[num_features].median())
    test_data[num_features] = test_data[num_features].fillna(test_data[num_features].median())
    del num_features
    del tmp
    gc.collect()

    print("Encoding 2")
    for col in tqdm(cat_features):
        tmp = train_data.groupby(col)["Weight Capacity (kg)"].agg(STATS2)
        tmp.columns = [f"CFE_{col}_wc_{s}" for s in STATS2]
        train_data = train_data.merge(tmp, on=col, how="left")
        val_data = val_data.merge(tmp, on=col, how="left")
        test_data = test_data.merge(tmp, on=col, how="left")
        del tmp
        gc.collect()
            
    print("Encoding 3")
    TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean', output_type='numpy', seed=24)
    for col in tqdm(features):
        TE.fit(train_data[col].values, train_data['Price'].values)
        train_data[f"TE_{col}_mean"] = TE.transform(train_data[col].values)
        val_data[f"TE_{col}_mean"] = TE.transform(val_data[col].values)
        test_data[f"TE_{col}_mean"] = TE.transform(test_data[col].values)

    del TE
    gc.collect()
        
    TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='median', output_type='numpy', seed=24)
    for col in tqdm(features):
        TE.fit(train_data[col].values, train_data['Price'].values)
        train_data[f"TE_{col}_median"] = TE.transform(train_data[col].values)
        val_data[f"TE_{col}_median"] = TE.transform(val_data[col].values)
        test_data[f"TE_{col}_median"] = TE.transform(test_data[col].values)
    
    del TE
    gc.collect()

    print(train_data.shape)

    xgb_model1 = xgb.XGBRegressor(
        random_state=42,
        device='cuda',
        max_depth= 7,
        colsample_bytree= 0.7,
        subsample= 0.85,
        n_estimators= 3000,
        learning_rate= 0.01,
        min_child_weight= 25,
        enable_categorical= True,
        reg_lambda= 8.0,
        reg_alpha= 1.8,
        early_stopping_rounds=100
    )

    catboost_model1 = CatBoostRegressor(
        random_seed=42,
        iterations=5000,
        task_type='GPU',
        devices='0',
        learning_rate=0.02,
        l2_leaf_reg=4.5,
        use_best_model=True,
        depth=9,
        gpu_cat_features_storage='CpuPinnedMemory'
    )

    lgbm_model1 = LGBMRegressor(
        device='gpu',
        random_state=42,
        early_stopping_round=100,
        verbosity=-1,
        n_estimators=5000,
        objective='regression_l2',
        learning_rate=0.01,
        num_leaves=40,
        reg_alpha= 0.7,
        reg_lambda= 1.2
    )
    print("Model initialized.")

    catboost_model1.fit(
        train_data.drop(columns=['Price']), train_data['Price'].values,
        eval_set=[(val_data.drop(columns=['Price']), val_data['Price'].values)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    xgb_model1.fit(
        train_data.drop(columns=['Price']), train_data['Price'].values,
        eval_set=[(val_data.drop(columns=['Price']), val_data['Price'].values)],
        verbose=100
    )

    lgbm_model1.fit(
        train_data.drop(columns=['Price']), train_data['Price'].values,
        eval_set=[(val_data.drop(columns=['Price']), val_data['Price'].values)],
        callbacks=[log_evaluation(100)],
        eval_metric='rmse',
    )

    aval = np.linspace(0,1,101)
    bval = np.linspace(0,1,101)
    
    xgbpred = xgb_model1.predict(val_data.drop(columns=['Price']))
    catpred = catboost_model1.predict(val_data.drop(columns=['Price']))
    lgbpred = lgbm_model1.predict(val_data.drop(columns=['Price']))

    y_true = val_data['Price'].values
    best_rmse = float('inf')
    best_a, best_b, best_c = 0, 0, 0
    for a in tqdm(aval):
        for b in bval:
            c = 1 - a - b
            if c < 0 or c > 1:
                continue  # Skip invalid weights
    
            # Compute ensemble prediction
            ensemble_pred = a * xgbpred + b * catpred + c * lgbpred
    
            # Compute RMSE
            rmse = np.sqrt(np.mean((ensemble_pred - y_true) ** 2))
    
            # Store best weights
            if rmse < best_rmse:
                best_rmse = rmse
                best_a, best_b, best_c = a, b, c    

    ensemble_weight.append([best_a, best_b, best_c])
    val_prediction = xgb_model1.predict(val_data.drop(columns=['Price'])) * best_a + catboost_model1.predict(val_data.drop(columns=['Price'])) * best_b + lgbm_model1.predict(val_data.drop(columns=['Price'])) * best_c
    test_prediction_fold = xgb_model1.predict(test_data.drop(columns=['Price', 'id'])) * best_a + catboost_model1.predict(test_data.drop(columns=['Price', 'id'])) * best_b + lgbm_model1.predict(test_data.drop(columns=['Price', 'id'])) * best_c
    
    val_rmse = np.sqrt(sklearn.metrics.mean_squared_error(val_data['Price'].values, val_prediction))
    val_rmse_all_folds.append(val_rmse)
    print(f"Fold {fold}, validation rmse = {val_rmse}, ensemble weighting = {best_a} {best_b} {best_c}")
    test_prediction.append(test_prediction_fold)
    del train_data
    del val_data
    del test_data
    del xgb_model1
    del catboost_model1
    del lgbm_model1
    gc.collect()


sum(np.array(val_rmse_all_folds))/len(val_rmse_all_folds)


id_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")['id']
submission = pd.DataFrame({
    'id': id_submission,
    'Price': np.average(np.array(test_prediction), axis=0)
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")


submission




