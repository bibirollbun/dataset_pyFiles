%load_ext cudf.pandas

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
import time
print(f"Using XGBoost version", xgb.__version__)
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv").drop("id", axis=1)
print("Train Shape: ", train.shape)
display(train.head())

test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv").drop("id", axis=1)
print("Test Shape: ", test.shape)
display(test.head())

original = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
print("Original Shape :", original.shape)
display(original.head())

train_plus = pd.concat([train, original], axis=0, ignore_index=True)
train_plus = train_plus.drop_duplicates()
train_plus = train_plus.dropna(subset="Listening_Time_minutes")
print("Train + Original data shape:", train_plus.shape)


def preprocessing_transform(train_plus, test):
    df = pd.concat([train_plus, test], axis=0, ignore_index=True)
    df["Number_of_Ads"] = df["Number_of_Ads"].astype("Float64").astype("Int64")
    cat = []
    num = []
    target = "Listening_Time_minutes"
    for c in df.columns:
        if c not in target:
            if df[c].dtype == "object" or df[c].dtype == "Int64":
                cat.append(c)
    print("-"*50)
    print(f"There are {len(cat)} categorical features:\n{cat}\n")

    num = [c for c in df.columns if df[c].dtype in [float] and c not in target]
    print(f"There are {len(num)} numerical features:\n{num}\n")
    print("-"*50)
    features = cat + num

    high_cardinality = []
    low_cardinality = []
    for c in cat:
        df[c] = df[c].fillna(np.nan)
        df[c], _ = df[c].factorize()
        df[c] -= df[c].min()
        n = df[c].nunique()
        print(f"{c} has {n} unique values")
        if n >=12: high_cardinality.append(c)
        else: low_cardinality.append(c)
    print()
    print(f"High cardinality features:\n{high_cardinality}")
    print(f"Low cardinality features:\n{low_cardinality}")
    
    print("-"*50)

    int_col = df.select_dtypes(include=int).columns.tolist()
    for col in int_col:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"The integer {col} has [Min: {min_val} Max: {max_val}]")
        # Choose appropriate type based on value range
        if min_val >= 0:
            if max_val < 256:
                print(f"{col} transformed to unit8\n")
                
                df[col] = df[col].astype('uint8')
    
            elif max_val < 65536:
                print(f"{col} transformed to uint16\n")
                df[col] = df[col].astype('uint16')
                
            elif max_val < 4294967296:
                print(f"{col} transformed to uint32\n")
                df[col] = df[col].astype('uint32')
                
        else:
            if min_val > -128 and max_val < 128:
                print(f"{col} transformed to int8\n")
                df[col] = df[col].astype('int8')
                
            elif min_val > -32768 and max_val < 32768:
                print(f"{col} transformed to int16\n")
                df[col] = df[col].astype('int16')
                
            elif min_val > -2147483648 and max_val < 2147483648:
                print(f"{col} transformed to int32\n")
                df[col] = df[col].astype('int32')
    
    print("-"*50)

    # Convert float64 columns to float32
    float_col = df.select_dtypes(include=float).columns.tolist()
    for col in float_col:
        df[col] = df[col].astype("float32")
    
    
    train_plus = df.iloc[:len(train_plus)].copy()
    test = df.iloc[len(train_plus):].copy()
    
   
    return train_plus, test, features, cat, num, high_cardinality, low_cardinality
    


train, test, features, cat, num, high_cardinality, low_cardinality = preprocessing_transform(train_plus, test)


def target_encode(train, valid, test, col, target, kfold=5, smooth=20, agg="mean"):
    # Set up cv folds
    train['kfold'] = ((train.index) % kfold) 
    
    # Create output col name
    col_name = '_'.join(col)
    train[f'TE_{agg.upper()}_' + col_name] = 0.

    # Global stats of the target variable based on agg parameter
    if agg=="mean": mn = train[target].mean()
    elif agg=="median": mn = train[target].median()
    elif agg=="min": mn = train[target].min()
    elif agg=="max": mn = train[target].max()
    elif agg=="nunique": mn = 0
    
    for i in range(kfold):
        # Get all the data except current fold to calculate encoding values
        df_tmp = train[train['kfold']!=i]
        # Group by col and calculate agg stats
        df_tmp = df_tmp[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
        df_tmp.columns = col + [agg, 'count']
        # Compute the smoothed encoding value
        if agg=="nunique":
            # Divides the number of unique values by the count
            df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
        else:
            # Bayesian smoothing
            df_tmp['TE_tmp'] = ((df_tmp[agg]*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
            # The smoothing prevent overfitting by "pulling" values from rare categories toward the global mean
        
        # Apply encoding to the current kold
        # df_tmp_m (subset of col, kfold, te col) left join (keeps all rows from the left (train)) with df_tmp
        df_tmp_m = train[col + ['kfold', f'TE_{agg.upper()}_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col)
        # Select the rows that belong to current fold (i) and update TE values with TE_tmp
        # Ensure that the encoding for each fold is based only on data from other folds
        df_tmp_m.loc[df_tmp_m['kfold']==i, f'TE_{agg.upper()}_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold']==i, 'TE_tmp']
        # Update the TE Col in the original dataframe train and replace NANS with the global statistic
        train[f'TE_{agg.upper()}_' + col_name] = df_tmp_m[f'TE_{agg.upper()}_' + col_name].fillna(mn).values  

    # After CV encoding, encodes the validation and test data using the entire training set
    # Groupby col and target and calculate agg stats + count
    df_tmp = train[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
    df_tmp.columns = col + [agg, 'count']
    
    # smoothed encoding value
    if agg=="nunique":
        df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
    else:
        # Bayesian
        df_tmp['TE_tmp'] = ((df_tmp[agg]*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
        
    # Apply encoding to validation set
    df_tmp_m = valid[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    valid[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    valid[f'TE_{agg.upper()}_' + col_name] = valid[f'TE_{agg.upper()}_' + col_name].astype("float32")
    
    # Apply encoding to validation set
    df_tmp_m = test[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    test[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    test[f'TE_{agg.upper()}_' + col_name] = test[f'TE_{agg.upper()}_' + col_name].astype("float32")
    
    # Drop kfold col and convert te values to float32 
    train = train.drop('kfold', axis=1)
    train[f'TE_{agg.upper()}_' + col_name] = train[f'TE_{agg.upper()}_' + col_name].astype("float32")

    return train, valid, test


# Baseline cv model function without te
def xgb_bs_oof_pred(train, test, ksplits=10):
    target = "Listening_Time_minutes"
    folds = ksplits
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=250)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    for f, (train_idx, valid_idx) in enumerate(kf.split(train)):
        
        print("-"*50)
        print(f"#### Fold {f + 1} ####")
        print("-"*50)
    
        X_train = train.loc[train_idx, features].copy()
        y_train = train.loc[train_idx, target]
        X_valid = train.loc[valid_idx, features].copy()
        y_valid = train.loc[valid_idx, target]
        x_test = test[features].copy()
    
        model = XGBRegressor(
                max_depth=11, 
                colsample_bytree=0.8, 
                subsample=0.96, 
                n_estimators=10000, 
                learning_rate=0.01, 
                early_stopping_rounds=100,  
                eval_metric="rmse",
                reg_alpha = 0.98,
                reg_lambda = 0.12,
                device="cuda:0"
            )
        model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],   
                verbose=600
            )
        # Store out-of-fold predictions
        oof[valid_idx] = model.predict(X_valid)
            
        # Add test predictions
        pred += model.predict(x_test)
            
        # Evaluate fold performance
        fold_score = np.sqrt(np.mean((y_valid - oof[valid_idx])**2))
        print(f"Fold {f+1} Score: {fold_score:.5f}")
            
        # Explicitly clean up memory
        import gc
        del X_train, X_valid, x_test, model
        gc.collect()
        
    # Average test predictions
    pred /= folds
        
    # Calculate overall CV score
    overall_score = np.sqrt(np.mean((train[target].values - oof)**2))
    print(f"Overall CV Score: {overall_score:.5f}")
    
    # Save oof and pred in a numpy array
    np.save(f"xgb_bs_CV{ksplits}F_oof.npy", oof)
    np.save(f"xgb_bs_CV{ksplits}F_pred.npy", pred)


# Baseline cv model function with te
def xgb_get_oof_pred_te(train, test, ksplits=10):
    target = "Listening_Time_minutes"
    folds = ksplits
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=250)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    for f, (train_idx, valid_idx) in enumerate(kf.split(train)):
        
        print("-"*50)
        print(f"#### Fold {f + 1} ####")
        print("-"*50)
    
        X_train = train.loc[train_idx, features + [target]].copy()
        y_train = train.loc[train_idx, target]
        X_valid = train.loc[valid_idx, features].copy()
        y_valid = train.loc[valid_idx, target]
        x_test = test[features].copy()

        start = time.time()
        for column in cat:
            col = [column] # Wrap column in a list for single column encoding
            
            if column in low_cardinality:
                X_train, X_valid, x_test = target_encode(X_train, X_valid, x_test, col, target=target, kfold=10, smooth=40, agg="mean")
    
            elif column in high_cardinality:
                X_train, X_valid, x_test = target_encode(X_train, X_valid, x_test, col, target=target, kfold=10, smooth=40, agg="mean")

        end = time.time()
        elapsed = end - start
        print(f"Feature Engineering took {elapsed:.1f} seconds")

        # Remove target from training features
        X_train = X_train.drop(target, axis=1)
        #X_train = X_train.drop([target] + cat, axis=1)
        #X_valid = X_valid.drop(cat, axis=1)
        #x_test = x_test.drop(cat, axis=1)
        
        print("X_train Shape:", X_train.shape, y_train.shape)
        print("X_valid Shape:", X_valid.shape, y_valid.shape)
        print("x_test Shape:", x_test.shape)

        print(f"{len(X_train.columns.tolist())} features used:\n{X_train.columns.tolist()}\n")
        

        model = XGBRegressor(
                max_depth=11, 
                colsample_bytree=0.8, 
                subsample=0.96, 
                n_estimators=10000, 
                learning_rate=0.01, 
                early_stopping_rounds=100,  
                eval_metric="rmse",
                reg_alpha = 0.98,
                reg_lambda = 0.12,
                device="cuda:0"
            )
        
        model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],   
                verbose=600
            )
        
        # Store out-of-fold predictions
        oof[valid_idx] = model.predict(X_valid)
            
        # Add test predictions
        pred += model.predict(x_test)
            
        # Evaluate fold performance
        fold_score = np.sqrt(np.mean((y_valid - oof[valid_idx])**2))
        print(f"Fold {f+1} Score: {fold_score:.5f}")
            
        # Explicitly clean up memory
        import gc
        del X_train, X_valid, x_test, model
        gc.collect()
        
    # Average test predictions
    pred /= folds
        
    # Calculate overall CV score
    overall_score = np.sqrt(np.mean((train[target].values - oof)**2))
    print(f"Overall CV Score: {overall_score:.5f}")
    
    # Save oof and pred in a numpy array
    np.save(f"xgb_te_CV{ksplits}F_oof.npy", oof)
    np.save(f"xgb_te_CV{ksplits}F_pred.npy", pred)


# Cv model function with te only in high cardinality features

def xgb_get_oof_pred_te_highc(train, test, ksplits=10):
    target = "Listening_Time_minutes"
    folds = ksplits
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=250)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    for f, (train_idx, valid_idx) in enumerate(kf.split(train)):
        
        print("-"*50)
        print(f"#### Fold {f + 1} ####")
        print("-"*50)
    
        X_train = train.loc[train_idx, features + [target]].copy()
        y_train = train.loc[train_idx, target]
        X_valid = train.loc[valid_idx, features].copy()
        y_valid = train.loc[valid_idx, target]
        x_test = test[features].copy()

        start = time.time()
        for column in cat:
            col = [column] # Wrap column in a list for single column encoding
            
            if column in low_cardinality: pass
            elif column in high_cardinality:
                X_train, X_valid, x_test = target_encode(X_train, X_valid, x_test, col, target=target, kfold=10, smooth=40, agg="mean")

        end = time.time()
        elapsed = end - start
        print(f"Feature Engineering took {elapsed:.1f} seconds")

        # Remove target from training features
        X_train = X_train.drop(target, axis=1)
        #X_train = X_train.drop([target] + cat, axis=1)
        #X_valid = X_valid.drop(cat, axis=1)
        #x_test = x_test.drop(cat, axis=1)
        
        print("X_train Shape:", X_train.shape, y_train.shape)
        print("X_valid Shape:", X_valid.shape, y_valid.shape)
        print("x_test Shape:", x_test.shape)

        print(f"{len(X_train.columns.tolist())} features used:\n{X_train.columns.tolist()}\n")
        

        model = XGBRegressor(
                max_depth=11, 
                colsample_bytree=0.8, 
                subsample=0.96, 
                n_estimators=10000, 
                learning_rate=0.01, 
                early_stopping_rounds=100,  
                eval_metric="rmse",
                reg_alpha = 0.98,
                reg_lambda = 0.12,
                device="cuda:0"
            )
        
        model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],   
                verbose=600
            )
        
        # Store out-of-fold predictions
        oof[valid_idx] = model.predict(X_valid)
            
        # Add test predictions
        pred += model.predict(x_test)
            
        # Evaluate fold performance
        fold_score = np.sqrt(np.mean((y_valid - oof[valid_idx])**2))
        print(f"Fold {f+1} Score: {fold_score:.5f}")
            
        # Explicitly clean up memory
        import gc
        del X_train, X_valid, x_test, model
        gc.collect()
        
    # Average test predictions
    pred /= folds
        
    # Calculate overall CV score
    overall_score = np.sqrt(np.mean((train[target].values - oof)**2))
    print(f"Overall CV Score: {overall_score:.5f}")
    
    # Save oof and pred in a numpy array
    np.save(f"xgb_te_hc_CV{ksplits}F_oof.npy", oof)
    np.save(f"xgb_bte_hc_CV{ksplits}F_pred.npy", pred)


xgb_bs_oof_pred(train, test)


xgb_get_oof_pred_te(train, test, ksplits=10)


xgb_get_oof_pred_te_highc(train, test, ksplits=10)

