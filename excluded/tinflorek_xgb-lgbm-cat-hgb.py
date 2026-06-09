import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import DMatrix
from xgboost import XGBRegressor
from catboost import Pool
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import optuna
import warnings

warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


print(f"Train data: {train_data.shape}, Extra train data: {extra_data.shape}")


full_train_data = pd.concat([train_data, extra_data], axis=0, ignore_index=True)


full_train_data.isnull().sum()


def feature_engineering(df):

    df_copy = df.copy()
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df_copy['Size_Num'] = df_copy['Size'].map(size_mapping)
    df_copy['Compartments_per_Size'] = df_copy['Compartments'] / df_copy['Size_Num']    
    df_copy['Weight_per_Compartment'] = df_copy['Weight Capacity (kg)'] / df_copy['Compartments'] 
    df_copy['Waterproof'] = df_copy['Waterproof'].map({'Yes': 1, 'No': 0})
    df_copy['Laptop Compartment'] = df_copy['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df_copy['Waterproof_Laptop'] = df_copy['Waterproof'] * df_copy['Laptop Compartment']
    df_copy['Is_Durable_Material'] = df_copy['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df_copy['Is_Lightweight_Material'] = df_copy['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df_copy['Luxury_Material'] = df_copy['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    df_copy['Professional_Style'] = df_copy['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df_copy['Casual_Style'] = df_copy['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    df_copy['Is_Premium_Brand'] = df_copy['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df_copy['Is_Budget_Brand'] = df_copy['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    df_copy['Is_Small'] = df_copy['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df_copy['Is_Medium'] = df_copy['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df_copy['Is_Large'] = df_copy['Size'].apply(lambda x: 1 if x == 'Large' else 0)
    
    return df_copy

train_data_engineered = feature_engineering(full_train_data)
test_data_engineered = feature_engineering(test_data)


categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)', 'Size_Num', 'Compartments_per_Size', 
                  'Weight_per_Compartment', 'Waterproof_Laptop', 'Is_Durable_Material', 
                  'Is_Lightweight_Material', 'Luxury_Material', 'Professional_Style', 
                  'Casual_Style', 'Is_Premium_Brand', 'Is_Budget_Brand', 
                  'Is_Small', 'Is_Medium', 'Is_Large']


categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_pipeline, categorical_cols),
    ('num', numerical_pipeline, numerical_cols)
])


X = preprocessor.fit_transform(train_data_engineered.drop('Price', axis=1))
y = full_train_data.Price
XX = preprocessor.fit_transform(test_data_engineered)


def feature_finder(model, X, y, model_name, random_state=123):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np

    if model_name == 'lgb':
        import lightgbm as lgb
        early_stop = lgb.early_stopping(200, verbose=0)
    elif model_name == 'xgb':
        import xgboost as xgb
        early_stop = xgb.callback.EarlyStopping(rounds=200)
    elif model_name == 'cat':
        from catboost import CatBoostRegressor
        early_stop = None
    elif model_name == 'hgb':
        early_stop = None
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Use 'lgb', 'xgb', 'cat', or 'hgb'")

    original_feature_count = X.shape[1]
    kept_feature_indices = list(range(original_feature_count))
    
    X_t = X.copy()
    y_t = y.copy()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=random_state)

    if model_name == 'lgb':
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
                 eval_metric='rmse', callbacks=[early_stop])
        weight_list = model.feature_importances_
        
    elif model_name == 'xgb':
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                 eval_metric='rmse', callbacks=[early_stop])
        weight_list = model.feature_importances_
        
    elif model_name == 'cat':
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                 early_stopping_rounds=200, verbose=False)
        weight_list = model.get_feature_importance()
        
    elif model_name == 'hgb':
        model.fit(X_train, y_train)
        
        weight_list = model.feature_importances_.copy() if hasattr(model, 'feature_importances_') else None

        if weight_list is None:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=random_state)
            weight_list = result.importances_mean
    
    y_pred = model.predict(X_valid)

    feature_importance_with_index = [(importance, idx) for idx, importance in enumerate(weight_list)]

    feature_importance_with_index.sort(key=lambda x: x[0])
    
    score_t = mean_squared_error(y_valid, y_pred, squared=False)
    print(f'Initial score with {original_feature_count} features: {score_t:.4f}')

    to_check = feature_importance_with_index.copy()
    removed_indices = []
    
    for importance, original_idx in to_check:
        if importance == 0:
            kept_feature_indices.remove(original_idx)
            removed_indices.append(original_idx)
            continue

        current_indices = [idx for idx in range(X_t.shape[1]) if idx not in removed_indices and idx != original_idx]
        temp_X = X_t[:, current_indices]

        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
            temp_X, y_t, test_size=0.20, random_state=random_state)

        if model_name == 'lgb':
            model.fit(X_train_reduced, y_train_reduced, 
                     eval_set=[(X_test_reduced, y_test_reduced)],
                     eval_metric='rmse', callbacks=[early_stop])
            
        elif model_name == 'xgb':
            model.fit(X_train_reduced, y_train_reduced,
                     eval_set=[(X_test_reduced, y_test_reduced)],
                     eval_metric='rmse', callbacks=[early_stop])
            
        elif model_name == 'cat':
            model.fit(X_train_reduced, y_train_reduced,
                     eval_set=(X_test_reduced, y_test_reduced),
                     early_stopping_rounds=200, verbose=False)
            
        elif model_name == 'hgb':
            model.fit(X_train_reduced, y_train_reduced)
        
        pred_i = model.predict(X_test_reduced)
        score_i = mean_squared_error(y_test_reduced, pred_i, squared=False)
        
        if score_i <= score_t:
            kept_feature_indices.remove(original_idx)
            removed_indices.append(original_idx)
            X_t = temp_X
            score_t = score_i
            print(f'Removed feature {original_idx}, remaining features: {len(kept_feature_indices)}, new score: {score_i:.4f}')

    final_X = X[:, kept_feature_indices]
    X_train_final, X_valid_final, y_train_final, y_valid_final = train_test_split(
        final_X, y, test_size=0.20, random_state=random_state)
    
    if model_name == 'lgb':
        model.fit(X_train_final, y_train_final, 
                 eval_set=[(X_valid_final, y_valid_final)],
                 eval_metric='rmse', callbacks=[early_stop])
        final_importances = model.feature_importances_
        
    elif model_name == 'xgb':
        model.fit(X_train_final, y_train_final,
                 eval_set=[(X_valid_final, y_valid_final)],
                 eval_metric='rmse', callbacks=[early_stop])
        final_importances = model.feature_importances_
        
    elif model_name == 'cat':
        model.fit(X_train_final, y_train_final,
                 eval_set=(X_valid_final, y_valid_final),
                 early_stopping_rounds=200, verbose=False)
        final_importances = model.get_feature_importance()
        
    elif model_name == 'hgb':
        model.fit(X_train_final, y_train_final)
        
        if hasattr(model, 'feature_importances_'):
            final_importances = model.feature_importances_
        else:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X_valid_final, y_valid_final, n_repeats=10, random_state=random_state)
            final_importances = result.importances_mean

    final_importance_with_index = [(final_importances[i], kept_feature_indices[i]) for i in range(len(kept_feature_indices))]

    final_importance_with_index.sort(key=lambda x: x[0], reverse=True)

    sorted_kept_feature_indices = [idx for _, idx in final_importance_with_index]
    
    return sorted_kept_feature_indices


def training(model, X, y, test, n_splits=5, random_state=42, verbose=0, model_name=None):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_rmse = []
    oof_test_preds = np.zeros(test.shape[0])
    oof_train_preds = np.zeros(len(y))

    if hasattr(X, 'toarray'):
        X = X.toarray()
    if hasattr(test, 'toarray'):
        test = test.toarray()

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        if model_name == 'xgb':
            model.fit(
                X_train, y_train, 
                eval_set=[(X_valid, y_valid)], 
                early_stopping_rounds=50, 
                verbose=verbose
            )
    
            booster = model.get_booster()

            if hasattr(model, 'best_ntree_limit'):
                best_ntree_limit = model.best_ntree_limit
            elif hasattr(model, 'best_iteration_'):
                best_ntree_limit = model.best_iteration_ + 1
            else:
                best_ntree_limit = model.n_estimators
            
            num_boosted_rounds = booster.num_boosted_rounds()

            best_ntree_limit = min(best_ntree_limit, num_boosted_rounds)

            y_pred = booster.predict(DMatrix(X_valid), iteration_range=(0, best_ntree_limit))
            test_pred = booster.predict(DMatrix(test), iteration_range=(0, best_ntree_limit))
            oof_train_preds[train_idx] = booster.predict(DMatrix(X_train), iteration_range=(0, best_ntree_limit))
        elif model_name == 'cat':
            trainPool = Pool(X_train ,y_train)
            testPool = Pool(test)
            validPool = Pool(X_valid, y_valid)

            model.fit(X=trainPool, eval_set=validPool, verbose=verbose, early_stopping_rounds=200)
            y_pred = model.predict(validPool)
            test_pred = model.predict(testPool)
            oof_train_preds[train_idx] = model.predict(Pool(X_train))
        elif model_name == 'hgb':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            test_pred = model.predict(test)
            oof_train_preds[train_idx] = model.predict(X_train)
        else:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[early_stopping(200, verbose=0)])
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            test_pred = model.predict(test, num_iteration=model.best_iteration_)
            oof_train_preds[train_idx] = model.predict(X_train, num_iteration=model.best_iteration_)

        oof_test_preds += test_pred
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        print(f"Fold {fold+1} --> RMSE: {rmse:.4f}")
        oof_rmse.append(rmse)

    print()
    print(f"Average Fold RMSE: {np.mean(oof_rmse):.4f}")
    return oof_test_preds/n_splits, oof_train_preds



test_preds, train_preds = pd.DataFrame(), pd.DataFrame()


xgb_params = {
    "device": "cuda",
    'max_depth': 3,
    'colsample_bytree': 0.6951021006703662,
    'subsample': 0.5389225032134102,
    'n_estimators': 3000,
    'learning_rate': 0.05711415050908186,
    'min_child_weight': 6,
    'enable_categorical': False,
    'reg_lambda': 2.750273397406342e-08,
    'reg_alpha': 7.686413430732185e-06}

xgb = XGBRegressor(**xgb_params)

#print('-------Feature Importance-------')
#features = feature_finder(xgb, X, y, model_name='xgb')
#print('------------Training------------')
#test_preds['xgb'], train_preds['xgb'] = training(xgb, X[:, features], y, XX[:, features], random_state=0, verbose=0, model_name='xgb')


lgb = LGBMRegressor(verbosity=-1, n_estimators=5000, learning_rate=0.1, device='gpu')

#print('-------Feature Importance-------')
#features = feature_finder(lgb, X, y, model_name='lgb')
#print('------------Training------------')
#test_preds['lgb'], train_preds['lgb'] = training(lgb, X[:, features], y, XX[:, features], random_state=101, model_name='lgb')


cat_params = {
    'n_estimators': 10000,
    'learning_rate': 0.05, 
    'verbose': False, 
    'task_type': 'GPU',
    'allow_writing_files': False,
}

cat = CatBoostRegressor(**cat_params)

print('-------Feature Importance-------')
features = feature_finder(cat, X, y, model_name='cat')
print('------------Training------------')
test_preds['cat'], train_preds['cat'] = training(cat, X[:, features], y, XX[:, features], random_state=21, model_name='cat')


hgb = HistGradientBoostingRegressor(max_iter=1000)

#print('-------Feature Importance-------')
#features = feature_finder(hgb, X, y, model_name='hgb')
#print('------------Training------------')
#test_preds['hgb'], train_preds['hgb'] = training(hgb, X[:, features], y, XX[:, features], random_state=0, model_name='hgb')


def objective(trial):
    lw = trial.suggest_float('lgb', 0.1, 5)
    cw = trial.suggest_float('cat', 0.1, 5)
    hw = trial.suggest_float('hgb', 0.1, 5)
    xw = trial.suggest_float('xgb', 0.1, 5)

    pred = np.average(train_preds.to_numpy(), weights=[lw, cw, hw, xw], axis=1)

    score = mean_absolute_percentage_error(y, pred)
    return score

#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=500)

#print(study.best_params)

#weights = study.best_params

#test_pred = np.average(test_preds.to_numpy(), weights=list(weights.values()),axis=1)


sub = pd.DataFrame({'id': test_data.index, 'Price': test_preds['cat']})
sub.to_csv('submission.csv', index=False)

