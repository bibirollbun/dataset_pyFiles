import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

def add_feature_cross_terms(df, features): 
    df = df.copy() 
    df = df.loc[:, ~df.columns.duplicated()]  
    
    if 'Height' in features and 'Weight' in features:
        df["BMI"] = df['Weight'] / (df['Height'] * df['Height'])
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
    return df_new

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new


# å�Ÿå§‹æ•°å€¼ç‰¹å¾�ï¼ˆå�«Heightã€�Weightï¼‰
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# 1. ç”Ÿæˆ�BMIï¼ˆåŸºäº�å�Ÿå§‹æ•°å€¼ç‰¹å¾�ä¸­çš„Heightã€�Weightï¼‰
train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)
new_numerical_features = [f for f in numerical_features if f not in ['Height', 'Weight']] + ['BMI']

# new_numerical_features = numerical_features
# æ­¤æ—¶ new_numerical_features = ['Age', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']

# 3. ç”Ÿæˆ�äº¤äº’ç‰¹å¾�ï¼ˆä»…åŸºäº�æ–°åˆ—è¡¨ï¼Œæ— Heightã€�Weightçš„äº¤äº’ï¼‰
train = add_interaction_features(train, new_numerical_features)
test = add_interaction_features(test, new_numerical_features)

# 4. ç”Ÿæˆ�ç»Ÿè®¡ç‰¹å¾�ï¼ˆå�¯é€‰æ‹©å�Ÿå§‹æ•°å€¼ç‰¹å¾�æˆ–æ–°åˆ—è¡¨ï¼Œè¿™é‡Œä¿�ç•™å�Ÿå§‹æ•°å€¼ç‰¹å¾�å�šç»Ÿè®¡ï¼‰
train = add_statistical_features(train, numerical_features)
test = add_statistical_features(test, numerical_features)

# 5. å¤„ç�†ç±»åˆ«ç‰¹å¾�Sex
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])
train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

# 6. å¤šé¡¹å¼�ç‰¹å¾�ï¼ˆä»…åŸºäº�æ–°åˆ—è¡¨ï¼Œé�¿å…�Heightã€�Weightçš„ä¹˜ç§¯ï¼‰
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_train = poly.fit_transform(train[new_numerical_features])
poly_test = poly.transform(test[new_numerical_features])
poly_feature_names = poly.get_feature_names_out(new_numerical_features)

poly_train_df = pd.DataFrame(poly_train, columns=poly_feature_names)
poly_test_df = pd.DataFrame(poly_test, columns=poly_feature_names)

# 7. å�ˆå¹¶æ‰€æœ‰ç‰¹å¾�
train = pd.concat([train.reset_index(drop=True), poly_train_df], axis=1)
test = pd.concat([test.reset_index(drop=True), poly_test_df], axis=1)

# å‡†å¤‡æ¨¡å�‹è¾“å…¥
X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  
X_test = test.drop(columns=['id'])

FEATURES = X.columns.tolist()


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# # ====================== 1. åŸºç¡€ç‰¹å¾�å®šä¹‰ï¼ˆéœ€æ��å‰�å‡†å¤‡ï¼‰ ======================
# # numerical_features: å�Ÿå§‹æ•°å€¼ç‰¹å¾�åˆ—è¡¨ï¼ˆå¦‚èº«é«˜ã€�ä½“é‡�ã€�æ—¶é•¿ç­‰ï¼‰
# numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']  

# # train/test: å�Ÿå§‹æ•°æ�®é›†ï¼ˆå�«idã€�Calories/Sexç­‰åˆ—ï¼‰
# # å�‡è®¾å·²åŠ è½½ï¼štrain = pd.read_csv('train.csv'), test = pd.read_csv('test.csv')  
# def add_feature_cross_terms(df, features): 
#     df = df.copy() 
#     df = df.loc[:, ~df.columns.duplicated()]  
    
#     # å�ªæ·»åŠ èº«é«˜å’Œä½“é‡�çš„äº¤å�‰é¡¹
#     if 'Height' in features and 'Weight' in features:
#         df["BMI"] = df['Weight'] / (df['Height'] * df['Height'])
    
#     return df

# train = add_feature_cross_terms(train, numerical_features)
# test = add_feature_cross_terms(test, numerical_features)
# # 2. æ�„å»ºâ€œæ–°æ•°å€¼ç‰¹å¾�åˆ—è¡¨â€�ï¼šç§»é™¤Heightã€�Weightï¼ŒåŠ å…¥BMI
# new_numerical_features = [f for f in numerical_features if f not in ['Height', 'Weight']] + ['BMI']
# # ====================== 2. ä»…ä¿�ç•™â€œå�Ÿå§‹ç‰¹å¾� + Sexç¼–ç �â€� ======================
# # 1. å¤„ç�†ç±»åˆ«ç‰¹å¾�Sexï¼ˆLabelEncoderç¼–ç �ï¼‰
# le = LabelEncoder()
# train['Sex'] = le.fit_transform(train['Sex'])
# test['Sex'] = le.transform(test['Sex'])
# train['Sex'] = train['Sex'].astype('category')  # æ ‡è®°ä¸ºç±»åˆ«ç‰¹å¾�ï¼ˆè‹¥æ¨¡å�‹æ”¯æŒ�ï¼‰
# test['Sex'] = test['Sex'].astype('category')

# # 2. æ�„é€ X/X_testï¼šä»…ä¿�ç•™å�Ÿå§‹æ•°å€¼ç‰¹å¾� + ç¼–ç �å��çš„Sexï¼Œä¸”å�»é‡�
# # å�Ÿå§‹ç‰¹å¾�åˆ— = æ•°å€¼ç‰¹å¾� + ç±»åˆ«ç‰¹å¾�(Sex)
# base_features = new_numerical_features + ['Sex']

# X = train.drop(columns=['id', 'Calories'])[base_features].loc[:, ~train[base_features].columns.duplicated()]
# y = np.log1p(train['Calories'])  # ç›®æ ‡å€¼ï¼ˆlog1pè½¬æ�¢ï¼Œå��ç»­è¿˜å�Ÿï¼‰

# X_test = test.drop(columns=['id'])[base_features].loc[:, ~test[base_features].columns.duplicated()]

# FEATURES = X.columns.tolist()  # åŸºç¡€ç‰¹å¾�åˆ—è¡¨ï¼ˆæ— ä»»ä½•è¡�ç”Ÿç‰¹å¾�ï¼‰


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import matplotlib.pyplot as plt
import seaborn as sns

FOLDS = 7  # KæŠ˜æ•°
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42) 

# å®šä¹‰æ¨¡å�‹ï¼ˆæ”¯æŒ�å�•æ¨¡å�‹æˆ–å¤šæ¨¡å�‹ï¼ŒæŒ‰éœ€æ³¨é‡Š/å�–æ¶ˆæ³¨é‡Šï¼‰
models = { 
    # ç¤ºä¾‹ï¼šå¤šæ¨¡å�‹ï¼ˆCatBoost + XGBoostï¼‰
    'CatBoost': CatBoostRegressor(
        task_type='GPU', devices='0', verbose=100, 
        random_seed=42, cat_features=['Sex'], early_stopping_rounds=100
    ), 
    'XGBoost': XGBRegressor(
        tree_method='gpu_hist', gpu_id=0, max_depth=10, 
        colsample_bytree=0.7, subsample=0.9, n_estimators=2000, 
        learning_rate=0.02, gamma=0.01, max_delta_step=2, 
        early_stopping_rounds=100, eval_metric='rmse', 
        enable_categorical=True, random_state=42
    ) 
    # ç¤ºä¾‹ï¼šå�•æ¨¡å�‹ï¼ˆä»…XGBoostï¼‰
    # 'XGBoost': XGBRegressor(...)
}  

# ====================== 2. ç»“æ�œä¸�ç‰¹å¾�é‡�è¦�æ€§åˆ�å§‹åŒ– ======================
results = {name: {'oof': np.zeros(len(X)), 'pred': np.zeros(len(X_test)), 'rmsle': []} for name in models}
feature_importances = {name: [] for name in models}
feature_names = X.columns[~X.columns.duplicated()].tolist()  # å�»é‡�ç‰¹å¾�å��  


# ====================== 3. KæŠ˜è®­ç»ƒå¾ªç�¯ï¼ˆå�«ç‰¹å¾�é‡�è¦�æ€§æ”¶é›†ï¼‰ ======================
for name, model in models.items(): 
    print(f"\n=== Training {name} ===\n") 
    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)): 
        print(f"----- Fold {i+1} è®­ç»ƒå¼€å§‹ -----\n") 
        x_train, y_train = X.iloc[train_idx], y[train_idx] 
        x_valid, y_valid = X.iloc[valid_idx], y[valid_idx] 
        
        # ç‰¹å¾�å�»é‡�ï¼ˆé�¿å…�é‡�å¤�åˆ—å¹²æ‰°ï¼‰
        x_train = x_train.loc[:, ~x_train.columns.duplicated()] 
        x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()] 
        x_test = X_test.loc[:, ~X_test.columns.duplicated()].copy() 

        start = time.time() 
        
        # æ¨¡å�‹è®­ç»ƒï¼ˆæ ¹æ�®æ¨¡å�‹ç±»å�‹é€‚é…�fité€»è¾‘ï¼‰
        if name == 'XGBoost': 
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100) 
        elif name == 'CatBoost': 
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid)) 
        else:  # LightGBMç­‰å…¶ä»–æ¨¡å�‹
            model.fit(x_train, y_train) 

        # ç”Ÿæˆ�é¢„æµ‹ç»“æ�œ
        oof_pred = model.predict(x_valid)  # éªŒè¯�é›†OOFé¢„æµ‹
        test_pred = model.predict(x_test)  # æµ‹è¯•é›†é¢„æµ‹ï¼ˆå��ç»­å¹³å�‡ï¼‰
        
        # å­˜å‚¨ç»“æ�œ
        results[name]['oof'][valid_idx] = oof_pred 
        results[name]['pred'] += test_pred / FOLDS  # æµ‹è¯•é›†é¢„æµ‹å¹³å�‡
        
        # è®¡ç®—å½“å‰�æŠ˜RMSLEï¼ˆè¿˜å�Ÿlog1pï¼‰
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred))) 
        results[name]['rmsle'].append(rmsle) 
        
        # æ‰“å�°è®­ç»ƒä¿¡æ�¯
        print(f"Fold {i+1} RMSLE: {rmsle:.4f}") 
        print(f"Fold {i+1} è®­ç»ƒè€—æ—¶: {time.time() - start:.1f} ç§’\n") 
        
        # æ”¶é›†ç‰¹å¾�é‡�è¦�æ€§
        if name == 'XGBoost': 
            importances = model.feature_importances_
        elif name == 'CatBoost': 
            importances = model.get_feature_importance()
        else:  # å…¶ä»–æ¨¡å�‹é»˜è®¤é€»è¾‘
            importances = model.feature_importances_
        feature_importances[name].append(importances)  


# ====================== 4. æ¨¡å�‹æ€§èƒ½è¯„ä¼° ======================
print("\n=== æ¨¡å�‹æ•´ä½“æ€§èƒ½è¯„ä¼° ===\n") 
for name in models: 
    mean_rmsle = np.mean(results[name]['rmsle']) 
    std_rmsle = np.std(results[name]['rmsle']) 
    print(f"{name} - å¹³å�‡RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")  


# ====================== 5. ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��ä¸�å�¯è§†åŒ– ======================
for name in models:
    print(f"\n=== {name} ç‰¹å¾�é‡�è¦�æ€§åˆ†æ�� ===\n")
    
    # è®¡ç®—å¤šæŠ˜å¹³å�‡é‡�è¦�æ€§
    avg_importances = np.mean(feature_importances[name], axis=0)
    
    # æ�„å»ºTop20ç‰¹å¾�DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': avg_importances
    }).sort_values('Importance', ascending=False).head(20)
    
    # å�¯è§†åŒ–Top20ç‰¹å¾�
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'Top 20 Feature Importance - {name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()  # é�¿å…�æ ‡ç­¾æˆªæ–­
    plt.savefig(f'{name}_feature_importance.png', dpi=300)  # ä¿�å­˜é«˜æ¸…å›¾
    plt.show()
    
    # æ‰“å�°Top10ç‰¹å¾�
    print(f"\n{name} Top 10 é‡�è¦�ç‰¹å¾�:\n")
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")  


# ====================== 6. é¢„æµ‹ä¸�æ��äº¤ï¼ˆè‡ªåŠ¨åˆ¤æ–­å�•/å¤šæ¨¡å�‹ï¼‰ ======================
oof_preds = {name: np.expm1(results[name]['oof']) for name in models}  # è¿˜å�Ÿlog1pçš„OOFé¢„æµ‹
test_preds = {name: np.expm1(results[name]['pred']) for name in models}  # è¿˜å�Ÿlog1pçš„æµ‹è¯•é›†é¢„æµ‹
y_true = np.expm1(y)  # è¿˜å�Ÿlog1pçš„çœŸå®�ç›®æ ‡
model_names = list(models.keys())

if len(models) == 1:
    # ---------- å�•æ¨¡å�‹é€»è¾‘ ----------
    selected_model = model_names[0]
    print(f"\nâœ… ä½¿ç”¨[{selected_model}]å�•æ¨¡å�‹é¢„æµ‹\n") 
    
    # éªŒè¯�é›†æ€§èƒ½
    single_rmsle = np.sqrt(mean_squared_log_error(y_true, oof_preds[selected_model]))
    print(f"{selected_model} éªŒè¯�é›†RMSLE: {single_rmsle:.4f}") 
    
    # æµ‹è¯•é›†é¢„æµ‹ï¼ˆæˆªæ–­èŒƒå›´ï¼‰
    test_pred = test_preds[selected_model]
    test_pred = np.clip(test_pred, 1, 314)  # æŒ‰éœ€è°ƒæ•´èŒƒå›´
    
    # ä¿�å­˜æ��äº¤
    submission['Calories'] = test_pred 
    submission.to_csv('submission.csv', index=False) 
    
    # æ‰“å�°æ��äº¤ä¿¡æ�¯
    print("\næ��äº¤æ–‡ä»¶å‰�5è¡Œ:")
    print(submission.head()) 
    print(f"\né¢„æµ‹å€¼å�‡å€¼: {test_pred.mean():.2f}") 
    print(f"é¢„æµ‹å€¼ä¸­ä½�æ•°: {np.median(test_pred):.2f}") 

else:
    # ---------- å¤šæ¨¡å�‹æ‰‹åŠ¨åŠ æ�ƒé€»è¾‘ ----------
    # æ‰‹åŠ¨è®¾ç½®æ�ƒé‡�ï¼ˆç¤ºä¾‹ï¼Œå�¯æ ¹æ�®éœ€æ±‚è°ƒæ•´ï¼‰
    manual_weights = {
        'CatBoost': 0.4,
        'XGBoost': 0.6
    }
    # ç¡®ä¿�æ�ƒé‡�å’Œä¸º1
    weight_sum = sum(manual_weights.values())
    for name in manual_weights:
        manual_weights[name] /= weight_sum
    
    print(f"\nâœ… æ‰‹åŠ¨è®¾ç½®çš„æ¨¡å�‹æ�ƒé‡�:\n")
    for name, weight in manual_weights.items():
        print(f"{name}: {weight:.4f}") 
    
    # éªŒè¯�é›†è��å�ˆé¢„æµ‹
    blended_oof = np.zeros(len(X))
    for name in model_names:
        blended_oof += manual_weights[name] * oof_preds[name]
    manual_rmsle = np.sqrt(mean_squared_log_error(y_true, blended_oof))
    print(f"\næ‰‹åŠ¨åŠ æ�ƒæ¨¡å�‹ éªŒè¯�é›†RMSLE: {manual_rmsle:.4f}") 
    
    # å�•æ¨¡å�‹éªŒè¯�é›†æ€§èƒ½å¯¹æ¯”
    for name in model_names:
        single_rmsle = np.sqrt(mean_squared_log_error(y_true, oof_preds[name]))
        print(f"{name} éªŒè¯�é›†RMSLE: {single_rmsle:.4f}") 
    
    # æµ‹è¯•é›†è��å�ˆé¢„æµ‹ï¼ˆæˆªæ–­èŒƒå›´ï¼‰
    blended_preds = np.zeros(len(X_test))
    for name in model_names:
        blended_preds += manual_weights[name] * test_preds[name]
    blended_preds = np.clip(blended_preds, 1, 314)  # æŒ‰éœ€è°ƒæ•´èŒƒå›´
    
    # ä¿�å­˜æ��äº¤
    submission['Calories'] = blended_preds 
    submission.to_csv('submission.csv', index=False) 
    
    # æ‰“å�°æ��äº¤ä¿¡æ�¯
    print("\næ��äº¤æ–‡ä»¶å‰�5è¡Œ:")
    print(submission.head()) 
    print(f"\né¢„æµ‹å€¼å�‡å€¼: {blended_preds.mean():.2f}") 
    print(f"é¢„æµ‹å€¼ä¸­ä½�æ•°: {np.median(blended_preds):.2f}") 



# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_log_error
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# import time

# FOLDS = 7
# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# # models = {
# #     'CatBoost': CatBoostRegressor(verbose=100, random_seed=42, cat_features=['Sex'], early_stopping_rounds=100),
# #     'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=2000, learning_rate=0.02,
# #                             gamma=0.01, max_delta_step=2, early_stopping_rounds=100, eval_metric='rmse',
# #                             enable_categorical=True, random_state=42),
# #     'LightGBM': LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
# #                               subsample=0.9, random_state=42, verbose=-1)
# # }
# models = { 
#     'CatBoost': CatBoostRegressor(task_type='GPU', devices='0', verbose=100, 
#                                    random_seed=42, cat_features=['Sex'], early_stopping_rounds=100), 
#     'XGBoost': XGBRegressor(tree_method='gpu_hist', gpu_id=0, max_depth=10, 
#                            colsample_bytree=0.7, subsample=0.9, n_estimators=2000, 
#                            learning_rate=0.02, gamma=0.01, max_delta_step=2, 
#                            early_stopping_rounds=100, eval_metric='rmse', 
#                            enable_categorical=True, random_state=42)
#     ,
#     'LightGBM': LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0,
#                              n_estimators=2000, learning_rate=0.02, max_depth=10, 
#                              colsample_bytree=0.7, subsample=0.9, random_state=42, verbose=-1) 
# } 

# results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in models}
# feature_importances = {name: [] for name in models}
# feature_names = X.columns[~X.columns.duplicated()].tolist()  # å�»é‡�ç‰¹å¾�å��  


# for name, model in models.items():
#     print(f"\n=== Training {name} ===")
#     for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
#         print(f"\nFold {i+1}")
#         x_train, y_train = X.iloc[train_idx], y[train_idx]
#         x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
        
#         x_train = x_train.loc[:, ~x_train.columns.duplicated()]
#         x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()]
#         x_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

#         start = time.time()
        
#         if name == 'XGBoost':
#             model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
#         elif name == 'CatBoost':
#             model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
#         else:
#             model.fit(x_train, y_train)

#         oof_pred = model.predict(x_valid)
#         test_pred = model.predict(x_test)
        
#         results[name]['oof'][valid_idx] = oof_pred
#         results[name]['pred'] += test_pred / FOLDS
        
#         rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
#         results[name]['rmsle'].append(rmsle)
        
#         print(f"Fold {i+1} RMSLE: {rmsle:.4f}")
#         print(f"Training time: {time.time() - start:.1f} sec")
#         # æ”¶é›†ç‰¹å¾�é‡�è¦�æ€§
#         if name == 'XGBoost': 
#             importances = model.feature_importances_
#         elif name == 'CatBoost': 
#             importances = model.get_feature_importance()
#         else:  # å…¶ä»–æ¨¡å�‹é»˜è®¤é€»è¾‘
#             importances = model.feature_importances_
#         feature_importances[name].append(importances)  

# print("\n=== Model Comparison ===")
# for name in models:
#     mean_rmsle = np.mean(results[name]['rmsle'])
#     std_rmsle = np.std(results[name]['rmsle'])
#     print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")



# from sklearn.metrics import mean_squared_log_error

# # å‡†å¤‡é¢„æµ‹ç»“æ�œ
# oof_preds = {name: np.expm1(results[name]['oof']) for name in results}
# test_preds = {name: np.expm1(results[name]['pred']) for name in results}
# y_true = np.expm1(y)

# # ä½¿ç”¨ç®€å�•å¹³å�‡æ�ƒé‡�ï¼ˆæ¯�ä¸ªæ¨¡å�‹æ�ƒé‡�ç›¸ç­‰ï¼‰
# equal_weights = [1/3, 1/3, 1/3]  # ä¸‰ä¸ªæ¨¡å�‹æ�ƒé‡�ç›¸ç­‰

# print(f"\nâœ… å¹³å�‡æ�ƒé‡�:")
# print(f"CatBoost = {equal_weights[0]:.4f}")
# print(f"XGBoost  = {equal_weights[1]:.4f}")
# print(f"LightGBM = {equal_weights[2]:.4f}")

# # è®¡ç®—éªŒè¯�é›†ä¸Šçš„RMSLEï¼ˆå�¯é€‰ï¼Œç”¨äº�è¯„ä¼°å¹³å�‡åŠ æ�ƒæ•ˆæ�œï¼‰
# blended_oof = (
#     equal_weights[0] * oof_preds['CatBoost'] +
#     equal_weights[1] * oof_preds['XGBoost'] +
#     equal_weights[2] * oof_preds['LightGBM']
# )
# rmsle_score = np.sqrt(mean_squared_log_error(y_true, blended_oof))
# print(f"å¹³å�‡åŠ æ�ƒçš„éªŒè¯�é›†RMSLE: {rmsle_score:.4f}")

# # å¯¹æµ‹è¯•é›†è¿›è¡Œé¢„æµ‹
# blended_preds = (
#     equal_weights[0] * test_preds['CatBoost'] +
#     equal_weights[1] * test_preds['XGBoost'] +
#     equal_weights[2] * test_preds['LightGBM']
# )

# # é™�åˆ¶é¢„æµ‹å€¼èŒƒå›´
# blended_preds = np.clip(blended_preds, 1, 314)

# # ä¿�å­˜ç»“æ�œ
# submission['Calories'] = blended_preds
# submission.to_csv('submission.csv', index=False)

# print("\nSubmission Head:")
# print(submission.head())

# print(f"\nPredict Mean: {blended_preds.mean():.2f}")
# print(f"Predict Median: {np.median(blended_preds):.2f}")





# # ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��å’Œå�¯è§†åŒ–
# for name in models:
#     print(f"\n=== {name} Feature Importance Analysis ===")
    
#     # è®¡ç®—å¹³å�‡ç‰¹å¾�é‡�è¦�æ€§
#     avg_importances = np.mean(feature_importances[name], axis=0)
    
#     # åˆ›å»ºç‰¹å¾�é‡�è¦�æ€§DataFrame
#     importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': avg_importances
#     }).sort_values('Importance', ascending=False).head(10)
    
#     # å�¯è§†åŒ–ç‰¹å¾�é‡�è¦�æ€§
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
#     plt.title(f'Top 20 Feature Importance - {name}')
#     plt.xlabel('Importance Score')
#     plt.ylabel('Feature')
#     plt.tight_layout()
    
#     # ä¿�å­˜å›¾åƒ�
#     plt.savefig(f'{name}_feature_importance.png', dpi=300)
#     plt.show()
    
#     # æ‰“å�°å‰�10ä¸ªé‡�è¦�ç‰¹å¾�
#     print(f"\nTop 10 Features for {name}:")
#     for i, row in importance_df.head(10).iterrows():
#         print(f"{row['Feature']}: {row['Importance']:.4f}")



# from scipy.optimize import minimize
# from sklearn.metrics import mean_squared_log_error

# oof_preds = {name: np.expm1(results[name]['oof']) for name in results}
# test_preds = {name: np.expm1(results[name]['pred']) for name in results}
# y_true = np.expm1(y)

# def rmsle_loss(weights):
#     blended = (
#         weights[0] * oof_preds['CatBoost'] +
#         weights[1] * oof_preds['XGBoost'] +
#         weights[2] * oof_preds['LightGBM']
#     )
#     return np.sqrt(mean_squared_log_error(y_true, blended))

# initial_weights = [1/3, 1/3, 1/3]
# constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
# bounds = [(0, 1)] * 3

# res = minimize(rmsle_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
# best_weights = res.x

# print(f"\nâœ… Optimized Weights:")
# print(f"CatBoost = {best_weights[0]:.4f}")
# print(f"XGBoost  = {best_weights[1]:.4f}")
# print(f"LightGBM = {best_weights[2]:.4f}")

# blended_preds = (
#     best_weights[0] * test_preds['CatBoost'] +
#     best_weights[1] * test_preds['XGBoost'] +
#     best_weights[2] * test_preds['LightGBM']
# )

# blended_preds = np.clip(blended_preds, 1, 314)

# submission['Calories'] = blended_preds
# submission.to_csv('submission.csv', index=False)

# print("\nSubmission Head:")
# print(submission.head())

# print(f"\nPredict Mean: {blended_preds.mean():.2f}")
# print(f"Predict Median: {np.median(blended_preds):.2f}")



# import pandas as pd
# import numpy as np

# df1 = pd.read_csv("/kaggle/input/caloriecast-adaptive-ensemble-engine-for-s5e5/submission.csv")
# df2 = pd.read_csv("/kaggle/input/ensemble-of-solutions/submission.csv")
# df3 = pd.read_csv("/kaggle/input/ps-s5e5-log-blended-cat-xgboost-with-50-fold-cv/ensemble_submission.csv")


# ground_truth = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")  

# ground_truth['Calories'] = (0.4 * df1['Calories']) + (0.3 * df2['Calories'])+(.3 * df3['Calories'])
# ground_truth.to_csv('submission.csv', index=False)

