import pandas as pd
df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_add = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')

df_full = pd.concat([df_train,df_test])
df_full = pd.concat([df_full,df_add])


import numpy as np

for i in range(1, len(df_full)):
    if pd.isna(df_full.iloc[i]['id']):
        df_full.iat[i, df_full.columns.get_loc('id')] = df_full.iloc[i-1]['id'] + 1


df_full.drop(columns=['Personality']).isnull().sum(axis=1).value_counts().sort_index()


# 不处理的列（id 和最终预测目标）
exclude_cols = ['id', 'Personality']

# 遍历所有其他列，添加缺失标志列
for col in df_full.columns:
    if col not in exclude_cols:
        missing_col_name = col + '_missing'
        df_full[missing_col_name] = df_full[col].isnull().astype(int)


# 填入缺失值为字符串 'None'
df_full['Stage_fear'] = df_full['Stage_fear'].fillna('None')
df_full['Drained_after_socializing'] = df_full['Drained_after_socializing'].fillna('None')


df_full.drop(columns=['Personality']).isnull().sum(axis=1).value_counts().sort_index()


import pandas as pd
import numpy as np

numerical_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]
cat_cols = ['Stage_fear', 'Drained_after_socializing']
all_cols = numerical_cols + cat_cols

# 复制数据用于填补
df_imputed = df_full.copy()

# 多个分箱策略，从细到粗
bins_list = [
    {
        'Time_spent_Alone': [0, 3, 7, 11],
        'Social_event_attendance': [0, 3, 6, 10],
        'Going_outside': [0, 3, 7],
        'Friends_circle_size': [0, 4, 10, 15],
        'Post_frequency': [0, 3, 6, 10]
    },
    {
        'Time_spent_Alone': [0, 5, 11],
        'Social_event_attendance': [0, 5, 10],
        'Going_outside': [0,3,7],
        'Friends_circle_size': [0, 8, 15],
        'Post_frequency': [0, 5, 10]
    },
    {
        'Time_spent_Alone': [0, 11],
        'Social_event_attendance': [0, 10],
        'Going_outside': [0, 7],
        'Friends_circle_size': [0, 15],
        'Post_frequency': [0, 10]
    }
]

# 分多轮执行填补
for bins_dict in bins_list:
    # 1. 复制数据用于分箱
    df_binned = df_imputed.copy()

    # 2. 按当前 bins_dict 分箱
    for col in numerical_cols:
        bins = bins_dict[col]
        df_binned[col + '_bin'] = pd.cut(df_binned[col], bins=bins, include_lowest=True)

    # 3. 类别列转字符串
    for col in cat_cols:
        df_binned[col + '_bin'] = df_binned[col].astype(str)

    # 4. 遍历每一行，填补尚未填补的缺失值
    for i in range(len(df_binned)):
        missing_cols = [col for col in all_cols if pd.isnull(df_imputed.iloc[i][col])]
        if not missing_cols:
            continue

        available_cols = [col for col in all_cols if col not in missing_cols]
        group_keys = {col + '_bin': df_binned.iloc[i][col + '_bin'] for col in available_cols}

        mask = pd.Series(True, index=df_binned.index)
        for k, v in group_keys.items():
            mask &= df_binned[k].eq(v)

        group_df = df_imputed.loc[mask]

        for col in missing_cols:
            if group_df[col].notnull().any():
                median_val = group_df[col].median()
                df_imputed.iat[i, df_imputed.columns.get_loc(col)] = median_val

# 最终检查
print(df_imputed[all_cols].isnull().sum())



#import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
#import numpy as np

# 假设填补后的完整数据是 df_imputed
#df_final = df_imputed.copy()

# 1. 分为训练集和测试集
#filledtrain = df_final[df_final['Personality'].notnull()].copy()
#filledtest = df_final[df_final['Personality'].isnull()].copy()

# 2. 准备特征列（去掉 id, Personality, 以及 *_missing 列）
#drop_cols = ['id', 'Personality'] + [col for col in df_final.columns if col.endswith('_missing')]
#feature_cols = [col for col in df_final.columns if col not in drop_cols]

# 3. 拆分出数值列和分类列
#X_raw = filledtrain[feature_cols]
#X_test_raw = filledtest[feature_cols]

#num_features = [col for col in X_raw.columns if X_raw[col].dtype in ['int64', 'float64']]
#cat_features = [col for col in X_raw.columns if col not in num_features]

# 4. One-Hot 编码分类特征
#X = pd.get_dummies(X_raw, columns=cat_features)
#X_test = pd.get_dummies(X_test_raw, columns=cat_features)

# 5. 对齐 train/test 特征列
#X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 6. 目标变量编码
#le = LabelEncoder()
#y = le.fit_transform(filledtrain['Personality'])  # Introvert -> 0, Extrovert -> 1

# 7. Logistic Regression + 7-fold Cross Validation
#model = LogisticRegression(max_iter=1000)
#cv_scores = cross_val_score(model, X, y, cv=7)

#print("7-fold CV accuracy scores:", cv_scores)
#print("Mean CV accuracy:", np.mean(cv_scores))

# 8. 在完整训练集上拟合
#model.fit(X, y)

# 9. 预测测试集
#test_preds = model.predict(X_test)
#test_preds_labels = le.inverse_transform(test_preds)

# 10. 生成提交文件
#submission = filledtest[['id']].copy()
#submission['Personality'] = test_preds_labels
#submission.to_csv('submission.csv', index=False)

#print("✅ Submission file saved as submission.csv")



#import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
#import numpy as np

# 假设填补后的完整数据是 df_imputed
#df_final = df_imputed.copy()

# 1. 分为训练集和测试集
#filledtrain = df_final[df_final['Personality'].notnull()].copy()
#filledtest = df_final[df_final['Personality'].isnull()].copy()

# 2. 准备特征列（去掉 id, Personality, 以及 *_missing 列）
#drop_cols = ['id', 'Personality'] + [col for col in df_final.columns if col.endswith('_missing')]
#feature_cols = [col for col in df_final.columns if col not in drop_cols]

# 3. 拆分出数值列和分类列
#X_raw = filledtrain[feature_cols]
#X_test_raw = filledtest[feature_cols]

#num_features = [col for col in X_raw.columns if X_raw[col].dtype in ['int64', 'float64']]
#cat_features = [col for col in X_raw.columns if col not in num_features]

# 4. One-Hot 编码分类特征
#X = pd.get_dummies(X_raw, columns=cat_features)
#X_test = pd.get_dummies(X_test_raw, columns=cat_features)

# 5. 对齐 train/test 特征列
#X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 6. 目标变量编码
#le = LabelEncoder()
#y = le.fit_transform(filledtrain['Personality'])  # Introvert -> 0, Extrovert -> 1

# 7. 决策树模型 + 7-fold Cross Validation
#model = DecisionTreeClassifier(random_state=42)
#cv_scores = cross_val_score(model, X, y, cv=7)

#print("7-fold CV accuracy scores:", cv_scores)
#print("Mean CV accuracy:", np.mean(cv_scores))

# 8. 在完整训练集上拟合
#model.fit(X, y)

# 9. 预测测试集
#test_preds = model.predict(X_test)
#test_preds_labels = le.inverse_transform(test_preds)

# 10. 生成提交文件
#submission = filledtest[['id']].copy()
#submission['Personality'] = test_preds_labels
#submission.to_csv('submission.csv', index=False)

#print("✅ Submission file saved as submission.csv")



#import pandas as pd
#import numpy as np
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
#from lightgbm import LGBMClassifier

# 假设填补后的完整数据是 df_imputed
#df_final = df_imputed.copy()

# 1. 分为训练集和测试集
#filledtrain = df_final[df_final['Personality'].notnull()].copy()
#filledtest = df_final[df_final['Personality'].isnull()].copy()

# 2. 准备特征列（去掉 id, Personality, 以及 *_missing 列）
#drop_cols = ['id', 'Personality'] + [col for col in df_final.columns if col.endswith('_missing')]
#feature_cols = [col for col in df_final.columns if col not in drop_cols]

# 3. 拆分出数值列和分类列
#X_raw = filledtrain[feature_cols]
#X_test_raw = filledtest[feature_cols]

#num_features = [col for col in X_raw.columns if X_raw[col].dtype in ['int64', 'float64']]
#cat_features = [col for col in X_raw.columns if col not in num_features]

# 4. One-Hot 编码分类特征
#X = pd.get_dummies(X_raw, columns=cat_features)
#X_test = pd.get_dummies(X_test_raw, columns=cat_features)

# 5. 对齐 train/test 特征列
#X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 6. 目标变量编码
#le = LabelEncoder()
#y = le.fit_transform(filledtrain['Personality'])  # Introvert -> 0, Extrovert -> 1

# 7. LightGBM 模型 + 7-fold Cross Validation
#model = LGBMClassifier(random_state=42)
#cv_scores = cross_val_score(model, X, y, cv=7)

#print("7-fold CV accuracy scores:", cv_scores)
#print("Mean CV accuracy:", np.mean(cv_scores))

# 8. 在完整训练集上拟合
#model.fit(X, y)

# 9. 预测测试集
#test_preds = model.predict(X_test)
#test_preds_labels = le.inverse_transform(test_preds)

# 10. 生成提交文件
#submission = filledtest[['id']].copy()
#submission['Personality'] = test_preds_labels
#submission.to_csv('submission.csv', index=False)

#print("✅ Submission file saved as submission.csv")



#import pandas as pd
#import numpy as np
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
#from xgboost import XGBClassifier

# 假设填补后的完整数据是 df_imputed
#df_final = df_imputed.copy()

# 1. 分为训练集和测试集
#filledtrain = df_final[df_final['Personality'].notnull()].copy()
#filledtest = df_final[df_final['Personality'].isnull()].copy()

# 2. 准备特征列（去掉 id, Personality, 以及 *_missing 列）
#drop_cols = ['id', 'Personality'] + [col for col in df_final.columns if col.endswith('_missing')]
#feature_cols = [col for col in df_final.columns if col not in drop_cols]

# 3. 拆分数值和分类列
#X_raw = filledtrain[feature_cols]
#X_test_raw = filledtest[feature_cols]

#num_features = [col for col in X_raw.columns if X_raw[col].dtype in ['int64', 'float64']]
#cat_features = [col for col in X_raw.columns if col not in num_features]

# 4. One-Hot 编码分类特征
#X = pd.get_dummies(X_raw, columns=cat_features)
#X_test = pd.get_dummies(X_test_raw, columns=cat_features)

# 5. 对齐 train/test 特征列
#X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 6. 编码目标变量
#le = LabelEncoder()
#y = le.fit_transform(filledtrain['Personality'])  # Introvert -> 0, Extrovert -> 1

# 7. XGBoost 模型 + 7-fold 交叉验证
#model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#cv_scores = cross_val_score(model, X, y, cv=7)

#print("7-fold CV accuracy scores:", cv_scores)
#print("Mean CV accuracy:", np.mean(cv_scores))

# 8. 在完整训练集上拟合
#model.fit(X, y)

# 9. 预测测试集
#test_preds = model.predict(X_test)
#test_preds_labels = le.inverse_transform(test_preds)

# 10. 生成提交文件
#submission = filledtest[['id']].copy()
#submission['Personality'] = test_preds_labels
#submission.to_csv('submission.csv', index=False)

#print("✅ Submission file saved as submission.csv")



#import pandas as pd
#import numpy as np
#import optuna
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
#from lightgbm import LGBMClassifier

# 准备训练集和测试集（使用你之前处理好的 filledtrain 和 filledtest）
#df_final = df_imputed.copy()

#filledtrain = df_final[df_final['Personality'].notnull()].copy()
#filledtest = df_final[df_final['Personality'].isnull()].copy()

#drop_cols = ['id', 'Personality'] + [col for col in df_final.columns if col.endswith('_missing')]
#feature_cols = [col for col in df_final.columns if col not in drop_cols]

#X_raw = filledtrain[feature_cols]
#X_test_raw = filledtest[feature_cols]

#num_features = [col for col in X_raw.columns if X_raw[col].dtype in ['int64', 'float64']]
#cat_features = [col for col in X_raw.columns if col not in num_features]

#X = pd.get_dummies(X_raw, columns=cat_features)
#X_test = pd.get_dummies(X_test_raw, columns=cat_features)
#X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

#le = LabelEncoder()
#y = le.fit_transform(filledtrain['Personality'])

# 目标函数：Optuna 会最小化这个函数
#from sklearn.model_selection import StratifiedKFold

#cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

#def objective(trial):
#    params = {
#        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#        'max_depth': trial.suggest_int('max_depth', 3, 12),
#        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#        'n_estimators': trial.suggest_int('n_estimators', 25, 500),
#        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
#        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#        'random_state': 42,
#    }
#    model = LGBMClassifier(**params)
#    score = cross_val_score(model, X, y, cv=cv, scoring='f1_macro').mean()
#    return 1 - score  # 因为 log_loss 越小越好



# 启动调参
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=300)  # 可调节 n_trials 数量，越多越精细

#print("Best parameters:", study.best_params)
#print("Best CV accuracy:", 1.0 - study.best_value)

# 使用最优参数训练并预测
#best_model = LGBMClassifier(**study.best_params)
#best_model.fit(X, y)
#test_preds = best_model.predict(X_test)
#test_preds_labels = le.inverse_transform(test_preds)

# 保存提交文件
#submission = filledtest[['id']].copy()
#submission['Personality'] = test_preds_labels
#submission.to_csv('submission.csv', index=False)
#print("✅ submission_lgb_optuna.csv saved.")



import pandas as pd
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# 用你自己的填补后的数据
df_final = df_imputed.copy()

# 1. 拆分训练/测试集
filledtrain = df_final[df_final['Personality'].notnull()].copy()
filledtest = df_final[df_final['Personality'].isnull()].copy()

# 2. 特征列处理
drop_cols = ['id', 'Personality'] + [col for col in df_final.columns if col.endswith('_missing')]
feature_cols = [col for col in df_final.columns if col not in drop_cols]

X_raw = filledtrain[feature_cols]
X_test_raw = filledtest[feature_cols]

num_features = [col for col in X_raw.columns if X_raw[col].dtype in ['int64', 'float64']]
cat_features = [col for col in X_raw.columns if col not in num_features]

# 3. One-Hot 编码 + 特征对齐
X = pd.get_dummies(X_raw, columns=cat_features)
X_test = pd.get_dummies(X_test_raw, columns=cat_features)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 4. Label Encoding 目标
le = LabelEncoder()
y = le.fit_transform(filledtrain['Personality'])  # Introvert->0, Extrovert->1

# 5. 模型参数（根据你提供的）
scale_pos_weight = 1.0  # 你可以改成适合你数据的值
best_params_dict = {
    'XGBoost': {
        'max_depth': 10, 
        'learning_rate': 0.013683607181209666, 
        'n_estimators': 735,
        'subsample': 0.8526, 
        'colsample_bytree': 0.7839,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1
    },
    'CatBoost': {
        'iterations': 894, 
        'depth': 6, 
        'learning_rate': 0.01525,
        'class_weights': [scale_pos_weight, 1],
        'random_seed': 42, 'verbose': 0
    },
    'LightGBM_gbdt': {
        'boosting_type': 'gbdt', 
        'num_leaves': 48, 
        'learning_rate': 0.01403,
        'n_estimators': 696, 
        'subsample': 0.7586, 
        'colsample_bytree': 0.8226,
        'class_weight': {0: scale_pos_weight, 1: 1},
        'random_state': 42, 'verbosity': -1
    },
    'LightGBM_goss': {
        'boosting_type': 'goss', 
        'num_leaves': 56, 
        'learning_rate': 0.02046,
        'n_estimators': 750, 
        'subsample': 0.9276, 
        'colsample_bytree': 0.7538,
        'class_weight': {0: scale_pos_weight, 1: 1},
        'random_state': 42, 'verbosity': -1
    },
    'HistGB': {
        'max_iter': 300, 
        'max_depth': 8, 
        'learning_rate': 0.02019,
        'min_samples_leaf': 20, 
        'class_weight': 'balanced',
        'random_state': 42
    }
}

# 6. 初始化模型
xgb = XGBClassifier(**best_params_dict['XGBoost'])
cat = CatBoostClassifier(**best_params_dict['CatBoost'])
lgbm_gbdt = LGBMClassifier(**best_params_dict['LightGBM_gbdt'])
lgbm_goss = LGBMClassifier(**best_params_dict['LightGBM_goss'])
hgb = HistGradientBoostingClassifier(**best_params_dict['HistGB'])

base_models = [
    ('xgb', xgb), ('cat', cat),
    ('lgbm_gbdt', lgbm_gbdt),
    ('lgbm_goss', lgbm_goss),
    ('hgb', hgb)
]

# 7. Stacking - OOF + Test predictions
def get_oof_predictions(models, X, y, X_test, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((X.shape[0], len(models)))
    test_preds = np.zeros((X_test.shape[0], len(models)))

    for idx, (name, model) in enumerate(models):
        print(f"\nTraining model: {name}")
        test_fold_preds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)

            oof_preds[val_idx, idx] = model.predict_proba(X_val)[:, 1]
            test_fold_preds.append(model.predict_proba(X_test)[:, 1])

        test_preds[:, idx] = np.mean(test_fold_preds, axis=0)

    return oof_preds, test_preds

# 8. 获取OOF与测试集预测
oof_preds, test_preds = get_oof_predictions(base_models, X, y, X_test)

# 9. Meta模型（Logistic回归）
meta_model = LogisticRegression(C=3.1566, penalty='l1', solver='liblinear', max_iter=2000)
meta_model.fit(oof_preds, y)

final_preds = meta_model.predict(test_preds)
final_labels = le.inverse_transform(final_preds)

# 10. 生成提交文件
submission = filledtest[['id']].copy()
submission['Personality'] = final_labels
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as submission.csv")





