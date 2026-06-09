import pandas as pd
import numpy as np

# 导入训练集和测试集和提交样例
train_path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path = "/kaggle/input/playground-series-s5e10/test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("数据集导入成功！")
print(train_data.head())


#频率编码与数值分箱
def create_new_features(train_df,test_df,num_cols,cat_cols):
    train = train_df.copy()
    test = test_df.copy()

    #频率编码,为具有类别特征的数据增加频率这一输入
    for col in cat_cols:
        #freq均使用训练集数据
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq)

    #对数值数据进行分箱
    for col in num_cols:
        for q in [5,10,15]:
            train[f"{col}_bin{q}"],bins = pd.qcut(train[col],q=q,labels=False,retbins=True,duplicates="drop")
            test[f"{col}_bin{q}"] = pd.cut(test[col],bins=bins,labels=False,include_lowest=True)

    return train,test


all_cols = train_data.drop(columns=['accident_risk','id']).columns.tolist()

#初步分类
primary_nondigit_cols = [col for col in all_cols if train_data[col].dtype in ["object","category","bool"]]
primary_num_cols = [col for col in all_cols if col not in primary_nondigit_cols]

#根据类别数精确分类
LIMIT = 6
nondigit_cols = list(primary_nondigit_cols)
num_cols = []
for col in primary_num_cols:
    num = train_data[col].nunique()
    if num<LIMIT:
        nondigit_cols.append(col)
    else:
        num_cols.append(col)
        
train_enriched,test_enriched = create_new_features(
    train_data, test_data,
    num_cols, nondigit_cols
)

#删除噪音
remove_cols = ["time_of_day","num_lanes","road_type","road_signs_present"]
train_enriched = train_enriched.drop(columns=remove_cols)
test_enriched = test_enriched.drop(columns=remove_cols)

#去重
train_enriched = train_enriched.drop_duplicates()

print(f"已分拣 {len(nondigit_cols)} 个类别特征, {len(num_cols)} 个数值特征。")


y_train = train_enriched['accident_risk']
x_train = train_enriched.drop('accident_risk', axis=1)
x_test = test_enriched.copy()

#将类别数据转换为category
combined_df = pd.concat([x_train,x_test],axis=0)
bin_cols = [col for col in combined_df.columns if '_bin' in col]
cat_cols = nondigit_cols + bin_cols
cat_cols = [c for c in cat_cols if c not in remove_cols]
print(f"找到 {len(cat_cols)} 个“逻辑类别”列需要修正类型。")

for col in cat_cols:
    combined_df[col] = combined_df[col].astype('category')

x_train = combined_df.iloc[:len(train_enriched)]
x_test = combined_df.iloc[len(train_enriched):]

test_ids = x_test['id']
x_train = x_train.drop('id',axis=1)
x_test = x_test.drop('id',axis=1)

print(f"--- 模型数据准备完毕 ---")
print("x_train:")
print(x_train.head())


import xgboost as xgb

dtrain = xgb.DMatrix(
    x_train,
    label=y_train,
    enable_categorical=True
)

xgb_params = {
    'objective': 'reg:squarederror', 
    'eval_metric': 'rmse',         
    'max_depth': 11, 'learning_rate': 0.011,
    'subsample': 0.82, 'colsample_bytree': 0.81,
    'min_child_weight': 3, 'gamma': 0.011,
    'reg_alpha': 0.12, 'reg_lambda': 0.4,
    'max_delta_step': 1, 'colsample_bylevel': 0.86,
    'colsample_bynode': 0.88,
    'max_bin': 512, 'tree_method': 'hist',
    'device': 'cuda', 
    'random_state': 42,
}

print("\n开始运行 5-Fold K-Fold")
cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,           # 传入 DMatrix
    num_boost_round=2000,    # 给一条长跑道 (boosting轮数)
    nfold=5,
    stratified=False,
    metrics='rmse',          # 监控的指标
    verbose_eval=100,        # 每 100 轮打印一次日志
    early_stopping_rounds=50 # 50 轮不进步就停
)

best_round = cv_results['test-rmse-mean'].idxmin()
best_score = cv_results['test-rmse-mean'][best_round]
print(f"\n--- XGBoost 调优完毕！最佳轮数: {best_round}, 最佳K-Fold RMSE: {best_score:.7f} ---")


xgb_params['n_estimators'] = best_round

model = xgb.XGBRegressor(**xgb_params,enable_categorical=True)
model.fit(x_train,y_train)


predictions = model.predict(x_test)
submission = pd.DataFrame({
    'id':test_ids,
    'accident_risk':predictions
})

print("------------ 提交结果预览 ------------")
print(submission.head())

submission.to_csv("submission.csv",index=False)

