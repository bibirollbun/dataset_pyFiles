import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import gc
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


df_sub=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train = pd.concat([df_train, df_original], axis=0, ignore_index=True)
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


df_train.info()


categorical_columns = df_train.select_dtypes(include=['object']).columns
unique_values = {col: df_train[col].nunique() for col in categorical_columns}
for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count} unique values")
    
gc.collect()


categorical_columns = df_test.select_dtypes(include=['object']).columns
unique_values = {col: df_test[col].nunique() for col in categorical_columns}
for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count} unique values")
    
gc.collect()


missing_train = df_train.isna().mean() * 100
missing_test = df_test.isna().mean() * 100

print("Columns in df_train with more than 10% missing values:")
print(missing_train[missing_train > 0])

print("\nColumns in df_test with more than 10% missing values:")
print(missing_test[missing_test > 0])


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import category_encoders as ce

missing_threshold = 0.95

high_missing_columns = df_train.columns[df_train.isnull().mean() > missing_threshold]

df_train = df_train.drop(columns=high_missing_columns)
df_test = df_test.drop(columns=high_missing_columns)

for column in df_train.columns:
    if df_train[column].isnull().any():      
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]
            df_train[column].fillna(mode_value, inplace=True)
            df_test[column].fillna(mode_value, inplace=True)     
        else:
            median_value = df_train[column].median()
            df_train[column].fillna(median_value, inplace=True)
            df_test[column].fillna(median_value, inplace=True)


df_train.columns



cat_cols_train = df_train.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'Fertilizer Name']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df_train[cat_cols_train] = ordinal_encoder.fit_transform(df_train[cat_cols_train].astype(str))
df_test[cat_cols_train] = ordinal_encoder.transform(df_test[cat_cols_train].astype(str))

cat_features = ['Soil Type','Crop Type']
df_train['Soil Type'] = df_train['Soil Type'].astype('category').cat.codes
df_train['Crop Type'] = df_train['Crop Type'].astype('category').cat.codes

df_test['Soil Type'] = df_test['Soil Type'].astype('category').cat.codes
df_test['Crop Type'] = df_test['Crop Type'].astype('category').cat.codes




df_train.head()


df_test.head()


le = LabelEncoder()
df_train['Fertilizer Name'] = le.fit_transform(df_train['Fertilizer Name'])

y = df_train['Fertilizer Name'] 
X = df_train.drop(['Fertilizer Name'],axis=1)


# KFold setup
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros((len(df_train), len(np.unique(y))))
pred = np.zeros((len(df_test), len(np.unique(y))))
logloss = []

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = df_test.copy()

    #weights = compute_sample_weight(class_weight='balanced', y=y_train)
    # 将数据转换为 XGBoost 的 DMatrix 格式
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    dtest = xgb.DMatrix(x_test)

    # XGBoost 参数
    params = {
        'objective': 'multi:softprob',  # 多分类概率输出
        'num_class': len(np.unique(y)),  # 类别数
        'max_depth': 16,
        'learning_rate': 0.03,
        'min_child_weight' : 2,
        'alpha': 0.8, 
        'reg_lambda': 4.0, 
        'colsample_bytree': 0.3,
        'subsample': 0.8,
        'max_bin': 128,
        'colsample_bytree': 0.3, 
        'colsample_bylevel': 1,  
        'colsample_bynode': 1,  
        'tree_method': 'hist',  
        'random_state': 42,
        'eval_metric': 'mlogloss',
        #'tree_method': 'hist',  
        #'device': 'cpu'                 
    }

    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=200
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(dvalid)
    pred += model.predict(dtest)

    log_loss_value = log_loss(y_valid, oof[valid_idx])
    print(f"Fold {i+1} log_loss: {log_loss_value:.4f}")
    logloss.append(log_loss_value)

# Average test predictions
pred /= FOLDS
log_loss_value = np.mean(logloss)

print(f"\nFinal CV log_loss: {log_loss_value:.4f}")


top_3_preds = np.argsort(pred, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y]

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
map3_score = mapk(actual, top_3_preds)
print(f"✅ MAP@3 Score: {map3_score:.5f}")


top_3_preds = np.argsort(pred, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")


