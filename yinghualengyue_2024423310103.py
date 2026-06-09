import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

def compute_map5(y_actual, y_probs):
    top5_indices = np.argsort(-y_probs, axis=1)[:, :5]
    ap_list = []
    for i in range(len(y_actual)):
        true_label = y_actual[i]
        pred_group = top5_indices[i]
        score_val = 0.0
        hit_count = 0.0
        for pos in range(min(5, len(pred_group))):
            if pred_group[pos] == true_label:
                hit_count += 1
                score_val += hit_count / (pos + 1)
        ap_list.append(score_val / hit_count if hit_count > 0 else 0.0)
    return np.mean(ap_list)

def build_features(input_df):
    df_copy = input_df.copy()
    rename_dict = {
        'Temparature': 'temperature',
        'Phosphorous': 'P',
        'Nitrogen': 'N',
        'Potassium': 'K',
        'Moisture': 'moisture'
    }
    df_copy.rename(columns=rename_dict, inplace=True)
    
    df_copy['N_P_ratio'] = df_copy['N'] / (df_copy['P'] + 1e-6)
    df_copy['N_K_ratio'] = df_copy['N'] / (df_copy['K'] + 1e-6)
    df_copy['P_K_ratio'] = df_copy['P'] / (df_copy['K'] + 1e-6)
    df_copy['nutrient_total'] = df_copy['N'] + df_copy['P'] + df_copy['K']
    df_copy['nutrient_balance'] = (df_copy['N'] + df_copy['P'] + df_copy['K']) / 3
    
    df_copy['temp_humidity_inter'] = df_copy['temperature'] * df_copy['Humidity']
    df_copy['temp_moisture_inter'] = df_copy['temperature'] * df_copy['moisture']
    df_copy['humidity_moisture_inter'] = df_copy['Humidity'] * df_copy['moisture']
    df_copy['env_total_inter'] = df_copy['temperature'] * df_copy['Humidity'] * df_copy['moisture']
    
    return df_copy

TRAIN_CSV = "/kaggle/input/playground-series-s5e6/train.csv"
TEST_CSV = "/kaggle/input/playground-series-s5e6/test.csv"

train_data = pd.read_csv(TRAIN_CSV)
test_data = pd.read_csv(TEST_CSV)

X_features = train_data.drop(['id', 'Fertilizer Name'], axis=1)
y_labels = train_data['Fertilizer Name']
test_ids_list = test_data['id']
X_test_features = test_data.drop('id', axis=1)

label_encoder = LabelEncoder()
y_encoded_labels = label_encoder.fit_transform(y_labels)
class_list = label_encoder.classes_
class_count = len(class_list)

print(f"\n肥料类别数量: {class_count}")
print("肥料类别映射:")
for idx, class_name in enumerate(class_list):
    print(f"{idx}: {class_name}")

print("\n执行特征工程...")
X_engineered = build_features(X_features)
X_test_engineered = build_features(X_test_features)

cat_feature_cols = ['Soil Type', 'Crop Type']
num_feature_cols = [col for col in X_engineered.columns if col not in cat_feature_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num_transform', 'passthrough', num_feature_cols),
        ('cat_transform', OneHotEncoder(handle_unknown='ignore'), cat_feature_cols)
    ])

print("应用预处理...")
X_processed = preprocessor.fit_transform(X_engineered)
X_test_processed = preprocessor.transform(X_test_engineered)

print("预处理后训练集形状:", X_processed.shape)
print("预处理后测试集形状:", X_test_processed.shape)

model_hyperparams = {
    'objective': 'multi:softprob',
    'num_class': class_count,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'tree_method': 'hist',
    'n_estimators': 2000
}

n_folds = 5
skf_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_probs = np.zeros((len(X_processed), class_count))
test_probs = np.zeros((len(X_test_processed), class_count))

print(f"\n开始 {n_folds} 折交叉验证训练...")
for fold_idx, (train_split, valid_split) in enumerate(skf_cv.split(X_processed, y_encoded_labels)):
    print(f"\n=== Fold {fold_idx + 1}/{n_folds} ===")
    
    X_train_fold = X_processed[train_split]
    X_valid_fold = X_processed[valid_split]
    y_train_fold = y_encoded_labels[train_split]
    y_valid_fold = y_encoded_labels[valid_split]
    
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
    dtest = xgb.DMatrix(X_test_processed)
    
    xgb_model = xgb.train(
        model_hyperparams,
        dtrain,
        num_boost_round=model_hyperparams['n_estimators'],
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    oof_probs[valid_split] = xgb_model.predict(dvalid)
    test_probs += xgb_model.predict(dtest) / n_folds
    
    fold_map5 = compute_map5(y_valid_fold, oof_probs[valid_split])
    print(f"Fold {fold_idx + 1} MAP@5: {fold_map5:.5f}")

overall_map5 = compute_map5(y_encoded_labels, oof_probs)
print(f"\n整体交叉验证 MAP@5: {overall_map5:.5f}")

top5_pred_indices = np.argsort(-test_probs, axis=1)[:, :5]

flat_top5 = top5_pred_indices.ravel()
flat_top5_decoded = label_encoder.inverse_transform(flat_top5)
top5_decoded = flat_top5_decoded.reshape(top5_pred_indices.shape)

# 修改列名为 'Fertilizer Name'
submission_df = pd.DataFrame({
    'id': test_ids_list,
    'Fertilizer Name': [' '.join(row) for row in top5_decoded.astype(str)]
})

submission_df.to_csv('submission.csv', index=False)
print("\n提交文件已保存: submission.csv")

