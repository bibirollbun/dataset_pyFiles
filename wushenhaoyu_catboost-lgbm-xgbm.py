import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
from scipy.optimize import minimize
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
original_data = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')
train_ex_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train_data.drop(columns=['id'],inplace=True)
test_data.drop(columns=['id'],inplace=True)
train_ex_data.drop(columns=['id'],inplace=True)


# Drop null values from original_data
original_data = original_data.dropna()

# Print the count of null values in original_data
print(original_data.isnull().sum())

# Combine original_data with train_data
#train_data = pd.concat([train_data, original_data], axis=0).reset_index(drop=True)
train_data = pd.concat([train_data, original_data, train_ex_data], axis=0).reset_index(drop=True)


# Define imputation strategies
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

# Fill categorical missing values with mode (most frequent value)
for col in categorical_features:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    test_data[col].fillna(test_data[col].mode()[0], inplace=True)

# Fill numerical missing values with median
for col in numerical_features:
    train_data[col].fillna(train_data[col].median(), inplace=True)
    test_data[col].fillna(test_data[col].median(), inplace=True)


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def perform_feature_engineering(df):
    # Brand Material Interaction - Certain materials may be common for specific brands
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']

    # Brand & Size Interaction - Some brands may produce only specific sizes
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']

    # Has Laptop Compartment - Convert Yes/No to 1/0 for easier analysis
    df['Has_Laptop_Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})

    # Is Waterproof - Convert Yes/No to 1/0 for easier analysis
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})

    # Compartments Binning - Group compartments into categories
    df['Compartments_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])

    # Weight Capacity Ratio - Normalize weight capacity using the max value
    df['Weight_Capacity_Ratio'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()

    # Interaction Feature: Weight vs. Compartments - Some bags may hold more with less compartments
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)  # Avoid division by zero

    # Style and Size Interaction - Certain styles may correlate with sizes
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    

    return df

# Apply the function to the training data
train_data = perform_feature_engineering(train_data)

# Apply the function to the test data
test_data = perform_feature_engineering(test_data)


y = train_data['Price']


# Selecting specific columns for encoding
columns_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color','Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment','Is_Waterproof', 'Compartments_Category', 'Style_Size']
train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]

# Dropping selected columns for scaling
train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training data
minmax_scaler.fit(train_data_to_scale.drop(['Price'], axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['Price'], axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['Price'], axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)


# Concatenate train datasets
train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)

# Concatenate test datasets
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)


random_forest_params = {
    'n_estimators': 360,  # 树的数量
    'max_depth': 7,      # 每棵树的最大深度
    'min_samples_split': 2,  # 分裂内部节点所需的最小样本数
    'min_samples_leaf': 1,   # 在叶子节点处需要的最小样本数
    'max_features': 'sqrt',  # 在寻找最佳分割时考虑的特征数量
    'random_state': 42,      # 随机种子
    'n_jobs': -1             # 使用所有可用的CPU核心
}

xgbm_params = {'tree_method': 'hist',
                         'n_estimators': 3800,
                         'objective': 'reg:squarederror',
                         'random_state': 42,
                         'enable_categorical': True,
                         'verbosity': 0,
                         'early_stopping_rounds': 50,
                         'eval_metric': 'rmse',
                         'booster': 'gbtree',
                         'max_depth': 3,
                         'min_child_weight': 13,
                         'subsample': 0.6791128709665586,
                         'reg_alpha': 0.0912837368699186,
                         'reg_lambda': 0.8697948615643227,
                         'colsample_bytree': 0.9592753750165163,
                         'n_jobs': -1,
                         'learning_rate': 0.01,
                         'max_bin': 8000,
                         "device": "cuda",
                       }
lgbm_params = {'random_state': 42,
                'early_stopping_round': 50,
                'verbose': -1,
                'boosting_type': 'gbdt',
                'n_estimators': 2000,
                'eval_metric': 'rmse',
                'objective': 'regression_l2',
                'max_depth': 4,
                'num_leaves': 7,
                'min_child_samples': 35,
                'min_child_weight': 11,
                'colsample_bytree': 0.4759506289207658, 
                'reg_alpha': 0.28461417683987383, 
                'reg_lambda': 0.6555944495127437,
                'max_bin': 255,
                'learning_rate': 0.01,
                'device': 'gpu' 
              }

catboost_params =  {'verbose': 0,
               'random_state': 42,
               'early_stopping_rounds': 50,
               'eval_metric': "RMSE",
               'n_estimators' : 3000,
               'objective': 'RMSE', 
               'depth': 3,
               'min_data_in_leaf': 20,
               'l2_leaf_reg': 0.3349374242775052,
               'bagging_temperature': 0.8315027960954179, 
               'random_strength': 0.309798135191685,
               'learning_rate': 0.01,
               'max_bin': 8000,
               'bootstrap_type': 'Poisson',
               "task_type": "GPU",
            }


models = {
        "LightGBM": (LGBMRegressor, lgbm_params),
        "CatBoost": (CatBoostRegressor, catboost_params),
    "XGBoost": (XGBRegressor, xgbm_params),
    #"RandomForest": (RandomForestRegressor, random_forest_params),
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {model_name: {"rmse_scores": [], "mae_scores": [], "test_preds": np.zeros(len(test_data_combined))} for model_name in models.keys()}

# 手动实现 5 折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data_combined)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = train_data_combined.iloc[train_idx], train_data_combined.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for model_name, (model_class, params) in models.items():
        print(f"Training {model_name} on Fold {fold + 1}...")

        # 初始化模型
        model = model_class(**params)

        # 特殊处理 RandomForest（不支持 early_stopping_rounds 和 eval_set）
        if model_name == "RandomForest":
            model.fit(X_train, y_train)
        elif model_name == "LightGBM":
            model = model_class(**params)
        

        
            # 训练模型
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse'
            )
        else:
            # 其他模型支持 early_stopping_rounds 和 eval_set
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0  # 禁用日志输出
            )

        # 验证集预测
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)

        # 存储验证集分数
        results[model_name]["rmse_scores"].append(rmse)
        results[model_name]["mae_scores"].append(mae)

        print(f"{model_name} Fold {fold + 1} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 测试集预测并取平均
        results[model_name]["test_preds"] += model.predict(test_data_combined) / kf.get_n_splits()

# 对所有模型的测试集预测结果再取平均
final_test_preds = np.zeros(len(test_data_combined))
for model_name in models.keys():
    final_test_preds += results[model_name]["test_preds"] / len(models)

# 输出每种模型的平均 RMSE 和 MAE
for model_name, metrics in results.items():
    avg_rmse = np.mean(metrics["rmse_scores"])
    avg_mae = np.mean(metrics["mae_scores"])
    print(f"{model_name} Mean RMSE: {avg_rmse:.4f}, Mean MAE: {avg_mae:.4f}")

# 输出最终的测试集预测结果
print("Final test predictions (averaged across models and folds):")
print(final_test_preds)


submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
submission['Price'] = final_test_preds
submission.to_csv('submission.csv', index=False)




