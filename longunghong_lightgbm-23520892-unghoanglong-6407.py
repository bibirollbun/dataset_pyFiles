import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import category_encoders as ce
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool, cv

from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm



!pip install imbalanced-learn==0.11.0



df79 = pd.read_parquet('/kaggle/input/dataset/data_7_9_silver.parquet')
df79.info()


# tạo thêm 2 thuộc tính mới lấy ngày và thứ của VSD
# Tạo ngày trong tháng từ VSD
df79['SO_DAY_OF_MONTH_VSD'] = df79['VSD'].dt.day

# Tạo thứ trong tuần (Monday=1, ..., Saturday=6, Sunday=7)
df79['SO_DAY_OF_WEEK_VSD'] = df79['VSD'].dt.weekday + 1


df79['SUPPLIER_DIV'] = df79['SUPPLIER_DIV'].astype('Int64')


columns_79 = df79.columns
cols_to_drop = ['Order date', 'SUBSIDIARY_CD', 'GLOBAL_NO', 'SO_TIME', 'VSD'] # loại bỏ các cột không cần tính IV
columns_79 = list(columns_79)
columns_79 = [col for col in columns_79 if col not in cols_to_drop]
print(columns_79)


df79['OTHER AREA SHIP DIV'] = df79['OTHER AREA SHIP DIV'].fillna('0')
df79['OTHER AREA SHIP DIV'] = df79['OTHER AREA SHIP DIV'].astype('int')


categorical_columns = ['CLASSIFY_CD', 'CUST_CD', 'BRAND_CD', 'INNER_CD', 'SUPPLIER_CD', 'Stock class', 'Consider count hodiday Saturday', 'PACKING RANK', 'PRODUCT_CD', 'PRODUCT ATTRIBUTION', 'SPECIAL DIV', 'LOGICAL PLANT', 'DIRECT SHIP FLG', 'DELI_DIV', 'Ship Mode', 'SHIP DECISION NO', 'SUPPLIER_DIV', 'SPECIAL_DIV', 'SO_DAY_OF_MONTH', 'SO_DAY_OF_WEEK', 'REASON_CD', 'SOUF_RCV_NO', 'QTUF_RCV_NO', 'SO_DAY_OF_MONTH_VSD', 'SO_DAY_OF_WEEK_VSD']
continuous_columns = ['Sales order line number', 'SO QTY', 'OTHER AREA SHIP DIV', 'ALLOCATION QTY', 'SUPPLIER INV AMOUNT', 'PURCHASE AMOUNT', 'PACK QTY', 'WEIGHT PER PIECE',  ]


features_gr2 = ['DELI_DIV', 'DIRECT SHIP FLG', 'SUPPLIER INV AMOUNT', 'WEIGHT PER PIECE', 'PRODUCT_CD', 'CUST_CD', 'SUPPLIER_CD', 'SO QTY', 'OTHER AREA SHIP DIV', 'CLASSIFY_CD', 'INNER_CD', 'Consider count hodiday Saturday', 'PACKING RANK', 'PRODUCT ATTRIBUTION', 'Ship Mode']


categorical_cols = ['CUST_CD', 'DIRECT SHIP FLG', 'DELI_DIV', 'PRODUCT_CD', 'SUPPLIER_CD', 'CLASSIFY_CD','INNER_CD', 'Consider count hodiday Saturday', 'PACKING RANK', 'PRODUCT ATTRIBUTION', 'Ship Mode']  # các cột dạng category


df79[features_gr2].isnull().sum()


X = df79[features_gr2].copy()
Y = df79['label'].copy()


# undersampling
rus = RandomUnderSampler(sampling_strategy={0: 10 * sum(Y == 1), 1: sum(Y == 1)}, random_state=42) # (1:5)
X_resampled, Y_resampled = rus.fit_resample(X, Y)



# Chuyển về dạng category để các model tree based có xử lý categorical feature mà không cần encoding có thể dùng
X_resampled[categorical_cols] = X_resampled[categorical_cols].astype('category')


N_TRIALS = 30  # Số lượng trials bạn muốn chạy

# Thanh tiến trình tqdm
progress_bar = tqdm(total=N_TRIALS, desc="Tuning LightGBM")

def objective_lightgmb(trial):
    param = {
        'objective': 'binary',
        'metric': 'None',  # Để tránh mâu thuẫn với f1_macro bên ngoài
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'cpu',  # CHẠY BẰNG CPU
        'max_depth': trial.suggest_int('max_depth', 10, 20),
        'num_leaves': trial.suggest_int('num_leaves', 64, 512),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.1),
        'max_bin': trial.suggest_int('max_bin', 255, 1024),  # thêm max_bin (dùng tốt với CPU)
    }

    model_lightgbm = lgb.LGBMClassifier(**param)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(
        model_lightgbm,
        X_resampled,
        Y_resampled,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1
    )

    progress_bar.update(1)  # Cập nhật tiến trình

    return np.mean(scores)

# Tạo Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective_lightgmb, n_trials=N_TRIALS)

# Kết thúc thanh tiến trình
progress_bar.close()


best_params = study.best_trial.params
print("Best parameters:", best_params)



# Cập nhật tham số
best_params.update({
    'device': 'cpu',
    'random_state': 42,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
})

# Train lại mô hình với toàn bộ dữ liệu đã resample
final_model_2_2 = lgb.LGBMClassifier(**best_params)
final_model_2_2.fit(X_resampled, Y_resampled)


df10 = pd.read_csv('/kaggle/input/dataset/PILOT_10.csv')
df10.info()


# Kiểu ngày
df10['Order date'] = pd.to_datetime(df10['Order date']) # Đã biết theo format deffault là YYYY-MN-DD => áp hàm chuyển kiểu luôn.
df10['VSD'] = pd.to_datetime(df10['VSD'])


df10.dtypes


# features có số giá trị rỗng > 0
blank_counts = df10.apply(lambda col: (col == ' ').sum() if col.dtype == 'object' else 0)
blank_counts = blank_counts[blank_counts > 0]  # chỉ giữ lại những cột có > 0 giá trị rỗng
print(blank_counts)


(df10 == '').sum()


# tạo thêm 2 thuộc tính mới lấy ngày và thứ của VSD
# Tạo ngày trong tháng từ VSD
df10['SO_DAY_OF_MONTH_VSD'] = df10['VSD'].dt.day

# Tạo thứ trong tuần (Monday=1, ..., Saturday=6, Sunday=7)
df10['SO_DAY_OF_WEEK_VSD'] = df10['VSD'].dt.weekday + 1


df10['SUPPLIER_DIV'] = df10['SUPPLIER_DIV'].astype('Int64')


features_gr2


df10[features_gr2].isnull().sum()


df10[features_gr2].dtypes


X_10 = df10[features_gr2]


X_10[categorical_cols] = X_10[categorical_cols].astype('category')


X_10


# Dự đoán trên X_10
y_pred_10_2_2 = final_model_2_2.predict(X_10)


# Tạo DataFrame kết quả
df_result_2_2 = pd.DataFrame({
    'ID': df10['ID'].values,
    'label': y_pred_10_2_2
})
df_result_2_2.to_csv("df_results_2_lan6.csv", index=False)


pip install -U optuna

