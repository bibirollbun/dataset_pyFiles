import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from category_encoders.hashing import HashingEncoder
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb


pip install -U scikit-learn imbalanced-learn xgboost lightgbm category_encoders



# Đọc file cụ thể
delay_4_6 = pd.read_csv(f"/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
not_delay_4_6 = pd.read_csv(f"/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv", dtype={10: str, 42: str})
delay_7_9 = pd.read_csv(f"/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv")
not_delay_7_9 = pd.read_csv(f"/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv", dtype={34: str})


pilot = pd.read_csv(f"/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv")
sample_solution = pd.read_csv(f"/kaggle/input/ds-108-p-21-assigment-06/sample_Solution.csv")


(delay_4_6.dtypes == not_delay_4_6.dtypes).all()  # True nếu kiểu dữ liệu khớp


# Kiểm tra xem ko khớp ở đâu
delay_4_6.dtypes
not_delay_4_6.dtypes


(delay_7_9.dtypes == not_delay_7_9.dtypes).all()  # True nếu kiểu dữ liệu khớp


not_delay_4_6["Consider count hodiday Saturday"] = pd.to_numeric(not_delay_4_6["Consider count hodiday Saturday"], errors="coerce").fillna(0).astype("int64")


(delay_4_6.dtypes == not_delay_4_6.dtypes).all()  # True nếu kiểu dữ liệu khớp


#Bỏ các dữ liệu trùng lặp
delay_4_6 = delay_4_6.drop_duplicates()
not_delay_4_6 = not_delay_4_6.drop_duplicates()
delay_7_9 = delay_7_9.drop_duplicates()
not_delay_7_9 = not_delay_7_9.drop_duplicates()


print("Cột có ở delay_4_6 nhưng không có ở delay_7_9:", set(delay_4_6.columns) - set(delay_7_9.columns))
print("Cột có ở delay_7_9 nhưng không có ở delay_4_6:", set(delay_7_9.columns) - set(delay_4_6.columns))


cols_to_drop = [
    'SPECIFY_PRODUCTION_DAYS', 'EXPENSIVE_FLG', 'IO_UNFIT_FLG',
    'ACTUAL_SHIP_DAYS', 'PRODUCT_ASSORT', 'SUPPLIER_CATEGORY_CD',
    'SPECIFY_SHIP_DAYS', 'HAZARD_FLG', 'WEIGHT_UNIT', 'HEAVY_FLG'
]

delay_4_6 = delay_4_6.drop(columns=cols_to_drop)
not_delay_4_6 = not_delay_4_6.drop(columns=cols_to_drop)


set(delay_4_6.columns) == set(delay_7_9.columns)


# Gộp df_train và df_test thành df_A
df_A = pd.concat([delay_4_6, not_delay_4_6], ignore_index=True)
# Gộp df_train_7_9 và df_test_7_9 thành df_B
df_B = pd.concat([delay_7_9, not_delay_7_9], ignore_index=True)


print(f"Số dữ liệu từ tháng 7 - 9: {df_B.shape} ")
print(f"Số dữ liệu từ tháng 4 - 6: {df_A.shape} ")


df_A = df_A.sample(frac=1, random_state=42)  # Lấy toàn bộ dữ liệu
df_B = df_B.sample(frac=0.75, random_state=42) #Lấy 75% dữ liệu
#Giảm dữ liệu để khi train không bị crash out


# Kết hợp A và B
df = pd.concat([df_A, df_B], ignore_index=True)


print("Cột có ở pilot nhưng không có ở df:", set(pilot.columns) - set(df.columns))
print("Cột có ở df nhưng không có ở pilot:", set(df.columns) - set(pilot.columns))


# Chỉnh dữ liệu lại các kiểu dữ liệu thời gian
df["VSD"] = pd.to_datetime(df["VSD"], format='ISO8601')
df["Order date"] = pd.to_datetime(df["Order date"], format='ISO8601')


df[['VSD', "Order date"]]


pilot[['VSD', "Order date"]]


# Chỉnh dữ liệu lại các kiểu dữ liệu thời gian
pilot["VSD"] = pd.to_datetime(pilot["VSD"], format='%m/%d/%Y')
pilot["Order date"] = pd.to_datetime(pilot["Order date"], format='%m/%d/%Y')


print(df.dtypes[["VSD", "Order date"]])


print(pilot.dtypes[["VSD", "Order date"]])


def plot_missing_values(df, threshold=0.2, show_top=None):
    """
    Hiển thị biểu đồ tỷ lệ missing value của các cột trong DataFrame.

    Parameters:
        df (pd.DataFrame): Dataset cần kiểm tra.
        threshold (float): Ngưỡng tỷ lệ missing để lọc (default = 0.2).
        show_top (int): Nếu muốn chỉ hiển thị top N cột missing cao nhất.

    Returns:
        missing_df (pd.DataFrame): Thống kê missing.
        high_missing_cols (List[str]): Các cột có missing > threshold.
    """
    cols = df.columns
    count = [df[col].isnull().sum() for col in cols]
    percent = [i / len(df) for i in count]

    missing = pd.DataFrame({'column': cols, 'proportion': percent})
    missing = missing.sort_values(by='proportion', ascending=False)

    if show_top is not None:
        missing = missing.head(show_top)

    plt.figure(figsize=(20, max(10, len(missing) * 0.3)))
    plt.title('Missing values in each column', fontsize=16)
    ax = sns.barplot(x='proportion', y='column', data=missing, palette='viridis')

    for p in ax.patches:
        value = p.get_width() * 100
        ax.text(p.get_width() + 0.005, p.get_y() + p.get_height() / 2,
                f"{value:.2f}%", va='center')

    mean = np.mean(missing['proportion'])
    std = np.std(missing['proportion'])
    plt.xlabel('Proportion')
    plt.ylabel('Columns')
    plt.plot([], [], ' ', label=f'Average missing: {mean:.2%} \u00B1 {std:.2%}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    high_missing_cols = missing[missing['proportion'] > threshold]['column'].tolist()
    return missing, high_missing_cols


missing_df, high_missing_cols = plot_missing_values(df, threshold=0.2)
print("Cột có missing > 20%:", high_missing_cols)


cols_to_check = ['QTUF_RCV_NO', 'SOUF_RCV_NO', 'REASON_CD', "OTHER AREA SHIP DIV"]

for col in cols_to_check:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))


# Đầu tiên chuyển tất cả sang string (để xử lý 1.0 và 1 giống nhau)
df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].astype(str).replace('1.0', '1')

# Nếu muốn giữ là số nguyên:
df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].replace({'1.0': 1, '1': 1})


# Đầu tiên chuyển tất cả sang string (để xử lý 1.0 và 1 giống nhau)
pilot['OTHER AREA SHIP DIV'] = pilot['OTHER AREA SHIP DIV'].astype(str).replace('1.0', '1')

# Nếu muốn giữ là số nguyên:
pilot['OTHER AREA SHIP DIV'] = pilot['OTHER AREA SHIP DIV'].replace({'1.0': 1, '1': 1})


print(df["OTHER AREA SHIP DIV"].value_counts(dropna=False))


df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].replace(['nan', 'NaN'], '0')


pilot['OTHER AREA SHIP DIV'] = pilot['OTHER AREA SHIP DIV'].replace(['nan', 'NaN'], '0')


print(df["OTHER AREA SHIP DIV"].value_counts(dropna=False))


df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].astype(str)
pilot['OTHER AREA SHIP DIV'] = pilot['OTHER AREA SHIP DIV'].astype(str)


# Impute cột categorical bằng mode (most frequent)
cat_imputer = SimpleImputer(strategy="most_frequent")
df[["Ship Mode", "SHIP DECISION NO"]] = cat_imputer.fit_transform(df[["Ship Mode", "SHIP DECISION NO"]])


# Impute cột categorical bằng mode (most frequent)
cat_imputer = SimpleImputer(strategy="most_frequent")
pilot[["Ship Mode", "SHIP DECISION NO"]] = cat_imputer.fit_transform(pilot[["Ship Mode", "SHIP DECISION NO"]])


df.drop(columns = ['QTUF_RCV_NO', 'SOUF_RCV_NO', 'REASON_CD'], inplace = True)
# drop luôn REASON_CD do tỉ lệ thiếu cao và có thể là biến được tạo ra khi đã giao hàng rồi


pilot.drop(columns = ['QTUF_RCV_NO', 'SOUF_RCV_NO', 'REASON_CD'], inplace = True)
# # drop luôn REASON_CD do tỉ lệ thiếu cao và có thể là biến được tạo ra khi đã giao hàng rồi


def winsorize_column(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    original_outliers = ((data[column] < lower) | (data[column] > upper)).sum()

    # Clip lại
    data[column] = data[column].clip(lower, upper)

    print(f"[{column}] Winsorized: {original_outliers} outliers clipped.")
    return data


num_cols = ["Sales order line number","SO QTY","ALLOCATION QTY","SUPPLIER INV AMOUNT","PURCHASE AMOUNT","PACK QTY",
            "WEIGHT PER PIECE"]


for col in num_cols:
    df = winsorize_column(df, col)
    pilot = winsorize_column(pilot,col)


df.describe()


# Loại các cột bị toàn NaN hoặc 1 giá trị
num_filtered_cols = [col for col in num_cols if df[col].nunique(dropna=True) > 1 and df[col].notnull().sum() > 0]
corr = df[num_filtered_cols].corr(numeric_only=True)  # Tính hệ số tương quan
plt.figure(figsize=(20, 16))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Ma trận tương quan giữa các biến số")
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

# Thử xem biến nào dự đoán label tốt hơn trong các biến có tương quan cao
X = df[['SO QTY', 'ALLOCATION QTY', 'SUPPLIER INV AMOUNT', 'PURCHASE AMOUNT']].copy()
y = df['label'].copy()

model = LinearRegression()
model.fit(X, y)
print(model.coef_)


df = df.drop(columns = ['SO QTY','PURCHASE AMOUNT'])
pilot = pilot.drop(columns = ['SO QTY','PURCHASE AMOUNT'])


cat_cols = ['SUBSIDIARY_CD', 'GLOBAL_NO', 'CLASSIFY_CD', 'CUST_CD',
       'BRAND_CD', 'INNER_CD', 'SUPPLIER_CD',
       'Stock class', 'Consider count hodiday Saturday', 'OTHER AREA SHIP DIV',
       'PACKING RANK', 'PRODUCT_CD', 'PRODUCT ATTRIBUTION', 'SPECIAL DIV',
       'LOGICAL PLANT', 'DIRECT SHIP FLG', 'DELI_DIV', 'Ship Mode',
        'SHIP DECISION NO', 'SUPPLIER_DIV', 'SPECIAL_DIV', 'SO_DAY_OF_MONTH',
       'SO_DAY_OF_WEEK', 'SO_TIME']


for col in cat_cols:
    print(f"Biến: {col} có {df[col].nunique()} giá trị khác nhau")


# Do em nhận thấy 'SO_TIME' có dạng hh:mm:ss, nên sẽ chuyển thành chuỗi và chỉ lấy 2 kí tự đầu trích xuất thành 'hour'
df['SO_TIME_str'] = df['SO_TIME'].astype(str).str.zfill(6)
df['hour'] = pd.to_datetime(df['SO_TIME_str'], format='%H%M%S').dt.hour


# Chuyển sang int trước để loại bỏ phần thập phân, sau đó về chuỗi
pilot['SO_TIME_str'] = pilot['SO_TIME'].astype(float).astype(int).astype(str)
# Bổ sung số 0 đầu nếu cần (đảm bảo độ dài 6 ký tự)
pilot['SO_TIME_str'] = pilot['SO_TIME_str'].str.zfill(6)

pilot['hour'] = pd.to_datetime(pilot['SO_TIME_str'], format='%H%M%S').dt.hour


def get_time_period(hour):
    if 5 <= hour < 12:
        return 'Sáng'
    elif 12 <= hour < 18:
        return 'Chiều'
    elif 18 <= hour < 23:
        return 'Tối'
    else:
        return 'Khuya'

df['time_period'] = df['hour'].apply(get_time_period)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.countplot(data=df, x='time_period', order=df['time_period'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Phân bố giá trị của biến time_period')
plt.ylabel('Số lượng')
plt.xlabel('time_period')
plt.tight_layout()
plt.show()


# Gom nhóm cuối tuần hay không
df['IS_WEEKEND'] = df['SO_DAY_OF_WEEK'].isin([5, 6]).astype(int)

# Gom tuần đầu/thứ 2/giữa/cuối tháng
df['MONTH_PHASE'] = pd.cut(df['SO_DAY_OF_MONTH'], bins=[0, 7, 14, 31],
                           labels=['early', 'mid', 'late'])

# Đã có biến IS_WEEKEND & MONTH_PHASE rồi nên chỉ cần trích xuất tháng/quý
df['Order_month'] = df['Order date'].dt.month             # Tháng trong năm (1-12)
df['Order_quarter'] = df['Order date'].dt.quarter         # Quý (1-4)

# Tương tự với 'VSD'
df['VSD_month'] = df['VSD'].dt.month
df['VSD_quarter'] = df['VSD'].dt.quarter

df['VSD_IS_WEEKEND'] = df['VSD'].dt.dayofweek.isin([5,6]).astype(int)
df['VSD_MONTH_PHASE'] = pd.cut(df['VSD'].dt.day, bins=[0, 7, 14, 31],
                           labels=['early', 'mid', 'late'])


# Gom nhóm cuối tuần hay không
pilot['IS_WEEKEND'] = pilot['SO_DAY_OF_WEEK'].isin([5, 6]).astype(int)

# Gom tuần đầu/thứ 2/giữa/cuối tháng
pilot['MONTH_PHASE'] = pd.cut(pilot['SO_DAY_OF_MONTH'], bins=[0, 7, 14, 31],
                           labels=['early', 'mid', 'late'])

# Đã có biến IS_WEEKEND & MONTH_PHASE rồi nên chỉ cần trích xuất tháng/quý
pilot['Order_month'] = pilot['Order date'].dt.month             # Tháng trong năm (1-12)
pilot['Order_quarter'] = pilot['Order date'].dt.quarter         # Quý (1-4)

#Tương tự với 'VSD'
pilot['VSD_month'] = pilot['VSD'].dt.month
pilot['VSD_quarter'] = pilot['VSD'].dt.quarter

pilot['VSD_IS_WEEKEND'] = pilot['VSD'].dt.dayofweek.isin([5,6]).astype(int)
pilot['VSD_MONTH_PHASE'] = pd.cut(pilot['VSD'].dt.day, bins=[0, 7, 14, 31],
                           labels=['early', 'mid', 'late'])


# Tạo biến ngày dự kiến sẽ giao hàng
df['Expected_delivery_days'] = (df['VSD'] - df['Order date']).dt.days
# Clipping
df = winsorize_column(df,'Expected_delivery_days')
df['Expected_delivery_days']


pilot['Expected_delivery_days'] = (pilot['VSD'] - pilot['Order date']).dt.days
#Clipping
pilot = winsorize_column(pilot,'Expected_delivery_days')
pilot['Expected_delivery_days']


# Trung bình ngày giao hàng dự kiến của từng đơn vị vận chuyển
df['delay_per_supplier'] = df.groupby('SUPPLIER_CD')['Expected_delivery_days'].transform('mean')
df = winsorize_column(df,'delay_per_supplier')
pilot['delay_per_supplier'] = pilot.groupby('SUPPLIER_CD')['Expected_delivery_days'].transform('mean')
pilot = winsorize_column(pilot,'delay_per_supplier')


# Kiểm tra biến 'SPECIAL_DIV' và  "SPECIAL DIV" có hoàn toàn giống nhau ko
(df['SPECIAL DIV'] == df['SPECIAL_DIV']).all()


df = df.drop('SPECIAL DIV', axis=1)
pilot = pilot.drop('SPECIAL DIV', axis=1)


df = df.drop(columns = ['GLOBAL_NO', 'INNER_CD', 'PRODUCT_CD', 'SHIP DECISION NO','SO_TIME', 'SO_TIME_str','hour','SO_DAY_OF_WEEK','SO_DAY_OF_MONTH','Order date', 'VSD'])
pilot = pilot.drop(columns = ['GLOBAL_NO', 'INNER_CD', 'PRODUCT_CD', 'SHIP DECISION NO','SO_TIME', 'SO_TIME_str','hour','SO_DAY_OF_WEEK','SO_DAY_OF_MONTH','Order date', 'VSD'])


df = pd.read_csv(f"/content/drive/MyDrive/Kì 4 - Năm 2/delay_prediction/df_train_gold_final.csv")
pilot = pd.read_csv(f"/content/drive/MyDrive/Kì 4 - Năm 2/delay_prediction/df_test_gold_final.csv")
sample_solution = pd.read_csv(f"/content/drive/MyDrive/Kì 4 - Năm 2/delay_prediction/sample_Solution.csv")


X = df.drop('label', axis=1)
y = df['label']

# Chia thành train (85%) và test (15%)
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)


X_test = pilot.copy().drop(columns = 'ID')
y_test = sample_solution.copy().drop(columns = 'ID')


for col in X_train.columns:
    types = set(X_train[col].apply(type))
    if len(types) > 1:
        print(f"Cột {col} có kiểu hỗn hợp: {types}")
print(f"Không có kiểu hỗn hợp.")


target_encode = ['CLASSIFY_CD', 'CUST_CD', 'BRAND_CD', 'SUPPLIER_CD']
one_hot_encode = ['SUBSIDIARY_CD', 'time_period', 'Stock class',
                    'Consider count hodiday Saturday', 'OTHER AREA SHIP DIV',
                    'PACKING RANK', 'PRODUCT ATTRIBUTION',
                    'LOGICAL PLANT', 'DIRECT SHIP FLG', 'DELI_DIV', 'Ship Mode',
                    'SUPPLIER_DIV', 'SPECIAL_DIV', 'IS_WEEKEND', 'MONTH_PHASE',
                    'Order_month', 'Order_quarter', 'VSD_month', 'VSD_quarter',
                    'VSD_IS_WEEKEND', 'VSD_MONTH_PHASE']


num_cols_new = ["Sales order line number","ALLOCATION QTY","SUPPLIER INV AMOUNT","PACK QTY",
            "WEIGHT PER PIECE", 'Expected_delivery_days',
       'delay_per_supplier']


threshold = 0.01 * len(X_train)  # Gom nhóm giá trị dưới 1%
for col in ['CLASSIFY_CD', 'CUST_CD', 'BRAND_CD', 'SUPPLIER_CD']:
    value_counts = X_train[col].value_counts()
    X_train[col] = X_train[col].apply(lambda x: x if value_counts.get(x, 0) >= threshold else 'Other')
    X_test[col] = X_test[col].apply(lambda x: x if value_counts.get(x, 0) >= threshold else 'Other')


from category_encoders import TargetEncoder

numeric_transformer = StandardScaler()
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
target_transformer = TargetEncoder(min_samples_leaf=5, smoothing=1.0)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols_new),
         ('target', target_transformer, target_encode),
        ('onehot', onehot_transformer, one_hot_encode)
    ])


# Fit preprocessor 
X_train_transformed = preprocessor.fit_transform(X_train,y_train)
X_dev_transformed = preprocessor.transform(X_dev)
X_test_transformed = preprocessor.transform(X_test)


# SMOTE cho tập dev để mô phỏng tỷ lệ của tập test
smote = SMOTE(sampling_strategy=1/2, random_state=42)
X_dev_smote, y_dev_smote = smote.fit_resample(X_dev_transformed, y_dev)


# Áp dung undersampling
undersampler = RandomUnderSampler(sampling_strategy=1/20, random_state=42)
X_train_final, y_train_final = undersampler.fit_resample(X_train_transformed, y_train)


xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=1,
    use_label_encoder=False
)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_xgb = GridSearchCV(
    xgb_model,
    param_grid=param_grid_xgb,
    scoring='f1_macro',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_xgb.fit(X_train_final, y_train_final)

print("Best XGBoost params:", grid_xgb.best_params_)
print("Best XGBoost F1_macro:", grid_xgb.best_score_)


y_pred_xgb = grid_xgb.predict(X_dev_smote)
print(classification_report(y_dev_smote, y_pred_xgb))


# y_pred_test_xgb = grid_xgb.predict(X_test_transformed)
# print(classification_report(y_test, y_pred_test_xgb))


onehot_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(one_hot_encode)

# 2. Từ TargetEncoder (giữ nguyên tên cột gốc)
target_feature_names = target_encode  # TargetEncoder không thay đổi tên cột

# 3. Từ StandardScaler (giữ nguyên tên cột số)
scaler_feature_names = num_cols_new

# Kết hợp tất cả tên đặc trưng
feature_names = np.concatenate([onehot_feature_names, target_feature_names, scaler_feature_names])


import pandas as pd
feature_importance = pd.DataFrame({
    'feature': feature_names,  # feature_names được định nghĩa từ preprocessor
    'importance': grid_xgb.best_estimator_.feature_importances_
}).sort_values(by='importance', ascending=False)
print(feature_importance)


df_submission = pd.DataFrame({
    'ID': sample_solution['ID'],
    "label": y_pred_test_xgb
})


df_submission.to_csv('/kaggle/working/submission.csv', index=False)

