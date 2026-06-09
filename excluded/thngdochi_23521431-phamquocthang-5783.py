import pandas as pd
import numpy as np

data_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
data_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')

data_not_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
data_not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')

pilot = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv')


len(data_delay_4_6.columns), len(data_delay_7_9.columns), len(data_not_delay_4_6.columns), len(data_not_delay_7_9.columns), len(pilot.columns)


data_delay_4_6.columns.difference(data_delay_7_9.columns)


data_delay_7_9.columns.difference(pilot.columns)


data_delay_4_6.columns.difference(pilot.columns)


data_not_delay_4_6.columns.difference(data_delay_4_6.columns)


diff_features = ['ACTUAL_SHIP_DAYS', 'EXPENSIVE_FLG', 'HAZARD_FLG', 'HEAVY_FLG',
       'IO_UNFIT_FLG', 'PRODUCT_ASSORT', 'SPECIFY_PRODUCTION_DAYS',
       'SPECIFY_SHIP_DAYS', 'SUPPLIER_CATEGORY_CD', 'WEIGHT_UNIT']

data_delay_4_6 = data_delay_4_6.drop(columns=diff_features)
data_not_delay_4_6 = data_not_delay_4_6.drop(columns=diff_features)


len(data_delay_4_6.columns), len(data_delay_7_9.columns), len(data_not_delay_4_6.columns), len(data_not_delay_7_9.columns), len(pilot.columns)


data_4_6 = pd.concat([data_delay_4_6, data_not_delay_4_6], axis=0)
data_4_6 = data_4_6.sample(frac=1, random_state=42)
data_4_6.head(5)


data_7_9 = pd.concat([data_delay_7_9, data_not_delay_7_9], axis=0)
data_7_9 = data_7_9.sample(frac=1, random_state=42)
data_7_9.head(5)


full_data = pd.concat([data_4_6, data_7_9], axis=0)
full_data = full_data.sample(frac=1, random_state=42)
full_data


import matplotlib.pyplot as plt

missing_count = full_data.isnull().sum()
missing_count = missing_count[missing_count > 0]
missing_info = pd.DataFrame({'missing': missing_count,'percentage': (missing_count / len(full_data)) * 100})
missing = missing_info.sort_values(by='missing', ascending=True)

plt.figure(figsize=(10, 6))
ax = missing['missing'].plot(kind='barh', color='orange')
plt.grid(axis='x')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.title('Missing values in data')
plt.xlabel('Number of missing values')
plt.ylabel('Columns')

for i, (val, pct) in enumerate(zip(missing['missing'], missing_info['percentage'])):
    ax.text(val + len(full_data)*0.002, i, f'{pct:.2f}%', va='center', fontsize=10)


drop_cols = missing[missing['percentage'] > 50].index


full_data = full_data.drop(columns=drop_cols)


for col in missing_info[missing_info['percentage'] < 50].index:
    print(f"Column {col} has {missing_info['missing'][col]} missing values, which is {missing_info['percentage'][col]:.2f}% of the total data")
    print(f"Data type: {full_data[col].dtype}")
    print(f"Unique values: {full_data[col].unique()}\n")


full_data.dropna(subset=['SUPPLIER_DIV'], inplace=True)
full_data['Ship Mode'] = full_data['Ship Mode'].fillna(full_data['Ship Mode'].mode()[0])
full_data['SHIP DECISION NO'] = full_data['SHIP DECISION NO'].fillna(-1)


missing_count = pilot.isnull().sum()
missing_count = missing_count[missing_count > 0]
missing_info = pd.DataFrame({'missing': missing_count,'percentage': (missing_count / len(pilot)) * 100})
missing = missing_info.sort_values(by='missing', ascending=True)

plt.figure(figsize=(10, 6))
ax = missing['missing'].plot(kind='barh', color='orange')
plt.grid(axis='x')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.title('Missing values in data')
plt.xlabel('Number of missing values')
plt.ylabel('Columns')

for i, (val, pct) in enumerate(zip(missing['missing'], missing_info['percentage'])):
    ax.text(val + len(pilot)*0.002, i, f'{pct:.2f}%', va='center', fontsize=10)


drop_cols = missing[missing['percentage'] > 50].index


pilot = pilot.drop(columns=drop_cols)


for col in missing_info[missing_info['percentage'] < 50].index:
    print(f"Column {col} has {missing_info['missing'][col]} missing values, which is {missing_info['percentage'][col]:.2f}% of the total data")
    print(f"Data type: {pilot[col].dtype}")
    print(f"Unique values: {pilot[col].unique()}\n")


pilot['SUPPLIER_DIV'] = pilot['SUPPLIER_DIV'].fillna(-1)
pilot['Ship Mode'] = pilot['Ship Mode'].fillna(pilot['Ship Mode'].mode()[0])


invalid_values = ['', ' ', None]

for col in full_data.columns:
    if full_data[col].isin(invalid_values).any():
        print(f"Cột '{col}' chứa giá trị rỗng: {full_data[col][full_data[col].isin(invalid_values)].unique()}")
full_data = full_data.replace(['', ' ', None], -1).fillna(-1)


for col in pilot.columns:
    if pilot[col].isin(invalid_values).any():
        print(f"Cột '{col}' chứa giá trị rỗng: {pilot[col][pilot[col].isin(invalid_values)].unique()}")
pilot = pilot.replace(['', ' ', None], -1).fillna(-1)


full_data.columns.difference(pilot.columns)


full_data = full_data.drop(columns='SHIP DECISION NO')


full_data['VSD'] = pd.to_datetime(full_data['VSD'], format='mixed')
full_data['Order date'] = pd.to_datetime(full_data['Order date'], format='mixed')

full_data['VSD'] = full_data['VSD'].dt.date
full_data['Order date'] = full_data['Order date'].dt.date

full_data['VSD'] = pd.to_datetime(full_data['VSD'], errors='coerce', infer_datetime_format=True)
full_data['Order date'] = pd.to_datetime(full_data['Order date'], errors='coerce', infer_datetime_format=True)


pilot['VSD'] = pd.to_datetime(pilot['VSD'], format='mixed')
pilot['Order date'] = pd.to_datetime(pilot['Order date'], format='mixed')

pilot['VSD'] = pilot['VSD'].dt.date
pilot['Order date'] = pilot['Order date'].dt.date

pilot['VSD'] = pd.to_datetime(pilot['VSD'], errors='coerce', infer_datetime_format=True)
pilot['Order date'] = pd.to_datetime(pilot['Order date'], errors='coerce', infer_datetime_format=True)


date_time_cols = ['VSD', 'Order date']
for col in date_time_cols:
    full_data[f'day_of_week_{col}'] = full_data[col].dt.dayofweek
    full_data[f'month_{col}'] = full_data[col].dt.month
    full_data[f'week_{col}'] = full_data[col].dt.isocalendar().week

for col in date_time_cols:
    pilot[f'day_of_week_{col}'] = pilot[col].dt.dayofweek
    pilot[f'month_{col}'] = pilot[col].dt.month
    pilot[f'week_{col}'] = pilot[col].dt.isocalendar().week


full_data.columns.difference(pilot.columns)


cols_type = {
    'SUBSIDIARY_CD': 'str',
    'GLOBAL_NO': 'str',
    'CLASSIFY_CD': 'str',
    'CUST_CD': 'str',
    'BRAND_CD': 'str',
    'INNER_CD': 'str',
    'SUPPLIER_CD': 'str',
    'Sales order line number': 'int64',
    'Stock class': 'str',
    'Consider count hodiday Saturday': 'int64',
    'SO QTY': 'int64',
    'ALLOCATION QTY': 'int64',
    'SUPPLIER INV AMOUNT': 'float64',
    'PACKING RANK': 'str',
    'PRODUCT_CD': 'str',
    'PRODUCT ATTRIBUTION': 'str',
    'SPECIAL DIV': 'str',
    'LOGICAL PLANT': 'str',
    'PURCHASE AMOUNT': 'float64',
    'DIRECT SHIP FLG': 'str',
    'DELI_DIV': 'str',
    'Ship Mode': 'str',
    'PACK QTY': 'int64',
    'WEIGHT PER PIECE': 'float64',
    'SUPPLIER_DIV': 'str',
    'SPECIAL_DIV': 'str',
    'SO_DAY_OF_MONTH': 'int64', 
    'SO_DAY_OF_WEEK': 'int64', 
    'SO_TIME': 'int64'
}


for col, col_type in cols_type.items():
    if col in full_data.columns:
        full_data[col] = full_data[col].astype(col_type)

for col, col_type in cols_type.items():
    if col in pilot.columns:
        pilot[col] = pilot[col].astype(col_type)


import seaborn as sns

numerical_features = [
    'Sales order line number',
    'Consider count hodiday Saturday',
    'SO QTY',
    'ALLOCATION QTY',
    'SUPPLIER INV AMOUNT',
    'PURCHASE AMOUNT',
    'PACK QTY',
    'WEIGHT PER PIECE',
    'SO_DAY_OF_MONTH', 
    'SO_DAY_OF_WEEK', 
    'SO_TIME'
]

plt.figure(figsize=(10, 6))
numerical_data = full_data[numerical_features].corr()
sns.heatmap(numerical_data, annot=True, fmt='.4f', cmap='Blues', mask=np.triu(numerical_data.corr()))


corr_matrix = full_data[numerical_features].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [
    (row, col, upper_tri.loc[row, col])
    for row in upper_tri.index
    for col in upper_tri.columns
    if upper_tri.loc[row, col] > 0.9
]
label_corr = full_data[numerical_features + ['label']].corr()['label'].abs().drop('label')
to_drop = set()
for row, col, corr_val in high_corr_pairs:
    if label_corr[col] > label_corr[row]:
        if row != "SO QTY":
            to_drop.add(row)
    else:
        if col != "SO QTY":
            to_drop.add(col)

to_drop = list(to_drop)
full_data = full_data.drop(columns=to_drop)


plt.figure(figsize=(10, 6))
numerical_data = pilot[numerical_features].corr()
sns.heatmap(numerical_data, annot=True, fmt='.4f', cmap='Blues', mask=np.triu(numerical_data.corr()))


drop = ['ALLOCATION QTY', 'PURCHASE AMOUNT']
pilot = pilot.drop(columns=drop)


def frequency_encoding(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        freq = df[col].value_counts() / len(df)
        df[col] = df[col].map(freq)
    return df


categorical_features = [col for col, col_type in cols_type.items() if col_type == 'str']
print(categorical_features)


full_data_cat_fq = frequency_encoding(full_data, categorical_features)
pilot_cat_fq = frequency_encoding(pilot, categorical_features)


cat_corr = full_data_cat_fq[categorical_features].corr()
mask = np.triu(np.ones_like(cat_corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(cat_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap of categorical features")
plt.show()


corr_matrix = full_data_cat_fq[categorical_features].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

label_corr = full_data_cat_fq[categorical_features + ['label']].corr()['label'].abs().drop('label')

to_drop = set()
dropped_pairs = []

for col in upper_tri.columns:
    for row in upper_tri.index:
        if upper_tri.loc[row, col] > 0.9:
            dropped_pairs.append((row, col, upper_tri.loc[row, col]))
            if label_corr.get(col, 0) >= label_corr.get(row, 0):
                to_drop.add(row)
            else:
                to_drop.add(col)

# Drop 
full_data_cat_fq = full_data_cat_fq.drop(columns=list(to_drop))

# Update categorical_features
categorical_features = [col for col in categorical_features if col not in to_drop]


for row, col, corr_value in dropped_pairs:
    if row in to_drop or col in to_drop:
        print(f"Dropped one of: ({row}, {col}) | corr = {corr_value:.4f}")


cat_corr = pilot_cat_fq[categorical_features].corr()
mask = np.triu(np.ones_like(cat_corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(cat_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap of categorical features")
plt.show()


pilot_cat_fq = pilot_cat_fq.drop(columns=['DELI_DIV', 'SPECIAL DIV', 'Stock class'])


# Total Weight = 'WEIGHT PER PIECE' * 'SO QTY'
full_data_cat_fq['WEIGHT'] = full_data_cat_fq['WEIGHT PER PIECE'] * full_data_cat_fq['SO QTY']
pilot_cat_fq['WEIGHT'] = pilot_cat_fq['WEIGHT PER PIECE'] * pilot_cat_fq['SO QTY']

# Day range = 'VSD' - 'Order date'
full_data_cat_fq['day_range'] = (pd.to_datetime(full_data_cat_fq['VSD']) - pd.to_datetime(full_data_cat_fq['Order date'])).dt.days
pilot_cat_fq['day_range'] = (pd.to_datetime(pilot_cat_fq['VSD']) - pd.to_datetime(pilot_cat_fq['Order date'])).dt.days

# Drop 'WEIGHT PER PIECE', 'SO QTY', 'VSD', 'Order date'
drop_weight_features = ['WEIGHT PER PIECE', 'SO QTY', 'VSD', 'Order date']
full_data_cat_fq = full_data_cat_fq.drop(columns=drop_weight_features)
pilot_cat_fq = pilot_cat_fq.drop(columns=drop_weight_features)


pilot_cat_fq.shape, full_data_cat_fq.shape


from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.0001, random_state=42) 
preds = iso.fit_predict(full_data_cat_fq)

outlier_idx = full_data_cat_fq[preds == -1].index
print(f"Drop points: {len(outlier_idx)}")

full_data_cat_fq = full_data_cat_fq.drop(index=outlier_idx)


label = full_data_cat_fq['label']


plt.figure(figsize=(10, 6))

counts = label.value_counts()
percentages = (counts / counts.sum() * 100).round(2)
total = counts.sum()

ax = counts.plot(kind='bar', color=['r', 'y'])

for i, (count, pct) in enumerate(zip(counts, percentages)):
    ax.text(i, count + total * 0.01, f'{pct}%', ha='center', fontsize=12)

plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.title('Label Distribution')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


from sklearn.utils import resample

label_counts = full_data_cat_fq['label'].value_counts()
majority_class = label_counts.idxmax()
minority_class = label_counts.idxmin()

df_majority = full_data_cat_fq[full_data_cat_fq['label'] == majority_class]
df_minority = full_data_cat_fq[full_data_cat_fq['label'] == minority_class]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_undersampled = pd.concat([df_majority_downsampled, df_minority])
df_undersampled = df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)

print("After undersampling:")
print(df_undersampled['label'].value_counts())


from sklearn.utils import resample

label_counts = full_data_cat_fq['label'].value_counts()
majority_class = label_counts.idxmax()
minority_class = label_counts.idxmin()

df_majority = full_data_cat_fq[full_data_cat_fq['label'] == majority_class]
df_minority = full_data_cat_fq[full_data_cat_fq['label'] == minority_class]

df_minority_oversampled = resample(
    df_minority,
    replace=True,  
    n_samples=len(df_majority),  
    random_state=42
)

df_oversampled = pd.concat([df_majority, df_minority_oversampled])
df_oversampled = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)

print("After oversampling:")
print(df_oversampled['label'].value_counts())


pip install lightgbm xgboost catboost optuna


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# Undersampling
df_train_un, df_temp = train_test_split(df_undersampled, test_size=0.2, random_state=42, stratify=df_undersampled['label'])
df_val_un, df_test_un = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])

# Oversampling
df_train_ov, df_temp = train_test_split(df_oversampled, test_size=0.2, random_state=42, stratify=df_oversampled['label'])
df_val_ov, df_test_ov = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])

print("=====UNDERSAMPLING=====")
print("Train size:", df_train_un.shape)
print("Validation size:", df_val_un.shape)
print("Test size:", df_test_un.shape)

print("Label distribution in train:")
print(df_train_un['label'].value_counts())


print("=====OVERSAMPLING=====")
print("Train size:", df_train_ov.shape)
print("Validation size:", df_val_ov.shape)
print("Test size:", df_test_ov.shape)

print("Label distribution in train:")
print(df_train_ov['label'].value_counts())


def LightGBM(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='weighted'), params

# XGBoost
def XGBoost(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0),
        "enable_categorical": True,
        "use_label_encoder": False,
        "eval_metric": 'logloss'
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='weighted'), params

# CatBoost
def CatBoost(trial, X_train, y_train, X_val, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0),
        "verbose": 0,
        "random_seed": 42
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='weighted'), params

# SVM
def SVM(trial, X_train, y_train, X_val, y_val):
    params = {
        "C": trial.suggest_float("C", 0.1, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    model = SVC(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='weighted'), params


# Objective wrapper
def objective(model_func, X_train, y_train, X_val, y_val):
    def inner(trial):
        val_f1, params = model_func(trial, X_train, y_train, X_val, y_val)
        trial.set_user_attr("best_params", params)
        return val_f1
    return inner

# Cross validation
X_train = df_train_un.drop(columns=['label'])
y_train = df_train_un['label']
X_val = df_val_un.drop(columns=['label'])
y_val = df_val_un['label']
X_test = df_test_un.drop(columns=['label'])
y_test = df_test_un['label']

models = {
    "LightGBM": (LightGBM, lgb.LGBMClassifier),
    # "XGBoost": (XGBoost, xgb.XGBClassifier),
    # "CatBoost": (CatBoost, CatBoostClassifier),
    # "SVM": (SVM, SVC)
}

results = []

for name, (func, cls) in models.items():
    print(f"Tuning {name}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective(func, X_train, y_train, X_val, y_val), n_trials=5)
    
    best_params = study.best_trial.user_attrs['best_params']
    print(f"Best Params for {name}:", best_params)

    model = cls(**best_params)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    preds = model.predict(X_test)

    results.append({
        'Model': name,
        'Val F1': study.best_value,
        'Test F1': f1_score(y_test, preds, average='macro'),
        'Test Acc': accuracy_score(y_test, preds),
        'Best Params': best_params
    })

    # Submission
    pilot_preds = model.predict(pilot_cat_fq.drop(columns='ID'))
    submission = pd.DataFrame({
        'ID': pilot_cat_fq['ID'].values,
        'label': pilot_preds
    })
    submission.to_csv(f"submission.csv", index=False)

# Kết quả tổng hợp
results_df = pd.DataFrame(results)
print(results_df)


# # Cross validation
# X_train_ov = df_train_ov.drop(columns=['label'])
# y_train_ov = df_train_ov['label']
# X_val_ov = df_val_ov.drop(columns=['label'])
# y_val_ov = df_val_ov['label']
# X_test_ov = df_test_ov.drop(columns=['label'])
# y_test_ov = df_test_ov['label']
# models = {
#     "LightGBM": (LightGBM, lgb.LGBMClassifier),
#     # "XGBoost": (XGBoost, xgb.XGBClassifier),
#     # "CatBoost": (CatBoost, CatBoostClassifier),
#     # "SVM": (SVM, SVC)
# }
# results_ov = []
# for name, (func, cls) in models.items():
#     print(f"Tuning {name}...")
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective(func, X_train_ov, y_train_ov, X_val_ov, y_val_ov), n_trials=5)
    
#     best_params = study.best_trial.user_attrs['best_params']
#     print(f"Best Params for {name}:", best_params)
#     model = cls(**best_params)
#     model.fit(pd.concat([X_train_ov, X_val_ov]), pd.concat([y_train_ov, y_val_ov]))
#     preds = model.predict(X_test_ov)
#     results_ov.append({
#         'Model': name,
#         'Val F1': study.best_value,
#         'Test F1': f1_score(y_test_ov, preds, average='macro'),
#         'Test Acc': accuracy_score(y_test_ov, preds),
#         'Best Params': best_params
#     })
#     # Submission
#     pilot_preds = model.predict(pilot_cat_fq.drop(columns='ID'))
#     submission = pd.DataFrame({
#         'ID': pilot_cat_fq['ID'].values,
#         'label': pilot_preds
#     })
#     submission.to_csv(f"submission_{name}_ov.csv", index=False)
# # Kết quả tổng hợp
# results_ov_df = pd.DataFrame(results_ov)
# print(results_ov_df)


# results_df = pd.DataFrame(results)
# best_model_row = results_df.loc[results_df['Test F1'].idxmax()]
# best_model_name = best_model_row['Model']
# best_model_params = best_model_row['Best Params']

# # Best model
# if best_model_name == 'LightGBM':
#     best_model = lgb.LGBMClassifier(**best_model_params)
# elif best_model_name == 'XGBoost':
#     best_model = xgb.XGBClassifier(**best_model_params)
# elif best_model_name == 'CatBoost':
#     best_model = CatBoostClassifier(**best_model_params, verbose=0)


# best_model.fit(X_train, y_train)
# pilot_preds = best_model.predict(df_pilot_pca.drop(columns='ID'))
# submission = pd.DataFrame({
#     'ID': df_pilot_pca['ID'].values,
#     'label': pilot_preds
# })
# submission.to_csv("submission1.csv", index=False)

