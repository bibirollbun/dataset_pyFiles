import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from itertools import combinations



#data load and shows data size and the fisrt colums
sample_sub = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# show the unique
for col in train.columns:
    unique_vals = train[col].unique()
    print(f"Unique values in '{col}':")
    print(unique_vals)
    print(f"Count: {len(unique_vals)}")
    print("-" * 40)


#data Re-Processing
# 1-column selection
target = train["target"]
train_id = train["id"]
test_id = test["id"]

x = train.drop(["id", "target"], axis=1)
x_test = test.drop("id", axis=1)


#  Missing count
x["missing_count"] = x.isnull().sum(axis=1)
x_test["missing_count"] = x_test.isnull().sum(axis=1)


# 2- Fill missing value
x = x.fillna("Missing")
x_test = x_test.fillna("Missing")


print("Columns in x:", x.columns.tolist())


# حذف العمود bin_3 من البيانات
x = x.drop(columns=['bin_3', 'bin_3_inv', 'bin3_bin_0_interaction', 'bin3_bin_1_interaction', 'bin3_bin_2_interaction', 'bin_3_ratio'], errors='ignore')
x_test = x_test.drop(columns=['bin_3', 'bin_3_inv', 'bin3_bin_0_interaction', 'bin3_bin_1_interaction', 'bin3_bin_2_interaction', 'bin_3_ratio'], errors='ignore')

print("Columns in x after removing bin_3:", x.columns.tolist())
print("Columns in x_test after removing bin_3:", x_test.columns.tolist())


import numpy as np
import pandas as pd

for col in ["day", "month"]:
    if col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce").fillna(0)

if "day" in x.columns and "month" in x.columns:
    x["day_sin"] = np.sin(2 * np.pi * x["day"] / 7)
    x["day_cos"] = np.cos(2 * np.pi * x["day"] / 7)
    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12)

    x_test["day_sin"] = np.sin(2 * np.pi * x_test["day"] / 7)
    x_test["day_cos"] = np.cos(2 * np.pi * x_test["day"] / 7)
    x_test["month_sin"] = np.sin(2 * np.pi * x_test["month"] / 12)
    x_test["month_cos"] = np.cos(2 * np.pi * x_test["month"] / 12)

    x = x.drop(["day", "month"], axis=1)
    x_test = x_test.drop(["day", "month"], axis=1)


#binary


from itertools import combinations
import pandas as pd

bin_cols = ['bin_0', 'bin_1', 'bin_2']

for col in bin_cols:
    x[col] = pd.to_numeric(x[col], errors='coerce').fillna(0).astype(int)
    x_test[col] = pd.to_numeric(x_test[col], errors='coerce').fillna(0).astype(int)

x['bin_sum'] = x[bin_cols].sum(axis=1)
x['bin_mean'] = x[bin_cols].mean(axis=1)

x_test['bin_sum'] = x_test[bin_cols].sum(axis=1)
x_test['bin_mean'] = x_test[bin_cols].mean(axis=1)

for col1, col2 in combinations(bin_cols, 2):
    new_col = f"{col1}_{col2}_interaction"
    x[new_col] = x[col1] * x[col2]
    x_test[new_col] = x_test[col1] * x_test[col2]

x['bin0_bin1'] = x['bin_0'].astype(str) + "_" + x['bin_1'].astype(str)
x_test['bin0_bin1'] = x_test['bin_0'].astype(str) + "_" + x_test['bin_1'].astype(str)

x['bin0_bin1_num'] = x['bin_0']*2 + x['bin_1']*1
x_test['bin0_bin1_num'] = x_test['bin_0']*2 + x_test['bin_1']*1

print("Columns in x after binary features engineering:", x.columns.tolist())
print("Columns in x_test after binary features engineering:", x_test.columns.tolist())


# طباعة القيم الفريدة ونسبة كل قيمة
print(x['bin_4'].value_counts(normalize=True, dropna=False))


nan_ratio = x['bin_4'].isna().mean()
print(f"النسبة المئوية للقيم المفقودة في bin_4: {nan_ratio:.4f}")


# تحويل Y/N إلى 1/0
x['bin_4'] = x['bin_4'].map({'Y': 1, 'N': 0})
x_test['bin_4'] = x_test['bin_4'].map({'Y': 1, 'N': 0})

# عمود عكسي
x['bin_4_inv'] = 1 - x['bin_4']
x_test['bin_4_inv'] = 1 - x_test['bin_4']

# التفاعلات مع الأعمدة الثنائية الأخرى (bin_0, bin_1, bin_2)
for col in ['bin_0', 'bin_1', 'bin_2']:
    new_col = f'bin4_{col}_interaction'
    x[new_col] = x['bin_4'] * x[col]
    x_test[new_col] = x_test['bin_4'] * x_test[col]

# نسبة bin_4 مقارنة بمجموع الأعمدة الثنائية (bin_0, bin_1, bin_2, bin_4)
bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_4']
x['bin_4_ratio'] = x['bin_4'] / (x[bin_cols].sum(axis=1) + 1e-5)
x_test['bin_4_ratio'] = x_test['bin_4'] / (x_test[bin_cols].sum(axis=1) + 1e-5)

# التحقق من الأعمدة الجديدة
print("Columns in x after bin_4 feature engineering:", x.columns.tolist())
print("Columns in x_test after bin_4 feature engineering:", x_test.columns.tolist())


#nominal


nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

for col in nom_cols:
    # حساب frequency encoding
    freq_encoding = x[col].value_counts(normalize=True)
    x[col + '_freq'] = x[col].map(freq_encoding)
    x_test[col + '_freq'] = x_test[col].map(freq_encoding)

# التأكد من النتائج
for col in nom_cols:
    print(f"Sample of {col} encoding:")
    print(x[[col, col + '_freq']].head())
    print()



nom_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

for col in nom_cols:
    # تحديد القيم الأعلى شيوعًا (top_values) لكل عمود، الباقي يصبح "Other"
    value_counts = x[col].value_counts()
    threshold = 0.01  # مثلا أي قيمة أقل من 1% تتحول لـ "Other"
    top_values = value_counts[value_counts/len(x) > threshold].index.tolist()
    
    x[f'{col}_clean'] = x[col].where(x[col].isin(top_values), 'Other')
    x_test[f'{col}_clean'] = x_test[col].where(x_test[col].isin(top_values), 'Other')

    # Frequency encoding
    freq_enc = x[f'{col}_clean'].value_counts(normalize=True)
    x[f'{col}_freq'] = x[f'{col}_clean'].map(freq_enc)
    x_test[f'{col}_freq'] = x_test[f'{col}_clean'].map(freq_enc)

# التحقق من النتائج
for col in nom_cols:
    print(x[[col, f'{col}_clean', f'{col}_freq']].head())


#ordinal


import pandas as pd

x = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
y = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
print(x[['ord_0','ord_1','ord_2','ord_3','ord_4']].head(10))
print(x[['ord_0','ord_1','ord_2','ord_3','ord_4']].dtypes)
print(y[['ord_0','ord_1','ord_2','ord_3','ord_4']].head(10))
print(y[['ord_0','ord_1','ord_2','ord_3','ord_4']].dtypes)



# أولاً نحدد mapping لكل عمود ordinal
ord_0_map = {1.0: 1, 2.0: 2, 3.0: 3}  # نترك NaN كما هو
ord_1_map = {'Novice': 1, 'Contributor': 2, 'Expert': 3, 'Master': 4, 'Grandmaster': 5}
ord_2_map = {'Freezing': 1, 'Cold': 2, 'Warm': 3, 'Hot': 4, 'Boiling Hot': 5, 'Lava Hot': 6}
ord_3_map = {'n':1,'a':2,'m':3,'c':4,'h':5,'o':6,'b':7,'e':8,'k':9,'i':10,'d':11,'f':12,
             'g':13,'j':14,'l':15}
ord_4_map = {'N':1,'P':2,'Y':3,'A':4,'R':5,'U':6,'M':7,'X':8,'C':9,'H':10,'Q':11,'T':12,
             'O':13,'B':14,'E':15,'K':16,'I':17,'D':18,'F':19,'W':20,'Z':21,'S':22,'G':23,
             'V':24,'J':25,'L':26}

# قائمة الأعمدة والـ mapping الخاص بكل واحد
ord_cols = ['ord_0','ord_1','ord_2','ord_3','ord_4']
ord_maps = [ord_0_map, ord_1_map, ord_2_map, ord_3_map, ord_4_map]

# تطبيق الـ mapping على البيانات مع الاحتفاظ بالـ NaN
for col, mapping in zip(ord_cols, ord_maps):
    x[col] = x[col].map(mapping)
    x_test[col] = x_test[col].map(mapping)

# طباعة مثال للتأكد
print("Sample ordinals (train):")
print(x[ord_cols].head(10))
print("\nSample ordinals (test):")
print(x_test[ord_cols].head(10))


# حساب تكرار كل قيمة في ord_5 بالنسبة للـ train
ord_5_freq = x['ord_5'].value_counts(normalize=True)

# إنشاء العمود الجديد مع الـ frequency encoding
x['ord_5_freq'] = x['ord_5'].map(ord_5_freq)
x_test['ord_5_freq'] = x_test['ord_5'].map(ord_5_freq)

# ملء القيم المفقودة في test بالقيم الأقل شيوعاً (0 أو قيمة صغيرة جداً)
x_test['ord_5_freq'] = x_test['ord_5_freq'].fillna(0)

# طباعة مثال للتأكد
print(x[['ord_5','ord_5_freq']].head())
print(x_test[['ord_5','ord_5_freq']].head())


# selection ordinal columns(featurs)
cat_features = x.columns.tolist()


#Data partitioning
x_train, x_valid, y_train, y_valid = train_test_split(
    x, target, test_size=0.2, random_state=42, stratify=target
)


# build model (catBoostClassifer)
x = x.astype(str)
x_test = x_test.astype(str)

x_train = x_train.astype(str)
x_valid = x_valid.astype(str)

model = CatBoostClassifier(
    task_type="GPU",
    devices='0',
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    cat_features=cat_features,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=100,
    verbose=200,
    random_seed=42
)

model.fit(x_train, y_train, eval_set=(x_valid, y_valid), early_stopping_rounds=100)


valid_pred = model.predict_proba(x_valid)[:, 1]
auc = roc_auc_score(y_valid, valid_pred)
print(f"Validation AUC: {auc:.4f}")


test_pred = model.predict_proba(x_test)[:, 1]

submission = pd.DataFrame({
    "id": test_id,
    "target": test_pred
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv file saved!")

