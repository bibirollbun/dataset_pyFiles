import pandas as pd
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df=df.drop("id",axis=1)
test_df=test_df.drop("id",axis=1)
X = df.drop("y", axis=1)
y = df["y"]



df.info()


categorical_cols=[col for col in X.columns if X[col].dtype=="object" or X[col].nunique()<10]
numerical_cols=[col for col in X.columns if X[col].dtype=="int64" and col not in categorical_cols]
              


import seaborn as sns
import matplotlib.pyplot as plt

categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 10]

for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=X, x=col, palette='Set2')
    plt.title(f'Category counts for {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



test_df.drop(['default','poutcome'],axis=1,errors='ignore',inplace=True)
df.drop(['default','poutcome'],axis=1,errors='ignore',inplace=True)
X=df.drop("y",axis=1)
y=df["y"]



month_counts = df['month'].value_counts()
rare_months = month_counts[month_counts < 20000].index  

df['month'] = df['month'].apply(lambda x: 'Other' if x in rare_months else x)
test_df['month'] = test_df['month'].apply(lambda x: 'Other' if x in rare_months else x)
test_df['month'].value_counts()


job_counts = df['job'].value_counts()
rare_jobs = job_counts[job_counts < 25000].index

df['job'] = df['job'].apply(lambda x: 'Other' if x in rare_jobs else x)

test_df['job'] = test_df['job'].apply(lambda x: 'Other' if x in rare_jobs else x)
test_df['job'].value_counts()




df['education'] = df['education'].replace('unknown', 'missing')
test_df['education'] = test_df['education'].replace('unknown', 'missing')
df['education'].value_counts()
test_df['education'].value_counts()


df['contact'] = df['contact'].replace('unknown', 'missing')
test_df['contact'] = test_df['contact'].replace('unknown', 'missing')




categorical_cols=[col for col in X.columns if X[col].dtype=="object" or X[col].nunique()<10]
numerical_cols=[col for col in X.columns if X[col].dtype=="int64" and col not in categorical_cols]


X = df.drop("y", axis=1)
y = df["y"]
import seaborn as sns
import matplotlib.pyplot as plt

categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 10]

for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=X, x=col, palette='Set2')
    plt.title(f'Category counts for {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



df.isna().sum()


plt.figure(figsize=(10,6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between numerical features')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

for col in numerical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(14,4))  
    sns.histplot(df[col], bins=30, color='skyblue', ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')

    sns.boxplot(x=df[col], color='lightgreen', ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()



df.drop(['previous','pdays'],axis=1,inplace=True)
test_df.drop(['previous','pdays'],axis=1,inplace=True)
df


X=df.drop('y',axis=1)
y=df['y']
numerical_cols=[col for col in X.columns if X[col].dtype=="int64" and col not in categorical_cols]
for col in numerical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(14,4))  
    sns.histplot(df[col], bins=30, color='skyblue', ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')

    sns.boxplot(x=df[col], color='lightgreen', ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()



age_70_90 = df[(df['age'] >= 70) & (df['age'] <= 90)]
print(len(age_70_90))



import numpy as np

# balance sütunundaki negatif değerleri 0 ile değiştir, sonra log1p uygula
df['balance'] = np.log1p(df['balance'].clip(lower=0))
test_df['balance'] = np.log1p(test_df['balance'].clip(lower=0))

# Sonucu görselleştir
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['balance'])
plt.show()




df['duration'] = np.log1p(df['duration'])
test_df['duration']=np.log1p(test_df['duration'])
sns.boxplot(x=df['duration'])




Q1 = df['duration'].quantile(0.25)
Q3 = df['duration'].quantile(0.75)
IQR = Q3 - Q1
k = 2.95 
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR


outliers = df[(df['duration'] < lower_bound) | (df['duration'] > upper_bound)]

# Sonuçları yazdır
count = len(outliers)
ratio = count / len(df) * 100

print(f"duration sütununda aykırı değer sayısı: {count}")
print(f"Veri setindeki oranı: {ratio:.2f}%")
df['duration'] = df['duration'].clip(lower=lower_bound, upper=upper_bound)
test_df['duration'] = test_df['duration'].clip(lower=lower_bound, upper=upper_bound)



sns.boxplot(x=df['duration'])


Q1 = df['campaign'].quantile(0.25)
Q3 = df['campaign'].quantile(0.75)
IQR = Q3 - Q1

k = 1.5
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR

# Aykırı değerler
outliers = df[(df['campaign'] < lower_bound) | (df['campaign'] > upper_bound)]

# Sayı ve oran
outlier_count = outliers.shape[0]
outlier_ratio = outlier_count / df.shape[0]

print(f"Aykırı değer sayısı: {outlier_count}")
print(f"Aykırı değer oranı: {outlier_ratio:.2%}")  # Yüzde olarak gösterir



df['campaign'] = df['campaign'].clip(lower=lower_bound, upper=upper_bound)
Q1 = df['campaign'].quantile(0.25)
Q3 = df['campaign'].quantile(0.75)
IQR = Q3 - Q1

k = 1.5
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR

# Aykırı değerleri bul
outliers = df[(df['campaign'] < lower_bound) | (df['campaign'] > upper_bound)]

outlier_count = outliers.shape[0]
outlier_ratio = outlier_count / df.shape[0]

print(f"Aykırı değer sayısı: {outlier_count}")
print(f"Aykırı değer oranı: {outlier_ratio:.2%}")


df['campaign'] = df['campaign'].clip(lower=lower_bound, upper=upper_bound)
test_df['campaign'] = test_df['campaign'].clip(lower=lower_bound, upper=upper_bound)


df


df.isna().sum()


df.info()


from sklearn.preprocessing import LabelEncoder

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

categorical_cols


df.head()
test_df.head()



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

if train['y'].dtype == 'O':
    y = train['y'].map({'no':0, 'yes':1}).astype(int)
else:
    y = train['y'].astype(int)

X_raw  = train.drop(columns=['y', 'id'], errors='ignore').copy()
test_raw = test.drop(columns=['id'], errors='ignore').copy()

def feature_engineering(df: pd.DataFrame):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == 'O':
            df[c] = df[c].astype(str)

    # Log dönüşümleri
    for col in ['balance', 'duration', 'pdays', 'campaign', 'previous']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col + '_log1p'] = np.log1p(df[col].clip(lower=0))

    if 'pdays' in df.columns:
        df['pdays_is_minus1'] = (df['pdays'] == -1).astype(int)

    # duration flag
    if 'duration' in df.columns:
        df['is_long_duration'] = (df['duration'] > 500).astype(int)

    return df

X_fe = feature_engineering(X_raw)
test_fe = feature_engineering(test_raw)

cat_cols = X_fe.select_dtypes(include=['object']).columns.tolist()
X_lgb = X_fe.copy()
test_lgb = test_fe.copy()

if cat_cols:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(pd.concat([X_lgb[cat_cols], test_lgb[cat_cols]], axis=0).astype(str))
    X_lgb[cat_cols] = enc.transform(X_lgb[cat_cols].astype(str))
    test_lgb[cat_cols] = enc.transform(test_lgb[cat_cols].astype(str))


skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

oof = np.zeros(len(X_lgb))
pred_test = np.zeros(len(test_lgb))

lgb_params = dict(
    objective='binary',
    learning_rate=0.05,      
    num_leaves=50,            
    max_depth=-1,             
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=10000,
    early_stopping_rounds=200,
    random_state=42,
    verbose=-1
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lgb, y), 1):
    Xtr, Xva = X_lgb.iloc[tr_idx], X_lgb.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    oof[va_idx] = model.predict_proba(Xva)[:,1]
    pred_test += model.predict_proba(test_lgb)[:,1] / skf.n_splits

    auc = roc_auc_score(yva, oof[va_idx])
    print(f"[Fold {fold}] AUC={auc:.5f}")

print("OOF AUC:", roc_auc_score(y, oof))


submission = pd.DataFrame({
    "id": test["id"],
    "y": pred_test
})
submission.to_csv("submission.csv", index=False)



X,y


print((df['y']==1).sum())

