# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_theme(style="whitegrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train


test


test['y']=-1


train


import matplotlib.pyplot as plt

train['day']=train['day'].astype('str')

def kategorik_target_dagilim(df, target_col):
    kategorik_kolonlar = df.select_dtypes(include=["object", "category"]).columns

    for col in kategorik_kolonlar:
        if col == target_col:
            continue  # hedef değişkeni atla

        # Crosstab ile yüzde tablosu
        tablo = pd.crosstab(df[col], df[target_col], normalize="index") * 100
        print(tablo.round(2))
        
        # Grafik
        tablo.plot(kind="bar", stacked=True, figsize=(6,4))
        plt.title(f"{col} - {target_col} dağılımı (%)")
        plt.ylabel("Yüzde")
        plt.xlabel(col)
        plt.legend(title=target_col)
        plt.xticks(rotation=45)
        plt.show()


kategorik_target_dagilim(train, 'y')


import seaborn as sns
for feature in train.select_dtypes('int').columns.tolist():
    plt.figure(figsize=(8,4))
    sns.boxplot(x='y', y=feature, data=train)
    plt.show()


from scipy.stats import f_oneway
sayısal_features=train.select_dtypes('int').columns.tolist()
df=train
for feature in sayısal_features:
    groups = [df[df['y']==cat][feature] for cat in df['y'].unique()]
    f_stat, p_val = f_oneway(*groups)
    print(f"{feature}: F={f_stat:.2f}, p={p_val:.4f}")



cat=df.select_dtypes(include=['object','category']).columns.tolist()


percentiles = np.linspace(0.05, 0.95, 19)

summary = df.describe(percentiles=percentiles)




summary


df.columns


from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

# Bağımlı ve bağımsız değişkenleri ayır
X = df.drop(columns=['y','id'])
y = df['y']

# Sadece sayısal kolonlar
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols.remove("y")  # eğer y de numeric_cols'e girdiyse çıkar
numeric_cols.remove("id")
# Train-test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y,stratify=y, test_size=0.2, random_state=42
)

# PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True)

# Sadece numeric kolonları fit et
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = pt.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = pt.transform(X_test[numeric_cols])

# Eğer ayrı test datası varsa (örn. Kaggle test seti gibi)
test_scaled = test.copy()
test_scaled[numeric_cols] = pt.transform(test[numeric_cols])



X_train_scaled.describe()


X_train_scaled.columns


from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


model = CatBoostClassifier(
    iterations=1000,       # Boosting tur sayısı
    learning_rate=0.05,    # Öğrenme oranı
    depth=6,               # Ağaç derinliği
    eval_metric='AUC',     # Değerlendirme metriği
    random_seed=42,
    verbose=100
)

# Modeli eğit (Pool ile kategorik sütunları belirtiyoruz)
train_pool = Pool(X_train_scaled, y_train, cat_features=cat)
test_pool = Pool(X_test_scaled, y_test, cat_features=cat)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# Tahmin
y_pred = model.predict(test_pool)
y_prob = model.predict_proba(test_pool)[:,1]  # AUC vs için

# Sonuçları değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



sub=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

test_scaled=test_scaled.drop(columns=['id','y'])
submit_pool = Pool(test_scaled, cat_features=cat)
y_prob = model.predict_proba(submit_pool)[:, 1]

sub['y']=y_prob
sub.to_csv("submission.csv", index=False)


sub


from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, y_pred)

