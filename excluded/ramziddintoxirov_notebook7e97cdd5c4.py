# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import os

# Papka manzili
path = "/kaggle/input/alpha-radar-solana-sprint"
files = sorted([f for f in os.listdir(path) if f.endswith('.csv') and f.startswith('evaluation_set')])
print("Topilgan fayllar:", files)

# Fayllarni birlashtiramiz
dfs = []
for f in files:
    df = pd.read_csv(os.path.join(path, f))
    print(f"{f}: {df.shape}")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print("Birlashgan dataset shakli:", data.shape)
print(data.head())

# Sample datasetni ham oâ€˜qiymiz
sample = pd.read_csv(os.path.join(path, "Sample_Dataset.csv"))
print("\nSample_Dataset ustunlari:", sample.columns.tolist())
print(sample.head())



import pandas as pd
import os

path = "/kaggle/input/alpha-radar-solana-sprint"
print(os.listdir(path))

# Fayl nomlarini tekshirib chiqamiz



path = "../input/alpha-radar-solana-sprint"
sample = pd.read_csv(f"{path}/Sample_Dataset.csv")
print(sample.shape)
sample.head()



# 1ï¸�âƒ£ Faylni oâ€˜qiymiz
path = "../input/alpha-radar-solana-sprint"
sample = pd.read_csv(f"{path}/Sample_Dataset.csv")

# 2ï¸�âƒ£ Faqat raqamli ustunlarni tanlaymiz
numeric_cols = sample.select_dtypes(include=np.number).columns.tolist()

# 3ï¸�âƒ£ mint_token_id boâ€˜yicha raqamli ustunlardan oâ€˜rtacha qiymatlarni olamiz
test_agg = sample.groupby("mint_token_id")[numeric_cols].mean().reset_index()

# 4ï¸�âƒ£ Natijani tekshiramiz
print(test_agg.shape)
test_agg.head()



import pandas as pd
import numpy as np

sample = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/Sample_Dataset.csv")

numeric_cols = sample.select_dtypes(include=np.number).columns.tolist()

# mint_token_id boâ€˜yicha oâ€˜rtacha qiymatlarni hisoblash
agg_data = sample.groupby("mint_token_id")[numeric_cols].mean().reset_index()



target = 'market_cap_usd'
features = [col for col in agg_data.columns if col != 'mint_token_id' and col != target]

X = agg_data[features]
y = agg_data[target]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    verbose=100
)

model.fit(X_train, y_train)



from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))



import numpy as np
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)



import os

# Kaggle input papkasidagi barcha fayllarni ko'rsatamiz
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1ï¸�âƒ£ Faylni yuklash
df = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/Sample_Dataset.csv")

# 2ï¸�âƒ£ Katta dataset boâ€˜lsa, 5000 qator namunani olish (tezroq)
df_sample = df.sample(n=5000, random_state=42)

# 3ï¸�âƒ£ Raqamli ustunlarni ajratish (target tashlab)
numeric_cols = df_sample.select_dtypes(include=['float64', 'int64']).columns
X = df_sample[numeric_cols].drop('token_quantity', axis=1)
y = df_sample['token_quantity']

# 4ï¸�âƒ£ Train-test boâ€˜linishi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5ï¸�âƒ£ Model yaratish (tezroq variant)
rf_model = RandomForestRegressor(
    n_estimators=50,  # daraxtlar sonini kamaytirdik
    max_depth=10,     # maksimal chuqurlikni kamaytirdik
    random_state=42,
    n_jobs=-1
)

# 6ï¸�âƒ£ Modelni fit qilish
rf_model.fit(X_train, y_train)

# 7ï¸�âƒ£ Bashorat qilish
y_pred = rf_model.predict(X_test)

# 8ï¸�âƒ£ Modelni baholash
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))



import matplotlib.pyplot as plt

# 1ï¸�âƒ£ Feature importance olish
importances = rf_model.feature_importances_
features = X.columns

# 2ï¸�âƒ£ DataFrame ga aylantirish
feat_imp_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# 3ï¸�âƒ£ Natijani chiqarish
print(feat_imp_df)

# 4ï¸�âƒ£ Vizualizatsiya qilish
plt.figure(figsize=(12,6))
plt.bar(feat_imp_df['feature'], feat_imp_df['importance'], color='skyblue')
plt.xticks(rotation=90)
plt.title("Feature Importance for Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1ï¸�âƒ£ Datasetni yuklash
df = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/Sample_Dataset.csv")

# 2ï¸�âƒ£ Eng muhim 10 ustun
top_features = [
    'token_delta', 'token_volume', 'creator_fee', 'virtual_token_reserves',
    'liquidity_ratio', 'sol_delta', 'sol_volume', 'creator_fee_pump',
    'virtual_sol_reserves', 'buy_count'
]

X = df[top_features]
y = df['token_quantity']

# 3ï¸�âƒ£ Train-test boâ€˜linishi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ğŸ”¹ Trainni kichraytirib olish (tezroq fit qilish uchun)
sample_frac = 0.2  # trainning faqat 20% ishlatiladi
X_train_small = X_train.sample(frac=sample_frac, random_state=42)
y_train_small = y_train.loc[X_train_small.index]

# 4ï¸�âƒ£ Model yaratish
rf_model = RandomForestRegressor(
    n_estimators=50,    # daraxtlar soni kamaytirildi
    max_depth=10,       # maksimal chuqurlik kamaytirildi
    random_state=42,
    n_jobs=-1
)

# 5ï¸�âƒ£ Modelni fit qilish
rf_model.fit(X_train_small, y_train_small)

# 6ï¸�âƒ£ Bashorat qilish
y_pred = rf_model.predict(X_test)

# 7ï¸�âƒ£ Modelni baholash
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))



import matplotlib.pyplot as plt

# Feature importance olish
importances = rf_model.feature_importances_
features = X.columns

# DataFramega joylash
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Natijani ko'rsatish
print(feature_importance_df)

# Diagramma chizish
plt.figure(figsize=(10,6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.gca().invert_yaxis()  # eng muhim ustun tepada bo'lsin
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance olish
importances = rf_model.feature_importances_
features = X.columns

# DataFrame ga joylash
feat_imp = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Chizma
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_imp, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ma'lumotni yuklash
df = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/Sample_Dataset.csv")

# Eng muhim 5 ustun
top5_features = ['sol_delta', 'creator_fee', 'virtual_sol_reserves', 'token_volume', 'sol_volume']
X = df[top5_features]
y = df['token_quantity']

# Train-test boâ€˜linishi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model yaratish (tezroq ishlash uchun)
rf_model = RandomForestRegressor(
    n_estimators=50,  # avvalgi 200 oâ€˜rniga 50
    max_depth=10,     # avvalgi 15 oâ€˜rniga 10
    random_state=42,
    n_jobs=-1
)

# Modelni fit qilish
rf_model.fit(X_train, y_train)

# Bashorat qilish
y_pred = rf_model.predict(X_test)

# Natijani baholash
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))



import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1ï¸�âƒ£ Ma'lumotni yuklash
train = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/Sample_Dataset.csv")

# 2ï¸�âƒ£ Eng muhim 5 ustun
top5_features = ['sol_delta', 'creator_fee', 'virtual_sol_reserves', 'token_volume', 'sol_volume']
X_train = train[top5_features]
y_train = train['token_quantity']

# 3ï¸�âƒ£ Model yaratish (tezroq variant)
rf_model = RandomForestRegressor(
    n_estimators=50,   # daraxtlarni kamaytirdik
    max_depth=10,      # chuqurlikni kamaytirdik
    random_state=42,
    n_jobs=-1
)

# 4ï¸�âƒ£ Fit qilish
rf_model.fit(X_train, y_train)

# 5ï¸�âƒ£ Submission tayyorlash
import glob

# Barcha evaluation fayllarini oâ€˜qish va birlashtirish
eval_files = glob.glob("/kaggle/input/alpha-radar-solana-sprint/evaluation_set_30s_chunk_*.csv")
submission = pd.DataFrame()

for f in eval_files:
    df = pd.read_csv(f)
    X_eval = df[top5_features]
    df['token_quantity'] = rf_model.predict(X_eval)
    submission = pd.concat([submission, df[['mint_token_id', 'token_quantity']]], axis=0)

# 6ï¸�âƒ£ CSV sifatida saqlash
submission.to_csv("submission.csv", index=False)
print("Submission tayyor!")



import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ma'lumotni yuklash
df = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/Sample_Dataset.csv")

features = ['sol_delta', 'creator_fee', 'virtual_sol_reserves', 'token_volume', 'sol_volume']
X = df[features]
y = df['token_quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM formatiga oâ€˜tkazish
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# Parametrlar
params = {
    'objective': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 10,
    'metric': 'rmse'
}

# Modelni fit qilish (early stopping callbacks bilan)
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)

# Bashorat qilish
y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

# Natijani baholash
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))



import pandas as pd

# 1ï¸�âƒ£ Test datasetni yuklash (o'zingizga mos yoâ€˜l bilan)
X_test_submission = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/evaluation_set_30s_chunk_001.csv")

# 2ï¸�âƒ£ Faqat modelda ishlatilgan ustunlarni tanlaymiz
top_features = ['sol_delta', 'creator_fee', 'virtual_sol_reserves', 'token_volume', 'sol_volume']
X_test_submission = X_test_submission[top_features]

# 3ï¸�âƒ£ Bashorat qilish
# Agar siz RandomForest modelini ishlatayotgan boâ€˜lsangiz:
y_test_pred = rf_model.predict(X_test_submission)

# Agar LightGBM modelini ishlatayotgan boâ€˜lsangiz:
# y_test_pred = lgb_model.predict(X_test_submission)

# 4ï¸�âƒ£ Submission DataFrame yaratish
submission = pd.DataFrame({
    "mint_token_id": X_test_submission.index,  # yoki test faylida token_id ustuni boâ€˜lsa: "X_test_submission['mint_token_id']"
    "token_quantity": y_test_pred
})

# 5ï¸�âƒ£ CSV faylga saqlash
submission.to_csv("submission.csv", index=False)

print("Submission CSV tayyor!")



# Fit qilish uchun ishlatilgan ustunlar
used_features = top5_features  # sizning eng muhim 5 ustun



X_test_submission = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/evaluation_set_30s_chunk_001.csv")

# Test datasetda kerakli ustunlar borligini tekshiring
X_test_submission = X_test_submission[used_features]  # faqat fit qilgan ustunlar



y_test_pred = rf_model.predict(X_test_submission)



submission = pd.DataFrame({
    'index': range(len(y_test_pred)),  # 0,1,2,...
    'token_quantity': y_test_pred
})
submission.to_csv("submission.csv", index=False)



import pandas as pd

# test faylini yuklash (faqat uzunligini bilish uchun)
test_df = pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/evaluation_set_30s_chunk_001.csv")

# model bashorati (y_test_pred) tayyor boâ€˜lgan deb faraz qilamiz
# Submission DataFrame yaratish
submission = pd.DataFrame({
    'index': range(len(y_test_pred)),  # 0,1,2,...
    'token_quantity': y_test_pred
})

# CSV faylga saqlash
submission.to_csv("submission.csv", index=False)

print("Submission CSV tayyor!")


