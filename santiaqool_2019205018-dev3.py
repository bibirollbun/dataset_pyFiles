import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Veri setlerini pandas DataFrame olarak okuyalım
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print(f"Eğitim verisi boyutu: {train_df.shape}")
print(f"Test verisi boyutu: {test_df.shape}")
train_df.head()


import matplotlib.pyplot as plt


# genel şekil ve veri tipi analizi
print(train_df.shape)
print(test_df.shape)
display(train_df.head())
print(train_df.dtypes.value_counts())




# Hedef değişkenin dağılımı
plt.figure()
train_df['accident_risk'].hist(bins=30)
plt.xlabel('accident_risk')
plt.ylabel('Frekans')
plt.title('Hedef Değişken Dağılımı')
plt.show()

#kategori ile ilgili olan sütunları seçme
cat_cols = list(train_df.select_dtypes(include=['object', 'bool']).columns)

#bütün kategoriler için risk ile bağlantı grafiği gözle görmek adına
for col in cat_cols:
    top_cats = train_df[col].value_counts().index
    tmp = train_df[train_df[col].isin(top_cats)].groupby(col)['accident_risk'].mean().sort_values()
    plt.figure()
    tmp.plot(kind='barh')
    plt.title(f'{col} kırılımında accident_risk ortalaması')
    plt.xlabel('accident_risk ortalaması')
    plt.show()



def make_features(df):
# speed_limit varsa aralıklara bölelim daha fazla anlam için
    bins = [0, 30, 50, 70, 90, 110, 1000]
    labels = ['0-30', '30-50', '50-70', '70-90', '90-110', '110+']
    df['speed_limit_bin'] = pd.cut(df['speed_limit'], bins=bins, labels=labels, include_lowest=True)

# Hem train hem test için uygulayalım
make_features(train_df)
make_features(test_df)


# 'id' sütunlarını daha sonra gönderim dosyası için ayıralım
train_ids = train_df['id']
test_ids = test_df['id']

# Modelde kullanmayacağımız 'id' sütununu kaldıralım
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Kategorik ve boolean sütunları seçelim
#buraya ekstra bir object ekledim hızlar sebebiyle
categorical_cols = train_df.select_dtypes(include=['object', 'bool', 'category']).columns


# pd.get_dummies() ile bu sütunları otomatik olarak 0'lara ve 1'lere çevirelim
train_processed = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_processed = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Sütunları hizalamak için train ve test setlerini hizalayalım
# Bu, her iki veri setinin de tam olarak aynı sütunlara sahip olmasını sağlar
train_labels = train_processed['accident_risk']
X = train_processed.drop('accident_risk', axis=1)
X_test = test_processed.reindex(columns=X.columns, fill_value=0)

X.head()


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


X_train, X_val, y_train, y_val = train_test_split(X, train_labels, test_size=0.2, random_state=40)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_val)
lr_pred = np.clip(lr_pred, 0, 1)  

rmse_lr = mean_squared_error(y_val, lr_pred, squared=False)
print(f"LinearRegression RMSE: {rmse_lr:.5f}")


rf_model = RandomForestRegressor(n_estimators=120, random_state=40)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_val)
rf_pred = np.clip(rf_pred, 0, 1)

rmse_rf = mean_squared_error(y_val, rf_pred, squared=False)
print(f"RandomForestRegressor RMSE: {rmse_rf:.5f}")


gbr_model = GradientBoostingRegressor(random_state=40)
gbr_model.fit(X_train, y_train)

gbr_pred = gbr_model.predict(X_val)
gbr_pred = np.clip(gbr_pred, 0, 1)

rmse_gbr = mean_squared_error(y_val, gbr_pred, squared=False)
print(f"GradientBoostingRegressor RMSE: {rmse_gbr:.5f}")



final_model = gbr_model
importances = final_model.feature_importances_
idx = np.argsort(importances)[-15:]
names = np.array(X.columns)[idx]
vals = importances[idx]

plt.figure(figsize=(8,5))
y_pos = np.arange(len(names))
plt.barh(y_pos, vals)
plt.yticks(y_pos, names)
plt.title("En Önemli 15 Özellik")
plt.xlabel("Önem (göreli)")
plt.tight_layout()
plt.show()



#tüm veri setiyle eğitim
final_model = GradientBoostingRegressor(random_state=40)
final_model.fit(X, train_labels)

test_pred = final_model.predict(X_test)
test_pred = np.clip(test_pred, 0, 1)

submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': test_pred})
OUT_CSV = 'submission_homework3.csv'  
submission_df.to_csv(OUT_CSV, index=False)
print(f"{OUT_CSV} dosya kaydedildi")
display(submission_df.head())


