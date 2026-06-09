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


# Kullandığım kütüphane importlarımı ve fonksiyonlarımı ekledim
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import  seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def grab_col_names(dataframe,cat_th=10,car_th=20):
    cat_cols = [col for col in dataframe
                if str(dataframe[col].dtypes) in ["object","category"]]
    num_but_cat = [col for col in dataframe
                    if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes in ["int" ,"float"]]
    cat_but_car = [col for col in dataframe if
                   dataframe[col].nunique() > car_th and
                   str(dataframe[col].dtypes) in ["category","object"]]

    cat_cols = cat_cols+num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe if dataframe[col].dtypes in ["float","int"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols,num_cols,cat_but_car


df_tr_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_tr = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_te = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# Bu proje kapsamında daha önce denediğm submissionlar sonucu extra veri setini eklemeyi yararlı buldum
df_tr = pd.concat([df_tr, df_tr_extra])


# Kategorik numerik kolonlarımı ayrıştırdım veri seti hakkında ufak bilgilere baktım
cat_cols_te,num_cols_te,cat_but_car_te =grab_col_names(df_te)
cat_cols_tr,num_cols_tr,cat_but_car_tr =grab_col_names(df_tr)


# train ve test veri setimde herhangi bir outlier veriye rastlamadım burada cpu zorlandığığı kodları yorum satırı olarak verdim


"""
for col in num_cols_tr:
    plt.figure(figsize=(10, 5))
    sns.boxplot(df_tr[col])
    plt.show(block=True)
"""


"""
for col in num_cols_te:
    plt.figure(figsize=(10, 5))
    sns.boxplot(df_te[col])
    plt.show(block=True)
"""


# önceki denemelerim kapsamında eksik verilerimden sadece sayısal bir kolonu median
# ile doldurdum modelime iyi bir etki yarattı


missing_percentage = df_tr.isnull().mean() * 100
print("\nEksik Veri Yüzdesi:\n", missing_percentage)


# eksik değerleri median ile doldurdum tek sayısal değişkenim vardı
df_tr["Weight Capacity (kg)"] = df_tr["Weight Capacity (kg)"].fillna(df_tr["Weight Capacity (kg)"].median())


missing_percentage = df_te.isnull().mean() * 100
print("\nEksik Veri Yüzdesi:\n", missing_percentage)


df_te["Weight Capacity (kg)"] = df_te["Weight Capacity (kg)"].fillna(df_te["Weight Capacity (kg)"].median())


# Cep sayısının fazla olması ile ilgili bir değişken oluşturdum 
df_tr["NEW_Many_Compartments"] = df_tr["Compartments"].apply(lambda x: 1 if x > 7 else 0)
df_te["NEW_Many_Compartments"] = df_te["Compartments"].apply(lambda x: 1 if x > 7 else 0)


# Laptop bölmesi varsa rengi siyahi koyu renk mi
df_tr.loc[ (df_tr["Laptop Compartment"] == "Yes") &
           ( (df_tr["Color"] == "Black") | (df_tr["Color"] == "Gray") ) , "NEW_Businness_Backpack"] = 1
df_tr["NEW_Businness_Backpack"].fillna(0,inplace=True)

df_te.loc[ (df_te["Laptop Compartment"] == "Yes") &
           ( (df_te["Color"] == "Black") | (df_te["Color"] == "Gray") ) , "NEW_Businness_Backpack"] = 1
df_te["NEW_Businness_Backpack"].fillna(0,inplace=True)


# Çanta küçükse renklimi
df_tr.loc[ (df_tr["Size"] == "Small") &
           ( (df_tr["Color"] != "Black") | (df_tr["Color"] != "Gray") ) , "NEW_Child_Backpack"] = 1
df_tr["NEW_Child_Backpack"].fillna(0,inplace=True)

df_te.loc[ (df_te["Size"] == "Small") &
           ( (df_te["Color"] != "Black") | (df_te["Color"] != "Gray") ) , "NEW_Child_Backpack"] = 1
df_te["NEW_Child_Backpack"].fillna(0,inplace=True)


# kategorik değişkenlerim kırılımında numerik değişkenlerimin betimsel istatistiklerini 
# inceliyorum iterafif bir şekilde bir fonksiyon yazıyorum 
num_cols_tr = [col for col in num_cols_tr if "Price" != col]
def feature_match(df,num_cols,cat_cols):
    for n_col in num_cols:
        for col in cat_cols:
            print(f"#####{col}###{n_col}##")
            print(df.groupby(col)[n_col].describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.99]))


feature_match(df_tr,num_cols_tr,cat_cols_tr)


# Material Canvas olan çantaların Compartment sayısında bir ilişki yakalayabilirmiyim bakıyorum  
df_tr.groupby(["Material","Size"])["Compartments"].describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.99])


# Material Canvas olan çantaların bölme sayısı ağırlıklarıyla bir değişken türetiyorum
df_tr["NEW_Canvas_Compartment_Interaction"] = (df_tr["Material"] == "Canvas").astype(int) * df_tr["Compartments"]

df_te["NEW_Canvas_Compartment_Interaction"] = (df_te["Material"] == "Canvas").astype(int) * df_te["Compartments"]


# çanta su geçirmez ise daha mı ağır Waterproof(su geçirmeme) yes olan çantalar
df_tr["NEW_Waterproof_Binary"] = df_tr["Waterproof"].map({"Yes": 1, "No": 0})
df_tr.drop(columns=["Waterproof"], inplace=True)

df_te["NEW_Waterproof_Binary"] = df_te["Waterproof"].map({"Yes": 1, "No": 0})
df_te.drop(columns=["Waterproof"], inplace=True)

# Bu değişken, yalnızca su geçirmez olanların ağırlığını korur, diğerlerini sıfırlar.
df_tr["NEW_Waterproof_Weight"] = df_tr["Weight Capacity (kg)"] * df_tr["NEW_Waterproof_Binary"]

df_te["NEW_Waterproof_Weight"] = df_te["Weight Capacity (kg)"] * df_te["NEW_Waterproof_Binary"]


# Compartments değişkenini kategorikleştiriyorum
df_tr["NEW_Compartments"] = pd.qcut(df_tr["Compartments"], q=6,
                                    labels=["Very Low", "Low", "Below Average", "Average", "Above Average", "High"])

df_te["NEW_Compartments"] = pd.qcut(df_te["Compartments"], q=6,
                                    labels=["Very Low", "Low", "Below Average", "Average", "Above Average", "High"])



df_tr.shape
df_te.shape


dff_tr = pd.get_dummies(df_tr, drop_first=True)
dff_te = pd.get_dummies(df_te, drop_first=True)


# Eğitim veri setini yükle
X_train = dff_tr.drop(columns=['Price'])  # Eğitim verisi, 'target' dışındaki özellikler
y_train = dff_tr['Price']  # Eğitim verisinin hedef değeri
# Modeli oluştur ve eğit
# XGBRegressor modelini oluştur ve eğit
model = XGBRegressor(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


# Eğitim verisi üzerinde tahmin yap
y_train_pred = model.predict(X_train)

# MSE hesapla
mse_train = mean_squared_error(y_train, y_train_pred)

# RMSE hesapla
rmse_train = np.sqrt(mse_train)

print(f"Eğitim Verisi RMSE: {rmse_train}")


# Özellik önem dereceleri için grafik oluşturalım
feature_importances = model.feature_importances_
features = X_train.columns

# Özellik önemlerini bir DataFrame'e koy
feat_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Hata kontrolü: Boş olup olmadığını kontrol et
if feat_importance_df.empty:
    print("Hata: Özellik önem dereceleri boş! Model eğitildi mi?")
else:
    # Veriyi sıralayarak ilk 15 özelliği al
    feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

    # Seaborn ile grafik çizdir
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_importance_df, palette='Blues_r')
    plt.xlabel("Önem Derecesi")
    plt.ylabel("Özellikler")
    plt.title("XGBoost Özellik Önem Dereceleri")
    plt.show()


print(num_cols_te.columns)




