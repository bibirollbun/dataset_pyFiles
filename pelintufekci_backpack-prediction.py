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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# === VERİ YÜKLEME VE BİRLEŞTİRME ===
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
training_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df = pd.concat([train, training_extra, test], ignore_index=True)

# === VERİ GÖZLEM FONKSİYONU ===
def check_df(dataframe):
    print("Shape:", dataframe.shape)
    print("Types:\n", dataframe.dtypes)
    print("Head:\n", dataframe.head(3))
    print("Tail:\n", dataframe.tail(3))
    print("NA counts:\n", dataframe.isnull().sum())
    num_cols = dataframe.select_dtypes(include=["float64", "int64"]).columns
    print("Quantiles:\n", dataframe[num_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# === SÜTUN TİPLERİNİ YAKALAMA ===
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}, Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}, num_cols: {len(num_cols)}, cat_but_car: {len(cat_but_car)}, num_but_cat: {len(num_but_cat)}")
    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)


######################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, True)


######################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)  # Görsel çıktı istemiyorsak False

######################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
######################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Price", col)


# Bağımlı değişkenin incelenmesi
plt.figure(figsize=(10, 6))
sns.histplot(train['Price'], kde=True)
plt.title('Price Distribution')
plt.show()

######################################
# Eksik Değer Analizi
######################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)



# Weight Capacity ve Laptop Compartment etkileşimli yeni özellikler
# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.01, up_quantile=0.99):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "Price" and col != "id":
        print(col, check_outlier(df, col))



# === TEMİZLİK VE DÖNÜŞÜMLER ===
df.drop("Color", axis=1, inplace=True)
# Tüm sayısal sütunları yakalayalım (Price dahil değil)
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [col for col in num_cols if col not in ["Price", "id"]]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
print(df[num_cols].isnull().sum())
cat_cols = [col for col in cat_cols if col != "Color"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# <<< Price DÜZELTME ADIMI >>>
# Fiyatı 150 olanların medyanla değiştirilmesi
price_median = df.loc[df["Price"] != 150, "Price"].median()
df["Price"] = df["Price"].apply(lambda x: price_median if x == 150 else x)


# === LABEL ENCODING & ONE-HOT ENCODING ===
binary_cols = ['Laptop Compartment', 'Waterproof']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
ohe_cols = ['Brand', 'Material', 'Size', 'Style']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# 1) Ağırlık başına bölme var mı?
df["WeightPerCompartment"] = df["Weight Capacity (kg)"] / (df["Laptop Compartment"] + 1)

# 2) Laptop bölmesi varsa ağırlık çarpanı
df["WeightXCompartment"] = df["Weight Capacity (kg)"] * df["Laptop Compartment"]

# 3) Bölmenin ağırlığa oranı
df["Weight_Compartment_Ratio"] = df["Laptop Compartment"] / df["Weight Capacity (kg)"]


# === TRAIN / TEST AYIRIMI ===
train_df = df[df["Price"].notnull()].copy()
test_df = df[df["Price"].isnull()].copy()
CATS = [col for col in cat_cols if col in train_df.columns]


# === YENİ: WEIGHT BASED FEATURE ENGINEERING ===


# === MODELLEME ===
X = train_df.drop(columns=['Price', 'id'])
y = train_df['Price']
X_train_model, X_test_model, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_model)
X_test_scaled = scaler.transform(X_test_model)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_model.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_model.columns)

lgb = LGBMRegressor(random_state=42)
lgb.fit(X_train_scaled_df, y_train)
y_pred = lgb.predict(X_test_scaled_df)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"LightGBM: RMSE = {rmse:.4f}, R² = {r2:.4f}")

# === FEATURE IMPORTANCE ===
def plot_importance(model, features, num=20, save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    feature_imp = feature_imp.sort_values("Value", ascending=False)

    plt.figure(figsize=(10, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.iloc[:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig("importances.png")

plot_importance(lgb, X)



########################################
# test dataframeindeki boş olan salePrice değişkenlerini tahminleyiniz ve
# Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturunuz. (Id, SalePrice)
########################################

# Test verisi için tahminler
X_test_submission = test_df.drop(["id", "Price"], axis=1)
test_predictions = lgb.predict(X_test_submission)


# Submission dosyasını oluşturalım
submission = pd.DataFrame({
    "id": test_df["id"].astype(int),
    "Price": test_predictions
})

# Dosyayı kaydedelim
submission.to_csv("submission.csv", index=False)

print("İşlem tamamlandı. submission.csv dosyası oluşturuldu.")

