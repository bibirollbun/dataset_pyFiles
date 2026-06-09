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


!pip install pyspark


import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random
import os

from pyspark.sql import SparkSession 
from pyspark.ml  import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql import SQLContext  
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
import pyspark.sql.functions as F
from pyspark.sql.functions import col, hour, minute, dayofweek
from pyspark.sql.types import DoubleType
import math
from pyspark.sql.functions import max
from pyspark.sql.functions import col, exp
from math import radians, sin, cos, sqrt, atan2
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator






spark = SparkSession.builder.appName("taxi").getOrCreate()


df_taxi= spark.read.csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv'
                        , inferSchema = True
                        , header=True)


df_taxi.limit(5).toPandas()


df_taxi.printSchema()


pd.DataFrame(df_taxi.dtypes, columns = ['Column Name','Data type'])


def count_missings(spark_df,sort=True):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'string', 'date')]).toPandas()

    if len(df) == 0:
        print("Tidak ada missing value!")
        return None

    if sort:
        return df.rename(index={0: 'count'}).T.sort_values("count",ascending=False)

    return df


count_missings(df_taxi)


#Disampling untuk visualisasi
sampled_df = df_taxi.sample(withReplacement=False, fraction=0.05,seed=1)
sampled_df = sampled_df.toPandas()


# #Sampling cara lain
# df_taxi.createOrReplaceTempView("taxi_table")

# sampled_sql_df = spark.sql("""
#     SELECT *
#     FROM taxi_table
#     WHERE RAND() <= 0.05
# """)

# sampled_df = sampled_sql_df.toPandas()


sampled_df.isnull().sum()


sampled_df = sampled_df.dropna()


sampled_df.isnull().sum()


sampled_df.describe()


sampled_df.info()


sampled_df.count()


sampled_df.head()


sampled_df[sampled_df['passenger_count']==208]


sampled_df.columns


num_feature=['fare_amount', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']

for kolom in num_feature:
    plt.figure(figsize=(10,6))
    sampled_df[kolom].plot(kind='box')
    plt.title(f'Distribusi {kolom}')
    plt.ylabel('Frekuensi')


sampled_df['fare_amount'].max()


sampled_df_clean = sampled_df

for column in sampled_df_clean.columns:

    Q1 = sampled_df_clean[column].quantile(0.25)
    Q3 = sampled_df_clean[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    sampled_df_clean = sampled_df_clean[
        (sampled_df_clean[column] >= lower) & (sampled_df_clean[column] <= upper)
    ]



num_feature=['fare_amount', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']

for kolom in num_feature:
    plt.figure(figsize=(10,6))
    sampled_df_clean[kolom].plot(kind='box')
    plt.title(f'Distribusi {kolom}')
    plt.ylabel('Frekuensi')



sampled_df_clean['pickup_datetime'] = pd.to_datetime(sampled_df_clean['pickup_datetime'])

# fungsi haversine untuk menghitung jarak dari 2 titik kordinat di bumi
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

# Apply fungsinya
sampled_df_clean['traveled_distance'] = sampled_df_clean.apply(
    lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'],
                          row['dropoff_latitude'], row['dropoff_longitude']), axis=1)



# ekstraksi jam, menit, tahun dan bulan
sampled_df_clean['pickup_hour'] = sampled_df_clean['pickup_datetime'].dt.hour
sampled_df_clean['pickup_minute'] = sampled_df_clean['pickup_datetime'].dt.minute
sampled_df_clean['pickup_year'] =sampled_df_clean['pickup_datetime'].dt.year
sampled_df_clean['pickup_month'] =sampled_df_clean['pickup_datetime'].dt.month

# jarak/penumpang
sampled_df_clean['distance_per_passenger'] = sampled_df_clean['traveled_distance'] / sampled_df_clean['passenger_count']

# Hari (1=Senin, 7=Minggu)
sampled_df_clean['pickup_day_of_week'] = sampled_df_clean['pickup_datetime'].dt.dayofweek + 1

# Akhir Pekan (1 jika ya, 0 jika tidak)
sampled_df_clean['is_weekend'] = np.where(sampled_df_clean['pickup_day_of_week'].isin([6, 7]), 1, 0)


# Fitur jam sibuk (07:00–09:00 atau 17:00–19:00)
sampled_df_clean['is_rush_hour'] = np.where(
    ((sampled_df_clean['pickup_hour'] >= 7) & (sampled_df_clean['pickup_hour'] < 9)) | 
    ((sampled_df_clean['pickup_hour'] >= 17) & (sampled_df_clean['pickup_hour'] < 19)), 
    1, 0
)

# Fitur jam istirahat (20:00–06:00)
sampled_df_clean['is_rest_time'] = np.where(
    ((sampled_df_clean['pickup_hour'] >= 20) | (sampled_df_clean['pickup_hour'] < 6)),
    1, 0
)






sampled_df_clean.head()


sampled_df_clean.columns


num_df= sampled_df_clean.select_dtypes(include=['int64', 'float64','int32'])

# Plot heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


sampled_df_clean['traveled_distance'].plot(kind='hist',bins=50)


sampled_df_clean['fare_amount'].plot(kind='hist',bins=50)


plt.figure(figsize=(10, 12))
sns.regplot(
    data=sampled_df_clean,
    x='fare_amount',
    y='traveled_distance',
    scatter_kws={'alpha': 0.003},
    line_kws={'color': 'red'},
    ci=None
)
plt.title('Scatter Plot: Trip Distance vs Fare Amount + Regression Line')
plt.xlabel('Fare Amount (USD)')
plt.ylabel('Trip Distance (miles)')
plt.grid(True)
plt.show()


df_taxi.printSchema()


df_taxi.count()


df_clean = df_taxi.dropna()


df_clean = df_clean.dropna()


df_clean.count()


df_clean.select(max(df_clean.fare_amount)).show()


for column in sampled_df.columns:
    Q1 = sampled_df[column].quantile(0.25)
    Q3 = sampled_df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    print(f"Kolom '{column}':")
    print(f"  - Batas bawah: {lower}")
    print(f"  - Batas atas: {upper}")
    print('-' * 30)


#Drop outlier sesuai batasan yang didapat dari sample (Beberapa diubah sesuai kira-kira dan dibulatkan)
df_clean = df_clean.filter('fare_amount > 0 and fare_amount <= 40')
df_clean = df_clean.filter('pickup_longitude >= -75.0295855 and pickup_longitude <= -72.9295975')
df_clean = df_clean.filter('pickup_latitude >= 39.68659242938233 and pickup_latitude <= 41.8154557423706')
df_clean = df_clean.filter('dropoff_longitude >= -75.0295855 and dropoff_longitude <= -72.9295975')
df_clean = df_clean.filter('dropoff_latitude >= 39.68659242938233 and dropoff_latitude <=  41.8154557423706')
df_clean = df_clean.filter('passenger_count > 0 and passenger_count <= 7')




print("\nJumlah data setelah filter outlier:")
print(df_clean.count())


# Mendaftarkan fungsi UDF (User Defined Function) untuk menghitung jarak haversine di PySpark
haversine_udf = F.udf(haversine, DoubleType())

# Menghitung jarak tempuh menggunakan rumus Haversine dan menyimpannya di kolom 'traveled_distance'
df_clean = df_clean.withColumn(
    'traveled_distance', 
    haversine_udf(
        df_clean['pickup_latitude'], 
        df_clean['pickup_longitude'], 
        df_clean['dropoff_latitude'], 
        df_clean['dropoff_longitude']
    )
)

# Mengekstrak jam dari waktu penjemputan
df_clean = df_clean.withColumn('pickup_hour', F.hour(df_clean['pickup_datetime']))

# Mengekstrak menit dari waktu penjemputan
df_clean = df_clean.withColumn('pickup_minute', F.minute(df_clean['pickup_datetime']))

# Menghitung jarak per penumpang
df_clean = df_clean.withColumn(
    'distance_per_passenger', 
    df_clean['traveled_distance'] / df_clean['passenger_count']
)

# Mendapatkan hari dalam seminggu dari waktu penjemputan (1 = Minggu, 2 = Senin, ..., 7 = Sabtu)
df_clean = df_clean.withColumn('pickup_day_of_week', F.dayofweek(df_clean['pickup_datetime']))

# Menambahkan fitur apakah penjemputan terjadi di akhir pekan (1 = akhir pekan, 0 = bukan)
df_clean = df_clean.withColumn(
    'is_weekend', 
    F.when(df_clean['pickup_day_of_week'].isin([7, 1]), 1).otherwise(0)
)

# Menambahkan fitur apakah waktu penjemputan berada pada jam sibuk (07:00–09:00 atau 17:00–19:00)
df_clean = df_clean.withColumn(
    'is_rush_hour', 
    F.when(
        ((df_clean['pickup_hour'] >= 7) & (df_clean['pickup_hour'] < 9)) | 
        ((df_clean['pickup_hour'] >= 17) & (df_clean['pickup_hour'] < 19)),
        1
    ).otherwise(0)
)

# Menambahkan fitur apakah waktu penjemputan berada pada jam istirahat (20:00–06:00)
df_clean = df_clean.withColumn(
    'is_rest_time', 
    F.when(
        (df_clean['pickup_hour'] >= 20) | (df_clean['pickup_hour'] < 6),
        1
    ).otherwise(0)
)

# Mengekstrak tahun dari waktu penjemputan
df_clean = df_clean.withColumn('pickup_Year', F.year(df_clean['pickup_datetime']))

# Mengekstrak bulan dari waktu penjemputan
df_clean = df_clean.withColumn('pickup_Month', F.month(df_clean['pickup_datetime']))



df_clean.limit(10).toPandas()


df_clean = df_clean.na.fill(0) 


# Membagi data menjadi 70% data latih dan 30% data uji
train_data, test_data = df_clean.randomSplit([0.7, 0.3], seed=42)

# Daftar kolom fitur yang digunakan dalam model
feature_columns = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 
                   'passenger_count', 'pickup_hour', 'pickup_minute', 'distance_per_passenger', 
                   'pickup_day_of_week', 'is_weekend', 'is_rush_hour', 'is_rest_time',
                   'pickup_Year', 'pickup_Month', 'traveled_distance']

# Mengubah semua kolom fitur ke dalam tipe data numerik
for col_name in feature_columns:
    df_clean = df_clean.withColumn(col_name, col(col_name).cast("double"))

# Menggabungkan kolom fitur menjadi satu vektor fitur
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")


#jumlah train data (pakai print biar cepet, udah pernah di run 1x)
#train_data.count()
print(36802678)


#jumlah test data
print(52573248-36802678)


best_model = None
best_rmse = float("inf")
best_numTrees = None
metrics_dict = {}
evaluator_rmse = RegressionEvaluator(labelCol='fare_amount', predictionCol="prediction", metricName="rmse")

for num_trees in [10,30,50,70,90]:
    rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount", numTrees=num_trees)
    pipeline = Pipeline(stages=[assembler, rf])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    
    rmse = evaluator_rmse.evaluate(predictions)
    metrics_dict[num_trees] = {
        "model": model,
        "predictions": predictions,
        "rmse": rmse
    }

    print(f"numTrees={num_trees}, RMSE={rmse}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_predictions = predictions
        best_numTrees = num_trees



# Evaluator metrik lainnya
evaluator_r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="mae")
evaluator_mse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="mse")

# Evaluasi akhir dari model terbaik
rmse = evaluator_rmse.evaluate(best_predictions)
r2 = evaluator_r2.evaluate(best_predictions)
mae = evaluator_mae.evaluate(best_predictions)
mse = evaluator_mse.evaluate(best_predictions)

print(f"\n==== Model Terbaik: Random Forest dengan numTrees={best_numTrees} ====")
print(f"RMSE : {rmse}")
print(f"R²   : {r2}")
print(f"MAE  : {mae}")
print(f"MSE  : {mse}")


# Ambil sample data prediksi untuk visualisasi
sampled_pred = best_predictions.select("fare_amount", "prediction").sample(False, 0.01, seed=42).toPandas().reset_index(drop=True)


sampled_pred.head()


# Hitung error/residuals
residuals = sampled_pred['fare_amount'] - sampled_pred['prediction']

# Plot 
plt.figure(figsize=(12, 6))
plt.plot(sampled_pred.index[:100], residuals[:100], label='Residuals', color='red', alpha=0.6)
plt.axhline(0, color='black', linestyle='--')  #(Garis di 0 kalau nggak ada error)
plt.xlabel("Sample Index")
plt.ylabel("Error (Nilai Asli - Prediksi)")
plt.title("Plot Error RF (Sample)")
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(12, 6))
plt.scatter(sampled_pred.index[:200], sampled_pred['fare_amount'][:200], label='Nilai Asli', color='red', alpha=0.3)
plt.scatter(sampled_pred.index[:200], sampled_pred['prediction'][:200], label='Prediksi', color='blue', alpha=0.3)
plt.xlabel("Sample Index")
plt.ylabel("Fare Amount")
plt.title("Prediksi RF vs Nilai Asli (Sample)")
plt.legend()
plt.grid(True)
plt.show()



# Plot prediksi dan nilai asli
plt.figure(figsize=(20, 6))
plt.plot(sampled_pred['fare_amount'][:100], label='Nilai Asli', color='red', alpha=0.5)
plt.plot(sampled_pred['prediction'][:100], label='Prediksi', color='blue', alpha=0.5)
plt.xlabel("Sample Index")
plt.ylabel("Fare Amount")
plt.title("Prediksi RF vs Nilai Asli (Sample)")
plt.legend()
plt.grid(True)
plt.show()


# Nilai-nilai elasticNetParam yang ingin diuji
elastic_net_values = [0.0, 0.25, 0.5, 0.75, 1.0]

best_model_en = None
best_rmse_en = float("inf")
best_en_param = None
metrics_dict_en = {}

for en_param in elastic_net_values:
    lr = LinearRegression(featuresCol="features", labelCol="fare_amount", elasticNetParam=en_param)
    pipeline = Pipeline(stages=[assembler, lr])
    
    model = pipeline.fit(train_data)
    predictions_en = model.transform(test_data)
    
    rmse = evaluator_rmse.evaluate(predictions_en)
    metrics_dict_en[en_param] = {
        "model": model,
        "predictions_en": predictions_en,
        "rmse": rmse
    }
    
    print(f"elasticNetParam={en_param}, RMSE={rmse}")
    
    if rmse < best_rmse_en:
        best_rmse_en = rmse
        best_model_en = model
        best_predictions_en = predictions_en
        best_en_param = en_param

# Evaluasi model terbaik dari looping
rmse_en = evaluator_rmse.evaluate(best_predictions_en)
r2_en = evaluator_r2.evaluate(best_predictions_en)
mae_en = evaluator_mae.evaluate(best_predictions_en)
mse_en = evaluator_mse.evaluate(best_predictions_en)

print(f"\n==== Model Terbaik Linear Regression (elasticNetParam={best_en_param}) ====")
print(f"RMSE : {rmse_en}")
print(f"R²   : {r2_en}")
print(f"MAE  : {mae_en}")
print(f"MSE  : {mse_en}")


# Convert hasil prediksi ke Pandas untuk visualisasi
sampled_pred_en = best_predictions_en.select("fare_amount", "prediction").sample(False, 0.01, seed=42).toPandas().reset_index(drop=True)


sampled_pred_en.head()


# Hiting error/residuals
residuals_lr = sampled_pred_en['fare_amount'] - sampled_pred_en['prediction']

# Plot 
plt.figure(figsize=(12, 6))
plt.plot(sampled_pred_en.index[:100], residuals_lr[:100], label='Residuals', color='Red', alpha=0.6)
plt.axhline(0, color='black', linestyle='--')  #(Garis di 0 kalau nggak ada error)
plt.xlabel("Sample Index")
plt.ylabel("Error (Nilai Asli - Prediksi)")
plt.title("Plot Error LR (Sample)")
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(12, 6))
plt.scatter(sampled_pred_en.index[:200], sampled_pred_en['fare_amount'][:200], label='Nilai Asli', color='red', alpha=0.3)
plt.scatter(sampled_pred_en.index[:200], sampled_pred_en['prediction'][:200], label='Prediksi', color='blue', alpha=0.3)
plt.xlabel("Sample Index")
plt.ylabel("Fare Amount")
plt.title(" Prediksi LR vs Nilai Asli (Sample)")
plt.legend()
plt.grid(True)
plt.show()



# Plot prediksi dan nilai asli
plt.figure(figsize=(20, 6))
plt.plot(sampled_pred_en['fare_amount'][:100], label='Nilai Asli', color='red', alpha=0.5)
plt.plot(sampled_pred_en['prediction'][:100], label='Prediksi', color='blue', alpha=0.5)
plt.xlabel("Sample Index")
plt.ylabel("Fare Amount")
plt.title(" Prediksi LR vs Nilai Asli (Sample)")
plt.legend()
plt.grid(True)
plt.show()


# Plot prediksi 2 model dan nilai asli
plt.figure(figsize=(20, 6))
plt.plot(sampled_pred_en['fare_amount'][:100], label='Nilai Asli', color='red', alpha=0.5)
plt.plot(sampled_pred_en['prediction'][:100], label='Prediksi LR', color='green', alpha=0.5)
plt.plot(sampled_pred['prediction'][:100], label='Prediksi RF', color='blue', alpha=0.5)
plt.xlabel("Sample Index")
plt.ylabel("Fare Amount")
plt.title("Prediksi vs Nilai Asli (LR dan RF)")
plt.legend()
plt.grid(True)
plt.show()


# Plot 
residuals_lr = sampled_pred_en['fare_amount'] - sampled_pred_en['prediction']
residuals = sampled_pred['fare_amount'] - sampled_pred['prediction']


plt.figure(figsize=(12, 6))
plt.plot(sampled_pred_en.index[:100], residuals_lr[:100], label='Residuals LR', color='Red', alpha=0.6)
plt.plot(sampled_pred.index[:100], residuals[:100], label='Residuals RF', color='Blue', alpha=0.6)

plt.axhline(0, color='black', linestyle='--')  #(Garis di 0 kalau nggak ada error)
plt.xlabel("Sample Index")
plt.ylabel("Error (Nilai Asli - Prediksi)")
plt.title("Plot Error (RF vs LR)")
plt.legend()
plt.grid(True)
plt.show()


# Box plot for actual vs RF and LR predictions
plt.figure(figsize=(12, 6))

data = [sampled_pred_en['fare_amount'][:100], 
        sampled_pred_en['prediction'][:100], 
        sampled_pred['prediction'][:100]]

box = plt.boxplot(data, 
                  notch=True, 
                  patch_artist=True, 
                  vert=False,  
                  flierprops=dict(markerfacecolor='white', marker='o', markersize=5, alpha=0),
                  medianprops=dict(color='black'))  

colors = ['red', 'green', 'blue'] 

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.yticks([1, 2, 3], ['Nilai Asli', 'Prediksi LR', 'Prediksi RF'])

plt.xlabel("Fare Amount")
plt.title("Boxplot Sampel Nilai Asli vs RF vs LR")
plt.grid(True)
plt.show()




