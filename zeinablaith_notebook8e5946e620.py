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


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # سنحتفظ بها لعدم كسر أي اعتماديات، لكن لن نستخدمها للرسوم البيانية
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tabulate import tabulate # مكتبة جديدة لطباعة الجداول الجميلة (قد تحتاج لتثبيتها: pip install tabulate)

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import ClusteringEvaluator

# ================================================================
# 1) إعدادات الملف وميزات البيانات (IEEE Fraud Detection Dataset)
# ================================================================
FILE_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
FEATURES = ['TransactionAmt', 'TransactionDT', 'D15', 'C13'] 

print("Ready using IEEE Fraud Detection Dataset (~1.4 GB)!")

# ================================================================
# 2) تطبيق Python/Pandas (قياس الزمن الكلي)
# ================================================================
print("\n--- Starting Python (Pandas) Implementation ---")
start_time_pandas = time.time()
pandas_duration = 0 

try:
    df_pandas = pd.read_csv(FILE_PATH, usecols=FEATURES) 
    df_pandas = df_pandas.dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_pandas)

    kmeans_pandas = KMeans(n_clusters=4, random_state=42, n_init='auto')
    kmeans_pandas.fit(scaled_features)

    end_time_pandas = time.time()
    pandas_duration = end_time_pandas - start_time_pandas

    print(f"Python (Pandas) Time (Full Dataset): {pandas_duration:.2f} seconds")

except Exception as e:
    print(f"Python (Pandas) Crashed or Failed!: {e}")
    pandas_duration = 0

# ================================================================
# 3) تطبيق PySpark (المُحسَّن)
# ================================================================
print("\n--- Starting PySpark (Optimized Mode) ---")

spark = SparkSession.builder \
    .appName("IEEEFraud_Clustering_Optimized") \
    .config("spark.driver.memory", "24g") \
    .config("spark.executor.memory", "24g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
start_time_spark_total = time.time()

# --- مرحلة القراءة والتحضير (ETL) ---
df_spark = spark.read.csv(FILE_PATH, header=True, inferSchema=False)

for column in FEATURES:
    df_spark = df_spark.withColumn(column, col(column).cast(DoubleType()))

df_spark = df_spark.select(FEATURES).na.drop() 

assembler = VectorAssembler(inputCols=FEATURES, outputCol="features_vec")
df_vec = assembler.transform(df_spark)

scaler_spark = SparkScaler(inputCol="features_vec", outputCol="features", withStd=True, withMean=False)
scalerModel = scaler_spark.fit(df_vec)
df_scaled = scalerModel.transform(df_vec) 

end_time_spark_prep = time.time()
spark_prep_time = end_time_spark_prep - start_time_spark_total
print(f"PySpark Preparation Time: {spark_prep_time:.2f} seconds")

# --- مرحلة التدريب (Modeling) ---
kmeans_spark = SparkKMeans(k=4, seed=42, featuresCol="features") 
model_spark = kmeans_spark.fit(df_scaled) 
centers = model_spark.clusterCenters()

end_time_spark_training = time.time()
spark_training_time = end_time_spark_training - end_time_spark_prep
spark_total_time = end_time_spark_training - start_time_spark_total

print(f"PySpark Training Time: {spark_training_time:.2f} seconds")
print(f"PySpark Total Time: {spark_total_time:.2f} seconds")

# ================================================================
# 4) تقييم الجودة (Silhouette Score)
# ================================================================
print("\n--- Calculating Quality Metrics ---")
# 1. تقييم PySpark
predictions_spark = model_spark.transform(df_scaled)
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', metricName='silhouette')
silhouette_spark = evaluator.evaluate(predictions_spark)
cost_spark = model_spark.summary.trainingCost

# 2. تقييم Python (Pandas)
silhouette_pandas = 0
inertia_pandas = 0

if pandas_duration > 0: 
    sample_size = 50000 
    if len(df_pandas) > sample_size:
        df_sample = df_pandas.sample(n=sample_size, random_state=42)
        data_sample = scaler.transform(df_sample)
        labels_sample = kmeans_pandas.predict(data_sample)
        silhouette_pandas = silhouette_score(data_sample, labels_sample)
    else:
        silhouette_pandas = silhouette_score(scaled_features, kmeans_pandas.labels_)

    inertia_pandas = kmeans_pandas.inertia_

# ================================================================
# 5) عرض توزيع التجمعات في PySpark (باستخدام جدول)
# ================================================================
print("\n--- PySpark Cluster Distribution Table ---")

cluster_counts = predictions_spark.groupBy('prediction').count().sort('prediction')
counts_pd = cluster_counts.toPandas()

# إعداد جدول توزيع التجمعات
counts_table = counts_pd.rename(columns={'prediction': 'Cluster ID', 'count': 'Number of Points'})
print(tabulate(counts_table, headers='keys', tablefmt='fancy_grid', showindex=False))


# ================================================================
# 6) جدول مقارنة الأداء (السرعة)
# ================================================================
print("\n--- Performance Comparison (Time in Seconds) ---")

performance_data = [
    ["Python (Pandas) Total", f"{pandas_duration:.2f}"],
    ["PySpark Preparation (ETL)", f"{spark_prep_time:.2f}"],
    ["PySpark Training (Modeling)", f"{spark_training_time:.2f}"],
    ["PySpark Total", f"{spark_total_time:.2f}"],
]

headers = ["Method", "Time (Seconds)"]
print(tabulate(performance_data, headers=headers, tablefmt="fancy_grid"))

# ================================================================
# 7) جدول مقارنة الجودة (Silhouette Score)
# ================================================================
print("\n--- Quality Comparison (Silhouette Score) ---")

quality_data = [
    ["Python (Pandas) - Sample", f"{silhouette_pandas:.4f}", f"{inertia_pandas:.0f}"],
    ["PySpark - Full Data", f"{silhouette_spark:.4f}", f"{cost_spark:.0f}"],
]

headers = ["Method", "Silhouette Score (Higher is Better)", "Cost/Inertia (Lower is Better)"]
print(tabulate(quality_data, headers=headers, tablefmt="fancy_grid"))


# إنهاء جلسة Spark
spark.stop()
print("\nSpark session stopped.")

