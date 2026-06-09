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


!pip install -q pyspark==3.5.1 pyarrow==15.0.2


from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("nyc-taxi-fare")
         .master("local[*]")                    
         .config("spark.driver.memory", "12g")  
         .config("spark.sql.shuffle.partitions", "200") 
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())

spark


INPUT_DIR  = "/kaggle/input/new-york-city-taxi-fare-prediction"
WORK_DIR   = "/kaggle/working"  # untuk output/cache kamu

TRAIN_CSV  = f"{INPUT_DIR}/train.csv"
TEST_CSV   = f"{INPUT_DIR}/test.csv"
SAMPLE_SUB = f"{INPUT_DIR}/sample_submission.csv"


from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

schema_train = StructType([
    StructField("key", StringType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("pickup_datetime", TimestampType(), True),
    StructField("pickup_longitude", DoubleType(), True),
    StructField("pickup_latitude", DoubleType(), True),
    StructField("dropoff_longitude", DoubleType(), True),
    StructField("dropoff_latitude", DoubleType(), True),
    StructField("passenger_count", IntegerType(), True),
])

schema_test = StructType([
    StructField("key", StringType(), True),
    StructField("pickup_datetime", TimestampType(), True),
    StructField("pickup_longitude", DoubleType(), True),
    StructField("pickup_latitude", DoubleType(), True),
    StructField("dropoff_longitude", DoubleType(), True),
    StructField("dropoff_latitude", DoubleType(), True),
    StructField("passenger_count", IntegerType(), True),
])



# — Load TRAIN
df_train_csv = (spark.read
    .option("header", True)
    .schema(schema_train)
    .csv(TRAIN_CSV))

# — Load TEST
df_test_csv = (spark.read
    .option("header", True)
    .schema(schema_test)
    .csv(TEST_CSV))

# Simpan sebagai Parquet (sekali saja; berikutnya baca dari Parquet)
df_train_csv.write.mode("overwrite").parquet(f"{WORK_DIR}/train_parquet")
df_test_csv.write.mode("overwrite").parquet(f"{WORK_DIR}/test_parquet")

# Baca kembali sebagai DataFrame Parquet (cepat)
train = spark.read.parquet(f"{WORK_DIR}/train_parquet")
test  = spark.read.parquet(f"{WORK_DIR}/test_parquet")



print("TRAIN schema:")
train.printSchema()

print("TEST schema:")
test.printSchema()

print("TRAIN count (butuh waktu, sabar):", train.count())
print("TEST  count:", test.count())

train.show(5, truncate=False)
test.show(5, truncate=False)



from pyspark.sql import functions as F

# Soft checks (tidak menghapus dulu—sekadar lihat ada apa)
summary_basic = (train
    .select(
        F.count("*").alias("rows"),
        F.count("fare_amount").alias("fare_not_null"),
        F.count(F.when(F.col("fare_amount") < 0, 1)).alias("fare_negative"),
        F.count(F.when((F.col("passenger_count") < 1) | (F.col("passenger_count") > 8), 1)).alias("passenger_out_of_range"),
    )
).collect()[0]

summary_basic.asDict()


# Jumlah baris dan kolom
print(f"Jumlah baris: {train.count():,}")
print(f"Jumlah kolom: {len(train.columns)}")
print("Kolom:", train.columns)

# Tipe data
train.printSchema()


numeric_cols = ["fare_amount", "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude", "passenger_count"]

train.select(numeric_cols).describe().show()



(train
 .select(
     F.mean("fare_amount").alias("mean"),
     F.expr("percentile_approx(fare_amount, 0.5)").alias("median"),
     F.min("fare_amount").alias("min"),
     F.max("fare_amount").alias("max"))
).show()



(train.groupBy("passenger_count")
      .agg(F.count("*").alias("jumlah"))
      .orderBy("passenger_count")
      .show())



train.select(
    F.min("pickup_longitude").alias("min_pickup_long"),
    F.max("pickup_longitude").alias("max_pickup_long"),
    F.min("pickup_latitude").alias("min_pickup_lat"),
    F.max("pickup_latitude").alias("max_pickup_lat"),
    F.min("dropoff_longitude").alias("min_dropoff_long"),
    F.max("dropoff_longitude").alias("max_dropoff_long"),
    F.min("dropoff_latitude").alias("min_dropoff_lat"),
    F.max("dropoff_latitude").alias("max_dropoff_lat")
).show()



train = train.withColumn("year", F.year("pickup_datetime")) \
             .withColumn("month", F.month("pickup_datetime")) \
             .withColumn("hour", F.hour("pickup_datetime")) \
             .withColumn("dayofweek", F.date_format("pickup_datetime", "E"))

(train.groupBy("year")
      .agg(F.count("*").alias("jumlah"))
      .orderBy("year")
      .show())



from pyspark.sql.functions import radians, sin, cos, asin, sqrt

def haversine(df):
    R = 6371  # radius bumi (km)
    return df.withColumn(
        "distance_km",
        2 * R * asin(
            sqrt(
                sin((radians(F.col("dropoff_latitude") - F.col("pickup_latitude")) / 2))**2 +
                cos(radians(F.col("pickup_latitude"))) *
                cos(radians(F.col("dropoff_latitude"))) *
                sin((radians(F.col("dropoff_longitude") - F.col("pickup_longitude")) / 2))**2
            )
        )
    )

train = haversine(train)

train.select(F.corr("fare_amount", "distance_km").alias("corr_fare_distance")).show()


