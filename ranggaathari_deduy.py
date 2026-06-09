!sudo apt update
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
#Check this site for the latest download link https://www.apache.org/dyn/closer.lua/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
!wget -q https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
!tar xf spark-3.2.1-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install pyspark
!pip install py4j

import os
import sys
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"


import findspark
findspark.init()
findspark.find()

import pyspark

from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F

spark= SparkSession \
       .builder \
       .appName("Our First Spark Example") \
       .getOrCreate()

spark


csv_file_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"

df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(csv_file_path))


df.printSchema()


df.describe(["fare_amount", "pickup_longitude", "pickup_latitude",
             "dropoff_longitude", "dropoff_latitude", "passenger_count"]).show()


print("Jumlah baris:", df.count())
print("Jumlah kolom:", len(df.columns))
print("Nama kolom:", df.columns)

