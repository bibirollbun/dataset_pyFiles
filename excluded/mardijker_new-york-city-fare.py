from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os


# Initialize SparkSession
spark = (
    SparkSession.builder
      .appName("NYC Taxi Fare Prediction")
      .config("spark.sql.shuffle.partitions", "100")   # adjust for your cluster
      .config("spark.driver.memory", "4g")
      .getOrCreate()
)

# For consistent train/val splits
RANDOM_SEED = 42


train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"

df_train = spark.read.option("header", True).option("inferSchema", True).csv(train_path)

print("** SCHEMA **")
df_train.printSchema()
print("** ROW COUNT **", df_train.count())

df_train.show(5, truncate=False)


# Count nulls per column in train
null_counts = (
    df_train.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_train.columns])
).toPandas().T
null_counts.columns = ["null_count"]
display(null_counts)


df_train.select(
    F.mean("fare_amount").alias("mean_fare"),
    F.expr("percentile(fare_amount, array(0.5))")[0].alias("median_fare"),
    F.min("fare_amount").alias("min_fare"),
    F.max("fare_amount").alias("max_fare")
).show()

df_train.groupBy("passenger_count").count().orderBy("passenger_count").show()


sample_pd = df_train.sample(fraction=0.02, seed=RANDOM_SEED).limit(100_000).toPandas()

plt.figure(figsize=(6,4))
sns.boxplot(x=sample_pd["passenger_count"])
plt.title("Boxplot of passenger_count (detecting outliers)")
plt.xlabel("Number of Passengers")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=sample_pd["fare_amount"])
plt.title("Boxplot of fare_amount (detecting outliers)")
plt.xlabel("Fare Amount ($)")
plt.show()


print("Initial train count:", df_train.count())

df = df_train.dropna(subset=[
    "fare_amount", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
])
print("After dropna:", df.count())


print("Before filtering outlier in fare:", df.count())
df = df.filter((F.col("fare_amount") > 0) & (F.col("fare_amount") < 500))
print("After filtering outlier in fare:", df.count())

print("Before filtering outlier in passenger_count:", df.count())
df = df.filter((F.col("passenger_count") >= 1) & (F.col("passenger_count") <= 6))
print("After filtering outlier in passenger_count:", df.count())


print("Before filtering coordinates outside New York:", df.count())
df = df.filter(
    (F.col("pickup_latitude").between(40.0, 42.0)) &
    (F.col("dropoff_latitude").between(40.0, 42.0)) &
    (F.col("pickup_longitude").between(-74, -72)) &
    (F.col("dropoff_longitude").between(-74, -72))
)
print("After filtering coordinates outside New York:", df.count())


df = df.withColumn(
    "pickup_datetime_ts", 
    F.to_timestamp("pickup_datetime", "yyyy-MM-dd HH:mm:ss.SSSSSSS")
)
df = df.withColumn("hour", F.hour("pickup_datetime_ts")) \
       .withColumn("day_of_week", F.dayofweek("pickup_datetime_ts"))

print("After datetime parsing:", df.count())


import math
def haversine_spark(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

haversine_udf = F.udf(haversine_spark, DoubleType())

df = df.withColumn(
    "distance_km",
    haversine_udf(
      F.col("pickup_longitude"), F.col("pickup_latitude"),
      F.col("dropoff_longitude"), F.col("dropoff_latitude")
    )
).filter(F.col("distance_km") > 0)  # remove zero-distance
print("After adding distance:", df.count())


df.show(5, truncate=False)


feature_cols = ["distance_km", "passenger_count", "hour", "day_of_week"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")


scaler = StandardScaler(
    inputCol="raw_features", 
    outputCol="features", 
    withMean=True, 
    withStd=True
)


base_pipeline_stages = [assembler, scaler]


train_df, val_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
print("Train set count:", train_df.count(), " Val set count:", val_df.count())


evaluator = RegressionEvaluator(
    labelCol="fare_amount", predictionCol="prediction", metricName="rmse"
)


lr = LinearRegression(featuresCol="features", labelCol="fare_amount", maxIter=50, regParam=0.1)
lr_pipeline = Pipeline(stages=base_pipeline_stages + [lr])
lr_model = lr_pipeline.fit(train_df)

lr_preds = lr_model.transform(val_df)

lr_preds.select(
    "fare_amount", 
    F.round("prediction", 2).alias("predicted_fare"),
    "distance_km", "passenger_count", "hour", "day_of_week"
).show(5)


lr_rmse = evaluator.evaluate(lr_preds)
print(f"Linear Regression Validation RMSE: {lr_rmse:.4f}")


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (
    ParamGridBuilder()
      .addGrid(lr.regParam, [0.0, 0.01, 0.1, 1.0])
      .build()
)

cv = CrossValidator(
    estimator=Pipeline(stages=base_pipeline_stages + [lr]),
    evaluator=evaluator,
    estimatorParamMaps=paramGrid,
    numFolds=3,
    seed=RANDOM_SEED
)

cv_model = cv.fit(train_df)

best_lr_model = cv_model.bestModel
best_regParam = float(best_lr_model.stages[-1]._java_obj.getRegParam())
print(f"Best regParam found: {best_regParam}")


tuned_preds = best_lr_model.transform(val_df)
tuned_rmse = evaluator.evaluate(tuned_preds)
print(f"Tuned Linear Regression RMSE = {tuned_rmse:.4f}")

tuned_preds.select(
    "fare_amount",
    F.round("prediction", 2).alias("predicted_fare"),
    "distance_km", "passenger_count", "hour", "day_of_week"
).show(10)

