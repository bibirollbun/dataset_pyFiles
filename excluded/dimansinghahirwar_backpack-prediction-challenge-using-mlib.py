


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when, count,lit
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px
import pandas as pd


spark = SparkSession.builder \
    .appName("Kaggle Backpack Price Prediction") \
    .getOrCreate()


train_df = spark.read.csv('/kaggle/input/playground-series-s5e2/train.csv', header=True, inferSchema=True)
train_extra_df = spark.read.csv('/kaggle/input/playground-series-s5e2/training_extra.csv', header=True, inferSchema=True)
test_df = spark.read.csv('/kaggle/input/playground-series-s5e2/test.csv', header=True, inferSchema=True)


# Combine train datasets
train_df = train_df.union(train_extra_df)

# Add missing column "Price" to test_df
test_df = test_df.withColumn("Price", lit(None).cast("double"))

# Handling Missing Values
for col_name in train_df.columns:
    mean_value = train_df.select(mean(col(col_name))).collect()[0][0]
    if mean_value is not None:  # Ensure the mean is not None before filling
        train_df = train_df.fillna({col_name: mean_value})

test_df = test_df.fillna({col_name: mean_value for col_name in test_df.columns if col_name != "id"})

# Handling Outliers (Using IQR method)
numeric_cols = ["Compartments", "Weight Capacity (kg)", "Price"]
for col_name in numeric_cols:
    quantiles = train_df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    IQR = quantiles[1] - quantiles[0]
    lower_bound, upper_bound = quantiles[0] - 1.5 * IQR, quantiles[1] + 1.5 * IQR
    train_df = train_df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))

# Encoding Categorical Features
categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
all_data = train_df.unionByName(test_df, allowMissingColumns=True)  # Combine train and test for consistent indexing

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep").fit(all_data) for col in categorical_cols]
for indexer in indexers:
    train_df = indexer.transform(train_df)
    test_df = indexer.transform(test_df)

# Ensure no nulls in categorical indexes
for col_name in [col+"_index" for col in categorical_cols]:
    train_df = train_df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))
    test_df = test_df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))



# Feature Engineering
feature_cols = ["Compartments", "Weight Capacity (kg)"] + [col+"_index" for col in categorical_cols]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
train_df = vector_assembler.transform(train_df)
test_df = vector_assembler.transform(test_df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(train_df)
train_df = scaler_model.transform(train_df)
test_df = scaler_model.transform(test_df)

# Split into train and validation sets
train_data, val_data = train_df.randomSplit([0.8, 0.2], seed=42)

# Train Models
models = {
    "LinearRegression": LinearRegression(featuresCol="scaled_features", labelCol="Price"),
    "RandomForest": RandomForestRegressor(featuresCol="scaled_features", labelCol="Price"),
    "GBTRegressor": GBTRegressor(featuresCol="scaled_features", labelCol="Price")
}

evaluator = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")

best_model = None
best_rmse = float("inf")


for name, model in models.items():
    model_fit = model.fit(train_data)
    predictions = model_fit.transform(val_data)
    rmse = evaluator.evaluate(predictions)
    print(f"{name} RMSE: {rmse}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model_fit

# Make Predictions on Test Set
test_predictions = best_model.transform(test_df).select("id", "prediction")
test_predictions = test_predictions.withColumnRenamed("prediction", "Price")

# Convert to Pandas and Save for Submission
test_predictions_pd = test_predictions.toPandas()
test_predictions_pd.to_csv("submission.csv", index=False)


# Visualization using Plotly
train_pd = train_df.select("Price", "Compartments", "Weight Capacity (kg)").toPandas()
fig = px.scatter(train_pd, x="Weight Capacity (kg)", y="Price", color="Compartments")
fig.show()

