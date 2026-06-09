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


%pip install pyspark
%pip install matplotlib
%pip install numpy


# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Home Credit Default Risk") \
    .getOrCreate()

# Define data path (use the correct Kaggle input path)
data_path = "/kaggle/input/home-credit-default-risk/application_train.csv"

# Read data
try:
    data = spark.read.csv(data_path, header=True, inferSchema=True)
except Exception as e:
    print(f"Error reading file: {e}")
    spark.stop()
    raise

# Split data into train and test set
train_data, temp_data = data.randomSplit([0.7, 0.3], seed=42)
test_data , validation_data = temp_data.randomSplit([0.2,0.1] , seed=42)


# Define categorical columns
categorical_columns = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

# Impute missing categorical values
for col_name in categorical_columns:
    train_data = train_data.withColumn(
        col_name, when(col(col_name).isNull(), "Unknown").otherwise(col(col_name))
    )


def barplot(df, col_name):
    # Group by the column and TARGET, count occurrences
    plot_data = (df
                 .groupBy(col_name, "TARGET")
                 .agg(count("*").alias("count"))
                 .orderBy(col_name, "TARGET"))

    # Pivot to get counts for TARGET=0 and TARGET=1 as columns
    pivot_data = (plot_data
                  .groupBy(col_name)
                  .pivot("TARGET")
                  .sum("count")
                  .na.fill(0))

    # Rename columns for clarity (0 for not defaulted, 1 for defaulted)
    pivot_data = pivot_data.withColumnRenamed("0", "not_defaulted").withColumnRenamed("1", "defaulted")

    # Add a total count column for sorting
    pivot_data = pivot_data.withColumn("total_count", pivot_data["not_defaulted"] + pivot_data["defaulted"])

    # Sort by total_count in descending order
    pivot_data = pivot_data.orderBy(pivot_data["total_count"].desc())

    # Collect data to driver for plotting
    collected_data = pivot_data.collect()

    # Extract categories and counts
    categories = [row[col_name] for row in collected_data]
    not_defaulted_counts = [row["not_defaulted"] for row in collected_data]
    defaulted_counts = [row["defaulted"] for row in collected_data]

    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the not_defaulted bars as the bottom layer
    ax.bar(categories, not_defaulted_counts, label='Not Defaulted (0)', color='skyblue')

    # Plot the defaulted bars stacked on top of the not_defaulted bars
    ax.bar(categories, defaulted_counts, bottom=not_defaulted_counts, label='Defaulted (1)', color='salmon')

    # Customize plot
    ax.set_xlabel(col_name)
    ax.set_ylabel('Count')
    ax.set_title(f'Default Status by {col_name} (Sorted by Total Count)')
    plt.xticks(rotation=45, ha='right')
    ax.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()





for col in categorical_columns:
    barplot(train_data, col)


# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import Row, DoubleType, FloatType, IntegerType, LongType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Home Credit Default Risk") \
    .getOrCreate()

# Define data path
data_path = "/kaggle/input/home-credit-default-risk"

# Read data
data = spark.read.csv(f"{data_path}/application_train.csv", header=True, inferSchema=True)

# Check for duplicate SK_ID_CURR
count = data.groupBy("SK_ID_CURR").count().filter(col("count") > 1).count()
if count > 0:
    raise ValueError(f"Duplicate SK_ID_CURR found: {count} duplicates")
else:
    print("All SK_ID_CURR values are unique.")

# Split data into train and test set
train_data, validation_data ,  test_data = data.randomSplit([0.7, 0.2 , 0.1], seed=42)

# Display dataset size
print('Training data points:', train_data.count())
print('Number of features:', len(train_data.columns))
print('Features:', train_data.columns)

# MapReduce for missing value imputation
# Computes mean for all numerical columns to impute missing values
numerical_types = [DoubleType, FloatType, IntegerType, LongType]
numerical_columns = [field.name for field in data.schema.fields if isinstance(field.dataType, tuple(numerical_types))]

# Custom map function: Emits (column_name, (value, 1)) for non-null values
def map_function(row):
    result = []
    for col_name in numerical_columns:
        value = row[col_name]
        if value is not None:
            result.append((col_name, (float(value), 1)))
    return result

# Custom reduce function: Aggregates (sum, count) for each column
def reduce_function(value1, value2):
    return (value1[0] + value2[0], value1[1] + value2[1])

# Execute MapReduce to compute mean for each numerical column in train and test data
mean_rdd = (train_data.rdd
            .flatMap(map_function)
            .reduceByKey(reduce_function)
            .mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0.0)
            .coalesce(1))
mean_dict = dict(mean_rdd.collect())

mean_rdd_test = (test_data.rdd
            .flatMap(map_function)
            .reduceByKey(reduce_function)
            .mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0.0)
            .coalesce(1))
mean_dict_test = dict(mean_rdd_test.collect())

mean_rdd_validation =  (validation_data.rdd
            .flatMap(map_function)
            .reduceByKey(reduce_function)
            .mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0.0)
            .coalesce(1))
mean_dict_validation = dict(mean_rdd_validation.collect())

def impute_row(row, mean_dict):
    row_dict = row.asDict()
    for col_name in numerical_columns:
        if row_dict[col_name] is None:
            row_dict[col_name] = mean_dict.get(col_name)
    return Row(**row_dict)

imputed_train_data = spark.createDataFrame(train_data.rdd.map(lambda row: impute_row(row, mean_dict)), schema=train_data.schema)
imputed_test_data = spark.createDataFrame(test_data.rdd.map(lambda row: impute_row(row, mean_dict_test)), schema=data.schema)
imputed_validation_data = spark.createDataFrame(train_data.rdd.map(lambda row: impute_row(row, mean_dict_validation)), schema=data.schema)
# Feature selection based on EDA
selected_features = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
]

categorical_columns = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

# Impute missing categorical values
for col_name in categorical_columns:
    imputed_train_data = imputed_train_data.withColumn(
        col_name, when(col(col_name).isNull(), "Unknown").otherwise(col(col_name))
    )
    imputed_test_data = imputed_test_data.withColumn(
        col_name, when(col(col_name).isNull(), "Unknown").otherwise(col(col_name))
    )
    imputed_validation_data = imputed_validation_data.withColumn(
         col_name, when(col(col_name).isNull(), "Unknown").otherwise(col(col_name))
    )

# Preprocessing pipeline
categorical_columns = [col for col in selected_features if data.schema[col].dataType == StringType()]
numerical_columns_selected = [col for col in selected_features if col in numerical_columns]

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in categorical_columns]
encoders = [OneHotEncoder(inputCols=[f"{col}_index"], outputCols=[f"{col}_encoded"]) for col in categorical_columns]
assembler = VectorAssembler(inputCols=numerical_columns_selected + [f"{col}_encoded" for col in categorical_columns], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="TARGET")

# Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, lr])

# Train model
model = pipeline.fit(imputed_train_data)

# Evaluate model
evaluator_acc = MulticlassClassificationEvaluator(labelCol="TARGET", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="TARGET", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

train_pred = model.transform(imputed_train_data)
test_pred = model.transform(imputed_test_data)
validate_pred = model.transform(imputed_validation_data)

print("Train Accuracy:", evaluator_acc.evaluate(train_pred))
print("Train AUC:", evaluator_auc.evaluate(train_pred))
print("Test Accuracy:", evaluator_acc.evaluate(test_pred))
print("Test AUC:", evaluator_auc.evaluate(test_pred))
print("Validation AUC:", evaluator_auc.evaluate(validate_pred))
print("Validation Accuracy :", evaluator_acc.evaluate(validate_pred))

spark.stop()

