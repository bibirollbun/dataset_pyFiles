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

# Split data into train, validation, and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Display dataset size
print('Training data points:', train_data.count())
print('Number of features:', len(train_data.columns))
print('Features:', train_data.columns)

# # Plot target distribution
target_counts = train_data.groupBy("TARGET").count().collect()
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
recipe = ["Will Repay", "Will not Repay"]
data_counts = [row['count'] for row in sorted(target_counts, key=lambda x: x['TARGET'])]
wedges, texts = ax.pie(data_counts, wedgeprops=dict(width=0.5), startangle=-40)
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)
ax.set_title("Loan Repayment Distribution")

# MapReduce for missing value imputation
# Computes mean for all numerical columns to impute missing values
numerical_types = [DoubleType, FloatType, IntegerType, LongType]
all_numerical_columns = [field.name for field in data.schema.fields if isinstance(field.dataType, tuple(numerical_types))]
rdd = train_data.rdd
rdd.persist() # Persist RDD for performance

# Custom map function: Emits (column_name, (value, 1)) for non-null values
def map_function(row):
    result = []
    for col_name in all_numerical_columns:
        value = row[col_name]
        if value is not None:
            result.append((col_name, (float(value), 1)))
    return result

# Custom reduce function: Aggregates (sum, count) for each column
def reduce_function(value1, value2):
    return (value1[0] + value2[0], value1[1] + value2[1])

# Execute MapReduce with retries

mean_rdd = (rdd
            .flatMap(map_function)
            .reduceByKey(reduce_function)
            .mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0.0)
            .coalesce(1))  # Single partition for collect
mean_dict = dict(mean_rdd.collect())

rdd.unpersist()  # Release persisted RDD

# Apply imputation to numerical columns
def impute_row(row):
    row_dict = row.asDict()
    for col_name in all_numerical_columns:
        if row_dict[col_name] is None:
            row_dict[col_name] = mean_dict.get(col_name, 0.0)
    return Row(**row_dict)

imputed_rdd = train_data.rdd.map(impute_row)
imputed_train_data = spark.createDataFrame(imputed_rdd, schema=train_data.schema)

# Apply imputation to test set
imputed_test_data = spark.createDataFrame(test_data.rdd.map(impute_row), schema=data.schema)

# Impute missing categorical values
categorical_columns = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for col_name in categorical_columns:
    imputed_train_data = imputed_train_data.withColumn(
        col_name, when(col(col_name).isNull(), "Unknown").otherwise(col(col_name))
    )
    imputed_test_data = imputed_test_data.withColumn(
        col_name, when(col(col_name).isNull(), "Unknown").otherwise(col(col_name))
    )

# Feature selection based on EDA
selected_features = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
]

# Preprocessing pipeline
categorical_columns = [col for col in selected_features if data.schema[col].dataType == StringType()]
numerical_columns_selected = [col for col in selected_features if col in all_numerical_columns]

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

print("Train Accuracy:", evaluator_acc.evaluate(train_pred))
print("Train AUC:", evaluator_auc.evaluate(train_pred))
print("Test Accuracy:", evaluator_acc.evaluate(test_pred))
print("Test AUC:", evaluator_auc.evaluate(test_pred))

spark.stop()


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
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Display dataset size
print('Training data points:', train_data.count())
print('Number of features:', len(train_data.columns))
print('Features:', train_data.columns)

# MapReduce for missing value imputation
# Computes mean for all numerical columns to impute missing values
numerical_types = [DoubleType, FloatType, IntegerType, LongType]
numerical_columns = [field.name for field in data.schema.fields if isinstance(field.dataType, tuple(numerical_types))]
rdd = train_data.rdd
rdd.persist() # Persist RDD for performance

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

# Execute MapReduce with retries

mean_rdd = (rdd
            .flatMap(map_function)
            .reduceByKey(reduce_function)
            .mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0.0)
            .coalesce(1))  # Single partition for collect
mean_dict = dict(mean_rdd.collect())

rdd.unpersist()  # Release persisted RDD

# Apply imputation to numerical columns
def impute_row(row):
    row_dict = row.asDict()
    for col_name in numerical_columns:
        if row_dict[col_name] is None:
            row_dict[col_name] = mean_dict.get(col_name, 0.0)
    return Row(**row_dict)

imputed_rdd = train_data.rdd.map(impute_row)
imputed_train_data = spark.createDataFrame(imputed_rdd, schema=train_data.schema)

# Apply imputation to test set
imputed_test_data = spark.createDataFrame(test_data.rdd.map(impute_row), schema=data.schema)

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

# Preprocessing pipeline
categorical_columns = [col for col in selected_features if data.schema[col].dataType == StringType()]
numerical_columns_selected = [col for col in selected_features if col in all_numerical_columns]

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

print("Train Accuracy:", evaluator_acc.evaluate(train_pred))
print("Train AUC:", evaluator_auc.evaluate(train_pred))
print("Test Accuracy:", evaluator_acc.evaluate(test_pred))
print("Test AUC:", evaluator_auc.evaluate(test_pred))

spark.stop()

