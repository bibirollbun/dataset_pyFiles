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


import zipfile
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import when, col

# Initialize Spark session
spark = SparkSession.builder.master("local[2]").appName("Airbnb-ML").getOrCreate()

# Helper function to extract ZIP file
def extract_zip(zip_file_path, extract_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        return zip_ref.namelist()  # Return the list of files extracted

# Paths to ZIP files
zip_paths = {
    'countries': '/kaggle/input/airbnb-recruiting-new-user-bookings/countries.csv.zip',
    'age': '/kaggle/input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv.zip',
    'sessions': '/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip',
    'test': '/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip',
    'train': '/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip'
}

# Folder to extract the ZIP files
extract_folder = '/path/to/extracted_files'

# Extract and read data
df_countries = spark.read.csv(os.path.join(extract_folder, extract_zip(zip_paths['countries'], extract_folder)[0]), header=True, inferSchema=True)
df_age = spark.read.csv(os.path.join(extract_folder, extract_zip(zip_paths['age'], extract_folder)[0]), header=True, inferSchema=True)
df_sessions = spark.read.csv(os.path.join(extract_folder, extract_zip(zip_paths['sessions'], extract_folder)[0]), header=True, inferSchema=True)
df_test = spark.read.csv(os.path.join(extract_folder, extract_zip(zip_paths['test'], extract_folder)[0]), header=True, inferSchema=True)
df_train = spark.read.option("encoding", "ISO-8859-1").csv(os.path.join(extract_folder, extract_zip(zip_paths['train'], extract_folder)[0]), header=True, inferSchema=True)

# Exploratory Data Analysis (EDA)
df_train.printSchema()
df_test.printSchema()

# Data Preprocessing
# I. Handling unknown values
df_train = df_train.withColumn("gender", 
                               when(col("gender") == "-unknown-", None).otherwise(col("gender")))
df_train = df_train.withColumn("first_browser", 
                               when(col("first_browser") == "-unknown-", None).otherwise(col("first_browser")))

# II. Handling Null Values
df_train = df_train.filter(~(col('age').isNull() & (col('country_destination') == 'NDF')))

# III. Feature Engineering
df_train = df_train.filter((col('age') >= 18) & (col('age') <= 120))

# IV. Handling Categorical Features
indexer_gender = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="skip")
indexer_browser = StringIndexer(inputCol="first_browser", outputCol="browser_index", handleInvalid="skip")
indexer_country = StringIndexer(inputCol="country_destination", outputCol="country_destination_index", handleInvalid="skip")

# Assemble features into a feature vector
assembler = VectorAssembler(inputCols=["age", "gender_index", "browser_index"], outputCol="features")

# Train-Test Split
train_data, test_data = df_train.randomSplit([0.8, 0.2], seed=1234)

# Model Building: Random Forest Classifier
rf = RandomForestClassifier(labelCol="country_destination_index", featuresCol="features", numTrees=100, maxBins=50)

# Pipeline for training
pipeline_training = Pipeline(stages=[indexer_gender, indexer_browser, indexer_country, assembler, rf])

# Fit the model on training data
model_train = pipeline_training.fit(train_data)


pipeline_prediction = Pipeline(stages=[indexer_gender, indexer_browser, assembler])
df_test_prepared = pipeline_prediction.fit(df_test).transform(df_test)

# Apply the trained RandomForest model directly
predictions_test = model_train.stages[-1].transform(df_test_prepared)
print("predictions_test",predictions_test)
# Display Predictions
#predictions_test.select("id", "prediction").show(5)

# Model Evaluation on Training-Test Split
predictions = model_train.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="country_destination_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Pipeline for prediction (without indexer_country)
#pipeline_prediction = Pipeline(stages=[indexer_gender, indexer_browser, assembler, rf])
#model_prediction = pipeline_prediction.fit(train_data)

# Make Predictions on Test Data
#predictions_test = model_prediction.transform(df_test)

# Display Predictions
#predictions_test.select("id", "prediction").show(5)

# Model Evaluation on Training Test Split
#predictions = model_train.transform(test_data)
#evaluator = MulticlassClassificationEvaluator(labelCol="country_destination_index", predictionCol="prediction", metricName="accuracy")
#accuracy = evaluator.evaluate(predictions)
#print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Feature Importances (Optional)
rf_model = model_train.stages[-1]  # RandomForest model
importances = rf_model.featureImportances
print("Feature Importances: ", importances)

# Save the model (Optional)
model_train.save("/path/to/save/random_forest_model")





