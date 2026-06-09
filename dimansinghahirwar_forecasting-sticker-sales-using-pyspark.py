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


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofweek, lag, avg, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px
import pandas as pd


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofweek, lag, avg, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder.appName("Sticker Sales Forecasting").getOrCreate()

# Load Data
train_path = "/kaggle/input/playground-series-s5e1/train.csv"
test_path = "/kaggle/input/playground-series-s5e1/test.csv"
train_df = spark.read.csv(train_path, header=True, inferSchema=True)
test_df = spark.read.csv(test_path, header=True, inferSchema=True)

# Handle Missing Values in num_sold
train_df = train_df.withColumn("num_sold", col("num_sold").cast("double"))
train_df = train_df.na.fill({"num_sold": 0.0})

# Feature Engineering for Training Data
train_df = train_df.withColumn("year", year(col("date")))\
                   .withColumn("month", month(col("date")))\
                   .withColumn("dayofweek", dayofweek(col("date")))

# Prepare Features for Training Data
feature_cols = ["year", "month", "dayofweek"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df)

# Train Model
train_data = train_df.select("features", "num_sold")
model = GBTRegressor(featuresCol="features", labelCol="num_sold")
trained_model = model.fit(train_data)

# Feature Engineering for Test Data
test_df = test_df.withColumn("year", year(col("date")))\
                 .withColumn("month", month(col("date")))\
                 .withColumn("dayofweek", dayofweek(col("date")))

# Prepare Features for Test Data
test_df = assembler.transform(test_df)

# Predict on Test Set
predictions = trained_model.transform(test_df).select("id", col("prediction").alias("num_sold"))

# Convert to Pandas for Visualization
predictions_pd = predictions.toPandas()
predictions_pd.to_csv("submission.csv", index=False)

print("Pipeline Completed Successfully!")


import plotly.express as px
import pandas as pd

# Convert Spark DataFrame to Pandas DataFrame for Visualization
pandas_train_df = train_df.select("date", "num_sold", "store", "product", "country").toPandas()

# Sales Trend Over Time
fig1 = px.line(pandas_train_df, x='date', y='num_sold', title='Sales Trend Over Time')
fig1.show()

# Sales Distribution by Store
fig2 = px.box(pandas_train_df, x='store', y='num_sold', title='Sales Distribution by Store')
fig2.show()

# Sales Distribution by Product
fig3 = px.box(pandas_train_df, x='product', y='num_sold', title='Sales Distribution by Product')
fig3.show()

# Sales Distribution by Country
fig4 = px.box(pandas_train_df, x='country', y='num_sold', title='Sales Distribution by Country')
fig4.show()


import plotly.graph_objects as go

# Ensure the `date` column is included in the `test_df` DataFrame
test_df = test_df.withColumn("date", col("date"))

# Predict on Test Set
predictions = trained_model.transform(test_df).select("id", col("date"), col("prediction").alias("num_sold"))

# Convert to Pandas for Visualization
predictions_pd = predictions.toPandas()

# Convert the date column to datetime format
pandas_train_df['date'] = pd.to_datetime(pandas_train_df['date'])
predictions_pd['date'] = pd.to_datetime(predictions_pd['date'])

# Merge the actual and predicted data on the date
merged_df = pandas_train_df.merge(predictions_pd, on='date', how='inner', suffixes=('_actual', '_predicted'))

# Plot the actual vs predicted sales
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['num_sold_actual'], mode='lines', name='Actual Sales'))
fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['num_sold_predicted'], mode='lines', name='Predicted Sales'))
fig.update_layout(title='Actual vs Predicted Sales', xaxis_title='Date', yaxis_title='Number of Items Sold')
fig.show()





