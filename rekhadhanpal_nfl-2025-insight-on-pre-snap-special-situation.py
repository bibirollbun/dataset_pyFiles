!pip install --upgrade pyspark[pandas_on_spark]


import pyspark
import pyspark.pandas as ps

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import warnings


# Create a SparkSession
spark = SparkSession.builder \
    .appName("NFL2025 Data Processing") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()


# Load datasets
path="/kaggle/input/nfl-big-data-bowl-2025"
games = spark.read.csv("/kaggle/input/nfl-big-data-bowl-2025/games.csv", header=True, inferSchema=True)
plays = spark.read.csv("/kaggle/input/nfl-big-data-bowl-2025/plays.csv", header=True, inferSchema=True)
player_play = spark.read.csv("/kaggle/input/nfl-big-data-bowl-2025/player_play.csv", header=True, inferSchema=True)
players = spark.read.csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv", header=True, inferSchema=True)

# Load tracking data for all weeks
tracking_files = [
    f"/kaggle/input/nfl-big-data-bowl-2025/tracking_week_{i}.csv" for i in range(1, 10)
]
tracking_data = spark.read.csv(tracking_files, header=True, inferSchema=True)


#  Merge datasets
plays_with_games = plays.join(games, "gameId", "inner")
plays_with_player = plays_with_games.join(player_play, ["gameId", "playId"], "inner")
full_data = plays_with_player.join(tracking_data, ["gameId", "playId", "nflId"], "inner")


#  Filter for Red Zone and Third-and-Long Situations
red_zone_full_data = full_data.filter(col("absoluteYardlineNumber") <= 20)
third_and_long_full_data = full_data.filter((col("down") == 3) & (col("yardsToGo") >= 8))


# Save processed Red Zone and Third-and-Long data to /kaggle/working/submission.csv
# red_zone_full_data.toPandas().to_csv("/kaggle/working/submission_red_zone.csv", index=False)
# third_and_long_full_data.toPandas().to_csv("/kaggle/working/submission_third_and_long.csv", index=False)

# # Fill null values in the DataFrame
# red_zone_full_data = red_zone_full_data.fillna("NA")
# third_and_long_full_data = third_and_long_full_data.fillna("NA")

# # Cast columns to appropriate types
# from pyspark.sql.types import StringType, DoubleType

# red_zone_full_data = red_zone_full_data.withColumn("timeToThrow", col("timeToThrow").cast(DoubleType()))
# third_and_long_full_data = third_and_long_full_data.withColumn("timeToThrow", col("timeToThrow").cast(DoubleType()))

# # Use "overwrite" mode to ensure the directory is clean
# red_zone_full_data.write.mode("overwrite").csv("/kaggle/working/submission_red_zone", header=True)
# third_and_long_full_data.write.mode("overwrite").csv("/kaggle/working/submission_third_and_long", header=True)



from pyspark.sql.functions import abs

# Add derived features
def preprocess_data(data):
    return data.withColumn(
        'playAction_encoded', when(col('playAction') == 'True', 1).otherwise(0)
    ).withColumn(
        'isDropback_encoded', col('isDropback').cast('int')
    ).withColumn(
        'winProbabilityDelta', abs(col('preSnapHomeTeamWinProbability') - col('preSnapVisitorTeamWinProbability'))
    ).withColumn(
        'yardsFromEndZone', 100 - col('absoluteYardlineNumber')
    )

red_zone_full_data = preprocess_data(red_zone_full_data)
third_and_long_full_data = preprocess_data(third_and_long_full_data)


# Feature Selection
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

red_zone_features = ['playAction_encoded', 'isDropback_encoded', 'yardsToGo', 'yardsFromEndZone', 
                     'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability', 'expectedPoints']
third_and_long_features = ['playAction_encoded', 'isDropback_encoded', 'yardsToGo', 'expectedPoints', 
                           'timeToThrow', 'passLength', 'dropbackType']

# Cast numerical columns to DoubleType
red_zone_full_data = red_zone_full_data.withColumn("timeToThrow", col("timeToThrow").cast(DoubleType()))
red_zone_full_data = red_zone_full_data.withColumn("passLength", col("passLength").cast(DoubleType()))
third_and_long_full_data = third_and_long_full_data.withColumn("timeToThrow", col("timeToThrow").cast(DoubleType()))
third_and_long_full_data = third_and_long_full_data.withColumn("passLength", col("passLength").cast(DoubleType()))

# Encode categorical column 'dropbackType' using StringIndexer
indexer = StringIndexer(inputCol="dropbackType", outputCol="dropbackType_index")
# red_zone_full_data = indexer.fit(red_zone_full_data).transform(red_zone_full_data)
third_and_long_full_data = indexer.fit(third_and_long_full_data).transform(third_and_long_full_data)


# Remove the original dropbackType column if necessary
import pandas as pd

if "dropbackType" in third_and_long_full_data.columns:
    third_and_long_full_data = third_and_long_full_data.drop("dropbackType")

third_and_long_features = [
    'playAction_encoded', 'isDropback_encoded', 'yardsToGo', 'expectedPoints', 
    'timeToThrow', 'passLength', 'dropbackType_index'
]


# Define target labels
red_zone_full_data = red_zone_full_data.withColumn(
    'success', when(col('expectedPointsAdded') > 0, 1).otherwise(0)
)
third_and_long_full_data = third_and_long_full_data.withColumn(
    'conversion', when(col('yardsToGo') <= col('yardsGained'), 1).otherwise(0)
)


# Train/Test Split
from pyspark.sql.functions import rand

# Random split for training and testing data
red_zone_train, red_zone_test = red_zone_full_data.randomSplit([0.8, 0.2], seed=42)
third_and_long_train, third_and_long_test = third_and_long_full_data.randomSplit([0.8, 0.2], seed=42)

# Train Models
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.feature import StringIndexer

red_zone_full_data = red_zone_full_data.fillna({
    "timeToThrow": 0.0,
    "passLength": 0.0,
    "yardsToGo": 0.0,
    "expectedPoints": 0.0
})

third_and_long_full_data = third_and_long_full_data.fillna({
    "timeToThrow": 0.0,
    "passLength": 0.0,
    "dropbackType_index": -1.0,  # Default index for missing categorical data
    "yardsToGo": 0.0,
    "expectedPoints": 0.0
})

from pyspark.ml.feature import VectorAssembler

# Prepare data for ML pipeline
def prepare_data(data, features, label):
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    return assembler.transform(data).select("features", label)

# Prepare data for training and testing
# Use the full processed DataFrame for the assembler
red_zone_train_transformed = prepare_data(red_zone_full_data, red_zone_features, "success")
third_and_long_train_transformed = prepare_data(third_and_long_full_data, third_and_long_features, "conversion")

# Split the transformed data into train and test sets
red_zone_train, red_zone_test = red_zone_train_transformed.randomSplit([0.8, 0.2], seed=42)
third_and_long_train, third_and_long_test = third_and_long_train_transformed.randomSplit([0.8, 0.2], seed=42)

# Red Zone
rf_red_zone = RandomForestClassifier(labelCol="success", featuresCol="features", seed=42)
rf_red_zone_model = rf_red_zone.fit(red_zone_train)

# Third-and-Long
gbt_third_and_long = GBTClassifier(labelCol="conversion", featuresCol="features", seed=42)
gbt_third_and_long_model = gbt_third_and_long.fit(third_and_long_train)


# Evaluate Models
evaluator = BinaryClassificationEvaluator(
    labelCol="success",  # The true label column
    rawPredictionCol="probability",  # Use the probability column for evaluation
    metricName="areaUnderROC"  # Metric: Area Under ROC Curve
)

# Red Zone
red_zone_preds = rf_red_zone_model.transform(red_zone_test)
red_zone_auc = evaluator.evaluate(red_zone_preds, {evaluator.metricName: "areaUnderROC"})


# Configure evaluator for Third-and-Long
third_and_long_evaluator = BinaryClassificationEvaluator(
    labelCol="conversion",  # Use the correct label column
    rawPredictionCol="probability",  # Use the prediction probabilities
    metricName="areaUnderROC"  # Evaluate using Area Under ROC Curve
)
# Third-and-Long
third_and_long_preds = gbt_third_and_long_model.transform(third_and_long_test)
third_and_long_auc = third_and_long_evaluator.evaluate(third_and_long_preds, {third_and_long_evaluator.metricName: "areaUnderROC"})


# Save predictions for Red Zone model
red_zone_preds.toPandas().to_csv("/kaggle/working/submission_red_zone_predictions.csv", index=False)

# Save predictions for Third-and-Long model
third_and_long_preds.toPandas().to_csv("/kaggle/working/submission_third_and_long_predictions.csv", index=False)


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="success", predictionCol="prediction", metricName="weightedPrecision"
)
red_zone_precision = precision_evaluator.evaluate(red_zone_preds)

# Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="success", predictionCol="prediction", metricName="weightedRecall"
)
red_zone_recall = recall_evaluator.evaluate(red_zone_preds)

# F1-Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="success", predictionCol="prediction", metricName="f1"
)
red_zone_f1 = f1_evaluator.evaluate(red_zone_preds)


# Repeat for Third-and-Long
precision_evaluator.setParams(labelCol="conversion")
third_and_long_precision = precision_evaluator.evaluate(third_and_long_preds)

recall_evaluator.setParams(labelCol="conversion")
third_and_long_recall = recall_evaluator.evaluate(third_and_long_preds)

f1_evaluator.setParams(labelCol="conversion")
third_and_long_f1 = f1_evaluator.evaluate(third_and_long_preds)


import matplotlib.pyplot as plt
import numpy as np

# Metrics data
metrics = ["ROC AUC", "Precision", "Recall", "F1-Score"]
red_zone_metrics = [red_zone_auc, red_zone_precision, red_zone_recall, red_zone_f1]
third_long_metrics = [third_and_long_auc, third_and_long_precision, third_and_long_recall, third_and_long_f1]

# Bar chart setup
x = np.arange(len(metrics))  # Positions for the metrics
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for Red Zone and Third-and-Long
bar1 = ax.bar(x - width / 2, red_zone_metrics, width, label="Red Zone", color="blue")
bar2 = ax.bar(x + width / 2, third_long_metrics, width, label="Third-and-Long", color="green")

# Add labels and title
ax.set_xlabel("Metrics")
ax.set_ylabel("Scores")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add data labels to the bars
for bar in bar1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

for bar in bar2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

# Show the plot
plt.tight_layout()
plt.show()


from pyspark.sql.functions import col, abs
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Step 1: Handle Missing Values
red_zone_full_data = red_zone_full_data.fillna({
    "timeToThrow": red_zone_full_data.select("timeToThrow").dropna().agg({"timeToThrow": "mean"}).collect()[0][0],
    "passLength": 0.0,
    "expectedPoints": 0.0,
})

# Step 2: Address Class Imbalance
success_data = red_zone_full_data.filter(col("success") == 1)
failure_data = red_zone_full_data.filter(col("success") == 0)
oversampled_success_data = success_data.sample(withReplacement=True, fraction=2.0)
balanced_data = failure_data.union(oversampled_success_data)

# Step 3: Feature Engineering
# Add Win Probability Delta
balanced_data = balanced_data.withColumn(
    "winProbabilityDelta", abs(col("preSnapHomeTeamWinProbability") - col("preSnapVisitorTeamWinProbability"))
)
# Drop columns if they already exist
columns_to_drop = ["playAction_index", "offenseFormation_index", "playAction_encoded", "offenseFormation_encoded"]
for col_name in columns_to_drop:
    if col_name in balanced_data.columns:
        balanced_data = balanced_data.drop(col_name)


# Convert BooleanType to StringType for playAction
balanced_data = balanced_data.withColumn("playAction", col("playAction").cast(StringType()))

# One-hot encode categorical features
indexer = StringIndexer(inputCols=["playAction", "offenseFormation"], outputCols=["playAction_index", "offenseFormation_index"])
balanced_data = indexer.fit(balanced_data).transform(balanced_data)

encoder = OneHotEncoder(inputCols=["playAction_index", "offenseFormation_index"], outputCols=["playAction_encoded", "offenseFormation_encoded"])
balanced_data = encoder.fit(balanced_data).transform(balanced_data)
# Add interaction features
balanced_data = balanced_data.withColumn(
    "timeToThrow_expectedPoints", col("timeToThrow") * col("expectedPoints")
)

# Step 4: Define and Train GBT Classifier
gbt = GBTClassifier(labelCol="success", featuresCol="features", seed=42)

# Step 5: Hyperparameter Tuning
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5, 7]) \
    .addGrid(gbt.stepSize, [0.1, 0.2, 0.3]) \
    .build()

crossval = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="success", metricName="areaUnderROC"),
    numFolds=3
)

# Train the model
cv_model = crossval.fit(red_zone_train)
best_model = cv_model.bestModel

# Step 6: Evaluate the Model
refined_preds = best_model.transform(red_zone_test)
evaluator = BinaryClassificationEvaluator(labelCol="success", rawPredictionCol="probability", metricName="areaUnderROC")
refined_auc = evaluator.evaluate(refined_preds)

print(f"Refined Red Zone ROC AUC: {refined_auc}")

# Step 7: Feature Importance
import matplotlib.pyplot as plt
import numpy as np

feature_importances = best_model.featureImportances.toArray()
plt.barh(np.array(red_zone_features), feature_importances)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance: Refined Red Zone Model")
plt.show()


# Save predictions for refined Red Zone model
refined_preds.toPandas().to_csv("/kaggle/working/submission_refined_red_zone_predictions.csv", index=False)


# Create evaluators for Precision, Recall, and F1-Score
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="success", predictionCol="prediction", metricName="weightedPrecision"
)
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="success", predictionCol="prediction", metricName="weightedRecall"
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="success", predictionCol="prediction", metricName="f1"
)

redifine_red_zone_precision = precision_evaluator.evaluate(refined_preds)
redifine_red_zone_recall = recall_evaluator.evaluate(refined_preds)
redifine_red_zone_f1 = f1_evaluator.evaluate(refined_preds)

#Printing all evaluation
print(f"Refined Red Zone ROC AUC: {refined_auc}")
print(f"Refined Red Zone precision: {redifine_red_zone_precision}")
print(f"Refined Red Zone Recall: {redifine_red_zone_recall}")
print(f"Refined Red Zone F1-Score: {redifine_red_zone_f1}")

print(f"Third-and-Long AUC: {third_and_long_auc}")
print(f"Third-and-Long Precision: {third_and_long_precision}")
print(f"Third-and-Long Recall: {third_and_long_recall}")
print(f"Third-and-Long F1-Score: {third_and_long_f1}")


# Create a DataFrame for evaluation metrics
evaluation_metrics = pd.DataFrame({
    "Model": ["Refined Red Zone", "Third-and-Long"],
    "ROC AUC": [refined_auc, third_and_long_auc],
    "Precision": [redifine_red_zone_precision, third_and_long_precision],
    "Recall": [redifine_red_zone_recall, third_and_long_recall],
    "F1-Score": [redifine_red_zone_f1, third_and_long_f1]
})

# Save evaluation metrics to /kaggle/working/submission.csv
evaluation_metrics.to_csv("/kaggle/working/submission.csv", index=False)

