import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")



train_df = pd.read_csv(
    "/kaggle/input/playground-series-s5e5/train.csv"
)
test_df = pd.read_csv(
    "/kaggle/input/playground-series-s5e5/test.csv"
)


print(train_df.shape, test_df.shape)


train_df.drop(columns=["id"], inplace=True)


train_df.info()


train_df.describe(include="all")


train_df.head()


train_df["Sex"].value_counts()


numerical_features = train_df.select_dtypes(include=["int64", "float64"]).columns
categorical_features = train_df.select_dtypes(include=["object"]).columns


independent = train_df.drop(columns=["Calories"])
dependent = train_df["Calories"]


import scipy.stats as stats
sns.histplot(train_df["Calories"], kde=True, bins=30)
fig = plt.figure()
stats.probplot(train_df["Calories"], plot=plt)
plt.title("Calories Distribution")
plt.xlabel("Calories")
plt.ylabel("Count")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

num_features = len(numerical_features)
fig, axes = plt.subplots(nrows=num_features, ncols=2, figsize=(12, 4 * num_features))

for i, col in enumerate(numerical_features):
    # Histogram with KDE
    sns.histplot(train_df[col], kde=True, bins=30, ax=axes[i, 0])
    axes[i, 0].set_title(f"{col} Distribution")
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel("Count")

    # Scatter plot vs Calories
    sns.scatterplot(x=train_df[col], y=train_df["Calories"], ax=axes[i, 1])
    axes[i, 1].set_title(f"{col} vs Calories")
    axes[i, 1].set_xlabel(col)
    axes[i, 1].set_ylabel("Calories")

plt.tight_layout()
plt.show()



train_df.groupby('Sex')['Calories'].describe()


# checking the realtion of sex and calories
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sex', y='Calories', data=train_df)
plt.show()


# creating a new feature BMI from the height and weight
train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
train_df["BMI"] = train_df["BMI"].round(2)



# checking for correlation
numerical_features = train_df.select_dtypes(include=["int64", "float64"]).columns
correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix")
plt.show()



train_df.drop(columns=["Height", "Weight"], inplace=True)


def custom_BMI_feature(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df["BMI"] = df["BMI"].round(2)
    df.drop(columns=["Height", "Weight"], inplace=True)
    return df


test_df.drop(columns=["id"], inplace=True)
test_df = custom_BMI_feature(test_df)
test_df.head()


train_df.head()


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 6))
numerical_features = train_df.select_dtypes(include=["int64", "float64"]).columns
for i, col in enumerate(numerical_features):
    sns.boxplot(data=train_df, x=col, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f"Boxplot of {col}")
    axes[i//2, i%2].set_xlabel(col)
    axes[i//2, i%2].set_ylabel("Value")
plt.tight_layout()
plt.show()


def get_iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)]



outliers = get_iqr_outliers(train_df, 'Calories')
print("Number of outliers in Calories:", len(outliers))
outliers.head()


# Encode sex column for train_df and test_df
train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})


import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def tfdf_train_and_evaluate(df, target_col):
    # Split data into training and test sets (80-20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare datasets with the proper task (regression)
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=target_col, task=tfdf.keras.Task.REGRESSION)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=target_col, task=tfdf.keras.Task.REGRESSION)

    # Model configurations 
    model_configs = {
        "RandomForest": tfdf.keras.RandomForestModel,
        "GradientBoostedTrees": tfdf.keras.GradientBoostedTreesModel,
        "Cart": tfdf.keras.CartModel,
    }

    best_model_info = {
        "model_name": None,
        "model_instance": None,
        "score": np.inf,
    }

    # Train and evaluate each model
    for model_name, model in model_configs.items():
        print(f"\nTraining {model_name}...")

        # Build and train the model
        model_instance = model(task=tfdf.keras.Task.REGRESSION)
        model_instance.fit(train_ds, verbose=0)

        # Make predictions and evaluate
        preds = model_instance.predict(test_ds).flatten()
        y_true = test_df[target_col].values
        score = rmsle(y_true, preds)

        print(f"  RMSLE: {score:.4f}")

        # Track the best model
        if score < best_model_info["score"]:
            best_model_info.update({
                "model_name": model_name,
                "model_instance": model_instance,
                "score": score
            })

    print(f"\n Best Model: {best_model_info['model_name']}")
    print(f"RMSLE: {best_model_info['score']:.4f}")

    return best_model_info



best_model_info = tfdf_train_and_evaluate(train_df, "Calories")



# Convert test data to TensorFlow dataset (with proper semantics) for inference
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, task=tfdf.keras.Task.REGRESSION)

test_df_pred = best_model_info["model_instance"].predict(test_ds).flatten()


test_df_temp = pd.read_csv(
    "/kaggle/input/playground-series-s5e5/test.csv"
)
submission = pd.DataFrame({
    "id": test_df_temp["id"],
    "Calories": test_df_pred
})
submission.to_csv("submission.csv", index=False)

