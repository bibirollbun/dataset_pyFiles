!pip uninstall -qqy jupyterlab  # Remove unused packages from Kaggle's base image that conflict
!pip install -U -q "google-genai==1.7.0"


from google import genai
from google.genai import types

from IPython.display import HTML, Markdown, display


from google.api_core import retry


is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)


client = genai.Client(api_key=GOOGLE_API_KEY)

for model in client.models.list():
   print(model.name)



import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# model = genai.GenerativeModel('gemini-2.0-flash')

history = []

config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.0,
)

chat = client.chats.create(model='gemini-2.0-flash',  config=config_with_search, history = history)



prompt = """You are a Data Scientist that advises on data science problems. What is the goal of the  kaggle competion S5E4?
Analyse the task for the kaggle competion S5E4 and advise on the next steps. Seperate each step with the keyword " GenAI_step"  

use the headlines  
Exploratory Data Analysis (EDA)
Feature Engineering
Model Selection
Model Training
Prediction and Submission

Give precise instructions for implementation, but do not include code. 
"""
response = chat.send_message( prompt) 
#                             config=config_with_search)



Markdown(response.text)


import polars as pl

train_df = pl.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pl.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# train_df = train_df.sample (fraction = 0.2, shuffle = True)

print (train_df.head(4))

train_describe = train_df.describe ()
test_describe = test_df.describe ()

print (f"{train_describe}")



code_prompt = """suggest a python implemention for the following requirements :

using the 2 polars dataframes train_df and test_df to perform Exploratory Data Analysis (EDA) as suggested 

perform seperate analysis for the train and test data 

exclude the column "id" from the analysis 

store the results in a variable that can be used later in this chat so you can actually understand the EDA 
Ensure you use polars native syntax. 
I need to be able to copy and paste this code into a Python interpreter.
"""
response = chat.send_message([code_prompt, f"{train_df.schema}", f"{train_describe}", f"{test_describe}"])

Markdown(response.text)


# copied from output above 

def perform_eda(train_df: pl.DataFrame, test_df: pl.DataFrame) -> dict:
    """
    Performs Exploratory Data Analysis (EDA) on train and test DataFrames using Polars.

    Args:
        train_df: Polars DataFrame representing the training data.
        test_df: Polars DataFrame representing the test data.

    Returns:
        A dictionary containing EDA results for train and test DataFrames.
    """

    eda_results = {}

    # --- EDA for Training Data ---
    train_eda = {}
    train_df_no_id = train_df.drop("id")

    # 1. Descriptive Statistics
    train_eda["descriptive_stats"] = train_df_no_id.describe()

    # 2. Missing Value Analysis
    train_eda["missing_values"] = train_df_no_id.null_count().transpose()

    # 3. Data Types
    train_eda["data_types"] = pl.DataFrame({"column": train_df_no_id.columns, "dtype": [str(t) for t in train_df_no_id.dtypes]})

    # 4. Value counts for categorical columns
    categorical_columns = [col for col in train_df_no_id.columns if train_df_no_id[col].dtype in [pl.Utf8, pl.Categorical]]
    train_eda["value_counts"] = {}
    for col in categorical_columns:
        try:
            train_eda["value_counts"][col] = train_df_no_id.group_by(col).count().sort("count", descending=True)
        except Exception as e:
            train_eda["value_counts"][col] = f"Error calculating value counts for {col}: {e}"

    eda_results["train"] = train_eda

    # --- EDA for Test Data ---
    test_eda = {}
    test_df_no_id = test_df.drop("id")

    # 1. Descriptive Statistics
    test_eda["descriptive_stats"] = test_df_no_id.describe()

    # 2. Missing Value Analysis
    test_eda["missing_values"] = test_df_no_id.null_count().transpose()

    # 3. Data Types
    test_eda["data_types"] = pl.DataFrame({"column": test_df_no_id.columns, "dtype": [str(t) for t in test_df_no_id.dtypes]})

    # 4. Value counts for categorical columns
    categorical_columns = [col for col in test_df_no_id.columns if test_df_no_id[col].dtype in [pl.Utf8, pl.Categorical]]
    test_eda["value_counts"] = {}
    for col in categorical_columns:
        try:
            test_eda["value_counts"][col] = test_df_no_id.group_by(col).count().sort("count", descending=True)
        except Exception as e:
            test_eda["value_counts"][col] = f"Error calculating value counts for {col}: {e}"

    eda_results["test"] = test_eda

    return eda_results


# Example Usage (replace with your actual DataFrames)

eda_results = perform_eda(train_df, test_df)

# Now you can access the EDA results for train and test DataFrames
# For example:
print(eda_results["train"]["descriptive_stats"])
print(eda_results["test"]["missing_values"])
print(eda_results["train"]["value_counts"])


code_prompt = """
Create  an implementation for Feature Engineering as previously suggested on 
the 2 polars dataframes train_df and test_df and the results of eda below, 

extract cardinality from the eda below 


Do not include "id" in the analysis 
Do not use one hot encoding for features with a cardinality over 20
use .to_dummy for one hot encoding if needed
ensure that you include Categorical Feature Encoding for non numerical features. 
For each feature select the best encoding.
use numbers to represent the episode values 
"""

response = chat.send_message([code_prompt,f"{eda_results}"])

Markdown(response.text)



def feature_engineering(train_df: pl.DataFrame, test_df: pl.DataFrame, eda_results: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Performs feature engineering on train and test DataFrames.

    Args:
        train_df: Polars DataFrame representing the training data.
        test_df: Polars DataFrame representing the test data.
        eda_results: Dictionary containing EDA results.

    Returns:
        A tuple containing the feature-engineered train and test DataFrames.
    """

    # --- Feature Engineering for Training Data ---
    train_df_fe = train_df.drop("id")

    # 1. Episode_Length_minutes Imputation (using median)
    median_episode_length_train = train_df_fe["Episode_Length_minutes"].median()
    train_df_fe = train_df_fe.with_columns(
        pl.col("Episode_Length_minutes").fill_null(median_episode_length_train)
    )

    # 2. Guest_Popularity_percentage Imputation (using median)
    median_guest_popularity_train = train_df_fe["Guest_Popularity_percentage"].median()
    train_df_fe = train_df_fe.with_columns(
        pl.col("Guest_Popularity_percentage").fill_null(median_guest_popularity_train)
    )

    # 3. Number_of_Ads Imputation (using median)
    median_number_of_ads_train = train_df_fe["Number_of_Ads"].median()
    train_df_fe = train_df_fe.with_columns(
        pl.col("Number_of_Ads").fill_null(median_number_of_ads_train)
    )

    # 4. Categorical Feature Encoding
    # a. Podcast_Name:  Label Encoding or Leave as is if cardinality > 20, otherwise to_dummies
    podcast_name_cardinality = len(eda_results["train"]["value_counts"]["Podcast_Name"])
    if podcast_name_cardinality > 20:
        podcast_name_mapping = {name: i for i, name in enumerate(train_df_fe["Podcast_Name"].unique())}
        train_df_fe = train_df_fe.with_columns(
            pl.col("Podcast_Name").replace_strict(podcast_name_mapping).cast(pl.UInt8).alias("Podcast_Name_Encoded")
          
        )
        print ("Podcast_Name_Encoded UInt8")  
        train_df_fe = train_df_fe.drop("Podcast_Name")
    else:
        train_df_fe = train_df_fe.to_dummies(subset=["Podcast_Name"])

    # b. Episode_Title: Label Encoding (using numbers)
    episode_title_mapping = {title: i for i, title in enumerate(train_df_fe["Episode_Title"].unique())}
    train_df_fe = train_df_fe.with_columns(
        pl.col("Episode_Title").replace_strict (episode_title_mapping).cast (pl.UInt8).alias("Episode_Title_Encoded")
    )
    train_df_fe = train_df_fe.drop("Episode_Title")

    # c. rest of categorical features : to_dummies
    train_df_fe = train_df_fe.to_dummies(columns = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"])
    
    # --- Feature Engineering for Test Data ---
    test_df_fe = test_df.drop("id")

    # 1. Episode_Length_minutes Imputation (using median from train)
    test_df_fe = test_df_fe.with_columns(
        pl.col("Episode_Length_minutes").fill_null(median_episode_length_train)
    )

    # 2. Guest_Popularity_percentage Imputation (using median from train)
    test_df_fe = test_df_fe.with_columns(
        pl.col("Guest_Popularity_percentage").fill_null(median_guest_popularity_train)
    )

    # 3. Number_of_Ads Imputation (using median from train)
    test_df_fe = test_df_fe.with_columns(
        pl.col("Number_of_Ads").fill_null(median_number_of_ads_train)
    )

    # 4. Categorical Feature Encoding
    # a. Podcast_Name: Label Encoding or Leave as is if cardinality > 20, otherwise to_dummies
    if podcast_name_cardinality > 20:
        test_df_fe = test_df_fe.with_columns(
            pl.col("Podcast_Name").replace(podcast_name_mapping).fill_null(-1).cast (pl.UInt8).alias("Podcast_Name_Encoded")
        )
        test_df_fe = test_df_fe.drop("Podcast_Name")
    else:
        test_df_fe = test_df_fe.to_dummies(subset=["Podcast_Name"])

    # b. Episode_Title: Label Encoding (using the same mapping as train)
    test_df_fe = test_df_fe.with_columns(
        pl.col("Episode_Title").replace(episode_title_mapping).fill_null(-1).cast (pl.UInt8).alias("Episode_Title_Encoded")
    )
    test_df_fe = test_df_fe.drop("Episode_Title")
    # rest of the categorical features,  
    test_df_fe = test_df_fe.to_dummies(columns = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"])
    

    
    
    return train_df_fe, test_df_fe


# Example Usage (replace with your actual DataFrames)
# Assuming you have train_df, test_df and eda_results


train_df_fe, test_df_fe = feature_engineering(train_df, test_df, eda_results)

# Now you can access the feature engineered DataFrames
print("Feature Engineered Train DataFrame:")
print(train_df_fe.head())
print("\nFeature Engineered Test DataFrame:")
print(test_df_fe.head())


print(train_df_fe.schema)

with pl.Config(tbl_rows=100, tbl_cols = 21):
    print(train_df_fe.describe().transpose(include_header = True))
    print(test_df_fe.describe().transpose(include_header = True))
    print (train_df_fe.head())



code_prompt = """ 
adding to the existing code and using the polars dataframe train_df_engineered 
 
suggest an implementation for Model Selection and Model Training that allow a prediction on the test dataset.
assume that feature encoding was provided

use Gradient Boosting

add progress reports and save important intermediate results to disc.   
"""


response = chat.send_message(code_prompt)


Markdown(response.text)


import polars as pl
import pandas as pd  # For compatibility with scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle  # For saving models
import time
import os

def model_selection_and_training(train_df_engineered: pl.DataFrame, test_df_engineered: pl.DataFrame, target_column: str, model_save_path: str = "model.pkl") -> pl.DataFrame:
    """
    Performs model selection and training on the engineered training data using Gradient Boosting,
    and generates predictions on the engineered test data.

    Args:
        train_df_engineered: Polars DataFrame representing the feature-engineered training data.
        test_df_engineered: Polars DataFrame representing the feature-engineered test data.
        target_column: The name of the target column in the training data.
        model_save_path: Path to save the trained model.

    Returns:
        A Polars DataFrame containing the predictions on the test data.
    """

    start_time = time.time()

    # 1. Data Preparation
    print("Preparing data for model training...")
    # Convert Polars DataFrame to Pandas DataFrame for scikit-learn compatibility
    X = train_df_engineered.drop(target_column).to_pandas()
    y = train_df_engineered[target_column].to_pandas()

    # 2. Model Selection (Gradient Boosting)
    print("Selecting model (Gradient Boosting)...")
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)  # You can tune hyperparameters here

    # 3. Cross-Validation
    print("Performing cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Using KFold for more control
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse_scores = [-score**0.5 for score in cv_scores]  # Convert to RMSE
    print(f"Cross-validation RMSE scores: {cv_rmse_scores}")
    print(f"Mean cross-validation RMSE: {sum(cv_rmse_scores) / len(cv_rmse_scores)}")

    # 4. Model Training
    print("Training model...")
    model.fit(X, y)

    # 5. Save Model
    print(f"Saving model to {model_save_path}...")
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)

    # 6. Prediction on Test Data
    print("Preparing test data and generating predictions...")
    X_test = test_df_engineered.to_pandas()  # Convert test data to Pandas

    # Ensure that the test set has the same columns as the training set
    # This is crucial, especially after one-hot encoding
    missing_cols = set(X.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    # Ensure the order of column is the same
    X_test = X_test[X.columns]

    predictions = model.predict(X_test)

    # 7. Create Prediction DataFrame
    predictions_df = pl.DataFrame({"predictions": predictions})

    end_time = time.time()
    print(f"Model selection, training, and prediction completed in {end_time - start_time:.2f} seconds.")

    return predictions_df


# Example Usage (replace with your actual DataFrames)
# Assuming you have train_df_engineered and test_df_engineered


target_column = "Listening_Time_minutes"
model_save_path = "my_model.pkl"

predictions_df = model_selection_and_training(train_df_fe, test_df_fe, target_column, model_save_path)

print("\nPredictions on Test Data:")
print(predictions_df)


print (predictions_df.max())

print (predictions_df.mean())


prompt = """

analyse the output and consider that this is the result you presented previously 

Preparing data for model training...
Selecting model (Linear Regression)...
Performing cross-validation...
<ipython-input-41-e924107b909b>:40: RuntimeWarning: invalid value encountered in scalar power
  cv_rmse_scores = [-score**0.5 for score in cv_scores]  # Convert to RMSE
Cross-validation RMSE scores: [nan, nan, nan, nan, nan]
Mean cross-validation RMSE: nan
Training model...
Saving model to my_model.pkl...
Preparing test data and generating predictions...
Model selection, training, and prediction completed in 18.66 seconds.

Predictions on Test Data:
shape: (250_000, 1)
┌─────────────┐
│ predictions │
│ ---         │
│ f64         │
╞═════════════╡
│ 55.758518   │
│ 20.07975    │
│ 51.495083   │
│ 81.727627   │
│ 50.031742   │
│ …           │
│ 9.156532    │
│ 59.419567   │
│ 4.334019    │
│ 79.70863    │
│ 57.062222   │
└─────────────┘

"""

response = chat.send_message(prompt)


Markdown(response.text)





for c in train_df_fe.columns :
    print (f"column {c}, null = {train_df_fe.get_column (c).is_null().sum()}, infinite ={train_df_fe.get_column (c).is_infinite().sum()}")
    


code_prompt = """ 


adding to the existing code and finalize the Submission by providing a file submission.csv that combines the prediction results. 
assume that training was executed with the suggestion you provided. 


"""


response = chat.send_message(code_prompt)


Markdown(response.text)







def create_submission_file(test_df: pl.DataFrame, predictions_df: pl.DataFrame, submission_file_path: str = "submission.csv") -> None:
    """
    Creates a submission file in the format required by Kaggle, combining the test IDs with the model predictions.

    Args:
        test_df: The original Polars DataFrame representing the test data (containing the 'id' column).
        predictions_df: A Polars DataFrame containing the model predictions.
        submission_file_path: The path to save the submission file.
    """

    print("Creating submission file...")

    # 1. Extract IDs from the original test_df
    test_ids = test_df.select("id")

    # 2. Combine IDs and Predictions
    submission_df = test_ids.with_columns(predictions_df)

    # 3. Rename the prediction column to match the competition's requirements
    submission_df = submission_df.rename({"predictions": "Listening_Time_minutes"})

    # 4. Save to CSV
    submission_df.write_csv(submission_file_path)

    print(f"Submission file created successfully at {submission_file_path}")



create_submission_file(test_df, predictions_df)

