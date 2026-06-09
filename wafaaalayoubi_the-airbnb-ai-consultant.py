import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
import warnings

from google.cloud import bigquery
from google.cloud import storage
from google.cloud import vision
from google.api_core.client_options import ClientOptions

from kaggle_secrets import UserSecretsClient

from tqdm.auto import tqdm
import ast 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from scipy.stats import uniform, randint

import xgboost as xgb

from tqdm.notebook import tqdm

# Import VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk



# Ignore harmless warnings to keep the notebook clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


project_id = 'bq-hackathon-ab-123'
client = bigquery.Client(project=project_id)
print(f"Successfully connected to BigQuery project: {project_id}")


# Define the SQL query to select all data from our table
sql = """
    SELECT *
    FROM `bq-hackathon-ab-123.airbnb_analysis.raw_listings`
"""


# Run the query and load the results into a Pandas DataFrame
print("Running query to load data...")
df = client.query(sql).to_dataframe()


# Verify the data loaded correctly
print("\nData loaded successfully from BigQuery!")


# --- Step 1: The First Contact ---

# 1. Check the dimensions of the DataFrame (rows, columns)
print(f"Dataset Shape: {df.shape}")


# 2. Display the first 5 rows to get a feel for the data
print("\n--- First 5 Rows ---")
display(df.head())


# 3. Display the last 5 rows to check for any summary rows or weird data at the end
print("\n--- Last 5 Rows ---")
display(df.tail())


# 4. Get the concise summary of the DataFrame. This is the most critical part of this step!
# It tells us column names, non-null counts, and data types (Dtype).
print("\n--- DataFrame Info ---")
# Using verbose=True to ensure all columns are shown, even if there are many
df.info(verbose=True, show_counts=True)


# 5. Check for any completely duplicate rows
print(f"\n--- Duplicate Rows ---")
print(f"Number of completely duplicate rows: {df.duplicated().sum()}")


print("\n--- Summary Statistics for Numerical Columns ---")
df.describe()


golden_columns = [
    'id',
    'name',
    'description',
    'picture_url',
    'host_is_superhost',  
    'room_type',
    'property_type',
    'accommodates',
    'bathrooms_text',
    'bedrooms',
    'beds',
    'amenities',
    'price',
    'review_scores_rating'
]


df_selected = df[golden_columns].copy()


# Verify the new DataFrame
print("--- New DataFrame with Selected 'Golden' Columns ---")
print(f"The new shape is: {df_selected.shape}")
print("\nHere's a look at the first few rows:")
display(df_selected.head())


# n updated .info() to see the non-null counts for just these columns.
print("\n--- Updated Info for Selected Columns ---")
df_selected.info()


# Making a copy
df_cleaning = df_selected.copy()


# Handle Missing Values
# We will drop any rows where our essential columns are empty.
# These are 'review_scores_rating', 'description', and 'price'.
initial_rows = len(df_cleaning)
df_cleaning.dropna(subset=['review_scores_rating', 'description', 'price'], inplace=True)
final_rows = len(df_cleaning)


print("--- 1. Handling Missing Values ---")
print(f"Initial number of rows: {initial_rows}")
print(f"Number of rows after dropping essential NaNs: {final_rows}")
print(f"Number of rows removed: {initial_rows - final_rows}")
print(f"Percentage of data retained: {100 * final_rows / initial_rows:.2f}%")


# Check for Duplicates
initial_rows = len(df_cleaning)
df_cleaning.drop_duplicates(subset=['id'], inplace=True)
final_rows = len(df_cleaning)


print("\n--- Checking for Duplicates ---")
if initial_rows == final_rows:
    print("No duplicate listing IDs found. Good!")
else:
    print(f"Removed {initial_rows - final_rows} duplicate rows.")


# Correct Data Types for 'price'
# The 'price' column is an object (text) due to '$' and ','.
# We need to remove these characters and convert it to a float.
print("\n--- Correcting 'price' Data Type ---")
print(f"Original 'price' Dtype: {df_cleaning['price'].dtype}")


# This line uses string operations to remove '$' and ',' then converts to a numeric type
df_cleaning['price'] = df_cleaning['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)


print(f"New 'price' Dtype: {df_cleaning['price'].dtype}")
print("First 5 price values after cleaning:")
print(df_cleaning['price'].head())


# --- Final Cleaned DataFrame ---
# Let's assign this to a final, clean name for the rest of the project.
df_clean = df_cleaning.copy()


print("\n--- Final Cleaned DataFrame Info ---")
df_clean.info()


# Set the style for our plots for a professional look
sns.set_style('whitegrid')


# Distribution of Review Scores
plt.figure(figsize=(12, 6))
sns.histplot(df_clean['review_scores_rating'], bins=30, kde=True)
plt.title('Distribution of Review Scores Rating', fontsize=16)
plt.xlabel('Rating (1-5)', fontsize=12)
plt.ylabel('Number of Listings', fontsize=12)
plt.show()


# Distribution of Price (Checking for Outliers)
plt.figure(figsize=(12, 6))
sns.histplot(df_clean['price'], bins=100, kde=False) # kde=False as the long tail is extreme
plt.title('Distribution of Listing Prices', fontsize=16)
plt.xlabel('Price (in USD)', fontsize=12)
plt.ylabel('Number of Listings', fontsize=12)
plt.show()


# Let's also look at a boxplot for price to better see the outliers
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_clean['price'])
plt.title('Boxplot of Listing Prices to Identify Outliers', fontsize=16)
plt.xlabel('Price (in USD)', fontsize=12)
plt.xscale('log') # Use a log scale to better visualize the wide range of prices
plt.show()


# Composition of Room Types
plt.figure(figsize=(10, 6))
sns.countplot(x='room_type', data=df_clean, order=df_clean['room_type'].value_counts().index)
plt.title('Number of Listings by Room Type', fontsize=16)
plt.xlabel('Room Type', fontsize=12)
plt.ylabel('Number of Listings', fontsize=12)
plt.xticks(rotation=45)
plt.show()


# Review Scores by Room Type
plt.figure(figsize=(12, 7))
sns.boxplot(x='room_type', y='review_scores_rating', data=df_clean, order=df_clean['room_type'].value_counts().index)
plt.title('Review Scores Distribution by Room Type', fontsize=16)
plt.xlabel('Room Type', fontsize=12)
plt.ylabel('Review Scores Rating', fontsize=12)
plt.xticks(rotation=45)
plt.show()


# Selecting only the most relevant numerical columns for the heatmap
numerical_cols_for_corr = [
    'review_scores_rating',
    'price',
    'accommodates',
    'bedrooms',
    'beds'
]


# Calculate the correlation matrix
correlation_matrix = df_clean[numerical_cols_for_corr].corr()


# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Key Numerical Features', fontsize=16)
plt.show()


# --- Verify a few random image URLs ---

# Get 5 random listings from our clean DataFrame
random_samples = df_clean.sample(5)


# Loop through the random samples and print their info
for index, row in random_samples.iterrows():
    print("-" * 50)
    print(f"Listing Name: {row['name']}")
    print(f"Listing ID: {row['id']}")
    print(f"Image URL: {row['picture_url']}")
    print(f"Airbnb Listing URL: https://www.airbnb.com/rooms/{row['id']}")
    print("-" * 50)
    print("\n")


# Define the full table ID for our new table
# Format is: project_id.dataset_id.table_name
table_id = f"{project_id}.airbnb_analysis.cleaned_listings"


# Configure the job to write the DataFrame to BigQuery
# write_disposition='WRITE_TRUNCATE' means if the table already exists, overwrite it.
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
)


# This sends our pandas DataFrame 'df_clean' to be saved as the table specified by 'table_id'.
job = client.load_table_from_dataframe(
    df_clean, table_id, job_config=job_config
)


# Wait for the job to complete and print the result
job.result()  # Waits for the job to finish
print(f"Successfully saved {len(df_clean)} rows to the table: {table_id}")


# 1. Install the Google Cloud Vision library
!pip install google-cloud-vision -q


# Get the API key we stored securely in Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")


# Create the client using ONLY the client_options with our API key
# This is the direct and correct way to use an API key.
client_options = ClientOptions(api_key=api_key)
client = vision.ImageAnnotatorClient(client_options=client_options)

print("Vision API client created successfully.")


# # (RUN ONCE)
# # Test with a single, reliable image URL
# image_url = "https://a0.muscache.com/pictures/47759803/e70c0f85_original.jpg"
# image = vision.Image()
# image.source.image_uri = image_url


# # (RUN ONCE)
# # Perform label detection on the image
# print(f"\nAnalyzing test image: {image_url}")
# response = client.label_detection(image=image)
# labels = response.label_annotations


# # (RUN ONCE)
# # Print the results
# print("\n--- Labels Detected ---")
# for label in labels:
#     print(f"- {label.description} (Score: {label.score:.2f})")


def get_image_labels(image_url):
    """
    Analyzes an image from a URL using the Google Vision API.

    Args:
        image_url (str): The public URL of the image to analyze.

    Returns:
        list: A list of the top 5 label descriptions, or an empty list if an error occurs.
    """
    # Add a small delay to avoid overwhelming the API
    time.sleep(0.1)

    try:
        image = vision.Image()
        image.source.image_uri = image_url
        response = client.label_detection(image=image)
        
        # Check for errors in the API response itself
        if response.error.message:
            # print(f"Error processing {image_url}: {response.error.message}")
            return []

        # Extract just the description text from the top 5 labels
        labels = [label.description for label in response.label_annotations[:5]]
        return labels

    except Exception as e:
        # Catch any other exceptions (e.g., network errors, broken links)
        # print(f"An exception occurred for {image_url}: {e}")
        return []


# # (RUN ONCE)
# # --- Test the function with a sample URL from our dataset ---
# sample_url = df_clean['picture_url'].iloc[0] # Get the first image URL from our clean data
# print(f"Testing function with URL: {sample_url}")


# # (RUN ONCE)
# labels = get_image_labels(sample_url)
# print("\n--- Labels returned by our function ---")
# print(labels)


# # --- Create and Save a Permanent Stratified Sample Flag (RUN ONCE) ---

# # Create the new column, default to False
# df_clean['is_in_1k_sample'] = False

# # Get the indices of our desired stratified sample
# # We are just getting the IDs/indices of the rows, not the data itself yet.
# _, sample_indices = train_test_split(
#     df_clean.index,  # We stratify on the index
#     test_size=1000,
#     stratify=df_clean['room_type'],
#     random_state=42
# )


# (RUN ONCE)
# # Set the flag to True for our chosen sample indices
# df_clean.loc[sample_indices, 'is_in_1k_sample'] = True


# (RUN ONCE)
# # Verify the flag was set correctly
# print("--- Verification of the new flag ---")
# print(df_clean['is_in_1k_sample'].value_counts())


# (RUN ONCE)
# # --- Confirm the Stratified Sample is Balanced ---

# # 1. Define our new sample and the full dataset
# flagged_sample = df_clean[df_clean['is_in_1k_sample'] == True]
# full_df = df_clean

# print("--- Final Comparison: Flagged Sample vs. Full Dataset ---")

# # 2. Compare the proportions of 'room_type'
# sample_counts = flagged_sample['room_type'].value_counts(normalize=True).rename('Sample')
# full_counts = full_df['room_type'].value_counts(normalize=True).rename('Full Dataset')

# # Combine into a single DataFrame for plotting and viewing
# comparison_df = pd.concat([sample_counts, full_counts], axis=1) * 100

# # 3. Plot the comparison
# comparison_df.plot(kind='bar', figsize=(12, 7))
# plt.title('Comparison of Room Type Proportions (%) - STRATIFIED SAMPLE', fontsize=16)
# plt.ylabel('Percentage of Listings (%)')
# plt.xticks(rotation=0)
# plt.show()

# # 4. Print the exact percentages to confirm
# print("\n--- Room Type Proportions ---")
# print("Note how the percentages in the 'Sample' column are now almost identical to the 'Full Dataset' column.")
# print(comparison_df)


bq_client = bigquery.Client(project=project_id)
print("BigQuery client re-initialized.")


# # Save this new master DataFrame back to BigQuery, overwriting the old 'cleaned_listings' # (RUN ONCE)
# table_id = f"{project_id}.airbnb_analysis.cleaned_listings"
# job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
# job = bq_client.load_table_from_dataframe(
#     df_clean, table_id, job_config=job_config
# )
# job.result()  # Wait for the job to finish


# print(f"\nSuccessfully saved 'df_clean' with the new 'is_in_1k_sample' flag back to {table_id}")


# # --- Select the Flagged Sample and Analyze Images (RUN ONCE) ---

# # Select ONLY the rows we just flagged
# df_api_sample = df_clean[df_clean['is_in_1k_sample'] == True].copy()
# print(f"Selected our permanent sample of {len(df_api_sample)} images to analyze.")


# # (RUN ONCE)
# tqdm.pandas()
# print("Analyzing images... (This will take a few minutes)")
# df_api_sample['image_labels'] = df_api_sample['picture_url'].progress_apply(get_image_labels)


# # (RUN ONCE)
# output_filename = 'airbnb_stratified_1k_sample_with_labels.csv'
# df_api_sample.to_csv(output_filename, index=False)
# print(f"\nImage analysis complete! Results saved to '{output_filename}'")


# Ensure we are authenticated with Google Cloud
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)
print("Authenticated with Google Cloud.")


# Define the direct GCS path to our results file
gcs_path = 'gs://bq-hackathon-wa-data/results/airbnb_stratified_1k_sample_with_labels.csv'

# Use pandas to read the CSV directly from the GCS path
print(f"Loading data from: {gcs_path}")
df_final_sample = pd.read_csv(gcs_path, converters={'image_labels': ast.literal_eval})


# --- Verify the Results ---
print("\nSuccessfully loaded enriched data!")
print(f"The final sample has {df_final_sample.shape[0]} rows and {df_final_sample.shape[1]} columns.")
print("\nHere's a look at our final, model-ready data:")
display(df_final_sample[['name', 'review_scores_rating', 'image_labels']].head())


# --- Part 3.1: Feature Engineering - One-Hot Encode Image Labels ---

# This will be our main modeling DataFrame
df_model = df_final_sample.copy()


# 1. Initialize the MultiLabelBinarizer
# This is a special tool from scikit-learn for handling columns that contain lists of labels.
mlb = MultiLabelBinarizer()


# 2. Fit and transform the 'image_labels' column
# This creates a new DataFrame where each column is a unique label and each value is 1 or 0.
encoded_labels = mlb.fit_transform(df_model['image_labels'])


# 3. Create a new DataFrame from the encoded labels
# We'll give the new columns clear names, like 'has_label_Bed'.
label_df = pd.DataFrame(encoded_labels, columns=[f"has_label_{cls.replace(' ', '_').lower()}" for cls in mlb.classes_])


# 4. Concatenate the new label DataFrame with our main DataFrame
# We need to make sure the indices align for a correct merge.
df_model.reset_index(drop=True, inplace=True)
label_df.reset_index(drop=True, inplace=True)
df_model = pd.concat([df_model, label_df], axis=1)


# --- Verify the Results ---
print("Successfully created one-hot encoded features from image labels.")
print(f"The DataFrame now has {df_model.shape[1]} columns.")
print("\nHere's a sample of the new 'has_label_' columns:")

# To make the output readable, find a few interesting columns to display
# Let's find columns for common labels like 'bed', 'swimming_pool', and 'kitchen'
display_cols = ['image_labels']
if 'has_label_bed' in df_model.columns: display_cols.append('has_label_bed')
if 'has_label_swimming_pool' in df_model.columns: display_cols.append('has_label_swimming_pool')
if 'has_label_kitchen' in df_model.columns: display_cols.append('has_label_kitchen')

display(df_model[display_cols].head())


# --- Part 3.2: Feature Engineering - TF-IDF for Descriptions ---

# 1. Initialize the TfidfVectorizer
# We'll set some parameters to get the best results:
# - max_features=500: We'll only keep the top 500 most important words. This prevents having too many columns.
# - stop_words='english': Automatically removes common English words like 'the', 'a', 'is'.
# - min_df=5: Only consider words that appear in at least 5 different descriptions. This removes very rare, possibly misspelled words.
tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=5)


# 2. Fit and transform the 'description' column
# This creates a sparse matrix where each column is a word and each value is its TF-IDF score.
tfidf_features = tfidf.fit_transform(df_model['description'])


# 3. Create a new DataFrame from the TF-IDF features
# We'll give the new columns clear names, like 'keyword_beach'.
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f"keyword_{word}" for word in tfidf.get_feature_names_out()])


# 4. Concatenate the new TF-IDF DataFrame with our main DataFrame
# The indices should already be aligned, but resetting is a safe practice.
df_model.reset_index(drop=True, inplace=True)
tfidf_df.reset_index(drop=True, inplace=True)
df_model = pd.concat([df_model, tfidf_df], axis=1)


# --- Verify the Results ---
print("Successfully created TF-IDF features from descriptions.")
print(f"The DataFrame now has {df_model.shape[1]} columns.")
print("\nHere's a sample of the new 'keyword_' columns:")

# Let's display a few potential keywords to see the result
display_cols = ['description']
if 'keyword_beach' in df_model.columns: display_cols.append('keyword_beach')
if 'keyword_cozy' in df_model.columns: display_cols.append('keyword_cozy')
if 'keyword_luxury' in df_model.columns: display_cols.append('keyword_luxury')

display(df_model[display_cols].head())


# --- Part 3.3: Train and Evaluate the Predictive Model ---

# 1. Define our Features (X) and Target (y)
# We select all columns we want the model to potentially learn from.
y = df_model['review_scores_rating']
X = df_model.drop(columns=[
    'review_scores_rating', 'id', 'name', 'description',
    'picture_url', 'amenities', 'image_labels'
])


# 2. Split the Data FIRST to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Create the preprocessing pipeline
# This pipeline will handle different column types separately and correctly.
# It handles imputation and encoding INSIDE the pipeline, after the split.

# Identify categorical and numerical columns
categorical_features = ['room_type', 'property_type', 'bathrooms_text']
numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep the other columns (our TF-IDF and image labels)
)


# 4. Define the model
# We'll use XGBoost, a powerful and popular gradient boosting model.
model = xgb.XGBRegressor(random_state=42)


# 5. Create the full pipeline
# This chains the preprocessing and the model together.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])


# 6. Train the pipeline on the TRAINING data
print("--- Training the model ---")
pipeline.fit(X_train, y_train)
print("Model training complete.")


# 7. Make predictions on the unseen TESTING data
print("\n--- Evaluating the model ---")
y_pred = pipeline.predict(X_test)


# 8. Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Let's look at a few predictions vs. actuals
prediction_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)
print("\nSample of Predictions vs. Actuals:")
print(prediction_comparison)


# --- Part 3.4: Iteration 1 - Remove Price Outliers ---

# 1. Determine the 99th percentile price threshold
price_cap = df_model['price'].quantile(0.99)
print(f"The 99th percentile price is: ${price_cap:.2f}. We will remove listings priced above this.")


# 2. Create the new, filtered DataFrame
df_no_outliers = df_model[df_model['price'] < price_cap].copy()
print(f"Original number of listings: {len(df_model)}")
print(f"Number of listings after removing outliers: {len(df_no_outliers)}")


# 3. Re-define X and y with the new outlier-removed data
y_no_outliers = df_no_outliers['review_scores_rating']
X_no_outliers = df_no_outliers.drop(columns=[
    'review_scores_rating', 'id', 'name', 'description',
    'picture_url', 'amenities', 'image_labels'
])


# 4. Split the new data
X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(
    X_no_outliers, y_no_outliers, test_size=0.2, random_state=42
)


# 5. Train the same pipeline on the new, cleaner data
print("\n--- Training model on data with outliers removed ---")
pipeline.fit(X_train_no, y_train_no)
print("Model training complete.")


# 6. Evaluate the new model
print("\n--- Evaluating the improved model ---")
y_pred_no = pipeline.predict(X_test_no)

mse_no = mean_squared_error(y_test_no, y_pred_no)
r2_no = r2_score(y_test_no, y_pred_no)

print(f"New Mean Squared Error (MSE): {mse_no:.4f}")
print(f"New R-squared (R²): {r2_no:.4f}")

# Compare to our baseline
print(f"\nBaseline R² was: {r2:.4f}")
print(f"Improvement in R²: {r2_no - r2:.4f}")


# --- Part 3.5: Final Model with Price Per Person Feature ---

# 1. Start with our best dataset (outliers removed)
df_ppp = df_no_outliers.copy()


# 2. Engineer the new 'price_per_person' feature
# We will add a small number (1) to 'accommodates' to avoid any division-by-zero errors, just in case.
df_ppp['price_per_person'] = df_ppp['price'] / df_ppp['accommodates']


# Download the VADER lexicon (required)
nltk.download('vader_lexicon')
tqdm.pandas()  # enable progress_apply


# 3. Add our sentiment feature (which we know is powerful)

analyzer = SentimentIntensityAnalyzer()
def get_sentiment_score(text):
    return analyzer.polarity_scores(str(text))['compound']
df_ppp['description_sentiment'] = df_ppp['description'].progress_apply(get_sentiment_score)


# 4. Create our binary target variable
df_ppp['is_top_tier'] = (df_ppp['review_scores_rating'] >= 4.9).astype(int)


# 5. Re-run the entire classification pipeline with ALL our best features
y_final_ppp = df_ppp['is_top_tier']
X_final_ppp = df_ppp.drop(columns=[
    'review_scores_rating', 'is_top_tier', 'id', 'name', 'description',
    'picture_url', 'amenities', 'image_labels'
])

X_train_ppp, X_test_ppp, y_train_ppp, y_test_ppp = train_test_split(
    X_final_ppp, y_final_ppp, test_size=0.2, random_state=42, stratify=y_final_ppp
)


print("\n--- Training the FINAL CHAMPION model ---")
# We use the same classifier pipeline as before.
# It will automatically handle our two new numerical features: 'description_sentiment' and 'price_per_person'
classifier_model = xgb.XGBClassifier(random_state=42)
pipeline_class = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', classifier_model)])
pipeline_class.fit(X_train_ppp, y_train_ppp)
print("Model training complete.")


# 6. Evaluate the final champion model
print("\n--- Evaluating the FINAL CHAMPION model ---")
y_pred_final_ppp = pipeline_class.predict(X_test_ppp)

accuracy_final = accuracy_score(y_test_ppp, y_pred_final_ppp)
precision_final = precision_score(y_test_ppp, y_pred_final_ppp)
recall_final = recall_score(y_test_ppp, y_pred_final_ppp)
f1_final = f1_score(y_test_ppp, y_pred_final_ppp)

print(f"Final Champion Accuracy: {accuracy_final:.4f}")
print(f"Final Champion Precision: {precision_final:.4f}")
print(f"Final Champion Recall: {recall_final:.4f}")
print(f"Final Champion F1-Score: {f1_final:.4f}")


# 7. Visualize the Final Champion Confusion Matrix
print("\n--- Final Champion Confusion Matrix ---")
cm_final = confusion_matrix(y_test_ppp, y_pred_final_ppp)
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Top Tier', 'Top Tier'],
            yticklabels=['Not Top Tier', 'Top Tier'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# --- Part 4.1: Extract and Visualize Feature Importances ---

# 1. Extract the trained XGBoost model from our champion pipeline
final_model = pipeline_class.named_steps['classifier']


# 2. Get the feature names from the preprocessor step of our champion pipeline
# This correctly gets the name for every single feature, including the one-hot encoded and TF-IDF columns.
feature_names = pipeline_class.named_steps['preprocessor'].get_feature_names_out()


# 3. Create a DataFrame of feature importances
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)


# 4. Display the Top 20 most important features
print("--- Top 20 Most Important Features (The 'Secret Formula') ---")
display(feature_importance_df.head(20))


# 5. Visualize the Top 20 most important features
plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
plt.title('Top 20 Most Important Features for Predicting a "Top Tier" Listing', fontsize=16)
plt.xlabel('Importance Score (XGBoost)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()


# --- Part 4.1: Identify Underperforming Listings ---

# 1. Start with our final, best DataFrame
# This is the 'df_ppp' DataFrame which includes 'price_per_person' and 'sentiment'.
df_final = df_ppp.copy()


# 2. Define the correct feature matrix (X) and pipeline
# These must match the variables used to train our best model.
X_final_features = df_final.drop(columns=[
    'review_scores_rating', 'is_top_tier', 'id', 'name', 'description',
    'picture_url', 'amenities', 'image_labels'
])
final_pipeline = pipeline_class  # This is our champion pipeline


# 3. Use our trained champion pipeline to make predictions on the ENTIRE sample
print("--- Using the champion model to predict tiers for all 990 listings ---")
all_predictions = final_pipeline.predict(X_final_features)


# 4. Add the predictions as a new column to our DataFrame
df_final['predicted_tier'] = all_predictions


# 5. Create our list of "underperformers"
# These are listings that are ACTUALLY 'Not Top Tier' OR that our model PREDICTS are 'Not Top Tier'.
underperformers_df = df_final[
    (df_final['is_top_tier'] == 0) | (df_final['predicted_tier'] == 0)
].copy()


# 6. For our demo, we'll select a small, manageable sample of 5 underperformers to improve.
scorecard_candidates = underperformers_df.sample(5, random_state=42)

print(f"\nIdentified {len(underperformers_df)} potential underperformers.")
print(f"Selected 5 candidates to generate an AI Scorecard for.")
display(scorecard_candidates[['name', 'review_scores_rating', 'is_top_tier', 'predicted_tier']])


# 1. Install the Google Generative AI library
!pip install -q google-generativeai

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient



user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 2. Get our "Secret Formula" (no change here)
top_features = feature_importance_df['feature'].head(5).tolist()
top_features_clean = [name.replace('num__keyword_', '').replace('_', ' ') for name in top_features]

# 3. Create our Generative Model (no change here)
model = genai.GenerativeModel('gemini-2.5-flash')


# Loop through our 5 candidate listings
print("--- Generating AI Scorecards with a DIRECT COMMAND PROMPT ---")

results_list = []
for index, row in scorecard_candidates.iterrows():
    print(f"\n--- Analyzing Listing: {row['name']} ---")


    prompt = f"""
    **ROLE:** You are an expert Airbnb host consultant named ListingLens.

    **TASK:** Analyze the provided data for an underperforming Airbnb listing and generate a concise "AI Scorecard".

    **DATA ANALYSIS:**
    - **Description Sentiment Score:** {row['description_sentiment']:.2f} (A score of 0.9+ is excellent).
    - **Main Photo Content:** The primary photo features the following elements: {row['image_labels']}
    - **Top Keywords for Success:** Our AI model shows the most important keywords for a top-tier listing are: {top_features_clean}

    **OUTPUT INSTRUCTIONS:**
    You MUST format your response using the following template. Do NOT add any extra conversational text.

    **Photo Score (out of 10):** [Give a score based on whether the photo features any of the 'Top Keywords for Success'. A photo with a top keyword gets a higher score.]
    **Why:** [Explain in one sentence why you gave that score. For example: "The main photo showcases general furniture but misses the opportunity to highlight a key feature."]

    **Description Score (out of 10):** [Give a score based on the sentiment. A sentiment score above 0.9 is a 9/10 or 10/10. A score around 0.5 is a 5/10.]
    **Why:** [Explain in one sentence why you gave that score. For example: "The sentiment is moderately positive but lacks the enthusiastic language of top-tier listings."]

    **Final AI Suggestion:** [In one or two sentences, give a single, powerful recommendation that combines the photo and description analysis. For example: "Your photo shows a 'patio', which is a top feature! Update your description to prominently mention your beautiful patio to attract more guests."]
    """

# ... (the rest of the code is the same) ...

    # Generate the content
    try:
        response = model.generate_content(prompt)
        suggestion = response.text
        print("AI SUGGESTION:")
        print(suggestion)
        results_list.append({'name': row['name'], 'review_scores_rating': row['review_scores_rating'], 'suggestion': suggestion})
    except Exception as e:
        # ... (error handling is the same) ...
        print("Error")
    
    time.sleep(20)

# (Saving the results code is the same)
# ...


# --- NEW, IMPORTANT PART: SAVE THE RESULTS ---
# 5. Create a final DataFrame from our list of results
final_scorecard_df = pd.DataFrame(results_list)

# 6. Define a filename and save the DataFrame to a CSV
final_output_filename = 'AI_Scorecard_Results.csv'
final_scorecard_df.to_csv(final_output_filename, index=False)


# 7. Display the final DataFrame and confirm the save
print("\n\n--- FINAL AI SCORECARD RESULTS ---")
print(f"All suggestions have been generated and saved to '{final_output_filename}'")
print("You can download this file from the 'Data' tab on the right.")
display(final_scorecard_df)


# # --- Part 3.5: Iteration 2 - Add Amenity Count Feature ---

# # 1. Engineer the new feature
# # The 'amenities' column is a string that looks like a list (e.g., '["TV", "Wifi"]').
# # We can count the number of commas and add 1 to get a good estimate of the amenity count.
# # We'll apply this to our outlier-removed DataFrame from the previous step.
# df_amenities = df_no_outliers.copy()
# df_amenities['amenity_count'] = df_amenities['amenities'].str.count(',') + 1

# # Handle cases where there are no amenities (which would result in a count of 1).
# # If the amenities string is empty ('[]'), the count should be 0.
# df_amenities.loc[df_amenities['amenities'] == '[]', 'amenity_count'] = 0

# print("--- 1. New 'amenity_count' feature created ---")
# print("Sample of the new feature:")
# display(df_amenities[['amenities', 'amenity_count']].head())


# # 2. Re-define X and y with this new feature included
# y_amenities = df_amenities['review_scores_rating']
# X_amenities = df_amenities.drop(columns=[
#     'review_scores_rating', 'id', 'name', 'description',
#     'picture_url', 'amenities', 'image_labels'
# ])


# # 3. Split the new data
# X_train_am, X_test_am, y_train_am, y_test_am = train_test_split(
#     X_amenities, y_amenities, test_size=0.2, random_state=42
# )


# # 4. Train the same pipeline on the newly enriched data
# # The pipeline will automatically handle our new numerical 'amenity_count' column.
# print("\n--- 2. Training model with 'amenity_count' feature ---")
# pipeline.fit(X_train_am, y_train_am)
# print("Model training complete.")


# # 5. Evaluate the new model
# print("\n--- 3. Evaluating the improved model ---")
# y_pred_am = pipeline.predict(X_test_am)

# mse_am = mean_squared_error(y_test_am, y_pred_am)
# r2_am = r2_score(y_test_am, y_pred_am)

# print(f"New Mean Squared Error (MSE): {mse_am:.4f}")
# print(f"New R-squared (R²): {r2_am:.4f}")

# # Compare to our previous best
# print(f"\nPrevious R² was: {r2_no:.4f}")
# print(f"Improvement in R²: {r2_am - r2_no:.4f}")


# # --- Part 3.6: Iteration 3 - Hyperparameter Tuning with RandomizedSearchCV ---

# # 1. Use our best dataset so far: the one with outliers removed.
# y_best = y_no_outliers
# X_best = X_no_outliers

# # We'll use the full dataset for the search, as RandomizedSearchCV has built-in cross-validation.
# # We still need to split a final hold-out test set to evaluate the *final* best model.
# X_train_best, X_test_final, y_train_best, y_test_final = train_test_split(
#     X_best, y_best, test_size=0.2, random_state=42
# )


# # 2. Define the hyperparameter search space
# # We are telling RandomizedSearchCV which settings to try and what range of values to test.
# param_dist = {
#     'regressor__n_estimators': randint(100, 1000),
#     'regressor__learning_rate': uniform(0.01, 0.3),
#     'regressor__max_depth': randint(3, 10),
#     'regressor__subsample': uniform(0.7, 0.3),
#     'regressor__colsample_bytree': uniform(0.7, 0.3)
# }


# # 3. Set up the RandomizedSearchCV
# # n_iter=25: It will try 25 different random combinations of the settings above.
# # cv=3: It will use 3-fold cross-validation for each combination.
# # n_jobs=-1: It will use all available CPU cores to speed up the search.
# random_search = RandomizedSearchCV(
#     pipeline, # We are tuning the entire pipeline
#     param_distributions=param_dist,
#     n_iter=25,
#     cv=3,
#     scoring='r2',
#     n_jobs=-1,
#     random_state=42,
#     verbose=1 # This will print progress updates
# )


# # 4. Run the search
# # This is the longest-running step. It will train 25 * 3 = 75 models.
# print("--- Starting Hyperparameter Search (This will take ~10-15 minutes) ---")
# random_search.fit(X_train_best, y_train_best)
# print("\nSearch complete.")


# # 5. Get the best model and its parameters
# print("\nBest R² score found during search:", random_search.best_score_)
# print("Best parameters found:")
# print(random_search.best_params_)

# best_model = random_search.best_estimator_


# # 6. Evaluate the FINAL best model on the hold-out test set
# print("\n--- Evaluating the FINAL Tuned Model ---")
# y_pred_final = best_model.predict(X_test_final)

# mse_final = mean_squared_error(y_test_final, y_pred_final)
# r2_final = r2_score(y_test_final, y_pred_final)

# print(f"Final Mean Squared Error (MSE): {mse_final:.4f}")
# print(f"Final R-squared (R²): {r2_final:.4f}")

# # Compare to our previous best
# print(f"\nPrevious best R² was: {r2_no:.4f}")
# print(f"Improvement from tuning: {r2_final - r2_no:.4f}")


# # --- Final Model - Add One-Hot Encoded Amenities ---
# from sklearn.feature_extraction.text import CountVectorizer

# # We start from our best dataset: the one with outliers removed.
# df_final_model = df_no_outliers.copy()

# # 1. Clean the amenities text
# def clean_amenities(text):
#     text = text.replace('[', '').replace(']', '').replace('"', '')
#     return ' '.join([amenity.strip().lower() for amenity in text.split(',')])

# df_final_model['amenities_cleaned'] = df_final_model['amenities'].apply(clean_amenities)

# # 2. Use CountVectorizer to one-hot encode the top 30 amenities
# vectorizer = CountVectorizer(max_features=30, binary=True)
# amenity_features = vectorizer.fit_transform(df_final_model['amenities_cleaned'])
# amenity_df = pd.DataFrame(amenity_features.toarray(), columns=[f"has_amenity_{name}" for name in vectorizer.get_feature_names_out()])

# # 3. Combine the new features with our main DataFrame
# df_final_model.reset_index(drop=True, inplace=True)
# amenity_df.reset_index(drop=True, inplace=True)
# df_final_model = pd.concat([df_final_model, amenity_df], axis=1)

# print(f"Successfully created {amenity_df.shape[1]} new amenity features.")

# # 4. Re-run the full modeling pipeline with these new features
# y_final = df_final_model['review_scores_rating']
# X_final = df_final_model.drop(columns=[
#     'review_scores_rating', 'id', 'name', 'description', 'picture_url',
#     'amenities', 'amenities_cleaned', 'image_labels'
# ])

# X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
#     X_final, y_final, test_size=0.2, random_state=42
# )

# print("\n--- Training the FINAL model ---")
# pipeline.fit(X_train_final, y_train_final)
# print("Model training complete.")

# print("\n--- Evaluating the FINAL model ---")
# y_pred_final = pipeline.predict(X_test_final)

# mse_final = mean_squared_error(y_test_final, y_pred_final)
# r2_final = r2_score(y_test_final, y_pred_final)

# print(f"Final Mean Squared Error (MSE): {mse_final:.4f}")
# print(f"Final R-squared (R²): {r2_final:.4f}")

# # Compare to our previous best
# print(f"\nPrevious best R² was: {r2_no:.4f}")
# print(f"Improvement in R²: {r2_final - r2_no:.4f}")


# # --- Part 3.5: Final Model - Classification ---

# # 1. Start with our best dataset (outliers removed)
# df_class = df_no_outliers.copy()
# # 2. Create our new binary target variable 'is_top_tier'
# # We define "Top Tier" as a rating of 4.9 or higher.
# df_class['is_top_tier'] = (df_class['review_scores_rating'] >= 4.9).astype(int)

# print("--- 1. Created New Binary Target: 'is_top_tier' ---")
# print(df_class['is_top_tier'].value_counts(normalize=True))
# # 3. Define our Features (X) and new Target (y)
# y_class = df_class['is_top_tier']
# X_class = df_class.drop(columns=[
#     'review_scores_rating', 'is_top_tier', 'id', 'name', 'description',
#     'picture_url', 'amenities', 'image_labels'
# ])
# # 4. Split the data
# X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
#     X_class, y_class, test_size=0.2, random_state=42, stratify=y_class # Stratify on y is important for classification
# )
# # 5. Define a new XGBoost CLASSIFIER model within our pipeline
# # We replace the regressor with a classifier.
# classifier_model = xgb.XGBClassifier(random_state=42)
# pipeline_class = Pipeline(steps=[('preprocessor', preprocessor),
#                                  ('classifier', classifier_model)])
# # 6. Train the classification pipeline
# print("\n--- 2. Training the Classification Model ---")
# pipeline_class.fit(X_train_c, y_train_c)
# print("Model training complete.")
# # 7. Make predictions and evaluate
# print("\n--- 3. Evaluating the Classification Model ---")
# y_pred_c = pipeline_class.predict(X_test_c)

# # Calculate metrics
# accuracy = accuracy_score(y_test_c, y_pred_c)
# precision = precision_score(y_test_c, y_pred_c)
# recall = recall_score(y_test_c, y_pred_c)
# f1 = f1_score(y_test_c, y_pred_c)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")
# # 8. Visualize the Confusion Matrix
# print("\n--- Confusion Matrix ---")
# cm = confusion_matrix(y_test_c, y_pred_c)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Not Top Tier', 'Top Tier'],
#             yticklabels=['Not Top Tier', 'Top Tier'])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()




