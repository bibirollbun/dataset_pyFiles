# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import mean_squared_error

# Models
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

# Optional: To display images if saved (though plt.show() is preferred for direct display)
# from PIL import Image
# from IPython.display import display, Image as IPImage

# Set plot style
sns.set_style("whitegrid")


# Load the training dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")

print("Training data loaded successfully.")


# Get a concise summary of the dataframe
print("--- Data Info ---")
df.info()
print("-" * 30)


# Calculate descriptive statistics for numerical columns
print("--- Descriptive Statistics ---")
print(df.describe())
print("-" * 30)

# Check for missing values
print("\n--- Missing Value Counts ---")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0]) # Show only columns with missing values
print("-" * 30)
# Note: Significant missing values in Episode_Length_minutes, Guest_Popularity_percentage, and a single one in Number_of_Ads.


# Analyze the distribution of the target variable 'Listening_Time_minutes'
print("--- Target Variable Analysis (Listening_Time_minutes) ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(df['Listening_Time_minutes'], kde=True, ax=axes[0], bins=50)
axes[0].set_title('Distribution of Listening Time')
axes[0].set_xlabel('Listening Time (minutes)')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=df['Listening_Time_minutes'], ax=axes[1])
axes[1].set_title('Box Plot of Listening Time')
axes[1].set_xlabel('Listening Time (minutes)')

plt.tight_layout()
plt.show()
# Observation: The target variable covers a wide range and seems somewhat multi-modal.


### 3.4 Numerical Feature Distributions


# Analyze distributions of key numerical features
print("--- Numerical Feature Distributions ---")
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='skyblue', bins=40) # dropna for plotting original dist
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

    # Add a boxplot for outlier visualization
    ax_boxplot = axes[i].twinx()
    sns.boxplot(x=df[col].dropna(), ax=ax_boxplot, color='lightcoral', width=0.2)
    ax_boxplot.set(yticks=[])
    ax_boxplot.set_ylabel('')
    ax_boxplot.spines['right'].set_visible(False)
    ax_boxplot.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
# Observations:
# - Episode_Length_minutes and Number_of_Ads appear right-skewed.
# - Popularity percentages seem to have values exceeding 100, suggesting potential outliers or data entry issues.


# Analyze distributions of key categorical features
print("--- Categorical Feature Distributions ---")
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Determine layout based on number of categorical columns
n_cols = 2
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    order = df[col].value_counts().index
    sns.countplot(y=df[col], order=order, ax=axes[i], palette="viridis")
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel('Count')
    axes[i].set_ylabel(col)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# Observation: Shows the frequency of each category within these features. Genres seem relatively balanced.


# Analyze relationship between categorical features and the target variable
print("--- Categorical Features vs. Listening Time ---")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows)) # Reuse layout vars
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    # Calculate order based on median listening time for better visualization
    median_order = df.groupby(col)['Listening_Time_minutes'].median().sort_values().index
    sns.boxplot(x=col, y='Listening_Time_minutes', data=df, ax=axes[i], order=median_order)
    axes[i].set_title(f'{col} vs. Listening Time')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Listening Time (minutes)')
    axes[i].tick_params(axis='x', rotation=30) # Rotate labels slightly

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Print median listening times for reference
print("\n--- Median Listening Times by Category ---")
for col in categorical_cols:
    medians = df.groupby(col)['Listening_Time_minutes'].median().sort_values()
    print(f"\n{col}:\n{medians}")
print("-" * 30)
# Observation: There appear to be some differences in median listening time based on categories, e.g., Sentiment, Publication Time.


# Analyze correlations between the original numerical features
print("--- Numerical Feature Correlation Matrix ---")

# Select only the original numerical columns for this correlation analysis
original_numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
correlation_matrix = df[original_numerical_cols].corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features (Original)')
plt.show()

print("\nCorrelation with Target (Listening_Time_minutes):")
print(correlation_matrix['Listening_Time_minutes'].sort_values(ascending=False))
print("-" * 30)
# Observation: Episode_Length_minutes shows the strongest positive correlation with Listening_Time_minutes among these features. Correlations are generally weak otherwise.


# Impute missing values using the median, which is robust to outliers.
print("--- Handling Missing Values ---")

cols_to_impute = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

for col in cols_to_impute:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"Imputed missing values in '{col}' with median: {median_val:.2f}")

# Verify imputation
print("\nMissing values after imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0]) # Should be empty or show unrelated columns if any exist
print("-" * 30)


# Address potential outliers identified during EDA
print("--- Handling Outliers/Anomalies ---")

# Cap popularity percentages at 100
df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].clip(upper=100)
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(upper=100)
print("Capped Host and Guest Popularity at 100.")

# Cap Number_of_Ads at the 99th percentile to handle extreme values
ads_99th_percentile = df['Number_of_Ads'].quantile(0.99)
df['Number_of_Ads'] = df['Number_of_Ads'].clip(upper=ads_99th_percentile)
print(f"Capped Number_of_Ads at 99th percentile: {ads_99th_percentile:.2f}")

# Display descriptive statistics after capping
print("\nDescriptive statistics after outlier handling:")
print(df[['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']].describe())
print("-" * 30)


# Apply log transformation (log1p) to reduce skewness in specific numerical features.
# log1p is used instead of log to handle potential zero values gracefully (log(1+x)).
print("--- Applying Log Transformation ---")

cols_to_log = ['Episode_Length_minutes', 'Number_of_Ads']

fig, axes = plt.subplots(len(cols_to_log), 2, figsize=(12, 5 * len(cols_to_log)))

for i, col in enumerate(cols_to_log):
    # Plot original distribution
    sns.histplot(df[col], kde=True, ax=axes[i, 0], bins=40)
    axes[i, 0].set_title(f'Original Distribution: {col}')

    # Apply log1p transformation
    df[col] = np.log1p(df[col])
    print(f"Applied log1p transformation to '{col}'.")

    # Plot transformed distribution
    sns.histplot(df[col], kde=True, ax=axes[i, 1], bins=40)
    axes[i, 1].set_title(f'Log-Transformed Distribution: {col}')

plt.tight_layout()
plt.show()

print("\nFirst 5 rows showing transformed columns:")
print(df[cols_to_log].head())
print("-" * 30)


# Encode categorical features for model consumption.
print("--- Encoding Categorical Features ---")

# Ordinal Encoding for Episode_Sentiment (Negative=0, Neutral=1, Positive=2)
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_mapping)
print("Applied Ordinal Encoding to 'Episode_Sentiment'.")

# One-Hot Encoding for nominal features
nominal_cols = ['Genre', 'Publication_Day', 'Publication_Time']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
print(f"Applied One-Hot Encoding to: {', '.join(nominal_cols)} (with drop_first=True).")

# Display changes
print("\nDataFrame Head after Encoding:")
print(df.head())
print("\nDataFrame Info after Encoding:")
df.info()
print("-" * 30)


# Define the feature set (X) and target variable (y)
print("--- Defining Features and Target ---")

# Drop identifier columns and the target variable to create feature matrix X
X = df.drop(['id', 'Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis=1)
y = df['Listening_Time_minutes']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("Feature columns:", X.columns.tolist())
print("-" * 30)


# Split the data into training and testing sets
print("--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("-" * 30)


# Scale numerical features using StandardScaler.
# Fit the scaler ONLY on the training data, then transform both train and test data.
print("--- Scaling Features ---")

# Identify columns to scale (all columns in X are now numerical)
# Note: While one-hot encoded columns are binary, scaling them often doesn't hurt
# and can sometimes help certain algorithms. We'll scale all features here.
cols_to_scale = X_train.columns

scaler = StandardScaler()

# Fit on training data and transform
X_train_scaled = scaler.fit_transform(X_train[cols_to_scale])

# Transform test data using the *same* fitted scaler
X_test_scaled = scaler.transform(X_test[cols_to_scale])

# Convert scaled arrays back to DataFrames (maintains column names)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols_to_scale, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=cols_to_scale, index=X_test.index)

print("Features scaled successfully.")
print("\nScaled Training Data Head:")
print(X_train_scaled.head())
print("\nScaled Training Data Description:")
print(X_train_scaled.describe())
print("-" * 30)


# Define the primary evaluation metric
print("--- Evaluation Metric ---")
print("Root Mean Squared Error (RMSE) will be used.")
print("RMSE penalizes larger errors more heavily and is in the same units as the target variable.")
print("-" * 30)


# Train a simple baseline model
print("--- Baseline Model: Linear Regression ---")

# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("Linear Regression model trained.")

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate the model
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"Linear Regression Test RMSE: {rmse_lr:.4f}")
print("-" * 30)


# Train a more advanced gradient boosting model
print("--- Advanced Model: LightGBM ---")

# Initialize and train the model
lgbm_model = LGBMRegressor(random_state=42)
# LightGBM can sometimes handle non-scaled data well, but we'll use scaled data for consistency
lgbm_model.fit(X_train_scaled, y_train)
print("LightGBM model trained.")

# Make predictions on the test set
y_pred_lgbm = lgbm_model.predict(X_test_scaled)

# Evaluate the model
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f"LightGBM Test RMSE: {rmse_lgbm:.4f}")

# Compare with baseline
print(f"\nLightGBM improvement over Linear Regression RMSE: {rmse_lr - rmse_lgbm:.4f}")
print("-" * 30)
# Observation: LightGBM significantly outperforms the baseline Linear Regression model. ***


# Analyze feature importances from the trained LightGBM model
print("--- LightGBM Feature Importance Analysis ---")

# Extract feature importances
importances = lgbm_model.feature_importances_
feature_names = X_train_scaled.columns # Use columns from the scaled data used for training

# Create a DataFrame for easier handling
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Plot top N features
N = 20
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(N), palette='viridis')
plt.title(f'Top {N} Feature Importances (LightGBM)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Print the full list
print("\nFeature Importances (Ranked):")
print(feature_importance_df)
print("-" * 30)
# Observation: Episode Length, Host Popularity, and Guest Popularity appear to be the most influential features according to the LightGBM model.


df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df


# import seaborn as sns
# numerical_columns = [
#     'Episode_Length_minutes',
#     'Host_Popularity_percentage',
#     'Guest_Popularity_percentage',
#     'Number_of_Ads'
# ]
# df_clean = df[numerical_columns + ['Listening_Time_minutes']].dropna()
# sns.set(style="white", palette="muted")
# for col in numerical_columns:
#     plt.figure(figsize=(8, 6))
#     sns.kdeplot(
#         x=df_clean[col],
#         y=df_clean['Listening_Time_minutes'],
#         cmap="viridis",
#         fill=True,
#         thresh=0.05,
#         levels=100,
#         cbar=True
#     )
#     plt.title(f'Density Scatter Plot: {col} vs Listening_Time_minutes')
#     plt.xlabel(col)
#     plt.ylabel('Listening_Time_minutes')
#     plt.tight_layout()
#     plt.show()




import pandas as pd
import numpy as np

# ---------- Handling Missing Values ----------
# Create missing value indicators
df['Is_Episode_Length_Missing'] = df['Episode_Length_minutes'].isna().astype(int)
df['Is_Guest_Popularity_Missing'] = df['Guest_Popularity_percentage'].isna().astype(int)

# Fill missing values
df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean())
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].mean())
df1=df.copy()
# ---------- Numerical Interactions ----------
df['Host_Guest_Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
df['Popularity_Product'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']

# ---------- Time Features ----------
# Convert day to ordinal
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['Publication_Day_Ordinal'] = df['Publication_Day'].map({day: i for i, day in enumerate(day_order)})

# Is weekend
df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# Encode time of day
time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
df['Publication_Time_Code'] = df['Publication_Time'].map(time_mapping)

# ---------- Text Features ----------
# Extract episode number
df['Episode_Number'] = df['Episode_Title'].str.extract(r'(\d+)', expand=False).astype(float)

# ---------- Group-based Features ----------
# Average listening time by genre
genre_avg_listening = df.groupby('Genre')['Listening_Time_minutes'].mean().to_dict()
df['Genre_Avg_Listening'] = df['Genre'].map(genre_avg_listening)

# Count total episodes by podcast
podcast_counts = df['Podcast_Name'].value_counts().to_dict()
df['Total_Episodes_By_Podcast'] = df['Podcast_Name'].map(podcast_counts)

# ---------- Encoding ----------
# One-hot encode sentiment
df = pd.get_dummies(df, columns=['Episode_Sentiment'], prefix='Sentiment')

# Optional: one-hot encode categorical vars like 'Genre', 'Publication_Time'
# df = pd.get_dummies(df, columns=['Genre', 'Publication_Time'])

# Final check: drop unused original columns if needed
df.drop(['Episode_Title', 'Publication_Day', 'Publication_Time'], axis=1, inplace=True)

# View engineered DataFrame
print(df.head())




import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df.drop(['Host_Popularity_percentage','Publication_Time','Publication_Day'], axis=1)
df=df.dropna()
df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)

X = df.drop(columns=['Listening_Time_minutes', 'id', 'Podcast_Name', 'Episode_Title'])
y = df['Listening_Time_minutes']
num_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
cat_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_prep.shape[1],)),
    tf.keras.layers.Dense(150, activation='relu'),

    tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train_prep, y_train, epochs=10, batch_size=256, validation_split=0.5)
test_loss, test_mae = model.evaluate(X_test_prep, y_test)
print("Test MAE:", test_mae)
y_pred = model.predict(X_test_prep).flatten()  # Flatten in case it's shape (n,1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)



dft=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


dft['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
dft['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)

dft['Episode_Number'] = dft['Episode_Title'].str.extract(r'(\d+)').astype(float)

X_dft = dft.drop(columns=['id', 'Episode_Title'])

X_dft_prep = preprocessor.transform(X_dft)

y_dft_pred = model.predict(X_dft_prep).flatten()

submission = pd.DataFrame({
    'id': dft['id'],
    'Listening_Time_minutes': y_dft_pred
})

submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv created successfully!")












































