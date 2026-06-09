import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error
import ast # Used to safely evaluate a string containing a Python literal

train_file_path = '/kaggle/input/sparta-2024-data-science-competition/train.csv'
train_data = pd.read_csv(train_file_path)
test_file_path = '/kaggle/input/sparta-2024-data-science-competition/test.csv'
test_data = pd.read_csv(test_file_path)
test_ids = test_data['id']


# Cek korelasi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def find_high_correlation_pairs(df, threshold=0.5):
    """
    Finds and prints pairs of columns with a correlation coefficient
    above a specified threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The correlation threshold (between 0 and 1).
                           Defaults to 0.8.

    Returns:
        pd.DataFrame: A DataFrame containing the highly correlated pairs
                      and their correlation values.
    """
    # --- 1. Calculate the correlation matrix for numeric columns only ---
    # .corr() computes pairwise correlation of columns.
    # We use numeric_only=True to automatically ignore non-numeric columns.
    corr_matrix = df.corr(numeric_only=True)

    # --- 2. Unstack the matrix to get a list of all pairs ---
    # .unstack() creates a Series with a MultiIndex, making it easy to iterate through pairs.
    correlated_pairs = corr_matrix.unstack()

    # --- 3. Convert to a DataFrame and clean up ---
    # Reset index to turn the multi-index into columns
    correlated_pairs_df = correlated_pairs.reset_index()
    correlated_pairs_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # --- 4. Remove self-correlations and duplicates ---
    # Remove pairs where a variable is correlated with itself (Correlation == 1)
    correlated_pairs_df = correlated_pairs_df[correlated_pairs_df['Variable 1'] != correlated_pairs_df['Variable 2']]

    # Create a unique key for each pair by sorting the variable names,
    # so (A, B) is treated the same as (B, A).
    correlated_pairs_df['pair_key'] = correlated_pairs_df.apply(
        lambda row: tuple(sorted((row['Variable 1'], row['Variable 2']))), axis=1
    )
    # Drop the duplicate pairs
    correlated_pairs_df = correlated_pairs_df.drop_duplicates(subset=['pair_key'])
    correlated_pairs_df = correlated_pairs_df.drop('pair_key', axis=1) # Drop the helper column

    # --- 5. Filter for pairs with correlation above the threshold ---
    # We use abs() to find both strong positive and strong negative correlations.
    high_corr_df = correlated_pairs_df[abs(correlated_pairs_df['Correlation']) > threshold].sort_values(
        by='Correlation', ascending=False
    )

    return high_corr_df

def visualize_correlation_matrix(df):
    """
    Generates and displays a heatmap of the correlation matrix.
    """
    # --- 1. Calculate the correlation matrix ---
    corr_matrix = df.corr(numeric_only=True)

    # --- 2. Create the heatmap ---
    plt.figure(figsize=(18, 15)) # Set a larger figure size for better readability
    sns.heatmap(
        corr_matrix,
        annot=False, # Set to True if you have few columns and want to see values
        cmap='coolwarm', # Use a diverging colormap (red=positive, blue=negative)
        linewidths=0.5
    )
    plt.title('Heatmap of Feature Correlation', fontsize=16)
    plt.show()

correlation_threshold = 0.5
high_correlation_pairs = find_high_correlation_pairs(train_data, threshold=correlation_threshold)

if high_correlation_pairs.empty:
    print("No pairs found above the specified threshold.")
else:
    print(high_correlation_pairs.to_string(index=False))
    visualize_correlation_matrix(train_data)


def preprocess(train_data):
    train_data_subset = train_data.copy()

    # Ganti missing values int dan float jadi median
    imputer = SimpleImputer(strategy='median')
    num_cols = train_data_subset.select_dtypes(include=['float64', 'int64']).columns
    train_data_subset[num_cols] = imputer.fit_transform(train_data_subset[num_cols])

    # one hot encoding untuk room type
    train_data_processed = pd.get_dummies(train_data_subset, columns=['room_type'], drop_first=True)

    # Drop atribut yang tidak terpakai
    X = train_data_processed.drop(['id', 'name', 'description', 'neighborhood_overview', 'host_id',
                                  'host_name', 'host_location', 'host_about',
                                  'host_response_time', 'host_is_superhost', 'host_neighbourhood',
                                  'host_total_listings_count', 'host_verifications',
                                  'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
                                  'neighbourhood_cleansed', 'property_type', 'amenities', 'bathrooms_text', 'number_of_reviews', 'number_of_reviews_ltm',
                                  'number_of_reviews_l30d',
                                  'availability_eoy', 'number_of_reviews_ly', 'estimated_occupancy_l365d',
                                  'first_review',
                                  'review_scores_accuracy',
                                  'review_scores_cleanliness', 'review_scores_checkin',
                                  'review_scores_communication', 'review_scores_location',
                                  'review_scores_value', 'reviews_per_month', 'city'], axis=1)

    #Ubah tanggal menjadi datetime lalu menjadi float
    reference_date = pd.to_datetime('2025-08-02')
    X['host_since'] = pd.to_datetime(X['host_since'])
    X['last_review'] = pd.to_datetime(X['last_review'])
    X['host_since'] = (reference_date - X['host_since']).dt.days
    X['last_review'] = (reference_date - X['last_review']).dt.days

    #Ubah persentase menjadi float
    X['host_response_rate'] = X['host_response_rate'].str.rstrip('%').astype('float') / 100.0
    X['host_acceptance_rate'] = X['host_acceptance_rate'].str.rstrip('%').astype('float') / 100.0

    #Isi NaN dengan 0
    X.fillna({'host_response_rate': 0}, inplace=True)
    X.fillna({'host_acceptance_rate': 0}, inplace=True)
    X.fillna({'has_availability' : 0}, inplace=True)
    X.fillna({'last_review': 0}, inplace=True)
    X.fillna({'host_since': 0}, inplace=True)

    # Konversi true false menjadi 1 dan 0
    boolean_map = {
        't': 1, 'f': 0,
        'TRUE': 1, 'FALSE': 0,
        True: 1, False: 0
    }
    for col in ['has_availability', 'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room']:
        X[col] = X[col].replace(boolean_map)

    return X

X = preprocess(train_data)
test_preprocessed = preprocess(test_data)
# Define the target variable (price) and features
y = X['price']
X = X.drop(columns=['price'])

# Align columns - crucial for making predictions
# This ensures the test set has the same columns as the training set
train_cols = X.columns
X_test = X.copy()
X_test = X_test.reindex(columns=train_cols, fill_value=0)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Engineering done
# Buat ngecek hasil preproses
# X_head = (X_train.head()).to_csv('x_head.csv', index=False)
# test_head = (test_data.head()).to_csv('test_head.csv', index=False)

print(f"Training data has {X.shape[1]} features.")
print(f"Test data has {X_test.shape[1]} features.")


# Pemilihan Model
from sklearn.ensemble import RandomForestRegressor

# Instantiate the Random Forest model
# n_estimators is the number of trees in the forest. 100 is a good starting point.
rf_model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Prediksi dengan RandomForestRegressor
test_predictions = rf_model.predict(X_test)

# Evaluasi Model
print("Original Test Prices (y_test):")
print(y_test.values)

print("\nModel's Predicted Prices:")
print(test_predictions)

# Hitung RMSE
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Hitung RMSE untuk data test yang telah dipreproses
test_predictions_final = rf_model.predict(test_preprocessed)
submission_df = pd.DataFrame({'id': test_ids, 'price': test_predictions_final})

submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")


# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

# # Import necessary libraries for neural network regression
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input

# # 1. Scale the features (Essential for Neural Networks)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # Also scale the final test data for submission
# test_preprocessed_scaled = scaler.transform(test_preprocessed)

# # 2. Define the Neural Network architecture
# nn_model = Sequential([
#     # Input layer: needs to know the number of features
#     Input(shape=(X_train_scaled.shape[1],)),
#     # Hidden layer 1: 64 neurons, 'relu' is a standard activation function
#     Dense(64, activation='relu'),
#     # Hidden layer 2: 32 neurons
#     Dense(32, activation='relu'),
#     # Output layer: 1 neuron for the single price prediction, 'linear' activation for regression
#     Dense(1, activation='linear')
# ])

# # 3. Compile the model
# # We define the optimizer, and the loss function to minimize (mean squared error for regression)
# nn_model.compile(optimizer='adam', loss='mean_squared_error')

# # 4. Train the model
# # epochs = number of times the model will cycle through the data
# # batch_size = number of samples per gradient update
# nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# # 5. Make predictions on the scaled validation set
# # .predict() returns a 2D array, so we use .flatten() to make it 1D for evaluation
# test_predictions = nn_model.predict(X_test_scaled).flatten()

# # Evaluasi Model
# print("Original Test Prices (y_test):")
# print(y_test.values)

# print("\nModel's Predicted Prices:")
# print(test_predictions)

# # Hitung RMSE
# mse = mean_squared_error(y_test, test_predictions)
# rmse = np.sqrt(mse)
# print(f"RMSE: {rmse}")

# # 6. Make final predictions on the scaled test set for submission
# test_predictions_final = nn_model.predict(test_preprocessed_scaled).flatten()
# submission_df = pd.DataFrame({'id': test_ids, 'price': test_predictions_final})

# submission_df.to_csv('/kaggle/working/submission.csv', index=False)

# print("Submission file 'submission.csv' created successfully!")

