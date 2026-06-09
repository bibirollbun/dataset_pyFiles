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


target_columns = ['Id', 'companyID', 'corporateTariffCode', 'frequentFlyer',
       'nationality', 'isAccess3D', 'isVip', 'legs0_arrivalAt',
       'legs0_departureAt', 'legs0_duration', 'legs0_segments0_aircraft_code',
       'legs0_segments0_arrivalTo_airport_city_iata',
       'legs0_segments0_arrivalTo_airport_iata',
       'legs0_segments0_baggageAllowance_quantity',
       'legs0_segments0_baggageAllowance_weightMeasurementType',
       'legs0_segments0_cabinClass',
       'legs0_segments0_departureFrom_airport_iata',
       'legs0_segments0_duration', 'legs0_segments0_flightNumber',
       'legs0_segments0_marketingCarrier_code',
       'legs0_segments0_operatingCarrier_code',
       'legs0_segments0_seatsAvailable', 'legs0_segments1_aircraft_code',
       'legs0_segments1_arrivalTo_airport_city_iata',
       'legs0_segments1_arrivalTo_airport_iata',
       'legs0_segments1_baggageAllowance_quantity',
       'legs0_segments1_baggageAllowance_weightMeasurementType',
       'legs0_segments1_cabinClass',
       'legs0_segments1_departureFrom_airport_iata',
       'legs0_segments1_duration', 'legs0_segments1_flightNumber',
       'legs0_segments1_marketingCarrier_code',
       'legs0_segments1_operatingCarrier_code',
       'legs0_segments1_seatsAvailable', 'legs1_arrivalAt',
       'legs1_departureAt', 'legs1_duration', 'legs1_segments0_aircraft_code',
       'legs1_segments0_arrivalTo_airport_city_iata',
       'legs1_segments0_arrivalTo_airport_iata',
       'legs1_segments0_baggageAllowance_quantity',
       'legs1_segments0_baggageAllowance_weightMeasurementType',
       'legs1_segments0_cabinClass',
       'legs1_segments0_departureFrom_airport_iata',
       'legs1_segments0_duration', 'legs1_segments0_flightNumber',
       'legs1_segments0_marketingCarrier_code',
       'legs1_segments0_operatingCarrier_code',
       'legs1_segments0_seatsAvailable', 'miniRules0_monetaryAmount',
       'miniRules0_statusInfos', 'miniRules1_monetaryAmount',
       'miniRules1_statusInfos', 'pricingInfo_isAccessTP', 'profileId',
       'ranker_id', 'searchRoute', 'sex', 'taxes', 'totalPrice',
       'requestDate_year', 'requestDate_month', 'requestDate_day',
       'requestDate_hour', 'requestDate_dow']


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import time
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
MODEL_TYPE = "xgboost"  # Options: "xgboost" or "lightgbm"
USE_GPU = True  # Set to True to enable GPU acceleration

def train_and_save_model(encoded_train_file, model_output_file):
    """Train model on already encoded dataset and save for later use"""
    # Load already encoded training data
    print("Loading encoded training data...")
    df = pd.read_csv(encoded_train_file)
    print(f"Data shape: {df.shape}")
    
    # Prepare data
    X = df.drop(columns=['selected', 'Id', 'ranker_id'])  # Features
    y = df['selected']  # Target
    
    # Handle class imbalance
    class_counts = y.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"\nClass distribution:\n{class_counts}")
    print(f"Scale positive weight: {scale_pos_weight:.2f}")
    
    # Initialize model with GPU support if available
    if MODEL_TYPE == "xgboost":
        print("\nTraining XGBoost model on full dataset...")
        gpu_params = {'tree_method': 'gpu_hist', 'gpu_id': 0} if USE_GPU else {'tree_method': 'hist'}
        model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **gpu_params
        )
    elif MODEL_TYPE == "lightgbm":
        print("\nTraining LightGBM model on full dataset...")
        from lightgbm import LGBMClassifier
        gpu_params = {'device': 'gpu'} if USE_GPU else {}
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            is_unbalance=True,
            **gpu_params
        )
    
    # Train model on full dataset
    print(f"Using GPU: {USE_GPU}")
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_n = min(30, len(feature_importance))
    
    # Create and save importance dataframe
    importance_df = pd.DataFrame({
        'Feature': X.columns[sorted_idx][:top_n],
        'Importance': feature_importance[sorted_idx][:top_n]
    })
    importance_df.to_csv('feature_importance.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importance ({MODEL_TYPE.upper()})')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Save model
    joblib.dump(model, model_output_file)
    print(f"\nModel saved to {model_output_file}")
    print("Feature importance saved to feature_importance.csv and feature_importance.png")
    
    return model

# def predict_and_rank(model, encoded_test_file, output_file):
#     """Make predictions on encoded test set and rank flights, outputting with original values"""
#     # Load the encoded test data (already preprocessed)
#     print("Loading encoded test data...")
#     encoded_test_df = pd.read_csv(encoded_test_file)
    
#     # Load the encoders
#     print("Loading encoders...")
#     try:
#         encoders = joblib.load('/kaggle/input/encoded-test-dataset/label_encoders.pkl')
#     except FileNotFoundError:
#         raise FileNotFoundError("Encoder file not found. Please ensure label_encoders.pkl exists")
    
#     # Create a copy for output with just the identifier columns
#     output_df = encoded_test_df[['Id', 'ranker_id']].copy()
    
#     # Make predictions (probability of being selected)
#     print("Making predictions...")
#     # Drop identifier columns for prediction
#     features = encoded_test_df.drop(columns=['Id', 'ranker_id'])
#     # Get probability predictions for class 1 (selected)
#     pred_proba = model.predict_proba(features)[:, 1]
#     encoded_test_df['prediction_score'] = pred_proba
    
#     # Rank flights within each ranker_id group (higher score = better rank)
#     print("Ranking flights within each session...")
#     # Calculate ranks within each group
#     ranks = encoded_test_df.groupby('ranker_id')['prediction_score'].rank(
#         ascending=False, method='first'
#     ).astype(int)
#     output_df['selected'] = ranks
    
#     # Decode identifier columns back to original values (if they were encoded)
#     print("Decoding identifier columns...")
#     for col in ['Id', 'ranker_id']:
#         if col in encoders:
#             le = encoders[col]
#             output_df[col] = le.inverse_transform(output_df[col])
    
#     # Save results with required columns
#     output_df = output_df[['Id', 'ranker_id', 'selected']]
#     output_df.to_csv(output_file, index=False)
#     print(f"\nPredictions saved to {output_file}")
    
#     return output_df
if __name__ == "__main__":
    # Paths configuration
    ENCODED_TRAIN_FILE = "/kaggle/input/chopped-train-data/chopped_cleaned_train.csv"  
    ENCODED_TEST_DATA =  "/kaggle/input/encoded-test-dataset/encoded_test_data.csv"  # Your preprocessed test DataFrame
    OUTPUT_FILE = "flight_rank_predictions.csv"
    MODEL_FILE = f"flight_selection_model_{MODEL_TYPE}.pkl"
    
    # Train and save model
    # model = train_and_save_model(ENCODED_TRAIN_FILE, MODEL_FILE)
    # model =  joblib.load("/kaggle/input/xgboost-2/flight_selection_model_xgboost (1).pkl")
    
    # # Make predictions on test set and rank flights
    # try:
    #     ranked_predictions = predict_and_rank(
    #         model, 
    #         encoded_test_file=ENCODED_TEST_DATA,
    #         output_file=OUTPUT_FILE
    #     )
    #     print("\nSample of ranked predictions:")
    #     print(ranked_predictions.head())
    # except Exception as e:
    #     print(f"\nError processing test set: {str(e)}")


import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_TYPE = "xgboost"  # Options: "xgboost" or "lightgbm"


# Load the pre-trained model
model_path = "/kaggle/input/xgboost-2/flight_selection_model_xgboost (1).pkl"
model = joblib.load(model_path)
print("Model loaded successfully!")


# Load test data
test_data_path = "/kaggle/input/encoded-test-dataset/encoded_test_data.csv"
encoded_test_df = pd.read_csv(test_data_path)
print(f"Test data loaded. Shape: {encoded_test_df.shape}")
display(encoded_test_df.head())


# Load encoders
try:
    encoders = joblib.load('/kaggle/input/encoded-test-dataset/label_encoders.pkl')
    print("Encoders loaded successfully!")
except FileNotFoundError:
    print("Encoder file not found. Please ensure label_encoders.pkl exists")


# Prepare features for prediction (drop ID columns)
features = encoded_test_df.drop(columns=['Id', 'ranker_id'])
print("Features prepared for prediction")


# Make predictions
encoded_test_df['prediction_score'] = model.predict_proba(features)[:, 1]
print("Predictions completed")
display(encoded_test_df[['Id', 'ranker_id', 'prediction_score']].head())


# Rank flights within each group
submission = encoded_test_df[['Id', 'ranker_id']].copy().astype(int)
submission['selected'] = encoded_test_df.groupby('ranker_id')['prediction_score'].rank(
    ascending=False, method='first'
).astype(int)
print("Ranking completed")
display(submission.head())


# Decode ranker_id back to original string format
if 'ranker_id' in encoders:
    submission['ranker_id'] = encoders['ranker_id'].inverse_transform(submission['ranker_id'])
    print("ranker_id decoded back to original strings")
else:
    print("Warning: No encoder found for ranker_id - keeping as is")

# Verify the decoded values
print("\nDecoded submission sample:")
display(submission.head())


# Save final output
output_file = "submission.csv"
submission.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

