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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')


TRAIN_FILE_PATH = '/kaggle/input/playground-series-s5e5/train.csv'
TEST_FILE_PATH = '/kaggle/input/playground-series-s5e5/test.csv'
SUBMISSION_FILE_PATH = "submission.csv"


MODEL_FEATURES_FOR_CLUSTERING = ['Duration', 'Heart_Rate', 'Body_Temp']
TARGET_COLUMN = 'Calories'
RANDOM_STATE = 42


def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
        if 'Height' in df.columns and 'Weight' in df.columns and 'Age' in df.columns and 'Sex' in df.columns:
            # Calculate BMI (Height is in cm, convert to meters)
            df['BMI'] = df['Weight'] / (df['Height'] / 100)**2
            # Calculate BMR using the Mifflin-St Jeor equation
            df['BMR'] = np.where(df['Sex'] == 0,
                                 (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) + 5,
                                 (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) - 161)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as your script.")
        return None



def feature_engineer_with_clustering(df, features_to_cluster, n_clusters=3):
    if df is None:
        return None, None
    print(f"\n--- Applying KMeans Clustering with {n_clusters} clusters ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
    df['Cluster_ID'] = kmeans.fit_predict(df[features_to_cluster])
    print("Clustering complete. 'Cluster_ID' feature added.")
    return df, kmeans


def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, target_scaler):
    print(f"\n--- Training and Evaluating {model_name} ---")
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_pred_original_scale = target_scaler.inverse_transform(y_pred_scaled)
    y_pred_original_scale = np.maximum(y_pred_original_scale, 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_original_scale))
    rmsle_score = rmsle(y_test, y_pred_original_scale)
    r2 = r2_score(y_test, y_pred_original_scale)
    individual_sq_log_error = (np.log1p(y_test.values.flatten()) - np.log1p(y_pred_original_scale.flatten())) ** 2
    print(f"{model_name} Results:")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Root Mean Squared Logarithmic Error (RMSLE): {rmsle_score:.4f}")
    print(f"  R-squared: {r2:.4f}")
    return {
        'model': model,
        'name': model_name,
        'rmse': rmse,
        'rmsle': rmsle_score,
        'r2': r2,
        'y_pred': y_pred_original_scale,
        'individual_sq_log_error': individual_sq_log_error
    }


def main():
    df_train = load_and_preprocess_data(TRAIN_FILE_PATH)
    if df_train is None:
        return
    features_for_clustering = MODEL_FEATURES_FOR_CLUSTERING
    df_train, kmeans_model = feature_engineer_with_clustering(df_train, features_for_clustering)
    if df_train is None:
        return
    MODEL_FEATURES = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Height', 'Weight', 'Sex', 'BMI', 'BMR', 'Cluster_ID']
    print(f"Updated features for the model: {MODEL_FEATURES}")
    X_raw = df_train[MODEL_FEATURES]
    y_raw = df_train[[TARGET_COLUMN]]
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE
    )
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    X_test_scaled = feature_scaler.transform(X_test_raw)
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    print("\n--- Data Scaling Complete ---")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"y_train_scaled shape: {y_train_scaled.shape}")
    models_to_evaluate = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }
    results = {}
    for name, model in models_to_evaluate.items():
        results[name] = train_and_evaluate_model(
            model, name, X_train_scaled, y_train_scaled.ravel(), X_test_scaled, y_test_raw, target_scaler
        )
    print("\n--- Model Performance Comparison ---")
    best_model_name = min(results, key=lambda k: results[k]['rmsle'])
    best_model_info = results[best_model_name]
    for name, info in results.items():
        print(f"\n{name}:")
        print(f"  RMSE: {info['rmse']:.4f}")
        print(f"  RMSLE: {info['rmsle']:.4f}")
        print(f"  R-squared: {info['r2']:.4f}")
    print(f"\n--- The best performing model based on RMSLE is: {best_model_name} ---")
    print("\n--- Individual Squared Logarithmic Errors (Sample) ---")
    print("This shows the prediction error for each data point after logging and squaring.")
    individual_errors_df = pd.DataFrame({
        'Actual_Calories': y_test_raw.values.flatten(),
        'Predicted_Calories': best_model_info['y_pred'].flatten(),
        'Individual_Sq_Log_Error': best_model_info['individual_sq_log_error']
    })
    print(individual_errors_df.head())
    print("\n--- Generating Visualization ---")
    df_full_for_plot = load_and_preprocess_data(TRAIN_FILE_PATH)
    df_full_for_plot['Cluster_ID'] = kmeans_model.predict(df_full_for_plot[features_for_clustering])
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(MODEL_FEATURES):
        plt.subplot(4, 3, i + 1)
        sns.scatterplot(x=col, y="Calories", data=df_full_for_plot, hue='Cluster_ID', palette='viridis', alpha=0.6)
        plt.title(f'{col} vs. Calories (Colored by Cluster)')
        plt.xlabel(col)
        plt.ylabel('Calories')
    plt.tight_layout()
    plt.show()
    print("\n--- Generating Submission File ---")
    test_df = load_and_preprocess_data(TEST_FILE_PATH)
    if test_df is None:
        return
    test_df['Cluster_ID'] = kmeans_model.predict(test_df[features_for_clustering])
    X_test_submission_raw = test_df[MODEL_FEATURES]
    X_test_submission_scaled = feature_scaler.transform(X_test_submission_raw)
    best_model = best_model_info['model']
    test_predict_scaled = best_model.predict(X_test_submission_scaled)
    if test_predict_scaled.ndim == 1:
        test_predict_scaled = test_predict_scaled.reshape(-1, 1)
    test_predict_original = target_scaler.inverse_transform(test_predict_scaled)
    test_predict_original = np.maximum(test_predict_original, 0)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Calories': test_predict_original.flatten()
    })
    submission.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission file saved to {SUBMISSION_FILE_PATH}")
    joblib.dump(best_model, f'{best_model_name.replace(" ", "_").lower()}_model.pkl')
    joblib.dump(feature_scaler, 'feature_minmax_scaler.pkl')
    joblib.dump(target_scaler, 'target_minmax_scaler.pkl')
    joblib.dump(kmeans_model, 'kmeans_clusterer.pkl')
    print("\nModel and scalers saved.")


if __name__ == "__main__":
    main()

