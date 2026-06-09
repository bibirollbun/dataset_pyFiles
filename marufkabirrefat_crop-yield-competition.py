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


import pandas as pd
import os

# Define the path to your data directory
data_directory = '/kaggle/input/agriyield-2025'

# 1. List the files in the directory to find the correct filename
files = os.listdir(data_directory)
print(f"Files found in the directory: {files}")

# 2. Construct the full path to the CSV file
#    Replace 'your_file_name.csv' with the actual filename printed above
#    For example, if the output is ['yield_data.csv'], use that name.
try:
    # Assuming there's only one CSV file, let's build the path automatically
    csv_file_name = [f for f in files if f.endswith('.csv')][0]
    full_path = os.path.join(data_directory, csv_file_name)
    print(f"Reading file from: {full_path}")
    
    # 3. Read the CSV file into a DataFrame
    df = pd.read_csv(full_path)
    
    # Display the first few rows of the dataframe
    print("\nSuccessfully loaded data:")
    display(df.head())

except IndexError:
    print("\nError: No CSV file found in the directory.")
except Exception as e:
    print(f"\nAn error occurred: {e}")


# Install necessary libraries if they are not already in the Kaggle environment
!pip install shap catboost -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
import catboost as cbt

# Explainable AI
import shap
import lime
import lime.lime_tabular

print("All libraries imported successfully!")


# Load data from Kaggle's default input directory
try:
    # Update this path if your dataset has a different folder name
    train_df = pd.read_csv('/kaggle/input/train.csv')
    test_df = pd.read_csv('/kaggle/input/test.csv')
except FileNotFoundError:
    print("Please make sure your data is in the '/kaggle/input/<your-dataset-folder>/' directory.")
    # Fallback for demonstration if files are not found
    train_df = pd.DataFrame(np.random.rand(1000, 9), columns=['field_id', 'soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi', 'yield'])
    test_df = pd.DataFrame(np.random.rand(500, 8), columns=['field_id', 'soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi'])

# --- Preprocessing ---
# Convert 'yield' (a regression target) to a binary classification target
median_yield = train_df['yield'].median()
train_df['yield_class'] = (train_df['yield'] > median_yield).astype(int)
print(f"Target variable 'yield' converted to binary 'yield_class' with median: {median_yield:.2f}")

# Separate features (X) and target (y)
features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']
X = train_df[features]
y = train_df['yield_class']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape data for LSTM/TCN models (which require a 3D input)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

print("Data preprocessing complete.")
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)


# ==============================================================================
results_list = []

def evaluate_model(name, y_true, y_pred_probs, is_dl=False):
    """Calculates metrics and prints results for a model."""
    y_pred = (y_pred_probs > 0.5).astype("int32") if is_dl else y_pred_probs
    
    print(f"\n--- {name} Evaluation ---")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    
    results_list.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1 Score': f1
    })



print("\nTraining LSTM...")
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
lstm_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
evaluate_model("LSTM", y_val, lstm_model.predict(X_val_reshaped), is_dl=True)



print("\nTraining DBN (Deep MLP)...")
dbn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3), Dense(64, activation='relu'), Dropout(0.3), Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
dbn_model.compile(optimizer='adam', loss='binary_crossentropy')
dbn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
evaluate_model("DBN", y_val, dbn_model.predict(X_val), is_dl=True)


print("\nTraining TCN (1D CNN)...")
tcn_model = Sequential([
    Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    MaxPooling1D(pool_size=1), Flatten(), Dense(16, activation='relu'), Dense(1, activation='sigmoid')
])
tcn_model.compile(optimizer='adam', loss='binary_crossentropy')
tcn_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
evaluate_model("TCN", y_val, tcn_model.predict(X_val_reshaped), is_dl=True)



ml_models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Multilayer Perceptron": MLPClassifier(random_state=42, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "CatBoost": cbt.CatBoostClassifier(verbose=0, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Neural Network (NN)": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
}

for name, model in ml_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    evaluate_model(name, y_val, model.predict(X_val))

print("\nAll models trained and evaluated successfully.")


print("\n--- Generating Explainable AI Plots ---")

# We will use CatBoost as it's a strong performer and works well with SHAP
model_for_explanation = ml_models['CatBoost']


explainer = shap.TreeExplainer(model_for_explanation)
shap_values = explainer.shap_values(X_val)
print("Displaying SHAP Summary Plot...")
plt.title("SHAP Feature Importance")
shap.summary_plot(shap_values, X_val, feature_names=features, show=False)
plt.show()


lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=features, class_names=['Low Yield', 'High Yield'], mode='classification'
)
# Explain a single instance from the validation set
instance_idx = 50
lime_explanation = lime_explainer.explain_instance(X_val[instance_idx], model_for_explanation.predict_proba, num_features=len(features))
print("\nDisplaying LIME explanation for a single prediction...")
lime_explanation.show_in_notebook(show_table=True)


df_results = pd.DataFrame(results_list).set_index('Model').sort_values(by='F1 Score', ascending=False)

# Apply styling to the DataFrame
styled_df = df_results.style.format("{:.3f}") \
    .background_gradient(cmap='Blues') \
    .set_properties(**{
        'font-family': 'Arial, sans-serif',
        'border': '1.5px solid black',
        'text-align': 'center'
    }) \
    .set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#1c2e47'), 
            ('color', 'white'),
            ('font-weight', 'bold')
        ]},
        {'selector': 'tbody th', 'props': [('font-weight', 'bold')]},
        {'selector': 'tbody tr:hover', 'props': [('background-color', '#a8d1ff')]}
    ]) \
    .set_caption("<h2>ðŸ“Š Final Model Performance Comparison</h2>")


print("\n" + "="*50)
print("           FINAL MODEL COMPARISON RESULTS")
print("="*50)
display(styled_df)

