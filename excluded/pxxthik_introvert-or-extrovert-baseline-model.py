import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!pip install mlflow==2.15.0  > /dev/null 2>&1
!pip install dagshub==0.3.34  > /dev/null 2>&1


import mlflow
import mlflow.sklearn
import dagshub
import os

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
dagshub_token = user_secrets.get_secret("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


dagshub_url = "https://dagshub.com"
repo_owner = "pxxthik"
repo_name = "Predict-the-Introverts-from-the-Extroverts"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


with mlflow.start_run():
    mlflow.log_param("Parameter", "value")
    mlflow.log_metric("Metric", 1)


import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


df = df.drop_duplicates()
df = df.dropna()


df.sample(5)


categorical_columns = ['Stage_fear', 'Drained_after_socializing', 'Personality']

# Apply Ordinal Encoding
encoder = OrdinalEncoder()

# Encoding categorical columns
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])


# Split the dataset into features (X) and target (y)
X = df.drop(columns=['id', 'Personality'])  # Drop non-features and target columns
y = df['Personality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Track the model with MLflow
mlflow.set_experiment("baseline model")
with mlflow.start_run():
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log parameters (e.g., model type)
    mlflow.log_param("model_type", "LogisticRegression")

    print("Model and metrics logged to MLflow.")


encoder.transform([["Yes", "Yes","Extrovert"]])


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PCA for dimensionality reduction to 2D or 3D
pca = PCA(n_components=3)  # You can change this to 2 if you want 2D visualization
X_pca = pca.fit_transform(X)

# PCA for dimensionality reduction to 2D or 3D
pca = PCA(n_components=3)  # You can change this to 2 if you want 2D visualization
X_pca = pca.fit_transform(X)

# Create a 3D plot for PCA
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(121, projection='3d')
scatter_pca = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
cbar = plt.colorbar(scatter_pca)
cbar.set_label('Personality (Encoded)')


import plotly.graph_objects as go

# Create the 3D scatter plot with Plotly
fig = go.Figure(data=go.Scatter3d(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    z=X_pca[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=y,  # Color by the encoded personality values
        colorscale='Viridis',  # Color scale (you can choose other scales like 'Cividis', 'Jet', etc.)
        colorbar=dict(title='Personality (Encoded)', tickvals=[0, 1, 2]),  # Customize colorbar
    ),
))

# Update the layout with axis labels and title
fig.update_layout(
    title='PCA: 3D Visualization of Personality and Features',
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    margin=dict(l=0, r=0, b=0, t=40)  # Margin adjustments
)

# Show the plot
fig.show()


!pip install mlflow==2.15.0  > /dev/null 2>&1
!pip install dagshub==0.3.34  > /dev/null 2>&1




