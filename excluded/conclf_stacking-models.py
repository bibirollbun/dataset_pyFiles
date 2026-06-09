import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim



# Custom PCA-based threshold classifier
class PC2ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=3)
        X_pca = self.pca.fit_transform(X_scaled)
        pc2 = X_pca[:, 1]
        thresholds = np.linspace(pc2.min(), pc2.max(), 200)
        accuracies = [(y == (pc2 < t)).mean() for t in thresholds]
        self.best_threshold = thresholds[np.argmax(accuracies)]
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        pc2 = self.pca.transform(X_scaled)[:, 1]
        proba = (pc2 < self.best_threshold).astype(float)
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5

# PyTorch MLP wrapped for scikit-learn compatibility
class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, lr=5e-4):
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 4), nn.ReLU(),
            nn.Linear(4, 1)
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            probs = torch.sigmoid(self.model(X)).numpy()
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5




import os
data_path = "/kaggle/input/playground-series-s5e3"

# Load data
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

X_train = train_df.drop(columns=["id", "day", "rainfall"])
y_train = train_df["rainfall"]

X_test = test_df.drop(columns=["id", "day"])
# Impute NaN with mean in test data
X_test = X_test.fillna(X_test.mean())
test_ids = test_df["id"]

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




from sklearn.impute import KNNImputer

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
train_extra=pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
submission100 = pd.read_csv("/kaggle/input/cp-sat-ensemble-100/submission.csv")
train_extra.columns = train_extra.columns.str.replace(' ', '')
train_extra = train_extra[train_extra.columns].copy()
train_extra['rainfall'] = train_extra['rainfall'].map({'no': 0, 'yes': 1})
train_extra['humidity']=train_extra['humidity'].astype(float)
train_extra['cloud']=train_extra['cloud'].astype(float)
train_features=list(train)
train_extra=train_extra[train_features]

train = pd.concat([train, train_extra], axis=0, ignore_index=True)
train = train.drop_duplicates()
test['winddirection']=test['winddirection'].fillna(value=test['winddirection'].mean())
train['winddirection']=train['winddirection'].fillna(value=train['winddirection'].mean())
train['windspeed']=train['windspeed'].fillna(value=train['windspeed'].mean()) 

def feature_engineering(df):
    """
    Create new features based on meteorological understanding and data analysis,
    with 'day' representing day of the year (1-365).
    Ensures no data leakage by avoiding use of the target variable (rainfall).
    """
    # Make a copy to avoid modifying the original dataframe
    enhanced_df = df.copy()
    
    # 1. temparature range (difference between max and min temparatures)
    enhanced_df['temp_range'] = enhanced_df['maxtemp'] - enhanced_df['mintemp']
    
    # 2. Dew point depression (difference between temparature and dew point)
    enhanced_df['dewpoint_depression'] = enhanced_df['temparature'] - enhanced_df['dewpoint']
    
    # 3. Pressure change from previous day
    enhanced_df['pressure_change'] = enhanced_df['pressure'].diff().fillna(0)
    
    # 4. Humidity to dew point ratio
    enhanced_df['humidity_dewpoint_ratio'] = enhanced_df['humidity'] / enhanced_df['dewpoint'].clip(lower=0.1)
    
    # 5. Cloud coverage to sunshine ratio (inverse relationship)
    enhanced_df['cloud_sunshine_ratio'] = enhanced_df['cloud'] / enhanced_df['sunshine'].clip(lower=0.1)
    
    # 6. Wind intensity factor (combination of speed and humidity)
    enhanced_df['wind_humidity_factor'] = enhanced_df['windspeed'] * (enhanced_df['humidity'] / 100)
    
    # 7. temparature-humidity index (simple version of heat index)
    enhanced_df['temp_humidity_index'] = (0.8 * enhanced_df['temparature']) + \
                                        ((enhanced_df['humidity'] / 100) * \
                                        (enhanced_df['temparature'] - 14.3)) + 46.4
    
    # 8. Pressure change rate (acceleration)
    enhanced_df['pressure_acceleration'] = enhanced_df['pressure_change'].diff().fillna(0)
    
    # 9. Seasonal features (based on day of year)
    # Convert day to month (1-365 to 1-12)
    enhanced_df['month'] = ((enhanced_df['day'] - 1) // 30) + 1
    enhanced_df['month'] = enhanced_df['month'].clip(upper=12)  # Ensure month doesn't exceed 12
    
    # 10. Convert day to season (1-365 to 1-4)
    enhanced_df['season'] = ((enhanced_df['month'] - 1) // 3) + 1
    
    # 11. Sine and cosine transformations to capture cyclical nature of days in a year
    enhanced_df['day_of_year_sin'] = np.sin(2 * np.pi * enhanced_df['day'] / 365)
    enhanced_df['day_of_year_cos'] = np.cos(2 * np.pi * enhanced_df['day'] / 365)
    
    # 12. Rolling averages for key meteorological variables
    for window in [3, 7, 14]:
        enhanced_df[f'temparature_rolling_{window}d'] = enhanced_df['temparature'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'pressure_rolling_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'humidity_rolling_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'cloud_rolling_{window}d'] = enhanced_df['cloud'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'windspeed_rolling_{window}d'] = enhanced_df['windspeed'].rolling(window=window, min_periods=1).mean()
    
    # 13. Weather pattern change features
    # temparature trend
    enhanced_df['temp_trend_3d'] = enhanced_df['temparature'].diff(3).fillna(0)
    # Pressure trend
    enhanced_df['pressure_trend_3d'] = enhanced_df['pressure'].diff(3).fillna(0)
    # Humidity trend
    enhanced_df['humidity_trend_3d'] = enhanced_df['humidity'].diff(3).fillna(0)
    
    # 14. Extreme weather indicators
    enhanced_df['extreme_temp'] = (enhanced_df['temparature'] > enhanced_df['temparature'].quantile(0.95)) | \
                                 (enhanced_df['temparature'] < enhanced_df['temparature'].quantile(0.05))
    enhanced_df['extreme_temp'] = enhanced_df['extreme_temp'].astype(int)
    
    enhanced_df['extreme_humidity'] = (enhanced_df['humidity'] > enhanced_df['humidity'].quantile(0.95)) | \
                                     (enhanced_df['humidity'] < enhanced_df['humidity'].quantile(0.05))
    enhanced_df['extreme_humidity'] = enhanced_df['extreme_humidity'].astype(int)
    
    enhanced_df['extreme_pressure'] = (enhanced_df['pressure'] > enhanced_df['pressure'].quantile(0.95)) | \
                                     (enhanced_df['pressure'] < enhanced_df['pressure'].quantile(0.05))
    enhanced_df['extreme_pressure'] = enhanced_df['extreme_pressure'].astype(int)
    
    # 15. Interaction terms between key variables
    enhanced_df['temp_humidity_interaction'] = enhanced_df['temparature'] * enhanced_df['humidity']
    enhanced_df['pressure_wind_interaction'] = enhanced_df['pressure'] * enhanced_df['windspeed']
    enhanced_df['cloud_sunshine_interaction'] = enhanced_df['cloud'] * enhanced_df['sunshine']
    enhanced_df['dewpoint_humidity_interaction'] = enhanced_df['dewpoint'] * enhanced_df['humidity']
    
    # 16. Moving standard deviations for measuring variability
    for window in [7, 14]:
        enhanced_df[f'temp_std_{window}d'] = enhanced_df['temparature'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'pressure_std_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'humidity_std_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=4).std().fillna(0)
    
    return enhanced_df
imputer = KNNImputer(n_neighbors=5)
test["winddirection"] = imputer.fit_transform(test[["winddirection"]])

train_df = feature_engineering(train)
test_df = feature_engineering(test)

x = train_df.drop('rainfall',axis=1)
y = train_df["rainfall"]

sc = StandardScaler().fit(x.values)
x = pd.DataFrame(sc.transform(x), index=x.index, columns=x.columns)
test_df = pd.DataFrame(sc.transform(test_df), index=test_df.index, columns=test_df.columns)

# Apply oversampling
# ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
# X_resampled, y_resampled = ros.fit_resample(x, y)

# print(X_resampled.shape)
# print(y_resampled.shape)

# x2 = pd.DataFrame(X_resampled, columns = train_df.drop(columns = ["rainfall"]).columns)
# y2 = pd.DataFrame(y_resampled, columns = ["rainfall"])

# X_train, X_val, y_train, y_val = train_test_split(x2, y2, test_size=0.2, random_state=42)

# y_test = np.asarray(submission100['rainfall']).astype(int)


from sklearn.decomposition import PCA

pca = PCA(n_components = 3)

transformed = pca.fit_transform(x)
transformed



import plotly.express as px


fig = px.scatter_3d(x = transformed[:,0], y = transformed[:,1], z = transformed[:,2], color=y)
fig


# Define base and meta models
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('pca', PC2ThresholdClassifier()),
    ('mlp', PyTorchMLPClassifier(epochs=500))
]

meta_model = LogisticRegression(max_iter=1000)

# Stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True, cv=5)

# Fit stacking model
stacking_model.fit(np.array(x), y)



def predict(stacking_model, X_test_scaled, test_ids):
    # Predict on test set
    predictions = stacking_model.predict_proba(X_test_scaled)[:,1]
    print(predictions)
    
    # Prepare submission
    submission_df = pd.DataFrame({
        "id": test_ids,
        "rainfall": predictions
    })
    submission_df.to_csv("submission.csv", index=False)
    
    print("Submission file 'submission.csv' created.")




predict(stacking_model, np.array(test_df), test_df.index)

