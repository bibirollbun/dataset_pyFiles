import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
    return train, test


def add_features(df):
    df['Month'] = ((df['day'] - 1) // 30) + 1
    df['Day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['Day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dew_point_spread'] = df['temparature'] - df['dewpoint']
    df['relative_humidity'] = (df['humidity'] / 100) * df['temparature']
    df['cloud_cover_intensity'] = df['cloud'] * df['sunshine']
    df['wind_u'] = df['windspeed'] * np.cos(df['winddirection'] * np.pi / 180)
    df['wind_v'] = df['windspeed'] * np.sin(df['winddirection'] * np.pi / 180)
    
    # Rolling Mean (Avoiding Data Leakage)
    df['temp_rolling_3'] = df['temparature'].rolling(3, min_periods=1).mean()
    df['humidity_rolling_3'] = df['humidity'].rolling(3, min_periods=1).mean()
    
    # Lag Features (Avoiding Data Leakage)
    df['prev_day_humidity'] = df['humidity'].shift(1)
    df['prev_day_humidity'].fillna(0, inplace=True)
    
    df['temp_humidity'] = df['temparature'] * df['humidity']
    df['temp_windspeed'] = df['temparature'] * df['windspeed']
    df['windspeed_humidity'] = df['windspeed'] * df['humidity']
    df['windspeed_windspeed'] = df['windspeed'] * df['windspeed']
    
    return df


def preprocess_data(train, test):
    numerical_features = [
        'day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
        'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed',
        'Month', 'Day_sin', 'Day_cos', 'temp_range', 'dew_point_spread', 'relative_humidity',
        'cloud_cover_intensity', 'wind_u', 'wind_v', 'temp_rolling_3', 'humidity_rolling_3',
        'prev_day_humidity', 'temp_humidity', 'temp_windspeed', 'windspeed_humidity',
        'windspeed_windspeed'
    ]

    X = train[numerical_features]
    X_test_data = test[numerical_features]
    y = train['rainfall']

    # Train-Test Split (Before Imputation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Handle Missing Values (After Splitting to Avoid Leakage)
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_test_data = imputer.transform(X_test_data)  # For submission dataset

    return X_train, X_test, y_train, y_test, X_test_data, test


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)


def train_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return y_pred_proba


def create_submission(test, y_test_pred):
    submission = pd.DataFrame({'id': test['id'], 'rainfall': y_test_pred})
    submission.to_csv('submission.csv', index=False)
    print("Submission file created successfully!")


# Main Execution
train, test = load_data()

train = add_features(train)
test = add_features(test)

X_train, X_test, y_train, y_test, X_test_imputed, test = preprocess_data(train, test)

X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

# Apply Standard Scaling After SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
X_test_imputed_scaled = scaler.transform(X_test_imputed)  # For submission dataset

best_model = train_logistic_regression(X_train_scaled, y_train_resampled)

y_pred_proba = evaluate_model(best_model, X_test_scaled, y_test)

test_predictions_proba = best_model.predict_proba(X_test_imputed_scaled)[:, 1]
create_submission(test, test_predictions_proba)

