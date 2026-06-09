import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col = 0)
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col = 0)


print(train.shape)
train.head()


train.describe()


train.isna().sum()


test.isna().sum()


def plot_avg(dataset, column):
    # Aggregate data by day (average temperature per day across years)
    if 'day' not in dataset.columns or column not in dataset.columns:
        raise ValueError(f"Dataset must contain 'day' and {column} columns")
    
        # Aggregate data by day (mean and standard deviation)
    daily_avg = dataset.groupby('day')[column].mean()
    daily_std = dataset.groupby('day')[column].std()

    # Set fancy style
    plt.figure(figsize=(12, 6))
    #plt.style.use("seaborn-darkgrid")

    # Plot smoothed temperature curve
    plt.plot(daily_avg.index, daily_avg.values, color='darkblue', linewidth=2, label=f"Smoothed Avg {column}")

    # Fill between ±1 std deviation
    plt.fill_between(daily_avg.index, daily_avg - daily_std, daily_avg + daily_std,
                     color='lightblue', alpha=0.3, label="±1 Std Dev")

    # Highlight seasons
    plt.axvspan(1, 80, color='cyan', alpha=0.1, label="Winter")  # Winter
    plt.axvspan(81, 172, color='lightgreen', alpha=0.1, label="Spring")  # Spring
    plt.axvspan(173, 265, color='orange', alpha=0.1, label="Summer")  # Summer
    plt.axvspan(266, 365, color='brown', alpha=0.1, label="Fall")  # Fall

    # Titles and labels
    plt.xlabel("Day of the Year (1-365)", fontsize=12, fontweight='bold')
    plt.ylabel(f"{column}", fontsize=12, fontweight='bold')
    plt.title(f"Trend of {column} Over a Year", fontsize=14, fontweight='bold', color="darkblue")

    # Grid and legend
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)

    # Show plot
    plt.show()


for col in train.columns:
    if col in ['day', 'rainfall']:
        continue
    plot_avg(train, col)
    


import seaborn as sns

def plot_correlation_matrix(dataset):
    """
    Plots a correlation heatmap to see relationships between numerical features and rainfall.

    Parameters:
    dataset (pd.DataFrame): DataFrame with numerical features including 'rainfall'.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    corr = dataset.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold', color="darkblue")
    plt.show()


plot_correlation_matrix(train)


def plot_rainfall_vs_feature(dataset, feature):
    """
    Creates a boxplot of a numerical feature split by rainfall (0 or 1).

    Parameters:
    dataset (pd.DataFrame): DataFrame with a numerical feature and 'rainfall' column.
    feature (str): Name of the numerical feature.

    Returns:
    None
    """
    if feature not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError(f"Dataset must contain '{feature}' and 'rainfall' columns")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=dataset['rainfall'], y=dataset[feature], hue=dataset['rainfall'], palette=['orange', 'skyblue'])

    plt.xlabel("Rainfall (0 = No Rain, 1 = Rain)", fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.title(f"{feature} Distribution When It Rains vs. No Rain", fontsize=14, fontweight='bold', color="darkblue")
    plt.legend('', frameon=False)
    plt.show()


for col in train.columns:
    if col in ['day', 'rainfall']:
        continue
    plot_rainfall_vs_feature(train, col)


import matplotlib.pyplot as plt
import pandas as pd

def plot_rainfall_seasonality(dataset):
    """
    Plots the probability of rainfall (Rainfall = 1) over the days of the year to check for seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).

    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError("Dataset must contain 'day' and 'rainfall' columns")

    # Compute probability of rainfall per day
    rainfall_prob = dataset.groupby('day')['rainfall'].mean()

    # Plot probability of rainfall over days
    plt.figure(figsize=(12, 6))
    plt.plot(rainfall_prob.index, rainfall_prob.values, marker='', linestyle='-', color='blue', linewidth=2)

    plt.xlabel("Day of the Year (1-365)", fontsize=12, fontweight='bold')
    plt.ylabel("Probability of Rain", fontsize=12, fontweight='bold')
    plt.title("Rainfall Seasonality Check", fontsize=14, fontweight='bold', color="blue")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()


def plot_rolling_rainfall(dataset, window=30):
    """
    Plots a rolling mean of rainfall probability over days to visualize seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).
    window (int): Rolling window size (default = 30 days).

    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError("Dataset must contain 'day' and 'rainfall' columns")

    # Compute rolling mean of rainfall probability
    daily_rainfall = dataset.groupby('day')['rainfall'].mean()
    rolling_mean = daily_rainfall.rolling(window=window, center=False).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_rainfall.index, daily_rainfall.values, linestyle='--', alpha=0.5, label="Daily Probability", color='gray')
    plt.plot(rolling_mean.index, rolling_mean.values, color='blue', linewidth=2, label=f"{window}-Day Rolling Mean")

    # Highlight seasons
    plt.axvspan(1, 80, color='cyan', alpha=0.1, label="Winter")  # Winter
    plt.axvspan(81, 172, color='lightgreen', alpha=0.1, label="Spring")  # Spring
    plt.axvspan(173, 265, color='orange', alpha=0.1, label="Summer")  # Summer
    plt.axvspan(266, 365, color='brown', alpha=0.1, label="Fall")  # Fall


    plt.xlabel("Day of the Year (1-365)", fontsize=12, fontweight='bold')
    plt.ylabel("Probability of Rain", fontsize=12, fontweight='bold')
    plt.title("Rolling Mean of Rainfall Probability", fontsize=14, fontweight='bold', color="darkblue")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()


import numpy as np

def plot_fft_rainfall(dataset):
    """
    Performs a Fast Fourier Transform (FFT) on rainfall probability data to check for seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).

    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError("Dataset must contain 'day' and 'rainfall' columns")

    # Compute rainfall probability per day
    daily_rainfall = dataset.groupby('day')['rainfall'].mean()

    # Apply FFT
    fft_vals = np.fft.fft(daily_rainfall.values)
    fft_freqs = np.fft.fftfreq(len(daily_rainfall))

    # Plot FFT spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(fft_freqs[1:len(fft_freqs)//2], np.abs(fft_vals[1:len(fft_vals)//2]), color='lightblue')

    plt.xlabel("Frequency", fontsize=12, fontweight='bold')
    plt.ylabel("Magnitude", fontsize=12, fontweight='bold')
    plt.title("FFT Spectrum of Rainfall Data", fontsize=14, fontweight='bold', color="darkblue")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


import statsmodels.api as sm

def plot_acf_rainfall(dataset):
    """
    Plots the Autocorrelation Function (ACF) for rainfall probability to check for seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).

    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError("Dataset must contain 'day' and 'rainfall' columns")

    # Compute daily rainfall probability
    daily_rainfall = dataset.groupby('day')['rainfall'].mean()

    # Plot ACF
    sm.graphics.tsa.plot_acf(daily_rainfall, lags=180, alpha=0.05)
    plt.title("Autocorrelation of Rainfall Probability")
    plt.xlabel("Lag (Days)")
    plt.ylabel("ACF")
    plt.show()


#Mean Analysis
plot_avg(train, 'rainfall')

#Rolling Mean Analysis
plot_rolling_rainfall(train, window = 30)

#Fourier Transform Analysis
plot_fft_rainfall(train)

#Autocorrelation Check
plot_acf_rainfall(train)


# Create the bar plot for target variable (rainfall)
plt.figure(figsize=(7, 4))
ax = sns.countplot(x=train['rainfall'])

# Add labels on top of the bars
for container in ax.containers:
    ax.bar_label(container)

# Labels and title
plt.xlabel("Rainfall (0 = No Rain, 1 = Rain)", fontsize=12, fontweight='bold')
plt.ylabel("Count", fontsize=12, fontweight='bold')
plt.title("Rainfall Distribution", fontsize=12, fontweight='bold')

# Show the plot
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import optuna
from functools import partial


feature_columns = ['pressure', 'temparature',  'dewpoint','humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
target_column = 'rainfall'


# Split Data
y = train[target_column]
x = train[feature_columns]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# Scale Numerical Features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


def plot_roc_curve(y_true, y_pred):
    # Evaluate Model
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_pred):.4f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Rainfall Prediction", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid()
    plt.show()


# Define the Optuna Optimization Function
def objective(trial, x_train, x_test, y_train, y_test):
    # Tune hyperparameters for Logistic Regression
    lr_C = trial.suggest_float('lr_C', 1e-4, 10, log=True)

    # Tune hyperparameters for Random Forest
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
    rf_max_depth = trial.suggest_int('rf_max_depth', 3, 20)

    # Tune hyperparameters for XGBoost
    xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
    xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 20)
    xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)

    # Define the models with suggested parameters
    model1 = LogisticRegression(C=lr_C, max_iter=500)
    model2 = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
    model3 = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
                           learning_rate=xgb_learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Create Voting Classifier (Soft Voting)
    voting_clf = VotingClassifier(estimators=[
        ('lr', model1),
        ('rf', model2),
        ('xgb', model3)
    ], voting='soft')

    # Train the model
    voting_clf.fit(x_train, y_train)

    # Evaluate the model
    y_pred = voting_clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return roc_auc  # Optuna will try to maximize roc_auc


study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, x_train=x_train, x_test=x_val, y_train=y_train, y_test=y_val), n_trials=20)

# Best Parameters
print("\n Best Hyperparameters Found:")
print(study.best_params)

# Train the Final Model with Best Parameters
best_params = study.best_params

final_lr = LogisticRegression(C=best_params['lr_C'], max_iter=500)
final_rf = RandomForestClassifier(n_estimators=best_params['rf_n_estimators'], max_depth=best_params['rf_max_depth'], random_state=42)
final_xgb = XGBClassifier(n_estimators=best_params['xgb_n_estimators'], max_depth=best_params['xgb_max_depth'],
                          learning_rate=best_params['xgb_learning_rate'], use_label_encoder=False, eval_metric='logloss', random_state=42)

# Create Final Voting Classifier
final_voting_clf = VotingClassifier(estimators=[
    ('lr', final_lr),
    ('rf', final_rf),
    ('xgb', final_xgb)
], voting='soft')

# Train the Final Optimized Model
final_voting_clf.fit(x_train, y_train)

# Evaluate Final Model
y_pred_final = final_voting_clf.predict(x_val)
final_accuracy = accuracy_score(y_val, y_pred_final)
final_roc_auc = roc_auc_score(y_val, y_pred_final)
print(f"\n Final Optimized Accuracy: {final_accuracy:.4f}")
print(f"\n Final Optimized ROC_AUC: {final_roc_auc:.4f}")


plot_roc_curve(y_val, y_pred_final)


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col = 0)


test.isna().sum()


test[test.winddirection.isna()]


mode_winddirection_per_day = train.groupby('day')['winddirection'].agg(lambda x: x.mode()[0])

# Convert to DataFrame for better readability
mode_winddirection_per_day = mode_winddirection_per_day.reset_index()
mode_winddirection_per_day.columns = ['day', 'winddirection_mode']


mode_winddirection_per_day[mode_winddirection_per_day['day']==153]


test['winddirection'] = test.winddirection.fillna(220.0)


test.isna().sum()


x_test = test[feature_columns]
x_test.head()


x_test = scaler.transform(x_test)
x_test


y_test_pred = final_voting_clf.predict_proba(x_test)


y_test_pred


submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.head()                   


submission['rainfall'] = y_test_pred[:,1]


submission


submission.to_csv("submission.csv", index=False)

