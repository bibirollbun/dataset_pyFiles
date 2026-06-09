import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

sub_df = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub_df.head().T


from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report,
                            mean_squared_error, r2_score, brier_score_loss)
from sklearn.calibration import calibration_curve

def evaluate_binary_model(model, X_test, y_test):
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    
    # Classification metrics
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_prob):.4f}")  # Lower is better
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Rain', 'Rain'], 
                yticklabels=['No Rain', 'Rain'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC Curve
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title('ROC Curve')
    plt.show()
    
    # Reliability diagram (calibration curve)
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Reliability Diagram')
    plt.show()
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob)
    }


# Feature importance visualization
def plot_feature_importance(model, feature_names):
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("Model doesn't have feature_importances_ or coef_ attributes")
        return
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


train_df.head()


train_df.describe()


train_df.info()


train_df.rainfall.value_counts()


# Distribution of rainfall
plt.figure(figsize = (12,6))
sns.boxplot(data=train_df, x="rainfall", y="day")#, hue="alive")
plt.xlabel("Days of the Year")
plt.ylabel("Rainfall Frequency")
plt.title("Average Rainfall frequency by the Day of Year")
plt.show()


# Day x Rainfall
grouped_days = train_df.groupby(['day']).agg({'rainfall': 'mean'}).reset_index()

plt.figure(figsize = (12,6))
sns.lineplot(x="day", y="rainfall", data = grouped_days)
plt.xlabel("Days of the Year")
plt.ylabel("Rainfall Frequency")
plt.title("Average Rainfall frequency by the Day of Year")
plt.show()


corr_matrix = train_df.corr()

mask = np.zeros_like(corr_matrix, dtype = np.bool_)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize = (11,9))
sns.heatmap(corr_matrix, mask = mask, center = 0,
            square = True, linewidths=.5,
            cbar_kws = {"shrink": .9}, vmin = -1.2,
            vmax = 1.2, cmap = "coolwarm", annot = True)


train_df.head().T


## Feature Engineering
df = train_df.sort_values(by = 'day').copy()

df['cloud_humidity'] = df['cloud'] * df['humidity']
df['dew_temp_diff'] = df['temparature'] - df['dewpoint']

# Temporal features
df['pressure_3d_avg'] = df['pressure'].rolling(window=3).mean()
df['humidity_7d_avg'] = df['humidity'].rolling(window=7).mean()

# Pressure Gradient
df['pressure_change'] = df['pressure'].diff()

# Cyclical features for day of year
df['temp_dewpoint_spread'] = df['temparature'] - df['dewpoint']

# Pressure change indicators
df['pressure_rising'] = (df['pressure'].diff() > 0).astype(int)
df['pressure_system'] = df['pressure'].diff().apply(
    lambda x: 'rising' if x > 0.01 else (
        'falling' if x < -0.01 else 'stable'
    )
)

## Cloud buildup rate
df['cloud_buildup'] = df['cloud'].diff()

# Threshold-based features (corrected syntax)
df['humidity_high'] = (df['humidity'] > 70).astype(int)
df['dewpoint_depression_critical'] = ((df['temparature'] - df['dewpoint']))

## Lagged variables
df['rainfall_lag1'] = df['rainfall'].shift(1)
df['pressure_lag1'] = df['pressure'].shift(1)
df['cloud_change'] = df['cloud'] - df['cloud'].shift(1)

### domain-specific indices
# Simple CAPE approximation
df['simple_cape'] = df['humidity'] * df['temp_dewpoint_spread'] * (df['temparature'] > 20).astype(int)

# Storm Potential
df['storm_index'] = (df['humidity']/100) * df['cloud'] * (1 - df['pressure']/1013.25)


df.head().T


df.pressure_system.value_counts()


## Backward Fill
cols_to_fill = ['pressure_3d_avg', 'humidity_7d_avg', 'pressure_change', 'cloud_buildup', 
               'rainfall_lag1', 'pressure_lag1', 'cloud_change']

df[cols_to_fill] = df[cols_to_fill].fillna(method = 'bfill')


df.info()


## Feat. Normalization

num_features = df.select_dtypes("float64").columns.to_list()
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])


# a lil. more pre-processing
encoder = OneHotEncoder(sparse_output=False)

ohe = encoder.fit_transform(df[['pressure_system']])
ohd = pd.DataFrame(ohe, columns=encoder.get_feature_names_out(['pressure_system']))

df = pd.concat([df, ohd], axis = 1)
df.drop(columns = 'pressure_system', inplace = True)


X = df.drop(columns = "rainfall")
y = df['rainfall']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 234)

# model
rf_model = RandomForestRegressor()
rf_model.fit(
    X_train, y_train
)

# feat. importance
importances = pd.DataFrame(
    {
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }
).sort_values('importance', ascending=False).reset_index()

importances


df['cloud_sunshine'] = df['cloud'] * df['sunshine']
df['humidity_pressure'] = df['humidity'] * df['pressure_change']
df['cloud_pressure_change'] = df['cloud'] * df['pressure_change']
df['sunshine_humidity'] = df['sunshine'] * df['humidity']

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2, include_bias = False)
top_features = ["cloud_humidity", "sunshine", "cloud", 
                "windspeed", "pressure_change",
                "pressure_lag1", "dewpoint"]

model_features = top_features + ["cloud_sunshine", "humidity_pressure"]
poly_features = poly.fit_transform(df[top_features])


from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits = 5)
model = XGBClassifier(
    learning_rate = 0.05,
    n_estimators = 200,
    max_depth = 5,
    subsample=0.8,
    coolsample_bytree=0.8
)

for train_idx, test_idx in tscv.split(df[model_features]):
    X_train, X_test = df.iloc[train_idx][model_features], df.iloc[test_idx][model_features]
    y_train, y_test = df.iloc[train_idx]['rainfall'], df.iloc[test_idx]['rainfall']
    model.fit(X_train, y_train)
    
evaluate_binary_model(model, X_test, y_test)
    


## Validate Model
# feat. engineering function

def feat_eng(df: pd.DataFrame):
    df = train_df.sort_values(by = 'day').copy()

    df['cloud_humidity'] = df['cloud'] * df['humidity']
    df['dew_temp_diff'] = df['temparature'] - df['dewpoint']

    # Temporal features
    df['pressure_3d_avg'] = df['pressure'].rolling(window=3).mean()
    df['humidity_7d_avg'] = df['humidity'].rolling(window=7).mean()

    # Pressure Gradient
    df['pressure_change'] = df['pressure'].diff()

    # Cyclical features for day of year
    df['temp_dewpoint_spread'] = df['temparature'] - df['dewpoint']

    # Pressure change indicators
    df['pressure_rising'] = (df['pressure'].diff() > 0).astype(int)
    df['pressure_system'] = df['pressure'].diff().apply(
        lambda x: 'rising' if x > 0.01 else (
            'falling' if x < -0.01 else 'stable'
        )
    )

    ## Cloud buildup rate
    df['cloud_buildup'] = df['cloud'].diff()

    # Threshold-based features (corrected syntax)
    df['humidity_high'] = (df['humidity'] > 70).astype(int)
    df['dewpoint_depression_critical'] = ((df['temparature'] - df['dewpoint']))

    ## Lagged variables
    df['rainfall_lag1'] = df['rainfall'].shift(1)
    df['pressure_lag1'] = df['pressure'].shift(1)
    df['cloud_change'] = df['cloud'] - df['cloud'].shift(1)

    ### domain-specific indices
    # Simple CAPE approximation
    df['simple_cape'] = df['humidity'] * df['temp_dewpoint_spread'] * (df['temparature'] > 20).astype(int)

    # Storm Potential
    df['storm_index'] = (df['humidity']/100) * df['cloud'] * (1 - df['pressure']/1013.25)

    cols_to_fill = ['pressure_3d_avg', 'humidity_7d_avg', 'pressure_change', 'cloud_buildup', 
               'rainfall_lag1', 'pressure_lag1', 'cloud_change']

    df[cols_to_fill] = df[cols_to_fill].fillna(method = 'bfill')

    num_features = df.select_dtypes("float64").columns.to_list()
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    encoder = OneHotEncoder(sparse_output=False)

    ohe = encoder.fit_transform(df[['pressure_system']])
    ohd = pd.DataFrame(ohe, columns=encoder.get_feature_names_out(['pressure_system']))

    df = pd.concat([df, ohd], axis = 1)
    df.drop(columns = 'pressure_system', inplace = True)

    # model features
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    df['humidity_pressure'] = df['humidity'] * df['pressure_change']
    df['cloud_pressure_change'] = df['cloud'] * df['pressure_change']
    df['sunshine_humidity'] = df['sunshine'] * df['humidity']

    poly = PolynomialFeatures(2, include_bias = False)
    top_features = ["cloud_humidity", "sunshine", "cloud", 
                    "windspeed", "pressure_change",
                    "pressure_lag1", "dewpoint"]

    model_features = top_features + ["cloud_sunshine", "humidity_pressure"]
    poly_features = poly.fit_transform(df[top_features])

    return df, top_features, model_features, poly_features

df_test, top_f, model_f, poly_f = feat_eng(test_df)


test_X = df_test[model_f]

predictions = model.predict(test_X)
y_proba = model.predict_proba(test_X)[:, 1]

# 1. Check dimensions
print(f"Original test_df shape: {test_df.shape}")
print(f"Processed df_test shape: {df_test.shape}")
print(f"Predictions shape: {predictions.shape}")

# 2. Save the index from the original frame
test_indices = test_df.index.copy()

# 3. Make sure the predictions match the original test data length
if len(predictions) != len(test_df):
    # Option 1: If df_test has fewer rows, align with original indices
    if len(df_test) < len(test_df):
        # Create a DataFrame with original index and fill with NaN
        submission_df = pd.DataFrame(index=test_indices, columns=['rainfall'])
        # Use the index from df_test to fill in predictions
        processed_indices = df_test.index
        submission_df.loc[processed_indices, 'rainfall'] = predictions
        # Fill any remaining NaN values (if required by the competition)
        submission_df['rainfall'].fillna(0, inplace=True)  # or another appropriate value
    
    # Option 2: If df_test has more rows, truncate to original length
    else:
        # This is less common but possible with certain feature engineering
        submission_df = pd.DataFrame({
            'rainfall': predictions[:len(test_df)]
        }, index=test_indices)
else:
    # If lengths match, simply create the DataFrame
    submission_df = pd.DataFrame({
        'rainfall': predictions
    }, index=test_indices)

# 4. Add ID column if needed
if 'ID' in test_df.columns:
    submission_df['ID'] = test_df['ID']
else:
    # Use index as ID if required
    submission_df['ID'] = test_indices

# 5. Rearrange columns if ID should come first
if 'ID' in submission_df.columns:
    submission_df = submission_df[['ID', 'rainfall']]

# 6. Save submission
submission_df.to_csv('submission.csv', index=False)

print("\n", "Created Submission File")

