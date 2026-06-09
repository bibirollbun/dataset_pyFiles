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


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


print("Trainig Data Shape",df.shape)
print("Test Data Shape",df_test.shape)


df.sample(3)


print("Null Count in Training Data\n",df.isnull().sum())
print("Null Count in Test Data\n",df_test.isnull().sum())


df_test.fillna(df_test.mean(), inplace = True)


print("Description Of Training Data")
df.describe().T


print("Description Of Test Data")
df_test.describe().T


numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']


import seaborn as sns
import matplotlib.pyplot as plt


# Define a Custom color palette
custom_palette = ['#f1b963','#c4c1e0']

# Define numerical features
variables = [col for col in df.columns if col in numerical_variables]

# Function to create and display plots for a single numerical variable
def create_variable_plots(variable):
    sns.set_theme(style = 'whitegrid')

    df_temp = df.copy()
    df_test_temp = df_test.copy()

    df_temp["Dataset"] = "Train"
    df_test_temp["Dataset"] = "Test"
    combined_df = pd.concat([df_temp, df_test_temp])

    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # Box plot
    sns.boxplot(data=combined_df, x=variable, y="Dataset",
               palette = custom_palette, ax = axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_title(f"Box Plot of {variable}") 

    sns.histplot(data=df, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=df_test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of {variable} [Train, Test]")
    axes[1].legend()

    # Adjust spacing and show
    plt.tight_layout()
    plt.show()

# Perform univariate analysis for each numerical variable
for variable in variables:
    create_variable_plots(variable)



plt.figure(figsize=(6,4))
sns.countplot(x = df['rainfall'], palette='coolwarm')
plt.title('Rainfall Class Distribution')
plt.xlabel('Ranfall')
plt.ylabel('Count')
plt.show()


# KDE plot for Feature- Target Relationship
plt.figure(figsize=(14,10))
for i,col in enumerate(numerical_variables, 1):
    plt.subplot(3, 4, i)
    sns.kdeplot(df[col][df['rainfall']==1], color='red', label='Rainfall:1')
    sns.kdeplot(df[col][df['rainfall']==0], color='blue', label='Rainfall:0')
    plt.title(f'Distribution of{col} by Rainfall')
    plt.legend()

plt.tight_layout()
plt.show()


# Filter data based on rainfall
rain_data = df[df['rainfall'] > 0]
no_rain_data = df[df['rainfall'] == 0]

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

# First wind rose plot (rain)
ax1 = axes[0]
ax1.set_theta_direction(-1)
ax1.set_theta_offset(np.pi / 2.0)
bars1 = ax1.bar(
    np.deg2rad(rain_data['winddirection']),
    rain_data['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color="b"  
)
ax1.set_title('Wind Speed and Direction with Rain')

# Second wind rose plot (no rain)
ax2 = axes[1]
ax2.set_theta_direction(-1)
ax2.set_theta_offset(np.pi / 2.0)
bars2 = ax2.bar(
    np.deg2rad(no_rain_data['winddirection']),
    no_rain_data['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color="r"  
)
ax2.set_title('Wind Speed and Direction without Rain')

plt.tight_layout()
plt.show()



# Correlation Heatmap
def plot_correlation_heatmap(data, title, annot_size=12):
    plt.figure(figsize=(12,8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, annot_kws={"size": annot_size},cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Heatmap - {title}', fontsize=16)
    plt.show()

plot_correlation_heatmap(df, "Train Data")


def preprocess_weather_data(data):
    data["dew_humidity"] = data['dewpoint']*data["humidity"]
    data["cloud_windspeed"] = data["cloud"]* data["windspeed"]
    data["cloud_to_humidity"]= data["cloud"]/data["humidity"]
    data["temp_to_sunshine"] = data["sunshine"]/data["temparature"]
    data["wind_temp_interaction"] = data["windspeed"]*data["temparature"]
    data["dew_humidity/sun"] = data["dewpoint"]*data["humidity"]/(data["sunshine"]+1)
    data["dew_humidity_+"] = data["dewpoint"] * data["humidity"]
    data['humidity_sunshine_*'] = data["humidity"] * data['sunshine']
    data["cloud_humidity/pressure"] = (data["cloud"] * data["humidity"]) / data["pressure"]
    # Extract temporal features
    data['month'] = ((data['day'] - 1) // 30 + 1).clip(upper=12)
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5  # Spring
                                         else 2 if 6 <= x <= 8  # Summer
                                         else 3 if 9 <= x <= 11  # Autumn
                                         else 0)  # Winter
    # Seasonal trends
    #data['season_temp_trend'] = data['temparature'] * data['season']
    data['season_cloud_trend'] = data['cloud'] * data['season']
    

    # Seasonal deviation from mean values
    data['season_cloud_deviation'] = data['cloud'] - data.groupby('season')['cloud'].transform('mean')
    data['season_temperature'] = data['temparature'] * data['season']  # Interaction of temper



    
    data = data.drop(columns=["month"])
    #data['season_temp_trend'] = data['avg_temp'] * data['season']
    #data['season_dewpoint_trend'] = data['dewpoint'] * data['season']
    #data["dew_humidity_with_season"] = data['humidity'] * data['season']
    
    data = data.drop(columns=["maxtemp", "winddirection","humidity","temparature","pressure","day","season"])

    return data

# Apply to train and test datasets
df = preprocess_weather_data(df)
df_test = preprocess_weather_data(df_test)


from sklearn.preprocessing import StandardScaler, LabelEncoder


# Select feature and targeet Variable
X = df.drop(['rainfall','id'], axis=1)
y = df['rainfall']
X_test = df_test.drop(['id'], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


import math
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# Define model
models ={
    "Logistic Regression" : LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest" : RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting" : GradientBoostingClassifier(random_state=42),
    "Support Vector Machine" : SVC(probability=True, random_state=42),
    "K-Nearest Neighbors" : KNeighborsClassifier(),
    "Neural Network" : MLPClassifier(random_state=42, max_iter=100, hidden_layer_sizes=(10)),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=6),
    "CatBoost": CatBoostClassifier(random_state=42, iterations=100, learning_rate=0.14, depth=6, verbose=0)
}

# Train models using StratifiedKFold CV
FOLDS = 13
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
auc_scores = {}
roc_curves = {}

for name, model in models.items():
    oof_preds = np.zeros(len(y))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        
        if hasattr(model, 'fit'):
            if "eval_set" in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
            else:
                model.fit(X_train, y_train)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    auc_score = roc_auc_score(y, oof_preds)
    auc_scores[name] = auc_score
    fpr, tpr, _ = roc_curve(y, oof_preds)
    roc_curves[name] = (fpr, tpr, auc_score)
    print(f"{name}: AUC = {auc_score:.4f}")


# Plot ROC curves
plt.figure(figsize=(8, 6))
for model_name, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# Plot AUC scores
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=list(auc_scores.keys()), y=list(auc_scores.values()))

# Annotate the bars with AUC scores
for i, score in enumerate(auc_scores.values()):
    ax.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=45)
plt.ylabel("AUC Score")
plt.xlabel("Models")
plt.title("Model AUC Score Comparison")
plt.ylim(0.5, 1)  
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()


# Find the best model overall
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
print(f"Best Model Overall: {best_model_name} with AUC = {auc_scores[best_model_name]:.4f}")


# Check if the model has feature_importance_ attribute
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
else:
    # For logistic regression, use coefficients as importance
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'

feature_df = pd.DataFrame({
    'Feature': df.drop(['rainfall','id'], axis=1).columns,
    'Importance': feature_importance
})
# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title(f"{importance_type} ({best_model_name}) with Best AUC")
plt.show()


# Select the best model based on AUC
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]

# Check if the model has feature_importances_ attribute
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
else:
    # For logistic regression, use coefficients as importance
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'

# Create a DataFrame to combine feature names and their importance values
feature_df = pd.DataFrame({
    'Feature': df.drop(['rainfall', 'id'], axis=1).columns,
    'Importance': feature_importance
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)


# List of top N features to try
top_feature_counts = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# Variables to track the best AUC and corresponding top features
best_auc_top =0
best_top_n=0
best_oof_preds_top = None

# Loop over different top feature counts
for top_n in top_feature_counts:
    # Select the top N feature
    top_features = feature_df.head(top_n)['Feature']

    # Prepare the data with the selected top N features
    X_top = X[:,[df.drop(['rainfall','id'],axis=1).columns.get_loc(col) for col in top_features]]
    X_test_top = X_test[:, [df.drop(['rainfall','id'], axis=1).columns.get_loc(col) for col in top_features]]

    #Retrain the best model using the top N features
    best_model.fit(X_top, y)

    # Make predictions and calculate AUC for the top N features
    oof_preds_top = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X_top, y):
        X_train, X_val = X_top[train_idx], X_top[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        oof_preds_top[val_idx] = best_model.predict_proba(X_val)[:, 1]

    # Calculate and print AUC score for top N features model
    auc_score_top = roc_auc_score(y, oof_preds_top)
    print(f"AUC for top {top_n} features model: {auc_score_top:.4f}")

    if auc_score_top > best_auc_top:
        best_auc_top = auc_score_top
        best_top_n = top_n
        best_oof_preds_top = oof_preds_top


best_features = feature_df.head(best_top_n)


# Plotting the feature importance for the best model with the highest AUC
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=best_features, palette="mako")
plt.title(f"{importance_type} for Top {best_top_n} Features ({best_model_name})")
plt.show()

print("=" * 50)
print(f"ğŸ�† Best Model: {best_model_name}")
print(f"ğŸ�¯ Best AUC: {best_auc_top:.4f} using Top {best_top_n} Features")
print("=" * 50)


test_preds = best_model.predict_proba(X_test_top)[:, 1]

# Submission
submission = pd.DataFrame({'id': df_test['id'], 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")




