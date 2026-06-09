# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings as warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

train_df.head()


train_df.info()


train_df.isnull().sum()


train_df.drop(columns = 'day').describe()


test_df.head()


test_df.info()


test_df.describe()


test_df.isna().sum()


# Fill missing value with median column value
test_df['winddirection'].fillna(test_df['winddirection'].median(), inplace = True)


import matplotlib.pyplot as plt
import seaborn as sns


for col in train_df.columns[1:-1]:
    figure, axes = plt.subplots(1, 2, figsize = (15, 6))
    
    sns.boxplot(data = train_df,
                y = col,
                x = 'rainfall',
                palette = 'pastel',
                ax = axes[0])
    axes[0].set_title(f'Boxplot of {col}')

    sns.histplot(data = train_df,
                 x = col,
                 hue = 'rainfall',
                 kde = True,
                 palette = 'pastel',
                 ax = axes[1])
    axes[1].set_title(f"Histogram of {col}")
    plt.show()


# Rainfall Distribution
sns.countplot(train_df,
              x = 'rainfall',
              palette='coolwarm')
plt.title('Rainfall Class Distribution')
plt.xlabel('Rainfall')
plt.ylabel('Count')
plt.show()


# Checking Relationship between numerical variables
plt.figure(figsize = (15, 8))
sns.heatmap(data = train_df.corr(),
            annot = True,
            fmt = '.4f')
plt.title("Correlation between numerical features", fontsize = 18)
plt.show()


train_df.head()


train_df.describe()


def preprocess_data(data):
    # Feature Engineering
    data["cloud_windspeed"] = data["cloud"] * data["windspeed"]
    
    data["temp_range"] = data["maxtemp"] - data["mintemp"]
    
    data['pressure_temp_ratio'] = data['pressure'] / (data['temparature'] + 1)  # Avoid division by zero.
    
    data['dew_temp_diff'] = data['temparature'] - data['dewpoint']
    
    data['dew_humidity_ratio'] = data['dewpoint'] / (data['humidity'] + 1)
    
    data['dew_sunshine_ratio'] = data['dewpoint'] / (data['sunshine'] + 1)  # Avoid division by zero
    
    # Rainy condition indicators
    data['high_cloud_humidity'] = (data['cloud'] > 50) & (data['humidity'] > 60)
    
    data['cloud_coverage_rate'] = data['cloud'] / 100
    
    data['humidity_sunshine_ratio'] = data["humidity"] / (data['sunshine'] + 1)

    data['cloud*humidity/temparature_ratio'] = (data['cloud'] * data['humidity']) / data['temparature']

    # Seasons
    data['month'] = ((data['day'] - 1) // 30 + 1).clip(upper=12)
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5  # Spring
                                                   else 2 if 6 <= x <= 8  # Summer
                                                   else 3 if 9 <= x <= 11  # Autumn
                                                   else 0)  # Winter
    data = data.drop(columns=["month"])

    # Seasonal trends
    data['season_temp_trend'] = data['temparature'] * data['season']
    
    data['season_cloud_trend'] = data['cloud'] * data['season']

    
    data = data.drop(columns=["maxtemp", "winddirection","humidity","temparature","cloud"])
    
    return data

# Apply to train and test datasets
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)


train_df.head()


plt.figure(figsize = (15, 8))
sns.heatmap(data = train_df.corr(),
            fmt = '.2f',
            annot = True,
            cmap = 'coolwarm')
plt.title("Correlation between more feature-engineered data")
plt.show()


from sklearn.preprocessing import StandardScaler


# Select features and target variable
X = train_df.drop(['rainfall'], axis=1)
y = train_df['rainfall']
X_test = test_df

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

X.shape, y.shape, X_test.shape


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve


models = {
    'Logistic Regression' : LogisticRegression(random_state = 42, max_iter = 1000),
    'Support Vector Machine' : SVC(probability = True, random_state = 42),
    'Random Forest Classifier' : RandomForestClassifier(random_state = 42, n_estimators = 300),
    'K-Nearest Neighbors' : KNeighborsClassifier()
}

# Splitting to train and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Accuracy, Classification Report and AUC score
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    class_report = classification_report(y_pred, y_val)
    accuracy = accuracy_score(y_pred, y_val)

    print(f"Classification Report for {name} model:\n{class_report}\n")
    print(f"Accuracy Score for {name} model: {accuracy:.4f}\n\n")


roc_curves = {}
auc_scores = {}

for name, model in models.items():
    # Probability score used for calculating ROC instead of discrete class predictios(0, 1)
    y_probs = model.predict_proba(X_val)[:, 1]
    
    auc_score = roc_auc_score(y_val, y_probs)
    fpr, tpr, _ = roc_curve(y_val, y_probs)

    auc_scores[name] = auc_score
    roc_curves[name] = (fpr, tpr, auc_score)
    
    print(f"{name} : AUC score = {auc_score:.4f}")


# ROC curves
plt.figure(figsize=(8, 6))
for model_name, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()



# Find the best model overall
best_model_name = max(auc_scores, key = auc_scores.get)
best_model = models[best_model_name]
print(f"Best Model Overall: {best_model_name} with AUC = {auc_scores[best_model_name]:.4f}")


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
    'Feature': train_df.drop(['rainfall'], axis=1).columns,
    'Importance': feature_importance
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title(f"{importance_type} ({best_model_name}) with Best AUC")
plt.show()


# Predictions for the test set with the top N features
test_preds = best_model.predict_proba(X_test)[:, 1]

# Submission
submission = pd.DataFrame({'id': test_df['id'], 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")


submission.head()

