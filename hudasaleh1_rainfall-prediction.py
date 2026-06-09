# Install TabPFN
!pip install tabpfn 
!pip install tabpfn-extensions


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
from sklearn.metrics import roc_auc_score,auc, classification_report, confusion_matrix,roc_curve, accuracy_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import optuna
from boruta import BorutaPy
#from tabpfn import TabPFNClassifier
#from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

import gc
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
# select color palette
palette_color = sns.color_palette('dark')




# Read data files
train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
original_data = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")

original_data.columns = original_data.columns.str.strip()
original_data['rainfall'] = original_data['rainfall'].str.lower().map({'yes': 1, 'no': 0})
train_data = pd.concat([train_data, original_data], axis=0, ignore_index=True)


# Number of rows [examples], Number of columns [Features] 
print("Total rows in train data: {0}, Total columns in train data: {1}".
      format(train_data.shape[0], train_data.shape[1]))
      
print("Total rows in test data: {0}, Total columns in test data: {1}".
      format(test_data.shape[0], test_data.shape[1]))


train_data.info()


test_data.info()


X =  pd.concat([train_data.drop(['id','rainfall'], axis=1), test_data.drop('id', axis=1)], axis=0, ignore_index=True)

# Create the target variable y
y = [0] * len(train_data) + [1] * len(test_data)

# Train an XGBoost model
model = XGBClassifier(random_state=0)
cv_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1, method='predict_proba')

# Calculate ROC-AUC score
score = roc_auc_score(y_true=y, y_score=cv_preds[:, 1])
print(f"ROC-AUC score: {score:0.3f}")


# drop id
train_data.drop('id', axis=1, inplace=True)
test_data.drop('id', axis=1, inplace=True)

# fill the missing data as columns' mean
train_data['winddirection'].fillna(train_data["winddirection"].mean(), inplace=True)
train_data['windspeed'].fillna(train_data["windspeed"].mean(), inplace=True)
test_data['winddirection'].fillna(test_data["winddirection"].mean(), inplace=True)


num_cols = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud','sunshine','winddirection','windspeed']

def plot_distribution_pairs(train, test, feature, dataset="set", palette=None):
    data_df = train.copy()
    data_df[dataset] = 'train'
    data_df = pd.concat([data_df, test.copy()]).fillna('test')
    f, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, s in enumerate(data_df[dataset].unique()):
        selection = data_df.loc[data_df[dataset]==s, feature]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sns.histplot(selection, color=palette[i], ax=axes[0], label=s)
            sns.boxplot(x=dataset, y=feature, data=data_df, palette=palette, ax=axes[1])
    axes[0].set_title(f"Paired train/test distributions of {feature}")
    axes[1].set_title(f"Paired train/test boxplots of {feature}")
    axes[0].legend()
    axes[1].legend()
    plt.show()

for feature in num_cols:
    plot_distribution_pairs(train_data, test_data, feature, palette=palette_color)


plt.figure(figsize=(5, 5))
# Plotting
train_data['rainfall'].value_counts().plot.pie(
    colors=palette_color,
    autopct="%1.1f%%",
    startangle=140,  
    textprops={'fontsize': 14,'color': 'white'}
)

# Adding a title
plt.title('Target Distribution', fontsize=18, weight='bold')
plt.show()


train_data.skew()


num_cols = ['dewpoint', 'humidity','cloud','sunshine','winddirection','windspeed','rainfall']
sns.pairplot(train_data[num_cols], hue='rainfall',palette=palette_color)
plt.show()


def process_weather_data(data):
    # Temperature range (maxtemp - mintemp)
    data['temp_range'] = data['maxtemp'] - train_data['mintemp']

    # Dewpoint difference (temperature - dewpoint)
    data['dewpoint_diff'] = data['temparature'] - data['dewpoint']

    # Cloud cover and Temperature interaction feature
    data['cloud_sunshine_interaction'] = data['cloud'] * data['sunshine']

    # Pressure and Temperature interaction feature
    data['pressure_temp_interaction'] = data['pressure'] * data['temparature']

    # Humidity and Dewpoint ratio
    data['humidity_dewpoint_ratio'] = data['humidity'] / (data['dewpoint'] + 1e-5)  

    # Additional Cloud-Sunshine Interaction
    data['cloud_sunshine_interaction_v2'] = data['cloud'] * (1 - data['sunshine'] / 100)

    # Cloud coverage rate (normalized to 0-1 range)
    data['cloud_coverage_rate'] = data['cloud'] / 100

    # Weather severity metric
    data['weather_severity'] = (data['cloud'] * data['humidity']) / (data['pressure'] * (data['sunshine'] + 1))
    
    # Extract temporal features
    data['month'] = ((data['day'] - 1) // 30 + 1).clip(upper=12)
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5  # Spring
                                         else 2 if 6 <= x <= 8  # Summer
                                         else 3 if 9 <= x <= 11  # Autumn
                                         else 0)  # Winter
    data = data.drop(columns=["month"])

    # Seasonal trends
    data['season_temp_trend'] = data['temparature'] * data['season']
    
    data['season_cloud_trend'] = data['cloud'] * data['season']
    return data

train_data = process_weather_data(train_data)
test_data = process_weather_data(test_data)



plt.figure(figsize = (25, 12))
sns.heatmap(train_data.corr(), annot = True, cmap = "copper");


X = train_data.drop(['rainfall'],axis=1)
y = train_data['rainfall']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test = scaler.transform(test_data)

# Convert scaled X to DataFrame to retain column names
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=test_data.columns)


from sklearn.model_selection import TimeSeriesSplit
# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "Neural Network": MLPClassifier(random_state=42, max_iter=100, hidden_layer_sizes=(10,)),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Extra Tree": ExtraTreesClassifier(random_state=42),

}

# Train models using TimeSeriesSplit CV
FOLDS = 5
tscv = TimeSeriesSplit(n_splits=FOLDS)
auc_scores = {}
roc_curves = {}

for name, model in models.items():
    oof_preds = np.zeros(len(y))  # Out-of-fold predictions
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    auc_score = roc_auc_score(y, oof_preds)
    auc_scores[name] = auc_score
    fpr, tpr, _ = roc_curve(y, oof_preds)
    roc_curves[name] = (fpr, tpr, auc_score)
    print(f"{name}: AUC = {auc_score:.4f}")

# Plotting ROC curves 
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.show()


# Initialize the model which will be used for feature selection
model = RandomForestClassifier(random_state=42, verbose=0)

# Boruta feature selection
boruta = BorutaPy(estimator=model, n_estimators='auto', random_state=42)
boruta.fit(X_scaled, y.values)

# Get selected features
selected_features = X_scaled.columns[boruta.support_]
print("Selected Features:", selected_features)



def objective(trial):
    # Hyperparameters to tune for Logistic Regression
    param = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e2),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']), # solvers that support l1 and l2
    }

    FOLDS = 5
    tscv = TimeSeriesSplit(n_splits=FOLDS)
    auc_scores = []

    for train_idx, val_idx in tscv.split(X_scaled[selected_features]):
        X_train, X_val = X_scaled[selected_features].iloc[train_idx], X_scaled[selected_features].iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize and train Logistic Regression model
        model = LogisticRegression(random_state=42, **param)

        model.fit(X_train, y_train)

        # Make predictions and calculate AUC
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)




# Print the best trial and hyperparameters
print('Best trial:', study.best_trial)
print('Best hyperparameters:', study.best_trial.params)




# Retrain the model with the best parameters
best_params = study.best_trial.params
best_model = LogisticRegression(random_state=42, **best_params)
best_model.fit(X_scaled[selected_features], y)

# Predict probabilities for the ROC curve
y_pred_proba = best_model.predict_proba(X_scaled[selected_features])[:, 1]




# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



# Get feature coefficients from the Logistic Regression model
coefficients = best_model.coef_[0]  # Access the coefficients

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_scaled[selected_features].columns,
    'Importance': np.abs(coefficients)  # Use absolute values for importance
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='brown')
plt.xlabel('Feature Importance (Absolute Coefficient Value)')
plt.title('Feature Importance based on Logistic Regression Model')
plt.show()


# Classification Report
y_pred = (y_pred_proba>0.5).astype(int)
report = classification_report(y, y_pred)
print("Classification Report:")
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='copper', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



results = best_model.predict_proba(X_test[selected_features])[:, 1]
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = results
submission.to_csv('submission.csv', index=False)

