import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,PowerTransformer,StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import StackingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

import optuna
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


df= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df.drop('id',axis=1,inplace=True)

test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv",index_col='id')
test['winddirection']= test['winddirection'].fillna(test['winddirection'].median())



# here we see that their is no null values in our dataset
# every column is in numeric 
df.info()


# check basic stats of Dataset
df.describe()


# Histograms
df.hist(bins=15, figsize=(15, 10))
plt.suptitle("Feature Distributions")
plt.show()


# Boxplots
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns[:-1]):  # Exclude target
    plt.subplot(4, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()



# Skewness and Kurtosis
skew = df.skew()
kurtosis = df.kurtosis()
# Combine into a DataFrame
stats_df = pd.DataFrame({'Skewness': skew, 'Kurtosis': kurtosis})
stats_df



# Count plot for target variable
sns.countplot(x='rainfall', data=df)
plt.title("Rainfall Class Distribution")
plt.show()

# Class distribution
print("Rainfall Class Distribution:")
print(df['rainfall'].value_counts(normalize=True))



# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()



# # Pair plot for top correlated features
# sns.pairplot(df, hue='rainfall')
# plt.show()


from scipy import stats

# Z-score method
z_scores = stats.zscore(df.select_dtypes(include=[float, int]))
outliers = (z_scores > 3).sum(axis=0)
print("\nOutliers (Z-score > 3):")
print(outliers)

# IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers (IQR method):")
print(outliers_iqr)



# High-outlier columns
outlier_cols = ['cloud', 'humidity', 'dewpoint']

# Boxplots for high-outlier features
plt.figure(figsize=(15, 5))
for i, col in enumerate(outlier_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()



# Add extreme weather indicators
df['high_cloud'] = (df['cloud'] > df['cloud'].quantile(0.95)).astype(int)
df['low_humidity'] = (df['humidity'] < df['humidity'].quantile(0.05)).astype(int)
df['high_dewpoint'] = (df['dewpoint'] > df['dewpoint'].quantile(0.95)).astype(int)



# Define capping thresholds (1st and 99th percentiles)
def cap_outliers(df, col, lower=0.01, upper=0.99):
    lower_bound = df[col].quantile(lower)
    upper_bound = df[col].quantile(upper)
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

# Apply capping to selected features
for col in ['cloud', 'humidity', 'dewpoint']:
    df = cap_outliers(df, col)



# Interaction terms
df['dewpoint_humidity'] = df['dewpoint'] * df['humidity']

# Extreme cloud indicator (95th percentile)
df['high_cloud_intensity'] = (df['cloud'] > df['cloud'].quantile(0.95)).astype(int)


import numpy as np

# Left-skewed: square root or cube root transformations
df['dewpoint'] = np.cbrt(df['dewpoint'])
df['cloud'] = np.cbrt(df['cloud'])
df['rainfall'] = np.cbrt(df['rainfall'])

# Right-skewed: log transformations
df['sunshine'] = np.log1p(df['sunshine'])
df['winddirection'] = np.log1p(df['winddirection'])
df['windspeed'] = np.log1p(df['windspeed'])



df.head()



from sklearn.preprocessing import PowerTransformer

# Apply Yeo-Johnson transformation (handles zero and negative values)
pt = PowerTransformer(method='yeo-johnson')

columns = ['humidity', 'cloud', 'dewpoint', 'windspeed']
df[[f'{col}' for col in columns]] = pt.fit_transform(df[columns])



# Histograms
df.hist(bins=15, figsize=(15, 10))
plt.suptitle("Feature Distributions")
plt.show()


X =df.drop('rainfall',axis=1)
y=df['rainfall']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Feature Importances')
plt.show()


feature_importances


# Drop less important features
df.drop(['high_cloud','high_cloud_intensity','high_dewpoint','low_humidity'],axis=1,inplace=True)


# Interaction terms
test['dewpoint_humidity'] = test['dewpoint'] * test['humidity']


# Data preprocss and Feature Engineering
X = df.drop(columns=['rainfall'])
y = df['rainfall']
X_test = test.copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("Resampled Class Distribution:", pd.Series(y_resampled).value_counts())


X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(solver='liblinear',C=0.30608001233888504,penalty='l1'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    'XGboost':XGBClassifier(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    accuracy = log_loss(y_val, y_pred)
    results[name] = accuracy
    print(f"{name}: loss = {accuracy:.4f}")


import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 200, 2000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
    
    
    # Define and train model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=0.6,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate with log loss
    rf_preds = rf.predict_proba(X_val)[:, 1]
    logloss = log_loss(y_val, rf_preds)
    
    return logloss

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)   # I run 50 trails you can try more 


# Best hyperparameters
print("Best RF Hyperparameters:", study.best_params)
print("Best Log Loss:", study.best_value)

# Train model with best hyperparameters
best_rf = RandomForestClassifier(
    **study.best_params,max_features=0.6,
    random_state=42
)
best_rf.fit(X_train, y_train)

# Predictions and evaluation
rf_preds = best_rf.predict_proba(X_val)[:, 1]
print("Fine-tuned RF Log Loss:", log_loss(y_val, rf_preds))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store log loss for each fold
gb_log_losses = []
rf_log_losses = []
ensemble_log_losses = []

# Ensure y_train is a NumPy array
y_train_array = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train

for train_idx, val_idx in skf.split(X_train, y_train_array):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train_array[train_idx], y_train_array[val_idx]
    
    # Train Random Forest
    best_rf.fit(X_train_fold, y_train_fold)
    rf_preds = best_rf.predict_proba(X_val_fold)[:, 1]
    rf_log_losses.append(log_loss(y_val_fold, rf_preds))

# Average log loss across folds
print("Cross-Validated RF Log Loss:", np.mean(rf_log_losses))


# Train base models on the full training set
best_rf.fit(X_resampled, y_resampled)

# Final Predictions for Submission
final_pred = best_rf.predict_proba(X_test_scaled)[:, 1]


submission = pd.DataFrame({"id": test.index, "rainfall":final_pred})
submission.to_csv("submit\\submisson7.csv", index=False)
print("Submission file saved!")

