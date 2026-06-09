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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test.head()
df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.head()


##****EDA (Exploratory Data Analysis)*****
print("EDA (Exploratory Data Analysis)")

#Load data and check basic info (shape, dtypes, memory usage)
print("Data Types")
print(df.dtypes)
print("DF Shape")
print(df.shape)
print("Memory used by Each Column")
print(df.memory_usage(deep=True))

#Examine target variable distribution and class balance
print(df.value_counts(['Personality']))
print(df['Personality'].value_counts(normalize=True)*100)
proportions = df['Personality'].value_counts(normalize=True)*100
proportions.plot(kind='bar', color=['red','blue'])
plt.ylabel('Percentage')
plt.title('Personality Class Distribution')
plt.show()

#Check for missing values per column
print("Values Missing for each Column")
print(df.isnull().sum())
print("% of values Missing for each Column")
print(round((df.isnull().sum()/df.count())*100))

#Analyze numerical features (distributions, outliers, correlations)
print("Distribution")
df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()
print("outliers")
df.boxplot(figsize=(10,8))
plt.tight_layout()
plt.show()
print("Correlations")
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, cmap='coolwarm')
plt.tight_layout()
plt.show()

#Analyze categorical features (unique values, frequencies)
print(df.select_dtypes(include='object').apply(lambda x : x.value_counts()))



#****Data Preprocessing****
print("Data Preprocessing")

#Handle missing values (imputation strategy)
def handle_missing_values(df):
    for col in df.select_dtypes(include='number').columns:
        df.fillna({col: df[col].mean()}, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df.fillna({col:df[col].mode()}, inplace=True)

handle_missing_values(df)
handle_missing_values(df_test)

#Encode categorical variables (Label/One-hot/Ordinal encoding)
def encode(df):
    le = LabelEncoder() #used in Yes/No Scenarios
    df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
    df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])

encode(df)
encode(df_test)

#Handle outliers (remove/cap/transform)
#no outliers detected in this case

#Scale numerical features (StandardScaler/MinMaxScaler)
print("No High numerical data, so scaling not necessary, only dropping and saving id column, for later use, as it has vast data, and can skew the MODEL Training")
ids = df['id']
df = df.drop('id', axis=1)
ids_test = df_test['id']
x_test_final = df_test.drop('id', axis=1)

#Split features and target variables
X = df.drop('Personality', axis=1) #features
y = df['Personality'] #axis


#****FEATURE ENGINEERING****

#Create interaction features (multiply/divide related features)
def create_features(dframe):
    dframe['friends_per_event'] = dframe['Friends_circle_size'] / (dframe['Social_event_attendance'] + 1)
    dframe['Alone_x_Post'] = dframe['Time_spent_Alone'] * dframe["Post_frequency"]
    # Social interaction patterns
    dframe['social_interaction'] = dframe['Social_event_attendance'] * dframe['Going_outside']
    dframe['social_posting'] = dframe['Social_event_attendance'] * dframe['Post_frequency']
    dframe['outside_posting'] = dframe['Going_outside'] * dframe['Post_frequency']
    
    # Fear and social behavior
    dframe['fear_social_events'] = dframe['Stage_fear'] * dframe['Social_event_attendance']
    dframe['fear_going_out'] = dframe['Stage_fear'] * dframe['Going_outside']

    # Social ratios (add small epsilon to avoid division by zero)
    dframe['alone_to_social_ratio'] = dframe['Time_spent_Alone'] / (dframe['Social_event_attendance'] + 0.1)
    dframe['post_to_friends_ratio'] = dframe['Post_frequency'] / (dframe['Friends_circle_size'] + 0.1)
    dframe['social_to_friends_ratio'] = dframe['Social_event_attendance'] / (dframe['Friends_circle_size'] + 0.1)

    # Efficiency ratios
    dframe['social_efficiency'] = (dframe['Social_event_attendance'] + dframe['Going_outside']) / (dframe['Time_spent_Alone'] + 0.1)

    # Social activity score
    dframe['social_activity_score'] = dframe['Social_event_attendance'] + dframe['Going_outside'] + dframe['Post_frequency']
    
    # Introversion indicator
    dframe['introversion_indicator'] = dframe['Time_spent_Alone'] - dframe['Social_event_attendance']
    
    # Social confidence score
    dframe['social_confidence'] = (dframe['Social_event_attendance'] + dframe['Going_outside']) - dframe['Stage_fear']
    
    # Digital vs Physical social preference
    dframe['digital_social_preference'] = dframe['Post_frequency'] - dframe['Social_event_attendance']
    
    # Social drain vs activity balance
    dframe['social_drain_balance'] = dframe['Drained_after_socializing'] - dframe['social_activity_score']

create_features(df)
create_features(df_test)

def create_bins(dframe):
    # Create bins for Post_frequency
    dframe['post_frequency_binned'] = pd.cut(dframe['Post_frequency'], 
                                       bins=[0, 3, 7, 12, float('inf')], 
                                       labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # Create bins for Time_spent_Alone
    dframe['time_alone_binned'] = pd.cut(dframe['Time_spent_Alone'], 
                                   bins=[0, 2, 5, 8, float('inf')], 
                                   labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # Create bins for Friends_circle_size
    dframe['friends_size_binned'] = pd.cut(dframe['Friends_circle_size'], 
                                     bins=[0, 8, 15, 25, float('inf')], 
                                     labels=['Small', 'Medium', 'Large', 'Very_Large'])

create_bins(X)
create_bins(df_test)

#Extract date/time components if applicable
print("No Date/Time columns in the dataset")

df_test = df_test.reindex(columns=X.columns, fill_value=0)


from sklearn.preprocessing import OrdinalEncoder

# Encode new binned features
new_cat_cols = ['post_frequency_binned', 'time_alone_binned', 'friends_size_binned']
for col in new_cat_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[col] = encoder.fit_transform(X[[col]])
    df_test[col] = encoder.transform(df_test[[col]])

# Use median for skewed features, mean for normal features
from sklearn.impute import SimpleImputer

# Check skewness and apply appropriate imputation
numerical_features = X.select_dtypes(include=[np.number]).columns
skewed_features = []
normal_features = []

for col in numerical_features:
    skewness = X[col].skew()
    if abs(skewness) > 1:
        skewed_features.append(col)
    else:
        normal_features.append(col)

# Apply median imputation to skewed features
if skewed_features:
    median_imputer = SimpleImputer(strategy='median')
    X[skewed_features] = median_imputer.fit_transform(X[skewed_features])
    df_test[skewed_features] = median_imputer.transform(df_test[skewed_features])

# Apply mean imputation to normal features
if normal_features:
    mean_imputer = SimpleImputer(strategy='mean')
    X[normal_features] = mean_imputer.fit_transform(X[normal_features])
    df_test[normal_features] = mean_imputer.transform(df_test[normal_features])

from sklearn.preprocessing import StandardScaler

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(df_test)

# Convert back to DataFrame
X = pd.DataFrame(X_scaled, columns=X.columns)
df_test = pd.DataFrame(test_scaled, columns=df_test.columns)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Individual models with basic tuning
lr = LogisticRegression(random_state=42, max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42)


#Voting Classifier Implementation
from sklearn.ensemble import VotingClassifier
# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('gb', gb)
    ],
    voting='soft'  # Use probability-based voting
)




y_encoded = y.apply(lambda x: 1 if x == 'Extrovert' else 0)
y_encoded.head()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Set up cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"Fold {fold}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Predict
    predictions = ensemble.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    accuracies.append(accuracy)

print(f"\nMean CV Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies)*2:.4f})")


from sklearn.model_selection import cross_val_score
# Compare individual model performance
models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Ensemble': ensemble
}

for name, model in models.items():
    scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")


# Get feature importance from Random Forest
rf.fit(X, y_encoded)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))


from sklearn.preprocessing import LabelEncoder

# Train on full dataset
ensemble.fit(X, y_encoded)

# Generate predictions
test_predictions = ensemble.predict(df_test)


final_test_preds = ['Extrovert' if p == 1 else 'Introvert'
                    for p in test_predictions]



# Create submission file
submission = pd.DataFrame({
    'id': ids_test,  # Use original test id column
    'Personality': final_test_preds
})

submission.to_csv('submission.csv', index=False)
print("Medium level submission saved!")


rm -r medium_level_submission.csv




