import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# Load the datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


print("Train Data Info:")
train_df.info()
print("\nTest Data Info:")
test_df.info()

print("\nTrain Data Description (Numerical):")
print(train_df.describe())

print("\nValue Counts for Categorical Features in Train Data:")
for col in ['Stage_fear', 'Drained_after_socializing', 'Personality']:
    print(f"\n--- {col} ---")
    print(train_df[col].value_counts())
    print(train_df[col].value_counts(normalize=True))


# Univariate Histograms for numerical features
train_df.select_dtypes(include=np.number).drop('id', axis=1).hist(bins=15, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Univariate Histograms of Numerical Features (Train Data)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Univariate Count Plots for categorical features
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=train_df, x='Stage_fear')
plt.title('Distribution of Stage Fear')
plt.subplot(1, 2, 2)
sns.countplot(data=train_df, x='Drained_after_socializing')
plt.title('Distribution of Drained After Socializing')
plt.tight_layout()
plt.show()


# Multivariate Analysis: Correlation Matrix for numerical features
numerical_cols = train_df.select_dtypes(include=np.number).columns.drop('id')
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Multivariate Analysis: Box Plots for numerical features vs. Personality
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=train_df, x='Personality', y=col)
    plt.title(f'{col} vs. Personality')
    plt.show()


# Multivariate Analysis: Stacked Bar Plots for categorical features vs. Personality
for col in ['Stage_fear', 'Drained_after_socializing']:
    personality_counts = train_df.groupby([col, 'Personality']).size().unstack(fill_value=0)
    personality_counts.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title(f'{col} vs. Personality')
    plt.ylabel('Count')
    plt.show()


# Impute missing numerical values with the median
for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    median_val_train = train_df[col].median()
    train_df[col].fillna(median_val_train, inplace=True)
    test_df[col].fillna(median_val_train, inplace=True) # Use train median for test set


# Impute missing categorical values with the mode
for col in ['Stage_fear', 'Drained_after_socializing']:
    mode_val_train = train_df[col].mode()[0]
    train_df[col].fillna(mode_val_train, inplace=True)
    test_df[col].fillna(mode_val_train, inplace=True) # Use train mode for test set


# One-hot encode categorical features
categorical_features = ['Stage_fear', 'Drained_after_socializing']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


# Fit encoder on training data and transform both train and test
encoded_train_features = encoder.fit_transform(train_df[categorical_features])
encoded_test_features = encoder.transform(test_df[categorical_features])


# Create DataFrames from encoded features
encoded_train_df = pd.DataFrame(encoded_train_features, columns=encoder.get_feature_names_out(categorical_features))
encoded_test_df = pd.DataFrame(encoded_test_features, columns=encoder.get_feature_names_out(categorical_features))


# Concatenate encoded features with numerical features
X_train_processed = pd.concat([train_df.drop(['id', 'Personality'] + categorical_features + numerical_cols.tolist(), axis=1),
                                train_df[numerical_cols],
                                encoded_train_df], axis=1)
X_test_processed = pd.concat([test_df.drop(['id'] + categorical_features + numerical_cols.tolist(), axis=1),
                               test_df[numerical_cols],
                               encoded_test_df], axis=1)


# Ensure columns match between train and test after encoding
train_cols = X_train_processed.columns
test_cols = X_test_processed.columns


missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_processed[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train_processed[c] = 0

X_test_processed = X_test_processed[train_cols] # Align columns


# Encode the 'Personality' target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(train_df['Personality'])

print("\nTrain Data after Preprocessing Info:")
X_train_processed.info()
print("\nTest Data after Preprocessing Info:")
X_test_processed.info()


# Initialize and train the GradientBoostingClassifier
# Using n_estimators=200 for potentially better performance on small datasets, and random_state for reproducibility
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_processed, y_train_encoded)


# Make predictions on the preprocessed test set
predictions_encoded = model.predict(X_test_processed)

# Inverse transform the predictions to get original labels
predictions_personality = le.inverse_transform(predictions_encoded)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': predictions_personality})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")




