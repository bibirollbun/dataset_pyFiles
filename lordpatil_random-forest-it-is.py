import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


print("Loading data...")
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure train.csv, test.csv, and sample_submission.csv are in the same directory.")
    exit()


if 'Phosphor ous' in train_df.columns:
    train_df.rename(columns={'Phosphor ous': 'Phosphorous'}, inplace=True)
    print("Renamed 'Phosphor ous' to 'Phosphorous' in training data.")


print("Data loaded successfully.")
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission_df.shape}")


test_ids = test_df['id']


print("\n--- EDA ---")


print("\nTrain DataFrame head:")
print(train_df.head())
print("\nTest DataFrame head:")
print(test_df.head())


print("\nTrain DataFrame info:")
train_df.info()
print("\nTest DataFrame info:")
test_df.info()


print("\nTrain DataFrame describe:")
print(train_df.describe(include='all'))
print("\nTest DataFrame describe:")
print(test_df.describe(include='all'))


print("\nMissing values in train data:")
print(train_df.isnull().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum())


numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil Type', 'Crop Type']
target = 'Fertilizer Name'


print("\nPlotting distribution of numerical features in training data...")
for col in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col} in Training Data')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'eda_hist_{col}_train.png') # Save plots
    plt.show()


print("\nPlotting distribution of numerical features in test data...")
for col in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(test_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col} in Test Data')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'eda_hist_{col}_test.png') # Save plots
    plt.show()


print("\nPlotting count plots for categorical features in training data...")
for col in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=train_df[col], order=train_df[col].value_counts().index)
    plt.title(f'Count of {col} in Training Data')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.savefig(f'eda_countplot_{col}_train.png')
    plt.show()


print("\nPlotting count plots for categorical features in training data...")
for col in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=train_df[col], order=train_df[col].value_counts().index)
    plt.title(f'Count of {col} in Training Data')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.savefig(f'eda_countplot_{col}_train.png')
    plt.show()


print("\nPlotting correlation heatmap for numerical features in training data...")
plt.figure(figsize=(10, 8))
correlation_matrix = train_df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features (Train Data)')
plt.savefig('eda_correlation_heatmap_train.png')
plt.show()


le = LabelEncoder()
train_df[target] = le.fit_transform(train_df[target])
# Store mapping for later use if needed to revert to original names
label_classes = le.classes_
print(f"Target variable '{target}' encoded.")
print(f"Encoded classes: {dict(zip(range(len(label_classes)), label_classes))}")


combined_df = pd.concat([train_df.drop([target, 'id'], axis=1), test_df.drop('id', axis=1)], keys=['train', 'test'])


print(f"One-hot encoding categorical features: {categorical_features}")
combined_df = pd.get_dummies(combined_df, columns=categorical_features, prefix=categorical_features)


X_train = combined_df.loc['train']
X_test = combined_df.loc['test']


y_train = train_df[target]


print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")


train_cols = X_train.columns
test_cols = X_test.columns


missing_cols_train = set(test_cols) - set(train_cols)
for c in missing_cols_train:
    X_train[c] = 0


missing_cols_test = set(train_cols) - set(test_cols)
for c in missing_cols_test:
    X_test[c] = 0


X_test = X_test[train_cols] # Ensure order and presence of columns match


print("\n--- Model Training ---")
# Using RandomForestClassifier as it's good for this kind of data and can give feature importances
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced') # Added class_weight


print("Training RandomForestClassifier model...")
model.fit(X_train, y_train)
print("Model training complete.")


print("Predicting probabilities on the test set...")
probabilities = model.predict_proba(X_test)


top_3_indices = np.argsort(probabilities, axis=1)[:, -3:] # Get indices of top 3 scores
top_3_fertilizers_encoded = np.fliplr(top_3_indices) # Order from highest to lowest prob


predicted_fertilizer_names_list = []
for row_indices in top_3_fertilizers_encoded:
    # Get original names, handle cases where we might want fewer than 3 if probabilities are very low
    # For now, always take top 3 as per sample submission (or fewer if fewer classes exist than 3)
    num_classes_available = len(label_classes)
    actual_top_n = min(3, num_classes_available) # Ensure we don't try to get more classes than exist
    names = []
    # Iterate through the top N indices for the current sample
    for i in range(actual_top_n):
        class_idx = row_indices[i]
        # Ensure the index is valid (it should be if from argsort on predict_proba)
        if class_idx < num_classes_available:
             names.append(label_classes[class_idx])
    
    # The problem states "up to three value, space delimited."
    # The sample submission has exactly 3 for the first few lines, but it might not be required.
    # For now, let's join them. If a model is not confident, it might predict fewer.
    # Our current approach takes the top 3 regardless of confidence.
    # We need to ensure we only take as many as there are unique classes if less than 3.
    predicted_fertilizer_names_list.append(" ".join(names))


submission_df = pd.DataFrame({'id': test_ids, 'Fertilizer Name': predicted_fertilizer_names_list})


print("\nSubmission DataFrame head:")
print(submission_df.head())


submission_file = "submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"\nSubmission file '{submission_file}' created successfully.")

