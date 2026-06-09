import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(train.info())
print('-'*50)
print(test.info())
print('-'*50)
print(sample.info())
print('-'*50)


# My complete script for the Kaggle Playground Series S5E7
# My strategy is to follow Chris Deotte's advice for small datasets:
# 1. Keep preprocessing simple and robust.
# 2. Use a strong cross-validation (CV) to evaluate my model, not the public LB.
# 3. Train a baseline model.
# 4. Create a submission file.

# Part 1: I'll start by importing libraries and loading the data.
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# I'll wrap the data loading in a try-except block to run it locally or on Kaggle.
try:
    # This path works inside a Kaggle notebook
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
    print("Data loaded successfully from Kaggle environment.")
except FileNotFoundError:
    # This block will run if the Kaggle files are not found.
    # Make sure to place the CSV files in the same directory as the script if running locally.
    print("Kaggle files not found. Attempting to load from local directory.")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sample_submission = pd.read_csv('sample_submission.csv')
        print("Data loaded successfully from local directory.")
    except FileNotFoundError:
        print("Error: Could not find data files in Kaggle input or local directory. Please check file paths.")
        # Exit the script if no data can be loaded
        exit()


# Part 2: Preprocessing. My plan is to handle missing values and encode variables.

# First, I'll separate my features (X) and my target (y).
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']
X_test = test_df.drop('id', axis=1)

# The target 'Personality' is text, so I need to encode it into numbers.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# I'll identify which columns are numeric and which are categorical to treat them differently.
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Now, I'll create preprocessing pipelines for my numeric and categorical features.
# This keeps my code clean and prevents data leakage during cross-validation.

# For numeric features: I will fill missing values with the median and then scale the data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For categorical features: I will fill missing values with the most frequent value,
# then one-hot encode them. 'handle_unknown' is important for safety.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# I'll combine these steps into a single preprocessor object.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)


# Part 3: Modeling and Cross-Validation. This is where I test my strategy.

# I'll choose LightGBM as my first model. It's fast and powerful.
# I'll set a random_state for reproducibility.
lgbm = lgb.LGBMClassifier(random_state=42)

# I'll combine my preprocessor and model into a single pipeline.
# This is a best practice! It makes the whole workflow much cleaner.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm)
])

# Now for the most important part: Stratified K-Fold Cross-Validation.
# I will use 5 splits. I trust my CV score more than the public leaderboard.
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
fold_accuracies = []

print(f"\nStarting cross-validation with {N_SPLITS} folds...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # I'll fit my pipeline on the training data for this fold.
    model_pipeline.fit(X_train, y_train)

    # I'll make predictions on the validation data.
    val_preds = model_pipeline.predict(X_val)
    oof_preds[val_idx] = val_preds

    # I'll calculate and store the accuracy for this fold.
    accuracy = accuracy_score(y_val, val_preds)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.5f}")

mean_cv_accuracy = np.mean(fold_accuracies)
print(f"\nMean CV Accuracy: {mean_cv_accuracy:.5f}")
print(f"Overall OOF Accuracy: {accuracy_score(y_encoded, oof_preds):.5f}")


# Part 4: Final Training and Submission File Generation.

print("\nTraining final model on all data...")
# Now I'll train my pipeline on the entire training dataset.
model_pipeline.fit(X, y_encoded)

print("Making predictions on the test set...")
# I'll use my fully trained pipeline to predict on the test data.
test_predictions_encoded = model_pipeline.predict(X_test)

# I need to convert the numeric predictions back to their original text labels.
test_predictions = label_encoder.inverse_transform(test_predictions_encoded)

print("Creating submission file...")
# Finally, I'll create the submission DataFrame in the required format.
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nScript finished successfully! 'submission.csv' has been created.")




