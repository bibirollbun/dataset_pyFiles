!pip install kaggle


import pandas as pd

# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')

# Show basic information and a sample of the data
print("Train Data Info:")
print(train_data.info())
print("\nTrain Data Sample:")
print(train_data.head())

print("\nTest Data Info:")
print(test_data.info())
print("\nTest Data Sample:")
print(test_data.head())

print("\nSample Submission Info:")
print(sample_submission.info())
print("\nSample Submission Sample:")
print(sample_submission.head())



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt

# Assuming train_data and test_data are already loaded as DataFrames

# Separate features and target
X_train = train_data.drop(columns=['id', 'loan_status'])
y_train = train_data['loan_status']
X_test = test_data.drop(columns=['id'])

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define preprocessing steps for categorical and numerical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # Impute missing values with the mean
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical variables
])

# Combine transformations for both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a full pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'ROC-AUC on Validation Set: {roc_auc:.4f}')

# Classify performance based on ROC-AUC score
if roc_auc >= 0.9:
    classification = 'Excellent'
    color = 'green'
elif roc_auc >= 0.8:
    classification = 'Good'
    color = 'blue'
elif roc_auc >= 0.7:
    classification = 'Fair'
    color = 'orange'
else:
    classification = 'Poor'
    color = 'red'

# Compute ROC curve and plot
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color=color, label=f'ROC curve (AUC = {roc_auc:.4f})')

# Add diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Annotate "Good" and "Bad" areas of the ROC curve
plt.annotate(f'   Rate Classification: {classification}', xy=(0.5, 0.2), xytext=(0.5, 0.15),
             arrowprops=dict(facecolor=color, shrink=0.05),
             fontsize=12, color=color)

# Customize the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Show the plot
plt.show()

# Make predictions on the test set
test_pred = model.predict_proba(X_test)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_data['id'], 'loan_status': test_pred})

# Specify download path (make sure the path exists)
submission.to_csv(f' submission.csv', index=False)


