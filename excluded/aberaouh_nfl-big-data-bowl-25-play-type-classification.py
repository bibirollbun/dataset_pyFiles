import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib as plt

# Load the dataset from a CSV file
constructedData = pd.read_csv('/kaggle/input/combined-data/combined_data.csv')

# Select features for the Random Forest model
# Include columns starting with 'distance' except specific excluded ones
cols_rf = [colHeader for colHeader in list(constructedData.columns) 
           if (
               colHeader.startswith('distance') and 
               (colHeader not in ['distance_OLB', 'distance_DT', 'distance_ILB', 'distance_FS'])
           )
          ] + ['isDropback']

# Filter the data to only the selected columns and drop rows with missing values
data_rf = constructedData[cols_rf].dropna()

# Split the data into features (X) and target variable (y)
X = data_rf.drop(['isDropback'], axis=1)  # Features
y = data_rf['isDropback']                # Target variable

# Split the data into training and testing sets
XTrain_clf, XTest_clf, yTrain_clf, yTest_clf = train_test_split(X, y)

# Create a pipeline for data preprocessing and model training
pipeline_randomForest = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features by scaling to zero mean and unit variance
    ('model', RandomForestClassifier(class_weight='balanced', max_depth=12))  # Random Forest Classifier with balanced class weights and depth limit
])

# Train the pipeline on the training data
pipeline_randomForest.fit(XTrain_clf, yTrain_clf)

# Predict on the training set to evaluate the model
predicted_training_randomForest = pipeline_randomForest.predict(XTrain_clf)

# Print the accuracy of the model on the training data
print(f'Accuracy: {metrics.accuracy_score(yTrain_clf, predicted_training_randomForest)}')

# Print a detailed classification report
print(classification_report(yTrain_clf, predicted_training_randomForest))

# Generate and display a confusion matrix
metrics.ConfusionMatrixDisplay.from_predictions(yTrain_clf, predicted_training_randomForest)
plt.pyplot.title(f'Training Set (n={XTrain_clf.shape[0]}) Confusion Matrix for Random Forest')
plt.pyplot.show()


# Testing the model

print('Test')
predicted_test_randomForest = pipeline_randomForest.predict(XTest_clf)
y_probs_rf = pipeline_randomForest.predict_proba(XTest_clf)[:, 1]
print(f'Accuracy: {metrics.accuracy_score(yTest_clf, predicted_test_randomForest)}')
print(classification_report(yTest_clf, predicted_test_randomForest))
metrics.ConfusionMatrixDisplay.from_predictions(yTest_clf, predicted_test_randomForest)
plt.pyplot.title(f'Test Set (n={XTest_clf.shape[0]}) Confusion Matrix for Random Forest')
plt.pyplot.show()

# Cross-validation confirmation
    # Please refer to the github repo for data on CV

