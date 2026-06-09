import pandas as pd

try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_submission = pd.read_csv('sample_submission.csv')

    display(df_train.head())
    display(df_test.head())
    display(df_submission.head())
except FileNotFoundError:
    print("Error: One or more CSV files not found.")
except pd.errors.ParserError:
    print("Error: Could not parse the CSV files. Check file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Examine Data Shapes and Data Types
print("df_train shape:", df_train.shape)
print("df_test shape:", df_test.shape)
print("\ndf_train data types:\n", df_train.dtypes)
print("\ndf_test data types:\n", df_test.dtypes)

# Descriptive Statistics
print("\ndf_train descriptive statistics:\n", df_train.describe())

# Missing Value Analysis
print("\ndf_train missing values:\n", df_train.isnull().sum())
print("\ndf_test missing values:\n", df_test.isnull().sum())

# Outlier Detection (using boxplots)
import matplotlib.pyplot as plt
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    df_train.boxplot(column=col)
    plt.title(col)
plt.tight_layout()
plt.show()

# Target Variable Analysis
print("\ndf_train target variable value counts:\n", df_train['Fertilizer Name'].value_counts())
plt.figure(figsize=(10, 5))
df_train['Fertilizer Name'].value_counts().plot(kind='bar')
plt.title('Distribution of Fertilizer Names')
plt.xlabel('Fertilizer Name')
plt.ylabel('Frequency')
plt.show()


# Feature Relationships (example: scatter plot of Temperature vs. Nitrogen)
plt.figure(figsize=(8, 6))
plt.scatter(df_train['Temparature'], df_train['Nitrogen'], c=df_train['Fertilizer Name'].astype('category').cat.codes, cmap='viridis')
plt.xlabel('Temperature')
plt.ylabel('Nitrogen')
plt.title('Temperature vs. Nitrogen (colored by Fertilizer Name)')
plt.colorbar(label='Fertilizer Name')
plt.show()


from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
import numpy as np

# Separate features and target in training data
X_train = df_train.drop('Fertilizer Name', axis=1)
y_train = df_train['Fertilizer Name']
X_test = df_test.copy()

# Identify categorical and numerical columns
categorical_cols = ['Soil Type', 'Crop Type']
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_categorical = encoder.fit_transform(X_train[categorical_cols])
X_test_categorical = encoder.transform(X_test[categorical_cols])

# Create dataframes from encoded categorical features
df_train_categorical_encoded = pd.DataFrame(X_train_categorical, columns=encoder.get_feature_names_out(categorical_cols), index=X_train.index)
df_test_categorical_encoded = pd.DataFrame(X_test_categorical, columns=encoder.get_feature_names_out(categorical_cols), index=X_test.index)

# Scale numerical features
scaler = RobustScaler()
X_train_numerical_scaled = scaler.fit_transform(X_train[numerical_cols])
X_test_numerical_scaled = scaler.transform(X_test[numerical_cols])

# Create dataframes from scaled numerical features
df_train_numerical_scaled = pd.DataFrame(X_train_numerical_scaled, columns=numerical_cols, index=X_train.index)
df_test_numerical_scaled = pd.DataFrame(X_test_numerical_scaled, columns=numerical_cols, index=X_test.index)

# Combine encoded categorical and scaled numerical features for training and testing data
df_train_encoded = pd.concat([X_train[['id']], df_train_categorical_encoded, df_train_numerical_scaled], axis=1)
df_test_encoded = pd.concat([X_test[['id']], df_test_categorical_encoded, df_test_numerical_scaled], axis=1)

# Label encode the target variable in the training data
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
df_train_encoded['Fertilizer Name'] = y_train_encoded

display(df_train_encoded.head())
display(df_test_encoded.head())


from sklearn.preprocessing import PolynomialFeatures

numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Polynomial Features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Fit and transform on training data
X_train_poly = poly.fit_transform(df_train_encoded[numerical_cols])
X_test_poly = poly.transform(df_test_encoded[numerical_cols])

# Get feature names
poly_feature_names = poly.get_feature_names_out(numerical_cols)

# Create dataframes
df_train_poly = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=df_train_encoded.index)
df_test_poly = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=df_test_encoded.index)

# Drop original numerical columns from encoded dataframes before concatenating
df_train_encoded_dropped = df_train_encoded.drop(columns=numerical_cols)
df_test_encoded_dropped = df_test_encoded.drop(columns=numerical_cols)

# Combine with original dataframes
df_train_encoded = pd.concat([df_train_encoded_dropped, df_train_poly], axis=1)
df_test_encoded = pd.concat([df_test_encoded_dropped, df_test_poly], axis=1)

display(df_train_encoded.head())
display(df_test_encoded.head())


from sklearn.model_selection import train_test_split

X = df_train_encoded.drop(columns=['Fertilizer Name', 'id'])
y = df_train_encoded['Fertilizer Name']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_train_encoded['id'])


from sklearn.model_selection import train_test_split

X = df_train_encoded.drop(columns=['Fertilizer Name', 'id'])
y = df_train_encoded['Fertilizer Name']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf_classifier.predict(X_val)

# Evaluate the model's performance
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy of the RandomForestClassifier on the validation set: {accuracy}")


import numpy as np

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

# Predict probabilities
y_pred_proba = rf_classifier.predict_proba(X_val)

# Get top 3 predictions
y_pred_top3 = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :3]

# Convert predicted labels back to original form
y_pred_labels = label_encoder.inverse_transform(y_pred_top3)

# Convert y_val back to original form
y_true_labels = label_encoder.inverse_transform(y_val.values.reshape(-1, 1))

# Calculate MAP@3
map3 = mapk([[label] for label in y_true_labels], y_pred_labels, k=3)
print(f"MAP@3: {map3}")


import numpy as np

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

# Predict probabilities
y_pred_proba = rf_classifier.predict_proba(X_val)

# Get top 3 predictions
y_pred_top3 = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :3]

# Reshape y_pred_top3 before inverse transform
y_pred_labels = []
for row in y_pred_top3:
    y_pred_labels.extend(label_encoder.inverse_transform(row))
y_pred_labels = np.array(y_pred_labels).reshape(-1,3)

# Convert y_val back to original form
y_true_labels = label_encoder.inverse_transform(y_val.values.reshape(-1, 1))

# Calculate MAP@3
map3 = mapk([[label] for label in y_true_labels], y_pred_labels, k=3)
print(f"MAP@3: {map3}")


# Predict probabilities for the test set
y_pred_proba_test = rf_classifier.predict_proba(df_test_encoded.drop(columns=['id']))

# Get the top 3 predictions
y_pred_top3_test = np.argsort(y_pred_proba_test, axis=1)[:, ::-1][:, :3]

# Inverse transform the predictions
y_pred_labels_test = []
for row in y_pred_top3_test:
    y_pred_labels_test.extend(label_encoder.inverse_transform(row))
y_pred_labels_test = np.array(y_pred_labels_test).reshape(-1, 3)

# Create a list to store the formatted predictions
formatted_predictions = []
for i, id_val in enumerate(df_test_encoded['id']):
    fertilizer_names = ' '.join(y_pred_labels_test[i])
    formatted_predictions.append([id_val, fertilizer_names])

# Create submission dataframe
submission_df = pd.DataFrame(formatted_predictions, columns=['id', 'Fertilizer Name'])

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

display(submission_df.head())

