# Import the necessary libararies

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import datasets 

train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original_data = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


original_data.reset_index(drop=False, inplace=True)
original_data.rename(columns={'index': 'id'}, inplace=True)
original_data.head()


original_data.shape


# Check the shape of all datasets

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("submission_data shape :",submission_data.shape)


# Visualize training data 

train_data.head()


# Explore training data 
train_data.info()
train_data.describe()



def feature_preprocessing(train_data):
    categorical_features = ['Soil Type', 'Crop Type']
    target_column = 'Fertilizer Name'

    # Encode target
    label_encoder = LabelEncoder()
    train_data['Fertilizer_Label'] = label_encoder.fit_transform(train_data[target_column])

    # One-hot encode categorical features
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse=False),
                                                    categorical_features) ],
                                     remainder='passthrough' )

    # Select only features (excluding target)
    X_train = train_data.drop(columns=[target_column])
    
    # Apply one-hot encoding
    X_train_encoded = preprocessor.fit_transform(X_train)

    # Convert sparse matrix to dense DataFrame with column names
    feature_names = preprocessor.get_feature_names_out()
    encoded_train_data = pd.DataFrame(X_train_encoded, columns=feature_names)

    # Add the encoded label for correlation
    encoded_train_data['Fertilizer_Label'] = train_data['Fertilizer_Label']

    return encoded_train_data


# Compute correlation
encoded = feature_preprocessing(train_data)

# Get correlation with the target label
correlations = encoded.corr(numeric_only=True)['Fertilizer_Label'].sort_values(ascending=False)

print(correlations)




# Sort correlations (excluding target itself)
sorted_corr = correlations.drop('Fertilizer_Label').sort_values()

# Color mapping: red for negative, blue for positive
colors = ['red' if val < 0 else 'blue' for val in sorted_corr]

plt.figure(figsize=(14, 6))
sorted_corr.plot(kind='bar', color=colors)

plt.axhline(0, color='black', linewidth=1)  # horizontal line at 0
plt.title('Feature Correlation with Fertilizer_Label')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Let's split the train data to X_train and y_train

X_train = train_data.drop(["Fertilizer Name","Fertilizer_Label"], axis=1) 
y_train = train_data["Fertilizer Name"] 


# View X_train 

X_train.head()


# View y_train

y_train.head()


# Check X_train data types

X_train.dtypes 


# Build the pipeline

categorical_features = ['Soil Type', 'Crop Type']
target_column = 'Fertilizer Name'

# Encode target
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_data[target_column])

# One-hot encode categorical features
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse=False),
                                                    categorical_features) ], remainder='passthrough' )



pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100))
])

# Fit the pipeline

pipeline.fit(X_train, y_train_encoded)


# Import test data using pandas 

test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
X_test = test_data      

y_preds = pipeline.predict(X_test)



# Predict probabilities

probs = pipeline.predict_proba(X_test)

# Get top 3 predictions

top_3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Get top-3, descending order
top_3_preds = label_encoder.inverse_transform(top_3_indices.ravel()).reshape(-1, 3)

# Combine predictions

test_ids = test_data['id']
submission = pd.DataFrame({
    'id': test_ids,  # assuming test_ids is a list or Series of IDs
    'Fertilizer Name': [' '.join(row) for row in top_3_preds]
})

# Save submission file

submission.to_csv('submission.csv', index=False)
submission.set_index('id', inplace=True)
submission.head()





