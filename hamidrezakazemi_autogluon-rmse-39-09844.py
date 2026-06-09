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


!pip install featuretools



!pip install faiss-cpu



!pip install --force-reinstall --no-cache-dir numpy==1.24.4



!pip install --force-reinstall --no-cache-dir scikit-learn==1.2.2



# Import Libraries
import numpy as np
import pandas as pd
import faiss
import featuretools as ft
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


#Loading the Dataset
df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_trainExtra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')



#Let's concat the extra data train with train.csv
frame = [df_train , df_trainExtra]
df_train = pd.concat(frame)



# Save the id from test data
test_ids = df_test["id"].copy()
df_test.drop("id", inplace=True, axis=1)
df_train.drop("id", inplace=True, axis=1)



categorical_columns = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
label_encoders = {}

#Encoding for Categorical Features
for column in categorical_columns:
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column].astype(str))
    df_test[column] = le.transform(df_test[column].astype(str))
    label_encoders[column] = le

# KNN imputing via Faiss library
def faiss_knn_impute(data, k=5):
    mask = np.isnan(data)  #Finding Nan value
    index = faiss.IndexFlatL2(data.shape[1]) 
    index.add(data[~mask.any(axis=1)])  
    
    for i in range(data.shape[0]):
        if mask[i].any():  
            _, indices = index.search(data[i].reshape(1, -1), k)  
            neighbors = data[indices[0]]
            
            #use mean
            for j in range(data.shape[1]):
                if mask[i, j]:
                    data[i, j] = np.nanmean(neighbors[:, j])
    
    return data

# to float32
df_train[categorical_columns] = df_train[categorical_columns].astype(np.float32)
df_test[categorical_columns] = df_test[categorical_columns].astype(np.float32)


df_train[categorical_columns] = faiss_knn_impute(df_train[categorical_columns].values, k=5)
df_test[categorical_columns] = faiss_knn_impute(df_test[categorical_columns].values, k=5)


for column in categorical_columns:
    df_train[column] = label_encoders[column].inverse_transform(df_train[column].astype(int))
    df_test[column] = label_encoders[column].inverse_transform(df_test[column].astype(int))


numeric_columns = df_train.select_dtypes(include=[np.number]).columns
numeric_columns = [col for col in numeric_columns if col != "Price"]
df_train[numeric_columns] = faiss_knn_impute(df_train[numeric_columns].values, k=5)
df_test[numeric_columns] = faiss_knn_impute(df_test[numeric_columns].values, k=5)



# Remove "Price" column from the DataFrames before adding to EntitySet
df_train_with_price = df_train.copy() 
if "Price" in df_train.columns:
    df_train = df_train.drop(columns=["Price"])
    


# Create an explicit 'index' column if it doesn't exist
df_train['index_column'] = df_train.reset_index().index
df_test['index_column'] = df_test.reset_index().index

# Add DataFrame to EntitySet
es = ft.EntitySet(id="backpacks")
es = es.add_dataframe(dataframe_name="train", dataframe=df_train, index="index_column")
es = es.add_dataframe(dataframe_name="test", dataframe=df_test, index="index_column")



def custom_features(entityset, target_df):

    # Generate features
    feature_matrix, feature_defs = ft.dfs(
        entityset=entityset,
        target_dataframe_name=target_df,
        agg_primitives=["sum", "mean", "count", "mode", "std", "max", "min", "num_unique"],
        trans_primitives = [
    "divide_numeric",
    "multiply_numeric"
        ],
        features_only=False,
        verbose=True,
        max_depth=2,
        where_primitives=["count", "sum"]
    )
    
    # Process boolean columns correctly
    for col in ["Waterproof", "Laptop Compartment"]:
        if col in feature_matrix.columns:
            feature_matrix[col] = feature_matrix[col].fillna(True).astype(int)  # Convert boolean to int
    
    # Handle missing values
    feature_matrix.fillna(method='ffill', inplace=True)  # Forward fill missing values
    
    return feature_matrix

# Execute feature extraction for training and test data
df_train_features = custom_features(es, "train")
df_test_features = custom_features(es, "test")



#concat the Price column to df_train_features
df_train_features = df_train_features.reset_index(drop=True) # Reset index of df_train_features
df_train_with_price = df_train_with_price.reset_index(drop=True) # Reset index of df_train_with_price to ensure alignment
df_train_features = pd.concat([df_train_features, df_train_with_price["Price"]], axis=1)



!pip install ray==2.10.0



!pip install autogluon.tabular



!pip install -U ipywidgets



!pip install --force-reinstall --no-cache-dir scikit-learn==1.2.2
!pip install --force-reinstall --no-cache-dir numpy==1.24.4



from autogluon.tabular import TabularDataset, TabularPredictor



target = 'Price'

predictor = TabularPredictor(label=target, eval_metric='rmse',
                             problem_type="regression").fit(
    df_train_features,
    presets='best_quality',
    time_limit=3600 * 8,
    verbosity=3,
    ag_args_fit={'num_gpus': 1},
    hyperparameters={
        'NN_TORCH': {},  
        'GBM': {},    
        'XGB': {},
        "FASTAI" : {} 
 
    }
)

results = predictor.fit_summary()



# Print fit summary
print(results)



#See leaderboard
predictor.leaderboard()



y_pred = predictor.predict(df_test_features)



# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Price': y_pred
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)
submission.to_csv('submissionV1.csv', index=False)

# Display the first few rows of the predictions
print(submission.head(10))

