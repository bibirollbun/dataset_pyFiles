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


# Ignore the warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


# Load the Train Data

train_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
train_quant=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
train_categorical=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_solution=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")


train_categorical.info()


## Load the Test Data
test_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_categorical=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_quant=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")


# test_categorical.info()
print(test_categorical.columns)
test_categorical.head()


# Perform the join on the 'ID' feature
test_cat_quant = pd.merge(test_categorical, test_quant, on='participant_id', how='inner')


# Join train_categorical and train_quant on participant_id
# Perform the join on the 'ID' feature
train_cat_quant = pd.merge(train_categorical, train_quant, on='participant_id', how='inner')
test_cat_quant = pd.merge(test_categorical, test_quant, on='participant_id',how='inner')


test_cat_quant['Dataset'] = 'Test'
train_cat_quant['Dataset'] = 'Train'

# Union the two DataFrames
combined_cat_quant = pd.concat([test_cat_quant, train_cat_quant], ignore_index=True)
combined_cat_quant.fillna(combined_cat_quant.median(numeric_only=True), inplace=True)


# Fill nulls with median

combined_cat = combined_cat_quant.copy()  # Create a copy of the DataFrame
combined_cat = combined_cat.fillna(combined_cat_quant.median(numeric_only=True))



# Splitting the dataframe based on a condition 
feature_name = 'Dataset' 
v_split = 'Test' 

# Create two dataframes based on the condition
df_split_test = combined_cat[combined_cat[feature_name] == v_split]
df_split_train = combined_cat[combined_cat[feature_name] != v_split]


# Append Solution to df_split_train
train_solution = pd.merge(df_split_train, train_solution, on='participant_id', how='inner')
train_conn_solution = pd.merge(train_solution,train_connectome, on='participant_id', how='inner')
train_conn = pd.merge(df_split_test,test_connectome, on='participant_id')


# Append connectnomes to df_split_test
test_conn = pd.merge(df_split_test,test_connectome, on='participant_id', how='inner')


#drop dataset from train_solution
train_solution = train_solution.drop(columns=['Dataset'])


# Create a new dataset without the 'participant_id' and any features with too many values for a frequency plot to make sense
merged_data = train_solution.drop(columns=['participant_id','EHQ_EHQ_Total','MRI_Track_Age_at_Scan'])


# Drop non-feature columns
X = train_conn_solution.drop(columns=['participant_id', 'ADHD_Outcome', 'Sex_F'])
y_adhd = train_conn_solution['ADHD_Outcome']
y_gender = train_conn_solution['Sex_F']  
X_test = train_conn_solution.drop(columns=['participant_id'])


# Create a new dataset without the 'participant_id' and any features with too many values for a frequency plot to make sense
merged_data = train_solution.drop(columns=['participant_id','EHQ_EHQ_Total','MRI_Track_Age_at_Scan'])


# Encode categoricals
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)


X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)


from sklearn.preprocessing import StandardScaler

# Standardize Features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# Check F1-Score of Train Data
# Create ADHD_Outcome and Sex_f as singluar datasets to evaluate the fit of each prediction
from sklearn.metrics import f1_score
actual_adhd = train_conn_solution[['ADHD_Outcome']]
actual_sex = train_conn_solution[['Sex_F']]




# Drop non-feature columns
X = train_conn_solution.drop(columns=['participant_id', 'ADHD_Outcome', 'Sex_F'])
y_adhd = train_conn_solution['ADHD_Outcome']
y_gender = train_conn_solution['Sex_F']  
X_test = train_conn_solution.drop(columns=['participant_id'])





#check learning curves

history_df = pd.DataFrame(model_hist.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense





# Convert probabilities to binary outcomes
predicted_ADHD = (predictions[:, 0] > 0.5).astype(int)
predicted_Sex_F = (predictions[:, 1] > 0.5).astype(int)

train_predicted_ADHD = (train_predictions[:, 0] > 0.5).astype(int)
train_predicted_Sex_F = (train_predictions[:, 1] > 0.5).astype(int)


# Create a DataFrame with participant_id and predictions
test_predictions = test_conn[['participant_id']].copy()
test_predictions['ADHD_Outcome'] = predicted_ADHD
test_predictions['Sex_F'] = predicted_Sex_F

# Save results to CSV
test_predictions.to_csv("submission.csv", index=False)



from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# Generate sample data
X, y = make_regression(n_features=5, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Access feature importances
importances = model.feature_importances_

# Print feature importances
print(importances)


# Calculate F1 Score for adhd
adhd_f1 = f1_score(actual_adhd, train_predicted_ADHD)

print("F1 Score:", adhd_f1)


# Calculate F1 Score for female
sex_f1 = f1_score(actual_sex, train_predicted_Sex_F)

print("F1 Score:", sex_f1)

