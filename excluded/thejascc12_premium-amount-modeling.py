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


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")


train.head()


test.head()


train['source'] = 'train'
test['source'] = 'test'


combined = pd.concat([train,test],ignore_index = True )


combined['Premium Amount']


combined.info()


combined.describe()


combined.isnull().sum()


cat_col = combined.select_dtypes(include=['object']).columns
cat_col


num_col = combined.select_dtypes(include=['int64','float64']).columns
num_col


for col in cat_col:
    combined[col] = combined[col].fillna("unknow") 


for col in num_col:
    combined[col] = combined[col].fillna(combined[col].median())


combined.isnull().sum()


print(combined['source'].unique())


for col in num_col:
    sns.boxplot(x=combined[col])
    plt.show()
    


from scipy import stats

# Define a function to remove outliers based on Z-score
def remove_outliers_z(combined, out_layer, threshold=3):
    # Iterate over each column in out_layer
    for column in out_layer:
        # Calculate Z-scores
        z_scores = stats.zscore(combined[column])
        
        # Filter the data based on Z-score threshold
        combined = combined[(np.abs(z_scores) < threshold)]
    
    return combined

# Example usage for a list of columns
out_layer = ['Premium Amount', 'Previous Claims', 'Annual Income']  # Replace with your numeric column names
combined_cleaned = remove_outliers_z(combined, out_layer)

# Now 'combined_cleaned' is your cleaned data, which you can use for further analysis
print(combined_cleaned)



combined = combined_cleaned


for col in out_layer:
    sns.boxplot(x=combined_cleaned[col])
    plt.show()


scale = StandardScaler()
num_col_to_scale = [col for col in num_col if col not in ['Premium Amount', 'id']]
combined[num_col_to_scale] = scale.fit_transform(combined[num_col_to_scale])



combined.head()


# Initialize the encoder
encoder = LabelEncoder()

# List of columns to encode, excluding 'source'
columns_to_encode = [col for col in combined.select_dtypes(include=['object']).columns if col != 'source']

# Apply LabelEncoder to the selected columns
for col in columns_to_encode:
    combined[col] = encoder.fit_transform(combined[col])

# Check the encoded columns
print(combined.head())



train_cleaned = combined[combined['source'] == 'train'].drop('source', axis=1)
test_cleaned = combined[combined['source'] == 'test'].drop('source', axis=1)



train_cleaned





train_cleaned.head()


x = train_cleaned.drop("Premium Amount", axis=1)
y = train_cleaned["Premium Amount"]
X_test_cleaned = test_cleaned.drop(columns=["id"])



print(x.shape)
print(y.shape)


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")



def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


rmsle_score = rmsle(y_test, y_pred)
print(f"RMSLE: {rmsle_score:.4f}")



X_test


submission = pd.DataFrame({
    'id': X_test['id'],  # Correct 'id' from the test set
    'Premium Amount': y_pred  # Predicted 'Premium Amount'
})
submission = submission[['id', 'Premium Amount']]
submission.to_csv('submission.csv', index=False)



submission





