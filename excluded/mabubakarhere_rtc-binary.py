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


# Loading training and test data
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train_data.head() #Previewing the training data


#previewing the sample data
sample_data = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv") #Loading data
sample_data.head(10) #Previewing first 10 rows


from sklearn.ensemble import RandomForestClassifier #importing our model
p = train_data["y"] # Targated column
features = ["age", "job", "marital", "default", "balance", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "poutcome"] #Selecting the columns on the basis of which our model will predicts

#Convert our textual data into numerecal format. 
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Setting up our model
model = RandomForestClassifier(n_estimators = 142, max_depth = 20, random_state = 42)
model.fit(X, p) #Finds the relationship between our targat and features
predictions = model.predict_proba(X_test) #Making our predictions with percentage (_proba())
final_prediction = predictions[:, 1] #Slicing the 2d array so that we can get only results for positive value.

# Making predictions
outcome = pd.DataFrame({"id": test_data.id, "y": np.around(final_prediction, decimals = 1)}) #Making pandas dataframe to store our predictions and id
outcome.to_csv("submission.csv", index = False)#Stroing our dataframe in CSV file
print("Submission file saved successfully")


output = pd.read_csv("/kaggle/working/submission.csv")
output.head()

