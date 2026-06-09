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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


train_df =pd.read_csv(r'/kaggle/input/ml-league-supervised-learning-competition/train.csv')
test_df = pd.read_csv('/kaggle/input/ml-league-supervised-learning-competition/test.csv')



train_df.head(5)


train_df.columns


train_df.info()


train_df.isna().sum().sort_values(ascending=False)/len(train_df) * 100


train_df


X = train_df.drop(columns=['yield', 'id', 'Row#'])
y = train_df['yield']
X_test = test_df.drop(columns=['id', 'Row#'])




print(X)
print(y)


print(type(X), X.shape)
print(type(y), y.shape)



np.array(X) #converting into a 2d array 


np.array(y) #converting into 1D array 



# Step 2: Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X)#make all values betwenn 0-1
x_test_scaled = scaler.transform(X_test)#make all values between 0-1 



param_dist = {
    'n_estimators': randint(100, 500),#no of trees,default =100
    'max_depth': randint(5, 50),#depth of each tree,default =none 
    'min_samples_split': randint(2, 10),#split a node,default =2
    'min_samples_leaf': randint(1, 10), #sample at leaf node,default =1
    'max_features': ['sqrt', 'log2', None] #max feature at each node split
    
}



rf = RandomForestRegressor(random_state=42)



random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,                 # Number of parameter settings to try
    cv=3,                      # 3-fold cross-validation
    verbose=1,
    random_state=42,           #keep the order of dataset same 
    n_jobs=-1                  # Use all available cores
)
'''verbose=1,to show minimal info while running'''



random_search.fit(X, y)#training for best parameters  



print("Best Parameters:", random_search.best_params_)







#train the model
model = RandomForestRegressor(max_depth=8, max_features=None, min_samples_leaf=8,
                      min_samples_split=4, n_estimators=354, random_state=42)
model.fit(x_train_scaled, y)

# Step 4: Predict on test set
y_pred = model.predict(x_test_scaled)



ids = test_df['id']
submission = pd.DataFrame({
    'id': ids,
    'yield':y_pred
})

# Save to CSV
submission.to_csv("submission.csv", index=False)







