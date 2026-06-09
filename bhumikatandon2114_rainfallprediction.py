import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


train.info()


train.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = train.corr()

# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# Title
plt.title("Feature Correlation Heatmap")

# Show plot
plt.show()


train_df=train.drop(['id','day','winddirection','mintemp','rainfall'],axis=1)
y=train['id']


test_df=test.drop(['id','day','winddirection','mintemp'],axis=1)


test_df.info()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeRegressor
model = RandomForestClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


test_pred=model.predict(test_df)


test_pred


pred = pd.DataFrame({'id':test['id'],'rainfall': test_pred})

pred.to_csv('submit1.csv')


pred




