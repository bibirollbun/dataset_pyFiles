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


train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
train.head()


train.isna().sum()


train=train.drop_duplicates()
train.shape


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Assume you already have a DataFrame named df
# df = pd.read_csv("your_data.csv")  # Uncomment if loading from CSV

# Scatter plot using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train, x='Heart_Rate', y='Calories', color='red', alpha=0.6)

# Add a regression line (optional)
#sns.regplot(data=train, x='Heart_Rate', y='Calories', scatter=False, color='blue')

# Labels and title
plt.title('Heart Rate vs Calories Burned')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Calories Burned')
plt.grid(True)
plt.tight_layout()
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Assume your DataFrame is named df
# df = pd.read_csv("your_data.csv")  # Uncomment if needed

# Compute correlation matrix
corr_matrix = train.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Titles and layout
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()



train.columns



train.describe()


train.info()


features=['Duration', 'Heart_Rate','Body_Temp']
X=train[features]
y=train['Calories']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)





from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Save `le` and use the same one during prediction:
y_test = le.transform(y_test)


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier()
model = XGBClassifier(gamma=0,learning_rate=0.01,max_depth=5,min_child_weight=3,subsample=1.0)


model = XGBClassifier(
    gamma=0,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=3,
    subsample=1.0,
    n_estimators=500,        # More trees to compensate for low learning rate
    tree_method='hist',      # ✅ Much faster on CPU
    n_jobs=-1,               # ✅ Use all CPU cores
    verbosity=1              # Optional: show training progress
)


#model = XGBClassifier(gamma=0,learning_rate=0.01,max_depth=5,min_child_weight=3,subsample=1.0)

#model = XGBClassifier()
model.fit(x_train,y_train)
#predictions = model.predict(x_test)


predictions = model.predict(x_test)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test.head()


X_test=pd.get_dummies(test[features])


pred = model.predict(X_test)


output = pd.DataFrame({'id': test.id, 'Calories': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

