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


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df.head()


df.fillna(0)


#df['Stage_fear'] = df['Stage_fear'].map({'No':0, 'Yes':1})


#df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No':0, 'Yes':1})


df_updated = pd.get_dummies(df, drop_first = True)
train_columns = df_updated.columns


df_updated.head()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is already defined and 'personality' column exists
target = df_updated['Personality_Introvert']
features = df_updated.drop('Personality_Introvert', axis=1)

# Compute correlations of each feature with 'personality'
correlations = features.corrwith(target)

# Plot the correlation bar graph
plt.figure(figsize=(10,6))
sns.barplot(x=correlations.index, y=correlations.values, palette='viridis')
plt.title('Correlation of Personality with Other Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.ylim(-1, 1)
plt.grid(True)
plt.tight_layout()
plt.show()


df_updated.drop('id',axis=1,inplace = True)


df_updated.fillna(0)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X = df_updated.drop('Personality_Introvert',axis = 1)
y = df_updated['Personality_Introvert']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = XGBClassifier()
model.fit(X_train,y_train)

model.score(X_test,y_test)



testdf = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
testdf.head()


testdf_updated = pd.get_dummies(testdf, drop_first=True)
testdf_updated = testdf_updated.reindex(columns=train_columns, fill_value=0)

testdf_updated.head(20)


testdf_updated.fillna(0,inplace = True)


final_df = pd.DataFrame()


final_df['id'] = testdf_updated['id']


testdf_updated.drop(['id', 'Personality_Introvert'], axis=1, inplace=True)


testdf_updated.head()



pred = model.predict(testdf_updated)



final_df['Personality'] = pred


final_df.head()


final_df['Personality'] = final_df['Personality'].map({0:'Extrovert', 1:'Introvert'})


final_df


final_df.to_csv('submission.csv', index = False)




