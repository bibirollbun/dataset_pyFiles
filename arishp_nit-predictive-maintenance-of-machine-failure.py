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
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s3e17/train.csv")


df.head()


df.info()


df.describe()


cols  = df.columns
for i in cols[2:]:
    if(df[i].dtype.kind in 'if' and df[i].value_counts().count()>=5):
        sns.histplot(data=df,x=i,hue='Machine failure',kde=True)
    else:
        df[i] = df[i].astype(str)
        sns.countplot(data=df,x=i,hue="Machine failure",dodge=True)
    plt.show()
    


sns.countplot(data=df,x='Machine failure')
plt.show()


## Downsampling
from sklearn.utils import resample
majority_class = df[df['Machine failure'] == '0']  
minority_class = df[df['Machine failure'] == '1']  
majority_downsampled = resample(
    majority_class, 
    replace=False,    
    n_samples=len(minority_class), 
    random_state=42   
)

df_downsampled = pd.concat([majority_downsampled, minority_class])

df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

print("Original dataset size:", len(df))
print("Downsampled dataset size:", len(df_downsampled))
print(df_downsampled['Machine failure'].value_counts())


new_df = df_downsampled


new_df = new_df.iloc[:,2:]


new_df.dtypes


from sklearn.preprocessing import LabelEncoder
for i in new_df.columns:
    if(new_df[i].dtype.kind not in 'if'):
        new_df[i] = LabelEncoder().fit_transform(new_df[i])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
X = new_df.drop(columns=['Machine failure'])  
y = new_df['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': report['1']['precision'],  
        'Recall': report['1']['recall'],        
        'F1-Score': report['1']['f1-score']  
    }

results_df = pd.DataFrame(results).T
print(results_df)


model = models['Random Forest']


test = pd.read_csv("/kaggle/input/playground-series-s3e17/test.csv")


test.head()


ids = test.iloc[:,0]


test = test.iloc[:,2:]


for i in test.columns:
    if(test[i].dtype.kind not in 'if'):
        test[i] = LabelEncoder().fit_transform(test[i])


preds = model.predict(test)


dic = {
    "ids":ids,
    "prediction":preds
}


data_frame = pd.DataFrame(dic)


data_frame['prediction']




