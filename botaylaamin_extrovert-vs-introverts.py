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


pip install --upgrade scikit-learn imbalanced-learn


# for handel dataset
import pandas as pd
import numpy as np
# for visulaization
import matplotlib.pyplot as plt
import seaborn as sns
#for preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE  
from sklearn.metrics import accuracy_score, f1_score
# Disable warnings
import warnings
warnings.filterwarnings('ignore')


train_df  = pd.read_csv(r"/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv(r"/kaggle/input/playground-series-s5e7/test.csv")
sub_df = pd.read_csv(r"/kaggle/input/playground-series-s5e7/sample_submission.csv")


train_df.sample(10) # get random 10 samples of the train dataset


train_df.info()   # get summary info for train data


test_df.info() # get summary info for test data


# get statistical summary for train dataset
train_df.describe(include = 'all').T


# check for missing values  in train dataset
train_df.isnull().sum()


# check for missing values  in test dataset
test_df.isnull().sum()


# for column (Time_spent_Alone)
train_df["Time_spent_Alone"] = train_df["Time_spent_Alone"].replace(np.nan , 0)
test_df["Time_spent_Alone"] = test_df["Time_spent_Alone"].replace(np.nan , 0)

# for column (Stage_fear )
train_df["Stage_fear"] = train_df["Stage_fear"].replace(np.nan , "No")
test_df["Stage_fear"] = test_df["Stage_fear"].replace(np.nan , "No")

# for column (Social_event_attendance )
train_df["Social_event_attendance"] = train_df["Social_event_attendance"].replace(np.nan , 0)
test_df["Social_event_attendance"] = test_df["Social_event_attendance"].replace(np.nan , 0)

# for column (Going_outside )
train_df["Going_outside"] = train_df["Going_outside"].replace(np.nan , 0)
test_df["Going_outside"] = test_df["Going_outside"].replace(np.nan , 0)

# for column (Drained_after_socializing )
train_df["Drained_after_socializing"] = train_df["Drained_after_socializing"].replace(np.nan , "No")
test_df["Drained_after_socializing"] = test_df["Drained_after_socializing"].replace(np.nan , "No")

# for column (Friends_circle_size )
train_df["Friends_circle_size"] = train_df["Friends_circle_size"].replace(np.nan , 0)
test_df["Friends_circle_size"] = test_df["Friends_circle_size"].replace(np.nan , 0)

# for column (Post_frequency  )
train_df["Post_frequency"] = train_df["Post_frequency"].replace(np.nan , 0)
test_df["Post_frequency"] = test_df["Post_frequency"].replace(np.nan , 0)



# check for missing values  in test dataset
train_df.isnull().sum()


# check for duplicates
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(f"Number of duplicates in Train dataset : {train_duplicates}")
print(f"Number of duplicates in Test dataset : {test_duplicates}")


# define numerical , categorical , target columns

numerical_cols = train_df.select_dtypes(include =['int','float']).columns.drop("id")
categorical_cols = train_df.select_dtypes(include = ['O']).columns.drop("Personality")
target = "Personality"


train_df.info()


numerical_cols


fig, axes = plt.subplots(1,5 ,figsize=(20, 5))
# for box_plot 
for i,col in enumerate(numerical_cols) :
    sns.boxplot(data=train_df, y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
fig.tight_layout()  # Prevent overlap
plt.show()

fig, axes = plt.subplots(1,5 ,figsize=(20, 5))
# to display the distribution with histogram
for i, col in enumerate(numerical_cols) :
    sns.histplot(data=train_df, x=col, ax=axes[i],kde=True)
    axes[i].set_title(f"Histogram of {col}")

fig.tight_layout()
plt.show()


# check for skewness
train_df[numerical_cols].skew() 


#identify the number of outliers in (Time_spent_Alone) column

Q1 = train_df['Time_spent_Alone'].quantile(0.25)
Q3 = train_df['Time_spent_Alone'].quantile(0.75)
IQR = Q3 - Q1
outliers = train_df[(train_df['Time_spent_Alone'] > (Q3 + 1.5 * IQR)) | 
            (train_df['Time_spent_Alone'] < (Q1 - 1.5 * IQR))]
print(f"Number of outliers: {len(outliers)}")
print(f"The precentage of outliers: {round((len(outliers)/len(train_df))*100 , 2)} %")


categorical_cols


fig, axes = plt.subplots(1,2 ,figsize=(20, 5))
# countplot to show frequency for each category 
for i,col in enumerate(categorical_cols):
    sns.countplot(data = train_df , x =col , ax=axes[i])
    axes[i].set_title(f"Countplot for {col}")

fig.tight_layout()
plt.show()

print("\n"+100*'*'+"\n") # for displaying only

# Pie chart to show proportion of each category
fig, axes = plt.subplots(1,2 ,figsize=(20, 5))
for i,col in enumerate(categorical_cols):
    counts = train_df[col].value_counts()
    counts.plot.pie(autopct = '%1.1f%%',ax=axes[i])
    axes[i].set_title(f"Pie chart for {col}")

fig.tight_layout()
plt.show()
  


# analysis the relationship between (Drained_after_socializing) and (Personality)
pd.crosstab(train_df['Drained_after_socializing'], train_df['Personality'] , normalize ='index')


# analysis the relationship between (Drained_after_socializing) and (Personality)
pd.crosstab(train_df['Drained_after_socializing'], train_df['Personality'] ).plot.bar()


# analysis the relationship between (Stage_fear) and (Personality)
pd.crosstab(train_df['Stage_fear'], train_df['Personality'] , normalize = 'index')


pd.crosstab(train_df['Stage_fear'], train_df['Personality']).plot.bar()


# check for distribution of target column

counts = train_df[target].value_counts(normalize =True)
print("The frequency of each class : ")
print(counts)
counts.plot.bar()
plt.title(f"The Freuquency of target {target} classes")
plt.xticks(rotation=45)
plt.show()


# heatmap to show the relationship between columns and each other

corr_mat = train_df[numerical_cols].corr()  # calc correlation matrix for train data

#create the mask for the upper traingle 
mask = np.triu(np.ones_like(corr_mat,dtype=bool))

sns.heatmap(corr_mat , mask =mask , annot=True)
plt.show()


train_df = pd.get_dummies(train_df, columns=['Stage_fear', 'Drained_after_socializing'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Stage_fear', 'Drained_after_socializing'], drop_first=True)


le = LabelEncoder()
train_df['Personality'] = le.fit_transform(train_df['Personality'])


# # apply transformation to handle skewness
# train_df['Time_spent_Alone'] = np.log1p(train_df['Time_spent_Alone'])
# test_df['Time_spent_Alone'] = np.log1p(test_df['Time_spent_Alone'])

from sklearn.preprocessing import MinMaxScaler
# apply scalling
scaler = MinMaxScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


train_df.drop("id" , axis =1 , inplace = True)
test_id = test_df["id"]
test_df.drop("id" , axis =1 , inplace = True)


X = train_df.drop("Personality" , axis =1)
Y = train_df["Personality"]


train_df.info()





test = test_df
Y_test = sub_df["Personality"].map({'Extrovert' : 0  , "Introvert" :1})


import xgboost as xgb
from sklearn.metrics import accuracy_score


model = xgb.XGBClassifier()

model.fit(X,Y)


predictions = model.predict(test)

accuracy_score(Y_test ,predictions  )


models = {
    "Logistic Regression": LogisticRegression(max_iter=100,class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',scale_pos_weight=3),
    "LightGBM": LGBMClassifier(class_weight='balanced'),
    "CatBoost": CatBoostClassifier(verbose=0,auto_class_weights='Balanced')
}
results = []

for name, model in models.items():
    model.fit(X,Y)
    preds = model.predict(test)
    
    acc = accuracy_score(Y_test ,preds)
    f1 = f1_score(Y_test, preds)
    
    results.append((name, round(acc*100, 2), round(f1*100, 2)))



print(f"{'Model':<20} | {'Accuracy (%)':<12} | {'F1 Score (%)':<12}")
print("-" * 50)
for r in results:
    print(f"{r[0]:<20} | {r[1]:<12} | {r[2]:<12}")


smote = SMOTE(random_state=42)  
X_resampled, y_resampled = smote.fit_resample(X, Y)  


model = LogisticRegression(max_iter=100,class_weight="balanced")
model.fit(X,Y)


predictions = model.predict(test_df)
predictions



le.inverse_transform(predictions)


sub_df["Personality"]


output = pd.DataFrame({'id': test_id, 'Personalities': le.inverse_transform(predictions)})
output.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

