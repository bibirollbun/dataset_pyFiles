import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as plx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('/kaggle/input/playground-series-s3e3/train.csv')


df.head()


df.tail()


df.shape


df.isnull().sum()


df.duplicated().sum()


df.describe()


df['Attrition'].value_counts()


df = df.drop('id',axis = 1)


df_new = df[['BusinessTravel', 'Department', 'Education', 'EducationField',
            'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobRole',
            'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked', 'OverTime',
             'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
             'TrainingTimesLastYear', 'WorkLifeBalance', 'Attrition']]


for col in df_new.columns:
    plt.figure(figsize = (10,6))
    sns.countplot(x = col,data = df_new,palette = 'hls')
    plt.xticks(rotation = 90)
    plt.show()


for col in df_new.columns:
    plt.figure(figsize = (10,6))
    print("Pieplot for :",col)
    df_new[col].value_counts().plot(kind = 'pie',autopct = '%1.1f%%')
    plt.show()


int_cols = [col for col in df.columns if df[col].dtype == 'int64']
print('Integer columns:', int_cols)

print('\n')

obj_cols = [col for col in df.columns if df[col].dtype == 'object']
print('Object columns:', obj_cols)


plt.figure(figsize = (15,7))
sns.histplot(df['Age'],kde = 'True',bins = 5,palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


fig = plx.box(df,x = 'Attrition',y = 'Age')
fig.show()


fig = plx.box(df,x = 'Attrition',y = 'DailyRate')
fig.show()


fig = plx.box(df,x = 'Attrition',y = 'DistanceFromHome')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'Education')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'EnvironmentSatisfaction')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'JobLevel')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'JobSatisfaction')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'MonthlyIncome')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'PercentSalaryHike')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'RelationshipSatisfaction')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'TotalWorkingYears')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'WorkLifeBalance')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'YearsAtCompany')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'YearsInCurrentRole')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'YearsSinceLastPromotion')
fig.show()


fig = plx.box(df, x = 'Attrition', y = 'YearsWithCurrManager')
fig.show()


categorical_cols = df.select_dtypes(include=['object', 'category'])

# Get number of unique values per categorical column
unique_counts = categorical_cols.nunique()

# Display result
print(unique_counts)


le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


df.head()


X = df.drop('Attrition',axis = 1)
y = df['Attrition']


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state = 0)


model = ExtraTreesClassifier()
model.fit(X_train,y_train)


y_pred = model.predict(X_val)


print("Accuracy score:",accuracy_score(y_pred,y_val))


test_df = pd.read_csv('/kaggle/input/playground-series-s3e3/test.csv')


test_df_new = test_df.drop('id',axis = 1)


test_df_new


le = LabelEncoder()
for col in categorical_cols:
    test_df_new[col] = le.fit_transform(test_df_new[col])


y_test = model.predict(test_df_new)



submission = pd.DataFrame({
    'id': range(len(y_test)),
    'target': y_test  # replace 'target' with the actual column name required by the Kaggle competition
})

# Save to CSV (without index)
submission.to_csv("submission.csv", index=False)


