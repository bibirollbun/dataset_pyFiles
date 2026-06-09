import pandas as pd
import numpy as np
import random
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , f1_score

import warnings
warnings.filterwarnings("ignore")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
#df = pd.read_csv("train.csv")
df.head()


df.shape


test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
#test_data = pd.read_csv("test.csv")
test_data.head()


test_data.shape


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
#sample_sub = pd.read_csv("sample_submission.csv")
sample_sub.head()


df.head(10)


df[df['Personality'] == 'Introvert'][['Personality' , 'Drained_after_socializing']].head(20)

# for Introvert Drained_after_socializing is Yes.


df[df['Personality'] != 'Introvert'][['Personality' , 'Drained_after_socializing']].head(20)

# for extrovert Drained_after_socializing is No.


df[df.Drained_after_socializing.isna()]['Personality'].value_counts()


df['Drained_after_socializing'].isna() & (df['Personality'] == 'Introvert')


df.loc[df['Drained_after_socializing'].isna() & (df['Personality'] == 'Introvert'), 'Drained_after_socializing'] = 'Yes'

df.loc[df['Drained_after_socializing'].isna() & (df['Personality'] == 'Extrovert'), 'Drained_after_socializing'] = 'No'



df['Drained_after_socializing'].isna().sum()


df[df['Personality'] != 'Introvert'][['Personality' , 'Stage_fear']].head(20)


df[df['Personality'] == 'Introvert'][['Personality' , 'Stage_fear']].head(20)


df.loc[df['Stage_fear'].isna() & (df['Personality'] == 'Introvert'), 'Stage_fear'] = 'Yes'

df.loc[df['Stage_fear'].isna() & (df['Personality'] == 'Extrovert'), 'Stage_fear'] = 'No'


df.head()


df.groupby(['Time_spent_Alone', 'Personality']).size().reset_index(name='Count')

#EX - 3
#In - 8


import random

# For extroverts with missing 'Time_spent_Alone', assign a random int between 0 and 3
df.loc[df['Time_spent_Alone'].isna() & (df['Personality'] == 'Extrovert'), 'Time_spent_Alone'] = [
    random.randint(0, 3) for _ in range((df['Time_spent_Alone'].isna() & (df['Personality'] == 'Extrovert')).sum())
]

# For introverts, assign a random int between 4 and 8
df.loc[df['Time_spent_Alone'].isna() & (df['Personality'] != 'Extrovert'), 'Time_spent_Alone'] = [
    random.randint(4, 8) for _ in range((df['Time_spent_Alone'].isna() & (df['Personality'] != 'Extrovert')).sum())
]



df.groupby(['Social_event_attendance', 'Personality']).size().reset_index(name='Count')


df.loc[df['Social_event_attendance'].isna() & (df['Personality'] == 'Extrovert'), 'Social_event_attendance'] = [
    random.randint(5, 9) for _ in range((df['Social_event_attendance'].isna() & (df['Personality'] == 'Extrovert')).sum())
]

df.loc[df['Social_event_attendance'].isna() & (df['Personality'] != 'Extrovert'), 'Social_event_attendance'] = [
    random.randint(0, 3) for _ in range((df['Social_event_attendance'].isna() & (df['Personality'] != 'Extrovert')).sum())
]


df.groupby(['Going_outside', 'Personality']).size().reset_index(name='Count')



df.loc[df['Going_outside'].isna() & (df['Personality'] == 'Extrovert'), 'Going_outside'] = [
    random.randint(3, 7) for _ in range((df['Going_outside'].isna() & (df['Personality'] == 'Extrovert')).sum())
]

df.loc[df['Going_outside'].isna() & (df['Personality'] != 'Extrovert'), 'Going_outside'] = [
    random.randint(0, 2) for _ in range((df['Going_outside'].isna() & (df['Personality'] != 'Extrovert')).sum())
]


df.head()


df.groupby(['Friends_circle_size', 'Personality']).size().reset_index(name='Count')


df.loc[df['Friends_circle_size'].isna() & (df['Personality'] == 'Extrovert'), 'Friends_circle_size'] = [
    random.randint(7, 14) for _ in range((df['Friends_circle_size'].isna() & (df['Personality'] == 'Extrovert')).sum())
]

df.loc[df['Friends_circle_size'].isna() & (df['Personality'] != 'Extrovert'), 'Friends_circle_size'] = [
    random.randint(0, 5) for _ in range((df['Friends_circle_size'].isna() & (df['Personality'] != 'Extrovert')).sum())
]


df.groupby(['Post_frequency', 'Personality']).size().reset_index(name='Count')


df.loc[df['Post_frequency'].isna() & (df['Personality'] == 'Extrovert'), 'Post_frequency'] = [
    random.randint(4, 9) for _ in range((df['Post_frequency'].isna() & (df['Personality'] == 'Extrovert')).sum())
]

df.loc[df['Post_frequency'].isna() & (df['Personality'] != 'Extrovert'), 'Post_frequency'] = [
    random.randint(0, 3) for _ in range((df['Post_frequency'].isna() & (df['Personality'] != 'Extrovert')).sum())
]


test_data.info()

# Since we have Na in test we need to do Imputation in Pipeline


df.info()
# Imputed


df.Personality.value_counts('NORMALIZE')*100

# slightly imbalance


df.info()


df.head()


df.columns


sns.boxplot(data=df, x='Personality', y='Time_spent_Alone')
plt.title('Time Spent Alone Distribution by Personality')
plt.show()


mean_time = df.groupby('Personality')['Time_spent_Alone'].mean().reset_index()

sns.barplot(data=mean_time, x='Personality', y='Time_spent_Alone')
plt.title('Average Time Spent Alone by Personality')
plt.show()




numeric_cols = df.select_dtypes(include=['number']).columns
sns.heatmap(df[numeric_cols].corr(), annot=True)


df.head()


# Custom transformer for Yes/No and Personality
df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})


test_data.head()


df.head()





def text_to_int(df):
    df = df.map({'Yes':1 , 'No':0})
    return df


df['Stage_fear'] =text_to_int (df['Stage_fear'])
df['Drained_after_socializing'] =text_to_int (df['Drained_after_socializing'])

test_data['Stage_fear'] =text_to_int (test_data['Stage_fear'])
test_data['Drained_after_socializing'] =text_to_int (test_data['Drained_after_socializing'])


df.head()


test_data.head()


test_data.isna().sum()


LR = LogisticRegression()


num_cols = ['Time_spent_Alone', 'Going_outside', 'Post_frequency', 'Friends_circle_size']
binary_cols = ['Stage_fear', 'Social_event_attendance', 'Drained_after_socializing']


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])


binary_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('bin', binary_pipeline, binary_cols)
])






pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])





x = df.drop(columns = ['Personality','id'] )
y = df['Personality']



x_train , x_test , y_train , y_test =train_test_split ( x, y , random_state= 42 , test_size= 0.3 )


pipeline


pipeline.fit(x_train , y_train)


train_predict = pipeline.predict(x_train)



test_predict = pipeline.predict(x_test)


print('Train F1_score:',f1_score(y_train , train_predict))


print('Test F1_score:',f1_score(y_test , test_predict))


    print('Train Accuracy_score:\n',accuracy_score(y_train , train_predict))
    print('Test Accuracy_score:\n',accuracy_score(y_test , test_predict))
    print('Train confusion_matrix:\n',confusion_matrix(y_train , train_predict))
    print('Test confusion_matrix:\n',confusion_matrix(y_test , test_predict))





pipeline_1 = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression())
])


pipeline_1.fit(x_train , y_train)


train_predict_log = pipeline_1.predict(x_train)
test_predict_log = pipeline_1.predict(x_test)



print('Train F1_score:',f1_score(y_train , train_predict_log))
print('Test F1_score:',f1_score(y_test , test_predict_log))


    print('Train Accuracy_score:\n',accuracy_score(y_train , train_predict_log))
    print('Test Accuracy_score:\n',accuracy_score(y_test , test_predict_log))
    print('Train confusion_matrix:\n',confusion_matrix(y_train , train_predict_log))
    print('Test confusion_matrix:\n',confusion_matrix(y_test , test_predict_log))


pipeline_2 = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators= 200 , min_samples_split= 4 , min_samples_leaf= 3 , max_features= 'sqrt'))
])


pipeline_2.fit(x_train , y_train)
train_predict_RF = pipeline_2.predict(x_train)
test_predict_RF = pipeline_2.predict(x_test)



    print('Train F1_score:',f1_score(y_train , train_predict_RF))
    print('Test F1_score:',f1_score(y_test , test_predict_RF))
    print('Train Accuracy_score:\n',accuracy_score(y_train , train_predict_RF))
    print('Test Accuracy_score:\n',accuracy_score(y_test , test_predict_RF))
    print('Train confusion_matrix:\n',confusion_matrix(y_train , train_predict_RF))
    print('Test confusion_matrix:\n',confusion_matrix(y_test , test_predict_RF))


from sklearn.ensemble import AdaBoostClassifier , VotingClassifier , StackingClassifier
import xgboost as xgb


def results(x_train , x_test , y_train , y_test , model):
    model.fit(x_train,y_train)
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    print('Train F1_score:',f1_score(y_train , train_predict))
    print('Test F1_score:',f1_score(y_test , test_predict))
    print('Train Accuracy_score:\n',accuracy_score(y_train , train_predict))
    print('Test Accuracy_score:\n',accuracy_score(y_test , test_predict))
    print('Train confusion_matrix:\n',confusion_matrix(y_train , train_predict))
    print('Test confusion_matrix:\n',confusion_matrix(y_test , test_predict))


pipeline_Ada = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', AdaBoostClassifier())
])


results(x_train , x_test , y_train , y_test , pipeline_Ada)


pipeline_xgb = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', xgb.XGBClassifier())
])


results(x_train , x_test , y_train , y_test , pipeline_xgb)


from sklearn.model_selection import GridSearchCV


param_grid ={
    'n_estimators':[120 , 140 , 160],
    'min_samples_split':[3,5],
    'min_samples_leaf':[3,5],
    'max_features':[4 , 5]
}


grid = GridSearchCV( estimator= RandomForestClassifier() , param_grid= param_grid , cv= 10 , verbose= 2 , scoring= 'accuracy')


grid.fit(x_train,y_train)


grid.best_estimator_


grid.best_score_


grid.feature_names_in_


pipeline_RF_CV = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(max_features=4, min_samples_leaf=3, min_samples_split=3,
                       n_estimators=160))
])


results(x_train , x_test , y_train , y_test , pipeline_RF_CV)


test_data.head()


test_data_1 = test_data.drop(columns = 'id')


test_data_1


val_predict = pipeline_1.predict(test_data_1)


val_predict


sample_sub.head()


sample_sub['Personality'] = pd.DataFrame(val_predict)


sample_sub


sample_sub['Personality'] = sample_sub['Personality'].map({1:'Extrovert', 0:'Introvert'})


sample_sub


sample_sub.to_csv('Final_pre_log.csv',index=False)





num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])


binary_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('bin', binary_pipeline, binary_cols)
])


pipeline_RF_CV_without_scale = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(max_features=4, min_samples_leaf=3, min_samples_split=3,
                       n_estimators=160))
])


results(x_train , x_test , y_train , y_test , pipeline_RF_CV_without_scale)


pipeline_RF_CV_without_scale = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(max_features=5, min_samples_leaf=2, min_samples_split=2,
                       n_estimators=160))
])


results(x_train , x_test , y_train , y_test , pipeline_RF_CV_without_scale)


clf1 = xgb.XGBClassifier()
clf2 = xgb.XGBClassifier(max_depth=5 , min_samples_leaf= 10 , min_samples_split= 10)
clf3 = RandomForestClassifier(max_depth=20 , min_samples_leaf= 5 , min_samples_split= 5)
clf4 = RandomForestClassifier(max_features=5, min_samples_leaf=2, min_samples_split=2,n_estimators=160)
clf5 = RandomForestClassifier(max_depth=15 , min_samples_leaf= 5 , min_samples_split= 5)
clf6 = xgb.XGBClassifier(max_depth=10 , min_samples_leaf= 8 , min_samples_split= 8)


voting = VotingClassifier(estimators=
                          [
                              ('XGB1',clf1),
                              ('XGB2',clf2),
                              ('RF1',clf3),
                              ('RF2',clf4),
                              ('RF3',clf5),
                              ('XGB3',clf6)
                          ],voting='soft',n_jobs=-1)


results(x_train , x_test , y_train , y_test , voting )


x.head()


y.head()


test_data_1.head()


# !pip install -U scikit-learn imbalanced-learn
# #


# !pip install --upgrade scikit-learn==1.3.2 imbalanced-learn==0.12.0



# from imblearn.over_sampling import RandomOverSampler


# over_sampeling = RandomOverSampler()
# x_os , y_os = over_sampeling.fit_resample(x , y)


# x.shape


# x_os.shape


# train_os = pd.concat([x_os , y_os],axis=1)


# df['Personality'].value_counts()


# train_os['Personality'].value_counts()


#x_train , x_test , y_train , y_test = train_test_split(x_os , y_os , test_size= 0.3 , random_state= 42)

# over sampleing intrude biase.


# results(x_train , x_test , y_train , y_test , pipeline_2 )


from sklearn.neighbors import KNeighborsClassifier


pipeline_Knn = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', KNeighborsClassifier( n_neighbors= 10))
])


results(x_train , x_test , y_train , y_test , pipeline_Knn )


pipeline_Knn = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors= 8))
])


results(x_train , x_test , y_train , y_test , pipeline_Knn )


results(x_train , x_test , y_train , y_test , voting )




