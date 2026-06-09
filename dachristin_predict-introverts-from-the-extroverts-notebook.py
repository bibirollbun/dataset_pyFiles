#importing initial packagaes
import pandas as pd
import numpy as np

#reading in the data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
train.head()




#Viewing the break down of the dataset
train.info()


#looking at the count of the target variable to check catagory balance
train.Personality.value_counts()


#looking at the percentage of each
train.Personality.value_counts()/train.Personality.count()


#looking into the amount of NAs for each column
train.isna().sum()


#looking into the unique values for each variable
train.nunique()


#looking into the specific unique values of each variable, checking for inconsistencies
train['Time_spent_Alone'].unique()


train['Stage_fear'].unique()


train['Social_event_attendance'].unique()


train['Going_outside'].unique()


train['Drained_after_socializing'].unique()


train['Friends_circle_size'].unique()


train['Post_frequency'].unique()


#finding the initial means of the variables to later check my work
Time_spent_Alone_mean = train.groupby('Personality')['Time_spent_Alone'].mean()
print(Time_spent_Alone_mean)

Social_event_attendance_mean = train.groupby('Personality')['Social_event_attendance'].mean()
print(Social_event_attendance_mean)

Going_outside_mean = train.groupby('Personality')['Going_outside'].mean()
print(Going_outside_mean)

Friends_circle_size_mean = train.groupby('Personality')['Friends_circle_size'].mean()
print(Friends_circle_size_mean)

Post_frequency_mean = train.groupby('Personality')['Post_frequency'].mean()
print(Post_frequency_mean)




#finding the mode of yes's and no's for the object variables.
Stage_fear_mode = train.groupby('Personality')['Stage_fear'].agg(lambda x: x.mode().tolist())
print(Stage_fear_mode)

Drained_after_socializing_mode = train.groupby('Personality')['Drained_after_socializing'].agg(lambda x: x.mode().tolist())
print(Drained_after_socializing_mode)



#building a function to find group means of multiple variables
def na_means(df,group_col,target_cols):
    for col in target_cols:
        df[col] = df[col].fillna(df.groupby(group_col)[col].transform('mean').round())
        
    return df


#Calling the previous function
train_num_filled = na_means(train, group_col = 'Personality',target_cols=['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency'])
train_num_filled.isna().sum()


#train.loc[<condition>, <column>] = <value>
#Filling in 'Stage_fear' NAs
train_num_filled.loc[(train_num_filled['Personality']=='Extrovert') & (train_num_filled['Stage_fear'].isna()), 'Stage_fear'] = 'No'
train_num_filled.loc[(train_num_filled['Personality']=='Introvert') & (train_num_filled['Stage_fear'].isna()), 'Stage_fear'] = 'Yes'

#Filling in 'Drained_after_socializing NAs
train_num_filled.loc[(train_num_filled['Personality']=='Extrovert') & (train_num_filled['Drained_after_socializing'].isna()), 'Drained_after_socializing'] = 'No'
train_num_filled.loc[(train_num_filled['Personality']=='Introvert') & (train_num_filled['Drained_after_socializing'].isna()), 'Drained_after_socializing'] = 'Yes'

#Checking to see NAs have been cleared
train_num_filled.isna().sum()



#pulling apart extrovert and introvert rows
extroverts = train_num_filled[train_num_filled['Personality']=='Extrovert']
introverts = train_num_filled[train_num_filled['Personality']=='Introvert']

#randomly sampling 4825 Extroverts, the same amount of introverts
ex_sample = extroverts.sample(4825, random_state=42)

#putting introverts and extroverts back together
balanced = pd.concat([ex_sample,introverts])

#resetting the index
balanced = balanced.reset_index(drop=True)

#checking the new dataframe's personalities are balances
balanced['Personality'].value_counts()


#factoring the balance set to later be used in models
balanced['Stage_fear'] = balanced['Stage_fear'].map({'Yes': 1, 'No': 0})
balanced['Drained_after_socializing'] = balanced['Drained_after_socializing'].map({'Yes':1, 'No':0})


#sampling 70% of the balanced data to remain the training dataset
train_bl = balanced.sample(frac=.7,random_state=42)

#removing the training rows from the balanced dataset, the remaining rows are the validation set
vali_bl = balanced.drop(train_bl.index)


#checking the personality counts to make sure decent balance within training and validation sets
train_bl['Personality'].value_counts()


vali_bl['Personality'].value_counts()


#downloading the needed packages
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

#removing the labels from the testing and training set
x_train = train_bl.drop('Personality', axis=1)
y_train = train_bl['Personality']
x_val = vali_bl.drop('Personality', axis=1)
y_val = vali_bl['Personality']

#building the decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train,y_train)


#predicting with the model
dt_pred = dt_model.predict(x_val)

#evaluate the model
from sklearn.metrics import ConfusionMatrixDisplay
print(classification_report(y_val,dt_pred))
ConfusionMatrixDisplay.from_predictions(y_val, dt_pred)


#building the model
ft_model = RandomForestClassifier(random_state=42)
ft_model.fit(x_train,y_train)

#predicting
ft_pred = ft_model.predict(x_val)

#evaluating the model
print(classification_report(y_val, ft_pred))
ConfusionMatrixDisplay.from_predictions(y_val, ft_pred)


#standardizing my datasets with scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)

#creating the model
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train_scaled, y_train)

#predicting with kNN model
knn_pred = knn_model.predict(x_val_scaled)

#evaluate the model
print(classification_report(y_val, knn_pred))
ConfusionMatrixDisplay.from_predictions(y_val, knn_pred)


#installing needed packages
!pip install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

#converting training and validation labels to 0 and 1 for the model
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.fit_transform(y_val)

#initialize XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

#fitting the model
xgb_model.fit(x_train,y_train_enc)

#predicting
xgb_pred_enc = xgb_model.predict(x_val)

#converting predicted labels back to introvert and extrovert
xgb_pred = le.inverse_transform(xgb_pred_enc)

#evaluate
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_val, xgb_pred)) #96.545%


#importing the test set
test =  pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

#viewing test set
test.head()


#summary of test set
test.info()


#factoring the test set to use for models
test['Stage_fear'] = test['Stage_fear'].map({'Yes':1, 'No':0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes':1, 'No':0})


from sklearn.impute import SimpleImputer
#filling in NA's with most frequent to cover both numeric and catagorical variables 
imputer = SimpleImputer(strategy='most_frequent') 
test_imputed = imputer.fit_transform(test) 
test_imputed = pd.DataFrame(test_imputed, columns=test.columns) 
test_imputed.info()


# Predicting test with random forest
test_ftpred = ft_model.predict(test_imputed)

#creating a csv for submission
rf_submission1 = pd.DataFrame({
    'id': test.iloc[:, 0],
    'Personality': test_ftpred
})

rf_submission1.to_csv('comp_submission1.csv', index=False) #97.17%


# Predicting test with Boosting
test_gbpred_le = xgb_model.predict(test)

#converting predicted labels back to introvert and extrovert
test_gbpred = le.inverse_transform(test_gbpred_le)

#creating a csv for submission
gb_submission1 = pd.DataFrame({
    'id': test.iloc[:, 0],
    'Personality': test_gbpred
})

gb_submission1.to_csv('comp_submission9.csv', index=False) 

