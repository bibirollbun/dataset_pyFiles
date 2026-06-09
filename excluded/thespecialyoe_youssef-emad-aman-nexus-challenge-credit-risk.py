#Importing the necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


#Loading the application dataset into a dataframe using pandas
train_df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")


#First look on the data
train_df.head()


column_description = pd.read_csv("/kaggle/input/home-credit-default-risk/HomeCredit_columns_description.csv")
column_description.head()


#Getting the descriptions for the columns in the application dataset only
application_col_description = column_description[column_description['Table'] == 'application_{train|test}.csv'][['Row','Description']]
#Lets see if we extracted the right data
application_col_description.count()


#Exploring what does each column represent
pd.set_option('display.max_rows', 122)
application_col_description.head(122)


pd.set_option('display.max_rows', 50)


#Creating a function that returns description of any column (for future reference)
def col_description (col) :
    col_upper = col.upper()
    result = application_col_description[application_col_description['Row'] == col_upper]['Description'].values
    return result[0]


#Inspecting the data shape and column data types
print(train_df.shape)
train_df.info()


#We now have to deal with columns that contains null values
missing = train_df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

#Lets visualize the number of null values in all the columns 
missing.hist(bins=len(missing))
plt.title('Number of missing rows per column')
plt.xlabel('Number of missing rows')
plt.show()




#now lets drop the columns that has more than 140k missing value
more_than_140k_missing = missing[missing > 140000].index
train_df.drop(columns = more_than_140k_missing, inplace=True)
#creating a list to append any dropped columns to it (so it can be used when preprocessing test data)
dropped_cols= list(more_than_140k_missing)


print(train_df.shape)
train_df.isna().sum().sort_values(ascending=False).head(20)


#Grouping these cols into a list
credit_bureau_cols = ['AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR']
#Reusing the col_description function we created earlier and calculating the mean for each column (i am a bit suspisous that they are representing the same thing but at diff frequencies)
for col in credit_bureau_cols :
    print(col_description(col) + ' column mean equals ' + str(round(train_df[col].mean(),2)))


#Removing the yearly requests from list of credit bureau columns
credit_bureau_cols.remove('AMT_REQ_CREDIT_BUREAU_YEAR') 
#droping the AMT_REQ_CREDIT_BUREAU qrt, mon, week, day, hour columns
train_df.drop(columns=credit_bureau_cols,inplace=True)
dropped_cols = dropped_cols + credit_bureau_cols


#filling AMT_REQ_CREDIT_BUREAU_YEAR null values with the column mean
train_df['AMT_REQ_CREDIT_BUREAU_YEAR'] = train_df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(train_df['AMT_REQ_CREDIT_BUREAU_YEAR'].mean())


print(train_df.shape)
missing = train_df.isna().sum() 
missing = missing[missing >0].sort_values(ascending=False)
print(missing)


#Now lets investigate the 12 remaining columns with null values and understand what does each of them represent
missing_cols = missing.index
for col in missing_cols :
    print(col + ': ' + col_description(col))


#Lets deal with the occupation_type column first
#Initial thinking is to use total income amount column as a reference to the occupation
print(train_df[['OCCUPATION_TYPE','AMT_INCOME_TOTAL']].groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].mean().sort_values(ascending=False))



#since we have alot of missing values in the occupation column it may be a good idea to represent each occupation by the earnings
#lets see the distribution of the total income amount column
train_df['AMT_INCOME_TOTAL'].describe()


#Defining the function that would be used to create the OCCUPATION_INCOME_LEVEL feature
def income_level(income) :
    if income < 112500 :
        return 'low'
    elif income < 202500:
        return 'medium'
    elif income < 1.5*202500:
        return 'high'
    else :
        return 'very high'



#Applying the function to the data
train_df['OCCUPATION_INCOME_LEVEL'] = train_df['AMT_INCOME_TOTAL'].apply(income_level)
#dropping the occupation type column
train_df.drop(columns='OCCUPATION_TYPE',inplace=True)
dropped_cols.append('OCCUPATION_TYPE')


train_df['OCCUPATION_INCOME_LEVEL'].value_counts(normalize=True).plot(kind='bar')
plt.title('Distribution of Applicants by Income Level')
plt.ylabel('Proportion')
plt.xlabel('Income Level')
plt.yticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], labels=['0%', '10%', '20%', '30%', '40%', '50%'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#Lets see the remaining columns with the highest missing values
missing = train_df.isna().sum() 
missing = missing[missing>0].sort_values(ascending=False)
missing.head(10)


print(missing.index[0],col_description(missing.index[0]))


train_df.groupby('OCCUPATION_INCOME_LEVEL')['EXT_SOURCE_3'].mean()


train_df['EXT_SOURCE_3'].fillna(train_df['EXT_SOURCE_3'].median(),inplace=True)


#Lets see the remaining columns with missing values
missing = train_df.isna().sum() 
missing = missing[missing>0].sort_values(ascending=False)
missing


def filling_null_values_w_median_mode(df):
    for col in df.columns:
        if df[col].isna().sum() > 0 and df[col].dtype != 'object':
            df[col].fillna(df[col].median(), inplace=True)
        elif df[col].isna().sum() > 0 and df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)


#calling function on our dataframe
filling_null_values_w_median_mode(train_df)


#Lets see how the target class is distributed
train_df['TARGET'].value_counts(normalize=True).plot(kind='bar')
plt.title('Distribution of the Target Variable in the dataset')
plt.xticks(rotation = 0)
plt.show()


#Viewing the columns datatypes
train_df.dtypes.value_counts()


#Lets store categorical columns in a list and numercial columns in another list
cat_cols =[]
num_cols =[]
for col in train_df.columns :
    if train_df[col].dtype == 'object':
        cat_cols.append(col)
    else :
        num_cols.append(col)
        


#Viewing number of unique values in each categorical column
print('Number of unique values in the categorical columns')
print('------------------------------------')
for col in cat_cols:
    print(col + ' : ' + str(train_df[col].nunique()))


train_df['ORGANIZATION_TYPE'].value_counts()


train_df.drop('ORGANIZATION_TYPE',axis=1,inplace=True)
dropped_cols.append('ORGANIZATION_TYPE')
cat_cols.remove('ORGANIZATION_TYPE')


#Viewing unique values of each categorical coulmn
for col in cat_cols:
    print(col + ' : ' + str(train_df[col].unique()))


train_df['CODE_GENDER'].value_counts()


train_df['CODE_GENDER'] = train_df['CODE_GENDER'].replace('XNA','F')
train_df['CODE_GENDER'].value_counts()


train_df['FLAG_OWN_CAR'] = train_df['FLAG_OWN_CAR'].replace(['Y','N'],[1,0])
train_df['FLAG_OWN_REALTY'] = train_df['FLAG_OWN_REALTY'].replace(['Y','N'],[1,0])


def weekday_vs_weekend(day):
    day=day.upper()
    if day in ['FRIDAY','SATURDAY'] :
        return 0
    else : 
        return 1
train_df['WEEKDAY_APPR_PROCESS_START'] = train_df['WEEKDAY_APPR_PROCESS_START'].apply(weekday_vs_weekend)    


#Running this loop once again to get categorical columns to be used for one hot encoding
cat_cols =[]
num_cols =[]
for col in train_df.columns :
    if train_df[col].dtype == 'object':
        cat_cols.append(col)
    else :
        num_cols.append(col)


#One hot encoding the categorical columns 
encoded_df = pd.get_dummies(train_df,columns=cat_cols,drop_first=True)
encoded_df.shape


#final look on the dataset and ensuring that all columns are numerical after the one hot encoding
encoded_df.dtypes.value_counts()


#Scaling the features so we avoid having one feature dominating the model
from sklearn.preprocessing import MinMaxScaler

X = encoded_df.drop('TARGET',axis = 1)
y = encoded_df['TARGET']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


#Splitting the dataset into training data and test data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size=0.2 , random_state=42)


#First lets use random forest classifier and see how it will perform
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=25)
rf_model.fit(X_train,y_train)


#After training the model and fitting the data to the model, we are going to make the predictions for the test set
rf_predictions = rf_model.predict(X_test)


#Testing the model performance
print(classification_report(y_test,rf_predictions))


#Training the Logistic regression model with specific parameters
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
model = LogisticRegression(max_iter=1000,solver='liblinear',penalty='l1',class_weight='balanced')
model.fit(X_train,y_train)


#Predictions for the test set
log_predictions = model.predict(X_test)


#Classification report for the Logistic Regression model
print(classification_report(y_test,log_predictions))


from sklearn.metrics import roc_auc_score

y_probs = model.predict_proba(X_test)[:, 1]  # get probability of class 1
auc = roc_auc_score(y_test, y_probs)

print("AUC Score:", round(auc, 4))


#Saving the model, as it took nearly an hour to train
import joblib
joblib.dump(model,'logistic_regression_model.pk1')


#loading the saved model, if needed
import joblib
model =joblib.load('logistic_regression_model.pk1')


#This function is created to make sure that the columns used to build the model are the same as the test set
def align_columns(train_df, test_df):

    train_cols = train_df.columns
    test_df_aligned = test_df.copy()

    # Add missing columns to test set
    for col in train_cols:
        if col not in test_df_aligned.columns:
            test_df_aligned[col] = 0

    # Drop any extra columns in test that are not in train
    test_df_aligned = test_df_aligned[train_cols]

    return test_df_aligned
    


#    Cleans and preprocesses the test DataFrame using the same steps applied to training data
#    Returns a final, scaled version ready for prediction.
def clean_df(df):
    df['AMT_REQ_CREDIT_BUREAU_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(df['AMT_REQ_CREDIT_BUREAU_YEAR'].mean())
    df.drop(columns = dropped_cols, inplace =True)
    df['OCCUPATION_INCOME_LEVEL'] = df['AMT_INCOME_TOTAL'].apply(income_level)
    filling_null_values_w_median_mode(df)
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA','F')
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace(['Y','N'],[1,0])
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace(['Y','N'],[1,0])
    df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].apply(weekday_vs_weekend)
    encoded_test = pd.get_dummies(df,columns=cat_cols,drop_first=True)
    aligned_test =  align_columns(encoded_df.drop(columns=['TARGET']),encoded_test)
    final = scaler.transform(aligned_test)
    return final


# Load and clean the test set
test_set = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
final = clean_df(test_set)
final.shape


# Predicting probabilities for the test set
final_test_pred = model.predict_proba(final)[:,1]


# Creating the submission file
submission = pd.DataFrame(test_set[['SK_ID_CURR']])
submission['TARGET'] = final_test_pred
submission.to_csv(r'C:\Users\EGYPT_LAPTOP\Desktop\Aman coding challenge\final_submission.csv',index=False)

