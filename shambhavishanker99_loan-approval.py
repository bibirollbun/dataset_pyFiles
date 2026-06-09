import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
from sklearn.svm import SVC


#loadind model into pd datframe
df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')


#view datadrame
df.head()


#checking the no of rows and columns
df.shape


#checking if there is any missing vlaues
df.isnull().sum()


#check if there are duplicate values
df.duplicated().sum()


#geeting an overview 
df.info()


#dropping id as it is not needed
df.drop(columns='id',axis = 1,inplace=True)


#checking updated df
df.head()


#extarcting feature and target column
X = df.drop(columns='loan_status',axis=1)
Y = df['loan_status']


#view X dataframe
X.head()


#view Y dataframe
Y.head()


#saving categorical columns in object_col and numerical in num_col

num_col = [ col for col in X.columns if (X[col].dtype != 'O')]

object_col = [col for col in X.columns if(X[col].dtype == 'O')]

print('num col: ', num_col , '\n object_col: ', object_col)


#histogram
def histogram(df,column):
    plt.figure(figsize=(5,3))
    sns.histplot(df[column],kde=True)

    plt.title(column)
    plt.axvline(df[column].mean(), color='red' , linestyle='--')
    plt.axvline(df[column].median(), color='blue' , linestyle=':')
    plt.show()


#invoking hostogram function for num_col

for col in num_col:
    histogram(df,col)


#function for boxplot
def box_plot(column):
    plt.figure(figsize=(5,3))
    sns.boxplot(df[column])
    plt.title(column)


#invoking boxplot for col in num_col
for col in num_col:
    box_plot(col)


#countplot for target class
#plotting distribution of target variable
plt.figure(figsize=(5,3))
Y.value_counts().plot(kind='bar')   
plt.title('Distribution of Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()


#setting the threshold value to 30% and confirming if target variable is balanced or not

#calculating the percentage of samples for 'Survived' column
percentage = (Y.value_counts() / len(df) )* 100

#check if data is imbalanced or not
if percentage[0] > 30:
    print('Target class is imbalanced')
else:
    print('Target class is balanced')


#correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']].corr(), annot = True, cmap='coolwarm')
plt.title("Correlation Matrix")


#printing count of unique value in obect_col

for col in object_col:
    print(df[col].value_counts())
    print('*'*50)


#label encoding
#initialize label encoder
label_encoder = {}

#apply label encoding and store the label encoder in dictionary
for col in object_col:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoder[col] = le
    
#saving the label encoder in a file       
joblib.dump(label_encoder, '/kaggle/working/loan_label_encoder.pkl')
    
print(label_encoder)


#view updated df
df.head()


#Again extracting X,Y after performing label encoding
X = df.drop(columns='loan_status', axis=1)
Y= df['loan_status']



#Handling Outliers using IQR and Capping
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    #calcluating upar bound and lower bound
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # Cast bounds to the same dtype as the column
    lower_bound = lower_bound.astype(df[col].dtype)
    upper_bound = upper_bound.astype(df[col].dtype)

    
    df.loc[df[col] > upper_bound, col] = upper_bound
    df.loc[df[col] < lower_bound, col] = lower_bound


#invoking cap_outliers for num_col

for col in num_col:
    cap_outliers(df, col)


#checking for outliers using box_plot

for col in num_col:
    box_plot(col)


#loading smote

smote = SMOTE()


#printing the shape of x & y

print(X.shape,Y.shape)

#performing SMOTE
X,Y = smote.fit_resample(X,Y)

#print shape after smote
print(X.shape,Y.shape)


#checking distribution of Y
Y.value_counts()


#split the data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, stratify = Y, random_state=3) 


#print the size of traina nd test 
print(X_train.shape, Y_train.shape)
print(X_test.shape,Y_test.shape)


#XGBoost
# Defining the parameters
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# GridSearchCV
xgb_model = RandomizedSearchCV(xgb.XGBClassifier(objective='binary:logistic', random_state=42), param_dist, cv=5, n_jobs=-1)

# Fitting the model in GridSearch
xgb_model.fit(X_train, Y_train)

# Printing the best hyperparameters
print(xgb_model.best_params_)

# Printing the best accuracy score
print(xgb_model.best_score_)



#SVM
# Defining the parameters
param_dist = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'degree': [2, 3],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.1],
    'shrinking': [True],
    'probability': [False],
    'tol': [1e-3, 1e-4],
    'class_weight': [None, 'balanced']
}

# GridSearchCV
svm = RandomizedSearchCV(SVC(), param_dist, n_iter=50, cv=5, n_jobs=-1, random_state=42)

# Fitting the model in GridSearch
svm.fit(X, Y)

# Printing the best hyperparameters
print(svm.best_params_)

# Printing the best accuracy score
print(svm.best_score_)


#cross validation
# Define the XGBoost model with the best hyperparameters
xgb_model = xgb.XGBClassifier(
    subsample=0.8,
    reg_lambda=1.5,
    reg_alpha=0,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.2,
    gamma=0.2,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42
)

# Perform cross-validation"
cv_score_xgb = cross_val_score(xgb_model, X, Y, cv=5)

# Getting mean of cv_score_xgb
cv_score_xgb_mean = cv_score_xgb.mean()
print(cv_score_xgb_mean)


#fitting the model on training set
xgb_model.fit(X_train, Y_train)


#save the model_data
joblib.dump(xgb_model, '/kaggle/working/xgb_boost_model.pkl')


#predicting the model on testing set
Y_pred = xgb_model.predict(X_test)

#calculating the accuracy of the model
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")

#consufion matrix
cm = confusion_matrix(Y_test, Y_pred)

#plotting the confusion matrix
plt.figure(figsize=(5,3))
sns.heatmap(cm, annot=True, cmap='Blues') 



#loading test data
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


#view df
test.head()


#extracting passengerid in different 

ID = test['id'].tolist()


#load model
loaded_model = joblib.load('/kaggle/working/xgb_boost_model.pkl')



test.drop(columns='id',axis=1,inplace=True)


#view updated test df
test.head()


#label encoding of test df

#loading label encoder
#loading the encoders
label_encoder = joblib.load('/kaggle/working/loan_label_encoder.pkl')

#label encoding the categorical columns in df_input
for col in object_col:
    le = label_encoder[col]
    test[col] = le.transform(test[col])

test.head()


#making prediction
prediction = loaded_model.predict(test)


print(prediction)


#saving
results = pd.DataFrame({
    'id': ID,
    'loan_status': prediction
})

# Check the result
results.head()


# Convert the DataFrame to a CSV file and drop the index
results.to_csv('/kaggle/working/loan_predictions.csv', index=False)

