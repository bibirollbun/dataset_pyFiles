import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train=pd.read_csv("/kaggle/input/oilgas-field-prediction/train_oil.csv")


t=pd.read_csv("/kaggle/input/oilgas-field-prediction/train_oil.csv")
y_train=t["Onshore/Offshore"]


train.head()


test=pd.read_csv("/kaggle/input/oilgas-field-prediction/oil_test.csv")
test.head()


train.shape


test.shape


train.columns


test.columns


train.info()


test.isna().sum()


import seaborn as sns


sns.histplot(train['Latitude'], bins=10, kde=True)  # `kde=True` adds a Kernel Density Estimate (like `distplot`)



train.describe()


mean_value = train["Latitude"].mean()
train["Latitude"] = train["Latitude"].fillna(mean_value)



mean_value = test["Latitude"].mean()
test["Latitude"] = test["Latitude"].fillna(value=mean_value)



sns.histplot(train['Longitude'], bins=10, kde=True)
plt.show()


train["Longitude"].mean()


train["Longitude"] = train["Longitude"].fillna(value=-30)



test["Longitude"]=test["Longitude"].fillna(value=-30)


plt.figure(figsize=(8,5))

train['Country'].value_counts().plot(kind='pie', autopct='%.2f')
plt.show()


#we can see most occuring countrys are usa,canada,uk
#so we will fill this by USA
train["Country"] = train["Country"].fillna(value="USA")




test["Country"]= test["Country"].fillna(value="USA")


train["Region"].mode()


#so most occurin in this north america we will it by this only
train["Region"] = train["Region"].fillna(value="NORTH AMERICA")



test["Region"].mode()
test["Region"] = test["Region"].fillna(value="NORTH AMERICA")


train["Basin name"].mode()


train["Basin name"] = train["Basin name"].fillna(value="WESTERN CANADA")



test["Basin name"] = test["Basin name"].fillna(value="WESTERN CANADA")



test.isna().sum()


train.isna().sum()


plt.figure(figsize=(28, 15))
sns.barplot(x='Country', y='Longitude', data=train)
plt.xticks(rotation=45)
plt.show()



train.columns


train['Onshore/Offshore'].value_counts().plot(kind='pie', autopct='%.2f')


y_train.value_counts().plot(kind='pie', autopct='%.2f')


#some how oyr data is clean now of output
#now lets change the catagorical data into numerical data
train.head()


def drop_col(df):
    df=df.drop(["Field name","Reservoir unit","Country","Region","Basin name","Operator company","Tectonic regime"],axis=1,inplace=True)
    return df


test.head()


drop_col(train)


drop_col(test)


#=train["Onshore/Offshore"]
y_train.head()


#now drop output too
train.drop("Onshore/Offshore",axis=1,inplace=True)


train.columns


test.columns


# Replace the problematic lines with the following:
y_train = t["Onshore/Offshore"]  # Make sure y_train is assigned the correct column
y_train.replace(['OFFSHORE', 'ONSHORE', 'ONSHORE-OFFSHORE'], [0, 1, 2], inplace=True)
y_train = y_train.astype(int)  # Convert to integer type


y_train


train.head()


train['Hydrocarbon type'].value_counts().plot(kind='pie', autopct='%.2f')


train['Structural setting'].value_counts().plot(kind='pie', autopct='%.2f')


train.head()


train['Reservoir status'].value_counts().plot(kind='pie', autopct='%.2f')


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


train.head()


cols = ['Hydrocarbon type', 'Reservoir status', 'Structural setting', 'Reservoir period','Lithology'] # # Encode labels of multiple columns at once #
train[cols] = train[cols].apply(LabelEncoder().fit_transform) # # Print head # df.head()


train.head()


cols = ['Hydrocarbon type', 'Reservoir status', 'Structural setting', 'Reservoir period','Lithology'] # # Encode labels of multiple columns at once #
test[cols] = test[cols].apply(LabelEncoder().fit_transform) # # Print head # df.head()


test.head()



df = train.copy()
df.head()


df_test= test.copy()
df_test.head()


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

cat_col = ['Hydrocarbon type', 'Reservoir status', 'Structural setting', 'Reservoir period','Lithology']
num_col = list(df.drop([*cat_col], axis = 1).columns)
col_transf = ColumnTransformer([
                                ('num_scaling1', StandardScaler(), num_col),
])



 df=col_transf.fit_transform(df)



 df_test=col_transf.fit_transform(df_test)


df_test = pd.DataFrame(df_test, columns = num_col)
df_test.head()


df = pd.DataFrame(df, columns = num_col)
df.head()


x1=train.copy()


x1_test=test.copy()
x2_test=df_test.copy()


x2=df.copy()


x1['Latitude']=x2['Latitude']
x1['Longitude']=x2['Longitude']
x1['Depth']=x2['Depth']
x1['Thickness (gross average ft)']=x2['Thickness (gross average ft)']
x1['Thickness (net pay average ft)']=x2['Thickness (net pay average ft)']
x1['Porosity']=x2['Porosity']
x1['Permeability']=x2['Permeability']


x1_test['Latitude']=x2_test['Latitude']
x1_test['Longitude']=x2_test['Longitude']
x1_test['Depth']=x2_test['Depth']
x1_test['Thickness (gross average ft)']=x2_test['Thickness (gross average ft)']
x1_test['Thickness (net pay average ft)']=x2_test['Thickness (net pay average ft)']
x1_test['Porosity']=x2_test['Porosity']
x1_test['Permeability']=x2_test['Permeability']


x1_test


x1


from sklearn.model_selection import train_test_split

xtrain, xval, ytrain, yval = train_test_split(x1,y_train, train_size=0.7)


xtrain.shape, xval.shape, ytrain.shape, yval.shape


ytrain


#try another model
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(xtrain, ytrain)


y_pred1 = sgd_clf.predict(xval)


score = y_pred1 == yval               # correct prediction = True, incorrect prediction = False
accuracy = score.sum() / y_pred1.size    # score.sum() = count the total num of True (correct prediction)
print(accuracy)
#the rediction is no that much good so we have to hypertune over model


from sklearn import metrics
conf_mat = metrics.confusion_matrix(yval, y_pred1)
conf_mat


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


dt.fit(xtrain,ytrain)


y_pred2=dt.predict(xval)


score = y_pred2 == yval               # correct prediction = True, incorrect prediction = False
accuracy = score.sum() / y_pred2.size    # score.sum() = count the total num of True (correct prediction)
print(accuracy)
#the rediction is no that much good so we have to hypertune over model
#so this is the best model we gain so far


conf_mat = metrics.confusion_matrix(yval, y_pred2)
conf_mat


#now predict in out test cases
y_ans=dt.predict(x1_test)


y_ans


# prompt: random forest

from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(xtrain, ytrain)

# Make predictions on the validation set
y_pred_rf = rf_clf.predict(xval)

# Evaluate the model
score_rf = y_pred_rf == yval
accuracy_rf = score_rf.sum() / y_pred_rf.size
print(f"Random Forest Accuracy: {accuracy_rf}")

# Confusion matrix for Random Forest
conf_mat_rf = metrics.confusion_matrix(yval, y_pred_rf)
print(f"Random Forest Confusion Matrix:\n{conf_mat_rf}")

# Predict on the test set
y_ans_rf = rf_clf.predict(x1_test)
print(f"Random Forest Predictions on Test Set:\n{y_ans_rf}")



# prompt: give me a heat map explain

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'conf_mat_rf' is your confusion matrix from the RandomForestClassifier
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=['OFFSHORE', 'ONSHORE', 'ONSHORE-OFFSHORE'],
            yticklabels=['OFFSHORE', 'ONSHORE', 'ONSHORE-OFFSHORE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap (Random Forest)')
plt.show()



# prompt: Support Vector Machines

import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Initialize the Support Vector Classifier
svm_clf = SVC(random_state=42)

# Train the SVM classifier
svm_clf.fit(xtrain, ytrain)

# Make predictions on the validation set
y_pred_svm = svm_clf.predict(xval)

# Evaluate the model
score_svm = y_pred_svm == yval
accuracy_svm = score_svm.sum() / y_pred_svm.size
print(f"SVM Accuracy: {accuracy_svm}")

# Confusion matrix for SVM
conf_mat_svm = metrics.confusion_matrix(yval, y_pred_svm)
print(f"SVM Confusion Matrix:\n{conf_mat_svm}")

# Predict on the test set
y_ans_svm = svm_clf.predict(x1_test)
print(f"SVM Predictions on Test Set:\n{y_ans_svm}")

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_svm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['OFFSHORE', 'ONSHORE', 'ONSHORE-OFFSHORE'],
            yticklabels=['OFFSHORE', 'ONSHORE', 'ONSHORE-OFFSHORE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap (SVM)')
plt.show()



# prompt: Neural Networks

# Assuming 'y_ans_rf' contains your Random Forest predictions and 'y_ans_svm' contains your SVM predictions
# Create a Pandas DataFrame for the predictions
import pandas as pd
results_df = pd.DataFrame({'RandomForest': y_ans_rf, 'SVM': y_ans_svm})
print(results_df)

# Example of saving to CSV
results_df.to_csv('model_predictions.csv', index=False)



# prompt: use randomforest final output

y_ans_rf



df1=pd.DataFrame({"Onshore/Offshore":y_ans_rf});df1.index.name="index"
df1


df1["Onshore/Offshore"]=df1["Onshore/Offshore"].replace({0:"OFFSHORE",1:"ONSHORE",2:"ONSHORE/OFFSHORE"});df1


df1.to_csv("submission.csv")

