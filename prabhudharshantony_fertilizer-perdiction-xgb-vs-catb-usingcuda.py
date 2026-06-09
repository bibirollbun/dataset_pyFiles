# ==== Basic Libs ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==== ML Libs ====
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==== Ignore Warnings ====
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('train.csv')
df_train


df_train.isnull().sum() # Check for any null values


# ==== Assigning Target and Feature Values ====
le = LabelEncoder()
x = pd.DataFrame(df_train.drop(columns=['Fertilizer Name']))

# ==== Creating Columns for categorical data (Apparently, it's better for modeling. Let's find out)====
x_ = pd.get_dummies(x)

df_train['Soil Type'] = le.fit_transform(df_train['Soil Type'])
df_train['Crop Type'] = le.fit_transform(df_train['Crop Type'])
y = pd.DataFrame(le.fit_transform(df_train['Fertilizer Name']),columns=["Fertilizer Name"])



y


# ==== This is what it should look like, every feature is converted into it's own column and Marked as a Bool (True/False or 1/0)
x_


df_train_one_hot = pd.concat([x_, y['Fertilizer Name']], axis=1, ignore_index=False)
df_train_one_hot


# ==== Checking Correlation ====
plt.figure(figsize=(16,8))
sns.heatmap(df_train_one_hot.corr(), annot=True, fmt='0.2f')
plt.title('Correlation Matrix')
plt.show()


#Split for both orginal dataset and one hot
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2) #Orginal 

xtrain,xtest,ytrain,ytest = train_test_split(x_,y, random_state=42, test_size= 0.2) #One Hot


#==== Fuction to train both data and return accracy, Because train.csv doesn't cotain 3 values for target I cannot evaluate using MAP@3,hence i used this. ====
    # If someone's reading this and knows a better way to check model performance, please drop a comment. I would highly appreciate that.
    
def train_xg_cat(xtrain,ytrain,xtest,ytest):
    '''
    This fuction trains the model with XGBoost and 
    CatBoost. Later checking the performance metrics
    '''
    XG = XGBClassifier(
        tree_method = 'gpu_hist',
        perdictor = 'gpu_predictor',
        max_depth = 4,
        learning_rate = 0.1,
        n_estimators = 100,
        use_label_encoder = False,
        objective = 'reg:gamma'
    )

    Cat = CatBoostClassifier(
    iterations=1500,
    depth=7,
    learning_rate=0.1,
    task_type='GPU',
    devices='0',
    verbose=100
    )
    
    XG.fit(xtrain,ytrain)
    xg_pred = XG.predict(xtest)
    
    Cat.fit(xtrain,ytrain)
    cat_pred = Cat.predict(xtest)
    
    xg_accuracy = accuracy_score(y_true= ytest, y_pred= xg_pred)
    cat_accuracy = accuracy_score(y_true= ytest, y_pred= cat_pred)
    
    print('XG Boost accuracy: ', xg_accuracy)
    print('Cat Boost accuracy: ', cat_accuracy)


# Traning and checking performance of XGB and CatB on the One Hot Dataset
train_xg_cat(xtrain,ytrain,xtest,ytest)


# Traning and checking performance of XGB and CatB on the Orginal Dataset
train_xg_cat(x_train,y_train,x_test,y_test)


# Here CatBoost with the One Hot Dataset clearly performs better. We will use that to train the model 
Cat = CatBoostClassifier(
iterations=1500,
depth=7,
learning_rate=0.1,
task_type='GPU',
devices='0',
verbose=100
)
Cat.fit(x_,y)


#==== Reading Data for testing ====
df_test = pd.read_csv('test.csv')
df_test


x_final = pd.get_dummies(df_test)


# ==== Predic Proba returns the probability of each Fertilizer====
probs = Cat.predict_proba(x_final)

# ==== Saving only the top three predictions in a array
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# ==== Transforming the target's back to it's original Labels ====
le.fit(df_train['Fertilizer Name']) # not necessary, it's just to make sure the LebalEncode doesn't use other columns to decode

decoded_preds = [le.inverse_transform(row) for row in top3_preds]


# ==== Basic Data Transformation ====
submission_preds = [' '.join(row) for row in decoded_preds]
submission_preds[:5]


# ==== Creating the submission DataFrame ====
final_csv = pd.DataFrame({
    'id': df_test['id'],
    'Fertilizer Name': submission_preds
})

final_csv.to_csv('submission.csv',index= False)

