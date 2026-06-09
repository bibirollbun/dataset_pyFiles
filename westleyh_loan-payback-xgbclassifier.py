import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings # Import to ignore all the warnings
warnings.filterwarnings('ignore')

train_full = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
X_test_full = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')

print("All data files successfully uploaded!")

#After examining column cardinality and values, the below columns are set for
#  one-hot encoding and ordinal encoding
oh_cols = ['loan_purpose', 'gender', 'employment_status', 'marital_status']
ord_cols = ['education_level', 'grade_subgrade']


train_full.head()


# Collecting all the numerical columns
num_cols = [col for col in train_full if train_full[col].dtype in ['float64', 'int64']]

#Getting info for the heatmap
heatmap_data = train_full[num_cols].corr()


plt.title("Correlation of Cols to Target Var Before Encoding")
sns.heatmap(heatmap_data, annot=True)


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

#Setting up Ordinal Encoding
od_encoder = OrdinalEncoder(categories='auto') #Setting categories to auto as more than one col will be Ordinal Encoded

# Ordinal Encoding for both the train and test datasets
for col in ord_cols:
    train_full[col] = od_encoder.fit_transform(train_full[[col]])
    X_test_full[col] = od_encoder.fit_transform(X_test_full[[col]])

#Setting up OneHot Encoding
oh_encoder = OneHotEncoder(categories='auto', sparse=False)

# Creating dfs to hold the one-hot encoded data for train and test sets
oh_train_full = pd.DataFrame()
oh_X_test_full = pd.DataFrame()

# One hot encoding for the training dataset
for col in oh_cols:
    oh_temp = pd.DataFrame(oh_encoder.fit_transform(train_full[[col]])) #OH Encoding a column
    oh_temp.columns = oh_encoder.get_feature_names_out(oh_encoder.feature_names_in_) #Getting and placing column names on OH data
    oh_train_full = pd.concat([oh_train_full, oh_temp], axis=1) #Joining current OH encoded column onto the final dataset

# One Hot encoding for the testing dataset
for col in oh_cols:
    oh_ttemp = pd.DataFrame(oh_encoder.fit_transform(X_test_full[[col]])) #OH Encoding a column
    oh_ttemp.columns = oh_encoder.get_feature_names_out(oh_encoder.feature_names_in_) #Getting and placing column names on OH data
    oh_X_test_full = pd.concat([oh_X_test_full, oh_ttemp], axis=1) #Joining current OH encoded column onto the final dataset
    

#Replacing the indexs
oh_train_full.index = train_full.index
oh_X_test_full.index = X_test_full.index

#Dropping the categorical columns that were just one-hot encoded
train_full.drop(oh_cols, axis=1, inplace=True)
X_test_full.drop(oh_cols, axis=1, inplace=True)

#Joining one-hot data to previous dataset
train_full_final = pd.concat([train_full, oh_train_full], axis=1)
X_test_full_final = pd.concat([X_test_full, oh_X_test_full], axis=1)





# Separating out the target
y = train_full_final['loan_paid_back']
X = train_full_final.drop('loan_paid_back', axis=1)


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

X['annual_income'] = (X['annual_income'] - X['annual_income'].mean(axis=0) ) / X['annual_income'].std(axis=0)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca = pd.DataFrame({'PCA1': X_pca[:,0], 'PCA2': X_pca[:,1]})
X = pd.concat([X, X_pca], axis=1)


#Splitting the train data into train and valid data
X_train_final, X_valid_final, y_train_final, y_valid_final = train_test_split(X, y, test_size=.2, random_state=0)


#Standardizing the annual income for the test datasets
X_test_full_final['annual_income'] = ( X_test_full_final['annual_income'] - X_test_full_final['annual_income'].mean(axis=0) ) / X_test_full_final['annual_income'].std(axis=0)


test_pca = pca.fit_transform(X_test_full_final)
test_pca_df = pd.DataFrame({'PCA1': test_pca[:,0], 'PCA2': test_pca[:,1]})
test_pca_df.set_index(X_test_full_final.index, inplace=True)
#Concatting PCA into main df; had to reset index due to pca have sequential index and test having sporadic index
X_test_full_final = pd.concat([X_test_full_final, test_pca_df], axis=1)


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Function to test different XGBoost parameters 
def test_models(X_train, y_train, X_valid_final, y_valid_final):
    # The best score received 
    best_score =  0.9222085991358778   

    estimators = [400, 700, 1000, 1500] #List of estimators to test
    stopping = [5, 10, 20, 50, 100, 150] #List of stopping rounds to test
    learn_rate = [.1, .08, .05, .03, .01] # List of learning rates to test

    # Multiple for loops to test each variation of the above parameter lists
    for est in estimators:
        for rate in learn_rate:
            #Instantiating the model according to the current parameters
            model = XGBClassifier(random_state=0, n_estimators=est, learning_rate=rate, eval_metric='auc')
            for stop in stopping:
                model.fit(X_train, y_train, early_stopping_rounds=stop,
                         eval_set=[(X_valid_final, y_valid_final)], verbose=False)
                preds = model.predict_proba(X_valid_final)[:,1]
                # If a new best score has been achieved, print the information and adjust best score
                if (roc_auc_score(y_valid_final, preds) > best_score):
                    print('New Best Score!')
                    print('Score: ', roc_auc_score(y_valid_final, preds), ' Estimators: ', est, ' Learning Rate: ', rate, ' Stopping Rounds: ', stop)
                    print()
                    best_score = roc_auc_score(y_valid_final, preds)
    print('All testing has been completed!')
    


import warnings # Import to ignore all the warnings
warnings.filterwarnings('ignore')

# Calling the testing method
# ***Takes a while to run***
#test_models(X_train_final, y_train_final, X_valid_final, y_valid_final)



#Instantiating the model
model = XGBClassifier(random_state=0, n_estimators=1500, learning_rate=0.05, eval_metric='auc')

#Fitting the model
model.fit(X_train_final, y_train_final,
         early_stopping_rounds=100,
         eval_set=[(X_valid_final, y_valid_final)],
          verbose=False)

preds = model.predict_proba(X_valid_final)[:, 1]
print("ROC AUC Score: ", roc_auc_score(y_valid_final, preds))
#Best ROC AUC Score:  0.9222085991358778  



#Attempting to implement Stratified K Fold CV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

#X['annual_income'] = ( X['annual_income'] - X['annual_income'].mean(axis=0) ) / X['annual_income'].std(axis=0)

cv = StratifiedKFold(random_state=0, n_splits=2, shuffle=True)
model = XGBClassifier(random_state=0, n_estimators=1500, learning_rate=.05, eval_metric='auc')
best_fold_score = 0
best_model = XGBClassifier()
score_list = []
for i, (train_index, valid_index) in enumerate(cv.split(X, y)):
    
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    print("***FOLD", i+1, "*****")
    model.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)],
             early_stopping_rounds = 100,
             verbose=False)
    preds = model.predict_proba(X_valid)[:,1]
    fold_score = roc_auc_score(y_valid, preds)
    score_list.append(fold_score)
    print("Score:", fold_score)
    print()
    if (fold_score > best_fold_score):
        best_fold_score = fold_score
        best_model = model

print('BEST RESULTS')
print('Score:', best_fold_score)
print()

print('CV Stats')
print('Max:', np.max(score_list))
print('Mean:', np.mean(score_list))
print('Min:', np.min(score_list))
print('STD:', np.std(score_list))


final_preds = best_model.predict_proba(X_test_full_final)[:,1]

submission = pd.DataFrame({'id': X_test_full_final.index, 'loan_paid_back': final_preds})
submission.to_csv('submission.csv', index=False)
#submission.shape


# Generating the competition test prediction values
preds_final = model.predict_proba(X_test_full_final)[:, 1]

#Creating the competition dataframe with prediction id
prediction_results = pd.DataFrame({
    'id':  X_test_full_final.index,
    'loan_paid_back': preds_final
})
prediction_results.shape
#Transforming above df to a csv for competition submittal  
#prediction_results.to_csv('submission.csv', index=False)




