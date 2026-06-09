import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

train.drop(columns = 'id', inplace=True)
test.drop(columns = 'id', inplace = True)



train.head(2)


train.info()


test.info()


train = train.rename(columns = {'temparature' : 'temperature'})
test = test.rename(columns = {'temparature':'temperature'})

print(f"Dimension of train data is {train.shape}")
print(f"Dimension of test data is {test.shape}")


print(f"Missing data train \n\n{train.isnull().sum()}")
print(f"\n\nMissing data test \n\n{test.isnull().sum()}")


#filling missing value in wind direction
test['winddirection'].fillna(test['winddirection'].median(), inplace = True)


import scipy
from scipy.stats.stats import pearsonr

corr_df = pd.DataFrame()
p = []
feature_1 = []
feature_2 = []
correlation = []

for feat_1 in train.columns:
    if feat_1 not in ['rainfall']:
        feature_1.append(feat_1)
        corr, p_value = scipy.stats.pearsonr(train[feat_1], train['rainfall'])
        correlation.append(corr)
        p.append(p_value)

corr_df['feature_1'] = feature_1
corr_df['feature_2'] = 'rainfall'
corr_df['p_value']=p
corr_df['correlation']=correlation

corr_df.sort_values('correlation', ascending = False)


plt.figure(figsize=(8,6))
sns.set_theme(style = "ticks")

sns.scatterplot(data = corr_df, x = "feature_1", y = "correlation", s=80,  hue = "correlation")
plt.tick_params(axis='x', labelrotation=40)
plt.xlabel("Features")
plt.ylabel("Pearson Coefficient")
plt.title("Correlation: Rain with all features")


plt.figure(figsize = (8,6))
ax = sns.heatmap(train.corr(numeric_only = True), annot = True, fmt = ".1f", linewidth = 0.5, cmap="crest")


#checking the distribution of rainfall which does not have a balance
train['rainfall'].value_counts()


from sklearn import metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
mean_absolute_error, mean_squared_error)
from sklearn.model_selection import KFold


kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

def model_kfold(model, x , y , kf):

    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    mae_list, mse_list, rmse_list = [], [], []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:,1]

        #classification metrics
        acc_list.append(round(accuracy_score(y_test, y_pred) ,4))
        prec_list.append(round(precision_score(y_test, y_pred), 4))
        rec_list.append(round(recall_score(y_test, y_pred),4))
        f1_list.append(round(f1_score(y_test, y_pred),4))
        auc_list.append(round(roc_auc_score(y_test, y_pred),4))
        
        #regression metrics
        mae_list.append(round( mean_absolute_error(y_test, y_pred),4))
        mse_val = round(mean_squared_error(y_test, y_pred),4)
        mse_list.append(mse_val)
        rmse_list.append(mse_val)

    model_df = {"accuracy": acc_list,
                "precision": prec_list,
                "recall" : rec_list,
                "f1_score": f1_list,
                "roc_auc" : auc_list,
                "mae" : mae_list,
                "mse": mse_list,
                "rmse" : rmse_list}
    model_df = pd.DataFrame(model_df)

    return model_df



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

x = train.drop('rainfall', axis = 1)
y = train[['rainfall']] #double square brackets ensure y remains a dataframe, single would make it a series

#logistic regression
lr_model = LogisticRegression()
lr_results = model_kfold(lr_model, x, y, kf)

#knn
knn_model = KNeighborsClassifier(n_neighbors = 10)
knn_results = model_kfold(knn_model, x, y, kf)

#random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_results = model_kfold(rf_model, x, y, kf)


#print average result from the model results

def print_avg_scores(model_results, model):
    print(f"\nAverage of Metrics for {model} :")

    for metrics, values in model_results.items():
        print(f"{metrics} : {round(np.mean(values),2)}")


print_avg_scores(lr_results, lr_model)
print_avg_scores(knn_results, knn_model)
print_avg_scores(rf_results, rf_model)


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

final_model= LogisticRegression()
final_model.fit(x_train, y_train)

y_pred = final_model.predict(x_test)
y_prob = final_model.predict_proba(x_test)[:,1]


print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))



y_test_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv") #has 'id'

final_model.fit(x, y)
y_submission_pred = final_model.predict(test)
y_submission_prob = final_model.predict_proba(test)[:, 1]

submission_df = pd.DataFrame( {
    'id': y_test_submission['id'],
    'rainfall' : y_submission_pred
})

submission_df.to_csv("submission.csv", index=False)

