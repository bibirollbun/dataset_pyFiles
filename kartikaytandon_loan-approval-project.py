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




import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score, roc_curve


import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv", index_col=0)

test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv", index_col=0)


train.head()


print(f'Number of rows: {train.shape[0]} \nNumber of cols: {train.shape[1]}')



print('\n')



negative = train['loan_status'][train['loan_status'] == 0].count()

positive = train['loan_status'][train['loan_status'] == 1].count()



print(f'Negative: {negative}\nPositve: {positive} \n%Positive: {positive/(positive+negative):.2f}')


train.info()


train['person_income'] = train['person_income'].astype(float)

train['loan_amnt'] = train['loan_amnt'].astype(float)


# Remove person age and person employment lenght > 100



train_subset = train[(train['person_age'] < 100)]



train_subset = train_subset[train_subset['person_emp_length'] < 100]


train_subset.describe().T.round(2)


train_subset.isna().sum()


train_subset[train_subset.duplicated()]


train_viz = train_subset.copy()


# create an age group

train_viz['age_group'] = ['<=30' if val <= 30 else '31 - 40'

                          if val <= 40 else '41 - 50'

                          if val <= 50 else '51 - 60'

                          if val <=60 else '> 60'

                          for val in train_viz['person_age']]


# mapping grade 

grade_num = {'A':1, 'B':2, 'C':3, 'D':4,'E':5,'F':6, 'G':7}



train_viz['grade_num'] = train_viz['loan_grade'].map(grade_num)


train_viz['person_home_ownership'] = train_viz['person_home_ownership'].str.title()

train_viz['loan_intent'] = train_viz['loan_intent'].str.title()


train_viz['cb_person_default_on_file'] = train_viz['cb_person_default_on_file'].apply(lambda cap: 'No' if cap == 'N' else 'Yes')


train_viz['loan_status'] = train_viz['loan_status'].apply(lambda x: 'No' if x == 0 else 'Yes')


categorical_var = list(train_viz.select_dtypes('object').columns)

continous_var = list(train_viz.select_dtypes('float64').columns)

discrete_var = list(train_viz.select_dtypes('int').columns)


plt.pie(train_viz['loan_status'].value_counts(),

               autopct='%.1f%%')

plt.legend(train_viz['loan_status'].unique());


for col in categorical_var:

    if col != 'loan_status':

        plt.figure(figsize=(15,5,))

        sns.histplot(data=train_viz[[col,'loan_status']].value_counts().reset_index().sort_values('count', ascending=False),

                        x=col,

                        weights='count',

                        hue='loan_status',

                        multiple='stack',

                      #  palette=palette,

                        edgecolor='white',

                        shrink=0.8

                    );


train_subset


for col in categorical_var:

    if col not in ['age_group',' person_age', 'loan_status']:

        plt.figure(figsize=(15,7))

        sns.barplot(data = train_subset[[col, 'loan_status']].groupby(col).mean().reset_index().sort_values('loan_status', ascending=False),

                     x=col,

                     y = 'loan_status'

                   )

        plt.show()


palette = ['green', 'red']



for num_var in continous_var+discrete_var:    



    fig, ax = plt.subplots(1, 2, figsize=(15, 4))   

    

    # Histograms    

    sns.histplot(data=train_viz, 

                      x=num_var,

                      bins = 30,

                      hue = 'loan_status',

                      palette=palette,

                      kde=True,

                      alpha=0.3,

                      ax=ax[0]                          

                ) 

    # Mean vertical line

    ax[0].axvline(np.mean(train_viz[num_var]), color="red")   

    ax[0].ticklabel_format(style='plain', axis='both')

    # Boxplots

    sns.boxplot(data=train_viz, 

                     y=num_var,

                     hue = 'loan_status',

                     palette=palette,

                     #showfliers=False,

                     ax=ax[1]

                   );


for col in continous_var:



    plt.figure(figsize=(15,7))



    sns.barplot(data=train_subset[[col, 'loan_grade']].groupby('loan_grade').mean().reset_index().sort_values(col, ascending=False),

                x='loan_grade',

                y=col

               )

    plt.title('Loan grade by average loan interest rates');


sns.barplot(data=train_subset[['loan_grade', 'loan_percent_income']].groupby('loan_grade').mean().reset_index(),

            x='loan_grade',

            y='loan_percent_income'

           )

plt.title('Loan grade by average loan percentage income');


sns.scatterplot(data=train_viz,

                x = 'grade_num',

                y='loan_int_rate',

                hue='loan_status'

                );


sns.pairplot(train_viz)

plt.title('Pair Plot of Selected Features')

plt.show()


def ext_features(df):

    df['cb_person_cred_hist_length_ratio'] = df['cb_person_cred_hist_length']/df['person_age']

    df['financial_burden'] = df['loan_amnt'] * df['loan_int_rate']    

    df['loan_percent_income'] = df['loan_amnt']/df['person_income']

    return df


def enc_features(df):

    col_enc = ['person_home_ownership', 'loan_intent','cb_person_default_on_file']

    grade_num = {'A':1, 'B':2, 'C':3, 'D':4,'E':5,'F':6, 'G':7}



    

    for col in col_enc:

        df= pd.concat([df, pd.get_dummies(df[col], dtype=int, prefix=col,drop_first=True)], axis=1)

        df.drop(columns=col, inplace=True)

        

    df['loan_grade'] = df['loan_grade'].map(grade_num)



    return df


train_subset = ext_features(train_subset)

test = ext_features(test)


train_enc = enc_features(train_subset)

test_enc = enc_features(test)


plt.figure(figsize=(20, 15))

sns.heatmap(pd.DataFrame(

                train_enc, 

                columns=train_enc.columns).corr(), 

                annot=True, cmap='coolwarm')

plt.title('Correlation Heatmap')

plt.show()




X = train_enc.drop(columns='loan_status')



y = train_enc['loan_status']







# Oversample minority class using SMOTE

smote = SMOTE()

X_resampled, y_resampled = smote.fit_resample(X, y)



X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)


rf = RandomForestClassifier(random_state=42)


rf_best_params = {'max_depth': 4,

                 'max_features': 5,

                 'max_samples': 0.07,

                 'min_samples_leaf': 0.02,

                 'min_samples_split': 0.001,

                 'n_estimators': 400}


rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)



print(f'Accuracy: {accuracy_score(y_pred, y_test)}')



print(f'Precision: {precision_score(y_pred, y_test)}')



print(f'Recall: {recall_score(y_pred, y_test)}')


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot();


importances = rf.feature_importances_

indices = np.argsort(importances)

features=train_enc.columns



plt.title('Feature Importances Random Forest')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()


# probability results

proba_ = rf.predict_proba(X_test)[:, 1]



# eval metrics

precision, recall, thresholds = precision_recall_curve(y_test, proba_)

fp, tp, thresholds_roc = roc_curve(y_test, proba_)



auc_score = np.round(roc_auc_score(y_test, proba_), 4)    



close_default = np.argmin(np.abs(thresholds - 0.5))

close_zero = np.argmin(np.abs(thresholds_roc))



# Plot ROC Curve

plt.figure(figsize=(7, 5))



plt.plot(fp, tp, label="ROC curve")

plt.plot(fp[close_zero], 

         tp[close_zero], 'o', 

         c='r', markersize=10, 

         label='threshold 0', 

         fillstyle="none", mew=2)

plt.title(f"ROC performance: AUC Score {auc_score}")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive (Recall)")

plt.legend(loc='best')   



plt.show()   


xgb = xgb.XGBClassifier()


xgb.fit(X_train, y_train)


y_pred = xgb.predict(X_test)


print(f'Accuracy: {accuracy_score(y_pred, y_test)}')



print(f'Precision: {precision_score(y_pred, y_test)}')



print(f'Recall: {recall_score(y_pred, y_test)}')


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot();


importances = xgb.feature_importances_

indices = np.argsort(importances)

features=train_enc.columns



plt.title('Feature Importances XGBoost')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()


# probability results

proba_ = xgb.predict_proba(X_test)[:, 1]



# eval metrics

precision, recall, thresholds = precision_recall_curve(y_test, proba_)

fp, tp, thresholds_roc = roc_curve(y_test, proba_)



auc_score = np.round(roc_auc_score(y_test, proba_), 4)    



close_default = np.argmin(np.abs(thresholds - 0.5))

close_zero = np.argmin(np.abs(thresholds_roc))



# Plot ROC Curve

plt.figure(figsize=(7, 5))



plt.plot(fp, tp, label="ROC curve")

plt.plot(fp[close_zero], 

         tp[close_zero], 'o', 

         c='r', markersize=10, 

         label='threshold 0', 

         fillstyle="none", mew=2)

plt.title(f"ROC performance: AUC Score {auc_score}")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive (Recall)")

plt.legend(loc='best')   



plt.show()   


predictions = xgb.predict_proba(test_enc)


submission_df = pd.DataFrame({

    'id': test_enc.index,

    'loan_status': predictions[:,1]

})


submission_df.head()


submission_df.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")

