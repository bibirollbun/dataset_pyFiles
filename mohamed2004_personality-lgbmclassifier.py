import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV


from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
data.head()


data.info()


data = data.set_index('id')
data.head()


data.describe()


data['Stage_fear'].unique(), data['Drained_after_socializing'].unique(), data['Personality'].unique()



data['Stage_fear'] = data['Stage_fear'].replace({'Yes':1,'No':0})
data['Drained_after_socializing'] = data['Drained_after_socializing'].replace({'Yes':1,'No':0})
data['Personality'] = data['Personality'].replace({'Introvert':1,'Extrovert':0})



data.hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.tight_layout() 
plt.show()


plt.Figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


X = data.drop('Personality', axis=1, inplace=False)
y = data['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=44, shuffle =True)


# Define the pipeline
steps = [
    ('imputer', IterativeImputer(estimator=LGBMClassifier(),
                                 max_iter=1, random_state=0)),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=4)),
    ('model', LGBMClassifier(learning_rate=0.05, max_depth=3, 
                             n_estimators=90, subsample=0.8, verbose=-1))
]

PipelineLGBMClassifierModel = Pipeline(steps)



PipelineLGBMClassifierModel.fit(X_train, y_train)


#predict
y_pred = PipelineLGBMClassifierModel.predict(X_test)
print('y = ',y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

#calculate sensitivity, specificity, and accuracy

#accuracy
total_samples = sum(sum(cm))
accuracy = (cm[0,0] + cm[1,1])/total_samples #accuracy
print('Accuracy : ',accuracy)

#sensitivity
sensitivity = cm[0,0]/(cm[0,0] + cm[0,1])   
print('sensitivity',sensitivity)

#specificity
specificity = cm[1,1] / (cm[1,0]+ cm[1,1])
print( 'specificity :', specificity)


PipelineLGBMClassifierModel.fit(pd.concat([X_train,X_test]), pd.concat([y_train,y_test]))


#predict
y_pred = PipelineLGBMClassifierModel.predict(X_test)
print('y = ',y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

#calculate sensitivity, specificity, and accuracy

#accuracy
total_samples = sum(sum(cm))
accuracy = (cm[0,0] + cm[1,1])/total_samples #accuracy
print('Accuracy : ',accuracy)

#sensitivity
sensitivity = cm[0,0]/(cm[0,0] + cm[0,1])   
print('sensitivity',sensitivity)

#specificity
specificity = cm[1,1] / (cm[1,0]+ cm[1,1])
print( 'specificity :', specificity)


data_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



data_test = data_test.set_index('id')


data_test['Stage_fear'] = data_test['Stage_fear'].replace({'Yes':1,'No':0})
data_test['Drained_after_socializing'] = data_test['Drained_after_socializing'].replace({'Yes':1,'No':0})


y_submit = PipelineLGBMClassifierModel.predict(data_test)
y_submit


submit = pd.DataFrame(y_submit, columns=['Personality'],index=data_test.index)
submit


submit = submit.replace({1:'Introvert',0:'Extrovert'})
submit


submit.to_csv('submission.csv', index = True)

