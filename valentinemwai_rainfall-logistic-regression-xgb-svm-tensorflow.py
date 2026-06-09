import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


train = pd.read_csv('../input/playground-series-s5e3/train.csv')
test = pd.read_csv('../input/playground-series-s5e3/test.csv')
sub = pd.read_csv('../input/playground-series-s5e3/sample_submission.csv')


print(train.head())
print(test.head())


#dropping the ID column
train=train.drop(['id'], axis=1)
test =test.drop(['id'], axis=1)


#Descriptive Statistics
print(train.describe())
print(test.describe())


print(train.info())
print(test.info())


print(train.isnull().sum())



test['winddirection'].fillna(test['winddirection'].median(), inplace=True)
print(test.isnull().sum())


#count plot of rainfall
sns.countplot(x='rainfall', data=train)
plt.show()


#Temperature distribution by rainfall
sns.histplot(train, x= 'temparature', hue= 'rainfall', kde=True)
plt.figure(figsize =(8,6))
plt.show()


train_df = pd.DataFrame(train)
scaler=MinMaxScaler()
train_scaled=scaler.fit_transform(train)
# Convert back to DataFrame
train_scaled_df = pd.DataFrame(train_scaled, columns=train.columns)

# Now you can use .head()
print(train_scaled_df.head())


#Correlation
corr_matrix = train_scaled_df.corr()
print(corr_matrix)
plt.figure(figsize=(12,7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',fmt=".2f")
plt.show()


X= train_scaled_df.drop(['rainfall'], axis=1)
y=train_scaled_df['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.8, random_state =40)


model=LogisticRegression(C=0.08858667904100823, solver='newton-cg')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


#Prediction using the Test dataset
test_df = pd.DataFrame(test)
scaler=MinMaxScaler()
test_scaled=scaler.fit_transform(test)
# Convert back to DataFrame
test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns)

# Now you can use .head()
print(test_scaled_df.head())
pred = model.predict_proba(test_scaled_df)[:,1]


#submission
sub['rainfall'] = pred
sub.to_csv('submission.csv', index=False)
sub.head()





param_grid = [
    {'penalty':['l1','l2','elasticnet','none'],
    'C' : np.logspace(-4,4,20),
    'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter'  : [100,1000,2500,5000]
}
]



clf_1 = LogisticRegression(C=0.0001, penalty='none', solver='newton-cg') 
clf_1.fit(X_train, y_train)
y_pred_1 =  clf_1.predict(X_test)   
acc = accuracy_score(y_test,y_pred_1 ) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")


clf_2 = LogisticRegression(C=0.0001, penalty='none', solver='newton-cg')
#clf_2 = LogisticRegression(solver='saga',penalty='l2',max_iter=10000, random_state=42,C=1.0) #83.20
clf_2.fit(X_train, y_train)
    
y_pred_2 =  clf_2.predict(X_test)   
acc = accuracy_score(y_test,y_pred_2 ) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")


test_preds_2 = clf_2.predict_proba(test_scaled_df)[:,1]


sub_2 = pd.read_csv('../input/playground-series-s5e3/sample_submission.csv')
#submission
sub_2['rainfall'] = test_preds_2
sub_2.to_csv('submissionLR1.csv', index=False)
sub_2.head()


import tensorflow as tf
print(tf.__version__)


tf.random.set_seed(42)


model_3 = tf.keras.Sequential([

  tf.keras.layers.Dense(100), # add 100 dense neurons

  tf.keras.layers.Dense(10), # add another layer with 10 neurons

  tf.keras.layers.Dense(1)

])

model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),

                optimizer=tf.keras.optimizers.Adam(), 

                metrics=['accuracy'])

model_3.fit(X, y, epochs=100, verbose=0)


model_3.evaluate(X,y)


model_4 = tf.keras.Sequential([

                               tf.keras.layers.Dense(4, activation = 'relu'), #we may right it "tf.keras.activations.relu" too

                               tf.keras.layers.Dense(4, activation = 'relu'),

                               tf.keras.layers.Dense(1, activation = 'sigmoid')

])

model_4.compile( loss= tf.keras.losses.binary_crossentropy,

                optimizer = tf.keras.optimizers.Adam(),

                metrics = ['accuracy'])

model_4.fit(X_train, y_train, epochs = 100, verbose = 0)


model_4.evaluate(X_train,y_train)


loss, accuracy = model_4.evaluate(X_test, y_test)
print(f' Model loss on the test set: {loss}')
print(f' Model accuracy on the test set: {100*accuracy}')


pred_4 = model_4.predict(test_scaled_df)
pred_4


sub4= pd.read_csv('../input/playground-series-s5e3/sample_submission.csv')
#submission
sub4['rainfall'] = pred_4
sub4.to_csv('submission4.csv', index=False)
sub4.head()


#Import svm model
from sklearn import svm

#Create a svm Classifier
svm_model = svm.SVC(random_state=42, C=100, gamma=0.001,kernel='rbf',probability=True) # Linear Kernel

#Train the model using the training sets
svm_model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = svm_model.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))



pred_svm= svm_model.predict_proba(test_scaled_df)[:,1]


sub_svm= pd.read_csv('../input/playground-series-s5e3/sample_submission.csv')
#submission
sub_svm['rainfall'] = pred_svm
sub_svm.to_csv('submission_svm.csv', index=False)
sub_svm.head()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define pipeline with SVC
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf'))
])

# Define parameter grid for SVC
param_grid = {
    'svc__C': [1, 5, 10, 50,1000], 
    'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]
}

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)  # Corrected



import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report



# Convert dataset into DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',  # Logarithmic loss for evaluation
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100,
    'random_state': 42
}

# Train the model
xgb_clf = xgb.XGBClassifier(**params)
xgb_clf.fit(X_train, y_train)



# Predict on test set
y_pred = xgb_clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))



import matplotlib.pyplot as plt
xgb.plot_importance(xgb_clf)
plt.show()



# Predict probabilities on new test dataset
pred_xgb = xgb_clf.predict_proba(test_scaled_df)[:,1] # X_test is the test dataset

# Print probabilities for the first 5 test instances
print(pred_xgb[:5])



sub_xgb= pd.read_csv('../input/playground-series-s5e3/sample_submission.csv')
#submission
sub_xgb['rainfall'] = pred_xgb
sub_xgb.to_csv('submission_xgb.csv', index=False)
sub_xgb.head()

