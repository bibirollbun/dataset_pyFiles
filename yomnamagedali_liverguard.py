import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from warnings import filterwarnings
filterwarnings('ignore')



train_data = pd.read_csv('/kaggle/input/liver-guard-multi-class-prediction-for-cirrhosis/train.csv')
test_data = pd.read_csv('/kaggle/input/liver-guard-multi-class-prediction-for-cirrhosis/test.csv')


train_data.head(5)


train_data.info()


train_data.describe(include=['number'])


train_data.isnull().sum()


# Define numerical columns
to_standarize = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 
                  'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']
numerical_imputer = SimpleImputer(strategy='mean')
train_data[to_standarize] = numerical_imputer.fit_transform(train_data[to_standarize])
test_data[to_standarize] = numerical_imputer.transform(test_data[to_standarize])


# Scale numerical features (after imputation!)
scaler = StandardScaler()
train_data[to_standarize] = scaler.fit_transform(train_data[to_standarize])
test_data[to_standarize] = scaler.transform(test_data[to_standarize])


train_data.isnull().sum()


train_data.describe(include=['object'])


# Handle missing values in CATEGORICAL features (use mode)
fill_missing = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
for label in fill_missing:
    mode = train_data[label].mode()[0]
    train_data[label].fillna(mode, inplace=True)
    test_data[label].fillna(mode, inplace=True)


#Encoding the categorical variables {Sex, Ascites, Hepatomegaly, Spiders, Edema}
to_encode = ['Drug' # D-penicillamine: 0, Placebo:1, NaN:2
             , 'Sex' # F:0, M:1
             , 'Ascites' # N:0, Y:1, NaN:2
             , 'Hepatomegaly' # N:0, Y:1, NaN:2
             , 'Spiders' # N:0, Y:1, NaN:2
             , 'Edema' # N:0, Y:1
             ]
label_encoder = LabelEncoder()
for label in to_encode:
    train_data[label] = label_encoder.fit_transform(train_data[label])
    test_data[label] = label_encoder.transform(test_data[label])


 # Encode target classes C:0, CL:1, D:2
train_data['Status'] = label_encoder.fit_transform(train_data['Status'])


train_data.head(5)


train_data.isnull().sum()


#correlation between features
corr_matrix = train_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Features')
plt.show()




#scatter plot between features and target
sns.scatterplot(x='Stage', y='Status', data=train_data)


sns.scatterplot(x='Prothrombin', y='Status', data=train_data)



y_train = train_data['Status']
X_train = train_data.drop('Status', axis=1)


sns.pairplot(X_train)
plt.title('Pairplot')
plt.show()


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_train)
residuals = y_train - predictions

plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predictions")
plt.ylabel("Residuals")


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model = RandomForestClassifier()


#Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=5, 
                            scoring='accuracy',
                            n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


grid_search.best_estimator_.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

best_model2 = grid_search.best_estimator_


y_pred = best_model2.predict(X_test)


print("Accuracy:",accuracy_score(y_test, y_pred))
print("Classification report:\n",classification_report(y_test, y_pred))


y_test_pred = best_model2.predict(test_data)
probabilities = best_model2.predict_proba(test_data)


plt.figure(figsize=(12, 6))
sns.histplot(probabilities, kde=True, bins=30)
plt.title('Probability Distribution of Predictions')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()


#Submitting the predictions
class_probabilities = pd.DataFrame(probabilities, columns=['Status_C', 'Status_CL', 'Status_D'])
class_probabilities['id'] = test_data['id']
class_probabilities.to_csv('submission.csv', index=False)

