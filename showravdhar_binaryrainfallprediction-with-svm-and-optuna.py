import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.head()


trsum = train_df.isnull().sum()
print(f"Missing values in train: {trsum}")

tstsum = test_df.isnull().sum()
print(f"\nMissing values in test:{tstsum}")


# Replace NaN with specific values for each column
test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mean())
tstsum = test_df.isnull().sum()
print(f"\nMissing values in test:{tstsum}")


train_df.describe().T


data = train_df

colors = ['#446BAD','#A2A2A2']

l = list(data['rainfall'].value_counts())
circle = [l[0] / sum(l) * 100,l[1] / sum(l) * 100]

fig = plt.subplots(nrows = 1,ncols = 2,figsize = (20,5))
plt.subplot(1,2,1)
plt.pie(circle,labels = ['Rain Cases','Not_Rain Cases'],autopct = '%1.1f%%',startangle = 90,explode = (0.1,0),colors = colors,
       wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
plt.title('Rain - Not_Rain Case %');

plt.subplot(1, 2, 2)
ax = sns.countplot(x='rainfall', data=data, palette=colors, edgecolor='black')
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), 
            horizontalalignment='center', fontsize=11)
ax.set_xticklabels(['Not_Rain Cases', 'Rain Cases'])

plt.title('Number of Rain - Non-Not_Rain Cases')

plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


train_df1 = train_df.drop(columns = ['id','day'])

data = train_df1
col = list(data.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(data[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features :',*categorical_features)
print(f'Numerical Features :{numerical_features}')


features = train_df1.loc[:,numerical_features]
target = train_df1.loc[:,categorical_features]

best_features = SelectKBest(score_func = f_classif,k = 'all')
fit = best_features.fit(features,target)

featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['ANOVA Score']) 

plt.subplots(figsize = (5,5))
sns.heatmap(featureScores.sort_values(ascending = False,by = 'ANOVA Score'),annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',fmt = '.2f');
plt.title('Selection of Numerical Features');


train_df.drop(columns = ['mintemp','winddirection','day'],inplace = True)
train_df.describe().T


import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE


f1 = train_df.iloc[:,1:9].values
t1 = train_df.iloc[:,9].values
Counter(t1)


over = SMOTE()
f2 = train_df.iloc[:,1:9].values # features
t2 = train_df.iloc[:,9].values #target
f2, t2 = over.fit_resample(f2, t2)
Counter(t2)


fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(y=t1, ax=ax[0], )
ax[0].set_title("Class Distribution Before SMOTE")
ax[0].set_xlabel("Classes")
ax[0].set_ylabel("Frequency")

sns.countplot(y=t2, ax=ax[1])
ax[1].set_title("Class Distribution After SMOTE")
ax[1].set_xlabel("Classes")
ax[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


train_df.columns


X = train_df.drop(columns=['rainfall', 'id'])
y = train_df['rainfall']
X_test = test_df.drop(columns=['id','day','mintemp','winddirection'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


!pip install optuna


# Importing the required libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


import optuna
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Define the objective function for Optuna
def objective(trial):
    # Choose the algorithm to tune
    classifier_name = trial.suggest_categorical('classifier', ['SVM', 'RandomForest', 'GradientBoosting', 'DecisionTreeClassifier'])

    if classifier_name == 'SVM':
        # SVM hyperparameters
        c = trial.suggest_float('C', 0.1, 100, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        model = SVC(C=c, kernel=kernel, gamma=gamma, random_state=42, probability=True)

    elif classifier_name == 'RandomForest':
        # Random Forest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=42
        )

    elif classifier_name == 'GradientBoosting':
        # Gradient Boosting hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    elif classifier_name == 'DecisionTreeClassifier':
        # Decision Tree hyperparameters
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    # Perform cross-validation and return the mean accuracy
    score = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc').mean()
    return score


# Assuming X_train and y_train are defined (your training data)
# Example: X_train, y_train from your binary classification dataset

# Create and run the Optuna study
study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())  # Maximize accuracy
study.optimize(objective, n_trials=100)  # Run 100 trials

# Print the best result
print("Best trial:")
print("  Value (ROC-AUC):", study.best_trial.value)
print("  Params:", study.best_trial.params)


# Retrieve the best trial
best_trial = study.best_trial
print("Best trial parameters:", best_trial.params)
print("Best trial accuracy:", best_trial.value)


study.trials_dataframe()['params_classifier'].value_counts()


best_params = study.best_trial.params
if best_params['classifier'] == 'SVM':
    best_model = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        gamma=best_params['gamma'],
        random_state=42,
        probability=True
    )
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
print(f"ROC AUC Score: {roc_auc_score(y_val, y_pred_proba)}")
print(f"Classification Report:\n{classification_report(y_val, y_pred)}")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix")
plt.show()


test_predictions = best_model.predict_proba(X_test_scaled)[:, 1]
submission_df = pd.DataFrame({'id': test_df['id'], 'rainfall': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")




