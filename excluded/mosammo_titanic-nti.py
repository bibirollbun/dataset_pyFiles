import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split


warnings.filterwarnings("ignore")


def bar_plot(df, column, hue=None, size=(10, 6), title=None):
    if not title:
        title = f'Distribution of {column}'
    plt.figure(figsize=size)
    ax = sns.countplot(data=df, x=column, hue=hue)
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        ax.text(p.get_x() + p.get_width() / 2, height / 2, percentage, ha='center', va='center', fontsize=10, color='white')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def outlier_cut(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return np.where(df > upper_bound, upper_bound, np.where(df < lower_bound, lower_bound, df))


train = pd.read_csv('/kaggle/input/dsaa-6100-titanic-survival-using-decision-trees/train.csv')
test = pd.read_csv('/kaggle/input/dsaa-6100-titanic-survival-using-decision-trees/test.csv')
train.head()


feats = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train = train[feats + ['Survived']]
test = test[feats]


train.info()


test.isna().sum()


train.isna().sum()


test['Age'] = test['Age'].fillna(train['Age'].mean())
train['Age'] = train['Age'].fillna(train['Age'].mean())


test['Fare'] = test['Fare'].fillna(train['Fare'].mean())


train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])


train.isna().sum()


test.isna().sum()


train['Fare'] = outlier_cut(train['Fare'])


for column in ['Age', 'Fare']:
    plt.figure(figsize=(12, 4))  # Create a new figure for each feature

    # Subplot 1: KDE Plot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    sns.kdeplot(train[column], fill=True)
    plt.title(f'KDE Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

    # Subplot 2: Box Plot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    sns.boxplot(x=train[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot


labels = ['{0} - {1}'.format(i, i + 4) for i in range(1, 62, 5)]
labels[-1] = labels[-1][:2] + '+'
labels


train['Age'] = pd.cut(train['Age'], range(0, 67, 5), labels=labels)
test['Age'] = pd.cut(test['Age'], range(0, 67, 5), labels=labels)


for col in train.drop(columns=['Fare']).columns:
    bar_plot(train, column=col)


plt.figure(figsize=(10, 4))  # Create a new figure for each feature

# Subplot 1: KDE Plot
sns.kdeplot(train[train['Survived'] == 1]['Fare'], color='red', fill=True)
sns.kdeplot(train[train['Survived'] == 0]['Fare'], color='blue', fill=True)
plt.legend(['yes', 'no'], loc='upper right')
plt.title(f'KDE Plot of Fare')
plt.xlabel('Fare')
plt.ylabel('Density')

plt.show()  # Display the plot


for feat in train.drop(['Fare', 'Survived'], axis=1):
    bar_plot(train, column=feat, hue='Survived', size=(12,6))


for col in ['Sex', 'Age', 'Embarked']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


X = train.drop(columns=['Survived'])
y = train['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

rf = RandomForestClassifier()

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

rand_search.fit(X_train, y_train)


best_rf = rand_search.best_estimator_
max_depth = rand_search.best_params_['max_depth']
n_estimators = rand_search.best_params_['n_estimators']

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)


rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_importances.plot.bar()


sub = pd.read_csv('/kaggle/input/dsaa-6100-titanic-survival-using-decision-trees/gender_submission.csv')
sub['Survived'] = rf.predict(test)


sub.to_csv('sub.csv', index=False)




