import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train.head()


train.shape


train.isnull().sum()


train.describe()


train.duplicated().sum()


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test.head()


test.shape


test.isnull().sum()


test.describe()


train['y'].value_counts()


concat_df = pd.concat([train, test], axis = 0)
concat_df.shape


concat_df.head()


concat_df.info()


concat_df['poutcome'].value_counts()


concat_df['contact'].value_counts()


concat_df['loan'].value_counts()


concat_df['housing'].value_counts()


concat_df['default'].value_counts()


concat_df['education'].value_counts()


concat_df['marital'].value_counts()


concat_df['job'].value_counts()


concat_df['month'].value_counts()


train['y'].value_counts()


numerical_columns = concat_df.select_dtypes(exclude='object').drop(['id','y'], axis = 1)
numerical_columns.columns


fig, axes = plt.subplots(4, 2, figsize=(15,15))
axes = axes.flatten()

for i, col in enumerate(numerical_columns.columns):
    sns.boxplot(data = concat_df, y=col, ax=axes[i])
    axes[i].set_title(f'Bloxplot of {col}')

plt.tight_layout()
plt.show()


def replace_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3-q1
    lower = q1-1.5*iqr
    upper = q3+1.5*iqr
    median_value = df[column].median()
    df[column] = np.where((df[column]<lower)|(df[column]>upper), median_value, df[column])
    return df


for col in numerical_columns.columns:
    concat_df = replace_outliers(concat_df, col)


fig, axes = plt.subplots(4, 2, figsize=(15,15))
axes = axes.flatten()

for i, col in enumerate(numerical_columns.columns):
    sns.boxplot(data = concat_df, y=col, ax=axes[i])
    axes[i].set_title(f'Bloxplot of {col}')

plt.tight_layout()
plt.show()


concat_df['month'] = concat_df['month'].map({'aug':8, 'jun':6, 'may':5, 'feb':2, 'apr':4, 'nov':11, 'jul':7, 'jan':1, 'oct':10,
       'mar':3, 'sep':9, 'dec':12})


concat_df['job'] = concat_df['job'].map({'technician':1, 'blue-collar':2, 'student':3, 'admin.':4, 'management':5,
       'entrepreneur':6, 'self-employed':7, 'unknown':99, 'services':8, 'retired':9, 'housemaid':11, 'unemployed':12})


concat_df['education'] = concat_df['education'].map({'secondary':1, 'primary':2, 'tertiary':3, 'unknown':99})


concat_df['default'] = concat_df['default'].map({'no':0, 'yes':1})


concat_df['housing'] = concat_df['housing'].map({'no':0, 'yes':1})


concat_df['marital'] = concat_df['marital'].map({'married':1, 'single':2, 'divorced':3})


concat_df['loan'] = concat_df['loan'].map({'no':0, 'yes':1})


concat_df['contact'] = concat_df['contact'].map({'cellular':1, 'unknown':99, 'telephone':2})


concat_df['poutcome'] = concat_df['poutcome'].map({'unknown':99, 'other':1, 'failure':2, 'success':3})


concat_df.info()


concat_df = concat_df.drop('id', axis = 1)


concat_df['log_duration'] = np.log1p(concat_df['duration'])   # log transform
concat_df['duration_per_campaign'] = concat_df['duration'] / (concat_df['campaign'] + 1)
concat_df['campaign_per_previous'] = concat_df['campaign'] / (concat_df['previous'] + 1)


concat_df['has_any_loan'] = ((concat_df['housing'] == 'yes') | (concat_df['loan'] == 'yes')).astype(int)


concat_df['balance_to_age_ratio'] = concat_df['balance'] / (concat_df['age'] + 1)


newtrain = concat_df.iloc[0:750000, :]
newtest = concat_df.iloc[750000:, :].drop('y', axis = 1)


newtrain.shape


newtest.shape


from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

X = newtrain.drop(columns=['y'])
y = newtrain['y']

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

selector = SelectKBest(score_func=chi2, k=10)   # pick top 10
X_new = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]

print("Top 10 Selected Features:")
print(selected_features.tolist())



x = newtrain[['marital', 'housing', 'loan', 'contact', 'duration', 'campaign', 'poutcome', 'log_duration', 'duration_per_campaign', 'campaign_per_previous']]
y = newtrain['y']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


models = {'Logistic Regression': LogisticRegression(), 'Random Forest': RandomForestClassifier(),
         'Bagging': BaggingClassifier(), 'Extra Tree': ExtraTreesClassifier(), 'LightGBM': LGBMClassifier(),
         'Gradient Boosting': GradientBoostingClassifier(), 'Adaboost': AdaBoostClassifier(),
         'XGB': XGBClassifier(), 'KNN': KNeighborsClassifier()}


def evaluate_models(x_train,x_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        predictions = model.fit(x_train, y_train).predict(x_test)
        accuracy = roc_auc_score(y_test, predictions)
        results[name] = accuracy
    return results


results = evaluate_models(x_train,x_test, y_train, y_test, models)


best_model_name = max(results, key = results.get)
best_model = models[best_model_name]


print(f"best model is {best_model_name} with roc_auc_score {results[best_model_name]}")


y_pred = best_model.fit(x_train, y_train).predict(x_test)


print(roc_auc_score(y_test, y_pred))


feature_imp = pd.DataFrame(sorted(zip(best_model.feature_importances_, x.columns), reverse=True)[:20], columns=['Value','Feature'])
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title(f'{best_model_name} Features')
plt.tight_layout()
plt.show()


x_train = newtrain.drop('y', axis = 1)
y_train = newtrain['y']
x_test = newtest
y_pred = best_model.fit(x_train, y_train).predict(x_test)


solution = pd.DataFrame({'id':test['id'], 'y': y_pred})
solution.head()


solution.to_csv('Solution.csv', index = False)




