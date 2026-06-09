import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train.head()


train.shape


train.isnull().sum()


train.describe()


train.duplicated().sum()


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test.head()


test.shape


test.isnull().sum()


test.describe()


train['Personality'].value_counts()


concat_df = pd.concat([train, test], axis = 0)
concat_df.shape


concat_df.head()


concat_df.info()


concat_df['Stage_fear'] = concat_df['Stage_fear'].map({'Yes': 1, 'No': 0})
concat_df['Drained_after_socializing'] = concat_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})


for col in concat_df.drop(columns = ['id', 'Personality']).columns:
    concat_df[col] = concat_df[col].fillna(concat_df[col].mean())


concat_df.isnull().sum()


numerical_columns = concat_df.select_dtypes(exclude='object').drop('id', axis = 1)
numerical_columns.columns


fig, axes = plt.subplots(4, 2, figsize=(15,15))
axes = axes.flatten()

for i, col in enumerate(numerical_columns.columns):
    sns.boxplot(data = concat_df, y=col, ax=axes[i])
    axes[i].set_title(f'Bloxplot of {col}')

plt.tight_layout()
plt.show()


concat_df['fear_drained_interaction'] = concat_df['Stage_fear'] * concat_df['Drained_after_socializing']
concat_df['alone_to_friends_ratio'] = concat_df['Time_spent_Alone'] / (concat_df['Friends_circle_size'] + 1)
concat_df['outside_to_friends_ratio'] = concat_df['Going_outside'] / (concat_df['Friends_circle_size'] + 1)
concat_df['social_engagement_score'] = (concat_df['Social_event_attendance'] +concat_df['Going_outside'] +concat_df['Post_frequency'])
concat_df['introvert_score'] = (concat_df['Time_spent_Alone'] * 0.4 +concat_df['Drained_after_socializing'] * 0.3 -concat_df['Social_event_attendance'] * 0.3)
concat_df['extrovert_score'] = (concat_df['Social_event_attendance'] * 0.4 + concat_df['Going_outside'] * 0.3 - concat_df['Time_spent_Alone'] * 0.3 -
    concat_df['Drained_after_socializing'] * 0.3)


concat_df = concat_df.drop('id', axis = 1)


newtrain = concat_df.iloc[0:18524, :]
newtest = concat_df.iloc[18524:, :].drop('Personality', axis = 1)


newtrain.shape


newtest.shape


le = LabelEncoder()
newtrain['Personality'] = le.fit_transform(newtrain['Personality'])


x = newtrain.drop('Personality', axis = 1)
y = newtrain['Personality']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


models = {'Logistic Regression': LogisticRegression(), 'Random Forest': RandomForestClassifier(),
         'Bagging': BaggingClassifier(), 'Extra Tree': ExtraTreesClassifier(), 'LightGBM': LGBMClassifier(),
         'Gradient Boosting': GradientBoostingClassifier(), 'Adaboost': AdaBoostClassifier(),
         'XGB': XGBClassifier(), 'KNN': KNeighborsClassifier(), 'svm' : SVC()}


def evaluate_models(x_train,x_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        predictions = model.fit(x_train, y_train).predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
    return results


results = evaluate_models(x_train,x_test, y_train, y_test, models)


best_model_name = max(results, key = results.get)
best_model = models[best_model_name]


print(f"best model is {best_model_name} with accuracy {results[best_model_name]}")


y_pred = best_model.fit(x_train, y_train).predict(x_test)


print(accuracy_score(y_test, y_pred))


feature_imp = pd.DataFrame(sorted(zip(best_model.feature_importances_, x.columns), reverse=True)[:20], columns=['Value','Feature'])
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title(f'{best_model_name} Features')
plt.tight_layout()
plt.show()


x_train = newtrain.drop('Personality', axis = 1)
y_train = newtrain['Personality']
x_test = newtest
y_pred = best_model.fit(x_train, y_train).predict(x_test)


solution = pd.DataFrame({'id':test['id'], 'Personality': le.inverse_transform(y_pred)})
solution.head()


solution.to_csv('Solution WITHOUT SMOTE.csv', index = False)




