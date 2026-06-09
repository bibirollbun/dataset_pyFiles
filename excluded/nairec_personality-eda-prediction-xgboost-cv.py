import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
data.head()


data.info()


print(data['Stage_fear'].unique())
print(data['Drained_after_socializing'].unique())
print(data['Personality'].unique())


# count of NaN's per column
for column in data.columns:
    na_count = data[column].isna().sum()
    print(f'{column}: {na_count} - {(na_count / 21424)*100:.2f}%')


sns.histplot(data=data, x='Time_spent_Alone', hue='Personality', discrete=True, palette='deep')
plt.show()
sns.histplot(data=data, x='Social_event_attendance', hue='Personality', discrete=True, palette='deep')
plt.show()
sns.histplot(data=data, x='Going_outside', hue='Personality', discrete=True, palette='deep')
plt.show()
sns.histplot(data=data, x='Post_frequency', hue='Personality', discrete=True, palette='deep')
plt.show()
sns.histplot(data=data, x='Friends_circle_size', hue='Personality', discrete=True, palette='deep')
plt.show()


sns.violinplot(data=data, y='Time_spent_Alone', x='Personality', discrete=True, palette='dark')
plt.show()
sns.violinplot(data=data, y='Social_event_attendance', x='Personality', discrete=True, palette='dark')
plt.show()
sns.violinplot(data=data, y='Going_outside', x='Personality', discrete=True, palette='dark')
plt.show()
sns.violinplot(data=data, y='Post_frequency', x='Personality', discrete=True, palette='dark')
plt.show()
sns.violinplot(data=data, y='Friends_circle_size', x='Personality', discrete=True, palette='dark')
plt.show()


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


training_data = data
y = data.Personality
training_data = training_data.drop(['Personality'], axis=1)
y = y.replace('Extrovert', 0)
y = y.replace('Introvert', 1)
X_train, X_valid, y_train, y_valid = train_test_split(training_data, y, train_size=0.8, test_size=0.2)
 
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
X_valid


numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ]
)

     
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enconder', OrdinalEncoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


parameters = {'n_estimators':[150, 200, 250, 300], 'learning_rate': [0.5, 0.05, 0.025, 0.005]}
model = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.01,
        random_state=0,
    )

pipeline = Pipeline(
        steps=[
            ('preprocessing', preprocessor),
            ('model', model)
        ]
    )



scores = cross_val_score(pipeline, training_data, y, cv=10, scoring='f1', error_score='raise')
print("CV accuracies:", scores, "Mean:", scores.mean())


test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test = test.drop(columns=["id"])

pipeline.fit(training_data, y)
predictions = pipeline.predict(test)

submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission["Personality"] = predictions
submission["Personality"] = submission["Personality"].replace([0,1], ["Extrovert","Introvert"])
submission.to_csv("submission.csv", index=False)
submission.value_counts()

