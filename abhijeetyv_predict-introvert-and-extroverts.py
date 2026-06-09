# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')



df.head()


df.describe()


for col in df.columns:
    print(col)


print(df.info)


print(df.isnull().sum())


import seaborn as sns

print("\nPersonality class distribution:")
print(df['Personality'].value_counts())
sns.countplot(x='Personality', data=df)
plt.title("Distribution of Personality Types")
plt.show()





numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

df[numeric_cols].hist(figsize=(12, 8), bins=15)
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()




corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()




cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    sns.countplot(x=col, hue='Personality', data=df)
    plt.title(f"{col} vs Personality")
    plt.show()




df['Personality_encoded'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Personality', y=col, data=df)
    plt.title(f"{col} vs Personality")
    plt.show()




#voilin plots for this data

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='Personality', y=col, data=df)
    plt.title(f"{col} vs Personality (Violin Plot)")
    plt.show()



cat_cols = ['Stage_fear', 'Drained_after_socializing']

for col in cat_cols:
    plt.figure(figsize=(5, 4))
    sns.countplot(x=col, hue='Personality', data=df)
    plt.title(f"{col} vs Personality")
    plt.show()

    # Normalized % stacked bar chart
    cross_tab = pd.crosstab(df[col], df['Personality'], normalize='index') * 100
    cross_tab.plot(kind='bar', stacked=True)
    plt.ylabel('Percentage')
    plt.title(f"{col} vs Personality (Normalized)")
    plt.legend(title='Personality')
    plt.show()


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

# Prepare data
X = df[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
        'Friends_circle_size', 'Post_frequency']].copy()
y = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Handle missing values
X = X.fillna(X.median())

# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k='all')
fit = selector.fit(X, y)

# Show scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': fit.scores_
}).sort_values(by='Score', ascending=False)

print("\nTop Features by ANOVA F-test:")
print(feature_scores)



from sklearn.ensemble import RandomForestClassifier

# Include categorical features as well (encode Yes/No)
X = df[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
        'Friends_circle_size', 'Post_frequency', 
        'Stage_fear', 'Drained_after_socializing']].copy()

# Encode categorical features
X['Stage_fear'] = X['Stage_fear'].map({'No': 0, 'Yes': 1})
X['Drained_after_socializing'] = X['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
X = X.fillna(X.median())

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get importances
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances from Random Forest:")
print(importances)

# Plot
import matplotlib.pyplot as plt
sns.barplot(data=importances, x='Importance', y='Feature')
plt.title("Feature Importance (Random Forest)")
plt.show()



corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()




from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report

df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

X = df.drop(columns=['id', 'Personality'])
y = df['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                    'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))  # to avoid dummy trap
])

# Combine preprocessors
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# feature selection + classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#xgboost

from xgboost import XGBClassifier

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#grid search

from sklearn.model_selection import GridSearchCV

#  hyperparameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1', 
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)
print("\nBest Cross-Validated Score:")
print(grid_search.best_score_)


y_pred = grid_search.best_estimator_.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validated Score:")
print(grid_search.best_score_)


best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)




y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))



import pandas as pd
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


test_ids = test_df['id']
test_df = test_df.drop(columns=['id'])

test_preds = best_model.predict(test_df)

personality_mapping = {0: "Introvert", 1: "Extrovert"}
test_personality = [personality_mapping[pred] for pred in test_preds]



submission = pd.DataFrame({
    'id': test_ids,
    'Personality': test_personality
})

submission.to_csv('/kaggle/working/sample_submission.csv', index=False)

print(submission.head())





