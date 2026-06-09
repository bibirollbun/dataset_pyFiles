import numpy as np
import pandas as pd
import os
import matplotlib.pyplot  as plt
import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print(train_data.shape)
print(test_data.shape)
print(train_data.dtypes)


print("TRAIN")
print(train_data.isna().sum())
print("TEST")
print(test_data.isna().sum())


train_data.head()



introverts = train_data[train_data.Personality == 'Introvert']
extroverts = train_data[train_data.Personality == 'Extrovert']

fig, axs = plt.subplots(2, 2, figsize=(12, 5))

colors = ['#1f77b4', '#ff7f0e']  # niebieski, pomarańczowy
labels = ['Introvert', 'Extrovert']

# 1. Time spent alone
axs[0][0].hist(
    [introverts.Time_spent_Alone, extroverts.Time_spent_Alone],
    bins=10,
    stacked=True,
    label=labels,
    color=colors,
    edgecolor='black'
)
axs[0][0].set_xlabel('Time spent alone')
axs[0][0].set_ylabel('Frequency')
axs[0][0].set_title('Time Spent Alone by Personality')
axs[0][0].legend()

# 2. Social event attendance
axs[0][1].hist(
    [introverts.Social_event_attendance, extroverts.Social_event_attendance],
    bins=10,
    stacked=True,
    label=labels,
    color=colors,
    edgecolor='black'
)
axs[0][1].set_xlabel('Social event attendance')
axs[0][1].set_ylabel('Frequency')
axs[0][1].set_title('Social Event Attendance by Personality')
axs[0][1].legend()

# 3. Time going outside
axs[1][0].hist(
    [introverts.Going_outside, extroverts.Going_outside],
    bins=10,
    stacked=True,
    label=labels,
    color=colors,
    edgecolor='black'
)
axs[1][0].set_xlabel('Time Going Outside')
axs[1][0].set_ylabel('Frequency')
axs[1][0].set_title('Time Going Outside by Personality')
axs[1][0].legend()

# 4. Friends circle size
axs[1][1].hist(
    [introverts.Friends_circle_size, extroverts.Friends_circle_size],
    bins=10,
    stacked=True,
    label=labels,
    color=colors,
    edgecolor='black'
)
axs[1][1].set_xlabel('Friends Circle Size')
axs[1][1].set_ylabel('Frequency')
axs[1][1].set_title('Friends Circle Size by Personality')
axs[1][1].legend()
plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Select target variable
y = train_data.Personality
train_data.drop(["Personality"], axis=1, inplace=True)

# Split to taining and validation
X_train_full, X_valid_full, y_train, y_valid_raw = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=0)

# Get categorical features names
categorical_features = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == 'object']
print("Categorical features: ", categorical_features)

# Get low cardinality features
low_cardinality_features = [cname for cname in categorical_features if X_train_full[cname].nunique() < 10]
print("Low cardinality features: ", low_cardinality_features)

# Get numeric columns
numeric_features = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int', 'float']]

# Filter redundant columns from train, test and valid
selected_features = low_cardinality_features + numeric_features

X_train = X_train_full[selected_features].copy()
X_valid = X_valid_full[selected_features].copy()
X_test = test_data[selected_features].copy()

# Encode categorical target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_valid = label_encoder.transform(y_valid_raw)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])



from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def get_score(n_estimators, cv, lr, X, y):
    """Return the average MAE over CV folds of XGBClassifier.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=0))
    ])
    
    scores = cross_val_score(pipeline, X, y,
                                 cv=cv,
                                 scoring='accuracy')
    return scores.mean()
    
def plot_results():
    pass
    
results = {n_estimators: get_score(n_estimators, 5, 0.01, X_train, y_train) for n_estimators in np.arange(50, 600, 50)}



plt.plot(list(results.keys()), list(results.values()))
plt.show()


pipeline_XGBClassifier = Pipeline(steps=[
    ('prepocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=350, learning_rate=0.01, random_state=0))
])

pipeline_XGBClassifier.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

predictions = pipeline_XGBClassifier.predict(X_valid)

accuracy = accuracy_score(y_valid, predictions)

print("Accuracy: ", accuracy)


final_prediction = pipeline_XGBClassifier.predict(X_test)


final_prediction = label_encoder.inverse_transform(final_prediction)


submission = pd.DataFrame({
    "id": X_test["id"],
    "pred_class": final_prediction
})


submission.to_csv("submission.csv", index=False)

