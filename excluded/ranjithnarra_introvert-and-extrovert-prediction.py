import numpy as np
import pandas as pd


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
output=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


output.Personality.value_counts()


train.shape,test.shape


train.columns


test['Personality']=[0]*len(test)
test.head()


df=pd.concat([train,test])




df


df.Time_spent_Alone.value_counts()


df = df.drop_duplicates()


df.isnull().sum() ,df['Personality'].value_counts(),df.shape


df1=df.drop(columns=['Personality'])
df['has_missing'] = df1.isnull().any(axis=1)


# Total columns in the DataFrame
total_cols = df.shape[1]
missing_per_row = df.isnull().sum(axis=1)
percent_missing_per_row = (missing_per_row / total_cols) * 100
percent_missing_per_row.value_counts()


total_rows = len(df)
percent_missing_per_column = (df.isnull().sum() / total_rows) * 100
print(percent_missing_per_column)


test.isnull().sum(),train.isnull().sum()


total_rows = len(df)
percent_missing_per_column = (df.isnull().sum() / total_rows) * 100
print(percent_missing_per_column)


df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1})
df['Personality'] = df['Personality'].map({'Extrovert': 0, 'Introvert': 1})


df.corr()


df.corr(method='spearman')



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
features=['Drained_after_socializing']
# Fill numerical nulls
num_imputer = IterativeImputer(max_iter=10, random_state=0)
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill categorical nulls
cat_imputer = SimpleImputer(strategy='most_frequent')
df[features] = cat_imputer.fit_transform(df[features])

df



df


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. Split the data into "known" and "unknown" Stage_fear -------------
df_known  = df[df["Stage_fear"].notna()].copy()
df_unknown = df[df["Stage_fear"].isna()].copy()
df_known



X_train = df_known.drop(columns=["id","Personality","has_missing"])  # no leakage
print(X_train)
y_train = df_known.Stage_fear
print(y_train)
# Identify feature types
num_cols  = X_train.select_dtypes(include=["float64","int64"]).columns.tolist()
cat_cols  = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",  
    random_state=42
)

clf = Pipeline([
    ("prep", preprocess),
    ("rf",  model)])
clf.fit(X_train, y_train)
print(classification_report(y_train, clf.predict(X_train)))
X_unknown = df_unknown.drop(columns=["id","Personality"])
preds = clf.predict(X_unknown)
df_unknown["Stage_fear"] = pd.Series(preds)



nan_idx = df['Stage_fear'].isna()
df.loc[nan_idx, 'Stage_fear'] = preds


df.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt
df_numeric = df.select_dtypes(include=['float64', 'int64'])
corr = df_numeric.corr()
plt.figure(figsize=(12, 8))  # You can adjust the figure size
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()



df.info()


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=df, y='Time_spent_Alone')  # or any numeric column
plt.title("Boxplot of Stage Fear")
plt.show()


Q1 = train['Time_spent_Alone'].quantile(0.25)
Q3 = train['Time_spent_Alone'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Time_spent_Alone'] = df['Time_spent_Alone'].clip(lower_bound, upper_bound)



df



train,test=df[:len(train)],df[len(train):]


train.corr()


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, mean_absolute_error
import xgboost as xgb
from catboost import CatBoostClassifier

# Prepare data
train, test = df[:len(train)], df[len(train):]

y_train = train['Personality']
X_train = train.drop(columns=['Personality', 'id'])
X_test = test.drop(columns=['Personality', 'id'])
y_true = output['Personality'].map({'Extrovert': 0, 'Introvert': 1}).values

# Initialize individual models
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
cat_model = CatBoostClassifier(verbose=0, random_state=42)
rf_model = RandomForestClassifier(random_state=42)

# Create ensemble with hard voting (majority vote)
voting_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ],
    voting='hard'  # Use 'soft' for probability averaging (works better when models are well-calibrated)
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_model, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validated Accuracy Scores (Voting):", cv_scores)
print("Mean CV Accuracy (Voting):", np.mean(cv_scores))

# Train on full training data
voting_model.fit(X_train, y_train)

# Predict on test set
final_preds = voting_model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_true, final_preds))
print("MSE:", mean_squared_error(y_true, final_preds))
print("Accuracy:", accuracy_score(y_true, final_preds))
print("Confusion Matrix:\n", confusion_matrix(y_true, final_preds))

# Map predictions back to labels
prediction_labels = pd.Series(final_preds).map({0: 'Extrovert', 1: 'Introvert'})
print("\nSample Predictions:")
print(prediction_labels.head(10))



y=y_train
#training accuracy
y_preds = voting_model.predict(X_train)
print("MAE:", mean_absolute_error(y, y_preds))
print("MSE:", mean_squared_error(y, y_preds))
print("Accuracy:",accuracy_score(y, y_preds))
print("CM",confusion_matrix(y, y_preds))


output=pd.DataFrame({'id':test.id ,'Personality':prediction_labels})
output.to_csv('my_submission.csv',index=False)
ou=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
output,ou

