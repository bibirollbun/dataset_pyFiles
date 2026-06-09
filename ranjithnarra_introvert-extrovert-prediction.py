import numpy as np
import pandas as pd
train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
output=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
train.shape,test.shape


train.columns


test['Personality']=[0]*len(test)
df=pd.concat([train,test])
df


df = df.drop_duplicates()


df.isnull().sum() ,df['Personality'].value_counts(),df.shape


13699/4825


df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1})
df['Personality'] = df['Personality'].map({'Extrovert': 0, 'Introvert': 1})


df.corr()


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
features=['Drained_after_socializing']

num_imputer = IterativeImputer(max_iter=10, random_state=0)
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[features] = cat_imputer.fit_transform(df[features])

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


X_train = df_known.drop(columns=["id","Personality"])  # no leakage
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


print(len(nan_idx),len(preds))


import seaborn as sns
import matplotlib.pyplot as plt
df_numeric = df.select_dtypes(include=['float64', 'int64'])
corr = df_numeric.corr()
plt.figure(figsize=(12, 8))  # You can adjust the figure size
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


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



train,test=df[:len(train)],df[len(train):]
y = train['Personality']
X = train.drop(columns=['Personality','id'])



from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

train,test=df[:len(train)],df[len(train):]
y = train['Personality']
X = train.drop(columns=['Personality','id'])

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_holdout = scaler.transform(X_holdout)

# 4. Choose model with class_weight
clf = LogisticRegression(
    class_weight='balanced',  # auto-adjust weights
    solver='liblinear',       # good for small to medium data
    random_state=42,
    max_iter=500
)

# 5. Setup cross-validation depending on imbalance ratio
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f'Imbalance ratio: {ratio:.2f}:1')

if ratio <= 4:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
else:
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

# 6. Cross-validate
scores = cross_val_score(
    clf, X_train, y_train,
    scoring='f1',  # f1 is appropriate for imbalanced binary
    cv=cv,
    n_jobs=-1,
    verbose=1
)
print('CV F1 scores:', scores)
print('Mean F1:', np.mean(scores))

# 7. Final training & holdout evaluation
clf.fit(X_train, y_train)
y_pred = clf.predict(X_holdout)

print('\nConfusion Matrix:\n', confusion_matrix(y_holdout, y_pred))
print('\nClassification Report:\n', classification_report(y_holdout, y_pred))



test1=test
Xt = test.drop(columns=['Personality','id'])
y_preds = clf.predict(Xt)
out=output['Personality'].map({'Extrovert': 0, 'Introvert': 1})
print('\nConfusion Matrix:\n', confusion_matrix(out.values, y_preds))
print('\nClassification Report:\n', classification_report(out.values, y_preds))




prediction = pd.Series(y_preds).map({0: 'Extrovert', 1: 'Introvert'})
output=pd.DataFrame({'id':test1.id ,'Personality':prediction})
output.to_csv('my_submission.csv',index=False)
ou=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')




