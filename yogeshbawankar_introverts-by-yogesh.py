# Importing necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

df.head(5)


print(f"Numbers of rows : {df.shape[0]}\nNumbers of columns : {df.shape[1]}")


print("Check the total numbers NULL values for each columns")
df.isnull().sum()


df.info()


df.isnull().sum()


print("Check is there any duplicates ")
df.duplicated().sum()


# Handling null values 

cols_to_fill = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance','Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']  # etc.

df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill').fillna(method='bfill')

df.isnull().sum()


numrical_cols = df.select_dtypes(include=['int64','float64'])
categorical_cols = df.select_dtypes(include=['object'])
print("Numrical Columns : ",numrical_cols.columns)
print("Categorical Columns : ",categorical_cols.columns)


numerical_cols = df.select_dtypes(include='number').columns.tolist()

for col in numerical_cols:
    if col != 'id':
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
        print(df[col].describe())
        print("="*50)


# Time_spent_Alone vs Personality
sns.boxplot(x='Personality', y='Time_spent_Alone', data=df)
plt.title("Time Spent Alone vs Personality")
plt.show()


# Social_event_attendance vs Personality
pd.crosstab(df['Social_event_attendance'], df['Personality'], normalize='index').plot(kind='bar', stacked=True)
plt.title("Social Event Attendance vs Personality")
plt.ylabel("Proportion")
plt.show()


# Going_outside vs Personality
sns.boxplot(x='Personality', y='Going_outside', data=df)
plt.title("Going Outside Frequency vs Personality")
plt.show()


# Friends_circle_size vs Personality
sns.boxplot(x='Personality', y='Friends_circle_size', data=df)
plt.title("Friends Circle Size vs Personality")
plt.show()


# Co-relation

# Only on relevant numeric features
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Convert binary text to 0/1
df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
df['Social_event_attendance'] = df['Social_event_attendance'].map({'No': 0, 'Yes': 1})

# Step 2: Selecting Relevant Columns
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                'Going_outside', 'Drained_after_socializing',
                'Friends_circle_size', 'Post_frequency']

target_col = 'Personality'
df = df.dropna(subset=[target_col])


# Train Test Split
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Identifying columns type 
numerical_cols = X.select_dtypes(include='number').columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()


numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Optional for tree-based models, but good style
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=4))
])


# Fit the train data
clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Train the pipeline and assign it to best_model
best_model = clf_pipeline.fit(X_train, y_train)


# Run the prediction
X_submission = test_df[feature_cols]
X_submission = X_submission.copy()


# Fill missing values in safe way 
X_submission = X_submission.fillna(method='ffill').fillna(method='bfill')


preds_binary = best_model.predict(X_submission)

# Convert back 0 and 1 readable labels 
preds_labels = pd.Series(preds_binary).map({0: "Introvert", 1: "Extrovert"})


submission_df = pd.DataFrame({
    "id": test_df["id"],
    "Personality": preds_labels
})

submission_df.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")




