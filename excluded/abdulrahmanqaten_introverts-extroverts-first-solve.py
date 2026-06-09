import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(train.info())
print('--------------------------------------------------------')
print(test.info())


# General styling
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Select numeric columns
numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                   'Friends_circle_size', 'Post_frequency']


train[numeric_columns].hist(bins=30, color='skyblue', figsize=(15, 10))
plt.suptitle('Distribution of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()


sns.countplot(x='Personality', data=train, palette='Set2')
plt.title('Distribution of Personality Types')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()


for col in numeric_columns:
    plt.figure()
    sns.boxplot(x='Personality', y=col, data=train, palette='Set3')
    plt.title(f'{col} vs Personality')
    plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(train[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()


for df_name, df in [('train', train), ('test', test)]:
    print(f"\nStarting to handle missing values. DataFrame: {df_name}")

    for column in df.columns:
        if df[column].isnull().any():

            # If the column is numerical
            if pd.api.types.is_numeric_dtype(df[column]):
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
                print(f"[{df_name}]The column '{column}' was compensated with the average: {mean_value}")

            # If the column is text (object or string)
            elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
                if not df[column].mode().empty:
                    mode_value = df[column].mode()[0]
                    df[column] = df[column].fillna(mode_value)
                    print(f"[{df_name}] The column '{column}' has been compensated with the mode: {mode_value}")
                else:
                    print(f"[{df_name}] column '{column}' Does not contain a value for the mode; no substitution has been made.")



print(train.info())
print('--------------------------------------------------------')
print(test.info())


X_train = train.drop(columns=['id', 'Personality'])
y_train = train['Personality']

X_test = test.drop(columns=['id'])


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Stage_fear', 'Drained_after_socializing']

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le


target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_tr, y_tr)

y_pred = model.predict(X_val)
print("Model accuracy on the validation set: ", accuracy_score(y_val, y_pred))


y_test_pred = model.predict(X_test)

y_test_labels = target_encoder.inverse_transform(y_test_pred)

submission = test[['id']].copy()
submission['Personality'] = y_test_labels

submission.to_csv('submission.csv', index=False)




