import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
orig = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')
orig['y'] = orig['y'].map({'no': 0, 'yes': 1})
train_df = pd.concat([train_df, orig], ignore_index=True)
train_df = train_df.drop_duplicates()
train_df.head()



sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
train_df.drop(columns=['id']).describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].plot(kind='bar', figsize=(12, 6))
plt.title('Descriptive Statistics for Continuous Features')
plt.show()


warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*use_inf_as_na.*")

plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
sns.histplot(train_df['age'], kde=True, color='skyblue', bins=20)
plt.title('Distribution of Age')


plt.subplot(1, 2, 2)
sns.histplot(train_df['balance'], kde=True, color='orange', bins=20)
plt.title('Distribution of Balance')

plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=train_df, palette='Set2')
plt.title('Distribution of Target Variable (y)')
plt.show()




categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

plt.figure(figsize=(14, 12))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, data=train_df, palette='Set2')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()




plt.figure(figsize=(14, 6))


plt.subplot(1, 3, 1)
sns.boxplot(x='y', y='age', data=train_df, palette='Set2')
plt.title('Age vs Target Variable (y)')


plt.subplot(1, 3, 2)
sns.boxplot(x='y', y='balance', data=train_df, palette='Set2')
plt.title('Balance vs Target Variable (y)')


plt.subplot(1, 3, 3)
sns.boxplot(x='y', y='duration', data=train_df, palette='Set2')
plt.title('Duration vs Target Variable (y)')

plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 6))


sns.countplot(x='job', hue='y', data=train_df, palette='Set2')
plt.title('Job vs Target Variable (y)')
plt.xticks(rotation=45)
plt.show()


sns.countplot(x='marital', hue='y', data=train_df, palette='Set2')
plt.title('Marital Status vs Target Variable (y)')
plt.xticks(rotation=45)
plt.show()


y_train = train_df["y"]
train_id = train_df["id"]
train = train_df.drop(columns=['id'])
train = train.drop(columns=['y'])
test_ID = test_df["id"]
test = test_df.drop(columns=['id'])


all_data = pd.concat((train, test)).reset_index(drop=True)

all_data['job_marital'] = train_df['job'].astype(str) + '_' + train_df['marital'].astype(str)
all_data['job_education'] = train_df['job'].astype(str) + '_' + train_df['education'].astype(str)


all_data = all_data.astype('str')


X_train = all_data[:train_id.shape[0]]
X_test = all_data[train_id.shape[0]:]


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the CatBoostClassifier model
model = CatBoostClassifier(
    iterations=10000,       # Number of iterations, default is 1000, can be tuned based on needs
    learning_rate=0.05,    # Learning rate, lower learning rates may require more iterations
    depth=6,               # Tree depth, default is 6, can be adjusted to find the best value
    task_type='GPU',
    cat_features=X_train.columns.to_list(),  # Specify the categorical feature column indices
    verbose=500,           # Output information every 500 iterations to help monitor the training process
    loss_function='Logloss',  # Log loss function for binary classification task
    custom_metric=['AUC'],    # Additional evaluation metrics, such as AUC
    early_stopping_rounds=500  # Stop training if there is no improvement on the validation set for 500 iterations
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Prediction
y_pred = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'id': test_ID,
    'Personality': y_pred  # output raw probability decimals
})
submission.to_csv("submission.csv", index=False)
submission.head()

