# Basic data manipulation and analysis
import pandas as pd  # For dataframes and data handling
import numpy as np    # For numerical operations

# Visualization libraries
import seaborn as sns  # For statistical data visualization
import matplotlib.pyplot as plt  # For basic plotting

# Preprocessing tools
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer  

# Model and pipeline
from xgboost import XGBClassifier  # Extreme Gradient Boosting Classifier
from sklearn.pipeline import Pipeline  # To build machine learning pipelines

# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score  
# For measuring model performance

# Model selection and validation
from sklearn.model_selection import GridSearchCV, train_test_split

# TensorFlow deep learning library
import tensorflow as tf  # For building neural networks
from tensorflow.keras.callbacks import EarlyStopping  # Stop training early if no improvement

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

import os

# Suppress most TensorFlow/XLA CUDA plugin warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

# Optional: reduce absl logging
os.environ["ABSL_LOGGING_LEVEL"] = "3"



from sklearn.metrics import classification_report


# Load the Training Dataset
df_train=pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
# Load the Testing Dataset
df_test=pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
# Load the Submission
submission=pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')


# Lets check the First Five Rows of Training Data
df_train.head()


# Lets check the First Five Rows of Testing Data
df_test.head()


# Lets check the First Five Rows of Submission 
submission.head()


df = df_train


print(f"The Total Rows of training data are {df.shape[0]}")
print(f"The Total Columns of training data are {df.shape[1]}")


df.columns


# Basic info
df.info()


# Drop non-informative columns
df.drop(columns=['id', 'CustomerId', 'Surname'], inplace=True)


# Check for missing values
df.isnull().sum()



# countplot of Exited column 
sns.countplot(x='Exited', data=df, palette='Set2')
plt.title('Customer Churn Distribution')
plt.xlabel('Exited (1 = Churned)')
plt.ylabel('Count')
plt.show()

print(df['Exited'].value_counts(normalize=True).map("{:.2%}".format))


num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
df[num_cols].describe().T


# Histograms
df[num_cols].hist(bins=30, figsize=(14, 8), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()


cat_cols = ['Geography', 'Gender', 'HasCrCard','Tenure', 'IsActiveMember', 'NumOfProducts']

for col in cat_cols:
    sns.countplot(data=df, x=col, palette='Set2')
    plt.title(f"Distribution of {col}")
    plt.show()



# Churn rate across categories
for col in cat_cols:
    churn_rate = df.groupby(col)['Exited'].mean().sort_values(ascending=False)
    churn_rate.plot(kind='bar', color='coral')
    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.xlabel(col)
    plt.show()



# Age distribution by churn
sns.kdeplot(data=df, x='Age', hue='Exited', fill=True)
plt.title("Age vs Churn Distribution")
plt.show()

# Balance vs Churn
sns.boxplot(data=df[df['Balance'] > 0], x='Exited', y='Balance')
plt.title("Balance (non-zero) vs Churn")
plt.show()

# CreditScore
sns.kdeplot(data=df, x='CreditScore', hue='Exited', fill=True)
plt.title("Credit Score vs Churn")
plt.show()



corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()



for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()



def encode_labels(df):
    for col in df.select_dtypes(include='object').columns.tolist():
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


# Scale the data for better performance of the model 
def scale_data(df):
    df = encode_labels(df)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def transformation(df):
    plt_normal = QuantileTransformer(output_distribution='normal')
    df['CreditScore'] = plt_normal.fit_transform(df[['CreditScore']])
    df['Age'] = plt_normal.fit_transform(df[['Age']])
    return df


# Create feature and labels 
X = df.drop('Exited', axis=1)
y = df['Exited']
X = scale_data(X)

# split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# build the model 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# stop training early if the validation loss does not improve for 5 epochs
early_stopping = EarlyStopping(patience=5)

# train the model
history = model.fit(X_train, y_train, epochs=100,callbacks=[early_stopping], batch_size=32,validation_data=(X_test,y_test),verbose=1)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Training Loss : {loss:.2f}")
print(f"Training Accuracy : {accuracy:.2f}")


# check the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training VS Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train','Validation'],loc='upper right')
plt.show()


# check the training and validation loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training VS Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'],loc='upper right')
plt.show()


# train the xgboost model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# predict the test data
y_pred = xgb.predict(X_test)

print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='micro'))
print('Recall score: ', recall_score(y_test, y_pred, average='micro'))
print('F1 score: ', f1_score(y_test, y_pred, average='micro'))

# plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


from sklearn.model_selection import RandomizedSearchCV

# Initialize base model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [1, 1.5, 2.0],
    'reg_alpha': [0, 0.1, 0.5]
}



# Set up randomized search
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=30,
    scoring='precision',  # You can also try 'accuracy'
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit on training data
random_search.fit(X_train, y_train)

# Best parameters and model
best_model = random_search.best_estimator_
print("ğŸ”� Best Parameters:", random_search.best_params_)

# Predict and evaluate
y_pred = best_model.predict(X_test)

print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='micro'))
print('Recall score: ', recall_score(y_test, y_pred, average='micro'))
print('F1 score: ', f1_score(y_test, y_pred, average='micro'))
print("\nğŸ“‹ Classification Report:\n", classification_report(y_test, y_pred))



# plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Convert categorical columns to numerical values
df_test = encode_labels(df_test)

# predict the test data and create a submission file of probability of each class
y_pred = best_model.predict_proba(df_test.drop(['id', 'CustomerId', 'Surname'], axis=1))

submission['Exited'] = y_pred[:, 1]
submission.to_csv('submission.csv', index=False)


import pickle 
# Save the model to a file
pickle.dump(xgb, open('XGBoostClassifier.pkl', 'wb'))

