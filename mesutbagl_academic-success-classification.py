import pandas as pd
pd.set_option('display.max_columns',100)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


train_df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


train_df.head()


train_df.info()


train_df.shape


train_df.isnull().sum()


train_df.describe()


train_df['Target'].value_counts()


train_df['Marital status'].value_counts()


train_df['Course'].value_counts()


train_df.corr(numeric_only=True)


test_df.head()


test_df.shape


mapping = {
    'Enrolled': 1,
    'Graduate': 2,
    'Dropout': 0
}


train_df['Target_num'] = train_df['Target'].map(mapping)


# Plot distribution of cover types
plt.figure(figsize=(10, 6))
train_df['Target'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Target')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# Calculate correlation matrix
correlation_matrix = train_df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# Bar plot
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Marital status', hue='Target')
plt.title('Marital Status vs. Educational Outcome')
plt.xlabel('Marital Status (0: Single, 1: Married)')
plt.ylabel('Count')
plt.legend(title='Target', labels=['Dropout', 'Enrolled', 'Graduate'])
plt.show()


order = train_df['Marital status'].value_counts().index
ax = sns.countplot(x=train_df['Marital status'], order=order)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)


# Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_df, x='Target', y='Admission grade')
plt.title('Admission Grades by Educational Outcome')
plt.xlabel('Target')
plt.ylabel('Admission Grade')
plt.xticks(ticks=[0, 1, 2], labels=['Dropout', 'Enrolled', 'Graduate'])
plt.show()


# Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_df, x='Target', y='Course')
plt.title('Courses by Educational Outcome')
plt.xlabel('Target')
plt.ylabel('Course')
plt.xticks(ticks=[0, 1, 2], labels=['Dropout', 'Enrolled', 'Graduate'])
plt.show()


# Set the style of seaborn
sns.set(style="whitegrid")

# Create a histogram and a KDE plot
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age at enrollment'], bins=10, kde=True, color='blue', stat='density', alpha=0.5)

# Add titles and labels
plt.title('Distribution of Age at Enrollment')
plt.xlabel('Age at Enrollment')
plt.ylabel('Density')
plt.xlim(train_df['Age at enrollment'].min() - 1, train_df['Age at enrollment'].max() + 1)

# Show the plot
plt.show()


# Set the style of seaborn
sns.set(style="whitegrid")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Target', y='Curricular units 1st sem (grade)', palette='Set2')

# Add titles and labels
plt.title('1st Semester Grade Distribution by Target')
plt.xlabel('Target')
plt.ylabel('1st Semester Grade')
plt.xticks(ticks=[0, 1, 2], labels=['Dropout', 'Enrolled', 'Graduate'])

# Show the plot
plt.show()


# Set the style of seaborn
sns.set(style="whitegrid")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Target', y='Curricular units 2nd sem (grade)', palette='Set2')

# Add titles and labels
plt.title('2nd Semester Grade Distribution by Target')
plt.xlabel('Target')
plt.ylabel('2nd Semester Grade')
plt.xticks(ticks=[0, 1, 2], labels=['Dropout', 'Enrolled', 'Graduate'])

# Show the plot
plt.show()


upper_bound=train_df.quantile(q=.97,numeric_only=True)
lower_bound = train_df.quantile(q=.03,numeric_only=True)


# Filter out the outliers based on upper bounds
train_df = train_df[(train_df['Age at enrollment'] <= upper_bound['Age at enrollment'])]
train_df = train_df[(train_df['Course'] <= upper_bound['Course'])]
train_df = train_df[(train_df['Admission grade'] <= upper_bound['Admission grade'])]
# Filter out the outliers based on lower bounds
train_df = train_df[(train_df['Age at enrollment'] >= lower_bound['Age at enrollment'])]
train_df = train_df[(train_df['Course'] >= lower_bound['Course'])]
train_df = train_df[(train_df['Admission grade'] >= lower_bound['Admission grade'])]


# Dropping the columns with less correlation on Target
train_df=train_df.drop(columns=["Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation", 
                                "Father's occupation", "Educational special needs", "International", "Curricular units 1st sem (credited)",
                               "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
                               "Curricular units 2nd sem (without evaluations)", "Unemployment rate", "Inflation rate"])
# Dropping the same columns for test_df
test_df=test_df.drop(columns=["Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation", 
                                "Father's occupation", "Educational special needs", "International", "Curricular units 1st sem (credited)",
                               "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
                               "Curricular units 2nd sem (without evaluations)", "Unemployment rate", "Inflation rate"])


x=train_df.drop(['id','Target', 'Target_num'],axis=1)
y=train_df[['Target']]


scaler = StandardScaler()
x=scaler.fit_transform(x)


def classification_algo(x, y, confusion_mtr=False, classification_rpt=False):
    g = GaussianNB()
    b = BernoulliNB()
    l = LogisticRegression()
    d = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    h = GradientBoostingClassifier()
    k = KNeighborsClassifier()
    
    algos = [g, b, l, d, rf, h, k]
    algo_names = ['Gaussian NB', 'Bernoulli NB', 'Logistic Regression', 
                  'Decision Tree Classifier', 'Random Forest Classifier', 
                  'Gradient Boosting Classifier', 'KNeighbors Classifier']

    accuracy = []
    confusion = []
    classification = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Prepare a DataFrame to store results
    result = pd.DataFrame(columns=['Accuracy Score', 'Confusion Matrix', 'Classification Report'], 
                          index=algo_names)

    labels = sorted(y["Target"].unique())

    for algo in algos:
        p = algo.fit(x_train, y_train).predict(x_test)
        accuracy.append(accuracy_score(y_test, p))
        confusion.append(confusion_matrix(y_test, p, labels=labels))
        classification.append(classification_report(y_test, p))

    # Store results
    result['Accuracy Score'] = accuracy
    result['Confusion Matrix'] = confusion
    result['Classification Report'] = classification

    # Sort results by accuracy
    r_table = result.sort_values('Accuracy Score', ascending=False)
    
    if confusion_mtr:
        for index, row in r_table.iterrows():
            confusion_mat = np.array(row['Confusion Matrix'])
            print(f"Confusion Matrix of {index}")
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_mat, annot=True, fmt="d", 
                        xticklabels=labels, yticklabels=labels, cmap="Blues")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.show()
    
    if classification_rpt:
        for index, row in r_table.iterrows():
            print(f"Classification Report of {index}:")
            print(row['Classification Report'])

    return r_table[['Accuracy Score']]


classification_algo(x,y,confusion_mtr=True,classification_rpt=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
gb = GradientBoostingClassifier()
model1=gb.fit(x_train, y_train)


submission=pd.DataFrame({
    'id':test_df['id']}
)


test_df.drop('id',axis=1,inplace=True)


predictions=model1.predict(test_df)


submission['Target']=predictions


submission.to_csv("submission.csv", index=False)


import joblib
joblib.dump(model1, 'best_model.pkl')


# Prepare the data
x=train_df.drop(['id','Target', 'Target_num'],axis=1)
y=train_df[['Target']]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y['Target'])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model2=Sequential()
model2.add(Dense(8, activation='relu'))
model2.add(Dense(32,activation='relu')) 
model2.add(Dense(64,activation='relu')) 
model2.add(Dense(128,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(32,activation='relu'))
model2.add(Dense(7,activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# Summary of the model
model2.summary()

# Fit the model
history = model2.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=32)


predictions=model2.predict(x_test)
predictions_labels = np.argmax(predictions, axis=1)
accuracy_score(predictions_labels,y_test)
val_loss, val_accuracy = model2.evaluate(x_test, y_test)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')


# Load test data
test_df=pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
test_df=test_df.drop(columns=["Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation", 
                                "Father's occupation", "Educational special needs", "International", "Curricular units 1st sem (credited)",
                               "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
                               "Curricular units 2nd sem (without evaluations)", "Unemployment rate", "Inflation rate"])
# Prepare test data for predictions
submission2 = pd.DataFrame({'id': test_df['id']})
test_df.drop('id', axis=1, inplace=True)

# Make predictions on the test set
predictions = model2.predict(test_df)
predictions_labels = np.argmax(predictions, axis=1)


predictions_labels


# Define the mapping
label_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

# Map the numerical labels to their corresponding string labels
predictions_strings = np.vectorize(label_mapping.get)(predictions_labels)


# Add predictions to submission DataFrame
submission2['Target'] = predictions_strings

# Save submission
submission2.to_csv('submission2.csv', index=False)




