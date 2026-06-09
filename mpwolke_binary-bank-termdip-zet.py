# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.tail()


train.isnull().sum() 


train.info()


#Describe showing Only the requested statistics (mean, minimum and maximum). Then, transpose the table.

train.describe().loc[['mean','min','max']].T


numerical_cols = ['id',
 'age',
 'balance',
 'day',
 'duration',
 'campaign',
 'pdays',
 'previous',
 'y'
 ]


# OutlierPandas https://www.kaggle.com/code/abhyudaya456/s5e6-eda-for-predicting-optimal-fertilizers/notebook 
plt.figure(figsize=(10,6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='summer')
plt.title("Correlation Between Numerical Features")
plt.show()


#Original figsize 15,10
train[numerical_cols].hist(figsize=(20,15), bins=30, color='Green', edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.show()


#By Vishnupriya https://www.kaggle.com/code/vishnupriyagarige/predict-the-introverts-from-the-extroverts/notebook

#  columns to plot
cols = ['default', 'housing','loan', 'y']

# Set up 11x2 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))#Original 10,6
axes = axes.flatten()  # Flatten to iterate easily

# Generate pie charts
for i, col in enumerate(cols):
    train[col].value_counts().plot.pie(
        ax=axes[i],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        shadow=True
    )
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("")  # Remove y-label for cleaner plot

plt.tight_layout()
plt.show()


#By Pedro Andrade https://www.kaggle.com/code/pbizil/datahackers-managers-radiografia-dos-gestores

jobs_counts = train["job"].value_counts().head(12)#Try different values of head
sns.set(style="white")
plt.figure(figsize=(8, 6))
#x=ok_counts.index, y=ok_counts.values
ax = sns.barplot(x=jobs_counts.index, y=jobs_counts.values, color=sns.color_palette("Reds", n_colors=5)[3])
plt.title("Distribution of Jobs", fontsize=16)
plt.xlabel("Jobs", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=60, fontsize=11)
plt.yticks(fontsize=11)
sns.despine()

#+2 is good if chart is vertical. +20 worked for horizontal
for i, v in enumerate(jobs_counts.values):
    ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


#https://stackoverflow.com/questions/52273546/matplotlib-typeerror-axessubplot-object-is-not-subscriptable
#https://stackoverflow.com/questions/76543569/rotation-of-xticks-in-seaborn-subplots

fig , axarr = plt.subplots(nrows=1, ncols=2,squeeze=False, figsize = (12,8))

sns.countplot(data=train, x="marital", ax= axarr[0][0], hue='default')
ax= axarr[0][0].tick_params("x",labelrotation=45)


sns.countplot(data=train, x="education",ax=axarr[0][1], hue='default')
ax= axarr[0][1].tick_params("x",labelrotation=45)

plt.show()


#GeeksForGeeks https://www.geeksforgeeks.org/loan-eligibility-prediction-using-machine-learning-models-in-python/

import seaborn as sb
plt.subplots(figsize=(15, 10))
for i, col in enumerate(['contact','poutcome']):
    plt.subplot(1, 2, i+1)
    sb.countplot(data=train, x=col, hue='y')
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()


data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


#By Heidar Mirhaji Sadati https://www.kaggle.com/code/heidarmirhajisadati/loan-approval-intelligence-advanced-loan-approval/notebook

# Encode categorical variables
# Label encoding for binary categorical variables 
label_encoder = LabelEncoder()
data['default'] = label_encoder.fit_transform(data['default'])


#By Heidar Mirhaji Sadati https://www.kaggle.com/code/heidarmirhajisadati/loan-approval-intelligence-advanced-loan-approval/notebook

# One-hot encoding for other categorical variables
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'poutcome'], drop_first=True)

# Feature Scaling 
scaler = StandardScaler()
numeric_features = ['age',
 'default',
 'balance',
 'day',
 'duration',
 'campaign',
 'pdays',
 'previous']
data[numeric_features] = scaler.fit_transform(data[numeric_features])


#By Heidar Mirhaji Sadati https://www.kaggle.com/code/heidarmirhajisadati/loan-approval-intelligence-advanced-loan-approval/notebook

# Splitting data into features (X) and target (y)
X = data.drop(['y', 'id'], axis=1)  # Drop the target variable and id
y = data['y']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and test sets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


#By Heidar Mirhaji Sadati https://www.kaggle.com/code/heidarmirhajisadati/loan-approval-intelligence-advanced-loan-approval/notebook

# Model Training and Ensemble
# Initialize a list to store individual models
models = []

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)  
lr_model.fit(X_train, y_train)
models.append(('Logistic Regression', lr_model))

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
models.append(('Random Forest', rf_model))


# Ensemble Predictions
ensemble_predictions = np.zeros(X_test.shape[0])


# Apply each model and weight their probabilities
for name, model in models:
    predictions = model.predict_proba(X_test)[:, 1]  
    ensemble_predictions += predictions 


# Normalize the predictions by dividing by the number of models
ensemble_predictions /= len(models)


#By Heidar Mirhaji Sadati https://www.kaggle.com/code/heidarmirhajisadati/loan-approval-intelligence-advanced-loan-approval/notebook

# Convert probabilities to class labels (0 or 1) based on a threshold (e.g., 0.5)
ensemble_predictions_labels = (ensemble_predictions > 0.5).astype(int)


#By Heidar Mirhaji Sadati https://www.kaggle.com/code/heidarmirhajisadati/loan-approval-intelligence-advanced-loan-approval/notebook

# Evaluate the ensemble performance
accuracy = accuracy_score(y_test, ensemble_predictions_labels)
roc_auc = roc_auc_score(y_test, ensemble_predictions)
print(f"Ensemble Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(classification_report(y_test, ensemble_predictions_labels))


#By Heidar Mirhaji Sadati https://www.kaggle.com/code/heidarmirhajisadati/loan-approval-intelligence-advanced-loan-approval/notebook

# Visualizing ROC Curve for the Ensemble Model
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, ensemble_predictions)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble Model (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Ensemble Model')
plt.legend(loc="lower right")
plt.show()

