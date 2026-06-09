

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


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.tail()


train.info()


#Describe showing Only the requested statistics (mean, minimum and maximum). Then, transpose the table.

train.describe().loc[['mean','min','max']].T


correlation=train.corr(method='kendall')

# heatmap of the correlation 
plt.figure(figsize=(10,10))
plt.title('Correlation heatmap')
sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,center=1);


#Original figsize 15,10
train.hist(figsize=(20,15), bins=50, color='green', edgecolor='red')
plt.suptitle("Histogram of Features")
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
%matplotlib inline


from sklearn.preprocessing import LabelEncoder

#fill in mean for floats
for c in train.columns:
    if train[c].dtype=='float16' or  train[c].dtype=='float32' or  train[c].dtype=='float64':
        train[c].fillna(train[c].mean())

#fill in -999 for categoricals
train = train.fillna(-999)
# Label Encoding
for f in train.columns:
    if train[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))
        
print('Labelling done.')


#Linear regression, first create test and train dataset
X=train.loc[:,['RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']].values
y=train.loc[:,'BeatsPerMinute'].values


# Creating a test and training dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#Code by Arpita Gupta https://www.kaggle.com/arpita28/analysis-of-spotify-trends

# Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)


#Displaying the difference between the actual and the predicted
y_pred = regressor.predict(X_test)
train_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(train_output)


#Code by Arpita Gupta https://www.kaggle.com/arpita28/analysis-of-spotify-trends

#Checking the accuracy of Linear Regression
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#Code by Arpita Gupta https://www.kaggle.com/arpita28/analysis-of-spotify-trends

plt.figure(figsize=(10,10))
plt.plot(y_pred,y_test,color='black',linestyle='dashed',marker='o',markerfacecolor='green',markersize=10)
plt.title('Error analysis')
plt.xlabel('Predicted values')
plt.ylabel('Test values');


# 1. Define the features (X) and target (y) using the FULL training dataset
features = ['RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
X_full_train = train[features]
y_full_train = train['BeatsPerMinute']

# 2. Prepare the final test data from test.csv
X_final_test = test[features]

# 3. Initialize and train the Linear Regression model
# We use the same regressor you already defined
regressor = LinearRegression()
regressor.fit(X_full_train, y_full_train)

# 4. Make predictions on the final test data
final_predictions = regressor.predict(X_final_test)

# 5. Create the submission DataFrame
# It needs an 'id' column and a 'BeatsPerMinute' column
submission_df = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': final_predictions
})

# 6. Save the DataFrame to a CSV file
# index=False prevents pandas from writing row indices into the file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print("Here's a preview of your submission file:")
print(submission_df.head())

