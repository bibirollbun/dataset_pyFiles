# !pip install imblearn
!pip install scikit-learn --upgrade
!pip install imbalanced-learn --upgrade
# !uv pip install -q scikit-learn==1.6.1 imblearn --system


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv')


train.info()


# train.describe()


# train.head()


# fig = px.imshow(train.corr(numeric_only=True), text_auto=True, aspect='auto', width=800, height=800, color_continuous_scale='plotly3')
# fig.update_traces(textfont_size=10)
# fig.show()


# fig = px.histogram(train, x='y')
# fig.show()


# fig = make_subplots(rows=3, cols=3, subplot_titles=('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'))

# job_counts = train['job'].value_counts()
# fig.add_trace(go.Bar(x=job_counts.index, y = job_counts.values), row=1, col=1)

# marital_counts = train['marital'].value_counts()
# fig.add_trace(go.Bar(x=marital_counts.index, y = marital_counts.values), row=1, col=2)

# education_counts = train['education'].value_counts()
# fig.add_trace(go.Bar(x=education_counts.index, y = education_counts.values), row=1, col=3)

# default_counts = train['default'].value_counts()
# fig.add_trace(go.Bar(x=default_counts.index, y = default_counts.values), row=2, col=1)

# housing_counts = train['housing'].value_counts()
# fig.add_trace(go.Bar(x=housing_counts.index, y = housing_counts.values), row=2, col=2)

# loan_counts = train['loan'].value_counts()
# fig.add_trace(go.Bar(x=loan_counts.index, y = loan_counts.values), row=2, col=3)

# contact_counts = train['contact'].value_counts()
# fig.add_trace(go.Bar(x=contact_counts.index, y = contact_counts.values), row=3, col=1)

# month_counts = train['month'].value_counts()
# fig.add_trace(go.Bar(x=month_counts.index, y = month_counts.values), row=3, col=2)

# poutcome_counts = train['poutcome'].value_counts()
# fig.add_trace(go.Bar(x=poutcome_counts.index, y = poutcome_counts.values), row=3, col=3)

# fig.update_layout(height=900, width=900, title='Character Variable Distributions', showlegend=False)

# fig.show()


# train.info()


# fig = make_subplots(rows=4, cols=2, subplot_titles=('age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y'))

# fig.add_trace(go.Histogram(x=train['age']), row=1, col=1)

# fig.add_trace(go.Histogram(x=train['balance']), row=1, col=2)

# fig.add_trace(go.Histogram(x=train['day']), row=2, col=1)

# fig.add_trace(go.Histogram(x=train['duration']), row=2, col=2)

# fig.add_trace(go.Histogram(x=train['campaign']), row=3, col=1)

# fig.add_trace(go.Histogram(x=train['pdays']), row=3, col=2)

# fig.add_trace(go.Histogram(x=train['previous']), row=4, col=1)

# fig.add_trace(go.Histogram(x=train['y']), row=4, col=2)

# fig.update_layout(height=1000, width=1000, title='Numeric Variable Distributions', showlegend=False)

# fig.show()


# fig = px.scatter_matrix(train, dimensions=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y'], title='Feature Matrix')

# fig.update_traces(diagonal_visible=False, marker_line_color='white', marker_line_width=.25)

# fig.update_layout(height=1000, width=1000, title_text='Feature Scatter Matrix', showlegend=False)

# fig.show()


X = train.drop(columns=['id','y'], axis=1)
y = train['y']


cat = X.select_dtypes(include=['object']).columns
num = X.select_dtypes(exclude=['object']).columns


num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])


preprocessing = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    ]
)


# Synthetic Minority Over-Sampling Technique (SMOTE)
smote = SMOTE(sampling_strategy='minority')


final_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('smote', smote), # SMOTE step
    ('classifier', LogisticRegression(max_iter=1000))
    # ('knn', KNeighborsClassifier(n_neighbors=3))
])


cv = StratifiedKFold(n_splits=5)
# scores = cross_val_score(final_pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)


# print(f'Mean ROC AUC score: {np.mean(scores):.4f}')
# print(f'Standard Deviation of ROC AUC score: {np.std(scores):.4f}')


y_pred_proba = cross_val_predict(final_pipeline, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]


roc_auc = roc_auc_score(y, y_pred_proba)

print(f"ROC AUC on the entire dataset using cross_val_predict: {roc_auc:.4f}")


# final_pipeline.fit(X,y)


# y_pred_train = final_pipeline.predict(X) #making predictions based on text X variables using final pipeline transformations

# y_pred_train_probs = final_pipeline.predict_proba(X)[:,1]



y_pred_train = cross_val_predict(final_pipeline, X, y, cv=cv, n_jobs=-1)



#comparing the actual y values from the train dataset to the values we predicted from our model
results = pd.DataFrame({
    'actual': y,
    'predicted': y_pred_train,
    'predicted probability': y_pred_proba
})


results.head(20)


conf_matrix = confusion_matrix(y, y_pred_train)
print('Confusion Matrix:\n', conf_matrix)

z_annotations = [['TN', 'FP'], ['FN', 'TP']]

cm_display = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=['Predicted No', 'Predicted Yes'],
    y=['Actual No', 'Actual Yes'],
    colorscale='Viridis',
    colorbar=dict(title='Count'),
    text=z_annotations,
    texttemplate='%{text}'
))
cm_display.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual', width=600, height=600)
cm_display.show()


print('t\t\Classification Report')
print(classification_report(y, y_pred_train))


fpr, tpr, thresholds = roc_curve(y, y_pred_train)

roc_auc = auc(fpr, tpr)

youden_j = tpr - fpr
optimal_threshold_index = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_threshold_index]

print(f'Optimal Threshold: {optimal_threshold:.4f}')

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area={roc_auc:.2f})')
plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', marker='o', label=f'Optimal Threshold = {optimal_threshold:.4f}')
plt.plot([0,1],[0,1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# submission = test[['id']]
# submission['y'] = y_pred_test
# submission.head(35)


# submission.to_csv('submission.csv', index=False)

