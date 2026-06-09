# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "iframe"
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


training_dataset = "/kaggle/input/data-guild-will-you-be-productive-today/train.csv"
df = pd.read_csv(training_dataset)
df.head()


# Check data to find anomalies

def find_anomalies_of_dataset(df):
    missing_val = df.isnull().sum()
    if not missing_val.empty:
        missing_val = missing_val[missing_val > 0]
    
    duplicate_val = df.duplicated().sum()
    
    unique_val = df.nunique()
    
    data_types = df.dtypes
    
    dataset_distribution = df['productive_day'].value_counts()
    
    print('Sum of Missing values:')
    if not missing_val.empty:
        print(missing_val)
    else:
        print('No missing values in the dataset!!')
    print('-----------')
    
    print('Number of duplicated samples: ', duplicate_val)
    print('-----------')
    
    print('Number of unique samples: ')
    print(unique_val)
    print('-----------')
    
    print('Data types of the samples: ')
    print(data_types)
    print('-----------')
    
    print('Target distribution of the samples: ')
    print(dataset_distribution)
    print('-----------')


# We want to analyze each feature against the 'productive day' feature
target = 'productive_day'

num_features = df.shape[1] - 1  # excluding target
cols = 3
rows = (num_features // cols)

features = [col for col in df.columns if col != target]

fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[f"{col} by {target}" for col in features]
)

for idx, col in enumerate(features):
    row = idx // cols + 1
    col_pos = idx % cols + 1

    for category in df[target].unique():
        fig.add_trace(
            go.Box(
                y=df[df[target] == category][col],
                name=str(category),
                boxmean='sd',
                marker=dict(opacity=0.7),
                showlegend=(idx == 0)  # Show legend only on first subplot
            ),
            row=row,
            col=col_pos
        )

fig.update_layout(
    height=300 * rows,
    width=400 * cols,
    title_text="Box Plots of Features by Productive Day",
    boxmode='group'  # Group by category
)

fig.show()


df = df.fillna(df.median())


df.drop_duplicates(inplace=True)
find_anomalies_of_dataset(df)


z_scores = zscore(df['lines_of_code_written'])
outlier_threshold = 3.0

# Identify outliers based on threshold
outliers_mask = np.abs(z_scores) > outlier_threshold
cleaned_df = df[~outliers_mask]

target = 'productive_day'
filtered = 'lines_of_code_written'

fig = go.Figure()

for category in cleaned_df[target].unique():
    fig.add_trace(
        go.Box(
            y=cleaned_df[cleaned_df[target] == category][filtered],
            name=str(category),
            boxmean='sd',
            marker=dict(opacity=0.7)
        )
    )

# Update layout
fig.update_layout(
    title=f"Box Plot of {filtered} by {target}",
    yaxis_title=filtered,
    xaxis_title=target,
    boxmode='group'
)

fig.show()


correlation_matrix = cleaned_df.corr()

fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
pio.show(fig) 


target_correlation = correlation_matrix['productive_day'].abs().sort_values(ascending=False)
print('----------------')
print("Correlation with target variable")
print(target_correlation)


target = 'productive_day'
X = cleaned_df.drop(columns=[target])
y = cleaned_df[target]

preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_transformed = preprocessing_pipeline.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

print("Main Data     :", cleaned_df.shape)
print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)
print("X_test shape. :", X_test.shape)
print("y_test shape  :", y_test.shape)


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'AdaBoost Classifier': AdaBoostClassifier(),
    'XGBoost Classifier': XGBClassifier(),
    'Support Vector Classifier': SVC(),
    'MLP Classifier': MLPClassifier(max_iter=500, hidden_layer_sizes=(100,), solver='adam', learning_rate_init=0.001, random_state=42)
}

model_accuracies = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy


accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

fig = px.bar(accuracy_df, x='Model', y='Accuracy', color='Model', title='Model Accuracy Comparison')
fig.update_layout(xaxis_title='Model', yaxis_title='Accuracy')
pio.show(fig)


classifier = models['Logistic Regression']

test_df = pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/test.csv")

X_new_transformed = preprocessing_pipeline.transform(test_df)

y_pred_new = classifier.predict(X_new_transformed)

y_pred_new

