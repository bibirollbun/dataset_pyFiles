# for data manipulation
import pandas as pd
import numpy as np

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px



# for preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV


# for model training
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# import model for regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# import tensorflow for creating neural networks
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# import CatBoostClassifier
from catboost import CatBoostClassifier


# for model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# function for Sub-Heading
def heading(title):
    print('-'*80)
    print(title.upper())
    print('-'*80)


df_train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
df_submission= pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')


df_train.head()


df_train.info()


df_train.shape
print(f'There are {df_train.shape[0]} rows and {df_train.shape[1]} columns in training data.')


df_train.describe().style.format(precision=2).background_gradient(cmap="RdPu")


df_train.isnull().sum() / len(df_train) * 100


heading('visulalizing the missing values in the dataset')
# Set up the figure size for the plot
plt.figure(figsize=(18, 9))

# Create the heatmap to plot null values
sns.heatmap(df_train.isnull(), cbar=False, yticklabels=False, cmap='RdPu')

# Show the plot
plt.show()


df_train_copy = df_train.copy()

cat_cols = df_train_copy.select_dtypes(include=['object', 'category'])
for i in cat_cols.columns:
    df_train_copy[i] = LabelEncoder().fit_transform(df_train_copy[i])

cor_matrix = df_train_copy.corr()
plt.figure(figsize=(20, 14))
sns.heatmap(cor_matrix, annot=True, cmap='RdPu', square=False)
plt.title('Correlation Matrix', fontsize=24)
plt.xlabel('X_Features', fontsize=17)
plt.ylabel('Y_Features', fontsize=17
)
plt.show()


# Data for the pie chart
labels = ["Churn: Yes", "Churn: No"]
values = [1869, 5163]
colors = ['#ff6666', '#66b3ff']
explode = [0.1, 0]

# Create pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo='label+percent', 
                             marker=dict(colors=colors, line=dict(color='#000000', width=2)),
                             hoverinfo='label+percent', pull=[0.1, 0])])

# Update layout
fig.update_layout(title='Churn Distribution',
                  showlegend=False,
                  annotations=[dict(text='Churn', x=0.5, y=0.5, font_size=25, showarrow=False)],
                  width=800,
                  height=800
                  
                  )

# Show plot
fig.show()



import plotly.graph_objects as go

# Data for the pie chart
labels_gender = ["Female: Churn", "Female: No Churn", "Male: Churn", "Male: No Churn"]
sizes_gender = [939, 2544, 930, 2619]
colors_gender = ['#ff6666', '#ffb3e6', '#66b3ff', '#c2c2f0']
explode_gender = [0.1, 0, 0.1, 0]

# Create pie chart
fig_gender = go.Figure(data=[go.Pie(labels=labels_gender, values=sizes_gender, hole=0.5, textinfo='label+percent', 
                                     marker=dict(colors=colors_gender, line=dict(color='#000000', width=2)),
                                     hoverinfo='label+percent', pull=explode_gender)])

# Update layout
fig_gender.update_layout(title='Churn Distribution w.r.t Gender',
                         showlegend=False,
                         annotations=[dict(text='Gender', x=0.5, y=0.5, font_size=20, showarrow=False)],
                         width=800,
                         height=800)

# Show plot
fig_gender.show()



heading('Top 10 Credit Score')
df_train['CreditScore'].value_counts().sort_values(ascending=False).head(10)


pl = df_train['CreditScore'].value_counts().sort_values(ascending=False).head(10)

plt.figure(figsize=(14, 8))
sns.barplot(x=pl.index, y=pl, palette="RdBu")
plt.title('Top 10 Credit Score')
plt.show()


heading('Top 10 Tenure')
ten = df_train['Tenure'].value_counts().sort_values(ascending=False).head(10)
print(ten)


plt.figure(figsize=(14, 8))
sns.barplot(x=ten.index, y=ten, palette="RdBu")
plt.title('Top 10 Tenure')
plt.show()


top_balance_values = df_train['Balance'].nlargest(10)
print(top_balance_values)


plt.figure(figsize=(14, 8))
sns.barplot(x=top_balance_values.index, y=top_balance_values, palette="RdBu")
plt.title('Top 10 Balance')
plt.show()


age = df_train['Age'].nlargest(10)
print(age)



plt.figure(figsize=(14, 8))
sns.barplot(x=age.index, y=age, palette="RdBu")
plt.title('Top 10 Age')
plt.show()


top_salaries_by_geography = df_train.groupby('Geography')['EstimatedSalary'].nlargest(5)
print(top_salaries_by_geography)



bal = df_train.groupby('Geography')['Balance'].nlargest(5)
print(bal)

import plotly.express as px

# Reset the index of the DataFrame to make 'Geography' a regular column
bal = bal.reset_index()

# Plotting the top balances by geography
fig = px.bar(bal, x='Geography', y='Balance', color=bal.index, title='Top 5 Highest Balances by Geography', 
             labels={'Balance': 'Balance', 'Geography': 'Geography', 'index': 'Index'})
fig.update_layout(height=800)
fig.show()



ex = df_train['Exited'].value_counts()
print(ex)


# Data for the pie chart
labels = ["Exited: Yes", "Exited: No"]
values = ex.values
colors = ['#ff6666', '#66b3ff']
explode = [0.1, 0]

# Create pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo='label+percent', 
                             marker=dict(colors=colors, line=dict(color='#000000', width=2)),
                             hoverinfo='label+percent', pull=[0.1, 0])])

# Update layout
fig.update_layout(title='Exited Distribution',
                  showlegend=False,
                  annotations=[dict(text='Exited', x=0.5, y=0.5, font_size=25, showarrow=False)],
                  width=800,
                  height=800
                  )

# Show plot
fig.show()



geo_exi = df_train.groupby('Geography')['Exited'].value_counts()
print(geo_exi)


df_train.head()


num_cols


num_cols = []
for i in df_train.columns:
    if df_train[i].dtype == 'int64' or df_train[i].dtype == 'float64':
        num_cols.append(i)
        
num_cols.remove('id')
num_cols.remove('Exited')

fig, ax = plt .subplots(3, 3, figsize=(15, 12))
ax = ax.flatten()
for i, j in enumerate(num_cols):
    sns.kdeplot(df_train[j], ax=ax[i], palette='purp', color='blue')
    ax[i].set_title(f'{i} Distribution', size=14)
    ax[i].set_xlabel(None)
    
plt.suptitle('Distribution of Features', fontsize=24, fontweight='bold')
fig.legend(['Train'])
plt.tight_layout()


num_cols = []
for i in df_train.columns:
    if df_train[i].dtype == 'int64' or df_train[i].dtype == 'float64':
        num_cols.append(i)
        
num_cols.remove('id')
num_cols.remove('Exited')

fig, ax = plt .subplots(3, 3, figsize=(15, 12))
ax = ax.flatten()
for i, j in enumerate(num_cols):
    sns.boxplot(df_train[j], ax=ax[i], color='blue')
    ax[i].set_title(f'{i} Distribution', size=14)
    ax[i].set_xlabel(None)
    
plt.suptitle('Distribution of Features', fontsize=24, fontweight='bold')
fig.legend(['Train'])
plt.tight_layout()


Q1 = df_train['Age'].quantile(0.25)
Q3 = df_train['Age'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1- 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train = df_train[(df_train['Age'] > lower_bound) & (df_train['Age'] < upper_bound)]


test_df = df_test.drop(['id', 'CustomerId', 'Surname'], axis=1)
train_df = df_train.drop(['id', 'CustomerId', 'Surname'], axis=1)
X = train_df.drop(['Exited'], axis=1)
y = train_df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = {
    'XGBClassifier': {
        'model': XGBClassifier(),  # XGBClassifier model instance
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),  # RandomForestClassifier model instance
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),  # GradientBoostingClassifier model instance
    },
    'CatBoostClassifier': {
        'model': CatBoostClassifier(),  # CatBoostClassifier model instance
    }
}


tr1 = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1, 2])
], remainder='passthrough')

tr2 = ColumnTransformer([
    ('quantile_trasformer', QuantileTransformer(output_distribution='normal'), [1, 3, 7, 8])
])


# Initialize lists to store model names and evaluation metrics
model_names = []
accuracy_scores = []
precision_scores = []
recall_scores = []

for model_name, mp in model.items():
    # Create a pipeline for the current model
    pipe = Pipeline([
        ('tr1', tr1),  # Apply the first ColumnTransformer for data preprocessing
        ('tr2', tr2),  # Apply the second ColumnTransformer for data preprocessing
        ('model', mp['model'])  # Add the model to the pipeline
    ])
    
    # Fit the pipeline on the training data
    pipe.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = pipe.predict(X_test)
    
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Append model name and evaluation metrics to respective lists
    model_names.append(model_name)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)


# create a dataframe
evaluation_df = pd.DataFrame({
    'Model': model_name,
    'accuracy_scores': accuracy_scores,
    'precision_scores': precision_scores,
    'recall_scores': recall_scores

})
evaluation_df


catagorical_columns = ['Geography', 'Gender']
numerical_columns = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']

processor = ColumnTransformer([
    ('OHE', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), catagorical_columns),
    ('STD_SCL', StandardScaler(), numerical_columns)
], remainder='passthrough')

X_train_scaled_dl = processor.fit_transform(X_train)
X_test_scaled_dl = processor.transform(X_test)


%%time
from gc import callbacks
from tabnanny import verbose


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled_dl.shape[1],)),  # Input layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    tf.keras.layers.Dense(16, activation='relu'),  # Hidden layer with 16 neurons and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron (for regression) and linear activation
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10
)

history = model.fit(
    X_train_scaled_dl,
    y_train,
    epochs=100,
    callbacks=[early_stopping],
    batch_size=32,
    verbose=2,
    validation_data=[X_test_scaled_dl, y_test]
)


# Predict "Exited" values for the test data
test_data_processed = processor.transform(test_df)


# test_predictions = model.predict(test_data_processed)

# # Add the predicted "Exited" values to the submission DataFrame
# df_submission['Exited'] = test_predictions.flatten()

# # Save the updated submission DataFrame to a CSV file
# df_submission.to_csv('submission2nd.csv', index=False)

