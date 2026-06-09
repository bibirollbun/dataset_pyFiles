from IPython.display import Image, display
display(Image("/kaggle/input/bank-churn-case-study-main-picture/Churn-Prediction-scaled.jpg"))


# Import the data Libraries
import pandas as pd
import numpy as np

## Import Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Import Warnings for removing useless warnings
import warnings
warnings.filterwarnings("ignore")


# ML Libraries

## Preprocessing libraries
from sklearn.preprocessing import StandardScaler,LabelEncoder,PowerTransformer

# Spliting data Library
from sklearn.model_selection import train_test_split,GridSearchCV

# Model For Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Model For Prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Import the tensorflow Library
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Import the matrix
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,roc_auc_score,roc_curve


df_train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndatasetumgc/train.csv')
df_test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndatasetumgc/test.csv')
df_submission = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndatasetumgc/sample_submission.csv')


df = df_train.copy() # We have create copy of the df_train because we want the variable name to df instead to df_train
df.head()


# Checking the shape of the data
df.shape
print(f"This dataset has {df.shape[0]} rows with {df.shape[1]} columns")


# Checking the information about the datatypes of each columns
print(f"This dataset has 1 int column , 3 object columns , 10 float column with overall 14 columns")


# Checking the statistical Information about the dataset
df.describe().T


# Checking the Columns and print 
for col in df_train.columns:
    print(col)


# Creating Variables with each datatype column
numeric_cols = df[['CreditScore','Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember', 'EstimatedSalary', 'Exited']]
cat_cols = df[['Surname', 'CreditScore', 'Geography', 'Gender']]


((df.isnull().sum())/len(df))*100


sns.heatmap(numeric_cols,linewidths=0)


df.drop(['id', 'CustomerId', 'Surname'],axis=1,inplace=True)


# First Create the professional theme for dataset

greenish_grey_palette = ["#6B8E23",  # olive green
                         "#4B5563",  # dark grey
                         "#9CA3AF",  # light grey
                         "#2F4F4F",  # dark slate grey (greenish tint)
                         "#A8BFA1"]  # soft pastel gree

# Creating box plot based on the numeric columns
plt.figure(figsize=(12, 16))
n_cols = 2
n_rows = (len(numeric_cols.columns) + n_cols - 1) // n_cols  # ensures enough rows
for idx, col in enumerate(numeric_cols.columns):
    plt.subplot(n_rows, n_cols, idx+1)
    sns.boxplot(x=df_train[col], palette=greenish_grey_palette)
    plt.title(col)
plt.tight_layout()
plt.show()


df = df[~(df['EstimatedSalary'] > 200000) & (df['EstimatedSalary'] <= 1500000)]


sns.boxplot(df,x='EstimatedSalary',color="#6B8E23")


df.duplicated().any()


# Lets Create the professional theme for dataset

greenish_grey_palette = ["#6B8E23",  # olive green
                         "#4B5563",  # dark grey
                         "#9CA3AF",  # light grey
                         "#2F4F4F",  # dark slate grey (greenish tint)
                         "#A8BFA1"]  # soft pastel gree



df.groupby(['Geography','Gender'])['CreditScore'].mean()


df['Geography'].unique()
print("Geography or Location column has 3 unique Entities like Spain France Germany")


# Let's create a Pie plot that shows the distribution of the Gender column
figure = plt.figure(figsize=(8,5))
counts = df['Geography'].value_counts()
plt.title("Percentage of People based on Geograply in the dataset (Graph 1.1)")
labels = ["France", "Spain","Germany"]
# Plot pie char2
plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=greenish_grey_palette, startangle=90)
plt.show()



plt.figure(figsize=(8,6))
sns.countplot(df,y='Geography',hue='Gender' , palette=greenish_grey_palette)
plt.title("Location or Geography of Gender in the dataset (Graph 1.2)")
plt.xlabel("Count of Gender W.R.T Geography")
plt.ylabel("Area W.R.T Geography")
plt.show()



# Let's create a Pie plot that shows the distribution of the Gender column
figure = plt.figure(figsize=(8,5))
counts = df['Gender'].value_counts()
plt.title("Percentage of Gender in the dataset (Graph 2.1)")
labels = ["Male", "Female"]
# Plot pie char2
plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=greenish_grey_palette, startangle=90)
plt.show()


df['Age'].mean()


sns.histplot(df,x='Age',kde=True,color = '#6B8E23')
plt.title("Average Age Bracket Check (Graph 3.1)")
plt.xlabel("Age Of Customer")
plt.ylabel("Count of Customer")
plt.show()


df.groupby(['Geography','Gender'])['Tenure'].max()


sns.countplot(df,x='Tenure',hue='Gender',palette=greenish_grey_palette)
plt.title("Average Tenure Based On Gender (Graph 4.1)")
plt.xlabel("Tenure Period Of Customer")
plt.ylabel("Count of Tenure Period")
plt.show()


df.groupby('Gender')['Balance'].max()


df['Balance'].describe()


sns.countplot(df,x='NumOfProducts',hue='Geography',palette=greenish_grey_palette)
plt.title("Average Products Of The Bank Used By User Based On Gender (Graph 5.1)")
plt.xlabel("Average Products Of The Bank")
plt.ylabel("Count of Products Of The Bank")
plt.show()


sns.countplot(df,x='NumOfProducts',hue='Geography',palette=greenish_grey_palette)
plt.title("Average Products Of The Bank Used By User Based On Location (Graph 5.2)")
plt.xlabel("Average Products Of The Bank Based On Location")
plt.ylabel("Count of Products Of The Bank Based On Location")
plt.show()



# Let's create a Pie plot that shows the distribution of the Credit Card column
figure = plt.figure(figsize=(8,5))
counts = df['NumOfProducts'].value_counts()
plt.title("Percentage of Products Sold (Graph 5.3)")
labels = [f"Product{i}" for i in counts.index]
# Plot pie char2
plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=greenish_grey_palette, startangle=90)
plt.show()


sns.countplot(df,x='HasCrCard',hue='Geography',palette=greenish_grey_palette)
plt.title("Average Person Have Credit Card Based On Location (Graph 6.1)")
plt.xlabel("Average Credit Card Person Based On Location")
plt.ylabel("Count of Credit Card Person Based On Location")
plt.show()



# Let's create a Pie plot that shows the distribution of the Credit Card column
figure = plt.figure(figsize=(8,5))
counts = df['HasCrCard'].value_counts()
plt.title("Percentage of Credit Card (Graph 6.2)")
labels = ["Has_Credit_Card", "Not_Have_Credit_Card"]
# Plot pie char2
plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=greenish_grey_palette, startangle=90)
plt.show()


df['IsActiveMember'].value_counts()
df.groupby(['Gender','Geography'])['IsActiveMember'].value_counts()



# Let's create a Pie plot that shows the distribution of the Credit Card column
figure = plt.figure(figsize=(8,5))
counts = df['IsActiveMember'].value_counts()
plt.title("Percentage of Active and Non Active Members (Graph 7.1)")
labels = ["Active", "Non-Active"]
# Plot pie char2
plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=greenish_grey_palette, startangle=90)
plt.show()


df['EstimatedSalary'].describe()


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10,8))
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df[col], kde=True, bins=30,color="#6B8E23")
    plt.title(col, fontsize=12)
plt.tight_layout()
plt.show()


transformer = PowerTransformer(method='box-cox')
df['Box_Cox_Transform_CreditScore'] = transformer.fit_transform(df[['CreditScore']])


transformer = PowerTransformer(method='yeo-johnson')
df['Yeo_Transform_Age'] = transformer.fit_transform(df[['Age']])


scaler = StandardScaler()
cols = df[['Box_Cox_Transform_CreditScore','Yeo_Transform_Age','Tenure','Balance','NumOfProducts','EstimatedSalary']];
for col in cols:
    df[f'{col}_standardized'] = scaler.fit_transform(df[[col]])
# This Make the Data Scaled between 0 to 1.


df.drop(['CreditScore','Age','Box_Cox_Transform_CreditScore','Yeo_Transform_Age','Tenure','Balance','NumOfProducts','EstimatedSalary'],axis=1,inplace=True)


encoders = {}
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'category':
        df[col] = label_encoder.fit_transform(df[col])
        encoders = label_encoder;


# First Getting the required data for deep learning
X = df.drop('Exited',axis=1)
y = df['Exited'] # This is the targeted variable

# Spliting the data into train and test data 
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Creating the deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,activation='relu',input_shape=(x_train.shape[1],)), # Input Layer
    tf.keras.layers.Dense(32,activation='relu'), # Hidden layer,
    tf.keras.layers.Dropout(0.2), # Hidden layer,
    tf.keras.layers.Dense(16,activation='relu'), # Hidden layer
    tf.keras.layers.Dense(1,activation='sigmoid') # Output Layer
])

# Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Declare Early Stopping
early = EarlyStopping(patience=17)
# Train the model

history = model.fit(x_train,y_train,verbose=1,epochs=22,batch_size=64,validation_data=(x_test,y_test),callbacks=[early])

# Evaluating the model
loss = model.evaluate(x_test,y_test,verbose=0)
print(f"Binary_Cross_Entropy: {loss[0]}")

y_prob = model.predict(x_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 3. Calculate AUC score
auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)

# Plot the roc curve 

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Ploting the training and testing loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Validation'],loc='upper right')
plt.show()

y_pred = model.predict(x_test)
y_pred = (y_pred >= 0.28346297).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 4. Optional: Visualize it
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


J = tpr - fpr
ix = J.argmax()
best_threshold = thresholds[ix]

print("Best Threshold for Balance (Youden J):", best_threshold)


X = df.drop('Exited',axis=1)
y = df['Exited']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000),
    'SVC': SVC(probability=True),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

param_grids = {
    'Logistic_Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
    },
    'SVC': {
        'C': [0.1, 1, 10],
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'XGB': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}
best_models = []
best_estimators = {}
for name, model in models.items():
    print(f"ğŸ”� Tuning Hyperparameters for {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(x_train, y_train)

    best_estimator = grid.best_estimator_
    best_estimators[name] = best_estimator

    y_prob = best_estimator.predict(x_test)
    acc_score = accuracy_score(y_test, y_prob)
    prec_score = precision_score(y_test, y_prob)

    best_models.append((name, acc_score, prec_score, grid.best_params_))

# Sort models based on accuracy
best_models_sorted = sorted(best_models, key=lambda x: x[1], reverse=True)

# Display results
for model in best_models_sorted:
    print(f"âœ… {model[0]} â†’ Accuracy: {model[1]:.2f} | Precision: {model[2]:.2f} | Best Params: {model[3]}")

# set final_model to the fitted estimator (not a tuple)
best_name = best_models_sorted[0][0]
final_model = best_estimators[best_name]


cm = confusion_matrix(y_test, y_prob)
print("Confusion Matrix:")
print(cm)

# 4. Optional: Visualize it
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap = 'Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


encoders = {}
label_encoder = LabelEncoder()
for col in df_test.columns:
    if df_test[col].dtype == 'object' or df_test[col].dtype == 'category':
        df_test[col] = label_encoder.fit_transform(df_test[col])
        encoders = label_encoder;


transformer = PowerTransformer(method='box-cox')
df_test['Box_Cox_Transform_CreditScore'] = transformer.fit_transform(df_test[['CreditScore']])
transformer = PowerTransformer(method='yeo-johnson')
df_test['Yeo_Transform_Age'] = transformer.fit_transform(df_test[['Age']])


df_test.columns


scaler = StandardScaler()
cols = df_test[['Box_Cox_Transform_CreditScore','Yeo_Transform_Age','Tenure','Balance','NumOfProducts','EstimatedSalary']];
for col in cols:
    df_test[f'{col}_standardized'] = scaler.fit_transform(df_test[[col]])
# This Make the Data Scaled between 0 to 1.


cols_to_drop = ['id', 'CustomerId', 'Surname','Box_Cox_Transform_CreditScore','Yeo_Transform_Age','CreditScore','Age', 'Tenure', 'Balance', 'NumOfProducts','EstimatedSalary']
for cols in cols_to_drop:
    df_test.drop(cols,axis=1,inplace=True)



# prediction = final_model.predict(df_test)
# threshold = 0.5
# y_pred = (prediction > threshold).astype(int)
# df_submission.iloc[:, 1] = y_pred 
# df_submission.to_csv('/kaggle/input/binaryclassificationwithabankchurndatasetumgc/sample_submission.csv',index=False)

