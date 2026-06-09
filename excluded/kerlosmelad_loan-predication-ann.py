import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data=pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
print(train_data.head())

print('-----------------------------------------------')
print(test_data.head()) # like train data in features but without target column




train_data.info()


train_data.describe().T


test_data.describe().T


train_data.isnull().sum()


test_data.isnull().sum()


train_data=train_data.drop('id',axis=1)
test_data=test_data.drop('id',axis=1)


print(train_data.shape)
print(test_data.shape)


train_data.columns


sns.countplot(data=train_data,x='loan_status')


plt.figure(figsize=(10,8))
scatter = plt.scatter(train_data['person_income'], train_data['loan_amnt'], 
                      c=train_data['loan_status'], cmap='viridis',alpha=0.7)  # Add colormap

plt.xlabel('Person Income')
plt.ylabel('Loan Amount')
plt.title('Relation between Person Income and Loan Amount')

# Create a color legend
plt.colorbar(label="Loan Status")

plt.show()


sns.kdeplot(data=train_data,x='person_emp_length',hue='loan_status')


plt.figure(figsize=(12, 8))


plt.subplot(2, 1, 1)
sns.countplot(data=train_data, x='person_home_ownership', hue='loan_status')
plt.title("Loan Status by Home Ownership")
plt.legend()

# Second subplot: loan_intent vs loan_status
plt.subplot(2, 1, 2)
sns.countplot(data=train_data, x='loan_intent', hue='loan_status')
plt.title("Loan Status by Loan Intent")
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


train_data.groupby('cb_person_default_on_file').sum().sort_values('loan_int_rate',ascending=False)['loan_int_rate']


plt.figure(figsize=(10, 6))
avg_interest_rate = train_data.groupby('loan_grade')['loan_int_rate'].mean().reset_index()

# Sort the values for better visualization
avg_interest_rate = avg_interest_rate.sort_values(by='loan_int_rate', ascending=True)

avg_interest_rate


plt.figure(figsize=(18,10))
sns.boxplot(data=train_data,y='person_income',hue='loan_status')
plt.show()


sns.displot(data=train_data, 
            x='cb_person_cred_hist_length', 
            hue='loan_status', 
            kde=True, 
            multiple="stack",  # Use 'stack' or 'dodge' for clarity
            aspect=1.5)  # Adjust width

plt.xlabel("Credit History Length")
plt.title("Distribution of Credit History Length by Loan Status")
plt.show()



plt.figure(figsize=(8, 6))
sns.boxplot(data=train_data, x='loan_status', y='person_age', hue='loan_status', palette="Set2")

plt.xlabel("Loan Status (0 = Non-Defaulter, 1 = Defaulter)")
plt.ylabel("Applicant Age")
plt.title("Box Plot of Age by Loan Default Status")
plt.show()



train_data


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
x=train_data.drop('loan_status',axis=1).values
y=train_data['loan_status']


over_sampledf=RandomOverSampler(sampling_strategy=0.85)
undersampledf=RandomUnderSampler(sampling_strategy=1.0)
x_over,y_over=over_sampledf.fit_resample(x,y)
x_under,y_under=undersampledf.fit_resample(x,y)


y_over.value_counts()


y_under.value_counts()


import pandas as pd

# If x was originally a DataFrame, preserve column names

y_under=y_under.convert_dtypes()
x_under_df = pd.DataFrame(x_under, columns=test_data.columns)  # Use original column namess
y_under_series = pd.Series(y_under, name='loan_status')
x_under_df=x_under_df.convert_dtypes()
y_under_series=y_under_series.convert_dtypes()
train_under = pd.concat([x_under_df, y_under_series], axis=1)

for col in train_under.columns:
    if train_under[col].dtype == 'string' or train_under[col].dtype == 'object':
        train_under[col] = train_under[col].astype(object) 

plt.figure(figsize=(20, 10))

# Loop through columns
for col in train_under.columns:
    if train_under[col].dtype == 'object':  # Check for categorical columns
        contingency_table = pd.crosstab(train_under[col], train_under['loan_status'], normalize='index')

        plt.figure(figsize=(20, 4))  # Create a new figure for each categorical variable
        sns.set(style="whitegrid")
        contingency_table.plot(kind="bar", stacked=True)
        
        plt.title(f"Percentage Distribution of Target across {col}")
        plt.xlabel(col)
        plt.ylabel("Percentage")
        plt.legend(title="Target Class")

plt.show()



plt.figure(figsize=(20, 10))

for col in train_under.columns:
    if train_under[col].dtype == 'Float64': 
        contingency_table = pd.crosstab(train_under[col], train_under['loan_status'], normalize='index')

        plt.figure(figsize=(20, 4))
        sns.set(style="whitegrid")
        contingency_table.plot(kind="kde", stacked=True)
        
        plt.title(f"Percentage Distribution of Target across {col}")
        plt.xlabel(col)
        plt.ylabel("Percentage")
        plt.legend(title="Target Class")

plt.show()




plt.figure(figsize=(20, 10))

for col in train_under.columns:
    if train_under[col].dtype == 'Int64':  
        contingency_table = pd.crosstab(train_under[col], train_under['loan_status'], normalize='index')

        plt.figure(figsize=(20, 4))
        sns.set(style="whitegrid")
        contingency_table.plot(kind="hist", stacked=True)
        
        plt.title(f"Percentage Distribution of Target across {col}")
        plt.xlabel(col)
        plt.ylabel("Percentage")
        plt.legend(title="Target Class")

plt.show()


train_under


train_under.columns


import numpy as np
import pandas as pd

# تحديد الأعمدة العددية فقط
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 
                'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                'cb_person_cred_hist_length']


Q1 = train_under[numeric_cols].quantile(0.25)
Q3 = train_under[numeric_cols].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


train_under_cleaned = train_under[~((train_under[numeric_cols] < lower_bound) | (train_under[numeric_cols] > upper_bound)).any(axis=1)]




def feature_engineering(df_):  
    df = df_.copy()
    df['loantoincome'] = (df['loan_amnt'] / df['person_income']) - df['loan_percent_income']
    df['person_income'] = np.log(df['person_income'])
    return df

train, test = feature_engineering(train_under_cleaned), feature_engineering(test_data)


train


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2,l1


# Assuming train_data and test_data are already loaded
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Label Encoding for categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])  # Fit on training data
    test_data[col] = le.transform(test_data[col])  # Transform test data using the same encoder

# Features and target
X = train_data.drop(['loan_status'], axis=1)  # Use train_data instead of train
y = train_data['loan_status']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_val_scaled = scaler.transform(X_val)  # Only transform validation data
test_data_scaled = scaler.transform(test_data)  # Only transform test data

# Define the ANN model
input_dim = X_train_scaled.shape[1]  # Dynamically set input dimension based on data
model = Sequential()
model.add(Dense(128, input_dim=input_dim, activation='relu'))  # Use input_dim from data
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=26, validation_data=(X_val_scaled, y_val))

# Evaluate on the validation set
val_loss, val_auc = model.evaluate(X_val_scaled, y_val)
print(f'Validation Loss: {val_loss}, Validation AUC: {val_auc}')

# Predict on test data (if needed)
# predictions = model.predict(test_data_scaled)


X_train_scaled.shape


model.evaluate(X_train_scaled,y_train)


model.evaluate(X_val_scaled,y_val)


predict=model.predict(X_val_scaled)
predict = (predict > 0.5).astype(int)  
predict


from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold , cross_val_score
con=confusion_matrix(y_val,predict)
plot_confusion_matrix(con)
print(classification_report(y_val, predict))


predication=model.predict(test_data_scaled)
predication
predication = (predication > 0.5).astype(int)  # Convert to 0 or 1



submision_file = pd.DataFrame({
    'id': test_data.index,
    'loan_statue': predication.ravel()  
})

submision_file.to_csv("submission.csv", index=False)
submision_file


