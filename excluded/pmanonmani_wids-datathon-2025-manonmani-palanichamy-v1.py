# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
train_quant=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
train_cate=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_sol=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")


train_cate.info()


train_quant.info()


train_cat_quant = pd.merge(train_cate, train_quant, on='participant_id', how='inner')


test_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_cate=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_quant=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")


test_cate.info()


test_quant.info()


test_cat_quant = pd.merge(test_cate, test_quant, on='participant_id', how='inner')


test_cat_quant['Dataset'] = 'Test'
train_cat_quant['Dataset'] = 'Train'


combined_cat = pd.concat([test_cat_quant, train_cat_quant], ignore_index=True)

combined_cat.head()


test_cat_quant['Dataset'] = 'Test'
train_cat_quant['Dataset'] = 'Train'


combined_cat = pd.concat([test_cat_quant, train_cat_quant], ignore_index=True)
combined_cat.head()


combined_cat.info()


combined_cat_drop= combined_cat.drop(columns=['MRI_Track_Scan_Location','SDQ_SDQ_Externalizing','MRI_Track_Age_at_Scan','Basic_Demos_Enroll_Year','SDQ_SDQ_Hyperactivity','APQ_P_APQ_P_PP'])


combined_cat_drop.info()


combined_cat2 = combined_cat_drop.copy()


combined_cat2 = combined_cat2.fillna(combined_cat.median(numeric_only=True))


combined_cat2.info()


feature_name = 'Dataset'  # Example feature to split on
value_to_split = 'Test'  # Value to filter on



df_split_test = combined_cat2[combined_cat2[feature_name] == value_to_split]
df_split_train = combined_cat2[combined_cat2[feature_name] != value_to_split]


df_split_train.info()


train_solution=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
train_solution = pd.merge(df_split_train, train_solution, on='participant_id', how='inner')
train_solution.info()


train_solution = train_solution.drop(columns=['Dataset'])


train_solution.info()


merged_data_graph = train_solution.drop(columns=['participant_id','EHQ_EHQ_Total'])


import matplotlib.pyplot as plt

# Function to create frequency plots for each column
def create_frequency_plots(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 6))
        if df[column].dtype in ['int64', 'float64']:  # Numerical data
            df[column].value_counts().sort_index().plot(kind='bar')
            plt.title(f"Frequency Plot of {column} (Numerical)")
            plt.xlabel(column)
            plt.ylabel("Frequency")
        else:  # Categorical or object data
            df[column].value_counts().plot(kind='bar')
            plt.title(f"Frequency Plot of {column} (Categorical)")
            plt.xlabel(column)
            plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


create_frequency_plots(merged_data_graph, merged_data_graph.columns)


adhd_sex_counts = merged_data_graph.groupby(['Sex_F', 'ADHD_Outcome']).size().unstack()


ax = adhd_sex_counts.plot(kind='bar', figsize=(8, 6), rot=0)
plt.xlabel("Sex (0 = Male, 1 = Female)")
plt.ylabel("Count")
plt.title("ADHD Outcome by Sex")
plt.legend(title="ADHD Outcome")
plt.show()



plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

#drop participant_id from the data we will evaluate for correlation since it is unique to each person
corr_data_graph = train_solution.drop(columns=['participant_id'])

#Compute the correlation matrix
correlation_matrix = corr_data_graph.corr()

# Filter correlations greater than 0.5 (absolute value) 
# Change the correlation coefficient threshold to explore the data more, try 0.6 or 0.7
#Remember the closer to 1 means two variables are highly correlated
strong_correlations = correlation_matrix[abs(correlation_matrix) > 0.5].dropna(how='all', axis=0).dropna(how='all', axis=1)

# Display the filtered correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(strong_correlations, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Strong Correlations (> 0.5) Heatmap")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix including 'Sex_F'
correlation_matrix = corr_data_graph.corr()

# Extract correlations of features with respect to 'Sex_F'
correlation_with_sex = correlation_matrix['Sex_F'].drop('Sex_F').sort_values(ascending=False)

# Plot the correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_sex.index, y=correlation_with_sex.values, palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Correlation with Sex_F")
plt.title("Feature Correlations with Sex (Sex_F)")
plt.show()

# Display numerical correlation values
correlation_with_sex



from sklearn.feature_selection import mutual_info_regression

X = corr_data_graph.copy()
y = X.pop('Sex_F')

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes 
discrete_features = X.dtypes == int

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores


from sklearn.feature_selection import mutual_info_regression

X = corr_data_graph.copy()
y = X.pop('Sex_F')

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes 
discrete_features = X.dtypes == int

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores


train_connectome.info()



# Identify features with nulls and their null counts
null_counts = train_connectome.isnull().sum()

# Filter features with at least one null
features_with_nulls = null_counts[null_counts > 0]

features_with_nulls 


# Identify features with nulls and their null counts
null_counts2 = test_connectome.isnull().sum()

# Filter features with at least one null
features_with_nulls2 = null_counts2[null_counts2 > 0]

features_with_nulls2


# Append connectnomes to train_solution
train_conn_solution = pd.merge(train_solution,train_connectome, on='participant_id', how='inner')
train_conn_solution.info()


# Append connectnomes to df_split_test
test_conn = pd.merge(df_split_test,test_connectome, on='participant_id', how='inner')
test_conn.info()


# Drop Dataset feature from test_conn
test_conn = test_conn.drop(columns=['Dataset'])


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Drop 'participant_id' from train and test data
X_train = train_conn_solution.drop(columns=['participant_id', 'ADHD_Outcome', 'Sex_F'])
y_train = train_conn_solution[['ADHD_Outcome', 'Sex_F']]
X_test = test_conn.drop(columns=['participant_id'])

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
# Try different versions of the model by changing activation or number of layers
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['binary_accuracy'])

# Train the model
model_hist=model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions on test data
predictions = model.predict(X_test_scaled)
train_predictions = model.predict(X_train_scaled)


# Convert probabilities to binary outcomes
predicted_ADHD = (predictions[:, 0] > 0.5).astype(int)
predicted_Sex_F = (predictions[:, 1] > 0.5).astype(int)

train_predicted_ADHD = (train_predictions[:, 0] > 0.5).astype(int)
train_predicted_Sex_F = (train_predictions[:, 1] > 0.5).astype(int)


# Create a DataFrame with participant_id and predictions
test_predictions = test_conn[['participant_id']].copy()
test_predictions['ADHD_Outcome'] = predicted_ADHD
test_predictions['Sex_F'] = predicted_Sex_F

# Save results to CSV
test_predictions.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")


#Check Output 
test_predictions


#Check learning curves

history_df = pd.DataFrame(model_hist.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))


# Check F1-Score of Train Data
# Create ADHD_Outcome and Sex_f as singluar datasets to evaluate the fit of each prediction
actual_adhd = train_conn_solution[['ADHD_Outcome']]
actual_sex = train_conn_solution[['Sex_F']]


from sklearn.metrics import f1_score

# Calculate F1 Score for adhd
adhd_f1 = f1_score(actual_adhd, train_predicted_ADHD)

print("F1 Score:", adhd_f1)

