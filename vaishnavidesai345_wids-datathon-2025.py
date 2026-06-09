import pandas as pd

# Load training data
train_targets = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
train_categorical = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_quantitative = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
train_fmri = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")

# Load test data
test_categorical = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_quantitative = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_fmri = pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")

# Check data shapes
print(train_targets.shape, train_categorical.shape, train_quantitative.shape, train_fmri.shape)



print(train_targets.info(), train_categorical.info(), train_quantitative.info(), train_fmri.info())




train_targets.head()




train_categorical.head()
 


train_quantitative.head()



train_fmri.iloc[:, :10].head()


import seaborn as sns
import matplotlib.pyplot as plt


categorical_columns = ["Basic_Demos_Enroll_Year", "Basic_Demos_Study_Site", 
                       "PreInt_Demos_Fam_Child_Ethnicity", "PreInt_Demos_Fam_Child_Race", 
                       "MRI_Track_Scan_Location","Barratt_Barratt_P1_Edu", "Barratt_Barratt_P1_Occ", "Barratt_Barratt_P2_Edu","Barratt_Barratt_P2_Occ"]

num_cols = 3  # Number of columns per row
num_rows = (len(categorical_columns) + num_cols - 1) // num_cols  # Calculate needed rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

# Flatten axes array for easy iteration (handles cases where number of plots < total grid size)
axes = axes.flatten()

# Plot each categorical column
for i, col in enumerate(categorical_columns):
    sns.countplot(x=col, data=train_categorical, palette="coolwarm", ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].set_title(f"Distribution of {col}")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])  # Remove empty plots

plt.tight_layout()
plt.show()


import warnings

# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# Identifying quantitative columns (excluding participant_id)
quantitative_columns = [col for col in  train_quantitative.columns if col != "participant_id"]

# Set up the subplot grid (rows=ceil(len(columns)/3), cols=3)
num_cols = 3  # Number of columns per row
num_rows = (len(quantitative_columns) + num_cols - 1) // num_cols  # Calculate needed rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

# Flatten axes array for easy iteration
axes = axes.flatten()

# Plot each quantitative column
for i, col in enumerate(quantitative_columns):
    sns.histplot( train_quantitative[col], bins=10, kde=True, ax=axes[i], color="steelblue")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")
    axes[i].set_title(f"Distribution of {col}")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])  # Remove empty plots

plt.tight_layout()
plt.show()


train_targets['ADHD_Outcome'].value_counts()



train_targets['ADHD_Outcome'].value_counts().plot(kind='bar', color='purple')
plt.title('ADHD Outcome')
plt.xlabel('Outcome (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


train_targets['Sex_F'].value_counts()


train_targets['Sex_F'].value_counts().plot(kind='bar', color='purple')
plt.title('Gender Distribution')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(train_quantitative['SDQ_SDQ_Emotional_Problems'], kde=True, color='skyblue')
plt.title('Distribution of SDQ_SDQ_Emotional_Problems')
plt.xlabel('SDQ_SDQ_Emotional_Problems')
plt.ylabel('Frequency')
plt.show()


# Check for correlation with ADHD outcome
train_Quant_copy = train_quantitative.copy()
train_Quant_copy['ADHD_Outcome'] = train_targets['ADHD_Outcome']

plt.figure(figsize=(8, 6))
sns.boxplot(x='ADHD_Outcome', y='SDQ_SDQ_Emotional_Problems', data=train_Quant_copy)
plt.title('SDQ_SDQ_Emotional_Problems vs ADHD Outcome')
plt.xlabel('ADHD Outcome')
plt.ylabel('SDQ_SDQ_Emotional_Problems')
plt.show()





sns.countplot(data=train_categorical, x='Barratt_Barratt_P1_Edu', hue=train_targets['ADHD_Outcome'])
plt.title('ADHD Prevalence by Parent 1 Education')
plt.show()


train_categorical['Barratt_Barratt_P1_Edu'].value_counts()


# Add ADHD_Outcome directly to a copy of the train_cat dataset for grouping
train_cat_copy = train_categorical.copy()
train_cat_copy['ADHD_Outcome'] = train_targets['ADHD_Outcome']

adhd_percentages = train_cat_copy.groupby('Barratt_Barratt_P1_Edu')['ADHD_Outcome'].mean()
print(adhd_percentages)


train_categorical['Barratt_Barratt_P1_Edu'].value_counts()


for col in train_categorical.select_dtypes(include='int').columns:
    train_categorical[col] = train_categorical[col].astype('category')


# Creating a list of all of the columns except the first
columns_to_encode = train_categorical.columns[1:].tolist()

# Print the columns to encode
print("Columns to encode:", columns_to_encode)



train_categorical_encoded = pd.get_dummies(train_categorical[columns_to_encode], drop_first=True)
train_categorical_encoded = train_categorical_encoded.applymap(lambda x: 1 if x is True else (0 if x is False else x))


# Combine encoded columns with the rest of the DataFrame
cat_train_final = pd.concat([train_categorical.drop(columns=columns_to_encode), train_categorical_encoded], axis=1)

# ensure it looks correct
cat_train_final.head()


# load in test categorical dataframe

file_path_test = "/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx"
test_cat = pd.read_excel(file_path_test)
(test_cat.head())


# convert our int variables to categories
for col in test_cat.select_dtypes(include='int').columns:
    test_cat[col] = test_cat[col].astype('category')

# Encode categorical variables in test
test_encoded = pd.get_dummies(test_cat[columns_to_encode], drop_first=True)
test_encoded = test_encoded.applymap(lambda x: 1 if x is True else (0 if x is False else x))

# Ensure test_encoded has the same columns as train_encoded
missing_cols = set(train_categorical_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0  # Add missing columns with 0 values

# Ensure test_encoded columns are in the same order as train_encoded
test_encoded = test_encoded.reindex(columns=train_categorical_encoded.columns, fill_value=0)

# Combine encoded columns with the rest of the DataFrame
cat_test_final = pd.concat([test_cat.drop(columns=columns_to_encode), test_encoded], axis=1)

cat_test_final.head()


train_cat_FCM = pd.merge(cat_train_final, train_fmri, on = 'participant_id')


train_df = pd.merge(train_cat_FCM, train_quantitative, on = 'participant_id')

# ensure it looks accurate
train_df.head()


file_path_test_FCM = "/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_FCM = pd.read_csv(file_path_test_FCM)
#print(train_FCM.head())



file_path_testQ = "/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"
test_Quant = pd.read_excel(file_path_testQ)
#print(train_Quant.head())

test_cat_FCM = pd.merge(cat_test_final, test_FCM, on = 'participant_id')

test_df = pd.merge(test_cat_FCM, test_Quant, on = 'participant_id')


# ensure it looks accurate
test_df.head()


train_df.shape


print(train_df.isna().sum())


print(train_df.isnull().sum().sum())


train_df.fillna(train_df.select_dtypes(include=['number']).mean(), inplace=True)


train_df.fillna(train_df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)


print(train_df.isnull().sum().sum())


'''train_df.fillna({'MRI_Track_Age_at_Scan':train_df['MRI_Track_Age_at_Scan'].mean()}, inplace = True)
train_df.fillna({'PreInt_Demos_Fam_Child_Ethnicity':train_df['PreInt_Demos_Fam_Child_Ethnicity'].mean()}, inplace = True)

print(train_df.isna().sum().sum()) '''


train_df.ffill(inplace=True)
print(train_df.isna().sum().sum())


for col in test_df.columns:
    if test_df[col].isna().sum() > 0:  # Check if the column has NaN values
        if test_df[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            test_df[col] = test_df[col].fillna(test_df[col].mean())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")


file_path_trainS = "/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx"
train_Solutions = pd.read_excel(file_path_trainS)


X_train = train_df.drop(columns = ['participant_id'])
Y_train = train_Solutions.drop(columns = ['participant_id'])


from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

# Initialize the base classifier
xgb_classifier = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)


multioutput_classifier = MultiOutputClassifier(xgb_classifier)




multioutput_classifier.fit(X_train, Y_train)


participant_id = test_df['participant_id']

X_test = test_df.drop(columns = 'participant_id')

y_pred = multioutput_classifier.predict(X_test)


predictions_df = pd.DataFrame(
    y_pred,
    columns=['Predicted_Gender', 'Predicted_ADHD']
)

# Combine participant IDs with predictions
result_df = pd.concat([participant_id.reset_index(drop=True), predictions_df], axis=1)

# Print or save the DataFrame
print(result_df)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score


def multi_output_accuracy(y_true, y_pred):
    # Ensure y_true and y_pred are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Compute accuracy for each target variable and return the mean
    return np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])


multi_output_scorer = make_scorer(multi_output_accuracy)


'''import numpy as np
cv_scores = cross_val_score(multioutput_classifier, X_train, Y_train, cv=5, scoring=multi_output_scorer, n_jobs=-1)

# Output the cross-validation results
print("Cross-validation scores for each fold:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))'''



from sklearn.linear_model import LogisticRegression


model = LogisticRegression(max_iter=1000)
model.fit(train_df.drop(columns='participant_id'), train_Solutions['Sex_F'])


coefficients = pd.Series(model.coef_[0], index=train_df.drop(columns='participant_id').columns)


#Select top features for Sex prediction
top_features = coefficients.abs().nlargest(10)
print(top_features)


plt.figure(figsize=(10,6))
top_features.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 10 Features for Sex Outcome')
plt.ylabel('Features')
plt.xlabel('Absolute Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.show()


model = LogisticRegression(max_iter=1000)
model.fit(train_df.drop(columns='participant_id'), train_Solutions['ADHD_Outcome'])


coefficients = pd.Series(model.coef_[0], index=train_df.drop(columns='participant_id').columns)


top_features = coefficients.abs().nlargest(10)
print(top_features)


plt.figure(figsize=(10,6))
top_features.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 10 Features for ADHD Outcome')
plt.ylabel('Features')
plt.xlabel('Absolute Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.show()


model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(train_df.drop(columns='participant_id'), train_Solutions['Sex_F'])


selected_features_sex = train_df.drop(columns='participant_id').columns[model.coef_[0] != 0]
print(selected_features_sex)


model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(train_df.drop(columns='participant_id'), train_Solutions['ADHD_Outcome'])


selected_features_ADHD = train_df.drop(columns='participant_id').columns[model.coef_[0] != 0]
print(selected_features_ADHD)


common_features = list(set(selected_features_ADHD) & set(selected_features_sex))


X_train_2 = X_train[common_features]
X_test_2 = X_test[common_features]


xgb_classifier = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)


multioutput_classifier = MultiOutputClassifier(xgb_classifier)


multioutput_classifier.fit(X_train_2, Y_train)


y_pred_2 = multioutput_classifier.predict(X_test_2)


predictions_df_2 = pd.DataFrame(
    y_pred_2,
    columns=['Predicted_Gender', 'Predicted_ADHD']
)

# Combine participant IDs with predictions
result_df_2 = pd.concat([participant_id.reset_index(drop=True), predictions_df_2], axis=1)

result_df_2.head()


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score


multi_output_scorer = make_scorer(multi_output_accuracy)


import numpy as np
cv_scores_2 = cross_val_score(multioutput_classifier, X_train_2, Y_train, cv=5, scoring=multi_output_scorer)

# Output the cross-validation results
print("Cross-validation scores for each fold:", cv_scores_2)
print("Mean CV score:", np.mean(cv_scores_2))



print(train_targets.columns)



import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer

# Define Multi-Output Accuracy Scorer
def multi_output_accuracy(y_true, y_pred):
    return np.mean((y_pred == y_true).all(axis=1))

multi_output_scorer = make_scorer(multi_output_accuracy)

# Define target variables
Y = train_targets[['ADHD_Outcome', 'Sex_F']]  # Multi-output target

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X_train_2, Y, test_size=0.2, random_state=42)

# Models to train
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42)
}

# Train and Evaluate Models
for model_name, base_model in models.items():
    print(f"\nTraining {model_name}...")

    # Wrap model in MultiOutputClassifier
    multioutput_classifier = MultiOutputClassifier(base_model)
    
    # Train Model
    multioutput_classifier.fit(X_train, Y_train)

    # Predictions
    Y_pred = multioutput_classifier.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(multioutput_classifier, X_train_2, Y, cv=5, scoring=multi_output_scorer)
    print(f"Cross-validation scores for {model_name}: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores):.4f}")


