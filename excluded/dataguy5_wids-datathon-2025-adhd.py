!pip install -q umap-learn


# pandas, numpy, os
import pandas as pd
import numpy as np
import os


# matplotlib, seaborn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


# scipy
import scipy
from scipy.stats import zscore, pearsonr, uniform
from scipy.io import loadmat


# sklearn
import sklearn
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_curve, make_scorer, accuracy_score, r2_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, SelectKBest
from sklearn.linear_model import LogisticRegression,Ridge,RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import ExtraTreesRegressor,RandomForestClassifier
from sklearn.manifold import TSNE


# xgboost

import lightgbm as lgb, xgboost as xgb, catboost as cb
from xgboost import XGBClassifier
from gc import collect


import umap


# load Data dictionary & sample submission

data_dict_df = pd.read_excel('/kaggle/input/widsdatathon2025/Data Dictionary.xlsx')
sample_submission_df = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')


# load all the TRAINING datasets

training_solutions_df = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx')
train_cat_df = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx')
train_fcm_df = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')
train_quant_df = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx')


# load all the provided datasets

test_cat_df = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
test_fcm_df = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
test_quant_df = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')


data_dict_df.head()


data_dict_df.shape


data_dict_df.tail()


data_dict_df.info()


data_dict_df.describe()


sample_submission_df.head()


sample_submission_df.shape


training_solutions_df.head()


training_solutions_df.shape


train_cat_df.head()


train_cat_df.shape


train_cat_df.info()


train_cat_df.describe()


train_fcm_df.head()


train_fcm_df.tail()


train_fcm_df.info()


train_fcm_df.describe()


len(train_fcm_df.columns)


train_quant_df.head()


train_quant_df.tail()


train_quant_df.info()


train_quant_df.describe().round()


test_cat_df.head()


test_fcm_df.head()


test_quant_df.head()


#Barratt_Barratt_P2_Occ - Barratt Simplified Measure of Social Status - Parent 2 Occupation
train_cat_df['Barratt_Barratt_P2_Occ'].value_counts()

#Look back at the dictionary on Kaggle!
# to see what category these integers [0, 45, 35...] represent.


sns.countplot(x='Barratt_Barratt_P2_Occ', data=train_cat_df[['Barratt_Barratt_P2_Occ']])
plt.title(f"Distribution of Barratt_Barratt_P2_Occ")
plt.xticks(rotation=45)
plt.show()


# Distribution of MRI_Track_Age_at_Scan
train_quant_df['MRI_Track_Age_at_Scan'].hist(figsize=(12, 10), bins=20)
plt.suptitle("MRI_Track_Age_at_Scan Distributions")
plt.xlabel('MRI_Track_Age_at_Scan')
plt.ylabel('Frequency Count')
plt.show()


# ADHD distribution
training_solutions_df['ADHD_Outcome'].value_counts()


training_solutions_df['ADHD_Outcome'].value_counts().plot(kind='bar', color='blue')
plt.title('ADHD Outcome')
plt.xlabel('Outcome (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# Gender distribution
training_solutions_df['Sex_F'].value_counts()


training_solutions_df['Sex_F'].value_counts().plot(kind='bar', color='blue')
plt.title('Gender Distribution')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.show()


# Plot the distribution of the SDQ_SDQ_Emotional_Problems variable
plt.figure(figsize=(8, 6))
sns.histplot(train_quant_df['SDQ_SDQ_Emotional_Problems'], kde=True, color='skyblue')
plt.title('Distribution of SDQ_SDQ_Emotional_Problems')
plt.xlabel('SDQ_SDQ_Emotional_Problems')
plt.ylabel('Frequency')
plt.show()


# Check for correlation with ADHD outcome
# copying 'ADHD_Outcome' from training_solutions_df into train_quantitative_metadata_df
train_quant_copy_df = train_quant_df.copy()
train_quant_copy_df['ADHD_Outcome'] = training_solutions_df['ADHD_Outcome']

plt.figure(figsize=(8, 6))
sns.boxplot(x='ADHD_Outcome', y='SDQ_SDQ_Emotional_Problems', data=train_quant_copy_df)
plt.title('SDQ_SDQ_Emotional_Problems vs ADHD Outcome')
plt.xlabel('ADHD Outcome')
plt.ylabel('SDQ_SDQ_Emotional_Problems')
plt.show()



sns.countplot(data=train_cat_df, x='Barratt_Barratt_P1_Edu', hue=training_solutions_df['ADHD_Outcome'])
plt.title('ADHD Prevalence by Parent 1 Education')
plt.show()


train_cat_df['Barratt_Barratt_P1_Edu'].value_counts()


# Add ADHD_Outcome directly to a copy of the train_cat dataset for grouping
train_cat_copy_df = train_cat_df.copy()
train_cat_copy_df['ADHD_Outcome'] = training_solutions_df['ADHD_Outcome']

adhd_percentages = train_cat_copy_df.groupby('Barratt_Barratt_P1_Edu')['ADHD_Outcome'].mean()
print(adhd_percentages)


for col in train_cat_df.select_dtypes(include='int').columns:
    train_cat_df[col] = train_cat_df[col].astype('category')


# Creating a list of all of the columns except the first
columns_to_encode = train_cat_df.columns[1:].tolist()

# Print the columns to encode
print("Columns to encode:", columns_to_encode)


# encoding categorical data
train_encoded = pd.get_dummies(train_cat_df[columns_to_encode], drop_first=True)
train_encoded = train_encoded.map(lambda x: 1 if x is True else (0 if x is False else x))


# Combine encoded columns with the rest of the DataFrame
train_cat_final_df = pd.concat([train_cat_df.drop(columns=columns_to_encode), train_encoded], axis=1)

# ensure it looks correct
train_cat_final_df.head()


train_cat_final_df.tail()


# convert our int variables to categories
for col in test_cat_df.select_dtypes(include='int').columns:
    test_cat_df[col] = test_cat_df[col].astype('category')


# Encode categorical variables in test
test_encoded = pd.get_dummies(test_cat_df[columns_to_encode], drop_first=True)
test_encoded = test_encoded.map(lambda x: 1 if x is True else (0 if x is False else x))


# Ensure test_encoded has the same columns as train_encoded
missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0  # Add missing columns with 0 values


# Ensure test_encoded columns are in the same order as train_encoded
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)


# Combine encoded columns with the rest of the DataFrame
test_cat_final_df = pd.concat([test_cat_df.drop(columns=columns_to_encode), test_encoded], axis=1)


test_cat_final_df.head()


train_cat_fcm_df = pd.merge(train_cat_final_df, train_fcm_df, on = 'participant_id')


train_df = pd.merge(train_cat_fcm_df, train_quant_df, on = 'participant_id')

# ensure it looks accurate
train_df.head()


train_df.describe()


test_cat_fcm_df = pd.merge(test_cat_final_df, test_fcm_df, on = 'participant_id')


test_df = pd.merge(test_cat_fcm_df, test_quant_df, on = 'participant_id')


test_df.head()


# check how many NA values we have
print(train_df.isna().sum())


train_df.fillna({'MRI_Track_Age_at_Scan':train_df['MRI_Track_Age_at_Scan'].mean()}, inplace = True)
train_df.fillna({'PreInt_Demos_Fam_Child_Ethnicity':train_df['PreInt_Demos_Fam_Child_Ethnicity'].mean()}, inplace = True)

print(train_df.isna().sum().sum()) # should now be zero


# Fill NAs of test data

for col in test_df.columns:
    if test_df[col].isna().sum() > 0:  # Check if the column has NaN values
        if test_df[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            test_df[col] = test_df[col].fillna(test_df[col].mean())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")


print(test_df.isna().sum().sum()) # should now be zero


X_train = train_df.drop(columns = ['participant_id'])
Y_train = training_solutions_df.drop(columns = ['participant_id'])


# Initialize the base classifier
xgb_classifier = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)


# Wrap with MultiOutputClassifier for multi-target classification
multioutput_classifier = MultiOutputClassifier(xgb_classifier)


# Train the model

multioutput_classifier.fit(X_train, Y_train)


participant_id = test_df['participant_id']

X_test = test_df.drop(columns = 'participant_id')

y_pred = multioutput_classifier.predict(X_test)


# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(
    y_pred,
    columns=['Predicted_Gender', 'Predicted_ADHD']
)

# Combine participant IDs with predictions
result_df = pd.concat([participant_id.reset_index(drop=True), predictions_df], axis=1)

# Print or save the DataFrame
print(result_df)


# result_df.to_csv("submission.csv", index = False)


def multi_output_accuracy(y_true, y_pred):
    # Ensure y_true and y_pred are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Compute accuracy for each target variable and return the mean
    return np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])


# Create a scorer using scikit-learn's make_scorer
multi_output_scorer = make_scorer(multi_output_accuracy)


# Perform cross-validation on the training data
cv_scores = cross_val_score(multioutput_classifier, X_train, Y_train, cv=5, scoring=multi_output_scorer)

# Output the cross-validation results
print("Cross-validation scores for each fold:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))


model = LogisticRegression(max_iter=1000)
model.fit(train_df.drop(columns='participant_id'), training_solutions_df['Sex_F'])


# Get coefficients for Sex prediction
coefficients = pd.Series(model.coef_[0], index=train_df.drop(columns='participant_id').columns)


# Select top features for Sex prediction
top_features = coefficients.abs().nlargest(10)
print(top_features)


#Plotting the top 10 coefficents for Sex Outcome
plt.figure(figsize=(10,6))
top_features.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 10 Features for Sex Outcome')
plt.ylabel('Features')
plt.xlabel('Absolute Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.show()


model = LogisticRegression(max_iter=1000)
model.fit(train_df.drop(columns='participant_id'), training_solutions_df['ADHD_Outcome'])


# Get coefficients for ADHD_Outcome prediction
coefficients = pd.Series(model.coef_[0], index=train_df.drop(columns='participant_id').columns)


# Select top features for ADHD_Outcome prediction
top_features = coefficients.abs().nlargest(10)
print(top_features)


#Plotting the top 10 coefficents
plt.figure(figsize=(10,6))
top_features.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 10 Features for ADHD Outcome')
plt.ylabel('Features')
plt.xlabel('Absolute Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.show()


model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(train_df.drop(columns='participant_id'), training_solutions_df['Sex_F'])


selected_features_Sex = train_df.drop(columns='participant_id').columns[model.coef_[0] != 0]
print(selected_features_Sex)


model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(train_df.drop(columns='participant_id'), training_solutions_df['ADHD_Outcome'])


selected_features_ADHD = train_df.drop(columns='participant_id').columns[model.coef_[0] != 0]
print(selected_features_ADHD)


# Step 1: Find common features between ADHD and Sex selected features
common_features = list(set(selected_features_ADHD) & set(selected_features_Sex))


X_train_2 = X_train[common_features]
X_test_2 = X_test[common_features]


# Initialize the base classifier
xgb_classifier = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)


# Wrap with MultiOutputClassifier for multi-target classification
multioutput_classifier = MultiOutputClassifier(xgb_classifier)


# Train the model
multioutput_classifier.fit(X_train_2, Y_train)


y_pred_2 = multioutput_classifier.predict(X_test_2)


# Convert predictions to a DataFrame
predictions_df_2 = pd.DataFrame(
    y_pred_2,
    columns=['Predicted_Gender', 'Predicted_ADHD']
)

# Combine participant IDs with predictions
result_df_2 = pd.concat([participant_id.reset_index(drop=True), predictions_df_2], axis=1)

result_df_2.head()


result_df_2.tail()


result_df_2.to_csv("submission.csv", index = False)


# Create a scorer using scikit-learn's make_scorer
multi_output_scorer = make_scorer(multi_output_accuracy)


# Perform cross-validation on the training data
cv_scores_2 = cross_val_score(multioutput_classifier, X_train_2, Y_train, cv=5, scoring=multi_output_scorer, n_jobs=-1)

# Output the cross-validation results
print("Cross-validation scores for each fold:", cv_scores_2)
print("Mean CV score:", np.mean(cv_scores_2))


log_features = [f for f in features if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]


X_train, X_test, y_train, y_test = train_test_split(train.drop(targets,axis=1), 
                                                    y[targets], 
                                                    test_size=0.30, 
                                                    random_state=42)
model = MultiOutputClassifier(make_pipeline(
                        
                              ColumnTransformer([('imputer',SimpleImputer(),features)],
                                               remainder='passthrough',
                                               verbose_feature_names_out=False).set_output(transform='pandas'),
                              ColumnTransformer([('log', 
                                                 FunctionTransformer(np.log1p), log_features)],
                                                 remainder='passthrough'),
                              
                            MinMaxScaler(),    
                              
                            RidgeClassifier(alpha=100)))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('f1: ', f1_score(y_test,y_pred,average='micro'))


pca = make_pipeline(SimpleImputer(),StandardScaler(),PCA())
pca.fit(train[test.columns])
plt.figure(figsize=(7,5))
plt.plot(pca[-1].explained_variance_ratio_.cumsum())
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Principal Components Analysis')
plt.xlabel('component#')
plt.ylabel('explained variance ratio')
plt.yticks([0,0.5,0.85,0.90,0.95,1])
plt.xticks(range(0,1300,100))
plt.grid()
plt.show()


pipe = make_pipeline(SimpleImputer(),MinMaxScaler())
reducer = umap.UMAP()
x_scaler = pipe.fit_transform(train[features])
reducer.fit(x_scaler)
_, axs = plt.subplots(1,2, figsize=(5,3), constrained_layout=True)
embedding = reducer.transform(x_scaler)
for t,ax in zip(targets,axs.ravel()):    
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y[t], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    ax.set_title(f'{t}')
plt.suptitle('UMAP',fontsize=22);


model = MultiOutputClassifier(make_pipeline(ColumnTransformer([('imputer',SimpleImputer(),features)],
                                               remainder='passthrough',
                                               verbose_feature_names_out=False).set_output(transform='pandas'),
                                              ColumnTransformer([('log', 
                                                 FunctionTransformer(np.log1p), log_features)],
                                                 remainder='passthrough'),
                                            MinMaxScaler(),  
                                            PCA(1087),
                                            RidgeClassifier(alpha=100)))
model.fit(train.drop(targets,axis=1),
          y.drop('participant_id',axis=1))
y_pred = model.predict(test)
sub['ADHD_Outcome'] = y_pred[:,0]
sub['Sex_F'] = y_pred[:,1]
sub.to_csv('submission.csv',index=False)

