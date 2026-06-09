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


# create dataframe of the train datasets
connectome_train=pd.read_csv(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
quant_train=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cate_train=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_solution=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")


connectome_train.info()
connectome_train.head()


quant_train.info()


cate_train.info()


train_solution.head()


# create dataframes of the test data
connectome_test=pd.read_csv(f"/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
cate_test=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
quant_test=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")


connectome_test.head()


cate_test.info()


quant_test.info()


# merge the quantitative and categorical datasets by participant id
merged_train=pd.merge(quant_train, cate_train, on='participant_id', how='inner')
merged_test=pd.merge(quant_test, cate_test, on='participant_id', how='inner')


# merge the train dataset with their outcomes
merged_train=pd.merge(merged_train, train_solution, on='participant_id', how='inner')


merged_train.info()
merged_test.info()


# identify the numerical columns
numerical_columns_train = merged_train.select_dtypes(include=['int64', 'float64']).columns
numerical_columns_test = merged_test.select_dtypes(include=['int64', 'float64']).columns


# fill the missing values with medians
merged_train[numerical_columns_train] = merged_train[numerical_columns_train].fillna(merged_train[numerical_columns_train].median())
merged_test[numerical_columns_test] = merged_test[numerical_columns_test].fillna(merged_test[numerical_columns_test].median())


merged_train.info()
merged_test.info()


# check for the missing values of connect outcome matrix
train_NA = connectome_train.isnull().values.any()
print("Number of missing values in connectome_train:", train_NA.sum())
test_NA = connectome_test.isnull().values.any()
print("Number of missig values in connectome_test:", test_NA.sum())


# Firstly import the packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


train_numerical = merged_train[numerical_columns_train]
test_numerical = merged_test[numerical_columns_test]


# draw bar charts to illustrate the train solution
sns.countplot(x='ADHD_Outcome',hue='Sex_F',data = merged_train)
plt.show()


# Use box plots to show the distribution of outcomes based on the categorical factors



# create a column to show the combined outcome of ADHD and Sex
merged_train['ADHD-Sex'] = merged_train['ADHD_Outcome'].astype(str) + "_" +merged_train['Sex_F'].astype(str)


# draw bar charts to show the distribution categorica factors of outcomes
# extract numerical columns
numeric_cate = cate_train.select_dtypes(include=['int64','float64'])
for col in numeric_cate:
    counts = merged_train.groupby([col, 'ADHD-Sex']).size().reset_index(name='counts')

    # Plot grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=col, y='counts', hue='ADHD-Sex', data=counts)
    plt.title(f"Distribution of {col} by outcome")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


# Next, draw box plots to show the distribution of quantitative factors
# extract the numerical data index
numerical_quant = quant_train.select_dtypes(include=['int64','float64']).columns

for col in numerical_quant:
    plt.figure(figsize=(8,6))
    sns.boxplot(x='ADHD-Sex',y=col, data=merged_train)
    plt.title(f"Distribution of {col} by Sex_F and ADHD outcome")
    plt.show()



from sklearn.decomposition import PCA


connectome_train.info()
connectome_train.shape


# extract participant id
participant_id = connectome_train.iloc[:,0]
connectivity_features = connectome_train.iloc[:, 1:]
participant_id_test = connectome_test.iloc[:,0]
connectivity_features_test = connectome_test.iloc[:, 1:]


# fitting PCA without restrictions
pca_full = PCA()
pca_full.fit(connectivity_features)


# compute explained variance ratio
explained_variance_ratio = pca_full.explained_variance_ratio_
print("Explained Variance Ratios (first 10 components):", explained_variance_ratio[:10])


# compute cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)


# Scree Plot (Cumulative Explained Variance Plot)
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot: Cumulative Explained Variance vs. Number of Components')
plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(0.5, 0.85, '90% variance threshold', color='red', fontsize=12)
plt.grid(True)
plt.show()


# Determine the Optimal Number of Components:
# choose the number whose explained variance can reach 90%
n_components_90 = np.where(cumulative_variance >= 0.90)[0][0] + 1
print(f"Number of components to reach 90% explained variance: {n_components_90}")


# dimensionality reduction using PCA
pca = PCA(n_components=n_components_90)
pca_features = pca.fit_transform(connectivity_features)
print("Shape of PCA-reduced features:", pca_features.shape)


# Create a DataFrame for the PCA Results for the train data
reduced_connectome_train = pd.DataFrame(pca_features, index=participant_id,
                      columns=[f"PC{i+1}" for i in range(n_components_90)])
reduced_connectome_train.head()


# create dataframe for the PCA results for the test data

# ----------------------------------------------------------------------------
# Apply PCA with n_components=708 using the 'randomized' solver.
# Note: n_components must be <= min(n_samples, n_features). In our case, 708 <= 1214.
# ----------------------------------------------------------------------------


# Fit PCA on the training data only.
pca = PCA(n_components=n_components_90, svd_solver='randomized', random_state=42)
pca.fit(connectivity_features)  # Fit on training features only.

# Transform the data using the fitted PCA
train_pca = pca.transform(connectivity_features)
test_pca = pca.transform(connectivity_features_test)

# ----------------------------------------------------------------------------
# Create a DataFrame for the PCA-transformed data.
# We'll name the columns PC1, PC2, ..., PC708.
# ----------------------------------------------------------------------------
reduced_connectome_train = pd.DataFrame(train_pca, index=participant_id,
                      columns=[f"PC{i+1}" for i in range(n_components_90)])
print("Shape of PCA-transformed data of train:", reduced_connectome_train.shape)
reduced_connectome_train.head()

reduced_connectome_test = pd.DataFrame(test_pca, index=participant_id_test,
                      columns=[f"PC{i+1}" for i in range(n_components_90)])

print("Shape of PCA-transformed data of test:", reduced_connectome_test.shape)
reduced_connectome_train.head()



# integrate the reduced functional connectome data with other data
merged_train = pd.merge(merged_train, reduced_connectome_train,left_on="participant_id", right_index=True, how="inner")
print("Final Merged DataFrame shape:", merged_train.shape)
print(merged_train.head())

merged_test = pd.merge(merged_test, reduced_connectome_test,left_on="participant_id", right_index=True, how="inner")
print("Final Merged DataFrame shape:", merged_test.shape)
print(merged_test.head())


# draw a correlation heatmap of all numerical features in the integrated dataset
# extract numerical columns
numeric_cols = merged_train.select_dtypes(include=["float64", "int64"]).columns
plt.figure(figsize=(12, 10))
corr_matrix = merged_train[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score


# Separate Predictors and the Target Variable
x_train = merged_train.drop(["participant_id", "ADHD_Outcome", "Sex_F","ADHD-Sex"], axis=1)
y_train = merged_train["ADHD_Outcome"]
x_test = merged_train.drop(["participant_id"], axis=1)


f1_score_list = []
gbc = GradientBoostingClassifier(random_state=42)

for k in range(50,301,50):
    selector = SelectKBest(mutual_info_classif,k=k)
    selector.fit(x_train,y_train)

    sel_x_train = selector.transform(x_train)

    gbc.fit(sel_x_train, y_train)
    kbest_preds = gbc.predict(sel_x_train)

    f1_score_kbest = round(f1_score(y_train, kbest_preds, average="weighted"),3)

    f1_score_list.append(f1_score_kbest)


# Determine the best k based on the highest F1 score
best_index = np.argmax(f1_score_list)
k_range = range(50,301,50)
best_k = k_range[best_index]
print("Best k (number of features) based on F1 score:", best_k)


# Re-run the feature selection with the best k and fit the selector on x and y
selector = SelectKBest(mutual_info_classif,k=best_k)
selector.fit(x_train, y_train)
X_final_selected = selector.transform(x_train)
selected_feature_names = x_train.columns[selector.get_support()]
print("Selected feature names:", selected_feature_names.tolist())


#  Build the final DataFrame for train data
selected_features_df = pd.DataFrame(
    X_final_selected,
    index=x_train.index,
    columns=selected_feature_names
)

# Now, combine with participant_id and outcome from final_df.
final_train_df = pd.concat(
    [merged_train[['participant_id', 'ADHD_Outcome', 'Sex_F']], selected_features_df],
    axis=1
)

print(final_train_df.head())

# And for test
selected_features_df_test = merged_test[selected_feature_names.tolist()]

# Now, combine with participant_id and outcome from final_df.
final_test_df = pd.concat(
    [merged_test['participant_id'], selected_features_df_test],
    axis=1
)

print(final_test_df.head())
final_test_df.info()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


feature_cols = [c for c in final_train_df.columns 
                if c not in ("participant_id", "ADHD_Outcome", "Sex_F")]

X_train = final_train_df[feature_cols]
y_train_adhd = final_train_df["ADHD_Outcome"]
y_train_sex  = final_train_df["Sex_F"]


X_test = final_test_df[feature_cols]
ids    = final_test_df["participant_id"]


xgb_adhd = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
xgb_adhd.fit(X_train, y_train_adhd)


xgb_sex = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
xgb_sex.fit(X_train, y_train_sex)


pred_adhd = xgb_adhd.predict(X_test)
pred_sex  = xgb_sex.predict(X_test)


submission = pd.DataFrame({
    "participant_id": ids,
    "ADHD_Outcome":  pred_adhd,
    "Sex_F":         pred_sex
})

submission.to_csv("my_submission.csv", index=False)
print("Wrote my_submission.csv with", len(submission), "rows.")

