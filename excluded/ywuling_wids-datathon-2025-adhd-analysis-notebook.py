import numpy as np
import pandas as pd
import warnings

# Ignore runtime warnings
    # /usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1458: RuntimeWarning: invalid value encountered in greater
    #   has_large_values = (abs_vals > 1e6).any()
    # /usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in less
    #   has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


### LOAD TRAINING DATA
# Load quantitative metadata
train = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")

# Load categorical metadata
cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")

# Merge quantitative and categorical data
train = pd.merge(train, cate, on='participant_id', how='left')

# Load training solutions (only for TRAIN mode)
solution = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
train = pd.merge(train, solution, on='participant_id', how='left')


### LOAD TEST DATA
# Load quantitative metadata
test = pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")

# Load categorical metadata
cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")

# Merge quantitative and categorical data
test = pd.merge(test, cate, on='participant_id', how='left')


# Display the first few rows of the training data
train.head()


train.info()


test.head()


test.info()


# Based on the competition's provided Data Dictionary
data = pd.read_excel(f"/kaggle/input/widsdatathon2025/Data Dictionary.xlsx")

df = pd.DataFrame(data)
df.iloc[list(range(2,20)) + list(range(23,32)),1:4].style.set_table_styles( # select specific rows & columns
    [{'selector':'th','props':[('text-align','left')]}]).set_properties( # align column headers left for readability
    subset=pd.IndexSlice[:], **{'text-align':'left'}).hide() # align rows to the left + hide row indexes


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features and target variables
X = train.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1, errors='ignore')
y_adhd = train['ADHD_Outcome']
y_sex = train['Sex_F']

# Identify categorical and numerical features
    # categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    # numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()
    # above do not work because the given data defines categ features with numerical int values
categorical_features = X.iloc[:,18:27].columns.tolist()
# print(categorical_features)
numerical_features = X.iloc[:,:18].columns.tolist()
# print(numerical_features)

# Create preprocessing pipelines - library in sklearn to create data processing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # fills in missing vals
    ('scaler', StandardScaler()) # scales the vars - z score
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

preprocessor.set_output(transform="pandas")

# Apply preprocessing to training data
print("Preprocessing data...")
X_preprocessed = preprocessor.fit_transform(X)


train.info()


X_preprocessed.info()


train['ADHD_Outcome'].sum()/len(train)


train['Sex_F'].sum()/len(train)


import matplotlib.pyplot as plt # graphing libraries
import seaborn as sns

sns.boxplot(data=train, y = 'EHQ_EHQ_Total', x = 'Sex_F')
plt.title("Laterality Index (Score)")


sns.countplot(data=train, x = 'ColorVision_CV_Score', hue = 'Sex_F')
plt.title("Color Vision Test Score")


table = pd.crosstab(train['ColorVision_CV_Score'], train['Sex_F'], normalize='index')
print(table)


# APQ_P_APQ_P_CP	Corporal Punishment Score
# APQ_P_APQ_P_ID	Inconsistent Discipline Score
# APQ_P_APQ_P_INV	Involvement Score
# APQ_P_APQ_P_OPD	Other Discipline Practices Score (Not factored into total score but provides item level information)
# APQ_P_APQ_P_PM	Poor Monitoring/Supervision Score
# APQ_P_APQ_P_PP	Positive Parenting Score


sns.countplot(data=train, x = 'APQ_P_APQ_P_CP', hue = 'Sex_F')
plt.title("Corporal Punishment Score")


table = pd.crosstab(train['APQ_P_APQ_P_CP'], train['Sex_F'], normalize='index')
print(table)


sns.boxplot(data=train, y = 'APQ_P_APQ_P_ID', x = 'Sex_F')
plt.title("Inconsistent Discipline Score")


sns.boxplot(data=train, y = 'APQ_P_APQ_P_INV', x = 'Sex_F')
plt.title("Involvement Score")


sns.boxplot(data=train, y = 'APQ_P_APQ_P_OPD', x = 'Sex_F')


sns.boxplot(data=train, y = 'APQ_P_APQ_P_PM', x = 'Sex_F')


sns.boxplot(data=train, y = 'APQ_P_APQ_P_PP', x = 'Sex_F')


sns.countplot(data=train, x = 'SDQ_SDQ_Conduct_Problems', hue = 'ADHD_Outcome')
plt.title("Conduct Problems Scale from Training Dataset")


table = pd.crosstab(train['SDQ_SDQ_Conduct_Problems'], train['ADHD_Outcome'], normalize='index')
print(table)


sns.countplot(data=train, x = 'SDQ_SDQ_Conduct_Problems', hue = 'Sex_F')
plt.title("Conduct Problems Scale from Training Dataset")


table = pd.crosstab(train['SDQ_SDQ_Conduct_Problems'], train['Sex_F'], normalize='index')
print(table)


sns.boxplot(data=train, y = 'SDQ_SDQ_Difficulties_Total', x = 'ADHD_Outcome')
plt.title("Total Difficulties Score from Training Dataset")


table = pd.crosstab(train['SDQ_SDQ_Difficulties_Total'], train['ADHD_Outcome'], normalize='index')
print(table)


sns.boxplot(data=train, y = 'SDQ_SDQ_Difficulties_Total', x = 'Sex_F')
plt.title("Total Difficulties Score from Training Dataset")


table = pd.crosstab(train['SDQ_SDQ_Difficulties_Total'], train['Sex_F'], normalize='index')
print(table)


sns.boxplot(data=train, y = 'SDQ_SDQ_Hyperactivity', x = 'Sex_F')
plt.title("Hyperactivity Scale")


sns.boxplot(data=train, y = 'SDQ_SDQ_Internalizing', x = 'Sex_F')
plt.title("Internalizing Score")


sns.boxplot(data=train, y = 'MRI_Track_Age_at_Scan', x = 'Sex_F')
plt.title("Age at Time of MRI Scan")


table = pd.crosstab(train['MRI_Track_Age_at_Scan'], train['Sex_F'], normalize='index')
print(table)


sns.countplot(data=train, x = 'PreInt_Demos_Fam_Child_Ethnicity', hue = 'ADHD_Outcome')
plt.title("Child Ethnicity from Training Dataset")

# 0= Not Hispanic or Latino
# 1= Hispanic or Latino
# 2= Decline to specify
# 3= Unknown


table = pd.crosstab(train['PreInt_Demos_Fam_Child_Ethnicity'], train['ADHD_Outcome'], normalize='index')
print(table)


sns.countplot(data=train, x = 'PreInt_Demos_Fam_Child_Ethnicity', hue = 'Sex_F')
plt.title("Child Ethnicity from Training Dataset")

# 0= Not Hispanic or Latino
# 1= Hispanic or Latino
# 2= Decline to specify
# 3= Unknown


table = pd.crosstab(train['PreInt_Demos_Fam_Child_Ethnicity'], train['Sex_F'], normalize='index')
print(table)


sns.countplot(data=train, x = 'PreInt_Demos_Fam_Child_Race', hue = 'ADHD_Outcome')
plt.title("Child Race from Training Dataset")

# 0= White/Caucasian
# 1= Black/African American
# 2= Hispanic
# 3= Asian
# 4= Indian
# 5= Native American Indian
# 6= American Indian/Alaskan Native
# 7= Native Hawaiian/Other Pacific Islander
# 8= Two or more races
# 9= Other race
# 10= Unknown
# 11=Choose not to specify


table = pd.crosstab(train['PreInt_Demos_Fam_Child_Race'], train['ADHD_Outcome'], normalize='index')
print(table)


sns.countplot(data=train, x = 'PreInt_Demos_Fam_Child_Race', hue = 'Sex_F')
plt.title("Child Race from Training Dataset")

# 0= White/Caucasian
# 1= Black/African American
# 2= Hispanic
# 3= Asian
# 4= Indian
# 5= Native American Indian
# 6= American Indian/Alaskan Native
# 7= Native Hawaiian/Other Pacific Islander
# 8= Two or more races
# 9= Other race
# 10= Unknown
# 11=Choose not to specify


table = pd.crosstab(train['PreInt_Demos_Fam_Child_Race'], train['Sex_F'], normalize='index')
print(table)


import lightgbm as lgb # to train a series of decision trees
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Split data into training and validation sets
X_train, X_val, y_train_adhd, y_val_adhd, y_train_sex, y_val_sex = train_test_split(
    X_preprocessed, y_adhd, y_sex, test_size=0.2, random_state=42, stratify=y_sex
) # Can we stratify by multiple variables?

# Calculate class weights for ADHD and sex
adhd_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_adhd), y=y_adhd)
sex_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_sex), y=y_sex)

# Define LightGBM models
adhd_model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=63,
    learning_rate=0.01,
    n_estimators=1000, # specifies num of trees in forest, accuracy stabilizes after certain amount
    scale_pos_weight=adhd_weights[1] / adhd_weights[0], # more adhd than non; lower weight on adhd makes up for bias
    early_stopping_rounds=50,
    verbose=-1
)

sex_model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=100, # smaller tree better fit for small dataset
    learning_rate=0.1, # how much model adjusts based on new trees/boosting
    n_estimators=10, # num of decision trees
    scale_pos_weight=sex_weights[1] / sex_weights[0], # more males than females; weight on females makes up for bias
    early_stopping_rounds=50, # stop at this num of trees before reaching n_estimators num due to lack of improvement
    verbose=-1 # do not print updates during decision tree process
)

# Train ADHD model
print("Training ADHD LightGBM model...")
adhd_model.fit(X_train, y_train_adhd, eval_set=[(X_val, y_val_adhd)]) # eval_set = specify validation sets

# Train Sex model
print("Training Sex LightGBM model...")
sex_model.fit(X_train, y_train_sex, eval_set=[(X_val, y_val_sex)])


y_train_sex.sum()


y_train_sex


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Split dataset into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train_adhd, y_test_adhd, y_train_sex, y_test_sex = train_test_split(
    X_preprocessed, y_adhd, y_sex, test_size=0.2, random_state=10, stratify=y_sex
)

# Optimize best parameter values for decision tree
param_grid = {
    'max_depth': [3, 5, 7],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv=10)

# Find best parameters for ADHD model
grid_search.fit(X_train, y_train_adhd)
print("Best parameters for ADHD model:", grid_search.best_params_)
print("Best score for ADHD model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

clf_adhd = grid_search.best_estimator_

# Train ADHD model
print("Training ADHD Decision Tree model...")
clf_adhd.fit(X_train, y_train_adhd)

# Find best parameters for Sex model
grid_search.fit(X_train, y_train_sex)
print("\nBest parameters for Sex model:", grid_search.best_params_)
print("Best score for Sex model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

clf_sex = grid_search.best_estimator_

# Train Sex model
print("Training Sex Decision Tree model...")
clf_sex.fit(X_train, y_train_sex)


features_list = X_preprocessed.columns.tolist() # 80 "features" due to one-hot encoding

feature_importances = pd.Series(clf_adhd.feature_importances_, index=features_list).sort_values(ascending=False)
print("Feature Importances for ADHD Decision Tree:\n")
feature_importances


feature_importances = pd.Series(clf_sex.feature_importances_, index=features_list).sort_values(ascending=False)
print("Feature Importances for Sex Decision Tree:\n")
feature_importances


from sklearn.ensemble import RandomForestClassifier

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train_adhd, y_test_adhd, y_train_sex, y_test_sex = train_test_split(
    X_preprocessed, y_adhd, y_sex, test_size=0.2, random_state=10, stratify=y_sex
)

# Optimize best parameter values for random forest
param_grid = {
    'n_estimators': [10, 50, 100], # 150 and 200 produced same or worse results
    'max_depth': [7, 10, 13], # out of [3, 5, 7], 7 was the best-performing for both adhd and sex models (random_state=42)
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=10), param_grid, cv=10)

# Find best parameters for ADHD model
grid_search.fit(X_train, y_train_adhd)
print("Best parameters for ADHD model:", grid_search.best_params_)
print("Best score for ADHD model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

rf_adhd = grid_search.best_estimator_

# Train ADHD model
print("Training ADHD Random Forest model...")
rf_adhd.fit(X_train, y_train_adhd)

# Find best parameters for Sex model
grid_search.fit(X_train, y_train_sex)
print("\nBest parameters for Sex model:", grid_search.best_params_)
print("Best score for Sex model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

rf_sex = grid_search.best_estimator_

# Train Sex model
print("Training Sex Random Forest model...")
rf_sex.fit(X_train, y_train_sex)


from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf_adhd, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = features_list, class_names=['0','1']) # 0=No ADHD, 1=Has ADHD
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DecisionTreeADHD.png')
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", "")) # removing the black rectangle, a bug
Image(graph.create_png())


from sklearn.tree import export_text

tree_text = export_text(clf_adhd, feature_names=features_list)
print(tree_text)


tree_matrix = clf_adhd.decision_path(X_train).todense()
print(tree_matrix)
print("\n", tree_matrix[0]) # Instance #1 passes through Nodes 1,2,6,7 and none else


dot_data = StringIO()
export_graphviz(clf_sex, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = features_list, class_names=['0','1']) # 0=Male, 1=Female
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DecisionTreeSex.png')
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", "")) # removing the black rectangle, a bug
Image(graph.create_png())


tree_text = export_text(clf_sex, feature_names=features_list)
print(tree_text)


tree_matrix = clf_sex.decision_path(X_train).todense()
print(tree_matrix)
print("\n", tree_matrix[0])


from sklearn.metrics import f1_score

# The F1 score is the harmonic mean of precision and recall, representing the balance between these two metrics.
# Precision = True Positives / (True Positives + False Positives) # the accuracy of positive predictions
# Recall = True Positives / (True Positives + False Negatives) # the ability to find all positive instances in the dataset

# Make predictions on the validation set
adhd_pred = adhd_model.predict(X_val)

# Calculate F1 scores
index = (y_val_sex == 0)
adhd_f1_males = f1_score(y_val_adhd[index], adhd_pred[index])

index = (y_val_sex == 1)
adhd_f1_females = f1_score(y_val_adhd[index], adhd_pred[index])

weighted_f1 = (2 * adhd_f1_females + adhd_f1_males) / 3

print(f"ADHD F1 Score for Males: {adhd_f1_males:.4f}")
print(f"ADHD F1 Score for Females: {adhd_f1_females:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")


def weighted_f1_score(y_true_adhd, y_pred_adhd, y_true_sex, y_pred_sex):
    weights = [2 if (a == 1 and s == 1) else 1 # places more weight on individuals who are both female and have adhd
            for a, s in zip(y_true_adhd, y_true_sex)]

    def compute_f1(y_true, y_pred, weights):
        TP = sum(w for i, w in enumerate(weights) if y_true.iloc[i] == 1 and y_pred[i] == 1) # target vars are
        FP = sum(w for i, w in enumerate(weights) if y_true.iloc[i] == 0 and y_pred[i] == 1) # pandas objs, pred vals
        FN = sum(w for i, w in enumerate(weights) if y_true.iloc[i] == 1 and y_pred[i] == 0) # are numpy arrays

        if TP + FP == 0 or TP + FN == 0:
            return 0.0

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    f1_adhd = compute_f1(y_true_adhd, y_pred_adhd, weights)
    f1_sex = compute_f1(y_true_sex, y_pred_sex, weights)

    # Final F1 on the leaderboard
    return (f1_adhd + f1_sex) / 2

adhd_pred = adhd_model.predict(X_val)
sex_pred = sex_model.predict(X_val)

print(weighted_f1_score(y_val_adhd, adhd_pred, y_val_sex, sex_pred))


from sklearn.metrics import confusion_matrix
import numpy as np

# Make predictions on the testing set
adhd_pred = adhd_model.predict(X_val)
sex_pred = sex_model.predict(X_val)

# Generate the confusion matrix
cm_adhd = confusion_matrix(y_val_adhd, adhd_pred)
cm_sex = confusion_matrix(y_val_sex, sex_pred)
'''
TP FN
FP TN
'''

# Extract TP, TN, FP, FN from the confusion matrix
tn_adhd, fp_adhd, fn_adhd, tp_adhd = cm_adhd.ravel()
tn_sex, fp_sex, fn_sex, tp_sex = cm_sex.ravel()

# Calculate percentages
total = sum(cm_adhd.ravel())
tp_a_percent = (tp_adhd / total) * 100
tn_a_percent = (tn_adhd / total) * 100
fp_a_percent = (fp_adhd / total) * 100
fn_a_percent = (fn_adhd / total) * 100
total = sum(cm_sex.ravel())
tp_s_percent = (tp_sex / total) * 100
tn_s_percent = (tn_sex / total) * 100
fp_s_percent = (fp_sex / total) * 100
fn_s_percent = (fn_sex / total) * 100

print("ADHD Metrics")
print("True Positives:", tp_adhd)
print("True Negatives:", tn_adhd)
print("False Positives:", fp_adhd)
print("False Negatives:", fn_adhd)
print("True Positive Percentage:", tp_a_percent)
print("True Negative Percentage:", tn_a_percent)
print("False Positive Percentage:", fp_a_percent)
print("False Negative Percentage:", fn_a_percent, "\n")

print("Sex Metrics")
print("True Positives:", tp_sex)
print("True Negatives:", tn_sex)
print("False Positives:", fp_sex)
print("False Negatives:", fn_sex)
print("True Positive Percentage:", tp_s_percent)
print("True Negative Percentage:", tn_s_percent)
print("False Positive Percentage:", fp_s_percent)
print("False Negative Percentage:", fn_s_percent)


# Make predictions on the training set
adhd_pred = adhd_model.predict(X_train)
sex_pred = sex_model.predict(X_train)

# Generate the confusion matrix
cm_adhd = confusion_matrix(y_train_adhd, adhd_pred)
cm_sex = confusion_matrix(y_train_sex, sex_pred)
'''
TP FN
FP TN
'''

# Extract TP, TN, FP, FN from the confusion matrix
tn_adhd, fp_adhd, fn_adhd, tp_adhd = cm_adhd.ravel()
tn_sex, fp_sex, fn_sex, tp_sex = cm_sex.ravel()

# Calculate percentages
total = sum(cm_adhd.ravel())
tp_a_percent = (tp_adhd / total) * 100
tn_a_percent = (tn_adhd / total) * 100
fp_a_percent = (fp_adhd / total) * 100
fn_a_percent = (fn_adhd / total) * 100
total = sum(cm_sex.ravel())
tp_s_percent = (tp_sex / total) * 100
tn_s_percent = (tn_sex / total) * 100
fp_s_percent = (fp_sex / total) * 100
fn_s_percent = (fn_sex / total) * 100

print("ADHD Metrics")
print("True Positives:", tp_adhd)
print("True Negatives:", tn_adhd)
print("False Positives:", fp_adhd)
print("False Negatives:", fn_adhd)
print("True Positive Percentage:", tp_a_percent)
print("True Negative Percentage:", tn_a_percent)
print("False Positive Percentage:", fp_a_percent)
print("False Negative Percentage:", fn_a_percent, "\n")

print("Sex Metrics")
print("True Positives:", tp_sex)
print("True Negatives:", tn_sex)
print("False Positives:", fp_sex)
print("False Negatives:", fn_sex)
print("True Positive Percentage:", tp_s_percent)
print("True Negative Percentage:", tn_s_percent)
print("False Positive Percentage:", fp_s_percent)
print("False Negative Percentage:", fn_s_percent)


# Make predictions on the testing set
adhd_pred = clf_adhd.predict(X_test)
sex_pred = clf_sex.predict(X_test)

# Generate the confusion matrix
cm_adhd = confusion_matrix(y_test_adhd, adhd_pred)
cm_sex = confusion_matrix(y_test_sex, sex_pred)
'''
TP FN
FP TN
'''

# Extract TP, TN, FP, FN from the confusion matrix
tn_adhd, fp_adhd, fn_adhd, tp_adhd = cm_adhd.ravel()
tn_sex, fp_sex, fn_sex, tp_sex = cm_sex.ravel()

# Calculate percentages
total = sum(cm_adhd.ravel())
tp_a_percent = (tp_adhd / total) * 100
tn_a_percent = (tn_adhd / total) * 100
fp_a_percent = (fp_adhd / total) * 100
fn_a_percent = (fn_adhd / total) * 100
total = sum(cm_sex.ravel())
tp_s_percent = (tp_sex / total) * 100
tn_s_percent = (tn_sex / total) * 100
fp_s_percent = (fp_sex / total) * 100
fn_s_percent = (fn_sex / total) * 100

print("ADHD Metrics")
print("True Positives:", tp_adhd)
print("True Negatives:", tn_adhd)
print("False Positives:", fp_adhd)
print("False Negatives:", fn_adhd)
print("True Positive Percentage:", tp_a_percent)
print("True Negative Percentage:", tn_a_percent)
print("False Positive Percentage:", fp_a_percent)
print("False Negative Percentage:", fn_a_percent, "\n")

print("Sex Metrics")
print("True Positives:", tp_sex)
print("True Negatives:", tn_sex)
print("False Positives:", fp_sex)
print("False Negatives:", fn_sex)
print("True Positive Percentage:", tp_s_percent)
print("True Negative Percentage:", tn_s_percent)
print("False Positive Percentage:", fp_s_percent)
print("False Negative Percentage:", fn_s_percent)


# Make predictions on the testing set
adhd_pred = rf_adhd.predict(X_test)
sex_pred = rf_sex.predict(X_test)

# Generate the confusion matrix
cm_adhd = confusion_matrix(y_test_adhd, adhd_pred)
cm_sex = confusion_matrix(y_test_sex, sex_pred)
'''
TP FN
FP TN
'''

# Extract TP, TN, FP, FN from the confusion matrix
tn_adhd, fp_adhd, fn_adhd, tp_adhd = cm_adhd.ravel()
tn_sex, fp_sex, fn_sex, tp_sex = cm_sex.ravel()

# Calculate percentages
total = sum(cm_adhd.ravel())
tp_a_percent = (tp_adhd / total) * 100
tn_a_percent = (tn_adhd / total) * 100
fp_a_percent = (fp_adhd / total) * 100
fn_a_percent = (fn_adhd / total) * 100
total = sum(cm_sex.ravel())
tp_s_percent = (tp_sex / total) * 100
tn_s_percent = (tn_sex / total) * 100
fp_s_percent = (fp_sex / total) * 100
fn_s_percent = (fn_sex / total) * 100

print("ADHD Metrics")
print("True Positives:", tp_adhd)
print("True Negatives:", tn_adhd)
print("False Positives:", fp_adhd)
print("False Negatives:", fn_adhd)
print("True Positive Percentage:", tp_a_percent)
print("True Negative Percentage:", tn_a_percent)
print("False Positive Percentage:", fp_a_percent)
print("False Negative Percentage:", fn_a_percent, "\n")

print("Sex Metrics")
print("True Positives:", tp_sex)
print("True Negatives:", tn_sex)
print("False Positives:", fp_sex)
print("False Negatives:", fn_sex)
print("True Positive Percentage:", tp_s_percent)
print("True Negative Percentage:", tn_s_percent)
print("False Positive Percentage:", fp_s_percent)
print("False Negative Percentage:", fn_s_percent)


# Separate features and target variables
X_sex = train[['ColorVision_CV_Score', 'SDQ_SDQ_Emotional_Problems', 'APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_PM']]
y_sex = train['Sex_F']

numerical_features = X_sex.columns.tolist()
# print(numerical_features)

# Create preprocessing pipelines - library in sklearn to create data processing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # fills in missing vals
    ('scaler', StandardScaler()) # scales the vars - z score
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
    ]
)

preprocessor.set_output(transform="pandas")

# Apply preprocessing to training data
print("Preprocessing data...")
X_sex_preprocessed = preprocessor.fit_transform(X_sex)


# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train_sex, y_test_sex = train_test_split(
    X_sex_preprocessed, y_sex, test_size=0.2, random_state=10, stratify=y_sex
)

# Optimize best parameter values for decision tree
param_grid = {
    'max_depth': [3, 5, 7],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv=10)

# Find best parameters for Sex model
grid_search.fit(X_train, y_train_sex)
print("\nBest parameters for Sex model:", grid_search.best_params_)
print("Best score for Sex model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

clf_sex_new = grid_search.best_estimator_

# Train Sex model
print("Training Sex Decision Tree model...")
clf_sex_new.fit(X_train, y_train_sex)


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train_sex, y_test_sex = train_test_split(
    X_sex_preprocessed, y_sex, test_size=0.2, random_state=50, stratify=y_sex
)

clf_sex_new = DecisionTreeClassifier(max_depth=5)
clf_sex_new.fit(X_train, y_train_sex)

prob_pred = clf_sex_new.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.0, 1.0, step=0.01)
recall_scores = [recall_score(y_test_sex, prob_pred > t) for t in thresholds]
precis_scores = [precision_score(y_test_sex, prob_pred > t) for t in thresholds]
accura_scores = [accuracy_score(y_test_sex, prob_pred > t) for t in thresholds]
f1_scores = [f1_score(y_test_sex, prob_pred > t) for t in thresholds]


fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, recall_scores, label="Recall @ t")
ax.plot(thresholds, precis_scores, label="Precision @ t")
ax.plot(thresholds, accura_scores, label="Accuracy @ t")
ax.plot(thresholds, f1_scores, label="F1 @ t")
ax.axvline(0.5, c="gray", linestyle="--", label="Default Threshold")
ax.set_xlabel("Threshold")
ax.set_ylabel("Metric @ Threshold")
ax.set_box_aspect(1)
ax.legend()
plt.show()


from sklearn.metrics import accuracy_score

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train_sex, y_test_sex = train_test_split(
    X_sex_preprocessed, y_sex, test_size=0.2, random_state=10, stratify=y_sex
)

# param_grid = {
#     'max_depth': [3, 5, 7],
#     'class_weight': ['balanced', None]
# }
# grid_search = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv=10)

# grid_search.fit(X_train, y_train_sex)
# print("\nBest parameters for Sex model:", grid_search.best_params_)
# print("Best score for Sex model on training dataset:", grid_search.best_score_)

# cv_results = grid_search.cv_results_
# df_results = pd.DataFrame(cv_results)
# print(df_results[['param_max_depth','mean_test_score']])

# clf_sex_new = grid_search.best_estimator_

clf_sex_new = DecisionTreeClassifier(max_depth=5)
clf_sex_new.fit(X_train, y_train_sex)

probabilities = clf_sex_new.predict_proba(X_test)

# Define a custom threshold instead of the default 0.5
custom_threshold = 0.33

# For binary classification, probabilities[:, 1] refers to the probability of the positive class
custom_prediction = (probabilities[:, 1] >= custom_threshold).astype(int)

print(clf_sex_new.predict(X_test)[:10])
print(custom_prediction[:10])


# Make predictions on the testing set
sex_pred = clf_sex_new.predict(X_test)

# Generate the confusion matrix
cm_sex = confusion_matrix(y_test_sex, sex_pred)
'''
TP FN
FP TN
'''

# Extract TP, TN, FP, FN from the confusion matrix
tn_sex, fp_sex, fn_sex, tp_sex = cm_sex.ravel()

# Calculate percentages
total = sum(cm_sex.ravel())
tp_s_percent = (tp_sex / total) * 100
tn_s_percent = (tn_sex / total) * 100
fp_s_percent = (fp_sex / total) * 100
fn_s_percent = (fn_sex / total) * 100

print("Sex Metrics")
print("True Positives:", tp_sex)
print("True Negatives:", tn_sex)
print("False Positives:", fp_sex)
print("False Negatives:", fn_sex)
print("True Positive Percentage:", tp_s_percent)
print("True Negative Percentage:", tn_s_percent)
print("False Positive Percentage:", fp_s_percent)
print("False Negative Percentage:", fn_s_percent)


# # Preprocess test data
# test_preprocessed = preprocessor.transform(test.drop('participant_id', axis=1, errors='ignore'))

# # Make predictions
# test_adhd_pred = adhd_model.predict(test_preprocessed)
# test_sex_pred = sex_model.predict(test_preprocessed)

# # Create submission file
# submission = pd.DataFrame({
#     'participant_id': test['participant_id'],
#     'ADHD_Outcome': test_adhd_pred,
#     'Sex_F': test_sex_pred
# })

# # Save submission file
# submission.to_csv('/kaggle/working/submission.csv', index=False)
# print("Submission file saved!")

