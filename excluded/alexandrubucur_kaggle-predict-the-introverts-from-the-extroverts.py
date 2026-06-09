# Standard data manipulation and numerical computing
import pandas as pd
import numpy as np
from collections import Counter

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Model selection and evaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, learning_curve, RandomizedSearchCV

# Metrics for model evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, accuracy_score, auc

# Machine learning models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import xgboost as xgb
from scipy.stats import uniform, randint, loguniform
import lightgbm as lgb
import joblib
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


path = '/kaggle/input/playground-series-s5e7/' # path for data files
models_path = './' # path for saved models and saved preprocessing parameters


train = pd.read_csv(path + 'train.csv')
test  = pd.read_csv(path + 'test.csv')


train.head()


train.describe()


# drop outliers ouside 1.5 * interquartile range
def detect_outliers(df,n,features):
    outlier_indices = []

    for col in features:
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k,v in outlier_indices.items() if v > n)

    return multiple_outliers

Otliers_to_drop = detect_outliers(train,2,['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency'])


print(len(Otliers_to_drop))


# concatenate train and test data into a single dataframe for better processing
train_len = len(train)
dataset = pd.concat(objs=[train,test], axis=0)
dataset = dataset.drop('id', axis=1).reset_index(drop=True)
print(f"Shape of total dataset: {dataset.shape}\n")
print(dataset.tail())



fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 6))

# Bar plot for Time_spent_Alone
ax1.bar(np.arange(train['Time_spent_Alone'].max() + 1), train['Time_spent_Alone'].value_counts())
ax1.set_xlabel('Time Spent Alone')
ax1.set_ylabel('Count')

# Bar plot for Social_event_attendance
ax2.bar(np.arange(train['Social_event_attendance'].max() + 1), train['Social_event_attendance'].value_counts())
ax2.set_xlabel('Social Event Attendance')
ax2.set_ylabel('Count')

# Bar plot for Going_outside
ax3.bar(np.arange(train['Going_outside'].max() + 1), train['Going_outside'].value_counts())
ax3.set_xlabel('Going Outside')
ax3.set_ylabel('Count')

# Bar plot for Friends_circle_size
ax4.bar(np.arange(train['Friends_circle_size'].max() + 1), train['Friends_circle_size'].value_counts())
ax4.set_xlabel('Friends Circle Size')
ax4.set_ylabel('Count')

# Bar plot for Post_frequency
ax5.bar(np.arange(train['Post_frequency'].max() + 1), train['Post_frequency'].value_counts())
ax5.set_xlabel('Post Frequency')
ax5.set_ylabel('Count')

# Hide the last subplot
ax6.set_visible(False)

# Adjust layout to prevent overlap
plt.tight_layout()


counts = train['Personality'].value_counts()
print(counts)


sns.countplot(data=train, x='Personality')


train['Personality'] = train['Personality'].map({'Extrovert':1, 'Introvert':0}) # map labels


g = sns.heatmap(train[['Personality','Time_spent_Alone','Social_event_attendance','Going_outside',
                      'Friends_circle_size','Post_frequency']].corr(), cmap='coolwarm',annot=True)


g = sns.barplot(data=train, x="Stage_fear", y="Personality", palette="muted")


g = sns.barplot(data=train, x="Drained_after_socializing", y="Personality", palette="muted")


dataset.fillna(np.nan) # consistent values
print(dataset.isnull().sum())


imputer = SimpleImputer(strategy="median")


y_train = train['Personality']
X_train_num = train.select_dtypes(include=[np.number]).drop(['id','Personality'], axis=1, errors='ignore') # numerical features only
dataset_num = dataset[X_train_num.columns]


imputer.fit(X_train_num) # only on train data


file_path = models_path + 'imputer.pkl'
if os.path.exists(file_path):
    print(f"Imputer already saved at {file_path}")
else:
    joblib.dump(imputer, file_path)
    print(f"Imputer saved at {file_path}")



X = imputer.transform(dataset_num) # input on all data: train + test


dataset_num_imputed = pd.DataFrame(X, columns=dataset_num.columns)
train_num_imputed = dataset_num_imputed[:len(train)] # used later for training a scaler
len(train_num_imputed)


dataset_num_imputed.isna().sum()


X_train_cat = train.select_dtypes(include=[object]).drop(['Personality'], axis=1, errors='ignore')
dataset_cat = dataset[X_train_cat.columns]


dataset_cat.head() # verifying we selected the correct features


print(X_train_cat.isna().sum())
print(dataset_cat.isna().sum())


imputer_cat = SimpleImputer(strategy="most_frequent")


imputer_cat.fit(X_train_cat) # only on train data


file_path = models_path + 'imputer_cat.pkl' # imputer needed in production
if os.path.exists(file_path):
    print(f"Imputer already saved at {file_path}")
else:
    joblib.dump(imputer_cat, file_path)
    print(f"Imputer saved at {file_path}")


X_cat = imputer_cat.transform(dataset_cat) # on all data


dataset_cat_imputed = pd.DataFrame(X_cat, columns=dataset_cat.columns)


dataset_cat_imputed.head()


dataset_cat_imputed.isna().sum()


cat_encoder = OneHotEncoder()
dataset_cat_1hot = cat_encoder.fit_transform(dataset_cat_imputed)


file_path = models_path + '1hot_encoder.pkl' # needed in production
if os.path.exists(file_path):
    print(f"1HotEncoder already saved at {file_path}")
else:
    joblib.dump(cat_encoder, file_path)
    print(f"1HotEncoder saved at {file_path}")


#build dataframe from the sparse matrix, with new column names generated based on column_name__value
dataset_cat_1hot = pd.DataFrame(dataset_cat_1hot.toarray(),columns=cat_encoder.get_feature_names_out())


dataset_cat_1hot.head()


scaler = StandardScaler()


scaler.fit(train_num_imputed) # only on train data


file_path = models_path + 'scaler.pkl' # needed in production
if os.path.exists(file_path):
    print(f"Scaler already saved at {file_path}")
else:
    joblib.dump(scaler, file_path)
    print(f"1HotEncoder saved at {file_path}")


X_fit = scaler.transform(dataset_num_imputed) # on all data


dataset_num_std_scaled = pd.DataFrame(X_fit, columns=dataset_num_imputed.columns)


dataset_num_std_scaled.head()


dataset_final = pd.DataFrame(pd.merge(dataset_num_std_scaled,dataset_cat_1hot, left_index=True, right_index=True)) # for better processing


print(dataset_final.shape)


y_train


X_train = dataset_final[:len(train)]
X_test = dataset_final[len(train):]
Y_train = y_train


kfold = StratifiedKFold(n_splits=5) # between 5 and 10 is generally used; I chose 5 for training time performance


random_state = 42
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

# Corrected barplot call
g = sns.barplot(x="CrossValMeans", y="Algorithm", data=cv_res, palette="Set3", orient="h",
                order=cv_res.sort_values("CrossValMeans", ascending=False)["Algorithm"])
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


cv_res


### SVC classifier
file_path = models_path + 'rsSVMC.pkl'

if os.path.exists(file_path):
    rsSVMC = joblib.load(file_path)
    print(f"Successfully loaded {file_path}")
else:
    print(f"Model not found at {file_path}, computing algorithm")
    
    SVMC = SVC(probability=False) #probability=True takes more time to train; not necessary here

    # Define parameter distributions for RandomizedSearchCV
    svc_param_distributions = {'kernel': ['rbf', 'linear'], #use linear also, if data is linearly separable
                            'gamma': loguniform(1e-4, 1), # explore a large range of values, as gamma (and C) can be nonlinear
                            'C': loguniform(1e-1, 100)} 

    # Use RandomizedSearchCV instead of GridSearchCV
    rsSVMC = RandomizedSearchCV(SVMC, param_distributions=svc_param_distributions,
                                n_iter=20, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1, random_state=42)

    rsSVMC.fit(X_train,Y_train)

    SVMC_best = rsSVMC.best_estimator_

    # Best score
    rsSVMC.best_score_

    joblib.dump(rsSVMC,file_path)
    if os.path.exists(file_path):
        print(f"Successfully wrote {file_path}")
    else:
        print(f"Error: Could not save model at {file_path}")


SVMC_best=rsSVMC.best_estimator_
SVMC_best


### MLP classifier
file_path = models_path + 'rsMLP.pkl'

if os.path.exists(file_path):
    rsMLP = joblib.load(file_path)
    print(f"Successfully loaded {file_path}")
else:
    print(f"Model not found at {file_path}, computing algorithm")
    MLP = MLPClassifier() # can't use early stopping as this causes numerical instability

    # Define parameter distributions for RandomizedSearchCV
    mlp_param_distributions = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (50, 50, 50)], 
        'activation': ['relu', 'tanh', 'logistic'], #tanh and logistic may improve performance in binary classification
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive'], 
        'learning_rate_init': loguniform(1e-4, 1e-2),
        'alpha': loguniform(1e-4, 1e-2), 
        'batch_size': [32, 64, 128, 256], #explore larger batch sizes to reduce training time
        'max_iter': randint(200, 400) 
    }

    # Use RandomizedSearchCV instead of GridSearchCV
    rsMLP = RandomizedSearchCV(MLP, param_distributions=mlp_param_distributions, n_iter=30, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1, random_state=42)

    rsMLP.fit(X_train,Y_train)

    MLP_best = rsMLP.best_estimator_

    # Best score
    rsMLP.best_score_

    joblib.dump(rsMLP,file_path)
    if os.path.exists(file_path):
        print(f"Successfully wrote {file_path}")
    else:
        print(f"Error: Could not save model at {file_path}")


MLP_best=rsMLP.best_estimator_
MLP_best


# Logistic regression
file_path = models_path + 'rsLR.pkl'

if os.path.exists(file_path):
    rsLR = joblib.load(file_path)
    print(f"Successfully loaded {file_path}")
else:
    print(f"Model not found at {file_path}, computing algorithm")
    LR = LogisticRegression()

    # To handle the penalty/solver compatibility with RandomizedSearchCV,
    # we can either manually specify valid combinations or use a solver that supports
    # the desired penalties. 'saga' supports 'l1', 'l2', and 'elasticnet'.
    # 'liblinear' supports 'l1' and 'l2'.

    # Let's define separate distributions for solvers to ensure compatibility
    lr_param_distributions = [
        {
            'penalty': ['l1', 'l2'],
            'C': loguniform(1e-3, 1e2),
            'solver': ['liblinear'],
        },
        {
            'penalty': ['l2'],
            'C': loguniform(1e-3, 1e2),
            'solver': ['lbfgs'],  # the default solver, often faster for smaller datasets
        },
        {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': loguniform(1e-3, 1e2),
            'solver': ['saga'],
            'l1_ratio': uniform(0, 1),  # Only used for elasticnet penalty
        }
    ]


    # Use RandomizedSearchCV instead of GridSearchCV
    # n_iter: Number of parameter settings that are sampled. Increase for a more exhaustive search.
    rsLR = RandomizedSearchCV(LR, param_distributions=lr_param_distributions, n_iter=40, cv=kfold,
                            scoring="accuracy", n_jobs=-1, verbose=1, random_state=42)

    rsLR.fit(X_train,Y_train)

    LR_best = rsLR.best_estimator_

    # Best score
    rsLR.best_score_

    joblib.dump(rsLR,file_path)
    if os.path.exists(file_path):
        print(f"Successfully wrote {file_path}")
    else:
        print(f"Error: Could not save model at {file_path}")


LR_best =  rsLR.best_estimator_
LR_best


### xgboost 
file_path = models_path + 'rsXGB.pkl' 

if os.path.exists(file_path):
    rsXGB = joblib.load(file_path)
    print(f"Successfully loaded {file_path}")
else:
    print(f"Model not found at {file_path}, computing algorithm")
    XGB = xgb.XGBClassifier(objective='binary:logistic', # Binary classification
                            eval_metric='logloss',       # Evaluation metric
                            # use_label_encoder=False,     # Deprecated, set to False
                            random_state=42)

    # Define parameter distributions for RandomizedSearchCV for XGBoost
    xgb_param_distributions = {
        'n_estimators': randint(50, 400),  # narrowed for efficiency, use early stopping
        'learning_rate': uniform(0.01, 0.19), # focus on stable learning rates
        'max_depth': randint(3, 7),         # conservative for low feature count
        'min_child_weight': randint(1, 15),  # Minimum sum of instance weight (hessian) needed in a child, extended to prevent overfitting
        'gamma': uniform(0, 1),            # extended for stricter splits
        'subsample': uniform(0.5, 0.5),      # more aggresive subsampling
        'colsample_bytree': uniform(0.5, 0.5),# adjusted for small feature set
        'reg_alpha': loguniform(1e-3, 10),          # L1 regularization term on weights
        'reg_lambda': loguniform(1e-3, 10),         # L2 regularization term on weights
        }


    # Use RandomizedSearchCV instead of GridSearchCV
    # n_iter: Number of parameter settings that are sampled. Increase for a more exhaustive search.
    rsXGB = RandomizedSearchCV(XGB, param_distributions=xgb_param_distributions, n_iter=50, cv=kfold,
                            scoring="roc_auc", n_jobs=-1, verbose=1, random_state=42)

    rsXGB.fit(X_train,Y_train)

    XGB_best = rsXGB.best_estimator_

    # Best score
    rsXGB.best_score_

    joblib.dump(rsXGB,file_path)
    if os.path.exists(file_path):
        print(f"Successfully wrote {file_path}")
    else:
        print(f"Error: Could not save model at {file_path}")


XGB_best =  rsXGB.best_estimator_
XGB_best



# LightGBM
file_path = models_path + 'rsLGBM.pkl'

if os.path.exists(file_path):
    rsLGBM = joblib.load(file_path)
    print(f"Successfully loaded {file_path}")
else:
    print(f"Model not found at {file_path}, computing algorithm")

    lgbm_param_distributions = {
        'n_estimators': randint(50, 500),  # narrowed for efficiency
        'learning_rate': uniform(0.01, 0.19),  # matches xgboost for stable learning
        'num_leaves': randint(20, 70),  # reduced to prevent overfitting with 7 features
        'max_depth': randint(3, 10),  # shallower trees for low dimensionality
        'min_child_samples': randint(10, 100),  # extended for robust leaves
        'subsample': uniform(0.5, 0.5),  # matches xgboost for robustness
        'colsample_bytree': uniform(0.5, 0.5),  # matches xgboost for small feature set
        'reg_alpha': loguniform(1e-3, 10),  # log scale for sparsity
        'reg_lambda': uniform(1e-3, 10),  # log scale for smoothing
        'min_split_gain': uniform(0, 2)  # matches xgboost's gamma for split control
    }
    
    # Initialize LightGBM Classifier
    LGBM = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', 
                              random_state=42, verbose=-1)

    # Use RandomizedSearchCV with AUC scoring
    rsLGBM = RandomizedSearchCV(
        estimator=LGBM, 
        param_distributions=lgbm_param_distributions, 
        n_iter=20, 
        cv=kfold,
        scoring="roc_auc",  # Changed to AUC
        n_jobs=-1, 
        verbose=1, 
        random_state=42
    )

    # Fit the model with early stopping
    rsLGBM.fit(
        X_train, 
        Y_train,
        eval_set=[(X_train, Y_train)], 
        eval_metric='auc',  # Already set to AUC, which is fine
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    # Get the best estimator
    LGBM_best = rsLGBM.best_estimator_

    # Print best parameters and score
    print("Best Parameters:", rsLGBM.best_params_)
    print("Best Cross-Validation AUC:", rsLGBM.best_score_)

    # Save the model
    joblib.dump(rsLGBM, file_path)
    if os.path.exists(file_path):
        print(f"Successfully wrote {file_path}")
    else:
        print(f"Error: Could not save model at {file_path}")


LGBM_best = rsLGBM.best_estimator_
LGBM_best


test_Extrovert_MLP = pd.Series(MLP_best.predict(X_test), name="MLP")
test_Extrovert_LR = pd.Series(LR_best.predict(X_test), name="LR")
test_Extrovert_SVMC = pd.Series(SVMC_best.predict(X_test), name="SVMC")
test_Extrovert_XGB = pd.Series(XGB_best.predict(X_test), name="XGB")
test_Extrovert_LGBM = pd.Series(LGBM_best.predict(X_test), name="LGBM")

# Concatenate all classifier results
ensemble_results = pd.concat([test_Extrovert_MLP,test_Extrovert_LR,test_Extrovert_SVMC,test_Extrovert_XGB, test_Extrovert_LGBM],axis=1)

mapping = {1:1, 0:0, 'Extrovert':1, 'Introvert':0}
ensemble_results = ensemble_results.apply(lambda x: x.map(mapping))


g= sns.heatmap(ensemble_results.corr(),annot=True)
# mabe too much similariy


file_path = models_path + 'ensemble.pkl'

# Check if the model already exists
if os.path.exists(file_path):
    stackingC = joblib.load(file_path)
    print(f"Successfully loaded {file_path}")
else:
    print(f"Model not found at {file_path}, computing algorithm")

    # Ensure Y_train is numeric (0 for Introvert, 1 for Extrovert)
    mapping = {1:1, 0:0,'Extrovert': 1, 'Introvert': 0}
    Y_train_mapped = Y_train.map(mapping) if isinstance(Y_train, pd.Series) else pd.Series(Y_train).map(mapping)

    # Define the stacking classifier
    stackingC = StackingClassifier(
        estimators=[
            ('mlp', MLP_best),
            ('lr', LR_best),
            ('svc', SVMC_best),
            ('xgb', XGB_best),
            ('lgbm', LGBM_best)
        ],
        final_estimator=LogisticRegression(),
        n_jobs=-1
    )

    # Train the stacking classifier with mapped Y_train
    stackingC = stackingC.fit(X_train, Y_train_mapped)

    # Save the trained stacking classifier
    joblib.dump(stackingC, file_path)

    # Verify that the model has been saved
    if os.path.exists(file_path):
        print(f"Successfully saved the ensemble model to {file_path}")
    else:
        print(f"Error: Could not save the ensemble model at {file_path}")


stackingC


print(X_train)


# Get cross-validated predictions for the stacking classifier
Y_train_pred_stacking = cross_val_predict(stackingC, X_train, Y_train, cv=kfold)

# Calculate metrics for the stacking classifier
accuracy_stacking = accuracy_score(Y_train, Y_train_pred_stacking)
precision_stacking = precision_score(Y_train, Y_train_pred_stacking, pos_label=1)
recall_stacking = recall_score(Y_train, Y_train_pred_stacking, pos_label=1)
f1_stacking = f1_score(Y_train, Y_train_pred_stacking, pos_label=1)
conf_matrix_stacking = confusion_matrix(Y_train, Y_train_pred_stacking)


print("Metrics for Stacking Classifier:")
print(f"Accuracy: {accuracy_stacking:.4f}")
print(f"Precision: {precision_stacking:.4f}")
print(f"Recall: {recall_stacking:.4f}")
print(f"F1-score: {f1_stacking:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix_stacking)


# Get cross-validated probability predictions for the positive class (Extrovert)
Y_scores_stacking = cross_val_predict(stackingC, X_train, Y_train, cv=kfold, method="predict_proba")[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(Y_train, Y_scores_stacking)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(Y_train, Y_scores_stacking)
pr_auc = auc(recall, precision)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


test_Extrovert = pd.Series(stackingC.predict(X_test), name="Personality")

test_Extrovert


X_test = X_test[[]]
# test_Extrovert.values
X_test = X_test.assign(Personality=test_Extrovert.values)
X_test.Personality.replace({0:'Introvert',1:'Extrovert'},inplace=True)


X_test.index.name = 'id'
X_test


X_test.to_csv(models_path + 'submission.csv')

