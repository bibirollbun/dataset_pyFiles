from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split

#metrics
from sklearn.metrics import log_loss,make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

import lightgbm as lgb
import xgboost as xgb
# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import StratifiedKFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate,StratifiedShuffleSplit

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,LabelBinarizer

from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# GridSearchCV
from sklearn.model_selection import GridSearchCV


!pip install charliepy modelviz --quiet --upgrade


#Load training and test data into respective dataframes
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df =  pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


len(train_df)


len(test_df)


train_df.columns


#check out some lines from the training 
train_df.head(10)


test_df.head(10)


#Data clean up required
#remove all NaNs with column means

# Fill NaNs in all numeric columns with their respective means
# First lets do this for numeric data
for col in train_df.select_dtypes(include=np.number).columns:
    train_df.fillna({col:train_df[col].mean()}, inplace=True)
# Next lets do this for boolean data
for col in train_df.select_dtypes(include=object).columns:
    print( col)
    train_df.fillna({col:train_df[col].mode()[0]}, inplace=True)

#repeat above for Test data
# First lets do this for numeric data
for col in test_df.select_dtypes(include=np.number).columns:
    test_df.fillna({col:test_df[col].mean()}, inplace=True)
# Next lets do this for boolean data
for col in test_df.select_dtypes(include=object).columns:
    print( col)
    test_df.fillna({col:test_df[col].mode()[0]}, inplace=True)


train_df.describe()


full_df = pd.concat([train_df, test_df])


summary_df = full_df.describe()
summary_df


# Set thresholds and means/stds based on your summary
alone_75 = summary_df.loc['75%', "Time_spent_Alone"]
social_75 = summary_df.loc["75%", "Social_event_attendance"]
friends_75 = summary_df.loc['75%', "Friends_circle_size"]
alone_median = summary_df.loc["50%", "Time_spent_Alone"]


mean_alone, std_alone = summary_df.loc['mean', "Time_spent_Alone"], summary_df.loc['std', "Time_spent_Alone"]
mean_social, std_social = summary_df.loc["mean", "Social_event_attendance"], summary_df.loc["std", "Social_event_attendance"]
mean_outside, std_outside = summary_df.loc['mean', "Going_outside"], summary_df.loc['std', "Going_outside"]
mean_friends, std_friends = summary_df.loc['mean', "Friends_circle_size"], summary_df.loc['std', "Friends_circle_size"]
mean_post, std_post = summary_df.loc['mean', "Post_frequency"], summary_df.loc['std', "Post_frequency"]


def add_derived_columns(df):
    df['alone_to_social_ratio'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['outside_per_friend'] = df['Going_outside'] / (df['Friends_circle_size'] + 1)
    df['posts_per_friend'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    
    df['is_social_butterfly'] = ((df['Social_event_attendance'] >= social_75) | (df['Friends_circle_size'] >= friends_75)).astype(int)
    df['is_homebody'] = (df['Time_spent_Alone'] >= alone_75).astype(int)
    
    df['active_social_score'] = df['Social_event_attendance'] * df['Going_outside']
    df['social_engagement_index'] = (df['Social_event_attendance'] + df['Post_frequency']) * df['Friends_circle_size']
    
    df['alone_time_deviation'] = df['Time_spent_Alone'] - alone_median
    
    df['z_time_spent_alone'] = (df['Time_spent_Alone'] - mean_alone) / std_alone
    df['z_social_event_attendance'] = (df['Social_event_attendance'] - mean_social) / std_social
    df['z_going_outside'] = (df['Going_outside'] - mean_outside) / std_outside
    df['z_friends_circle_size'] = (df['Friends_circle_size'] - mean_friends) / std_friends
    df['z_post_frequency'] = (df['Post_frequency'] - mean_post) / std_post

    return df

# Apply to both train and test
train_df = add_derived_columns(train_df)
test_df = add_derived_columns(test_df)



def assign_social_group(row, se_75, fcs_75, tsa_75, se_25, fcs_25, pf_25):
    if row['Social_event_attendance'] >= se_75 and row['Friends_circle_size'] >= fcs_75:
        return 'Social Butterfly'
    elif row['Time_spent_Alone'] >= tsa_75:
        return 'Homebody'
    elif (row['Social_event_attendance'] <= se_25 and
          row['Friends_circle_size'] <= fcs_25 and
          row['Post_frequency'] <= pf_25):
        return 'Lurker'
    else:
        return 'Average Joe'


se_75 = full_df['Social_event_attendance'].quantile(0.75)
fcs_75 = full_df['Friends_circle_size'].quantile(0.75)
tsa_75 = full_df['Time_spent_Alone'].quantile(0.75)
se_25 = full_df['Social_event_attendance'].quantile(0.25)
fcs_25 = full_df['Friends_circle_size'].quantile(0.25)
pf_25 = full_df['Post_frequency'].quantile(0.25)


train_df['Social_Group'] = train_df.apply(
    lambda row: assign_social_group(row, se_75, fcs_75, tsa_75, se_25, fcs_25, pf_25), axis=1
)
test_df['Social_Group'] = test_df.apply(
    lambda row: assign_social_group(row, se_75, fcs_75, tsa_75, se_25, fcs_25, pf_25), axis=1
)


group_stats = train_df.groupby('Social_Group').agg({
    'Time_spent_Alone': 'mean',
    'Social_event_attendance': 'mean',
    'Going_outside': 'mean',
    'Friends_circle_size': 'mean',
    'Post_frequency': 'mean',
    # add more if you want!
})
print(group_stats)



import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='Social_Group', y='Time_spent_Alone', data=train_df)
plt.show()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df['Social_Group_Encoded'] = le.fit_transform(train_df['Social_Group'])
test_df['Social_Group_Encoded'] = le.transform(test_df['Social_Group'])


train_df.drop(["Social_Group"], axis=1, inplace=True)
test_df.drop(["Social_Group"], axis=1, inplace=True)


numeric_features = train_df.columns[train_df.dtypes == np.number]


numeric_features


X_train = train_df.copy()


X_train.columns


X_train['Stage_fear'] = train_df['Stage_fear'].map({'Yes': 1, 'No': 0})
X_train['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
# Drop personality and id from training data
X_train = X_train.drop(columns='Personality')
X_train = X_train.drop(columns='id')


X_train.head(20)


len(X_train)


X_test = test_df.copy()


X_test


X_test['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
X_test['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
# Drop personality and id from testing  data
X_test = X_test.drop(columns='id')


len(X_test)


y_train = train_df['Personality'].map({'Introvert': 1, 'Extrovert': 0})
y_train = pd.DataFrame(data=y_train)
y_train.head(5)


len(y_train)


from sklearn.model_selection import train_test_split

X_train_red, X_valid, y_train_red, y_valid = train_test_split(
    X_train, y_train, test_size = 0.1, random_state=42
)


print(f'The length of the reduced set is: {len(X_train_red)}.\n\nThe length of the original training is: {len(X_train)}.')


from charlie.models.ensemble import CHARLIE
import charlie
charlie.__version__


import numpy as np
import random
from tqdm import tqdm 

def random_search_charlie(
    X_tr, y_tr, X_val, y_val,
    param_dist,
    n_search=10,
    metric_fn=None,
    maximize_metric=True,
    verbose=True,
    charlie_verbose=False, 
    classification=True
):
    """
    Random search for CHARLIE model hyperparameters (train/val sets provided).
    Optimizes for *any* metric function (score or loss).

    Args:
        X_tr, y_tr: training data (NumPy arrays)
        X_val, y_val: validation data (NumPy arrays)
        param_dist: dict of parameter lists, e.g. {'rf_trees': [100, 200], ...}
        n_search: number of random samples to try
        metric_fn: function (y_true, y_prob) -> score or loss.
        maximize_metric: True if higher metric is better, False if lower is better.
        verbose: print progress

    Returns:
        best_model: trained CHARLIE model
        best_score: best metric value
        best_params: params of best model
        all_results: list of {'params', 'score'}
    """
    if metric_fn is None:
        from sklearn.metrics import roc_auc_score
        metric_fn = roc_auc_score

    best_score = None
    best_params = None
    best_model = None
    all_results = []

    # Add tqdm progress bar
    iterator = tqdm(range(n_search), desc="Random Search", unit="trial") if verbose else range(n_search)

    for i in iterator:
        # Randomly sample params
        params = {k: random.choice(v) for k, v in param_dist.items()}
        if verbose:
            print(f"Trial {i+1}/{n_search}: {params}")

        charlie = CHARLIE(
            input_dim=X_tr.shape[1],
            selected_features=params['selected_features'],
            rf_trees=params['rf_trees'],
            hidden_layers=params['hidden_layers'],
            classification=classification,
            logging_enabled=charlie_verbose,
            progress_bar=charlie_verbose
        )
        
        train_keys = ['epochs', 'lr'] 
        train_params = {k: v for k, v in params.items() if k in train_keys}
        charlie.train_model(X_tr, y_tr, **train_params)

        # Prediction
        probs = charlie.predict(X_val)
        # If output is logits, convert to probabilities as needed
        if hasattr(probs, 'shape') and len(probs.shape) == 2 and probs.shape[1] == 1:
            from scipy.special import expit
            probs_np = expit(probs).flatten()
        elif hasattr(probs, 'shape') and len(probs.shape) == 2 and probs.shape[1] == 2:
            # Softmax, prob of class 1
            probs_exp = np.exp(probs)
            probs_softmax = probs_exp / np.sum(probs_exp, axis=1, keepdims=True)
            probs_np = probs_softmax[:, 1].flatten()
        else:
            probs_np = np.asarray(probs).flatten()

        metric = metric_fn(np.array(y_val).flatten(), probs_np)
        all_results.append({'params': params, 'score': metric})

        if verbose:
            print(f"Metric: {metric:.4f}")

        if best_score is None or \
            (maximize_metric and metric > best_score) or \
            (not maximize_metric and metric < best_score):
            best_score = metric
            best_params = params
            best_model = charlie

    if verbose:
        print("\nBest metric:", best_score)
        print("Best parameters:", best_params)

    return best_model, best_score, best_params, all_results



col_len = int(len(train_df.columns))


# param_dist = {
#     'selected_features': [5, 10, col_len],
#     'rf_trees': [100, 200, 300],
#     'hidden_layers': [
#         [256, 128, 64],       
#         [256, 128, 64, 32],   
    
#     ],
#     'epochs': [50, 1000, 1500],
#     'lr': [0.001, 0.005]}



param_dist = {
    'selected_features': [19],
    'rf_trees': [200],
    'hidden_layers': [[128, 64, 32]],
    'epochs': [50],
    'lr': [0.005]
}



y_tr = np.ravel(np.array(y_train_red))
y_val = np.ravel(np.array(y_valid))



%%capture
best_model, best_score, best_params, all_results = random_search_charlie(
    np.array(X_train_red), y_tr, np.array(X_valid), y_val,
    param_dist=param_dist,
    n_search=10,
    metric_fn=None,  
    maximize_metric=True,
    verbose=True,
    charlie_verbose=False)



best_params


best_model


best_score


probs = best_model.predict(np.array(X_valid))
pos_val_probs = probs[:,1]


from modelviz.roc import plot_roc_curve_with_youdens_thresholds
def roc_stats_and_plot(
    y_true, y_probs, 
    theoretical_threshold=None, 
    model_name="Model",
    plot_func=None
):
    """
    Calculate ROC stats, Youden's J, plot ROC with both theoretical and Youden's thresholds,
    and return all metrics in a dictionary.

    Parameters:
        y_true: array-like, true labels
        y_probs: array-like, predicted probabilities (for positive class)
        theoretical_threshold: float or None, user-selected threshold to compare
        model_name: str, for plot title
        plot_func: callable, function to plot ROC curve (must accept required params)

    Returns:
        dict with ROC stats, thresholds, indices, and AUC
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    import numpy as np

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    youden_threshold = thresholds[best_idx]

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Youden's J best threshold: {youden_threshold:.4f} at index {best_idx}")
    if theoretical_threshold is not None:
        print(f"Theoretical threshold: {theoretical_threshold:.4f}")

    # Plot ROC if plotting function provided
    if plot_func is not None:
        plot_func(
            fpr, tpr, thresholds, roc_auc=roc_auc,
            model_name=model_name,
            adjusted_threshold=theoretical_threshold,
            youden_threshold=youden_threshold
        )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
        "youden_j": youden_j,
        "youden_best_idx": best_idx,
        "youden_threshold": youden_threshold,
        "theoretical_threshold": theoretical_threshold
    }



stats = roc_stats_and_plot(
    y_true=y_val, 
    y_probs=pos_val_probs,
    theoretical_threshold=0.5,  
    model_name='Introvert/Extrovert CHARLIE',
    plot_func=plot_roc_curve_with_youdens_thresholds
)


test_probs = best_model.predict(np.array(X_test))
test_probs = test_probs[:,1]

if len(test_probs) != len(X_test):
    raise ValueError('Size of the predicted probabilities does not match source dataset')


youdens_thresh = stats.get('youden_threshold')
y_pred_youden = (test_probs >= youdens_thresh).astype(int)
len(y_pred_youden)


y_pred = pd.DataFrame(y_pred_youden)


best_params = {
    'n_estimators': 1013,
    'max_depth': 3,
    'learning_rate': 0.04473761810915283,
    'subsample': 0.7472021066686094,
    'colsample_bytree': 0.6526442450606929,
    'gamma': 4.987525774261538,
    'reg_lambda': 0.1016293050091594,
    'reg_alpha': 0.8381641826774137,
    'min_child_weight': 10,
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'device': 'cuda',
    'random_state': 42
}

model = xgb.XGBClassifier(**best_params)


from sklearn.metrics import accuracy_score
models, scores = [], []
best_score = -np.inf
best_model = None

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = X_train_red
y = y_tr

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n[INFO] Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"[INFO] Accuracy: {acc:.6f}")
    models.append(model)
    scores.append(acc)
    
    # Track best model by accuracy (or change to your favorite metric)
    if acc > best_score:
        best_score = acc
        best_model = model

print("\n[INFO] Mean CV Accuracy:", np.mean(scores))
print("[INFO] Best CV Accuracy:", best_score)


plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), scores, color='skyblue')
plt.title("Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i+1 - 0.15, v + 0.01, f"{v:.2f}", fontweight='bold')
plt.show()


xgboost_test_preds = best_model.predict(X_test) 
xgboost_test_probs = best_model.predict_proba(X_test)[:, 1]
xgboost_test_ythres_label = (xgboost_test_probs >= youdens_thresh).astype(int)


xgboost_test_ythres_label


from sklearn.metrics import roc_auc_score, make_scorer
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)


from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb

estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=7)),
    ('gbc', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=100, random_state=42)),
]



from sklearn.linear_model import LogisticRegression
meta_learner = LogisticRegression(max_iter=1000, random_state=42)



from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5,
    passthrough=False,
    n_jobs=-1
)


from charlie.


from sklearn.metrics import roc_auc_score
rfe_cv_hyb = HybridRFECV(
    estimator=stacking, 
    step=1, 
    cv=5, 
    scoring=roc_auc_score, 
    min_features_to_select=2, verbose=1)


rfe_cv_hyb.fit(X, y)


X_val_reduced = rfe_cv_hyb.transform(X_test)
stack_proba = rfe_cv_hyb.best_model_.predict_proba(X_val_reduced)[:,1]


stack_proba


import lightgbm as lgb
lgbm_estimator = lgb.LGBMClassifier()



from sklearn.metrics import roc_auc_score
rfecv_lgbm = HybridRFECV(
    estimator=lgbm_estimator,
    cv=10,
    step=1,
    scoring=roc_auc_score, 
    min_features_to_select=1,
    verbose=2,
    random_state=42
)


rfecv_lgbm.fit(X, y)


support_mask = rfecv_lgbm.best_support_


selected_columns = X_train.columns[support_mask]
print("Selected features:", list(selected_columns))


# Reduce your test data to selected features
lgm_test_reduced = rfecv_lgbm.transform(X_test)

# Predict probabilities with the best model found by RFECV
lgm_probs = rfecv_lgbm.best_model_.predict_proba(lgm_test_reduced)[:, 1]



charlie_probs = test_probs
xgboost_probs = xgboost_test_probs

probs_df = pd.DataFrame({
    "CHARLIE_Probs": charlie_probs,
    "XGBoost_Probs": xgboost_probs,
    "Stacked_Probs": stack_proba,
    "LGBM_Probs": lgm_probs
})


from modelviz.relationships import plot_model_probs_scatter
import modelviz
modelviz.__version__


prob_pairs = [
    (charlie_probs, xgboost_probs),
    (stack_proba, xgboost_probs),
    (stack_proba, charlie_probs)
]

labels = ['CHARLIE vs XGB', 'Stacked vs XGB', "Stacked vs CHARLIE"]
colors = ['grey', 'black', 'blue']
markers = ['o', 'o', 'o']


plot_model_probs_scatter(probs_pairs=prob_pairs,
                         labels=labels, 
                         colors=colors, 
                         markers=markers, 
                         x_thresh=youdens_thresh, y_thresh=youdens_thresh,
                         title="Model Probability comparison", sigmoid_color='black',
                         x_thresh_color='#212121', y_thresh_color='#212121', 
                         alpha= 0.6, show_legend=True)


weights = [0.1, 0.1, 0.1, 0.7]
probs_list = [xgboost_probs, charlie_probs, stack_proba, lgm_probs]


from charlie.utils import weighted_ensemble
weight_probs = weighted_ensemble(probs_list, weights=weights)


weight_labels = (weight_probs >= youdens_thresh).astype(int)


weight_labels


lgm_labels = (lgm_probs >= youdens_thresh).astype(int)


temp = pd.DataFrame(pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")['id'])
temp['Personality'] = lgm_labels#y_pred
temp['Personality'] = temp['Personality'].map({1:'Introvert',0:'Extrovert'})


temp


temp.to_csv("../working/submission.csv", index = False)

