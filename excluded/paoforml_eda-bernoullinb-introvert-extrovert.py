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


#!ls -al | grep pandas


import warnings
warnings.filterwarnings('ignore')


#EDA libs
import matplotlib.pyplot as plt
import plotly.express as px

#model & metric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier



#load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train.head()


test =pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test.head()
test.shape


example_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
example_sub.shape


example_sub.head(2)


test.isnull().sum()


train.info()


train.describe()


train.columns


train.describe()


train['Personality'].value_counts().plot(kind='bar')


train['Personality'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)


train_not_NA = train.dropna(how='any').copy()


# dataset without any NULL values
train_not_NA['Personality'].value_counts().plot(kind='bar')


train_not_NA['Personality'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)


# Resample
g = train_not_NA.groupby('Personality')
data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop = True)))
data['Personality'].value_counts().plot(kind = "bar")
df_balance=data.reset_index(drop=True)


def make_dummies(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Returns a new DataFrame in which the specified columns are replaced
    by their one-hot (dummy) encodings.  All other columns are left intact.
    """
    return pd.get_dummies(df, columns=cols, drop_first=False)


#train_NA_balanced = data.reset_index(drop=True).copy()
train_not_NA.head()


cols_to_dummies = ['Stage_fear','Drained_after_socializing']

df_cleaned = make_dummies(train_not_NA, cols_to_dummies)
df_cleaned_balance = make_dummies(df_balance, cols_to_dummies)
df_cleaned.head(1)


plt.figure(figsize=(6, 4))
plt.violinplot( train_not_NA[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']], 
               positions=[1, 2,3,4,5], showmeans=True)
plt.xticks([1, 2,3,4,5] , ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size','Post_frequency'], rotation=45)   # rotated x-axis labels
plt.ylabel('value')
plt.title('Features in dataset without NULL values')
#plt.title('Violin plots of columns Time_spent_Alone and Social_event_attendance')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
plt.violinplot( df_cleaned_balance[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']], 
               positions=[1, 2,3,4,5], showmeans=True)
plt.xticks([1, 2,3,4,5] , ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size','Post_frequency'], rotation=45)   # rotated x-axis labels
plt.ylabel('value')
#plt.title('Violin plots of columns Time_spent_Alone and Social_event_attendance')
plt.tight_layout()
plt.title('Features in BALANCED dataset (NO null values)')
plt.show()


# Prepare Correlation Matrix
train_not_NA = make_dummies(train_not_NA, cols_to_dummies)
df = pd.get_dummies(train_not_NA, columns=['Personality'], prefix='personality')
df.head(1)


df.columns


ROI=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency', 'personality_Extrovert',
       'personality_Introvert', 'Stage_fear_No', 'Stage_fear_Yes',
       'Drained_after_socializing_No', 'Drained_after_socializing_Yes']
df = df[ROI]


corr_matrix = df.corr()
corr_matrix["personality_Extrovert"].sort_values(ascending = False)


corr_matrix["personality_Introvert"].sort_values(ascending = False)


# interactive heat-map
fig = px.imshow(
    corr_matrix,
    text_auto=".2f",         # show correlation values
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1
)
fig.update_layout(title="Correlation Matrix")
fig.show()


train.columns


ROI = ['Personality','Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size', 'Post_frequency']
mean_table= df_cleaned[ROI].groupby('Personality').agg({'Time_spent_Alone': 'mean', 'Social_event_attendance': 'mean',
                                       'Going_outside': 'mean','Friends_circle_size': 'mean','Post_frequency':'mean'})

mean_table


def imputation_by_pattern (df: pd.DataFrame,
                 col_null: str,
                cols_to_examinate,
                 mean_table: pd.DataFrame) -> pd.DataFrame:
    

    cols_to_examinate = [col for col in cols_to_examinate if col != col_null]

    # find all indexes of null values
    null_idx = df[df[col_null].isnull()].index.tolist()

    #iterate each row 
    for idx in null_idx:
        ex_count= 0
        in_count= 0
        for col in cols_to_examinate:
            if pd.isna(df.at[idx, col]):
                continue
            val = df.loc[idx,col]
            # check the distance from the mean Extrovert and Introvert
            mean_ex= round(mean_table.loc['Extrovert',col],2)
            mean_in = round(mean_table.loc['Introvert',col],2)
            if abs(val - mean_ex) > abs(val - mean_in):
                in_count+=1
            else:
                ex_count+=1
    
        if ex_count >= in_count:
            df.at[idx,col_null]=  mean_table.loc['Extrovert'][col_null]
        else:
            df.at[idx,col_null]=  mean_table.loc['Introvert'][col_null]
    
    return df[[col_null]]


df = df_cleaned.copy() #df_cleaned_balance


#encode Personality column
le = LabelEncoder()
df['personality_encoded'] = le.fit_transform(df['Personality'])


df.head(1)


ROI=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency','Stage_fear_No','Stage_fear_Yes',
       'Drained_after_socializing_No', 'Drained_after_socializing_Yes']

X = df[ROI]
y = df['personality_encoded'].values


# Train and split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for balanced classes
)


#MinMaxScaler is useful when the data has a bounded range or when the distribution is not Gaussian.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).copy()
X_val = scaler.transform(X_val).copy()


from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

# ------------------------------------------------------------------
# 1.  A tiny soft-voting meta-estimator
# ------------------------------------------------------------------
#
class NBForestEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble of:
        * BernoulliNB
        * RandomForestClassifier
    
    Each base model is trained on the *same* data (you can extend this
    to feature sub-spaces or stacking if you wish).
    """
    def __init__(self,
                 nb_alpha=1.0,
                 rf_n_estimators=200,
                 rf_max_depth = None,
                 voting_weight=None):
        """
        voting_weight : list or None
            If None -> [1, 1]  (equal weight).
            Otherwise [w_nb, w_rf].
        """
        self.nb_alpha = nb_alpha
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.voting_weight = voting_weight

    # --------------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse='csr')
        self.n_features_in_ = X.shape[1]

        # --- build the two base learners
        
        self.nb_ = BernoulliNB(alpha=self.nb_alpha)
      
        #bootstrap': True, 'class_weight': None, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200  
        self.rf_ = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            n_jobs=-1,
            min_samples_leaf=4,
            min_samples_split= 2,
            max_features='sqrt',
            class_weight= None, #'balanced',
            random_state=42
        )

        # --- fit them
        self.nb_.fit(X, y)
        self.rf_.fit(X, y)

        # --- remember classes
        self.classes_ = self.nb_.classes_
        return self

    # --------------------------------------------------------------
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')

        # probability matrices of shape (n_samples, 2)
        proba_nb = self.nb_.predict_proba(X)
        proba_rf = self.rf_.predict_proba(X)

        w = [1, 1] if self.voting_weight is None else self.voting_weight
        w = np.asarray(w) / np.sum(w)

        return w[0] * proba_nb + w[1] * proba_rf

    # --------------------------------------------------------------
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1))


clf = NBForestEnsemble(nb_alpha=0.5)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
import pandas as pd
'''

# ------------------------------------------------------------------
# 2. Define the parameter grid
# ------------------------------------------------------------------
param_grid = {
    "n_estimators":   [200, 400, 600],
    "max_depth":      [None, 3, 10, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":   ["sqrt", "log2", 0.5],   # 0.5 = 50 % of features
    "bootstrap":      [True, False],
    "class_weight":   [None, "balanced"]
}

# ------------------------------------------------------------------
# 3. Build the search
# ------------------------------------------------------------------
forest = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=forest,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",          # or  roc_auc"accuracy", "f1", etc.
    n_jobs=-1,
    verbose=2
)

# ------------------------------------------------------------------
# 4. Fit
# ------------------------------------------------------------------
grid.fit(X_train, y_train)

# ------------------------------------------------------------------
# 5. Inspect results
# ------------------------------------------------------------------
print("Best accuracy:", grid.best_score_)
print("Best params: ", grid.best_params_)
best_model = grid.best_estimator_

# Optional: evaluate on hold-out
#print("Hold-out ROC-AUC:", roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1]))

#-------------------- find the best --------------------------------------------------------------------------------------------------------------------------
#bootstrap': True, 'class_weight': None, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200


'''


from sklearn.svm import SVC

'''
clf = RandomForestClassifier ( n_estimators = 200,
            max_depth= 4,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42)

#BernoulliNB() 
#SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
#LogisticRegression()
'''

clf.fit(X_train, y_train)
#clf.class_prior_ = [0.2, 0.8]



y_pred = clf.predict(X_val)


'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=300,      # number of trees
    max_depth=None,        # let trees grow fully (or set an int)
    random_state=42,
    n_jobs=-1              # use all cores
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
'''


print("-------------------------------------------------------------------------")
print(f"The accuracy score is:   {accuracy_score(y_val,y_pred)}")
print("-------------------------------------------------------------------------")
print(f"The Confusion Matrix is:  \n{confusion_matrix(y_val,y_pred)}")
print("-------------------------------------------------------------------------")
print(f"The Classification Report is: \n {classification_report(y_val,y_pred)}")


y_true = y_val
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')


all_rows = np.arange(train.id.max()) 
without_null= train_not_NA.id #select rows without ANY NULL
discarded = all_rows[~np.isin(all_rows, without_null)]
len(discarded)


mask = train.index.isin(discarded)   # np_array is your 1-D numpy array of labels
train_null = train[mask].copy()


train_null.head(1)


train_null['Personality'].value_counts().plot(kind='bar')


train_null.isna().sum()



#Imputation with mean

train_null['Time_spent_Alone']= train_null['Time_spent_Alone'].fillna(train_null['Time_spent_Alone'].mean())

train_null['Social_event_attendance'] = train_null['Social_event_attendance'].fillna(train_null['Social_event_attendance'].mean())#inplace=True

train_null['Going_outside']= train_null['Going_outside'].fillna(train_null['Going_outside'].mean())

train_null['Friends_circle_size'] = train_null['Friends_circle_size'].fillna(train_null['Friends_circle_size'].mean())

train_null['Post_frequency'] =train_null['Post_frequency'].fillna(train_null['Post_frequency'].mean())

'''
cols_to_impute =['Time_spent_Alone', 'Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']
df_temp= train_null.copy()
for c in cols_to_impute:
    df_temp[c] = imputation_by_pattern(train_null, c, cols_to_impute, mean_table)
train_null = df_temp.copy()
'''



### No Imputation (Categorical values)


np.random.seed(42)
#mask = train_null['Stage_fear'].isna()
#train_null.loc[mask, 'Stage_fear'] = np.random.choice(['Yes', 'No'], size=mask.sum())

#mask = train_null['Drained_after_socializing'].isna()
#train_null.loc[mask, 'Drained_after_socializing'] = np.random.choice(['Yes', 'No'], size=mask.sum())


# check 
train_null.isna().sum()


train_null.head(2)


train_null['personality_encoded'] = le.transform(train_null['Personality'])


df.head(1)


train_null = make_dummies(train_null,cols_to_dummies).copy()


train_null.head(1)


ROI = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency', 'Stage_fear_No', 'Stage_fear_Yes',
       'Drained_after_socializing_No', 'Drained_after_socializing_Yes']
X = train_null[ROI]
y = train_null['personality_encoded'].values


X = scaler.fit_transform(X).copy()


y_pred = clf.predict(X)


y_val=y
print("-------------------------------------------------------------------------")
print(f"The accuracy score is:   {round(accuracy_score(y_val,y_pred),6)}")
print("-------------------------------------------------------------------------")
print(f"The Confusion Matrix is:  \n{confusion_matrix(y_val,y_pred)}")
print("-------------------------------------------------------------------------")
print(f"The Classification Report is: \n {classification_report(y_val,y_pred)}")


y_true = y_val
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')


train_null['predicted_personality'] = y_pred
train_null.head(1)


train_null['decoded_personality'] = le.inverse_transform(train_null['predicted_personality'])
train_null.head()


test.head(1)


example_sub.head(1)


# Step 1: imputation



test['Time_spent_Alone']= test['Time_spent_Alone'].fillna(test['Time_spent_Alone'].mean())

test['Social_event_attendance'] = test['Social_event_attendance'].fillna(test['Social_event_attendance'].mean())#inplace=True

test['Going_outside']= test['Going_outside'].fillna(test['Going_outside'].mean())

test['Friends_circle_size'] = test['Friends_circle_size'].fillna(test['Friends_circle_size'].mean())

test['Post_frequency'] = test['Post_frequency'].fillna(test['Post_frequency'].mean())
'''


cols_to_impute =['Time_spent_Alone', 'Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']
df_temp= test.copy()
for c in cols_to_impute:
    df_temp[c] = imputation_by_pattern(test, c, cols_to_impute, mean_table)
test = df_temp.copy()

'''


# Step 2: dummies
cols_to_dummies = ['Stage_fear','Drained_after_socializing']
test = make_dummies(test,cols_to_dummies).copy()
test.head(2)


# Step 3: Select ROI and normalize
ROI = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency', 'Stage_fear_No', 'Stage_fear_Yes',
       'Drained_after_socializing_No', 'Drained_after_socializing_Yes']

X = test[ROI]


X = scaler.fit_transform(X).copy()


# Step 4: predict!
y_pred = clf.predict(X)


predicted = le.inverse_transform(y_pred)
predicted


# Step 5 : Submission

# id_series and personality_series are your two pd.Series
sub = pd.DataFrame({
    'id': test.id,
    'Personality': predicted
})
sub.head()


sub.shape


sub.to_csv('submission.csv', index=False)


%pwd

