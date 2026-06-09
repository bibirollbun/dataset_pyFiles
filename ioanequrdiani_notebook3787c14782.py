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


train  = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")


train


train.columns


train.info


trainset = train.sample(frac = 0.8)
valset = train.drop(trainset.index)
testset = pd.read_csv("/kaggle/input/playground-series-s4e6/test.csv")


numeric_columns = ['Marital status', 'Application mode', 'Application order',
       'Course', 'Daytime/evening attendance', 'Previous qualification',
       'Previous qualification (grade)', 'Nacionality',
       "Mother's qualification", "Father's qualification",
       "Mother's occupation", "Father's occupation", 'Admission grade',
       'Displaced', 'Educational special needs', 'Debtor',
       'Tuition fees up to date', 'Gender', 'Scholarship holder',
       'Age at enrollment', 'International',
       'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
       'Inflation rate', 'GDP']



def prepare_data(df, status = 'Train', info_dct = None):
    df = df.copy()
    if status == "Train":
        info_dct = {}
    #
    # Drop unnec. columns:
    unnecessary_columns = ['id']
    if status == "Train":
        df = df.drop(unnecessary_columns, axis = 1)
        info_dct["unnecessary_columns"] = unnecessary_columns
    elif status == "Test":
        unnecessary_columns = info_dct.get("unnecessary_columns")
        df = df.drop(unnecessary_columns, axis = 1)
    #
    # Encode Target:
    if "Target" in df.columns:
        if status == 'Train':
            Target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
            info_dct['Target_map'] = Target_map
        else:
            Target_map = info_dct['Target_map']
        def map_Target(value):
            return Target_map[value]
        df['Target'] = df['Target'].apply(map_Target)
    #
    # Fill miss. values:
    if status == "Train":
        missing_dct = {}
        columns = df.columns
        for i in df.columns:
            col = df.loc[:, [i]]
            median = col.median()
            missing_dct[i] = median
            df.loc[:, [i]].fillna(median)
        info_dct["Missings"] = missing_dct
    elif status == "Test":
        missing_dct = info_dct.get("Missings")
        for i in df.columns:
            median = missing_dct.get(i)
            df.loc[:, [i]].fillna(median)
    # Make outliers
    def handle_outliers(series, status, col_name):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_whisker = Q1 - 1.5*IQR
        upper_whisker = Q3 + 1.5*IQR
        extreme_lower = Q1 - 3*IQR
        extreme_upper = Q3 + 3*IQR
        #
        if status == 'Train':
            info_dct[col_name+'_whiskers'] = (lower_whisker, upper_whisker)
        #
        def cap_value(x):
            if x < extreme_lower:
                return lower_whisker
            elif x > extreme_upper:
                return upper_whisker
            else:
                return x
        return series.apply(cap_value)
    for i in numeric_columns:
        df[i] = handle_outliers(df[i], status, i)
    #
    # Scaling
    for i in numeric_columns:
        if status == 'Train':
            mn = df[i].min()
            mx = df[i].max()
            info_dct[i + '_scale'] = (mn, mx)
        else:
            mn, mx = info_dct[i + '_scale']
    
        def fff(a):
            return (a - mn) / (mx - mn + 0.000000001)
    
        df[i] = df[i].apply(fff)
    # Drop columns with >75% same value
    def drop_highly_skewed_columns(df, threshold=0.75):
        drop_cols = []
        for col in df.columns:
            top_freq = df[col].value_counts(normalize=True).iloc[0]
            if top_freq > threshold:
                drop_cols.append(col)
        return df.drop(drop_cols, axis=1), drop_cols

    if status == 'Train':
        df, dropped_skewed = drop_highly_skewed_columns(df, threshold=0.75)
        info_dct['dropped_highly_skewed'] = dropped_skewed
    else:
        if 'dropped_highly_skewed' in info_dct:
            df = df.drop(info_dct['dropped_highly_skewed'], axis=1, errors='ignore')
    return df, info_dct



train


clean_trainset, info_dct = prepare_data(trainset, "Train")

clean_valset, _ = prepare_data(valset, "Test", info_dct = info_dct)

clean_testset, _ = prepare_data(testset, "Test", info_dct = info_dct)




# Categorical Columns
cat_cols = ['Marital status', 'Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Target']
#



clean_trainset


info_dct


target = 'Target'

X_train = clean_trainset.drop(columns=[target])
y_train = clean_trainset[target]

X_val = clean_valset.drop(columns=[target])
y_val = clean_valset[target]



from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score
)
import matplotlib.pyplot as plt




decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)



train_pred = decision_tree.predict(X_train)
val_pred = decision_tree.predict(X_val)



xxxx = classification_report(y_val, y_val_pred)
print(xxxx)


plt.figure(figsize=(30, 20), dpi=200)
plot_tree(decision_tree, feature_names=X_train.columns, class_names=['Dropout','Enrolled', 'Graduate'], filled=True, max_depth=2)
#



print(classification_report(y_val, y_val_pred))


from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
max_depth_list = [2, 3, 4, 5, None]
min_samples_split_list = [2, 5, 10, 20]
min_impurity_decrease_list = [0.0, 0.001, 0.01]

results = {}
best_cls = -1
best_model = None
best_params = None

for d in max_depth_list:
    for m in min_samples_split_list:
        for imp in min_impurity_decrease_list:

            model = DecisionTreeClassifier(
                max_depth=d,
                min_samples_split=m,
                min_impurity_decrease=imp,
                random_state=0
            )

            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]

            cls_rep = classification_report(y_val, y_val_pred)

            key = f"depth={d}, minsplit={m}, imp={imp}"
            results[key] = {
                "classification_report": cls_rep,
                "params": (d, m, imp)
            }
            cccc = classification_report(y_val, y_val_pred, output_dict = True)

            if cccc['macro avg']['f1-score'] > best_cls:
                best_cls = cccc['macro avg']['f1-score']
                best_model = model
                best_params = (d, m, imp)

print("BEST PARAMS:", best_params)
print("BEST F1:", best_cls)



print(xxxx)
print(classification_report(y_val, y_val_pred))



test_predictions = best_model.predict(clean_testset)
test_predictions


from sklearn.linear_model import LogisticRegression


lr = LogisticRegression()
lr.fit(X_train, y_train)


train_pred = lr.predict(X_train)
val_pred = lr.predict(X_val)


weights = lr.coef_[0]
weights


bias = lr.intercept_[0]
print(bias)


y_train_pred = lr.predict(X_train)
y_val_pred = lr.predict(X_val)


yyyy = classification_report(y_val, y_val_pred)
print(yyyy)


#print()


coefficients = pd.Series(lr.coef_[0], index=X_train.columns).sort_values(ascending=False)
coefficients


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Define hyperparameter options
fit_intercept_list = [True, False]
max_iter_list = [100, 500, 1000]

results = {}
best_cls = -1
best_model = None
best_params = None

for fit_int in fit_intercept_list:
    for max_it in max_iter_list:

        model = LogisticRegression(
            fit_intercept=fit_int,
            max_iter=max_it,
            solver='liblinear',   # liblinear works for both l1 and l2 penalties
            random_state=0
        )

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:,1]

        cls_rep = classification_report(y_val, y_val_pred)

        key = f"fit_intercept={fit_int}, max_iter={max_it}"
        results[key] = {
            "classification_report": cls_rep,
            "params": (fit_int, max_it)
        }
        cccc = classification_report(y_val, y_val_pred, output_dict = True)

        if cccc['macro avg']['f1-score'] > best_cls:
            best_cls = cccc['macro avg']['f1-score']
            best_model = model
            best_params = (fit_int, max_it)



results


print("BEST PARAMS:", best_params)
print("BEST F1:", best_cls)


print(yyyy)
print(classification_report(y_val, y_val_pred))


test_predictions = best_model.predict(clean_testset)
test_predictions


submission = pd.read_csv("/kaggle/input/playground-series-s4e6/sample_submission.csv")
submission


submission["Target"] = test_predictions



submission


Target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
tttt = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
submission['Target'] = submission['Target'].map(tttt)


submission


submission.to_csv("submission.csv", index=False)
submission.head()

