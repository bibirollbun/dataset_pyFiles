pip install scorecardbundle


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.impute import SimpleImputer
from scorecardbundle.feature_discretization import ChiMerge as cm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif, mutual_info_classif, RFECV
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.tree import _tree
import re


X_data = pd.read_csv('/kaggle/input/x-data/appended_x_data_discretized_no_scaling.csv')
X_data = X_data.set_index(['Unnamed: 0'])

X_train = X_data.loc['train']
X_test = X_data.loc['test']

y_data = pd.read_csv('/kaggle/input/y-data/appended_y_data_discretized_no_scaling.csv')
y_data = y_data.set_index(['Unnamed: 0','RefId'])

y_train = y_data.loc['train'].astype('str')
y_test = y_data.loc['test'].astype('str')

X_train.info()
y_train.info()


columns = X_train.columns
categorical_indices = [0, 2, 3, 4, 5, 7, 8, 9, 18, 19, 20]
continuous = X_train.select_dtypes(exclude=['object','category']).columns.tolist()
categorical = X_train.select_dtypes(include=['object','category']).columns.tolist()
ordinal = ['VehBCost_cat_cm','WarrantyCost_cat_cm']
nominal = [i for i in categorical if i not in ordinal]


one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
one_hot_encoded_train = one_hot_encoder.fit_transform(X_train[nominal])
one_hot_encoded_test = one_hot_encoder.transform(X_test[nominal])

one_hot_encoded_df_train = pd.DataFrame(one_hot_encoded_train, columns=one_hot_encoder.get_feature_names_out())
one_hot_encoded_df_test = pd.DataFrame(one_hot_encoded_test, columns=one_hot_encoder.get_feature_names_out())

X_train_dropped = X_train.drop(columns=nominal+ordinal)
X_test_dropped = X_test.drop(columns=nominal+ordinal)

X_train = pd.concat([X_train_dropped.reset_index(), one_hot_encoded_df_train, X_train[ordinal].reset_index(drop=True)], axis=1)
X_train = X_train.set_index(['RefId','Unnamed: 0'])
X_test = pd.concat([X_test_dropped.reset_index(), one_hot_encoded_df_test, X_test[ordinal].reset_index(drop=True)], axis=1)
X_test = X_test.set_index(['RefId','Unnamed: 0'])


clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=20,min_samples_leaf=10,
                             min_weight_fraction_leaf=0.0,random_state=880, min_impurity_decrease=0.0,
                             class_weight="balanced",ccp_alpha=0.0) 

clf = clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for test set:", accuracy)


clf.classes_
clf.feature_importances_
clf.feature_names_in_
clf.max_features_
clf.n_classes_
clf.n_features_in_
clf.n_outputs_
clf.tree_


y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)


accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

accuracy_diff = accuracy_train - accuracy_test

print("Accuracy for train set:", accuracy_train)
print("Accuracy for test set:", accuracy_test)
print("Difference in accuracy between train and test sets:", accuracy_diff)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=clf.classes_)
disp.plot()
plt.show()


report = classification_report(y_test, y_pred, labels=None, zero_division='warn')
print("Classification Report:")
print(report)


fpr, tpr, thresholds = roc_curve(y_test,y_prob[:,1], pos_label="1")
roc_auc = auc(fpr, tpr)

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Classification Tree')
display.plot()
plt.show()

roc_auc = roc_auc_score(y_test, y_prob[:,1])
print("ROC AUC Score:", roc_auc)


custom_scorer = make_scorer(f1_score, pos_label='1', average='binary')

param_grid = {
    'criterion':['gini', 'entropy'],
    'max_depth': [ 3, 4, 10],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [5, 8, 12],
    'class_weight': [None, 'balanced']
}


grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    scoring=custom_scorer,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_clf = grid_search.best_estimator_

accuracy = best_clf.score(X_test, y_test)
print("Test Accuracy:", accuracy)


best_clf.fit(X_train, y_train)

y_pred_train = best_clf.predict(X_train)
y_prob_train = best_clf.predict_proba(X_train)

conf_matrix_train = confusion_matrix(y_train, y_pred_train)
print("Confusion Matrix for Training Set:")
print(conf_matrix_train)

report_train = classification_report(y_train, y_pred_train, labels=None, zero_division='warn')
print("Classification Report fot Training Set:")
print(report_train)

roc_auc_train = roc_auc_score(y_train, y_prob_train[:,1])
print("ROC AUC Score for Training Set:", roc_auc_train)
print("#"*60)
#______________________________________________________________________________________________________

y_pred = best_clf.predict(X_test)
y_prob = best_clf.predict_proba(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for Test Set:")
print(conf_matrix)

report = classification_report(y_test, y_pred, labels=None, zero_division='warn')
print("Classification Report for Test Set:")
print(report)

roc_auc = roc_auc_score(y_test, y_prob[:,1])
print("ROC AUC Score for Test Set:", roc_auc)


importances = best_clf.feature_importances_
feature_names =best_clf.feature_names_in_

indices = np.argsort(importances)

plt.figure(figsize=(30, 20))
plt.title("Feature Importances")
plt.barh(range(len(feature_names)), importances[indices], color="navy", align="center")
plt.yticks(range(len(feature_names)), feature_names[indices])
plt.xlabel("Importance")
plt.ylabel("Features")
#plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


plt.figure(figsize=(40,30))  
plot_tree(best_clf,
         feature_names=best_clf.feature_names_in_,  
        class_names=[str(c) for c in best_clf.classes_],
        proportion=True,
          filled=True,
          rounded=True)
plt.show()


tree_rules = export_text(
    best_clf,
    feature_names=list(best_clf.feature_names_in_),
    spacing=3,
    decimals=2,
    show_weights=True
)

print(tree_rules)



def get_rules(tree, feature_names, task):  # task='reg' or 'class'
    tree_ = tree.tree_
    if task == 'class':
        class_name = tree.classes_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    antecedents = []
    consequents = []
    
    for path in paths:
        antecedent = ""
        consequent = ""
        
        for p in path[:-1]:
            if antecedent != "":
                antecedent += " and "
            antecedent += str(p)

        if task == 'reg':
            consequent += str(np.round(path[-1][0][0][0], 3))
        elif task == 'class':
            classes = path[-1][0][0]
            l = np.argmax(classes)
            consequent += str(class_name[l])

        antecedents.append(antecedent)
        consequents.append(consequent)
    
    rules = pd.DataFrame({'Antecedent':antecedents, 'Consequent':consequents})
        
    return rules



feature_names = X_train.columns.tolist()

pd.set_option('display.max_colwidth', None)

rules = get_rules(best_clf, feature_names, task='class')
rules


def calculate_rule_metrics(rules_df, input_data, target_variable, task):

    import re

    # Add columns for metrics
    rules_df['Sample'] = None
    rules_df['Support'] = None
    rules_df['Confidence'] = None
    rules_df['Lift'] = None
    
    # Calculate metrics for each rule
    for i, row in rules_df.iterrows():
        antecedent = row['Antecedent']
        consequent = row['Consequent']
        
        def filter_by_antecedent(data, antecedent):

            conditions = antecedent.split(" and ")
            filtered_data = data.copy()  # Avoid modifying original data
            pattern = r"(?P<col_name>\w+)\s*(?P<operator>[<>=]+)\s*(?P<threshold>[-.\d]+)"

            for condition in conditions:
                condition = condition.strip('()')
                match = re.match(pattern, condition)
                if not match:
                    raise ValueError(f"Invalid condition format: {condition}")

                col_name, operator, threshold = match.groups()
                # Access the appropriate comparison function directly using operator
                comparison_func = {
                    "<": pd.Series.__lt__,
                    ">": pd.Series.__gt__,
                    "<=": pd.Series.__le__,
                    ">=": pd.Series.__ge__,
                    "==": pd.Series.__eq__,
                }[operator]

                filtered_data = filtered_data[comparison_func(filtered_data[col_name], float(threshold))]

            return filtered_data



        # Filter train_data based on antecedent conditions (boolean mask)
        filtered_data = filter_by_antecedent(input_data, antecedent)
        index = filtered_data.index.tolist()

        # Calculate support (proportion of samples satisfying the rule)
        rule_support = len(filtered_data) / len(input_data)
        rules_df.at[i, 'Sample'] = len(filtered_data)
        rules_df.at[i, 'Support'] = rule_support

        # Calculate confidence (conditional probability of consequent given antecedent) and lift index
        if task == 'class':
            
            if len(filtered_data) == 0:
                confidence = None
                lift = None
            else:
                class_probabilities = target_variable.value_counts(normalize=True)
                if target_variable.index.name:
                    class_counts = target_variable.loc[index].value_counts()
                else:
                    class_counts = target_variable[index].value_counts()
                if consequent in class_counts:
                    correct_predictions = class_counts[consequent]
                else:
                    correct_predictions = 0
                confidence = (correct_predictions / len(filtered_data)) 
                lift = (confidence) / class_probabilities[consequent]       

        elif task == 'reg':

            # Confidence and Lift calculation for regression is not typically used
            confidence = None
            lift = None

        rules_df.at[i, 'Confidence'] = confidence
        rules_df.at[i, 'Lift'] = lift
        
    return rules_df


def calculate_rule_metrics(rules_df, input_data, target_variable, task):
    import re

    # Add columns for metrics
    rules_df['Sample'] = None
    rules_df['Support'] = None
    rules_df['Confidence'] = None
    rules_df['Lift'] = None

    def filter_by_antecedent(data, antecedent):
        conditions = antecedent.split(" and ")
        filtered_data = data.copy()
        pattern = r"(?P<col_name>\w+)\s*(?P<operator>[<>=]+)\s*(?P<threshold>[-.\d]+)"

        for condition in conditions:
            condition = condition.strip('()')
            match = re.match(pattern, condition)
            if not match:
                raise ValueError(f"Invalid condition format: {condition}")

            col_name, operator, threshold = match.groups()
            comparison_func = {
                "<": pd.Series.__lt__,
                ">": pd.Series.__gt__,
                "<=": pd.Series.__le__,
                ">=": pd.Series.__ge__,
                "==": pd.Series.__eq__,
            }[operator]

            filtered_data = filtered_data[comparison_func(filtered_data[col_name], float(threshold))]

        return filtered_data

    for i, row in rules_df.iterrows():
        antecedent = row['Antecedent']
        consequent = row['Consequent']

        # Filter input_data based on antecedent conditions
        filtered_data = filter_by_antecedent(input_data, antecedent)
        filtered_data = filtered_data.reset_index(drop=False)  # reset index but keep old index as column
        original_indices = filtered_data[input_data.index.name].tolist() if input_data.index.name else filtered_data.index.tolist()

        # Calculate support
        rule_support = len(filtered_data) / len(input_data)
        rules_df.at[i, 'Sample'] = len(filtered_data)
        rules_df.at[i, 'Support'] = rule_support

        if task == 'class':
            if len(filtered_data) == 0:
                confidence = None
                lift = None
            else:
                class_probabilities = target_variable.value_counts(normalize=True)

                # تبدیل ایندکس‌های filtered_data به موقعیت عددی در target_variable
                pos_indices = [target_variable.index.get_loc(idx) for idx in original_indices if idx in target_variable.index]

                # گرفتن مقادیر target_variable بر اساس موقعیت‌های عددی
                filtered_targets = target_variable.iloc[pos_indices]

                class_counts = filtered_targets.value_counts()
                correct_predictions = class_counts.get(consequent, 0)

                confidence = correct_predictions / len(filtered_data)
                lift = confidence / class_probabilities.get(consequent, 1e-10)  # جلوگیری از تقسیم بر صفر

        else:
            confidence = None
            lift = None

        rules_df.at[i, 'Confidence'] = confidence
        rules_df.at[i, 'Lift'] = lift

    return rules_df



calculate_rule_metrics(rules, X_train, y_train, task='class')







