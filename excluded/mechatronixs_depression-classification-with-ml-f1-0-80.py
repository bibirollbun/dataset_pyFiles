!pip install catboost


import warnings
warnings.simplefilter(action="ignore")

import numpy as np
import pandas as pd

import math
# For visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier


pd.options.display.max_rows = None
pd.options.display.max_columns = None





def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)


def grab_col_names(dataframe, cat_th=11, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# Data Visualization

def target_vs_category_visual(dataframe, target, categorical_col):
    plt.figure(figsize=(15, 8))
    sns.histplot(x=target, hue=categorical_col, data=dataframe, element="step", multiple="dodge")
    plt.title("State of Categorical Variables according to Churn ")
    plt.show()


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"Depression Mean": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")
    print("###################################")

def get_numerical_summary(dataframe):
    total = df.shape[0]
    missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    missing_percent = {}
    for col in missing_columns:
        null_count = df[col].isnull().sum()
        per = (null_count / total) * 100
        missing_percent[col] = per
        print("{} : {} ({}%)".format(col, null_count, round(per, 3)))
    return missing_percent



def feature_distribution_plot(df, col, target_col=None):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    if df[col].dtype != 'object':
        sns.histplot(data=df, x=col, hue=target_col)
    else:
        sns.countplot(data=df, x=col, hue=target_col)
    plt.ylabel('Count')
    plt.xlabel(f'{col}')
    plt.title(f'Histogram of {col}')

    plt.subplot(1,2,2)
    if df[col].dtype != 'object':
        sns.boxplot(data=df, x=col, hue=target_col)
        plt.title(f'Boxplot of {col}')
        plt.ylabel('Count')
        plt.xlabel(f'{col}')
    else:
        df[col].value_counts().plot(kind='pie', autopct='%.0f%%',pctdistance=0.85,fontsize=12)
        plt.gca().add_artist(plt.Circle((0,0),radius=0.7,fc='white'))
        plt.title(f'Pie Chart of {col}')
        plt.xlabel('')
        plt.ylabel('')


    plt.tight_layout()
    plt.suptitle(f'Distribution of {col}', y=1.05, size=24, weight='bold')
    plt.show()


def plot_distribution_pairs(train, test, feature, hue="set", palette=None):
    data_df = train.copy()
    data_df['set'] = 'train'
    data_df = pd.concat([data_df, test.copy()]).fillna('test')

    f, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, s in enumerate(data_df[hue].unique()):
        selection = data_df.loc[data_df[hue]==s, feature]
        # Filter 'selection' to include only the central 95% of the data
        q_025, q_975 = np.percentile(selection, [2.5, 97.5])
        selection_filtered = selection[(selection >= q_025) & (selection <= q_975)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sns.histplot(selection_filtered, color=palette[i], ax=axes[0], label=s)
            sns.boxplot(x=hue, y=feature, data=data_df, palette=palette, ax=axes[1])
    axes[0].set_title(f"Paired train/test distributions of {feature}")
    axes[1].set_title(f"Paired train/test boxplots of {feature}")
    axes[0].legend()
    axes[1].legend()
    plt.show()

color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0B2447"]


def plot_cdf_by_loan_status(df, num_cols, loan_status_col='loan_status'):
    """
    Plots the CDF (Cumulative Density Function) of numerical columns by loan status.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    num_cols (list): List of numerical column names to plot.
    loan_status_col (str): The column representing the loan status (default is 'loan_status').

    Returns:
    None (Displays plots)
    """
    for column in num_cols:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[df[loan_status_col] == 1][column], label='Default', fill=True)
        sns.kdeplot(df[df[loan_status_col] == 0][column], label='Non-Default', fill=True)
        plt.title(f'CDF of {column} by Loan Status')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.show()


def plot_avg_loan_by_grade(df, loan_grade_col, loan_amnt_col, loan_status_col):
    """
    Plots the average loan amount by loan grade and loan status.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    loan_grade_col (str): The categorical column representing the loan grade.
    loan_amnt_col (str): The numerical column representing the loan amount.
    loan_status_col (str): The column representing the loan status.

    Returns:
    None (Displays plot)
    """
    if not pd.api.types.is_categorical_dtype(df[loan_grade_col]) and not df[loan_grade_col].dtype == 'object':
        raise ValueError(f"{loan_grade_col} should be a categorical column.")
    if not pd.api.types.is_numeric_dtype(df[loan_amnt_col]):
        raise ValueError(f"{loan_amnt_col} should be a numeric column.")

    plt.figure(figsize=(12, 6))
    sns.barplot(x=loan_grade_col, y=loan_amnt_col, hue=loan_status_col, data=df, estimator=np.mean)
    plt.title('Average Loan Amount by Loan Grade and Status')
    plt.xlabel('Loan Grade')
    plt.ylabel('Average Loan Amount')
    plt.legend(title='Loan Status')
    plt.show()






def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def quick_missing_imp(dataframe, target, num_method="median", cat_length=20):
    variables_with_na = [col for col in dataframe.columns if
                         dataframe[col].isnull().sum() > 0]  # Eksik deÄŸere sahip olan deÄŸiÅŸkenler listelenir

    temp_target = dataframe[target]

    print("# BEFORE")
    print(dataframe[variables_with_na].isnull().sum(),
          "\n\n")  # Uygulama Ã¶ncesi deÄŸiÅŸkenlerin eksik deÄŸerlerinin sayÄ±sÄ±

    # deÄŸiÅŸken object ve sÄ±nÄ±f sayÄ±sÄ± cat_lengthe eÅŸit veya altÄ±ndaysa boÅŸ deÄŸerleri mode ile doldur
    dataframe = dataframe.apply(
        lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan deÄŸiÅŸkenlerin boÅŸ deÄŸerleri ortalama ile dolduruluyor
    if num_method == "mean":
        dataframe = dataframe.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan deÄŸiÅŸkenlerin boÅŸ deÄŸerleri ortalama ile dolduruluyor
    elif num_method == "median":
        dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    dataframe[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")

    return dataframe


def outlier_th(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Define a Function about checking outlier for data columns
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Define a Function about replace with threshold for data columns
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe



def train_test(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("Base Models....")
    classifiers = [  #('LR', LogisticRegression()),
        # ('KNN', KNeighborsClassifier()),
        # ("SVC", SVC()),
        ("CART", DecisionTreeClassifier(random_state=0)),
        ("RF", RandomForestClassifier(random_state=0, max_features='sqrt')),
        # ('Adaboost', AdaBoostClassifier(random_state=0)),
         ('GBM', GradientBoostingClassifier(max_depth=4,random_state=0)),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(random_state=0, verbose=-1)),
        ('CatBoost', CatBoostClassifier(verbose=False))
    ]
    print(classifiers)
    return X_train, X_test, y_train, y_test, classifiers

def models2(classfiers, X, y):
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # Calculating Cross-Validation scores for different metrics
        accuracy_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy',n_jobs=-1).mean()
        f1_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='f1',n_jobs=-1).mean()
        precision_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='precision',n_jobs=-1).mean()
        recall_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='recall',n_jobs=-1).mean()

        # Printing Cross-Validation scores
        print(f"Classifier: {name}")
        print("Cross Validation Scores:")
        print("Accuracy : ", '{0:.2%}'.format(accuracy_cv))
        print("F1 : ", '{0:.2%}'.format(f1_cv))
        print("Precision : ", '{0:.2%}'.format(precision_cv))
        print("Recall : ", '{0:.2%}'.format(recall_cv))

        # Accuracy on test data
        test_accuracy = accuracy_score(y_test, prediction)
        print("Test Accuracy : ", '{0:.2%}'.format(test_accuracy))


def plot_cm(ax, cm, class_names=('0','1'), title='Confusion Matrix', normalize=None):
    """
    normalize: None | 'true' (satÄ±r) | 'pred' (sÃ¼tun) | 'all'
    """
    data = cm.astype(float).copy()
    if normalize == 'true':
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, where=row_sums!=0)
    elif normalize == 'pred':
        col_sums = data.sum(axis=0, keepdims=True)
        data = np.divide(data, col_sums, where=col_sums!=0)
    elif normalize == 'all':
        s = data.sum()
        data = data / s if s else data

    sns.heatmap(data, annot=False, cmap='Blues', cbar=True, square=True, ax=ax,
                vmin=0, vmax=data.max() if data.max() > 0 else 1)

    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticklabels(class_names, rotation=0)

    # HÃ¼cre anotasyonu: TN/FP/FN/TP + sayÄ± + % (normalize seÃ§imine gÃ¶re)
    tags = np.array([['TN','FP'], ['FN','TP']])
    total = cm.sum()
    thresh = data.max() * 0.5
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            if normalize == 'true':
                denom = cm[i, :].sum()
            elif normalize == 'pred':
                denom = cm[:, j].sum()
            elif normalize == 'all':
                denom = total
            else:
                denom = total
            perc = (count / denom) if denom else 0.0
            color = 'white' if data[i, j] > thresh else 'black'
            ax.text(j + 0.5, i + 0.5,
                    f"{tags[i,j]}\n{count}\n{perc:.2%}",
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

def model_evaluation(classifiers, X_test, y_test, X_train, y_train):
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        cm = confusion_matrix(y_test, classifier.predict(X_test))
        names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        counts = [value for value in cm.flatten()]
        percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        # Her sÄ±nÄ±flandÄ±rÄ±cÄ± iÃ§in ayrÄ± bir grafik Ã§iz
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=labels, cmap='Blues', fmt='', square=True)
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # GÃ¶ster
        plt.show(block=True)

        # SÄ±nÄ±flandÄ±rma raporunu yazdÄ±r
        print(f'Classification Report for {name}:\n')
        print(classification_report(y_test, classifier.predict(X_test)))


def model_evaluation_s(classifiers, X_train, y_train, X_test, y_test,
                     class_names=('Non Churn','Churn'), normalize=None, ann_fs=9, tick_fs=9, title_fs=11, label_fs=10, line_spacing=0.9):
    for name, clf in classifiers:
        model = clone(clf).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        plot_cm(ax, cm, class_names=class_names, title=f'Confusion Matrix â€“ {name}', normalize=normalize)
        fig.tight_layout()
        plt.show()

        print(f'Classification Report â€” {name}\n')
        print(classification_report(y_test, y_pred, zero_division=0))




def feature_importances(classifiers, X, y,count=15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        feature_imp = pd.Series(classifier.feature_importances_,
                                index=X.columns).sort_values(ascending=False)[:count]

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('DeÄŸiÅŸken Ã–nem SkorlarÄ±')
        plt.ylabel('DeÄŸiÅŸkenler')
        plt.title(name)
        plt.show(block=True)


def hyperparameter_optimization(X, y, classifiers1, cv=5, main_scoring='accuracy'):
    print("Hyperparameter Optimization....")
    best_models = {}
    scoring_metrics = ['accuracy', 'f1', 'recall', 'precision']

    for name, classifier, params in classifiers1:
        print(f"########## {name} ##########")

        initial_scores = {}
        for metric in scoring_metrics:
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=metric)
            mean_score = round(cv_results['test_score'].mean(), 4)
            initial_scores[metric] = mean_score
            print(f"{metric} (Before): {mean_score}")

        # GridSearchCV ile hiperparametre optimizasyonu
        # RandomSearchCV
        gs_best = RandomizedSearchCV(classifier, params, cv=cv, scoring=main_scoring, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        print(f"{name} best params: {gs_best.best_params_}")

        # Optimizasyon sonrasÄ± skorlarÄ± hesaplama
        optimized_scores = {}
        for metric in scoring_metrics:
            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=metric)
            mean_score = round(cv_results['test_score'].mean(), 4)
            optimized_scores[metric] = mean_score
            print(f"{metric} (After): {mean_score}")

        best_models[name] = {
            'final_model': final_model,
            'initial_scores': initial_scores,
            'optimized_scores': optimized_scores
        }

    return best_models





def _plot_single_cm(ax, cm, class_names, title):
    """Tek bir confusion matrix'i anotasyonlu Ã§iz."""
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # HÃ¼cre iÃ§i sayÄ±m + yÃ¼zde, kontrasta gÃ¶re yazÄ± rengi
    total = cm.sum()
    thresh = cm.max() / 2.0
    tags = np.array([['TN','FP'], ['FN','TP']])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{tags[i,j]}\n{cm[i,j]:,}\n{cm[i,j]/total:.2%}"
            color = 'white' if cm[i,j] > thresh else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

def evaluate_models_confusions(
    models, X_train, y_train, X_test, y_test,
    class_names=('Non Churn','Churn'),
    normalize=None,        # None | 'true' | 'pred' | 'all'  (sklearn>=0.22)
    cols=3, figsize_per=(5,4), print_reports=False
):
    """
    TÃ¼m modeller iÃ§in confusion matrix'i hesapla ve Ã§iz.
    DÃ¶nÃ¼ÅŸ: (metrics_df, cms_dict)
    """
    n = len(models)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figsize_per[0], rows*figsize_per[1]))
    if n == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    metrics = []
    cms = {}

    for i, (name, model) in enumerate(models):
        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Confusion matrix (ham sayÄ±; normalize istersen ayrÄ±ca hesapla)
        cm_raw = confusion_matrix(y_test, y_pred)
        cm_to_plot = confusion_matrix(y_test, y_pred, normalize=normalize) if normalize else cm_raw

        cms[name] = cm_raw
        _plot_single_cm(axes[i], cm_to_plot, class_names, title=f'{name}')

        # Metrikler
        row = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
        }
        # AUC (varsa)
        try:
            if hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_test)
            else:
                scores = None
            if scores is not None:
                row['ROC_AUC'] = roc_auc_score(y_test, scores)
        except Exception:
            row['ROC_AUC'] = np.nan
        metrics.append(row)

        if print_reports:
            from sklearn.metrics import classification_report
            print(f'\nClassification report â€” {name}\n')
            print(classification_report(y_test, y_pred))

    # BoÅŸ kalan eksenleri kaldÄ±r
    for j in range(i+1, rows*cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    metrics_df = pd.DataFrame(metrics).round(4)
    return metrics_df, cms






def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]


    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

def hyperparameter_optimization(X, y, classifiers_hyp, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, clf, params in classifiers_hyp:
        print(f"########## {name} ##########")
        before = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)['test_score'].mean()
        print(f"{scoring} (Before): {before:.4f}")

        gs = GridSearchCV(clf, params, cv=cv, n_jobs=-1, scoring=scoring, verbose=0)
        gs.fit(X, y)
        best = gs.best_estimator_

        after = cross_validate(best, X, y, cv=cv, scoring=scoring, n_jobs=-1)['test_score'].mean()
        print(f"{scoring} (After):  {after:.4f}")
        print(f"{name} best params: {gs.best_params_}\n")

        best_models[name] = best
    return best_models

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]),
                                              ('XGBoost', best_models["XGBoost"])],
                                               #('CatBoost', best_models["CatBoost"]))],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


def select_and_save_best_base_model(X, y, metric="f1", cv_splits=5, out_path=None):
    """
    Base modelleri CV ile deÄŸerlendirir, seÃ§ilen metrike gÃ¶re en iyiyi X,y Ã¼zerinde fit eder
    ve joblib ile kaydeder. (metric: 'f1' | 'roc_auc' | 'accuracy' | 'recall' | 'precision')
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    rows = []
    name2est = dict(BASE_MODELS)

    for name, est in BASE_MODELS:
        scores = cross_validate(
            est, X, y, cv=cv, n_jobs=-1,
            scoring=["accuracy", "roc_auc", "recall", "precision", "f1"],
            return_train_score=False
        )
        rows.append({
            "Model": name,
            "CV_Accuracy":  scores["test_accuracy"].mean(),
            "CV_AUC":       scores["test_roc_auc"].mean(),
            "CV_Recall":    scores["test_recall"].mean(),
            "CV_Precision": scores["test_precision"].mean(),
            "CV_F1":        scores["test_f1"].mean(),
        })

    results = pd.DataFrame(rows)
    metric_col = {"f1":"CV_F1", "roc_auc":"CV_AUC", "accuracy":"CV_Accuracy",
                  "recall":"CV_Recall", "precision":"CV_Precision"}[metric]
    best_row  = results.sort_values(metric_col, ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_est  = clone(name2est[best_name]).fit(X, y)  # tÃ¼m veriyle yeniden eÄŸit

    # kaydet
    if out_path is None:
        out_path = f"best_base_{metric}.joblib"
    artifact = {
        "model": best_est,
        "model_name": best_name,
        "selected_metric": metric,
        "metric_value": float(best_row[metric_col]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": list(X.columns) if hasattr(X, "columns") else None,
    }
    joblib.dump(artifact, out_path)

    return best_est, results.sort_values(metric_col, ascending=False).reset_index(drop=True), out_path






import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
df.head()


check_data(df)


df.nunique()


df.info()


df.describe([0.5,0.7,0.8,0.9,0.95,1]).T


df["Depression"].value_counts()


labels = 'NotDepression', 'Depression'
sizes = [df.Depression[df['Depression']==1].count(), df.Depression[df['Depression']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 14)
plt.show()


df = df.drop(["id"], axis = 1)


cat_cols, num_cols, cat_but_car = grab_col_names(df)


cat_cols


num_cols


for col in num_cols:
    target_summary_with_num(df,"Depression",col)





for col in cat_cols:
    cat_summary(df, col)





for col in cat_cols:
    target_summary_with_cat(df, "Depression", col)





df.groupby("Work Pressure").agg({"Depression": "mean"})





df['Depression'] = df['Depression'].astype(str)


fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Job Satisfaction', hue = 'Depression',data = df, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Depression',data = df, ax=axarr[0][1])
sns.countplot(x='Academic Pressure', hue = 'Depression',data = df, ax=axarr[1][0])
sns.countplot(x='Study Satisfaction', hue = 'Depression',data = df, ax=axarr[1][1])





df["Age"].describe()


num_cols


  fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
  sns.boxplot(y='Age',x = 'Depression', hue = 'Depression',data = df , ax=axarr[0][1])
  sns.boxplot(y='CGPA',x = 'Depression', hue = 'Depression',data = df, ax=axarr[1][0])
  sns.boxplot(y='Work/Study Hours',x = 'Depression', hue = 'Depression',data = df, ax=axarr[1][1])








counts = df.groupby(['Gender','Depression']).size().reset_index(name='count')
labels = [f"{row['Gender']}, {row['Depression']}" for _, row in counts.iterrows()]

fig = go.Figure(data=[go.Pie(labels=labels, values=counts['count'])])
fig.update_layout(title_text="Distribution of Depression by Gender")
fig.show()



fig = go.Figure()

# Her Geography iÃ§in ayrÄ± bar serisi ekliyoruz
for geo in df['Job Satisfaction'].unique():
    sub = df[df['Job Satisfaction'] == geo]
    fig.add_trace(
        go.Bar(
            x=sub['Age'],
            y=sub['Depression'],
            name=geo
        )
    )

fig.update_layout(
    title="Dependent variable (Depression) by Age and Job Satisfaction",
    xaxis_title="Age",
    yaxis_title="Depression",
    barmode='group'   # 'stack' yaparsan Ã¼st Ã¼ste binmiÅŸ olur
)

fig.show()






df_cat   = df[cat_cols]

n_plots  = len(df_cat.columns)
n_cols   = 3
n_rows   = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten() if n_rows*n_cols > 1 else [axes]

for i, col in enumerate(df_cat.columns):
    ax = axes[i]
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_ylabel('Count')
    if col in ['HasCrCard','IsActiveMember']:
        ax.set_xticklabels(['No','Yes'])

# BoÅŸ kalan eksenleri kaldÄ±r
for j in range(i+1, n_rows*n_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()





for column in num_cols:
  plt.figure(figsize=(10, 6))
  sns.kdeplot(df[df['Depression'] == 1][column], label='Depression', fill=True)
  sns.kdeplot(df[df['Depression'] == 0][column], label='Non-Depression', fill=True)
  plt.title(f'{column} by Depression Status')
  plt.xlabel(column)
  plt.ylabel('Density')
  plt.legend()
  plt.show()





# Correlation Matrix
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df[num_cols].corr(method='spearman'), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


corr_matrix = df[num_cols].corr()
corr_matrix





target_corr = df[num_cols + ['Depression']].corr()['Depression'].sort_values(ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=target_corr.values, y=target_corr.index, palette="viridis")
plt.title("Correlation of Features with Exited", fontsize=16)
plt.show()


corr_with_target = df[num_cols + ['Depression']].corr()['Depression'].sort_values(ascending=False)
print(corr_with_target)


import plotly.express as px
corr = df[num_cols].corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
fig.show()





num_cols


from matplotlib import pyplot as plt
df['Age'].plot(kind='hist', bins=20, title='Age')
plt.gca().spines[['top', 'right',]].set_visible(False)


#df['log_age'] = np.log10(df['Age'] + 1)


sns.boxplot(x=df['Age'])
plt.title('Raw person_age')
plt.show()





df.isnull().sum()


# toplam satÄ±r sayÄ±sÄ±
n_rows = len(df)

# her sÃ¼tun iÃ§in boÅŸluk oranÄ±
missing_ratio = df.isna().sum() / n_rows

# %80 ve Ã¼zeri eksik olan sÃ¼tunlarÄ± bul
drop_cols = missing_ratio[missing_ratio >= 0.80].index.tolist()



drop_cols


# bu sÃ¼tunlarÄ± dÃ¼ÅŸÃ¼r
df_reduced = df.drop(columns=drop_cols).copy()

print(f"DÃ¼ÅŸÃ¼rÃ¼len sÃ¼tun sayÄ±sÄ±: {len(drop_cols)}")
print(drop_cols[:10])


df_reduced.head()


df_reduced.isnull().sum()





from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

cat_missing = ["Profession","Work Pressure","Job Satisfaction"]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=float)
Xc = enc.fit_transform(df_reduced[cat_missing])               # kategorikler sayÄ±ya

imputer = KNNImputer(n_neighbors=5, weights="distance")
Xc_imp = imputer.fit_transform(Xc)                 # sayÄ±sal KNN imputation

# En yakÄ±n tam sayÄ±ya yuvarla (kategorik indeksler)
Xc_imp = np.round(Xc_imp)

df_reduced[cat_missing] = enc.inverse_transform(Xc_imp)


def impute_group_mode(s, by):
    # s: doldurulacak sÃ¼tun (Series), by: gruplandÄ±rma anahtarÄ± (DataFrame cols)
    return s.fillna(
        s.groupby(by).transform(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    )


# Son durumda Eksik Verileri Silelim
df_reduced = df_reduced.dropna()


df_reduced.isnull().sum()


cat_cols, num_cols, cat_but_car = grab_col_names(df_reduced)


num_cols


for col in num_cols:
    print(col, check_outlier(df_reduced, col))


for col in num_cols:
    replace_with_thresholds(df_reduced, col)

for col in num_cols:
    print(col, check_outlier(df_reduced, col))


df_reduced.info()


def label_encoder(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df


binary_cols = [col for col in df_reduced.columns if df_reduced[col].dtype not in [int, float]
               and df_reduced[col].nunique() == 2]

for col in binary_cols:
    df_reduced = label_encoder(df_reduced, col)

ohe_cols = [col for col in df_reduced.columns if 10 >= df_reduced[col].nunique() > 2]


binary_cols


df.nunique()


df_reduced["Degree"].value_counts().head(10)


import re

# Ã¶rnek: df["Profession"] zaten var kabul ediyorum
prof = df_reduced["Profession"].fillna("Unknown").str.strip()

# nadir sÄ±nÄ±flarÄ± topla (Ã¶r. frekansÄ± < 0.5% veya min_count)
min_count = int(0.005 * len(df_reduced))  # %0.5 eÅŸiÄŸi, veri bÃ¼yÃ¼klÃ¼ÄŸÃ¼ne gÃ¶re deÄŸiÅŸtir
value_counts = prof.value_counts()
rare = set(value_counts[value_counts < min_count].index)

def map_profession(x: str) -> str:
    x_low = x.lower()

    # rare & unknown
    if x in rare or x_low in ("", "unknown", "na", "none"):
        return "Other"

    # LEGAL
    if re.search(r"\blawyer\b|\battorney\b|judge", x_low):
        return "Legal"

    # EDUCATION & TRAINING
    if re.search(r"teacher|educational consultant|lecturer|professor|tutor", x_low):
        return "Education"

    # HR / PEOPLE OPS
    if re.search(r"\bhr\b|human resources|recruit|talent", x_low):
        return "HR"

    # CONSULTING (biz/tech)
    if re.search(r"\bconsultant\b|advis(or|er)", x_low) and "educational" not in x_low:
        return "Consulting"

    # HEALTHCARE
    if re.search(r"doctor|physician|pharmacist|nurse|dentist|therap", x_low):
        return "Healthcare"

    # FINANCE & ACCOUNTING
    if re.search(r"financial analyst|accountant|auditor|finance|bank", x_low):
        return "Finance"

    # SALES & MARKETING & CX
    if re.search(r"marketing|sales|customer support|customer service|account manager|digital marketer|travel consultant", x_low):
        return "Sales_Marketing_CX"

    # TECH (data & software)
    if re.search(r"software engineer|developer|programmer|data scientist|ml|ai|analyst\b(?!.*financial)|data engineer|devops", x_low):
        return "Tech_Data"

    # RESEARCH / SCIENCE
    if re.search(r"researcher|scientist|chemist|biolog|physic|lab", x_low):
        return "Research_Science"

    # ARCHITECTURE & ENGINEERING (non-software)
    if re.search(r"architect|civil engineer|mechanical|electrical engineer|industrial engineer|architectural", x_low):
        return "Arch_Eng"

    # TRADES & OPERATIONS
    if re.search(r"plumber|electrician|chef|cook|mechanic|technician", x_low):
        return "Trades_Ops"

    # MANAGEMENT / GENERAL
    if re.search(r"\bmanager\b|management|project manager|product manager|operations manager", x_low):
        return "Management"

    # AVIATION
    if re.search(r"pilot|aircrew|flight", x_low):
        return "Aviation"

    # CREATIVE / CONTENT
    if re.search(r"graphic designer|designer|content writer|copywriter|illustrator|videograph|photograph|editor", x_low):
        return "Creative_Content"

    return "Other"

df_reduced["Profession_group"] = prof.apply(map_profession)




print(df_reduced["Profession_group"].value_counts(dropna=False).head(15))


col = "Sleep Duration"

# DÃœZELTÄ°LMÄ°Åž: np.nan yerine sadece float (NaN zaten float)
def parse_sleep_duration(x) -> float:
    """
    '5-6 hours'    -> 5.5
    'Less than 5'  -> 4.5
    'More than 8'  -> 8.5
    '9 hours'      -> 9.0
    SaÃ§ma/deÄŸersiz -> np.nan
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    # geÃ§ersiz/etiketsel deÄŸerler
    if s in {"", "na", "none", "no", "sleep_duration", "unhealthy", "moderate"}:
        return np.nan

    # less-than
    if "less" in s or s.startswith("<"):
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            up = float(m.group(1))
            return max(up - 0.5, 0.0)

    # more-than
    if "more" in s or s.startswith(">"):
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            lo = float(m.group(1))
            return lo + 0.5

    # sayÄ±/range yakalama
    nums = re.findall(r"\d+\.?\d*", s.replace(",", "."))
    if not nums:
        return np.nan

    vals = list(map(float, nums[:2]))  # ilk iki sayÄ±
    h = vals[0] if len(vals) == 1 else sum(vals) / 2.0

    # fiziksel mantÄ±k: 0.5â€“16 saat aralÄ±ÄŸÄ± dÄ±ÅŸÄ±nda ise NaN
    if not (0.5 <= h <= 16):
        return np.nan
    return h



df_reduced["Sleep_hours"] = df_reduced[col].apply(parse_sleep_duration)

# Bantlar (sÄ±ralÄ± kategori)
bins = [-np.inf, 5, 7, 9, np.inf]
labels = ["very_short", "short", "optimal", "long"]
df_reduced["Sleep_band"] = pd.cut(df_reduced["Sleep_hours"], bins=bins, labels=labels)

# HÄ±zlÄ± test
tests = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours",
         "9 hours", "40-45 hours", "No"]
for t in tests:
    print(t, "->", parse_sleep_duration(t))


df_reduced = df_reduced.drop(columns=["Sleep Duration","Profession"])


col = "Dietary Habits"

def clean_dietary(x: str) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    # saÄŸlam sÄ±nÄ±flar
    if s in {"moderate"}:
        return "Moderate"
    if s in {"healthy", "more healthy"}:
        return "Healthy"
    if s in {"unhealthy", "no healthy", "less than healthy", "less healthy"}:
        return "Unhealthy"

    # geri kalan nadir/yanlÄ±ÅŸ deÄŸerler
    return np.nan




# Temizle
df_reduced["Dietary_clean"] = df_reduced[col].apply(clean_dietary)

# Geri kalan NaN'leri "Other" veya "Moderate" ile doldur
df_reduced["Dietary_clean"] = df_reduced["Dietary_clean"].fillna("Other")

print(df_reduced["Dietary_clean"].value_counts(dropna=False))


df_reduced = df_reduced.drop(columns=["Dietary Habits"])


col = "Degree"

def clean_degree(x: str) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    # School / Pre-University
    if s in {"CLASS 12", "CLASS XII", "12", "HSC"}:
        return "School"

    # Bachelorâ€™s
    if s.startswith("B.") or s in {"BA", "BSC", "BSC.", "BBA", "BE", "BTECH", "B.TECH", "BCA", "BARCH", "B.ARCH",
                                   "BCOM", "B.COM", "BPHARM", "B.PHARM", "BHM", "LLB"}:
        return "Bachelors"

    # Masterâ€™s
    if s.startswith("M.") or s in {"MA", "M.SC", "MSC", "MBA", "MCA", "MTECH", "M.TECH", "M.ED", "MPHARM", "M.PHARM",
                                   "M.ARCH"}:
        return "Masters"

    # Doctoral / Professional
    if s in {"PHD", "PH.D", "MD", "MBBS", "LLM", "ME", "MHM"}:
        return "Doctoral_Professional"

    # Other / Unknown
    return "Other"





df_reduced["Degree_clean"] = df_reduced[col].apply(clean_degree)

# Eksikleri 'Other' ile doldur
df_reduced["Degree_clean"] = df_reduced["Degree_clean"].fillna("Other")

print(df_reduced["Degree_clean"].value_counts())


df_reduced = df_reduced.drop(columns=["Degree"])


df_reduced.nunique()


ohe_cols = [col for col in df_reduced.columns if 15 >= df_reduced[col].nunique() > 2]


ohe_cols


df_reduced.shape


df1 = df_reduced.copy()


df1 = one_hot_encoder(df1, ohe_cols)


df1.head()





#scaler = RobustScaler()
#df1[num_cols] = scaler.fit_transform(df1[num_cols])

#df1[num_cols].head()


cat_cols, num_cols, cat_but_car = grab_col_names(df1)


cat_cols


num_cols


df1.head()


y = df1["Depression"]
X = df1.drop(["Depression","Name","City"],axis=1)


# Train- Test Split
# Train-Test Separation
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=42)


X_train.shape


y_train.shape


models = []
models.append(('LR', LogisticRegression(random_state = 42)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state = 42)))
models.append(('RF', RandomForestClassifier(random_state = 42)))
models.append(('SVM', SVC(gamma='auto', random_state = 42)))
models.append(('XGB', GradientBoostingClassifier(random_state = 42)))
models.append(("LightGBM", LGBMClassifier(random_state = 42, verbosity=-1)))
models.append(("CatBoost", CatBoostClassifier(random_state = 42, verbose = False)))

# evaluate each model in turn
results = []
names = []


for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        msg = "%s: (%f)" % (name, accuracy)
        print(msg)





for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")





X_train, X_test, y_train, y_test,classifiers = train_test(X,y)


from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report





model_evaluation(classifiers, X_test, y_test, X_train, y_train)


classifiers = [  #('LR', LogisticRegression()),
        # ('KNN', KNeighborsClassifier()),
        # ("SVC", SVC()),
        ("CART", DecisionTreeClassifier(random_state=0)),
        ("RF", RandomForestClassifier(random_state=0, max_features='sqrt')),
        # ('Adaboost', AdaBoostClassifier(random_state=0)),
        # ('GBM', GradientBoostingClassifier(max_depth=4,random_state=0)),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(random_state=0, verbose=-1)),
        #('CatBoost', CatBoostClassifier(verbose=False))
    ]


X_train, X_test, y_train, y_test,classifiers = train_test(X,y)


models2(classifiers,X,y)





# Feature Importance chart of each classifier model
feature_importances(classifiers,X,y,count=15)





# CART (DecisionTreeClassifier)
cart_params = {
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3]
}

# RandomForestClassifier
rf_params = {
    "n_estimators": [300, 600],
    "max_depth": [None, 12, 20],
    "max_features": ["sqrt", 0.5],          # sqrt ya da %50 Ã¶zellik
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    # "class_weight": ["balanced"]          # dengesizlik yÃ¼ksekse ekle
}

# XGBoost (XGBClassifier)
xgboost_params = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 600],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1, 5]
}

# LightGBM (LGBMClassifier)
lightgbm_params = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 600],
    "num_leaves": [31, 63],
    "max_depth": [-1, 10],
    "min_child_samples": [20, 40],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# CatBoost (CatBoostClassifier)
catboost_params = {
    "learning_rate": [0.05, 0.1],
    "depth": [6, 8],
    "iterations": [500, 800],               # erken durdurmayÄ± kullanacaksan daha da daralt
    "l2_leaf_reg": [3, 5],
    "bagging_temperature": [0.0, 1.0],
    "auto_class_weights": [None, "Balanced", "SqrtBalanced"]
    # "loss_function": ["Logloss"],          # istersen sabitle
}

classifiers_hyp = [#('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(random_state=42), cart_params),
               ("RF", RandomForestClassifier(random_state=42), rf_params),
               ('XGBoost', xgb.XGBClassifier(eval_metric='logloss',random_state=42), xgboost_params)]
               #('LightGBM', LGBMClassifier(random_state=42,verbose=-1), lightgbm_params),
                #('CatBoost', CatBoostClassifier(verbose=False),catboost_params)]



best_models = hyperparameter_optimization(X,y,classifiers_hyp)



best_models


import time, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone

def _extract_estimator(obj):
    """best_models deÄŸerinden (estimator ya da dict) gerÃ§ek estimator'Ä± Ã§Ä±kar."""
    if hasattr(obj, "fit"):      # zaten estimator
        return obj
    if isinstance(obj, dict):
        for k in ["final_model", "best_estimator_", "estimator", "model"]:
            if k in obj and hasattr(obj[k], "fit"):
                return obj[k]
    raise TypeError(f"Estimator bekleniyordu, gelen tip: {type(obj)} -> {obj}")

def select_and_save_best_by_f1(best_models, X, y, cv_splits=5, out_path="best_model_f1.joblib"):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    rows, name_to_est = [], {}
    for name, obj in best_models.items():
        est = _extract_estimator(obj)
        name_to_est[name] = est
        scores = cross_validate(
            est, X, y, cv=cv, n_jobs=-1,
            scoring=["f1", "roc_auc", "accuracy", "precision", "recall"],
            return_train_score=False
        )
        rows.append({
            "Model": name,
            "CV_F1":        float(scores["test_f1"].mean()),
            "CV_AUC":       float(scores["test_roc_auc"].mean()),
            "CV_Accuracy":  float(scores["test_accuracy"].mean()),
            "CV_Precision": float(scores["test_precision"].mean()),
            "CV_Recall":    float(scores["test_recall"].mean()),
        })

    leaderboard = pd.DataFrame(rows).sort_values("CV_F1", ascending=False).reset_index(drop=True)
    best_name = leaderboard.loc[0, "Model"]
    best_est  = clone(name_to_est[best_name]).fit(X, y)

    artifact = {
        "model": best_est,
        "model_name": best_name,
        "selected_metric": "CV_F1",
        "metric_value": float(leaderboard.loc[0, "CV_F1"]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": list(X.columns) if hasattr(X, "columns") else None,
    }
    joblib.dump(artifact, out_path)

    print("Leaderboard:")
    print(leaderboard)
    print(f"\nKaydedildi â†’ {out_path} | SeÃ§ilen: {best_name} (CV_F1={leaderboard.loc[0,'CV_F1']:.4f})")
    return best_est, leaderboard, out_path



best_estimator, leaderboard, path = select_and_save_best_by_f1(best_models, X, y, cv_splits=5,
                                                               out_path="best_model_f1.joblib")









def main():
    df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/kaggle_dataset/exploring_mental_health/train.csv", delimiter=',')
    X, y = churn_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X,y,classifiers_hyp)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf


#if __name__ == "__main__":
 #   print("Ä°ÅŸlem baÅŸladÄ±")
 #   main()







