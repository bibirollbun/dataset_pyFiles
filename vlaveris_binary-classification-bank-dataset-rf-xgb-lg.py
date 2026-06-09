pip uninstall scikit-learn imbalanced-learn -y


pip install scikit-learn imbalanced-learn


pip install --upgrade xgboost scikit-learn


!pip install xgboost==1.7.6 scikit-learn==1.3.2


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


class DataAnalysis:
    def __init__(self, df):
        self.df = df

    # shows data
    def show_data(self):
        print(self.df.head())

    # shows dimensions
    def show_dimensions(self):
        print(self.df.shape)

    # shows statistics
    def statistics(self):
        print(self.df.describe())

    # Lets check missing values
    def missing_values(self):
        for feature in self.df.columns:
            missing_values = self.df[feature].isna().sum()
            percentage = missing_values/len(self.df) * 100
            print(f'{feature} - {missing_values} - {round(percentage, 1)} %')

    # draws a graph
    def draw_graph(self, feature, title, xlabel):
        counts = self.df[feature].value_counts()

        plt.figure(figsize=(10, 6))

        bars = plt.bar(range(len(counts)), counts.values, color='blue')

        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        plt.xticks(range(len(counts)), counts.index, rotation=45, ha='right')

        # for i, v in enumerate(counts.values):
        #     plt.text(i, v, str(v), ha='center', va='bottom')

        plt.tight_layout()

        plt.show()

    # Encodes cathegorical variables
    def encode(self):
        for feature in self.df.columns:
            if self.df[feature].dtype == "object":
                self.df = pd.get_dummies(self.df, columns=[feature], drop_first=True, dtype=float)
        return self.df

    # Check correlations
    def prints_correlations(self):
        corr = self.df.corr()
        return corr.style.background_gradient(cmap='coolwarm')

    # Removes features
    def remove_feature(self, feature):
        self.df = self.df.drop([feature], axis=1)
        return self.df

    def select_variables_from_heatmap(self, threshold=0.8):
        corr = self.df.corr().abs()
    
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
        to_drop = [column for column in upper.columns 
               if any(upper[column].dropna() > threshold) and column != "y"]
    
        df_reduced = self.df.drop(self.df[to_drop], axis=1)
        return df_reduced, to_drop


class Balancing:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def random_undersampling(self):
        rus = RandomUnderSampler(sampling_strategy=1)

        X_res, y_res = rus.fit_resample(self.x, self.y)

        ax = y_res.value_counts().plot.pie(autopct='%.2f')
        _ = ax.set_title("Under-sampling")

        return X_res, y_res

    def random_oversampling(self):
        ros = RandomOverSampler(sampling_strategy="not majority")
        X_res, y_res = ros.fit_resample(self.x, self.y)

        ax = y_res.value_counts().plot.pie(autopct='%.2f')
        _ = ax.set_title("Over-sampling")
        return X_res, y_res


class Classifying:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def random_forest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)

        param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

        rf = RandomForestClassifier()

        rand_search = RandomizedSearchCV(rf, 
                                         param_distributions = param_dist, 
                                         n_iter=5,
                                         cv=5)

        rand_search.fit(X_train, y_train)
        
        best_rf = rand_search.best_estimator_

        print('Best hyperparameters:',  rand_search.best_params_)

        y_pred = best_rf.predict(X_test)

        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy on Test Set: {accuracy:.4f}")
        print(f"ROC AUC on Test Set: {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return best_rf

    def xgb(self):
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [300, 400, 500]
        }

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        outer_scores = []

        for train_idx, test_idx in outer_cv.split(self.x, self.y):
            X_train, X_test = self.x.iloc[train_idx], self.x.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]


        model = XGBClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        print('Best hyperparameters:',  grid_search.best_params_)
        
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy on Test Set: {accuracy:.4f}")
        print(f"ROC AUC on Test Set: {roc_auc:.4f}")

        # Display a detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return best_model

    def light_gbm(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.25, random_state=42, stratify=self.y)
        
        lgbm = lgb.LGBMClassifier(random_state=42, num_boost_round=100, verbosity=-1)
        
        param_grid = {
            'n_estimators': [300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [3, 5, 7]
        }
    
        grid_search = GridSearchCV(
            estimator=lgbm,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=2 
        )
        
        print("Starting Grid Search...")
        grid_search.fit(X_train, y_train)

        print("\nGrid Search Complete.")
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best ROC AUC score during cross-validation: {grid_search.best_score_:.4f}")

        print("\n--- Evaluating Best Model on Test Set ---")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy on Test Set: {accuracy:.4f}")
        print(f"ROC AUC on Test Set: {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return best_model

        


analysis = DataAnalysis(train_df)


analysis.show_data()


analysis.show_dimensions()


analysis.statistics()


analysis.missing_values()


analysis.draw_graph("age", "Age distribution", "Age")


analysis.draw_graph("job", "Job distribution", "Job")


analysis.draw_graph("marital", "Marital distribution", "Marital")


analysis.draw_graph("education", "Education distribution", "Education")


analysis.draw_graph("default", "Default distribution", "Default")


analysis.draw_graph("housing", "Housing distribution", "Housing")


analysis.draw_graph("loan", "Loan distribution", "Loan")


analysis.draw_graph("contact", "Contact distribution", "Contact")


analysis.draw_graph("day", "Day distribution", "Day")


analysis.draw_graph("month", "Month distribution", "Month")


analysis.draw_graph("campaign", "Campaign distribution", "Campaign")


analysis.draw_graph("poutcome", "Poutcome", "Paoutcome")


analysis.draw_graph("y", "Y distribution", "Y")


analysis.encode()


analysis.remove_feature("id")


analysis.prints_correlations()


df_reduced, dropped = analysis.select_variables_from_heatmap(threshold=0.8)


print(dropped)


y = df_reduced["y"]


y


x = df_reduced.drop("y", axis=1)


balancing = Balancing(x, y)


X_un, y_un = balancing.random_undersampling()


X_ov, y_ov = balancing.random_oversampling()


classification_un = Classifying(X_un, y_un)


rf_model = classification_un.random_forest()


xg_model = classification_un.xgb()


lg_model = classification_un.light_gbm()


ids = test_df["id"]


test_df = test_df.drop(["id"], axis=1)


test_analysis = DataAnalysis(test_df)


test_analysis.missing_values()


test_analysis.encode()


test_data = test_analysis.remove_feature("poutcome_unknown")


for feature in x.columns:
    if feature not in test_data.columns:
        print(feature)


pred = xg_model.predict_proba(test_data)


pred


y = pred[:, 1]


predictions = pd.DataFrame({'y': y})


predictions['y'] = predictions['y'].round(1)


predictions


submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


submission


submission = submission.drop(['y'], axis=1)


submission['y'] = predictions


submission


submission.set_index('id', inplace=True)


submission.to_csv('submission.csv', index=True) 




