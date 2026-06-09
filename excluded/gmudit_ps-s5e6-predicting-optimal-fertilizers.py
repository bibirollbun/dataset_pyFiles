# Imports
import os
import sys
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from scipy.stats import ks_2samp
from catboost import CatBoostClassifier,Pool
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import shap


# Configuration
class CFG:
    seed = 42
    debug = False
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    target = 'Fertilizer Name'

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


set_seed(CFG.seed)
print(f"Debugging: {CFG.debug}\nUsing devide: {CFG.device}, Seed set to {CFG.seed}")


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df


df.info()



test.info()


# Check for any row in test that exists in train
is_subset = test.drop(columns = ["id"]).isin(df.drop(columns = ["id","Fertilizer Name"])).all(axis=1)
print("Rows in test also in train:", is_subset.sum(), "out of", len(test))

# Check if any row is common in both train and test
common = pd.merge(test.drop(columns = ["id"]), df.drop(columns = ["id","Fertilizer Name"]), how='inner')
print("Number of common rows:", len(common))



test


class DataInspector:
    def __init__(self, df, name="Dataset"):
        self.df = df
        self.name = name
        self.cat_cols = self.get_categorical_columns()
        self.cont_cols = self.get_continuous_columns()

    def get_categorical_columns(self):
        return [col for col in self.df.columns if self.df[col].dtype == "object"]

    def get_continuous_columns(self):
        return [col for col in self.df.columns if col not in self.cat_cols and col != "id"]

    def display_column_types(self):
        print(f"\nğŸŸ¦ Categorical Columns in {self.name}: {self.cat_cols}")
        print(f"ğŸŸ© Continuous Columns in {self.name}: {self.cont_cols}\n")

    def display_unique_values(self):
        print(f"\nğŸ”� Unique Values in {self.name}:\n")

        for col in self.cat_cols:
            unique_vals = sorted(self.df[col].unique())
            print(f"ğŸ“Œ {col} [Categorical] â�œ {unique_vals}")

        for col in self.cont_cols:
            unique_vals = sorted(self.df[col].unique())
            print(f"ğŸ“ˆ {col} [Continuous] â�œ {unique_vals}")


# Usage Example:

train_inspector = DataInspector(df, name="Train")
test_inspector = DataInspector(test, name="Test")

train_inspector.display_column_types()
test_inspector.display_column_types()

train_inspector.display_unique_values()
test_inspector.display_unique_values()

cat_cols = test_inspector.get_categorical_columns()
cont_cols = test_inspector.get_continuous_columns()


class DatasetComparer:
    def __init__(self, train_df, test_df):
        self.train = train_df
        self.test = test_df

    def compare_numerical_distributions(self, numerical_cols):
        print("\nğŸ”� Kolmogorov-Smirnov (KS) Test Results:\n")
        for col in numerical_cols:
            plt.figure(figsize=(6, 4))
            sns.kdeplot(self.train[col], label='Train', fill=True)
            sns.kdeplot(self.test[col], label='Test', fill=True)
            plt.title(f"Distribution of {col}")
            plt.legend()
            plt.tight_layout()
            plt.show()

            stat, p = ks_2samp(self.train[col], self.test[col])
            print(f"{col:<15}: KS Statistic = {stat:.3f}, p-value = {p:.3f}")

    def compare_categorical_counts(self, categorical_cols):
        for col in categorical_cols:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=col, data=self.train, color='blue', alpha=0.6, label='Train')
            sns.countplot(x=col, data=self.test, color='orange', alpha=0.6, label='Test')
            plt.title(f"Count Comparison of {col}")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


# âœ… Usage Example
comparer = DatasetComparer(df, test)
comparer.compare_numerical_distributions(cont_cols)
print("Plots for Categorical Features")
comparer.compare_categorical_counts(cat_cols)



class FertilizerModelExplainer:
    def __init__(self, df, target_col, id_col, categorical_cols, use_gpu=True):
        self.df = df
        self.target_col = target_col
        self.id_col = id_col
        self.categorical_cols = categorical_cols
        self.X = df.drop(columns=[id_col, target_col])
        self.y = df[target_col]
        self.model = CatBoostClassifier(cat_features=categorical_cols, task_type='GPU' if use_gpu else 'CPU', verbose=0)
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = self.label_encoder.classes_

    def train_model(self):
        self.model.fit(self.X, self.y)

    def show_feature_importances(self):
        importances = self.model.get_feature_importance()
        print("\nğŸ“Š Feature Importances:")
        for name, score in zip(self.X.columns, importances):
            print(f"{name:<15}: {score:.2f}")
    def pp(self):
        print(self.X)
    def explain_with_shap(self):
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.df[self.target_col])  # or use your encoded y
        class_names = le.classes_ 
        explainer = shap.TreeExplainer(self.model)
        X = self.df.drop(columns=["id","Fertilizer Name"])
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, class_names=class_names)

# âœ… Usage

categorical_features = ['Soil Type', 'Crop Type']
explainer = FertilizerModelExplainer(
    df=df,
    target_col="Fertilizer Name",
    id_col="id",
    categorical_cols=categorical_features,
    use_gpu=True
)

explainer.train_model()
explainer.show_feature_importances()
explainer.explain_with_shap()



def mapk(actuals, predictions, k=3):
    """
    Mean Average Precision at K
    """
    def apk(actual, pred, k):
        if actual in pred[:k]:
            return 1 / (pred[:k].index(actual) + 1)
        return 0

    return np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)])



# class FertilizerClassifier:
#     def __init__(self, cat_features, params, seed, debug):
#         self.seed = seed
#         self.debug = debug
#         self.cat_features = cat_features
#         self.params = params
#         self.model = CatBoostClassifier(**params)

#     def train(self, df, target_column, id_column, test_size=0.2):
#         X = df.drop(columns=[id_column, target_column])
#         y = df[target_column]

#         X_train, X_valid, y_train, y_valid = train_test_split(
#             X, y, test_size=test_size, random_state=self.seed, stratify=y
#         )

#         train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
#         valid_pool = Pool(X_valid, y_valid,  cat_features=self.cat_features)

#         self.model.fit(train_pool, eval_set=valid_pool, verbose=100)
#         sys.stdout.flush()

#         # Evaluate MAP@3 on validation set
#         probs = self.model.predict_proba(valid_pool)
#         top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
#         class_names = self.model.classes_
#         top3_labels = [[class_names[i] for i in row] for row in top3_preds]

#         score = mapk(y_valid.tolist(), top3_labels, k=3)
#         print(f"\nâœ… MAP@3 on validation set: {score:.4f}")
#         sys.stdout.flush()

#     def predict(self, test_df, droppable_column, top_k=3):
#         sys.stdout.flush()

#         test_df = test_df.copy()
#         if droppable_column in test_df.columns:
#             test_df.drop(droppable_column, axis=1, inplace=True)

#         if isinstance(test_df, pd.Series):
#             test_df = test_df.to_frame().T

#         test_pool = Pool(data=test_df, cat_features=self.cat_features)

#         if self.model is None:
#             raise ValueError("Model has not been trained yet")

#         probs = self.model.predict_proba(test_pool)
#         topk_indices = np.argsort(probs, axis=1)[:, -top_k:][:, ::-1]
#         class_names = self.model.classes_
#         topk_labels = [[class_names[i] for i in row] for row in topk_indices]

#         return topk_labels



# Create classifier
# classifier = FertilizerClassifier(
#     cat_features=cat_cols_test,
#     params=CFG.cat_boost_params,
#     seed=CFG.seed,
#     debug=CFG.debug
# )

# # Train and evaluate
# classifier.train(df, target_column=CFG.target, id_column="id")

# # Predict top-3 fertilizers
# top3_preds = classifier.predict(test, droppable_column="id", top_k=3)



# top3_labels = [" ".join(x) for x in top3_preds]


# submission = pd.DataFrame({
#     "id": test["id"],
#     "Fertilizer Name": top3_labels
# })
# submission
# submission.to_csv("submission.csv",index=False)


class XBGFertilizerClassifier:
    def __init__(self,data,target,seed = 42):
        self.seed = seed
        self.data  = data
        self.params = {
            'device': 'cuda',      # for GPU training
            'tree_method': 'hist', # GPU-capable histogram algorithm
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42
            }
        self.model = XGBClassifier(**self.params)
        self.target = target
        
    def encode_cats(self,test):
        le_target = LabelEncoder()
        self.data["Fertilizer Label"] = le_target.fit_transform(self.data["Fertilizer Name"])
        cat_cols = [cols for cols in test.columns if test[cols].dtype == "object"]

        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            test[col] = le.fit_transform(test[col])
            encoders[col] = le
        return test ,le_target

    def XYtest(self,test,features):
        self.X = self.data[features]
        self.Y = self.data["Fertilizer Label"]
        self.test = test[features]
        

    def train_validate(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.Y, test_size=0.2, 
                                                          random_state=self.seed, stratify=self.Y)
        self.model.fit(X_train,y_train)
        preds = self.model.predict(X_val)
        print(f"Accuracy Score = {accuracy_score(y_val,preds)}")

    def run_test(self, le_target):
        test_probs = self.model.predict_proba(self.test)
        top3 = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
        top3_labels = [[le_target.inverse_transform([i])[0] for i in row] for row in top3]
        return top3_labels

features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous']


xgbclassifier = XBGFertilizerClassifier( df, CFG.target,seed = CFG.seed)
test_df,le_target = xgbclassifier.encode_cats(test)
xgbclassifier.XYtest(test_df,features)
xgbclassifier.train_validate()
test_predictions = xgbclassifier.run_test(le_target)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in test_predictions]
})
submission.head()


# Save to CSV
submission.to_csv('submission.csv', index=False)

