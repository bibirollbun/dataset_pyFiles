import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

import warnings; warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/optimizingdefaultmodelbyfirstpaymentdefault/kaggle_dataset.csv", index_col="ID")
df


print("Number NaN each feature")
for feature in df.columns:
    print(f"{feature}: {df[feature].isna().sum()}")
print(f"Total NaN: {df.isna().sum().sum()}")


df = df.dropna(axis=0)
df


df.info()


df['target'].unique()


X, y = df.drop(columns=['target']), df['target']
X.shape, y.shape


scaler = MinMaxScaler()

normalized_X = scaler.fit_transform(X)
normalized_X.shape


X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.2, random_state=42)


search_space = {
    "criterion": Categorical(["gini", "entropy", "log_loss"]),
    "splitter": Categorical(["best", "random"]),
    "max_depth": Integer(1, 20),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 20),
    "max_features": Categorical([None, "sqrt", "log2"])
}

opt = BayesSearchCV(
    estimator=DecisionTreeClassifier(),
    search_spaces=search_space,
    n_iter=32,                
    cv=5,                      
    n_jobs=-1,                 
    random_state=42,
    verbose=1
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))


dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)


pred = dt_model.predict(X_test)
pred


accuracy = accuracy_score(y_test, pred)
roc = roc_auc_score(y_test, pred)
ks = ks_2samp(y_test, pred)


print(f"""
accuracy: {accuracy}
roc: {roc}
ks: {ks.statistic}
""")

