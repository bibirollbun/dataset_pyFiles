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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train_df.shape, test_df.shape


# columns 
train_df.columns


# shape of the dataframe
train_df.shape


# dataframe information
train_df.info()


# stats details
train_df.describe()


# null check
train_df.isnull().sum()


# duplicates check
train_df.duplicated().sum()


print(train_df.info())
train_df.head()


train_df.columns


df = train_df[['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'rainfall']]


plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True)
plt.show()


df.columns


sns.displot(df.corr()['rainfall'])
plt.show()


## split the data into training and for test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['rainfall']),
                                                    df['rainfall'],
                                                    test_size=0.25,
                                                    random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)


y_pred = sgd_clf.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report

print("accuracy:", accuracy_score(y_pred, y_test))
print("Report: ",classification_report(y_pred, y_test))


from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_test)


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)



# X_train_folds = X_train.iloc[train_index]  # Use `.iloc[]` for Pandas
# y_train_folds = y_train.iloc[train_index]


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize classifier
sgd_clf = SGDClassifier(random_state=42)

# Stratified K-Fold Cross-Validation
for train_index, test_index in skfolds.split(X_train, y_train):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train.iloc[train_index]  # Use .iloc for Pandas
    y_train_folds = y_train.iloc[train_index]
    X_test_fold = X_train.iloc[test_index]
    y_test_fold = y_train.iloc[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)

    n_correct = sum(y_pred == y_test_fold)
    accuracy = n_correct / len(y_pred)
    
    print(f"Fold Accuracy: {accuracy:.2f}")


from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_train_pred)


# fit the sgd classifier algorithms
sgd_clf.fit(X_train, y_train)


from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_train, sgd_clf.predict(X_train))
recall = recall_score(y_train, sgd_clf.predict(X_train))
print("precision: ", precision)
print("recall: ", recall)


from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, sgd_clf.predict(X_test))
recall = recall_score(y_test, sgd_clf.predict(X_test))
print("precision: ", precision)
print("recall: ", recall)


from sklearn.metrics import f1_score
f1_tr = f1_score(y_test, sgd_clf.predict(X_test))
f1_te = f1_score(y_train, sgd_clf.predict(X_train))
print("f1 score training: ", f1_tr)
print("f1 score test: ", f1_te)


y_score = cross_val_predict(sgd_clf, X_train, y_train, cv=3,
                            method="decision_function")


from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_train, y_score)


def plot_precision_recall_vs_threshold(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "b--", label='precision')
    plt.plot(threshold, recall[:-1], "g--", label="recall")
    plt.legend()
    plt.xlabel("Thresholds")

plot_precision_recall_vs_threshold(precision, recall, threshold)
plt.show()


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_score)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal

plot_roc_curve(fpr, tpr)
plt.show()


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train, y_score)


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

scaling =  StandardScaler()
X_train_scaled = scaling.fit_transform(X_train)
X_test_scaled = scaling.fit_transform(X_test)


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

y_probs_forest = cross_val_predict(rf_model, X_train, y_train,
                                   cv=3, method='predict_proba')


# cross_val_predict??


y_scores_forest = y_probs_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train,y_scores_forest)


plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()


roc_auc_score(y_train, y_scores_forest)


from sklearn.ensemble import RandomForestClassifier
def model_training(X, y):
    model=RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


model = model_training(X=X_train, y=y_train)
model


from sklearn.metrics import precision_score, recall_score, accuracy_score
def model_evaluate(model, X, y):
    y_pred = model.predict(X)
    precision = precision_score(y_pred, y)
    recall = recall_score(y_pred, y)
    acc = accuracy_score(y_pred, y)
    print("precision: ", precision)
    print("recall: ", recall)
    print("Accuracy score: ", acc)


model_evaluate(model=model, X=X_test, y=y_test)


model_evaluate(model, X_train, y_train)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
importance = model.feature_importances_

for feature, score in zip(X_train.columns, importance):
    print(f"{feature}: {score:.4f}")


selected_features = []
for feature, score in zip(X_train.columns, importance):
    selected_features.append(feature)
    print(f"{feature}")


rf_features = df[selected_features]
rf_features['rainfall'] = df['rainfall']


rf_features



## split the data into training and for test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(rf_features.drop(columns=['rainfall']),
                                                    rf_features['rainfall'],
                                                    test_size=0.25,
                                                    random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


corr_matrix = rf_features.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


preprocessor = ColumnTransformer(
    transformers = [
        ("scaler", StandardScaler(), X_train.columns)
    ], remainder='passthrough'
)


y_train.dtypes


pipeline = Pipeline(
    steps = [
        ("scaler", preprocessor),
        ("model", RandomForestClassifier())
    ]
)


pipeline.fit(X_train, y_train)


model_evaluate(pipeline, X_train, y_train)


model_evaluate(pipeline, X_test, y_test)





def model_trainig(model, train, test):

    # transform the data 
    preprocessor = ColumnTransformer(
    transformers = [
        ("scaler", StandardScaler(), X_train.columns)
        ], remainder='passthrough'
    )

    # create a pipeline for model and scaling numerical features
    pipeline = Pipeline(
    steps = [
        ("scaler", preprocessor),
        ("model", model)
        ]
    )
    
    model = pipeline.fit(train, test)
    return model


model = model_trainig(model=RandomForestClassifier(), train=X_train, test=y_train)


model


# evaluate on training data
model_evaluate(model, X_train, y_train)


# evaluate on test data
model_evaluate(model, X_test, y_test)


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
selector = RFE(model, n_features_to_select=5)
selector.fit(X_train, y_train)

print("Selected features:", X_train.columns[selector.support_])


rfe_features = df[['maxtemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'rainfall']]


## split the data into training and for test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(rfe_features.drop(columns=['rainfall']),
                                                    rfe_features['rainfall'],
                                                    test_size=0.25,
                                                    random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


model1 = model_trainig(model=RandomForestClassifier(), train=X_train, test=y_train)


# evaluate on training data
model_evaluate(model, X_train, y_train)


# evaluate on test data
model_evaluate(model, X_test, y_test)


main_df = df[['dewpoint', 'humidity', "cloud", 'winddirection', "windspeed", "rainfall"]]


corr_mxtrix = main_df.drop(columns='rainfall').corr()
sns.heatmap(corr_mxtrix, annot=True, center=True)
plt.show()


model3 = model_training(X_train, y_train)


model_evaluate(model3, X_test, y_test)


model_evaluate(model3, X_train, y_train)


test = df[model.feature_names_in_]
test.sample(10)


df['rainfall'].value_counts()


df.sample(2)


def model_testing(idx=0, df=df):
    df1 = df[model.feature_names_in_]
    input_data = df1.iloc[[idx]]
    actual = df['rainfall'].iloc[idx]
    prediction = model.predict(input_data)[0]
    print("Actual: ", actual)
    print("Predicted: ", prediction)


model_testing(idx=1484, df=df)


model_testing(idx=1255, df=df)


model_testing(idx=1528, df=df)


!pip install scikit-learn==1.7.0


!pip install imblearn


!pip uninstall sklearn -y
!pip uninstall scikit-learn -y
!pip install scikit-learn


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_resample(X, y)




counter = Counter(y_train)
print('Before', counter)

# oversampling the train dataset using SMOTE
smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_sm)
print('After', counter)




















import pickle
# save model
with open("rainfall_rf_ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

# save the 


test_df = test_df[model.feature_names_in_]


submition_df = {
    "id": [],
    "rainfall": []
}













