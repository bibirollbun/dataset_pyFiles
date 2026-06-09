import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df_train = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/train.csv")
df_test = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/test.csv")

#### DATA PREPARATION

train_features = df_train.columns.tolist()
test_features = df_test.columns.tolist()
features_toremove =  list(set(train_features) - set(test_features) - {'sii'})

del df_train['id']

for col in features_toremove:
    del df_train[col]

df_train.dropna(subset=['sii'], inplace=True)

df_train = df_train[df_train['Physical-BMI']!=0]
df_train = df_train[df_train['Physical-Weight']!=0]

physical_measures_df = pd.read_csv('/kaggle/input/nhanes-physical-measures/physical_measures.csv')

df_train = df_train.merge( physical_measures_df, on=['Basic_Demos-Age', 'Basic_Demos-Sex'], suffixes=('', '_avg'))
cols = ['Physical-BMI','Physical-Height','Physical-Weight','Physical-Waist_Circumference','Physical-Diastolic_BP','Physical-HeartRate','Physical-Systolic_BP']
tot_nan_phys = df_train[cols].isna().all(axis=1)

for col in cols:
    df_train.loc[tot_nan_phys, col] = df_train.loc[tot_nan_phys, f"{col}_avg"]
    del df_train[f"{col}_avg"]

threshold = int(df_train.shape[1] * 0.35)
df_train.dropna(thresh=threshold, inplace=True)

X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in X.dtypes])  
numerical_idx = np.flatnonzero(is_numerical) 
new_X_train = X_train.iloc[:, numerical_idx]
new_X_test = X_test.iloc[:, numerical_idx]


scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=3)

scaled_train = scaler.fit_transform(new_X_train)
X_array = imputer.fit_transform(scaled_train)
X_array = scaler.inverse_transform(X_array)
new_X_train = pd.DataFrame(X_array, columns=new_X_train.columns, index=new_X_train.index) # convert into a dataframe since X_array is of type ndarray

scaled_test = scaler.fit_transform(new_X_test)
X_array = imputer.fit_transform(scaled_test)
X_array = scaler.inverse_transform(X_array)
new_X_test = pd.DataFrame(X_array, columns=new_X_test.columns, index=new_X_test.index)

categorical_idx = np.flatnonzero(is_numerical==False)
categorical_X_train = X_train.iloc[:, categorical_idx]
categorical_X_test = X_test.iloc[:, categorical_idx]

imputer = SimpleImputer(strategy='most_frequent')
X_array = imputer.fit_transform(categorical_X_train)
categorical_X_train = pd.DataFrame(X_array, columns=categorical_X_train.columns, index=categorical_X_train.index)

X_array = imputer.fit_transform(categorical_X_test)
categorical_X_test = pd.DataFrame(X_array, columns=categorical_X_test.columns, index=categorical_X_test.index)


oh = OneHotEncoder(sparse_output=False)

oh.fit(categorical_X_train)
encoded = oh.transform(categorical_X_train)

for i, col in enumerate(oh.get_feature_names_out()):
    new_X_train = new_X_train.copy()
    new_X_train[col] = encoded[:, i]

oh.fit(categorical_X_test)
encoded = oh.transform(categorical_X_test)

for i, col in enumerate(oh.get_feature_names_out()):
    new_X_test = new_X_test.copy()
    new_X_test[col] = encoded[:, i]


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

base_model = RandomForestClassifier()
parameters = { 'n_estimators': [50, 100, 200],
    'max_leaf_nodes': [50, 80, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
    }
tuned_model = GridSearchCV(base_model, parameters, cv=5, scoring='accuracy', n_jobs=-1)
tuned_model.fit(new_X_train, y_train)
print ("Best Score: {:.3f}".format(tuned_model.best_score_) )
print("Best Params: ", tuned_model.best_params_)
test_acc = accuracy_score(y_true = y_test, y_pred = tuned_model.predict(new_X_test) )
print("Test Accuracy: {:.3f}".format(test_acc) )


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    estimator=tuned_model.best_estimator_,
    X=new_X_test, y=y_test,
    cmap = 'Blues_r')


base_model = RandomForestClassifier(class_weight='balanced') # give weight to the class that are inversely proportional to frequency
parameters = { 'n_estimators': [50, 100, 200],
    'max_leaf_nodes': [50, 80, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
    }
tuned_model = GridSearchCV(base_model, parameters, cv=5, scoring='accuracy', n_jobs=-1)
tuned_model.fit(new_X_train, y_train)
print ("Best Score: {:.3f}".format(tuned_model.best_score_) )
print("Best Params: ", tuned_model.best_params_)
test_acc = accuracy_score(y_true = y_test, y_pred = tuned_model.predict(new_X_test) )
print("Test Accuracy: {:.3f}".format(test_acc) )


ConfusionMatrixDisplay.from_estimator(
    estimator=tuned_model.best_estimator_,
    X=new_X_test, y=y_test,
    cmap = 'Blues_r')


feature_names = new_X_train.columns.tolist()

important_features = [name for name, importance in zip(feature_names, tuned_model.best_estimator_.feature_importances_) if importance > 0.025]
print(important_features)

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(range(new_X_train.shape[1]), sorted(tuned_model.best_estimator_.feature_importances_)[::-1])
ax.set_title("Feature Importances")
ax.set_yticks(range(new_X_train.shape[1]))
ax.set_yticklabels(np.array(feature_names)[np.argsort(tuned_model.best_estimator_.feature_importances_)[::-1]])
ax.invert_yaxis() 
ax.grid()


from sklearn.feature_selection import RFECV
model = RandomForestClassifier(class_weight='balanced', bootstrap=True, max_depth=20, max_leaf_nodes=100, min_samples_split=5, n_estimators=200)
selector = RFECV(model, step=5, cv=5, scoring='accuracy', n_jobs=-1)
selector.fit(new_X_train, y_train)
X_train_subset = new_X_train.iloc[:, selector.support_]
X_test_subset = new_X_test.iloc[:, selector.support_]
print(new_X_train.shape[1])
print(X_train_subset.shape[1])
X_train_subset.columns.to_list()

model.fit(X_train_subset, y_train)
test_acc = accuracy_score(y_true = y_test, y_pred = model.predict(X_test_subset) )
print("Test Accuracy: {:.3f}".format(test_acc) )


ConfusionMatrixDisplay.from_estimator(
    estimator=model,
    X=X_test_subset, y=y_test,
    cmap = 'Blues_r')


subset_feature_names = X_train_subset.columns.tolist()

important_features = [name for name, importance in zip(subset_feature_names, model.feature_importances_) if importance > 0.035]
print(important_features)

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(range(X_train_subset.shape[1]), sorted(model.feature_importances_)[::-1])
ax.set_title("Feature Importances")
ax.set_yticks(range(X_train_subset.shape[1]))
ax.set_yticklabels(np.array(subset_feature_names)[np.argsort(model.feature_importances_)[::-1]])
ax.invert_yaxis() 
ax.grid()


# Process the Data Set
df_test = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/test.csv")
ids = df_test['id']
del df_test['id']

physical_measures_df = pd.read_csv('/kaggle/input/nhanes-physical-measures/physical_measures.csv')

df_test = df_test.merge( physical_measures_df, on=['Basic_Demos-Age', 'Basic_Demos-Sex'], suffixes=('', '_avg'))
cols = ['Physical-BMI','Physical-Height','Physical-Weight','Physical-Waist_Circumference','Physical-Diastolic_BP','Physical-HeartRate','Physical-Systolic_BP']
tot_nan_phys = df_test[cols].isna().all(axis=1)

for col in cols:
    df_test.loc[tot_nan_phys, col] = df_test.loc[tot_nan_phys, f"{col}_avg"]
    del df_test[f"{col}_avg"]

X_test = df_test

is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in X_test.dtypes])  
numerical_idx = np.flatnonzero(is_numerical) 
new_X_test = X_test.iloc[:, numerical_idx]


scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=3)

scaled_test = scaler.fit_transform(new_X_test)
X_array = imputer.fit_transform(scaled_test)
X_array = scaler.inverse_transform(X_array)
new_X_test = pd.DataFrame(X_array, columns=new_X_test.columns, index=new_X_test.index)

categorical_idx = np.flatnonzero(is_numerical==False)
categorical_X_test = X_test.iloc[:, categorical_idx]

imputer = SimpleImputer(strategy='most_frequent')
X_array = imputer.fit_transform(categorical_X_test)
categorical_X_test = pd.DataFrame(X_array, columns=categorical_X_test.columns, index=categorical_X_test.index)

oh = OneHotEncoder(sparse_output=False)

oh.fit(categorical_X_test)
encoded = oh.transform(categorical_X_test)

for i, col in enumerate(oh.get_feature_names_out()):
    new_X_test = new_X_test.copy()
    new_X_test[col] = encoded[:, i]


features_to_add = list(set(feature_names) - set(new_X_test.columns.tolist()))
for feature in features_to_add:
    new_X_test[feature] = 0
    
# now we reorder the subset, then we are ready for the prediction task
new_X_test = new_X_test[feature_names]

X_test_model = new_X_test.iloc[:, selector.support_]
y_pred = model.predict(X_test_model)
y_pred


submission = pd.DataFrame({
    'id' : ids,
    'sii' : y_pred
})
submission.to_csv('submission.csv',index=False)


y_proba = model.predict_proba(X_test_subset) # for each instances get the probability that it belongs to each class
confidence = np.max(y_proba, axis=1) # we get with what probability an instance was mapped to a class label
results_df = pd.DataFrame({
    'true_label': y_test,
    'pred_label': model.predict(X_test_subset),
    'confidence': confidence
})
results_df['correct'] = results_df['true_label'] == results_df['pred_label']
results_df['index'] = y_test.index

most_wrong = results_df[~results_df['correct']].sort_values(by='confidence', ascending=False).head(500)
print(most_wrong.head(10))


feature_names = new_X_train.columns.tolist()

important_features = [name for name, importance in zip(feature_names, tuned_model.best_estimator_.feature_importances_) if importance > 0.034]
print(important_features)
n = len(important_features)



import seaborn as sns
fig, ax = plt.subplots(1, n, figsize=(20,n))

for i in range(len(important_features)):
    sns.kdeplot(X_test_subset[important_features[i]], label='General', ax=ax[i])
    sns.kdeplot(X_test_subset.loc[most_wrong['index']][important_features[i]], label='Most Wrong', ax=ax[i])

    ax[i].legend()
    ax[i].set_xlabel(important_features[i])
    ax[i].set_ylabel('Density')
    ax[i].set_title(f'Distribution of {important_features[i]}')


most_correct = results_df[results_df['correct']].sort_values(by='confidence', ascending=False).head(500)
print(most_correct.head(10))


fig, ax = plt.subplots(1, n, figsize=(20,n))

for i in range(len(important_features)):
    sns.kdeplot(X_test_subset[important_features[i]], label='General', ax=ax[i])
    sns.kdeplot(X_test_subset.loc[most_correct['index']][important_features[i]], label='Most Correct', ax=ax[i])

    ax[i].legend()
    ax[i].set_xlabel(important_features[i])
    ax[i].set_ylabel('Density')
    ax[i].set_title(f'Distribution of {important_features[i]}')

