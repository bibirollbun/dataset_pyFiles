import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/kaggle/input/playground-series-s5e8/train.csv'
bank_data = pd.read_csv(path, index_col='id')


print(bank_data.columns)
bank_data.head()


bank_data.describe()


bank_data.describe(include='object')


len(bank_data)


bank_data.hist(figsize=(20, 15), bins=50, xlabelsize=8, ylabelsize=8)


import matplotlib.pyplot as plt
# plt.bar(bank_data['job'].unique(), )
plt.figure(figsize=(12, 6))
plt.title("Jobs of customers")
plt.bar(bank_data['job'].unique(), bank_data['job'].value_counts())


import seaborn as sns
sns.countplot(bank_data, x = 'job')
sns.countplot(bank_data, x = 'marital')


plt.figure(figsize=(12, 12))
plt.title("Categoraical variables")
plt.axis("off")
plot_count = 0  # Initialize a counter for displayed images
categorical_variables = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for var in categorical_variables:
    ax = plt.subplot(3, 3, plot_count + 1)  # Adjust subplot position
    sns.countplot(bank_data, x = var)
    plot_count += 1  # Increment the counter
plt.show()


bank_data.isnull().sum()


import random
# Data preparation
def dataPrep(columns_to_keep=['age', 'y'], rows_to_keep=1):
    bank_data_clean = bank_data.copy()
    if ((rows_to_keep > 1) or (rows_to_keep) <= 0):
        rows_to_keep = 1
    max_row = int(len(bank_data_clean) * rows_to_keep)
    rows = random.sample(range(len(bank_data_clean)), max_row)
    bank_data_clean = bank_data_clean.loc[rows, columns_to_keep]
    # Dummy encoding
    bank_data_clean = pd.get_dummies(bank_data_clean, drop_first=True)
    # Making X and y variables
    X = bank_data_clean.drop(columns=['y'])
    y = bank_data_clean['y']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, test_size=0.3)
    return([train_X, val_X, train_y, val_y])


from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error

def printResults(model, train_X, val_X, train_y, val_y):
    # Training accuracy:
    predictions_train = rf_model.predict(train_X)
    print('training accuracy:', (predictions_train == train_y.values).mean())
    # Testing accuracy:
    predictions_test = rf_model.predict(val_X)
    print('testing accuracy:', (predictions_test == val_y.values).mean())
    # Finding the ROC
    print('train roc:', roc_auc_score(train_y, rf_model.predict_proba(train_X)[:, 1]))
    print('test ROC:', roc_auc_score(val_y, rf_model.predict_proba(val_X)[:, 1]))


bank_data_clean = bank_data.copy()


# Chosing columsn to keep
# days and pdays seems to have issues so these are excluded
columns_to_keep = ['age', 'balance', 'duration', 'campaign', 'previous', 'y'] 
bank_data_clean = bank_data_clean[columns_to_keep]


# Dummy encoding
bank_data_clean = pd.get_dummies(bank_data_clean, drop_first=True)
bank_data_clean.shape


bank_data_clean


from sklearn.model_selection import train_test_split
# Making X and y variables
X = bank_data_clean.drop(columns=['y'])
y = bank_data_clean['y']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=0)
lr_model.fit(train_X, train_y)


from sklearn.metrics import mean_absolute_error

#mean_absolute_error(predictions, val_y)
# Training accuracy:
predictions_train = lr_model.predict(train_X)
print('training accuracy:', (predictions_train == train_y.values).mean())
# Testing accuracy:
predictions_test = lr_model.predict(val_X)
print('testing accuracy:', (predictions_test == val_y.values).mean())



from sklearn.metrics import roc_auc_score
# Finding the ROC
print('train roc:', roc_auc_score(train_y, lr_model.predict_proba(train_X)[:, 1]))
print('test ROC:', roc_auc_score(val_y, lr_model.predict_proba(val_X)[:, 1]))


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(val_y, lr_model.predict_proba(val_X)[:, 1])
roc_auc = metrics.auc(fpr, tpr)
#fig, ax = plt.subplots()
#ax.axline((0, 0.5), slope=0.5, color="black", linestyle=(0, (5, 5)))
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

display.plot()
plt.show()


# Prepping the data
# All columns:
# ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
columns_to_keep = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_X, val_X, train_y, val_y = dataPrep(columns_to_keep)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model = rf_model.fit(train_X, train_y)


printResults(rf_model, train_X, val_X, train_y, val_y)


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(val_y, rf_model.predict_proba(val_X)[:, 1])
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

display.plot()
plt.show()


roc_auc


# Prepping the data
# All columns:
# ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
columns_to_keep = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_X, val_X, train_y, val_y = dataPrep(columns_to_keep, 0.01)


from sklearn.model_selection import RepeatedKFold, GridSearchCV

model = RandomForestClassifier()

max_features = [0.16, 0.2, 0.25, 0.3]
min_samples_split = [2, 3, 4, 5, 6]
min_samples_leaf = [1, 2, 3]
grid = dict(max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

gridSearch = GridSearchCV(
    estimator=model,
    param_grid=grid,
    verbose=1,
    n_jobs=-1,
	cv=5,
    scoring='roc_auc'
)


searchResults = gridSearch.fit(train_X, train_y)


bestModel = searchResults.best_estimator_
bestModel


import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
results_df = pd.DataFrame(searchResults.cv_results_['params'])
results_df['results'] = searchResults.cv_results_['mean_test_score']
results_df['min_samples_leaf'] = results_df['min_samples_leaf'].astype(str)
results_df['min_samples_split'] = results_df['min_samples_split'].astype(str)
leafs = []
for i in range(len(results_df)):
    leafs.append('min_samples_leaf: ' + str(results_df.loc[i, 'min_samples_leaf']) + ' min_samples_split: ' + str(results_df.loc[i, 'min_samples_split']))
results_df['leafs'] = leafs
plt.figure(figsize=(14, 7))
sns.lineplot(data=results_df, y='results', x='max_features', hue='min_samples_split')


columns_to_keep = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_X, val_X, train_y, val_y = dataPrep(columns_to_keep)
bestModel.fit(train_X, train_y)


printResults(bestModel, train_X, val_X, train_y, val_y)


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(val_y, bestModel.predict_proba(val_X)[:, 1])
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

display.plot()
plt.show()


roc_auc


# Importing the test dataset
bank_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
bank_test = pd.get_dummies(bank_test, drop_first=True)


# Prepping the training data
bank_data_clean = bank_data.copy()
bank_data_clean = pd.get_dummies(bank_data_clean, drop_first=True)
X = bank_data_clean.drop(columns=['y'])
y = bank_data_clean['y']


# Running the model
bestModel.fit(X, y)


predictions = bestModel.predict_proba(bank_test)[:, 1]
predictions


output = pd.DataFrame({'id': bank_test.index,
                       'y': predictions})
output.to_csv('submission.csv', index=False)

