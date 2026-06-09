import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, metrics, preprocessing, compose
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import tensorflow as tf
import warnings


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)



train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv').drop('id', axis=1)
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv').drop('id', axis=1)
print(f'Training data has shape of {train.shape[0]}')
print('\nTraining data:\n')
display(train.head(3))
print('=='*50)
print('=='*50)
print(f'Test data has shape of {test.shape}')
print('\nTest data:\n')
display(test.head(3))


# Fitting additional data
df = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
df = df[train.columns]
df


train = pd.concat([train, df], ignore_index=True)
train


train.info()


# Checking for duplicates
print('sum of duplicates in train data is ', train.duplicated().sum())


train.describe().T


train.describe(include=object).T


train.info()


y = train['diagnosed_diabetes']
X = train.drop('diagnosed_diabetes', axis=1)


y.value_counts(normalize=True)


numerical_columns = X.select_dtypes(exclude=object).columns.tolist()
categorical_columns = X.select_dtypes(include=object).columns.tolist()


# setting pLot defaults:
plt.rc('figure', figsize=(12, 8))
plt.rc('axes',
       labelsize=9,
       titleweight='bold',
       titlesize=21)
sns.set_theme(style='dark', rc={'figure.figsize':(15,8)})


# comment
for i, col in enumerate(numerical_columns):
    if col not in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
        plt.figure(figsize=(10,7))
        sns.boxplot(data=train, y=col, x='diagnosed_diabetes')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.tight_layout()


# Explain what is done and why
test_result = []
for col in numerical_columns:
    if col not in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
        corr, p_value = stats.pointbiserialr(train[col], train['diagnosed_diabetes'])
        test_result.append({'numeric_features':col, 'corr':corr, 'p-value':p_value})
        num_table = pd.DataFrame(test_result)
display(num_table)
print('='*50)
print('\nStatistically signicant:')
num_table_sig = num_table[num_table['p-value']< 0.05]
num_table_sig


#numeric_features = num_table_sig[abs(num_table_sig['corr'])>=0.05].numeric_features.tolist()
#numeric_features


X.describe(include=object)


for col in categorical_columns:
    print(col.ljust(30, '.'), X[col].unique())


for col in categorical_columns:
    plt.figure(figsize=(10,8))
    sns.countplot(data=train, x=col, hue='diagnosed_diabetes')
    plt.title('Countplot of {}'.format(col))
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(loc='best')


for col in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
    print(display(pd.crosstab(train[col], train['diagnosed_diabetes'], normalize='index')))


num_results = []
for col in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
    cross_table = pd.crosstab(train[col], train['diagnosed_diabetes'])
    chi2, pvalue, dof, expected = stats.chi2_contingency(cross_table)
    num_results.append({'feature':col, 'chi-square':round(chi2, 4), 'p-value':format(pvalue, '.4e'),
                    'dof':dof})
num_results = pd.DataFrame(num_results)
num_results[num_results['p-value'].astype(np.float32)<0.05]


#[numeric_features.append(i) for i in num_results.feature]
#numeric_features


results = []
for col in categorical_columns:
    cross_table = pd.crosstab(train[col], train['diagnosed_diabetes'])
    chi2, pvalue, dof, expected = stats.chi2_contingency(cross_table)
    results.append({'feature':col, 'chi-square':round(chi2, 4), 'p-value':format(pvalue, '.4e'),
                    'dof':dof})
results = pd.DataFrame(results)
display(results)
significant_results = results[results['p-value'].astype(np.float32)<=.05]
significant_results


#categorical_features = significant_results.feature.tolist()
#categorical_features


# Checking ranking of columns from mutual info scores
# Mutual info classif measures nonlinear relationship and since we'll be using logistics regression we will know feature to engineer

le = preprocessing.LabelEncoder()
X_copy = X.copy() # Getting copy of independent variables

for col in categorical_columns:
    X_copy[col] = le.fit_transform(X_copy[col])
mi = pd.Series(mutual_info_classif(X_copy, y, random_state=8, n_neighbors=100),
               index=X_copy.columns)
mi.sort_values(ascending=False)


mi.plot(kind='bar')


mi_features = mi[mi>=0.001].index
mi_features


numeric_features = [col for col in mi_features if X[col].dtype!=object]
categorical_features = [col for col in mi_features if X[col].dtype==object]

numeric_features, categorical_features


plt.figure(figsize=(19,10))
sns.heatmap(X[numeric_features].corr(), annot=True, cmap='viridis');


# log transforming skewed column 'physical_activity_minutes_per_week'
X['log_physical_activity_minutes_per_week'] = np.log1p(X['physical_activity_minutes_per_week'])
test['log_physical_activity_minutes_per_week'] = np.log1p(test['physical_activity_minutes_per_week'])

sns.histplot(X['log_physical_activity_minutes_per_week'])
plt.title('log_physical_activity_minutes_per_week')
plt.xlabel('Distribution of log_physical_activity_minutes_per_week')


labels=['poor', 'moderate', 'healthy']
#X['diet_score'] = pd.cut(X['diet_score'], labels=labels, bins=3).astype(object)
X.head()


X['alcohol_consumption_per_week'] = X['alcohol_consumption_per_week'].astype(object)
X['cholesterol_ratio'] = X['ldl_cholesterol'] / X['hdl_cholesterol']
X['triglycerides*hdl_cholesterol'] = X['hdl_cholesterol'] * X['triglycerides']
X['bmi*waist_tohipratio'] = X['bmi'] * X['waist_to_hip_ratio']

X['smoking_status'] = np.where(X['smoking_status']=='current',1,
                              np.where(X['smoking_status']=='former', -1, 0))
X['employment_status'] = np.where(X['employment_status']=='Employed',1,
                              np.where(X['employment_status']=='Retired', 2, 0))
#
X['smoking*family_history_diabetes'] = X['smoking_status'] * X['family_history_diabetes']



categorical_columns.append('alcohol_consumption_per_week')
categorical_columns.append('diet_score')
categorical_columns


numeric_features.append('log_physical_activity_minutes_per_week')
numeric_features.remove('physical_activity_minutes_per_week')
numeric_features.append('cholesterol_ratio')
numeric_features.append('triglycerides*hdl_cholesterol')
numeric_features.append('bmi*waist_tohipratio')
numeric_features.remove('cholesterol_total')
numeric_features.remove('diet_score')
#numeric_features.append('smoking_status')
#numeric_features.append('heart_rate')
numeric_features


print(f'Numeric features are: {list(numeric_features)}')


print(f'Categorical features are: {list(categorical_features)}')


# Splitting data  to train and valdation in the ration 0.9 and 0.1 reapectively
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.1, stratify=y, random_state=19)

# Preprocessor for different column type
numeric_transformer = preprocessing.StandardScaler()
ordinal_transformer = preprocessing.OrdinalEncoder(dtype=int)
nominal_transformer = preprocessing.OneHotEncoder(sparse_output=False, dtype=np.float32)


numeric_features, categorical_features


[categorical_features.append(col) for col in significant_results.feature] # Adding statistically significant columns
categorical_features


ordinal_columns = ['education_level', 'income_level']#, 'alcohol_consumption_per_week', 'diet_score']
nominal_columns = [col for col in categorical_features if col not in ordinal_columns]
nominal_columns


# Combining preprocessors
column_transformer = compose.ColumnTransformer([
    ('numerical', numeric_transformer, numeric_features),
    ('ordinals', ordinal_transformer, ordinal_columns),
    ('nominals', nominal_transformer, nominal_columns)
], remainder='drop').set_output(transform='pandas')


model = LogisticRegression(random_state=7)
model_pipeline = Pipeline([
    ('columTransformer', column_transformer),
    ('model', model)
])


# Handling outliers cliping with upper and lower quantiles
for col in numeric_features:
    if col not in ['family_history_diabetes', 'hypertension_history']:
        q1 = X[col].quantile(.25)
        q3 = X[col].quantile(.75)
        iqr = q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3+1.5*iqr
        X[col] = X[col].clip(lower_bound, upper_bound)


params = {'model__C':[0.001, .01, 1, 10],
          'model__class_weight':[{0:i, 1:1} for i in range(1, 4)],
          'model__penalty':['l1', 'l2', None]}
grid = model_selection.GridSearchCV(estimator=model_pipeline, param_grid=params,
                                    scoring={'recall':metrics.make_scorer(metrics.recall_score),
                                             'precision':metrics.make_scorer(metrics.f1_score)},
                                    refit='recall', cv=5)
grid.fit(X_train, y_train)


grid.best_params_


threshold = 0.5
train_probs = grid.predict_proba(X_train)[:, 1]
train_preds = (train_probs>threshold).astype(int)
y_probs = grid.predict_proba(X_valid)[:, 1]
y_preds = (y_probs>threshold).astype(int)
print('Training recall score of: {}'.format(metrics.recall_score(y_train, train_preds)))
print('Validation recall score of: {}'.format(metrics.recall_score(y_valid, y_preds)))
print('\nClassification report: \n{}'.format(metrics.classification_report(y_valid, y_preds)))


cm = metrics.confusion_matrix(y_valid, y_preds)
print(cm)
sns.heatmap(cm, cmap='viridis', annot=True)


# ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_valid, y_probs)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()




# Precisionâ€“Recall Curve
precision, recall, _ = metrics.precision_recall_curve(y_valid, y_probs)
ap = metrics.average_precision_score(y_valid, y_preds)

plt.figure(figsize=(7,5))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precisionâ€“Recall Curve")
plt.legend()
plt.show()




nn = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='tanh'),
    tf.keras.layers.Dense(2, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn.compile(loss='binary_crossentropy',
          optimizer='Adam',
          metrics=['recall'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    min_delta=0.001,
    patience=4,
    restore_best_weights=True,
)


X_train_processed = column_transformer.fit_transform(X_train)
X_valid_processed = column_transformer.transform(X_valid)


nn.fit(X_train_processed.values, y_train.values, callbacks=[early_stopping],
       validation_data=(X_valid_processed.values, y_valid.values), epochs=10)


nn_prob = nn.predict(X_valid_processed) 
nn_preds = (nn_prob>=0.5).astype(int)
metrics.recall_score(y_valid.values, nn_preds.flatten())


print(metrics.classification_report(y_valid, nn_preds))


cm = metrics.confusion_matrix(y_valid.values, nn_preds)
cm


#ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_valid, nn_prob)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()




# Precisionâ€“Recall Curve
precision, recall, _ = metrics.precision_recall_curve(y_valid, nn_prob)
ap = metrics.average_precision_score(y_valid, y_preds)

plt.figure(figsize=(7,5))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precisionâ€“Recall Curve")
plt.legend()
plt.show()



test['alcohol_consumption_per_week'] = test['alcohol_consumption_per_week'].astype(object)
test['cholesterol_ratio'] = test['hdl_cholesterol'] / test['ldl_cholesterol']
test['triglycerides*hdl_cholesterol'] = test['hdl_cholesterol'] * test['triglycerides']
test['bmi*waist_tohipratio'] = test['bmi'] * test['waist_to_hip_ratio']

test['smoking_status'] = np.where(test['smoking_status']=='current',-1,
                              np.where(test['smoking_status']=='former', 1, 0))
test['employment_status'] = np.where(test['employment_status']=='Employed',1,
                              np.where(test['employment_status']=='Retired', 2, 0))
test['smoking*family_history_diabetes'] = test['smoking_status'] * test['family_history_diabetes']



test_processed = column_transformer.transform(test)
test_preds = nn.predict(test_processed)
test_preds


sub_df = pd.DataFrame({'id':range(700000, len(test)+700000),
                      'diagnosed_diabetes':test_preds.ravel()})
sub_df.head()


sub_df.to_csv('submission_file.csv', index=False)

