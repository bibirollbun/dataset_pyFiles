import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


# # Load the 2 files of data set of competetion of train test and submission file
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

train.head()


test.head()


#Let's see the shape of the data
print(train.shape)
print(test.shape)
# print(sample_submission.shape)

# Also check the the how data is divided between train and test %
print(train.shape[0] / (train.shape[0] + test.shape[0]))
print(test.shape[0] / (train.shape[0] + test.shape[0]))


train.info()


train.columns


# id column
train['id'].head(7)


# day column
train['day'].head(7)


# train = train.drop('day', axis=1)



# pressure
train['pressure'].value_counts()


train.columns


# Check for missing values
print(train.isnull().sum())
print(test.isnull().sum())


# Let's fill missing value in column of wind direction using mean of the column
test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())

# Also print the missing values to see data is filled or not)
print(test.isnull().sum())



# Check the summary of the data
print(train.describe())
# print(test.describe())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

from sklearn.linear_model import LogisticRegression


def get_numeric_features(df: pd.DataFrame, columns: list) -> list:
    num_features = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    return num_features

def get_category_features(df: pd.DataFrame, columns: list) -> list:
    cat_features = [col for col in columns if pd.api.types.is_categorical_dtype(df[col])]
    return cat_features
    
def get_object_features(df: pd.DataFrame, columns: list) -> list:
    obj_features = [col for col in columns if pd.api.types.is_object_dtype(df[col])]
    return obj_features



corr_matrix = round(train.corr(), 2)
annot_matrix = corr_matrix.copy().astype(str)
np.fill_diagonal(annot_matrix.values, train.columns) 

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=annot_matrix, fmt="", cmap="mako", linewidths=0.5)
plt.title("Correlation Heatmap with Feature Names on Diagonal")


train.head()


train['high_cloud_humidity'] = ((train['humidity'] > 80) & (train['cloud'] > 60)).astype(int)
train['cloud_sunshine_ratio'] = train['cloud'] / (train['sunshine'] + 1)
train['humidity_sunshine_ratio'] = train['humidity'] / (train['sunshine'] + 1)
train['low_sunshine'] = (train['sunshine'] < 6).astype(int)
train['temperature'] = train[['mintemp', 'temparature', 'maxtemp']].mean(axis=1)
train['cloud_windspeed'] = train['cloud'] * train['windspeed']

train['humidity_previous_day'] = train['humidity'].shift(1).fillna(0)
train['pressure_previous_day'] = train['pressure'].shift(1).fillna(0)

test['humidity_previous_day'] = test['humidity'].shift(1).fillna(0)
test['pressure_previous_day'] = test['pressure'].shift(1).fillna(0)




test['high_cloud_humidity'] = ((test['humidity'] > 80) & (test['cloud'] > 60)).astype(int)
test['cloud_sunshine_ratio'] = test['cloud'] / (test['sunshine'] + 1)
test['humidity_sunshine_ratio'] = test['humidity'] / (test['sunshine'] + 1)
test['low_sunshine'] = (test['sunshine'] < 6).astype(int)
test['temperature'] = test[['mintemp', 'temparature', 'maxtemp']].mean(axis=1)
test['cloud_windspeed'] = test['cloud'] * test['windspeed']


train = train.drop(['mintemp', 'temparature', 'maxtemp'], axis=1)
test = test.drop(['mintemp', 'temparature', 'maxtemp'], axis=1)


train.head()


def assign_season(day: int):
    if not ( 1 <= day <= 366):
        return None
    
    if 80 <= day < 172: return 'spring'
    elif 172 <= day < 266: return 'summer'
    elif 266 <= day < 355: return 'autumn'
    else : return 'winter'

train['season'] = train['day'].map(assign_season).astype('category')
test['season'] = test['day'].map(assign_season).astype('category')

def get_month_from_day(day: int):
    if not (1 <= day <= 366):
        return "Invalid day. Must be between 1 and 366."
    
    if 1 <= day <= 31:
        return "January"
    if 32 <= day <= 60:
        return "February"
    if 61 <= day <= 91:
        return "March"
    if 92 <= day <= 121:
        return "April"
    if 122 <= day <= 152:
        return "May"
    if 153 <= day <= 182:
        return "June"
    if 183 <= day <= 213:
        return "July"
    if 214 <= day <= 244:
        return "August"
    if 245 <= day <= 274:
        return "September"
    if 275 <= day <= 305:
        return "October"
    if 306 <= day <= 335:
        return "November"
    if 336 <= day <= 366:
        return "December"
    
    return "Error: Day out of range"

train['month'] = train['day'].map(get_month_from_day).astype('category')
test['month'] = test['day'].map(get_month_from_day).astype('category')


train['month'].value_counts()


train.head()




# Prepare dataset
X = train.drop(['rainfall', 'id'], axis=1)
y = train['rainfall']


# LEt's apply season adn month into codes using label encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['season'] = le.fit_transform(X['season'])
X['month'] = le.fit_transform(X['month'])

train.head()


# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X.head()


# Lets ' select only important features
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, /random_state=42)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
selected_features = feature_importances.nlargest(8).index.tolist()  # Convert to list

# Print selected features
print("Selected Features:", selected_features)
# Create a new DataFrame with selected features
X_selected = X[selected_features]
# train_selected=train[selected_features]


# lets' use selected features to train the model
# train=train_selected
X=X_selected


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# shpae of train and test data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


X.head()


from scipy.stats import skew, kurtosis
new_num_features = get_numeric_features(train, train.columns)
new_num_features.remove('rainfall')

skewness = {col:skew(train[col]) for col in new_num_features}
kurtosis_value = {col:kurtosis(train[col]) for col in new_num_features}

for col, value in skewness.items():
    print(f"{col}:{value}")

print('*'*10)
for col, value in kurtosis_value.items():
    print(f"{col}:{value}")
    
features_to_transform = [col for col in new_num_features if abs(skewness[col]) > 0.5 or kurtosis_value[col] > 3]
features_to_transform




for feature in features_to_transform:
    train[f"{feature}_log"] = np.log1p(train[feature])
    test[f"{feature}_log"] = np.log1p(test[feature])


train = train.drop(features_to_transform, axis=1)
test = test.drop(features_to_transform, axis=1)


train = pd.get_dummies(train, columns=['season','month'], drop_first=True, dtype=int)
test = pd.get_dummies(test, columns=['season', 'month'], drop_first=True, dtype=int)


scaler = StandardScaler()
columns_to_scale = [col for col in train.columns if not (col.startswith("season_") or  col.startswith("month_") or col in ('rainfall', 'id'))]
train[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])
test[columns_to_scale] = scaler.transform(test[columns_to_scale])


strat_kfold = StratifiedKFold(n_splits=10,shuffle=True, random_state=42)
models = {
    'Logistic Regression': (LogisticRegression(solver='liblinear'), {
        'penalty': ['l1', 'l2'],
        'C': np.linspace(1e-2, 10, 100),
        'max_iter': [50,75,100]
    })
}   
 
best_models = {}

# Prepare dataset
X = train.drop(['rainfall', 'id'], axis=1)
y = train['rainfall']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Iterate over models and perform hyperparameter tuning
for model_name, (model, params) in models.items():
    print(f'\nTraining {model_name}...')
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=strat_kfold, verbose=1, scoring='neg_log_loss')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='weighted')
    recall = recall_score(y_valid, y_pred, average='weighted')
    precision = precision_score(y_valid, y_pred, average='weighted')
    auc = roc_auc_score(y_valid, y_pred_proba)
    
    print(f'Best parameters for {model_name}: {grid_search.best_params_}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'AUC: {auc:.4f}')



best_model = best_models['Logistic Regression']
X = test.drop('id', axis=1)
y_pred = best_model.predict_proba(X)[:, 1]
submission = pd.DataFrame({"id": test.id,
                           "rainfall": y_pred})
submission.to_csv('submission.csv', index=False)


submission.head()

