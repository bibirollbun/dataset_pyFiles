import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test_data = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')


train_data.shape, test_data.shape


train_data


test_data


train_data.info()


test_data.info()


train_data['datetime'] = pd.to_datetime(train_data['datetime'])
test_data['datetime'] = pd.to_datetime(test_data['datetime'])

train_data.info(), test_data.info()


train_data['year'] = train_data['datetime'].dt.year
train_data['year'] = train_data['year'].map({2011:0, 2012:1})
train_data['month'] = train_data['datetime'].dt.month
train_data['day'] = train_data['datetime'].dt.day
train_data['hour'] = train_data['datetime'].dt.hour

train_data.head()


train_data.drop('datetime', axis=1, inplace=True)
train_data.head()


train_data.info()


test_data['year'] = test_data['datetime'].dt.year
test_data['year'] = test_data['year'].map({2011:0, 2012:1})
test_data['month'] = test_data['datetime'].dt.month
test_data['day'] = test_data['datetime'].dt.day
test_data['hour'] = test_data['datetime'].dt.hour

test_data.head()


test_data.drop('datetime', axis=1, inplace=True)
test_data.head()


test_data.info()


train_data.isna().sum()


test_data.isna().sum()


train_data.duplicated().sum()


test_data.duplicated().sum()


for col in train_data.columns:
    q1 = train_data[col].quantile(0.25)
    q3 = train_data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = train_data[(train_data[col] < lower_bound) | (train_data[col] > upper_bound)]
    print(f"{col} : {len(outliers)}")


for col in test_data.columns:
    q1 = test_data[col].quantile(0.25)
    q3 = test_data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = test_data[(test_data[col] < lower_bound) | (test_data[col] > upper_bound)]
    print(f"{col} : {len(outliers)}")


train_data.describe(include='all')


test_data.describe(include='all')


train_data.hist(figsize=(15, 10), bins=30)
plt.show()


test_data.hist(figsize=(13, 8), bins=30)
plt.show()


corr_matrix = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Train Data')
plt.show()


corr_matrix = test_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Test Data')
plt.show()


columns = [col for col in train_data.columns if col != 'count']

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16,16))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.boxplot(
        data=train_data,
        x=col,
        y='count',
        ax=axes[i],
    )
    axes[i].set_title(f'Boxplot of {col} by count')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
sns.boxplot(x=train_data['count'])
plt.title('Boxplot of count')
plt.show()


q1 = train_data['count'].quantile(0.25)
q3 = train_data['count'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = train_data[(train_data['count'] < lower_bound) | (train_data['count'] > upper_bound)]
print(lower_bound, upper_bound)
outliers


outliers['hour'].value_counts()


q1 = train_data['casual'].quantile(0.25)
q3 = train_data['casual'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = train_data[(train_data['casual'] < lower_bound) | (train_data['casual'] > upper_bound)]
outliers['hour'].value_counts()


q1 = train_data['registered'].quantile(0.25)
q3 = train_data['registered'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = train_data[(train_data['registered'] < lower_bound) | (train_data['registered'] > upper_bound)]
outliers['hour'].value_counts()


train_data[(train_data['casual'] > 116 ) & (train_data['hour'] == 21)]


train_data[(train_data['casual'] > 116 ) & (train_data['hour'] == 22)]


data = train_data.drop(columns=['casual', 'registered'])
data


test_data


X_data = data.drop(columns=['count'])
y_data = data['count']


X_data


y_data


from sklearn.model_selection import train_test_split

# í•™ìŠµìš©, ê²€ì¦�ìš©, ì‹œí—˜ìš© ë�°ì�´í„° ë¶„ë¦¬ í•¨ìˆ˜  # # For learning, verification, test data separation function
def train_val_test_split(X, y):

# í•™ìŠµìš© : ì‹œí—˜ìš© = 8 : 2 ë¡œ ë¶„ë¦¬í•˜ê¸°  # # Learning: Examination = 8: 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ë‹¤ì‹œ í•™ìŠµìš©ì�„ í•™ìŠµìš© : ê²€ì¦�ìš© = 8 : 2 ë¡œ ë¶„ë¦¬í•˜ê¸°  # # For learning for learning again: Verification = 8: 2 to separate
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# train : 64 / val : 16 / test : 20 ì�˜ ë¹„ìœ¨ë¡œ ë‚˜ëˆ´ë‹¤.  # # Train: 64 / VAL: 16 / TEST: Divided in the ratio of 20.

# ì�¸ë�±ìŠ¤ ì´ˆê¸°í™”í•˜ê¸°  # # Initializing indexes
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

# X, yì�˜ train, val, test ê°’ì�„ ê°�ê°� ë°˜í™˜í•˜ê¸°  # # Returns the train, val, test values â€‹â€‹of x, y
    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_data, y_data)


X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


from sklearn.preprocessing import StandardScaler

# ìŠ¤íƒ ë‹¤ë“œ ìŠ¤ì¼€ì�¼ë§� í•¨ìˆ˜
def standard_scaler(X_train, X_val, X_test, test_data):

    # ì •ê·œí™”í•  ë�°ì�´í„°ë§Œ ë½‘ì•„ë‚´ê¸°
    X_train_num = X_train[['temp', 'atemp', 'humidity', 'windspeed']]
    X_val_num = X_val[['temp', 'atemp', 'humidity', 'windspeed']]
    X_test_num = X_test[['temp', 'atemp', 'humidity', 'windspeed']]

    # ë�¼ë²¨ë§�ì�´ ë�˜ì–´ì�ˆëŠ” ë�°ì�´í„°
    X_train_label = X_train.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_val_label = X_val.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_test_label = X_test.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])

    # ìŠ¤íƒ ë‹¤ë“œ ìŠ¤ì¼€ì�¼ë§�
    scaler = StandardScaler()

    # ìŠ¤ì¼€ì�¼ëŸ¬ ì �ìš©
    X_train_scaled = scaler.fit_transform(X_train_num)  # í•™ìŠµë�°ì�´í„°ì—�ë§Œ fit_transform
    X_val_scaled = scaler.transform(X_val_num)          # ê²€ì¦�ë�°ì�´í„°ì—�ëŠ” transform
    X_test_scaled = scaler.transform(X_test_num)        # ì‹œí—˜ë�°ì�´í„°ì—�ë�„ transform

    # ë�°ì�´í„°í”„ë ˆì�„ìœ¼ë¡œ ë³€í™˜
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=['temp', 'atemp', 'humidity', 'windspeed'])

    # ë�°ì�´í„° í•©ì¹˜ê¸°
    X_train_ss = pd.concat([X_train_label, X_train_scaled], axis=1)
    X_val_ss = pd.concat([X_val_label, X_val_scaled], axis=1)
    X_test_ss = pd.concat([X_test_label, X_test_scaled], axis=1)



    # í…ŒìŠ¤íŠ¸ ë�°ì�´í„° ìŠ¤ì¼€ì�¼ë§�
    test_num = test_data[['temp', 'atemp', 'humidity', 'windspeed']]
    test_label = test_data.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])

    # ë³€í™˜ ë°� ì �ìš©
    scaler.transform(test_num)

    # ë�°ì�´í„° í”„ë ˆì�„ìœ¼ë¡œ ë³€í™˜ ë°� í•©ì¹˜ê¸°
    test_scaled = pd.DataFrame(test_data, columns=['temp', 'atemp', 'humidity', 'windspeed'])
    test_ss = pd.concat([test_label, test_scaled], axis=1)

    return X_train_ss, X_val_ss, X_test_ss, test_ss


# ìŠ¤íƒ ë‹¤ë“œ ìŠ¤ì¼€ì�¼ëŸ¬ ì �ìš©  # # Standard scaler application
X_train_ss, X_val_ss, X_test_ss, test_ss = standard_scaler(X_train, X_val, X_test, test_data)


X_train_ss[['temp', 'atemp', 'humidity', 'windspeed']].describe() ,X_val_ss[['temp', 'atemp', 'humidity', 'windspeed']].describe(), X_test_ss[['temp', 'atemp', 'humidity', 'windspeed']].describe(), test_ss[['temp', 'atemp', 'humidity', 'windspeed']].describe()


from sklearn.preprocessing import MinMaxScaler

# ë¯¸ë‹ˆë§¥ìŠ¤ ìŠ¤ì¼€ì�¼ë§� í•¨ìˆ˜
def minmax_scaler(X_train, X_val, X_test, test_data):

    # ì •ê·œí™”í•  ë�°ì�´í„°ë§Œ ë½‘ì•„ë‚´ê¸°
    X_train_num = X_train[['temp', 'atemp', 'humidity', 'windspeed']]
    X_val_num = X_val[['temp', 'atemp', 'humidity', 'windspeed']]
    X_test_num = X_test[['temp', 'atemp', 'humidity', 'windspeed']]

    # ë�¼ë²¨ë§�ì�´ ë�˜ì–´ì�ˆëŠ” ë�°ì�´í„°
    X_train_label = X_train.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_val_label = X_val.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_test_label = X_test.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])

    # ë¯¸ë‹ˆë§¥ìŠ¤ ìŠ¤ì¼€ì�¼ë§�
    scaler = MinMaxScaler()

    # ìŠ¤ì¼€ì�¼ëŸ¬ ì �ìš©
    X_train_scaled = scaler.fit_transform(X_train_num)      # í•™ìŠµë�°ì�´í„°ì—�ë§Œ fit_transform
    X_val_scaled = scaler.transform(X_val_num)              # ê²€ì¦�ë�°ì�´í„°ì—�ëŠ” transform
    X_test_scaled = scaler.transform(X_test_num)            # ì‹œí—˜ë�°ì�´í„°ì—�ë�„ transform

    # ë�°ì�´í„°í”„ë ˆì�„ìœ¼ë¡œ ë³€í™˜
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=['temp', 'atemp', 'humidity', 'windspeed'])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=['temp', 'atemp', 'humidity', 'windspeed'])

    # ë�°ì�´í„° í•©ì¹˜ê¸°
    X_train_mm = pd.concat([X_train_label, X_train_scaled], axis=1)
    X_val_mm = pd.concat([X_val_label, X_val_scaled], axis=1)
    X_test_mm = pd.concat([X_test_label, X_test_scaled], axis=1)



    # í…ŒìŠ¤íŠ¸ ë�°ì�´í„° ìŠ¤ì¼€ì�¼ë§�
    test_num = test_data[['temp', 'atemp', 'humidity', 'windspeed']]
    test_label = test_data.drop(columns=['temp', 'atemp', 'humidity', 'windspeed'])

    # ë³€í™˜ ë°� ì �ìš©
    scaler.transform(test_num)

    # ë�°ì�´í„° í”„ë ˆì�„ìœ¼ë¡œ ë³€í™˜ ë°� í•©ì¹˜ê¸°
    test_scaled = pd.DataFrame(test_data, columns=['temp', 'atemp', 'humidity', 'windspeed'])
    test_mm = pd.concat([test_label, test_scaled], axis=1)


    return X_train_mm, X_val_mm, X_test_mm, test_mm


X_train_mm, X_val_mm, X_test_mm, test_mm = minmax_scaler(X_train, X_val, X_test, test_data)


X_train_mm[['temp', 'atemp', 'humidity', 'windspeed']].describe(), X_val_mm[['temp', 'atemp', 'humidity', 'windspeed']].describe(), X_test_mm[['temp', 'atemp', 'humidity', 'windspeed']].describe(), test_mm[['temp', 'atemp', 'humidity', 'windspeed']].describe()


X_train_ss[['year', 'month', 'day', 'hour', 'season', 'holiday', 'workingday', 'weather']].head()
X_val_ss[['year', 'month', 'day', 'hour', 'season', 'holiday', 'workingday', 'weather']].head()
X_test_ss[['year', 'month', 'day', 'hour', 'season', 'holiday', 'workingday', 'weather']].head()
X_train_mm[['year', 'month', 'day', 'hour', 'season', 'holiday', 'workingday', 'weather']].head()
X_val_mm[['year', 'month', 'day', 'hour', 'season', 'holiday', 'workingday', 'weather']].head()
X_test_mm[['year', 'month', 'day', 'hour', 'season', 'holiday', 'workingday', 'weather']].head()


import numpy as np

# yì—� 1ì�„ ë�”í•´ì£¼ëŠ” np.log1p()ë¥¼ ì‚¬ìš©í•œë‹¤.  # # Use np.log1p () that adds 1 to y.
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# ëª¨ë�¸ë§ˆë‹¤ ë”•ì…”ë„ˆë¦¬ì—� ì¶”ê°€  # # Add to dictionary for each model
models = {
    'LinearRegression' : LinearRegression(),
    'Ridge' : Ridge(alpha=1.0),
    'Lasso' : Lasso(alpha=0.1),
    'ElasticNet' : ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# ë°˜ë³µë¬¸ìœ¼ë¡œ ëª¨ë�¸ì—� ì �ìš©  # # Applied to the model with a loop
for name, model in models.items():
    model.fit(X_train_ss, y_train_log)
    print(f'{name} ëª¨ë�¸ X_train_ss í•™ìŠµ ì™„ë£Œ')  # print (f '{name} model x_train_ss learning complete')


# ëª¨ë�¸ë§ˆë‹¤ ê³„ìˆ˜ì™€ ì ˆí�¸, R2 ìŠ¤ì½”ì–´ êµ¬í•˜ê¸°  # # Finding r2 score, coefficient and fragment for each model
for name, model in models.items():
    print(f'{name}ì�˜ ê³„ìˆ˜ : {model.coef_}')  # Print
    print(f'{name}ì�˜ ì ˆí�¸ : {model.intercept_}')  # print (f '{name} section: {model.intercept_}')
    print(f'{name}ì�˜ R2 score : {model.score(X_train_ss, y_train_log)}')  # print (f '{name} R2 score: {model.Score (x_train_ss, y_train_log)}')

    plt.figure(figsize=(9, 5))
    plt.bar(X_train.columns, model.coef_)
    plt.xticks(rotation=20)
    plt.xlabel('Features')
    plt.ylabel('Coefficients')
    plt.title(f'Coefficients of {name}')
    plt.show()


from sklearn.model_selection import GridSearchCV

# parameter grid
param_grid = {
    'Ridge': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    'Lasso': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
    'ElasticNet': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
}

# ì„ í˜•íšŒê·€ëŠ” í•˜ì�´í�¼ íŒŒë�¼ë¯¸í„°ê°€ ì—†ìœ¼ë¯€ë¡œ ì œì™¸  # # Excluded because there is no hyper parameter for linear regression
models = {
    'Ridge' : Ridge(),
    'Lasso' : Lasso(),
    'ElasticNet' : ElasticNet()
}

# ê°�ê°�ì�˜ ëª¨ë�¸ì—�ì„œ ìµœì � ëª¨ë�¸ ì €ì�¥ìš© ë”•ì…”ë„ˆë¦¬  # # Dictionary for storage of optimal model in each model
best_models = {}

# ë°˜ë³µë¬¸ ë�Œë¦¬ê¸°  # # Turn the loop
for name, model in models.items():
    print(f'{name} ëª¨ë�¸ ì‹œì�‘')  # print (f'nate} model start ')

# ê·¸ë¦¬ë“œ ì„œì¹˜ ì„¤ì •  # # Grid search setting
    grid_search = GridSearchCV(
        model,
        param_grid[name],
        scoring='neg_mean_squared_log_error',
        refit=True,
        cv=5,
        n_jobs=1,
    )

# ê·¸ë¦¬ë“œ ì„œì¹˜ ì �ìš©  # # Grid search application
    grid_search.fit(X_train_ss, y_train_log)
    print(f'{name} ëª¨ë�¸ ì™„ë£Œ')  # print (f'nate} model completed ')

# ìµœì � ëª¨ë�¸ ì—…ë�°ì�´íŠ¸  # # Optimal model update
    best_models[name] = grid_search.best_estimator_
    print(f'{name}ì�˜ ìµœì � íŒŒë�¼ë¯¸í„° : {grid_search.best_estimator_}')  # PRINT (F '{name} optimal parameters: {grid_search.best_estimator_}')


best_models['LinearRegression'] = LinearRegression()

best_models


# ì„ í˜•íšŒê·€ëŠ” ì•„ì§� ì‹¤í–‰ë�˜ì§€ ì•Šì•˜ê¸°ê²Œ ì‹¤í–‰í•´ì•¼ í•œë‹¤.  # # Linear regression must be executed so that it has not been executed yet.
best_models['LinearRegression'].fit(X_train_ss, y_train_log)


from sklearn.metrics import mean_squared_log_error

# RMSLE í•¨ìˆ˜  # # Rmsle function
def rmsle(y_true_log, y_pred_log):
    return np.sqrt(mean_squared_log_error(y_true_log, y_pred_log))

# ë¹„êµ�ë¥¼ ìœ„í•´ ëª¨ë�¸ë³„ rmsleë¥¼ ì €ì�¥  # #Save RMSLE for each model for comparison
val_rmsle = {}

# ê²€ì¦�ìš© ë�°ì�´í„°ì—�ì„œ ìµœì �ì�˜ ëª¨ë�¸ ì°¾ê¸°  # # Find the best model in verification data
for name, model in best_models.items():
    y_val_pred_log = model.predict(X_val_ss)
    score = rmsle(y_val_log, y_val_pred_log)

# ë”•ì…”ë„ˆë¦¬ì—� ì €ì�¥  # #Save to the dictionary
    val_rmsle[name] = score

print(f'{name}ì�˜ Validation RMSLE : {score}')  # print (f '{name} validation rmsle: {score}')


min(val_rmsle, key=val_rmsle.get)


# LinearRegressionì�„ í…ŒìŠ¤íŠ¸ ë�°ì�´í„°ì—� ì �ìš©  # # Apply linearrregression to test data
y_test_pred_log = best_models['LinearRegression'].predict(X_test_ss)

score = rmsle(y_test_log, y_test_pred_log)
print(f'LinearRegressionì�˜ Test RMSLE : {score}')  # print


# LinearRegressionì�„ 'ì‹¤ì œ' í…ŒìŠ¤íŠ¸ ë�°ì�´í„°ì—� ì �ìš©
test_pred_log = best_models['LinearRegression'].predict(test_ss)

test_pred_log


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# ëª¨ë�¸ë§ˆë‹¤ ë”•ì…”ë„ˆë¦¬ì—� ì¶”ê°€  # # Add to dictionary for each model
models = {
    'LinearRegression' : LinearRegression(),
    'Ridge' : Ridge(alpha=1.0),
    'Lasso' : Lasso(alpha=0.1),
    'ElasticNet' : ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# ë°˜ë³µë¬¸ìœ¼ë¡œ ëª¨ë�¸ì—� ì �ìš©  # # Applied to the model with a loop
for name, model in models.items():
    model.fit(X_train_mm, y_train_log)
    print(f'{name} ëª¨ë�¸ X_train_mm í•™ìŠµ ì™„ë£Œ')  # print (f '{name} model x_train_mm learning completed')


# ëª¨ë�¸ë§ˆë‹¤ ê³„ìˆ˜ì™€ ì ˆí�¸, R2 ìŠ¤ì½”ì–´ êµ¬í•˜ê¸°  # # Finding r2 score, coefficient and fragment for each model
for name, model in models.items():
    print(f'{name}ì�˜ ê³„ìˆ˜ : {model.coef_}')  # Print
    print(f'{name}ì�˜ ì ˆí�¸ : {model.intercept_}')  # print (f '{name} section: {model.intercept_}')
    print(f'{name}ì�˜ R2 score : {model.score(X_train_mm, y_train_log)}')  # PRINT (f '{name} R2 score: {model.Score (x_train_mm, y_train_log)}')

    plt.figure(figsize=(9, 5))
    plt.bar(X_train.columns, model.coef_)
    plt.xticks(rotation=20)
    plt.xlabel('Features')
    plt.ylabel('Coefficients')
    plt.title(f'Coefficients of {name}')
    plt.show()


from sklearn.model_selection import GridSearchCV

# parameter grid
param_grid = {
    'Ridge': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    'Lasso': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
    'ElasticNet': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
}

# ì„ í˜•íšŒê·€ëŠ” í•˜ì�´í�¼ íŒŒë�¼ë¯¸í„°ê°€ ì—†ìœ¼ë¯€ë¡œ ì œì™¸  # # Excluded because there is no hyper parameter for linear regression
models = {
    'Ridge' : Ridge(),
    'Lasso' : Lasso(),
    'ElasticNet' : ElasticNet()
}

# ê°�ê°�ì�˜ ëª¨ë�¸ì—�ì„œ ìµœì � ëª¨ë�¸ ì €ì�¥ìš© ë”•ì…”ë„ˆë¦¬  # # Dictionary for storage of optimal model in each model
best_models = {}

# ë°˜ë³µë¬¸ ë�Œë¦¬ê¸°  # # Turn the loop
for name, model in models.items():
    print(f'{name} ëª¨ë�¸ ì‹œì�‘')  # print (f'nate} model start ')

# ê·¸ë¦¬ë“œ ì„œì¹˜ ì„¤ì •  # # Grid search setting
    grid_search = GridSearchCV(
        model,
        param_grid[name],
        scoring='neg_mean_squared_log_error',
        refit=True,
        cv=5,
        n_jobs=1,
    )

# ê·¸ë¦¬ë“œ ì„œì¹˜ ì �ìš©  # # Grid search application
    grid_search.fit(X_train_mm, y_train_log)
    print(f'{name} ëª¨ë�¸ ì™„ë£Œ')  # print (f'nate} model completed ')

# ìµœì � ëª¨ë�¸ ì—…ë�°ì�´íŠ¸  # # Optimal model update
    best_models[name] = grid_search.best_estimator_
    print(f'{name}ì�˜ ìµœì � íŒŒë�¼ë¯¸í„° : {grid_search.best_estimator_}')  # PRINT (F '{name} optimal parameters: {grid_search.best_estimator_}')


best_models['LinearRegression'] = LinearRegression()

best_models


# ì„ í˜•íšŒê·€ëŠ” ì•„ì§� ì‹¤í–‰ë�˜ì§€ ì•Šì•˜ê¸°ê²Œ ì‹¤í–‰í•´ì•¼ í•œë‹¤.  # # Linear regression must be executed so that it has not been executed yet.
best_models['LinearRegression'].fit(X_train_mm, y_train_log)


from sklearn.metrics import mean_squared_log_error

# RMSLE í•¨ìˆ˜  # # Rmsle function
def rmsle(y_true_log, y_pred_log):
    return np.sqrt(mean_squared_log_error(y_true_log, y_pred_log))

# ë¹„êµ�ë¥¼ ìœ„í•´ ëª¨ë�¸ë³„ rmsleë¥¼ ì €ì�¥  # #Save RMSLE for each model for comparison
val_rmsle = {}

# ê²€ì¦�ìš© ë�°ì�´í„°ì—�ì„œ ìµœì �ì�˜ ëª¨ë�¸ ì°¾ê¸°  # # Find the best model in verification data
for name, model in best_models.items():
    y_val_pred_log = model.predict(X_val_mm)
    score = rmsle(y_val_log, y_val_pred_log)

# ë”•ì…”ë„ˆë¦¬ì—� ì €ì�¥  # #Save to the dictionary
    val_rmsle[name] = score

print(f'{name}ì�˜ Validation RMSLE : {score}')  # print (f '{name} validation rmsle: {score}')


min(val_rmsle, key=val_rmsle.get)


# LinearRegressionì�„ í…ŒìŠ¤íŠ¸ ë�°ì�´í„°ì—� ì �ìš©  # # Apply linearrregression to test data
y_test_pred_log = best_models['LinearRegression'].predict(X_test_mm)

score = rmsle(y_test_log, y_test_pred_log)
print(f'LinearRegressionì�˜ Test RMSLE : {score}')  # print


# LinearRegressionì�„ 'ì‹¤ì œ' í…ŒìŠ¤íŠ¸ ë�°ì�´í„°ì—� ì �ìš©
test_pred_log = best_models['LinearRegression'].predict(test_mm)

test_pred_log

