import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df


df.info()


df = df.replace([np.inf, -np.inf], np.nan)  


df.isna().sum().sum()


df.duplicated().sum()


df.hist(figsize=(30, 15))


sns.countplot(df, x='rainfall')


plt.figure(figsize=(20, 20))
sns.heatmap(df.drop('id', axis=1).corr(), cmap='jet', annot=True)


plt.figure(figsize=(20, 20))
sns.heatmap(df.drop('id', axis=1).corr(method='spearman'), cmap='jet', annot=True)


# plt.figure(figsize=(50, 50))
# sns.pairplot(df, hue="rainfall", diag_kind="hist")


plt.figure(figsize=(14, 14))
sns.jointplot(df, x='cloud', y='humidity', hue='rainfall')


plt.figure(figsize=(14, 14))
sns.jointplot(df, x='temparature', y='pressure', hue='rainfall')


from sklearn.utils import resample
minority = df[df.rainfall == 0]
majority = df[df.rainfall == 1]
minority = resample(minority,
             replace=True,
             n_samples=len(majority),
             random_state=239301)
minority.shape, majority.shape


df = pd.concat([minority, majority])


# Convert 'day' to datetime
df['day'] = pd.to_datetime(df['day'], errors='coerce')

# Extract temporal features
df['month'] = df['day'].dt.month
df['day_of_week'] = df['day'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Temperature features
df['temp_range'] = df['maxtemp'] - df['mintemp']
df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
df['temp_deviation'] = df['temparature'] - df['avg_temp']

# Dew point depression
df['dew_point_depression'] = df['temparature'] - df['dewpoint']

# Wind direction - sine and cosine transformation
df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
df.drop(columns=['wind_dir_rad'], inplace=True)

# Wind chill factor (simplified version)
df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temparature'] * (df['windspeed']**0.16)

# Interaction features
df['humidity_temp'] = df['humidity'] * df['temparature']
df['cloud_sunshine'] = df['cloud'] * df['sunshine']

# Rolling statistical features
df['rolling_temp_mean'] = df['avg_temp'].rolling(window=7).mean()
df['rolling_wind_mean'] = df['windspeed'].rolling(window=7).mean()
df['rolling_humidity_mean'] = df['humidity'].rolling(window=7).mean()

# Lag features
df['temp_lag_1'] = df['avg_temp'].shift(1)
df['humidity_lag_1'] = df['humidity'].shift(1)
df['windspeed_lag_1'] = df['windspeed'].shift(1)

# Pressure-Temperature interaction
df['pressure_temp_interaction'] = df['pressure'] * df['avg_temp']
# Wind-Speed-Temperature interaction
df['windspeed_temp_interaction'] = df['windspeed'] * df['avg_temp']

# Sunshine-Cloud interaction
df['sunshine_cloud_interaction'] = df['sunshine'] * df['cloud']

# Season feature
df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                  'Summer' if 6 <= x <= 8 else
                                  'Autumn' if 9 <= x <= 11 else 'Winter')

for c in ['pressure', 'maxtemp', 'temparature', 'humidity']:
    for gap in [1]:
        df[c+f"_shift{gap}"] = df[c].shift(gap)
        df[c+f"_diff{gap}"] = df[c].diff(gap)

# Binary encoding for season
df = pd.get_dummies(df, columns=['season'], drop_first=True)
# Drop original 'day' column
df.drop(columns=['day'], inplace=True)

df = df.drop(['mintemp', 'maxtemp', 'id'], axis=1).dropna(axis=0)


df.isna().sum()


from sklearn.model_selection import train_test_split
y = df.rainfall
X = df.drop('rainfall', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=239301, test_size=0.1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=239301, max_iter=10000).fit(X_train, y_train)
log_reg.score(X_train, y_train), log_reg.score(X_test, y_test)


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
svm = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X_train, y_train)
svm.score(X_train, y_train), svm.score(X_test, y_test)


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

xgb = XGBClassifier().fit(X_train, y_train)
params = {'min_child_weight': [1, 2, 3],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.4, 0.5],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_depth': [3, 4, 5, 6, 7, 8]
        }

clf = GridSearchCV(xgb,params,verbose=1,n_jobs=-1)
clf.fit(X, y)
print(clf.best_score_)
print(clf.best_params_)


xgb = XGBClassifier(**clf.best_params_).fit(X_train, y_train)
xgb.score(X_train, y_train), xgb.score(X_test, y_test)


from sklearn.neighbors import KNeighborsClassifier as KNN
KNN().fit(X_train, y_train).score(X_test, y_test)


from sklearn.neighbors import KNeighborsClassifier as KNN
train_accuracies, test_accuracies = [], []

for n in range(1, 20):
    knn = KNN(n).fit(X_train, y_train)
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))


plt.figure(figsize=(20, 10))
plt.plot(range(1, 20), train_accuracies, label='Train')
plt.plot(range(1, 20), test_accuracies, label='Test')
plt.xticks(range(1, 20))
plt.title('KNN Performance')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


knn = KNN(6).fit(X_train, y_train)
knn.score(X_train, y_train), knn.score(X_test, y_test)


from sklearn.metrics import ConfusionMatrixDisplay
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
models = [log_reg, svm, xgb, knn]
titles = ['Logistic Regression', 'SVM', 'XGBoost', 'KNN']

for i in range(len(models)):
    ConfusionMatrixDisplay.from_estimator(models[i], X_train, y_train, ax=axs.flatten()[i], cmap='Reds')
    axs.flatten()[i].set_title(titles[i])


fig, axs = plt.subplots(2, 2, figsize=(16, 10))
for i in range(len(models)):
    ConfusionMatrixDisplay.from_estimator(models[i], X_test, y_test, ax=axs.flatten()[i], cmap='Blues')
    axs.flatten()[i].set_title(titles[i])


from sklearn.model_selection import learning_curve

fig, axs = plt.subplots(2, 2, figsize=(16, 9))

for ax, model, title in zip(axs.flatten(), models, titles):
    steps, train_scores, val_scores = learning_curve(model, X_train, y_train)
    
    ax.plot(steps, train_scores.mean(axis=1), label=f'{title} - Train', color="royalblue", marker='o')
    ax.plot(steps, val_scores.mean(axis=1), label=f'{title} - Val', color='mediumseagreen', marker='o')
    ax.fill_between(steps, train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), color="royalblue", alpha=0.3)
    ax.fill_between(steps, val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), color='mediumseagreen', alpha=0.3)
    
    ax.set_title(title)
    ax.legend()
    


from sklearn.metrics import RocCurveDisplay
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
for i, model in enumerate(models):
    display = RocCurveDisplay.from_estimator(model, X_train, y_train, ax=axs.flatten()[i])
    axs.flatten()[i].set_title(titles[i])
    axs.flatten()[i].fill_between(display.fpr, display.tpr, alpha=0.3)


from sklearn.metrics import RocCurveDisplay
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
for i in range(len(models)):
    display = RocCurveDisplay.from_estimator(models[i], X_test, y_test, ax=axs.flatten()[i])
    axs.flatten()[i].set_title(titles[i])
    axs.flatten()[i].fill_between(display.fpr, display.tpr, alpha=0.3)


from sklearn.metrics import PrecisionRecallDisplay
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
for i, model in enumerate(models):
    display = PrecisionRecallDisplay.from_estimator(model, X_train, y_train, ax=axs.flatten()[i])
    axs.flatten()[i].set_title(titles[i])
    axs.flatten()[i].fill_between(display.recall, display.precision, alpha=0.4)


fig, axs = plt.subplots(2, 2, figsize=(16, 10))

for i, model in enumerate(models):
    display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axs.flatten()[i])
    axs.flatten()[i].set_title(titles[i])
    axs.flatten()[i].fill_between(display.recall, display.precision, alpha=0.4)


from yellowbrick.classifier import DiscriminationThreshold
DiscriminationThreshold(log_reg).fit(X_test, y_test).poof()


DiscriminationThreshold(knn).fit(X_test, y_test).poof()


DiscriminationThreshold(svm).fit(X_test, y_test).poof()


DiscriminationThreshold(xgb).fit(X_test, y_test).poof()


from sklearn.metrics import classification_report
y_pred = log_reg.predict(X_train)
print(classification_report(y_pred, y_train))


y_pred = xgb.predict(X_test)
print(classification_report(y_pred, y_test))


y_pred = svm.predict(X_test)
print(classification_report(y_pred, y_test))


y_pred = knn.predict(X_test)
print(classification_report(y_pred, y_test))


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
# Convert 'day' to datetime
test['day'] = pd.to_datetime(test['day'], errors='coerce')

# Extract temporal features
test['month'] = test['day'].dt.month
test['day_of_week'] = test['day'].dt.dayofweek
test['is_weekend'] = test['day_of_week'].isin([5, 6]).astype(int)

# Temperature features
test['temp_range'] = test['maxtemp'] - test['mintemp']
test['avg_temp'] = (test['maxtemp'] + test['mintemp']) / 2
test['temp_deviation'] = test['temparature'] - test['avg_temp']

# Dew point depression
test['dew_point_depression'] = test['temparature'] - test['dewpoint']

# Wind direction - sine and cosine transformation
test['wind_dir_rad'] = np.deg2rad(test['winddirection'])
test['wind_dir_sin'] = np.sin(test['wind_dir_rad'])
test['wind_dir_cos'] = np.cos(test['wind_dir_rad'])
test.drop(columns=['wind_dir_rad'], inplace=True)

# Wind chill factor (simplified version)
test['wind_chill'] = 13.12 + 0.6215 * test['temparature'] - 11.37 * (test['windspeed']**0.16) + 0.3965 * test['temparature'] * (test['windspeed']**0.16)

# Interaction features
test['humidity_temp'] = test['humidity'] * test['temparature']
test['cloud_sunshine'] = test['cloud'] * test['sunshine']

# Rolling statistical features
test['rolling_temp_mean'] = test['avg_temp'].rolling(window=7).mean()
test['rolling_wind_mean'] = test['windspeed'].rolling(window=7).mean()
test['rolling_humidity_mean'] = test['humidity'].rolling(window=7).mean()

# Lag features
test['temp_lag_1'] = test['avg_temp'].shift(1)
test['humidity_lag_1'] = test['humidity'].shift(1)
test['windspeed_lag_1'] = test['windspeed'].shift(1)

# Pressure-Temperature interaction
test['pressure_temp_interaction'] = test['pressure'] * test['avg_temp']
# Wind-Speed-Temperature interaction
test['windspeed_temp_interaction'] = test['windspeed'] * test['avg_temp']

# Sunshine-Cloud interaction
test['sunshine_cloud_interaction'] = test['sunshine'] * test['cloud']

# Season feature
test['season'] = test['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                  'Summer' if 6 <= x <= 8 else
                                  'Autumn' if 9 <= x <= 11 else 'Winter')

for c in ['pressure', 'maxtemp', 'temparature', 'humidity']:
    for gap in [1]:
        test[c+f"_shift{gap}"] = test[c].shift(gap)
        test[c+f"_diff{gap}"] = test[c].diff(gap)

# Binary encoding for season
test = pd.get_dummies(test, columns=['season'], drop_first=True)
# Drop original 'day' column
test.drop(columns=['day'], inplace=True)

test = test.drop(['mintemp', 'maxtemp', 'id'], axis=1).dropna(axis=0)
test = scaler.transform(test)
preds = xgb.predict(test)


data = {'id': np.arange(2190, 2190+len(preds)), 'rainfall': preds}
pd.DataFrame(data).to_csv('submission.csv', index=False)

