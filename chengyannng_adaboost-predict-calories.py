import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


traindata = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv').drop(columns=['id'])
testdata = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


traindata.head()


testdata.head()


traindata.isnull().sum()


testdata.isnull().sum()


traindata.info()


testdata.info()


traindata.duplicated().sum()


traindata = traindata.drop_duplicates()


testdata.duplicated().sum()


traindata.describe().T


testdata.describe().T


def features_engineering(df):
    df = df.copy()
    if df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].replace({'male': 1, 'female': 0})
        
    df['BMI'] = np.round(df['Weight'] / ((df['Height']/100)**2), 2)
    # Mifflin-St Jeor(1990)
    # For men: BMR = 10W + 6.25H – 5A + 5
    # For women: BMR = 10W + 6.25H – 5A – 161
    # BMR Calculator: Learn Your Basal Metabolic Rate In 2025
    df['BMR'] = np.where(df['Sex'] == 1,
                         10*df['Weight']+6.25*df['Height']-5*df['Age']+5,
                         10*df['Weight']+6.25*df['Height']-5*df['Age']-161)

    # Dr. Tanaka(2001)
    # HRmax = 208 - ( 0.7 * A )
    df['HRmax'] = 208-(0.7*df['Age'])

    # Intensity = MET（Metabolic Equivalent） * Duration
    # How are METs calculated?
    # or replace MET with Heart_Rate
    # Intensity = Heart_Rate  * Duration
    df['Indensity_Duration'] = df['Heart_Rate']*df['Duration']
    
    # BodyTemp_Interaction = Body_Temp * Duration
    df['BodyXTemp_Duration'] = df['Body_Temp']*df['Duration']

    # Weight_Duration = Weight * Duration
    df['Weight_Duration'] = df['Weight']*df['Duration']

    # Age_HR = Age * Heart_Rate
    df['Age_HR'] = df['Age']*df['Heart_Rate']
    
    # df = df.drop(columns=['Height', 'Weight', 'Age', 'Duration', 'Heart_Rate', 'Body_Temp'])

    return df


traindata = features_engineering(traindata)


traindata.head()


testdata = features_engineering(testdata)


testdata.head()


# plt.figure(figsize=(10, 8))
# sns.heatmap(data=traindata.drop(columns='Sex').corr(), fmt='.2f', annot=True)
# plt.show()


# sex_counts = traindata['Sex'].value_counts().reset_index()
# sex_counts.columns = ['sex', 'counts']

# plt.figure(figsize=(4, 3))
# sns.barplot(data=sex_counts, x='sex', y='counts')
# plt.title('traindata')


# tsetsex_counts = testdata['Sex'].value_counts().reset_index()
# tsetsex_counts.columns = ['sex', 'counts']

# plt.figure(figsize=(4, 3))
# sns.barplot(data=tsetsex_counts, x='sex', y='counts')
# plt.title('testdata')


# sns.histplot(data=traindata, x='Age', fill=False, kde=True, binwidth=1)


# sns.scatterplot(data=traindata, x='Height', y='Weight', hue='Sex')


# sns.scatterplot(data=traindata, x='BMI', y='Calories', hue='Sex')


# sns.scatterplot(data=traindata, x='Duration', y='Calories', hue='Sex', style='Sex')


# sns.scatterplot(data=traindata, x='Body_Temp', y='Calories', hue='Sex')


# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA


# traindata['Sex'] = traindata['Sex'].replace({'male': 1, 'female': 0})


# scaler = StandardScaler()
# traindata_scaled = scaler.fit_transform(traindata.drop(columns='Calories'))
# pca = PCA(n_components=2)
# traindata_pca = pca.fit_transform(traindata_scaled)


# df = pd.DataFrame(traindata_pca)
# df.columns = ['PC1', 'PC2']
# df['Sex'] = traindata['Sex'].values
# df.head(3)


# sns.scatterplot(data=df, x='PC1', y='PC2', hue='Sex')


from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV


X = traindata.drop(columns='Calories')
y = traindata['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
abr = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, loss='exponential', random_state=42)
model = abr.fit(X_train, y_train)
y_pred = np.maximum(model.predict(X_test), 0)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"RMSLE: {rmsle:.4f}")



sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')


def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
plot_feature_importance(model, feature_names=X.columns)


# def cv_RMSLE(X, y, n_splits=10):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     RMSLE_scores = []

#     for train_x, test_x in kf.split(X):
#         X_train, X_test = X.iloc[train_x], X.iloc[test_x]
#         y_train, y_test = y.iloc[train_x], y.iloc[test_x]

#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         lr = LinearRegression()
#         model = lr.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)
#         y_pred = np.maximum(y_pred, 0)

#         score = np.sqrt(mean_squared_log_error(y_test, y_pred))
#         RMSLE_scores.append(score)

#     return np.mean(RMSLE_scores), RMSLE_scores



# male_data = traindata[traindata['Sex'] == 1]
# female_data = traindata[traindata['Sex'] == 0]

# x_male = male_data.drop(columns='Calories')
# y_male = male_data['Calories']
# x_female = female_data.drop(columns='Calories')
# y_female = female_data['Calories']

# mean_rmsle_male, scores_male = cv_RMSLE(x_male, y_male)
# mean_rmsle_female, scores_female = cv_RMSLE(x_female, y_female)

# print(f"male 5-Fold RMSLE mean: {mean_rmsle_male:.4f}")
# print(f"every fold score: {np.round(scores_male, 4)}\n")

# print(f"female 5-Fold RMSLE mean: {mean_rmsle_female:.4f}")
# print(f"every fold score: {np.round(scores_female, 4)}")



# def rmsle_f(y_test, y_pred):
#     y_pred = np.maximum(y_pred, 0)
#     return np.sqrt(mean_squared_log_error(y_test, y_pred))

# rmsle_score = make_scorer(rmsle_f, greater_is_better=False)


# param_grid = {'n_estimators': [100, 300],
#              'max_depth': [2, 5, 10], 
#              'min_samples_split': [2, 5],
#              'min_samples_leaf': [1, 3]}
# rf = RandomForestRegressor(random_state=42, n_jobs=-1)
# grid_search = RandomizedSearchCV(estimator=rf,
#                                  param_distributions=param_grid,
#                                  n_iter=30,
#                                  scoring=rmsle_score,
#                                  cv=3,
#                                  n_jobs=-1,
#                                  verbose=2,
#                                  random_state=42)
# grid_search.fit(X_sample, y_sample)


# * Best Params: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 3, 'max_depth': 10}
# * Best RMSLE: 0.07653252126600418


# print("Best Params:", grid_search.best_params_)
# print("Best RMSLE:", -(grid_search.best_score_))


# best_params = grid_search.best_params_
# best_rf = RandomForestRegressor(random_state=42, n_jobs=-1, 
#                                 n_estimators=300,
#                                 min_samples_split=2,
#                                 min_samples_leaf=3,
#                                 max_depth=10)
# best_model = best_rf.fit(X, y)


y_pred = np.maximum(model.predict(testdata.drop(columns='id')), 0)


results = pd.DataFrame({'id': testdata['id'], 'Calories': y_pred})


results.head()


results.shape


results.to_csv('submission.csv', index=False)

