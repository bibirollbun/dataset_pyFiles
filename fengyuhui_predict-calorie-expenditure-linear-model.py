import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

train_df.set_index('id', inplace=True)
test_df.set_index('id', inplace=True)


train_df.head()


train_df.shape


test_df.shape


test_df.info()


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,3))

# 1. plot original distribution
sns.histplot(x=train_df.Calories, ax=axes[0])
plt.title('Calories')

# 2. plot log calories
train_df['Calories_per_Duration'] = train_df.Calories/train_df.Duration
sns.histplot(
    x=train_df.loc[train_df.Calories_per_Duration<10,'Calories_per_Duration'],
    ax=axes[1])
plt.title('Calories/Duration')

# 3. calories per duration
train_df['log_Calories_per_Duration'] = np.log(train_df.Calories_per_Duration)
sns.kdeplot(
    data=train_df.loc[train_df.Calories_per_Duration<10,:],
    x='Calories_per_Duration',
    fill=True,
    hue='Sex', ax=axes[2])


numeric_features = ['Age','Height','Weight','Duration','Heart_Rate', 'Body_Temp']
categorical_features = ['Sex']


def numeric_feature_exploration(numeric_features=numeric_features):
    n_features = len(numeric_features)
    fig, axes = plt.subplots(nrows=n_features, 
                             ncols=3, 
                             figsize=(20,3*n_features))
    
    for i in range(n_features):
        feature = numeric_features[i]

        # 1. Compare distribution between train and test
        train_data = train_df.loc[:, [feature]]
        train_data['source'] = 'train'
        test_data = test_df.loc[:,[feature]]
        test_data['source'] = 'test'
        all_data = pd.concat([train_data, test_data])
        sns.kdeplot(data=all_data, x=feature, hue='source', 
                    fill=True, ax=axes[i][0])
        plt.title(feature)
        
        # 2. plot association with Calories
        sns.scatterplot(data=train_df, x=feature, y='Calories',
                        hue='Sex',
                        s=3, alpha=0.1, ax=axes[i][1])
        plt.title(f'{feature} vs Calories')

        # 3. plot association with Calories per Duration
        sns.scatterplot(data=train_df.loc[train_df.Calories_per_Duration<=10], 
                        x=feature, 
                        y='Calories_per_Duration',
                        hue='Sex',
                        s=3, alpha=0.1, ax=axes[i][2])
        plt.title(f'{feature} vs Calories per Duration')


#numeric_feature_exploration()


def explore_correlations(numeric_features=numeric_features):
    n_features = len(numeric_features)
    fig, axes = plt.subplots(nrows=n_features, 
                             ncols=n_features, 
                             figsize=(3*n_features,3*n_features))
    
    for i in range(n_features-1):
        for j in range(i+1, n_features):
            if i < j:
                sns.scatterplot(
                    data=train_df, 
                    x=numeric_features[i], 
                    y=numeric_features[j],
                    hue='Sex',
                    s=3, alpha=0.1, ax=axes[j][i])


#explore_correlations()


# One-Hot Encoding for Sex
train_df['female'] = 0
train_df.loc[train_df.Sex=='female','female'] = 1

test_df['female'] = 0
test_df.loc[test_df.Sex=='female','female'] = 1


# 1. Create Highter orders for Heart_Rate and Body_Temp

train_df['log_Heart_Rate'] = np.log(train_df.Heart_Rate)
test_df['log_Heart_Rate'] = np.log(test_df.Heart_Rate)

train_df['Heart_Rate_2'] = train_df.Heart_Rate ** 2
test_df['Heart_Rate_2'] = test_df.Heart_Rate ** 2

train_df['log_Body_Temp'] = np.log(train_df.Body_Temp)
test_df['log_Body_Temp'] = np.log(test_df.Body_Temp)

train_df['Body_Temp_2'] = train_df.Body_Temp ** 2
test_df['Body_Temp_2'] = test_df.Body_Temp ** 2

train_df['Duration_2'] = train_df.Duration ** 2
test_df['Duration_2'] = test_df.Duration ** 2

train_df['log_Duration'] = np.log(train_df.Duration)
test_df['log_Duration'] = np.log(test_df.Duration)

# 2. Weight By Height
train_df['Weight_by_Height'] = train_df.Weight/train_df.Height
test_df['Weight_by_Height'] = test_df.Weight/test_df.Height

train_df['Heart_Rate_by_Duration'] = train_df.Heart_Rate/train_df.Duration
test_df['Heart_Rate_by_Duration'] = test_df.Heart_Rate/test_df.Duration

train_df['Body_Temp_by_Duration'] = train_df.Body_Temp/train_df.Duration
test_df['Body_Temp_by_Duration'] = test_df.Body_Temp/test_df.Duration

train_df['Body_Temp_by_Heart_Rate'] = train_df.Body_Temp/train_df.Heart_Rate
test_df['Body_Temp_by_Heart_Rate'] = test_df.Body_Temp/test_df.Heart_Rate


# numeric_feature_exploration(
#     numeric_features=['Weight_by_Height','Duration_2']
# )


from sklearn.decomposition import PCA

pca_features = ['Age','Height','Weight','Duration','Heart_Rate', 'Body_Temp',
               'Weight_by_Height']

# - female
X_female = train_df.loc[train_df.female==1, pca_features]
X_female_scaled = (X_female - X_female.mean(axis=0))/X_female.std(axis=0)

X_female_test = test_df.loc[test_df.female==1, pca_features]
test_X_female_scaled = (X_female_test - X_female_test.mean(axis=0))/X_female_test.std(axis=0)

pca_female = PCA()
X_female_pca = pca_female.fit_transform(X_female_scaled)
test_X_female_pca = pca_female.transform(test_X_female_scaled)

# - male
X_male = train_df.loc[train_df.female==0, pca_features]
X_male_scaled = (X_male - X_male.mean(axis=0))/X_male.std(axis=0)

X_male_test = test_df.loc[test_df.female==0, pca_features]
test_X_male_scaled = (X_male_test - X_male_test.mean(axis=0))/X_male_test.std(axis=0)

pca_male = PCA()
X_male_pca = pca_male.fit_transform(X_male_scaled)
test_X_male_pca = pca_male.transform(test_X_male_scaled)

# - Create Dataset
component_names = [f'PC{i+1}' for i in range(X_female.shape[1])]

X_female_pca = pd.DataFrame(
    X_female_pca, 
    columns=component_names,
    index=train_df.index[train_df.female==1]
)
test_X_female_pca = pd.DataFrame(
    test_X_female_pca, 
    columns=component_names,
    index=test_df.index[test_df.female==1]
)

X_male_pca = pd.DataFrame(
    X_male_pca,
    columns=component_names,
    index=train_df.index[train_df.female==0]
)
test_X_male_pca = pd.DataFrame(
    test_X_male_pca, 
    columns=component_names,
    index=test_df.index[test_df.female==0]
)

# - Merge back 
train_df = train_df.merge(
    pd.concat([X_female_pca,X_male_pca], axis=0),
    on='id',how='left'
)

test_df = test_df.merge(
    pd.concat([test_X_female_pca, test_X_male_pca], axis=0),
    on='id', how='left'
)


# from sklearn.decomposition import PCA

# pca_features = ['Age','Height','Weight','Duration','Heart_Rate', 'Body_Temp']
# X = train_df.loc[:, pca_features]
# X_scaled = (X - X.mean(axis=0))/X.std(axis=0)

# X_test = test_df.loc[:, pca_features]
# test_X_scaled = (X_test - X_test.mean(axis=0))/X_test.std(axis=0)

# pca = PCA()
# X_pca = pca.fit_transform(X_scaled)
# test_X_pca = pca.transform(test_X_scaled)


# # Plot explained variance
# explained_variance = np.cumsum(pca.explained_variance_ratio_)
# plt.figure(figsize=(10,5))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Scree Plot')
# plt.show()


# # Include first 3 components
# component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
# X_pca = pd.DataFrame(X_pca, columns=component_names)
# test_X_pca = pd.DataFrame(test_X_pca, columns=component_names)

# train_df['PC1'] = X_pca.PC1
# train_df['PC2'] = X_pca.PC2
# train_df['PC3'] = X_pca.PC3

# test_df['PC1'] = test_X_pca.PC1
# test_df['PC2'] = test_X_pca.PC2
# test_df['PC3'] = test_X_pca.PC3


# Train Validation Split
from sklearn.model_selection import train_test_split

X = train_df.copy()
y = X.pop('Calories')
y_per_dur = X.pop('Calories_per_Duration')
y_log_per_dur = X.pop('log_Calories_per_Duration')
test_X = test_df.copy()

train_X,valid_X, train_y,valid_y, train_y_per_dur,valid_y_per_dur, train_y_log_per_dur, valid_y_log_per_dur= train_test_split(X, y, y_per_dur,y_log_per_dur, random_state = 1)


# Calculate RMSE
from sklearn.metrics import mean_squared_log_error

def calculate_RMSLE(pred_train, 
                   pred_valid,
                   train_y=train_y, 
                   valid_y=valid_y):

    train_msle = mean_squared_log_error(pred_train, train_y)
    train_rmsle = np.sqrt(train_msle)

    valid_msle = mean_squared_log_error(pred_valid, valid_y)
    valid_rmsle = np.sqrt(valid_msle)
    
    print(f'Train RMSLE is {train_rmsle}')
    print(f'Valid RMSLE is {valid_rmsle}')


# Plot feature vs predict
def resid_plot_feature_num(pred_valid_y,
                           num_features=numeric_features,
                           valid_X=valid_X,
                           valid_y=valid_y
                          ):

    fig, axes = plt.subplots(nrows=len(num_features), ncols=2,
                             figsize=(20, 3*len(num_features)))

    for i in range(len(num_features)):
        feature = num_features[i]

        # 1. valid, pred vs feature
        sns.scatterplot(data=valid_X, x=feature, y=pred_valid_y, hue='Sex',
                        s=1, alpha=0.3, ax=axes[i][0])

        # 2. valid, resid vs feature
        sns.scatterplot(data=valid_X, x=feature, y=valid_y - pred_valid_y, 
                        hue='Sex', s=1, alpha=0.5, ax=axes[i][1])


from sklearn.linear_model import LinearRegression

reg_features = ['Age','Height','Weight','Duration','Heart_Rate', 
                'Body_Temp', 'female']

reg = LinearRegression()
reg.fit(train_X.loc[:,reg_features], train_y)

pred_train = pd.DataFrame({
    'pred': reg.predict(train_X.loc[:,reg_features])
},index=train_X.index)
pred_train.loc[pred_train.pred<0, 'pred'] = 1.0

pred_valid= pd.DataFrame({
    'pred': reg.predict(valid_X.loc[:,reg_features])
}, index=valid_X.index)
pred_valid.loc[pred_valid.pred<0, 'pred'] = 1.0

calculate_RMSLE(pred_train.pred, pred_valid.pred, train_y, valid_y)


from sklearn.linear_model import LinearRegression

reg_features = ['Age','Height','Weight','Duration','Heart_Rate', 'Body_Temp',
               'log_Heart_Rate','log_Body_Temp',
               'Heart_Rate_2','Body_Temp_2',
               'Weight_by_Height',
               # add features
               'Heart_Rate_by_Duration',
               'Body_Temp_by_Duration']

# Male model
reg_male = LinearRegression()
reg_male.fit(train_X.loc[(train_X.female==0) & (train_y_per_dur<=10),reg_features],
             train_y_per_dur.loc[(train_X.female==0) & (train_y_per_dur<=10)])

# Female model
reg_female = LinearRegression()
reg_female.fit(train_X.loc[(train_X.female==1) & (train_y_per_dur<=10),reg_features], 
               train_y_per_dur.loc[(train_X.female==1) & (train_y_per_dur<=10)])


## Prediction
# Train
pred_train_sep = pd.DataFrame(
    {'Duration':train_X.Duration},
    index=train_X.index).merge(
    pd.concat([
        pd.DataFrame(
            {'pred':reg_male.predict(
                train_X.loc[train_X.female==0, reg_features])},
            index=train_X.index[train_X.female==0]),
        pd.DataFrame(
            {'pred':reg_female.predict(
                train_X.loc[train_X.female==1, reg_features])},
            index=train_X.index[train_X.female==1]),
    ]),
    on='id',how='left'
)
pred_train_sep['Calories_pred'] = pred_train_sep.Duration*pred_train_sep.pred 
pred_train_sep.loc[pred_train_sep.Calories_pred<0, 'Calories_pred'] = 1

# Validation
pred_valid_sep = pd.DataFrame(
    {'Duration':valid_X.Duration},
    index=valid_X.index).merge(
    pd.concat([
        pd.DataFrame(
            {'pred':reg_male.predict(
                valid_X.loc[valid_X.female==0, reg_features])},
            index=valid_X.index[valid_X.female==0]),
        pd.DataFrame(
            {'pred':reg_female.predict(
                valid_X.loc[valid_X.female==1, reg_features])},
            index=valid_X.index[valid_X.female==1]),
    ]),
    on='id',how='left'
)
pred_valid_sep['Calories_pred'] = pred_valid_sep.Duration*pred_valid_sep.pred
pred_valid_sep.loc[pred_valid_sep.Calories_pred<0, 'Calories_pred'] = 1

calculate_RMSLE(pred_train_sep.Calories_pred, 
               pred_valid_sep.Calories_pred)


# Submission
pred_test_sep = pd.DataFrame(
    {'Duration':test_X.Duration},
    index=test_X.index).merge(
    pd.concat([
        pd.DataFrame(
            {'pred':reg_male.predict(
                test_X.loc[test_X.female==0, reg_features])},
            index=test_X.index[test_X.female==0]),
        pd.DataFrame(
            {'pred':reg_female.predict(
                test_X.loc[test_X.female==1, reg_features])},
            index=test_X.index[test_X.female==1]),
    ]),
    on='id',how='left'
)

pred_test_sep['Calories_pred'] = pred_test_sep.Duration*pred_test_sep.pred 

pred_test_sep.loc[pred_test_sep.Calories_pred<0,'Calories_pred'] = 1

submission = pd.DataFrame({'id':test_df.index,
                           'Calories': pred_test_sep.Calories_pred})

submission.to_csv('submission.csv', index=False)


#resid_plot_feature_num(pred_valid_y=pred_valid_sep.Calories_pred)

