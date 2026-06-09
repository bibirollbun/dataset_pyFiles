import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(f'train shape {train_df.shape}')
print(f'test shape {test_df.shape}')
train_df.info()


train_df.groupby('Personality').size()/len(train_df.id)


train_df['Introvert'] = 0
train_df.loc[train_df['Personality']=='Introvert','Introvert'] = 1


# One-hot Encoding
# - Stage fear
train_df['Stage_fear_Yes'] = 0
train_df.loc[train_df.Stage_fear=='Yes',['Stage_fear_Yes']] = 1
train_df['Stage_fear_No'] = 0
train_df.loc[train_df.Stage_fear=='No',['Stage_fear_No']] = 1
train_df['Stage_fear_Nan'] = 0
train_df.loc[train_df.Stage_fear.isna(),['Stage_fear_Nan']] = 1

test_df['Stage_fear_Yes'] = 0
test_df.loc[test_df.Stage_fear=='Yes',['Stage_fear_Yes']] = 1
test_df['Stage_fear_No'] = 0
test_df.loc[test_df.Stage_fear=='No',['Stage_fear_No']] = 1
test_df['Stage_fear_Nan'] = 0
test_df.loc[test_df.Stage_fear.isna(),['Stage_fear_Nan']] = 1

# - Drained after socializing
train_df['Drained_Yes'] = 0
train_df.loc[train_df.Drained_after_socializing=='Yes',['Drained_Yes']] = 1
train_df['Drained_No'] = 0
train_df.loc[train_df.Drained_after_socializing=='No',['Drained_No']] = 1
train_df['Drained_Nan'] = 0
train_df.loc[train_df.Drained_after_socializing.isna(),['Drained_Nan']] = 1

test_df['Drained_Yes'] = 0
test_df.loc[test_df.Drained_after_socializing=='Yes',['Drained_Yes']] = 1
test_df['Drained_No'] = 0
test_df.loc[test_df.Drained_after_socializing=='No',['Drained_No']] = 1
test_df['Drained_Nan'] = 0
test_df.loc[test_df.Drained_after_socializing.isna(),['Drained_Nan']] = 1


# Explore categorical features
cat_features = [
    ['Stage_fear_Yes','Stage_fear_No','Stage_fear_Nan'],
    ['Drained_Yes','Drained_No','Drained_Nan']
]

n_features = len(cat_features)
fig, axes = plt.subplots(nrows = n_features,
                        ncols = 2,
                        figsize=(10, 3*n_features))

for i in range(n_features):
    feature = cat_features[i]
    # - compare distribution between train and test
    train_df_plot = pd.DataFrame(train_df.loc[:,feature].mean())
    train_df_plot['source'] = 'train'
    test_df_plot = pd.DataFrame(test_df.loc[:,feature].mean())
    test_df_plot['source'] = 'test'
    data_all = pd.concat([train_df_plot, test_df_plot]).reset_index()
    data_all.columns=['feature','percentage','source']
    sns.barplot(data_all, 
                x='feature', y='percentage', hue='source',
                ax=axes[i][0]
               )
    axes[i][0].set_title('Train vs Test')

    # - investigate association between introvert and extrovert
    train_dist = train_df.loc[:,['Personality']+feature].groupby('Personality').mean().reset_index()
    train_dist_long = train_dist.melt(id_vars='Personality', 
                                      value_vars=feature)
    sns.barplot(data=train_dist_long, 
                x='variable',y='value', hue='Personality',
               ax=axes[i][1])
    axes[i][1].set_title('Introverts vs Extroverts')
    


num_features = ['Time_spent_Alone','Social_event_attendance',
                'Going_outside','Friends_circle_size','Post_frequency']


n_features = len(num_features)
fig, axes = plt.subplots(nrows = n_features,
                         ncols = 2,
                         figsize=(10, 3*n_features))

train_df_plot = pd.DataFrame(train_df.loc[:,num_features])
train_df_plot['source'] = 'train'
test_df_plot = pd.DataFrame(test_df.loc[:,num_features])
test_df_plot['source'] = 'test'
data_all = pd.concat([train_df_plot, test_df_plot])

for i in range(n_features):
    feature = num_features[i]
    # - compare distribution between train and test
    sns.kdeplot(data_all.loc[~data_all[feature].isna()], 
                x=feature, hue='source',
                fill=True,
                ax=axes[i][0]
               )
    axes[i][0].set_title('Train vs Test')

    # - investigate association between introvert and extrovert
    sns.kdeplot(data=train_df, 
                x=feature,hue='Personality',
                fill=True,
               ax=axes[i][1])
    axes[i][1].set_title('Introverts vs Extroverts')
    


train_df.loc[:,num_features].isna().sum()


train_df.loc[:,num_features].describe()


# Handle missing values
for feature in num_features:
    # - create flag for missing values
    train_df["missing_"+feature] = 0
    train_df.loc[train_df[feature].isna(),'missing_'+feature] = 1

    test_df["missing_"+feature] = 0
    test_df.loc[test_df[feature].isna(),'missing_'+feature] = 1
    
    # - impute with mean
    train_df.loc[train_df[feature].isna(),feature] = train_df[feature].mean()
    test_df.loc[test_df[feature].isna(),feature] = test_df[feature].mean()


# Train Test Split
from sklearn.model_selection import train_test_split

X = train_df.copy()
y = X.pop('Introvert')
test_X = test_df.copy()

train_X,valid_X, train_y,valid_y = train_test_split(X, y, random_state = 1)


# accuracy score
from sklearn.metrics import accuracy_score

def model_eval(pred_train, pred_valid, train_y = train_y, valid_y = valid_y):
    acc_train = accuracy_score(pred_train, train_y)
    acc_valid = accuracy_score(pred_valid, valid_y)
    print(f'accuracy for train is {acc_train}')
    print(f'accuracy for valid is {acc_valid}')


from sklearn.linear_model import LogisticRegression

lr_features = ['Time_spent_Alone','missing_Time_spent_Alone',
               'Social_event_attendance','missing_Social_event_attendance',
               'Going_outside','missing_Going_outside',
               'Friends_circle_size','missing_Friends_circle_size',
               'Post_frequency','missing_Post_frequency',
               'Stage_fear_Yes','Stage_fear_No','Stage_fear_Nan',
               'Drained_Yes','Drained_No','Drained_Nan']

lr_model = LogisticRegression()
lr_model.fit(train_X.loc[:,lr_features], train_y)

train_pred = lr_model.predict(train_X.loc[:, lr_features])
valid_pred = lr_model.predict(valid_X.loc[:, lr_features])


model_eval(train_pred, valid_pred)


# submission
submission = pd.DataFrame({'id':test_X.id,
                          'pred':lr_model.predict(test_X.loc[:,lr_features])})

submission['Personality'] = 'Extrovert'
submission.loc[submission.pred==1,'Personality'] = 'Introvert'
submission = submission.loc[:,['id','Personality']]
submission.to_csv('submission.csv', index=False)

