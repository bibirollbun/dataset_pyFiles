from scipy.stats import chi2_contingency
import numpy as np 
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

warnings.filterwarnings("ignore",  category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning,)
warnings.filterwarnings("ignore", category=UserWarning)



train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


display(train_df.head(4))
display(test_df.head(4))


display(train_df.info())
print('')
display(test_df.info())


display("Summary for TRAIN data:",train_df.describe())
display("Summary for TEST data:",test_df.describe())


display("TRAIN data", train_df.nunique())
print('\n')
display('TEST data',test_df.nunique())


display("Duplicates in TRAIN data:", train_df.duplicated().sum())
display("Duplicates in TEST data:", test_df.duplicated().sum())


print("Missing values in TRAIN data:\n",train_df.isna().mean().apply(lambda x: f"{x:.2%}"))
print("\nMissing values  in TEST data:\n",test_df.isna().mean().apply(lambda x: f"{x:.2%}"))


missing_data= pd.DataFrame({
    'train': train_df.isna().mean()*100,
    'test': test_df.isna().mean()*100
})
missing_data.plot(kind='bar', title='Missing values in % for train and test')


for col in test_df.columns:
    if col != 'id' and test_df[col].dtype in[np.int64,np.float64]:
        sns.kdeplot(test_df[col], label='test', fill=True)
        sns.kdeplot(train_df[col], label='train', fill=True)
        plt.legend()
        plt.show()


sns.heatmap(train_df[['Time_spent_Alone', 'Social_event_attendance','Going_outside', 'Friends_circle_size','Post_frequency']].corr(),
            annot = True, cmap='coolwarm')


sns.heatmap(test_df[['Time_spent_Alone', 'Social_event_attendance','Going_outside', 'Friends_circle_size','Post_frequency']].corr(), 
            annot = True, cmap='coolwarm')


sns.countplot(data=train_df, x='Personality')
round(train_df['Personality'].value_counts(normalize=True)*100,2)


columns = [col for col in test_df.columns if col != 'id']
for col in columns:
    train_df.groupby([col,'Personality']).size().unstack().plot(kind='bar', stacked=True, title=col)


sns.catplot(
    data=train_df[train_df['Personality'].isin(['Introvert', 'Extrovert'])],  
    x="Stage_fear",
    hue="Personality",
    col="Drained_after_socializing",
    kind="count",
    height=4,
    aspect=1
)
plt.suptitle("Rozkład StageFear wg osobowości", y=1.05)
plt.show()


pd.crosstab([train_df['Personality'],train_df["Stage_fear"]],train_df["Drained_after_socializing"]  )


cat_cols1 = ['Stage_fear','Drained_after_socializing']
all_train_df=train_df[['Stage_fear','Drained_after_socializing','Personality']].copy()
all_train_df[cat_cols1]=all_train_df[cat_cols1].fillna('Missing').astype(str)
display(pd.crosstab([all_train_df['Personality'],all_train_df["Stage_fear"]],all_train_df["Drained_after_socializing"]  ))


help_train_df = train_df[['Stage_fear','Drained_after_socializing','Personality']].copy()
help_train_df['Stage_fear'] = help_train_df['Stage_fear'].mask(help_train_df['Stage_fear'].isna() & help_train_df['Drained_after_socializing']
                                            .notna(), help_train_df['Drained_after_socializing'])
help_train_df['Drained_after_socializing'] = help_train_df['Drained_after_socializing'].mask(help_train_df['Drained_after_socializing']
                                            .isna() & help_train_df['Stage_fear'].notna(), help_train_df['Stage_fear'])
help_train_df[cat_cols1]=help_train_df[cat_cols1].fillna('Missing').astype(str)

display(pd.crosstab(help_train_df['Stage_fear'],help_train_df['Personality']))


train_introvert_df = train_df[train_df['Personality']=='Introvert']
train_extrovert_df = train_df[train_df['Personality']=='Extrovert']


introvert_no_no_df = train_df[(train_df["Stage_fear"]=='No') & (train_df["Drained_after_socializing"]=='No') & (train_df['Personality']=='Introvert')]
extrovert_yes_yes_df = train_df[(train_df["Stage_fear"]=='Yes') & (train_df["Drained_after_socializing"]=='Yes') & (train_df['Personality']=='Extrovert')]


for col in columns:
    if train_df[col].dtype in[np.int64,np.float64]:
        sns.kdeplot(introvert_no_no_df[col], label='introvert with no-no', fill=True)
        sns.kdeplot(train_introvert_df[col], label='train_df introvert', fill=True)
        sns.kdeplot(train_extrovert_df[col], label='train_df extrovert', fill=True)
        plt.legend()
        plt.show()


for col in columns:
    if train_df[col].dtype in[np.int64,np.float64]:
        sns.kdeplot(extrovert_yes_yes_df[col], label='extrovert with yes-yes', fill=True)
        sns.kdeplot(train_introvert_df[col], label='train_df introvert', fill=True)
        sns.kdeplot(train_extrovert_df[col], label='train_df extrovert', fill=True)
        plt.legend()
        plt.show()


for col in columns:
    if train_df[col].dtype in[np.int64,np.float64]:
        sns.kdeplot(extrovert_yes_yes_df[col], label='extrovert with yes-yes', fill=True)
        sns.kdeplot(train_introvert_df[col], label='train_df introvert', fill=True)
        sns.kdeplot(train_extrovert_df[col], label='train_df extrovert', fill=True)
        sns.kdeplot(introvert_no_no_df[col], label='introvert with no-no', fill=True)
        plt.legend()
        plt.show()


for col in test_df.columns:
    if col != 'id' and test_df[col].dtype in[np.int64,np.float64]:
        sns.kdeplot(introvert_no_no_df[col], label='introvert with no-no', fill=True)
        sns.kdeplot(extrovert_yes_yes_df[col], label='extrovert with yes-yes', fill=True)

        plt.legend()
        plt.show()


excluded_cols = ['id', 'Personality']
all_columns = train_df.columns
for col in all_columns:
    if col not in excluded_cols:
        train_df[col + '_MISS'] = train_df[col].notna().astype(int)


columns = ['Time_spent_Alone_MISS','Stage_fear_MISS', 'Social_event_attendance_MISS', 'Going_outside_MISS','Drained_after_socializing_MISS', 
           'Friends_circle_size_MISS','Post_frequency_MISS']
for col in columns:
    train_df.groupby([col,'Personality']).size().unstack().plot(kind='bar', stacked=True, title=col)
    result = pd.crosstab(train_df[col],train_df['Personality'], normalize='index')*100
    chi2, p, _, _ = chi2_contingency(result)
    print(f"Chi2 = {chi2:.3f}, p-value = {p:.4f} Column: {col} ")


train_df['not_MISS_total'] = train_df[columns].sum(axis=1)
train_df.groupby(['not_MISS_total','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Data without NaN values')
pd.crosstab(train_df['not_MISS_total'],train_df['Personality'], normalize='index')*100


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size','Post_frequency']    
cat_cols = ['Stage_fear', 'Drained_after_socializing']     

cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_df[cat_cols] = cat_encoder.fit_transform(train_df[cat_cols])
num_imputer = IterativeImputer(estimator=LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, verbosity=-1),
                               max_iter=10, random_state=42)

train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
cat_imputer = IterativeImputer(estimator=LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, class_weght='balanced', verbosity=-1),
                               max_iter=10, random_state=42)

train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])

columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size','Post_frequency']
train_df[columns]=train_df[columns].round().astype(int)



train_df['Time_Alone_dev_Outside'] = train_df['Time_spent_Alone'] / train_df['Going_outside']
train_df['Time_Alone_dev_Outside']=train_df['Time_Alone_dev_Outside'].round(2).astype(float)
train_df.groupby(['Time_Alone_dev_Outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


def Time_Alone_dev_Outside (x):
    try:
        x=float(x)
        if x <= 1:
            return 0
        elif x > 1 and x < 2:
            return 1
        elif x >= 2 and x < 100:
            return 2
        else:
            return 3
    except ValueError:
        return 3

train_df['Time_Alone_dev_Outside']=train_df['Time_Alone_dev_Outside'].apply(Time_Alone_dev_Outside).astype('Int64')
train_df.groupby(['Time_Alone_dev_Outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')
pd.crosstab(train_df["Time_Alone_dev_Outside"],train_df['Personality'],normalize='index')*100 


train_df['Social_dev_Post'] = train_df['Social_event_attendance'] / train_df['Post_frequency']
train_df['Social_dev_Post']=train_df['Social_dev_Post'].round(2).astype(float)
train_df.groupby(['Social_dev_Post','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


## Alternatywne grupowanie
train_df['Social_dev_Post'] = train_df['Social_event_attendance'] / train_df['Post_frequency']
train_df['Social_dev_Post']=train_df['Social_dev_Post'].round(2).astype(float)

def Social_dev_Post (x):
    try:
        x=float(x)
        if x == 0:
            return 0
        elif x > 0 and x < 0.33:
            return 1
        elif x == 0.33:
            return 2
        elif x > 0.33 and x < 0.5:
            return 1
        elif x == 0.5:
            return 4
        elif x > 0.5 and x < 0.67:
            return 1
        elif x == 0.67:
            return 4
        elif x > 0.67 and x < 1:
            return 1
        elif x == 1:
            return 4
        elif x > 1 and x < 1.5:
            return 1
        elif x == 1.5:
            return 4
        elif x > 1.5 and x < 2:
            return 1
        elif x == 2:
            return 4
        elif x > 2 and x < 3:
            return 1
        elif x == 3:
            return 2
        elif x > 3 and x < 100:
            return 1
        else:
            return 0
    except ValueError:
        return 0

train_df['Social_dev_Post']=train_df['Social_dev_Post'].apply(Social_dev_Post).astype('Int64')
train_df.groupby(['Social_dev_Post','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')
pd.crosstab(train_df["Social_dev_Post"],train_df['Personality'], normalize='index')*100


train_df['Outside_mult_Friends'] = train_df['Going_outside'] * train_df['Friends_circle_size']
train_df.groupby(['Outside_mult_Friends','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


def Outside_mult_Friends (x):
    try:
        x=float(x)
        if x <= 11:
            return 0
        elif x > 11 and x <= 15:
            return 1
        elif x > 15 and x < 400:
            return 2
        else:
            return 2
    except ValueError:
        return 2

train_df['Outside_mult_Friends']=train_df['Outside_mult_Friends'].apply(Outside_mult_Friends).astype('Int64')
train_df.groupby(['Outside_mult_Friends','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')
pd.crosstab(train_df["Outside_mult_Friends"],train_df['Personality'], normalize='index')*100


train_df['Going_sub_Post']=train_df['Going_outside'] - train_df['Post_frequency']
train_df.groupby(['Going_sub_Post','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


def Outside_mult_Friends (x):
    try:
        x=float(x)
        if x <= -6:
            return 0
        elif x == -5 or x== -4 or x==4:
            return 1
        elif x == -3 or x== 3:
            return 2
        elif x == -2 or x== 2:
            return 3
        elif x == -1 or x== 0 or x==1:
            return 4
        elif x >= 5:
            return 5
        else:
            return 6
    except ValueError:
        return 6

train_df['Going_sub_Post']=train_df['Going_sub_Post'].apply(Outside_mult_Friends).astype('Int64')
train_df.groupby(['Going_sub_Post','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')
pd.crosstab(train_df["Going_sub_Post"],train_df['Personality'], normalize='index')*100


train_df['subtraction']=train_df['Friends_circle_size'] - train_df['Post_frequency']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Going_outside'] - train_df['Post_frequency']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Going_outside'] - train_df['Friends_circle_size']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Social_event_attendance'] - train_df['Post_frequency']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Social_event_attendance'] - train_df['Friends_circle_size']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Social_event_attendance'] - train_df['Going_outside']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Time_spent_Alone'] - train_df['Post_frequency']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Time_spent_Alone'] - train_df['Friends_circle_size']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Time_spent_Alone'] - train_df['Going_outside']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['subtraction']=train_df['Time_spent_Alone'] - train_df['Social_event_attendance']
train_df.groupby(['subtraction','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Summary']=train_df['Friends_circle_size'] + train_df['Post_frequency']
train_df.groupby(['Summary','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Summary']=train_df['Going_outside'] + train_df['Friends_circle_size']
train_df.groupby(['Summary','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Summary']=train_df['Going_outside'] + train_df['Post_frequency']
train_df.groupby(['Summary','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Summary']=train_df['Social_event_attendance'] + train_df['Post_frequency']
train_df.groupby(['Summary','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Summary']=train_df['Social_event_attendance'] + train_df['Going_outside']
train_df.groupby(['Summary','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Time_alona_outside'] = train_df['Going_outside'] / train_df['Post_frequency']
train_df.groupby(['Time_alona_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Time_alona_outside'] = train_df['Social_event_attendance'] / train_df['Post_frequency']
train_df.groupby(['Time_alona_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Time_alona_outside'] = train_df['Time_spent_Alone'] / train_df['Post_frequency']
train_df.groupby(['Time_alona_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Time_alona_outside'] = train_df['Time_spent_Alone'] / train_df['Friends_circle_size']
train_df.groupby(['Time_alona_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Time_alona_outside'] = train_df['Time_spent_Alone'] / train_df['Social_event_attendance']
train_df.groupby(['Time_alona_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Time_alona_outside'] = train_df['Time_spent_Alone'] / train_df['Going_outside']
train_df.groupby(['Time_alona_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Time_alone_friends_circle'] = train_df['Time_spent_Alone'] / train_df['Friends_circle_size']
train_df.groupby(['Time_alone_friends_circle','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Social_event_outside'] = train_df['Going_outside'] * train_df['Social_event_attendance']
train_df.groupby(['Social_event_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Social_event_outside'] = train_df['Social_event_attendance'] * train_df['Friends_circle_size']
train_df.groupby(['Social_event_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Social_event_outside'] = train_df['Social_event_attendance'] * train_df['Post_frequency']
train_df.groupby(['Social_event_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Social_event_outside'] = train_df['Going_outside'] * train_df['Friends_circle_size']
train_df.groupby(['Social_event_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


train_df['Social_event_outside'] = train_df['Going_outside'] * train_df['Post_frequency']
train_df.groupby(['Social_event_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')

train_df['Social_event_outside'] = train_df['Friends_circle_size'] * train_df['Post_frequency']
train_df.groupby(['Social_event_outside','Personality']).size().unstack().plot(kind='bar', stacked=True, title='Summary')


