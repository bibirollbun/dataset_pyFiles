import pandas as pd
import numpy as np
import seaborn as sns
import shap
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

sum_cmap = sns.light_palette("#BFC1C9", as_cmap=True)


# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Verify shapes
print("Train:",train_data.shape,"   Test:",test_data.shape)


from IPython.display import display_html

df1_styler = train_data.head(3).style.set_table_attributes("style='display:inline'").set_caption('Head Train Data').background_gradient(sum_cmap)
df2_styler = test_data.head(3).style.set_table_attributes("style='display:inline'").set_caption('Head Test Data').background_gradient(sum_cmap)
display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)


def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values 
    summ['%missing'] = df.isnull().sum().values / len(df)* 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ["MostFreqValue"] = df.mode().iloc[:1].T.squeeze()
    summ["MostFreqValueCount"] = df.apply(lambda col: col.value_counts().iloc[0])
    most_freq_count = df.apply(lambda col: col.value_counts().iloc[0])
    summ["MostFreqValueCountRatio"] = most_freq_count / len(df)
        
    return summ


summary(train_data).style.background_gradient(sum_cmap).format({"MostFreqValueCountRatio": "{:.1%}"}, precision=1)


summary(test_data).style.background_gradient(sum_cmap).format({"MostFreqValueCountRatio": "{:.1%}"}, precision=1)


for col in train_data.columns:
    if col in ['Genre', 'Publication_Day', 'Publication_Time', 'Number_of_Ads']:
        print(train_data[col].value_counts())
        print("-" * 50)


train_data.loc[train_data['Number_of_Ads'] > 3, 'Number_of_Ads'] = 3
test_data.loc[test_data['Number_of_Ads'] > 3, 'Number_of_Ads'] = 3


train_data[['Episode_Length_minutes']].sort_values(by='Episode_Length_minutes', ascending=False).head(10)


test_data[['Episode_Length_minutes']].sort_values(by='Episode_Length_minutes', ascending=False).head(10)


train_data.loc[train_data['Episode_Length_minutes'] > 120.99, 'Episode_Length_minutes'] = 120.99
test_data.loc[test_data['Episode_Length_minutes'] > 120.99, 'Episode_Length_minutes'] = 120.99


numerical_variables = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
categorical_variables = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

target_variable = 'Listening_Time_minutes' 


train_data['Episode_Length_minutes']=train_data['Episode_Length_minutes'].fillna(60)
test_data['Episode_Length_minutes']=test_data['Episode_Length_minutes'].fillna(60)


train_data['Guest_Popularity_percentage'] = train_data.groupby('Episode_Length_minutes')['Guest_Popularity_percentage']\
                                       .transform(lambda x: x.fillna(x.median()))
test_data['Guest_Popularity_percentage'] = test_data.groupby('Episode_Length_minutes')['Guest_Popularity_percentage']\
                                       .transform(lambda x: x.fillna(x.median()))


def fill_df(df):

    # Fill in the numerical gaps with the median
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Filling in categorical gaps with the most frequent value (mode)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


train_data = fill_df(train_data)
test_data = fill_df(test_data)


def plot_categorical_distributions(train_data, test_data, variables, custom_palette=['#E19757', '#99348A']):
    fig = plt.figure(constrained_layout=True, figsize=(20, 5 * len(variables)))
    grid = gridspec.GridSpec(ncols=1, nrows=len(variables), figure=fig)

    for idx, var in enumerate(variables):
        ax = fig.add_subplot(grid[idx])
        ax.set_title(f'{var} Distribution')

        sns.histplot(train_data[var], kde=True, ax=ax, label='Train', color=custom_palette[0])
        sns.histplot(test_data[var], kde=True, ax=ax, label='Test', color=custom_palette[1])

        ax.legend()

        labels = [label.get_text() for label in ax.get_xticklabels()]
        short_labels = [label[:10] for label in labels]
        ax.set_xticklabels(short_labels, rotation=45, ha="right")
        ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()], rotation=45, va="top")

    plt.show()

def plot_numerical_distributions(train_data, test_data, variables, bins=30, custom_palette=['#E19757', '#99348A']):
    fig, axes = plt.subplots(len(variables), 2, figsize=(20, 5 * len(variables)))
    
    if len(variables) == 1:
        axes = [axes]
    
    for idx, var in enumerate(variables):
        # Гистограмма
        sns.histplot(train_data[var], bins=bins, kde=True, ax=axes[idx][0], label='Train', color=custom_palette[0])
        sns.histplot(test_data[var], bins=bins, kde=True, ax=axes[idx][0], label='Test', color=custom_palette[1])
        axes[idx][0].set_title(f'{var} Histogram')
        axes[idx][0].legend()
        
        # Боксплот
        sns.boxplot(data=[train_data[var], test_data[var]], ax=axes[idx][1], palette=custom_palette)
        axes[idx][1].set_xticklabels(['Train', 'Test'])
        axes[idx][1].set_title(f'{var} Boxplot')
    
    plt.tight_layout()
    plt.show()


categorical_variables = [col for col in train_data.columns if col in categorical_variables]
plot_categorical_distributions(train_data, test_data, categorical_variables)


numerical_variables = [col for col in train_data.columns if col in numerical_variables]
plot_numerical_distributions(train_data, test_data, numerical_variables)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(train_data['Listening_Time_minutes'], kde=True, bins=30, color='#E19757')
plt.title(f"Histogram of Listening_Time_minutes")
plt.xlabel('Listening_Time_minutes')
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(x=train_data['Listening_Time_minutes'], color='#E19757')
plt.title(f"Box Plot of Listening_Time_minutes")

plt.tight_layout()
plt.show()


def plot_target_relationships(train_data, target_variable, numerical_variables, categorical_variables):

    total_numerical = len(numerical_variables)
    total_categorical = len(categorical_variables)
    
    total_plots = total_numerical + total_categorical
    num_columns = 2
    num_rows = (total_plots + num_columns - 1) // num_columns
    
    plt.figure(figsize=(16, num_rows * 6))
    plasma_colors = sns.color_palette("plasma", n_colors=5)

    for i, var in enumerate(numerical_variables, 1):
        plt.subplot(num_rows, num_columns, i)
        sns.regplot(x=var, y=target_variable, data=train_data, 
                    scatter_kws={'color': plasma_colors[0]},
                    line_kws={'color': plasma_colors[1]})
        plt.title(f'Relationship between {var} and {target_variable}')
        plt.xlabel(var)
        plt.ylabel(target_variable)

    for i, var in enumerate(categorical_variables, total_numerical + 1):
        plt.subplot(num_rows, num_columns, i)
        sns.violinplot(x=var, y=target_variable, data=train_data, palette='plasma')
        plt.title(f'Relationship between {var} and {target_variable}')
        plt.xlabel(var)
        plt.ylabel(target_variable)

    plt.tight_layout()
    plt.show()

def plot_target_relationships2(train_data, target_variable, categorical_variables):
    total_plots = len(categorical_variables)
    num_columns = 1
    num_rows = total_plots
    plt.figure(figsize=(16, num_rows * 6))

    for i, var in enumerate(categorical_variables, 1):
        plt.subplot(num_rows, num_columns, i)
        
        # Boxplot (график размаха)
        sns.boxplot(x=var, y=target_variable, data=train_data, palette='plasma', showcaps=False, whiskerprops={'linewidth': 0}, medianprops={'color': 'black'})
        
        # Stripplot (точки поверх boxplot)
        sns.stripplot(x=var, y=target_variable, data=train_data, color='black', alpha=0.3, jitter=True, size=1)

        plt.title(f'Relationship between {var} and {target_variable}')
        plt.xlabel(var)
        plt.ylabel(target_variable)
        plt.xticks(rotation=65, ha="right")

    plt.tight_layout()
    plt.show()


categorical_variables_gr1 = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
plot_target_relationships(train_data, target_variable, numerical_variables, categorical_variables_gr1)


categorical_variables_gr2 = ['Podcast_Name', 'Episode_Title']
plot_target_relationships2(train_data, target_variable, categorical_variables_gr2)


n = 50
top_podcast = train_data['Podcast_Name'].value_counts().head(n)

plt.figure(figsize=(10, 10))
sns.barplot(x=top_podcast.values, y=top_podcast.index, palette='plasma')
plt.title(f'Top {n} Podcast_Name by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Podcast Name')
plt.tight_layout()
plt.show()


n = 30
top_episode = train_data['Episode_Title'].value_counts().head(n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_episode.values, y=top_episode.index, palette='plasma')
plt.title(f'Top {n} Episode_Title by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Episode Title')
plt.tight_layout()
plt.show()


def segment_percentage(percentage):
    if percentage <= 20:
        return '0-20'
    elif percentage <= 40:
        return '20-40'
    elif percentage <= 60:
        return '40-60'
    elif percentage <= 80:
        return '60-80'
    elif percentage <= 100:
        return '80-100'
    else:
        return '100+'


def feature_engineering(df):
    
    columns = ['Podcast_Name', 'Episode_Title']
    for col in columns:
        counts = df[col].value_counts(normalize=True)
        df[f'{col}_popularity'] = df[col].map(counts)
        
    df['Host_Popularity_percentage_segment'] = df['Host_Popularity_percentage'].apply(segment_percentage)
    df['Guest_Popularity_percentage_segment'] = df['Guest_Popularity_percentage'].apply(segment_percentage)
    
    df['Host_vs_Guest_Popularity'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1e-5)
    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Avg_Episode_Length_By_Genre'] = df.groupby('Genre')['Episode_Length_minutes'].transform('mean')
    
    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-5)
    df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype(int)
    df['Ads_Intensity'] = pd.cut(df['Number_of_Ads'], bins=[-1, 0, 2, 3], labels=['No Ads', 'Few Ads', 'Many Ads'])
    
    df['Has_Sport'] = df['Podcast_Name'].str.contains(r'\b(Sports|Sport|Arena)\b', case=False, na=False).astype(int)
    df['Has_Music'] = df['Podcast_Name'].str.contains(r'\b(Music|Melody|Sound|Tune)\b', case=False, na=False).astype(int)
    df['Has_Comedy'] = df['Podcast_Name'].str.contains(r'\b(Joke|Funny|Comedy|Laugh|Humor)\b', case=False, na=False).astype(int)
    df['Has_Crime'] = df['Podcast_Name'].str.contains(r'\b(Mystery|Criminal|Crime|Detective)\b', case=False, na=False).astype(int)
    df['Has_Education'] = df['Podcast_Name'].str.contains(r'\b(Study|Educational|Learning|Brain)\b', case=False, na=False).astype(int)
    df['Has_Finance'] = df['Podcast_Name'].str.contains(r'\b(Business|Market|Money|Finance)\b', case=False, na=False).astype(int)
    
    # by yunsuxiaozi (https://www.kaggle.com/code/yunsuxiaozi/pss5e4-xgb-baseline)
    df['Episode_Title_num']=df['Episode_Title'].apply(lambda x:int(x[len('Episode '):]))
    df['sin_Episode_Title_num']=np.sin(2*np.pi*df['Episode_Title_num']/12)
    df['cos_Episode_Title_num']=np.cos(2*np.pi*df['Episode_Title_num']/12)
    df['sin_Episode_Length_minutes']=np.sin(2*np.pi*df['Episode_Length_minutes']/60)
    df['cos_Episode_Length_minutes']=np.cos(2*np.pi*df['Episode_Length_minutes']/60)
    
    return df


train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)


cat_features = ['Podcast_Name', 
                'Episode_Title', 
                'Genre', 
                'Publication_Day', 
                'Publication_Time', 
                'Episode_Sentiment', 
                'Host_Popularity_percentage_segment', 
                'Guest_Popularity_percentage_segment',
                'Ads_Intensity'
               ]
col_features = ['Episode_Length_minutes', 
                'Host_Popularity_percentage', 
                'Guest_Popularity_percentage', 
                'Number_of_Ads', 
                'Podcast_Name_popularity', 
                'Episode_Title_popularity', 
                'Host_vs_Guest_Popularity', 
                'Ads_per_Minute', 
                'Total_Popularity',
                'Avg_Episode_Length_By_Genre',
                'Has_Ads',
                'Has_Sport',
                'Has_Music',
                'Has_Comedy',
                'Has_Crime',
                'Has_Education',
                'Has_Finance',
                'sin_Episode_Title_num', 
                'cos_Episode_Title_num', 
                'sin_Episode_Length_minutes', 
                'cos_Episode_Length_minutes'
               ]

features = col_features + cat_features


from sklearn.preprocessing import LabelEncoder

def encode_columns(df, label_encoders, features):
    for column, le in label_encoders.items():
        if column in features:
            df[column] = le.transform(df[column])
    return df

label_encoders = {}
for column in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train_data[column], test_data[column]])
    le.fit(combined)
    label_encoders[column] = le


train_data = encode_columns(train_data, label_encoders, features)
test_data = encode_columns(test_data, label_encoders, features)


plt.figure(figsize=(18, 14))
corr_train = train_data.corr(method='pearson')
mask_train = np.triu(np.ones_like(corr_train))

sns.heatmap(
    corr_train,
    annot=True,
    fmt='.2f',
    mask=mask_train,
    cmap='Spectral',
    cbar=True,
    linewidths=1,
    annot_kws={"size": 8}
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Correlation matrix of features', fontsize=14, pad=20)
plt.tight_layout()
plt.show()


X = train_data[features]
y = train_data['Listening_Time_minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight= 4,
    colsample_bytree=0.66,
    subsample=0.9,
    gamma=1.6,
    reg_alpha=5.5,
    reg_lambda=8,
    eval_metric="rmse",
    early_stopping_rounds=100,
    random_state=1212,
    tree_method="hist",
    enable_categorical=True,
    verbosity=0
)

# Assuming X and y are already defined
kf = KFold(n_splits=10, shuffle=True, random_state=1212)
train_scores = []
val_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    # Splitting data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Fitting the model
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              verbose=False)

    # Predicting on the training and validation sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # RMSE scores
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    print(f"Fold {fold}: Train RMSE = {train_rmse:.4f}, Validation RMSE = {val_rmse:.4f}")

# Calculating mean RMSE scores
mean_train_rmse = np.mean(train_scores)
mean_val_rmse = np.mean(val_scores)

print(f"\nMean Train RMSE: {mean_train_rmse:.4f}")
print(f"Mean Validation RMSE: {mean_val_rmse:.4f}")


feature_importances = model.feature_importances_
feature_names = X.columns

importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
})

importances_df = importances_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importances_df['feature'], importances_df['importance'])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance from LightGBM")
plt.gca().invert_yaxis()
plt.show()


pred = model.predict(test_data[features])

submission = pd.DataFrame({
    'id': sample_data.id,
    'price': pred
})

submission.to_csv('submission.csv', index=False)


submission.head()

