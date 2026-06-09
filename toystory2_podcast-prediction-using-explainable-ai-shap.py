import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms #missing values graph
import shap as sp

import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


#model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

#SFS
from sklearn.feature_selection import SequentialFeatureSelector


#evaluate
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")



#!pip install ace-tools


#read the data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
samp_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train_df.head()


test_df.head()


# Convert all infinite values to NaN in one shot
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df .replace([np.inf, -np.inf], np.nan, inplace=True)



train_df.info()


ms.heatmap(train_df, cmap = 'viridis')
plt.title("Customized Missing Data Heatmap")
plt.show()


ms.heatmap(test_df, cmap = 'viridis')
plt.title("Customized Missing Data Heatmap")
plt.show()


train_df.isnull().sum()


test_df.isnull().sum()


#drop id columns from train and test data
train_df.drop(columns = 'id', axis = 1, inplace = True)
test_df.drop(columns = 'id', axis = 1, inplace = True)



#Fixing the missing values(training data)
# numerical missing val
train_df['Episode_Length_minutes'] = train_df.Episode_Length_minutes.fillna(train_df['Episode_Length_minutes'].median())     
train_df.Guest_Popularity_percentage = train_df.Guest_Popularity_percentage.fillna(train_df.Guest_Popularity_percentage.median())


#categorical missing val
mode_val = train_df['Number_of_Ads'].mode()[0]
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(mode_val)



#fixing missing values (testing data)
# numerical missing val
test_df['Episode_Length_minutes'] = test_df.Episode_Length_minutes.fillna(test_df['Episode_Length_minutes'].median())     
test_df.Guest_Popularity_percentage = test_df.Guest_Popularity_percentage.fillna(test_df.Guest_Popularity_percentage.median())


train_df.info()


test_df.info()


train_df['Episode_Num'] = train_df['Episode_Title']\
                            .str[8:].astype('int')



# Compute Spearmen correlation
eps = train_df['Episode_Num']
listen = train_df['Listening_Time_minutes']


ρ = eps.corr(listen, method='spearman')
print("Spearman ρ =", round(ρ, 3))


plt.figure(figsize=(10,6))
sns.scatterplot(x=eps, y=listen, alpha=0.3)
# draw a LOWESS (locally weighted) smoothing curve
sns.regplot(x=eps, y=listen, scatter=False, lowess=True, color='red')
plt.xlabel("Episode Number")
plt.ylabel("Listening Time (minutes)")
plt.title("Episode Number vs Listening Time (with LOWESS)")
plt.show()



# drop from both train and test 
train_df = train_df.drop(columns=['Episode_Title', 'Episode_Num'])
test_df  = test_df .drop(columns=['Episode_Title'])



# mapping dictionaries on train_df
genre_mean_map   = train_df.groupby('Genre')['Listening_Time_minutes'] \
                          .mean().to_dict()
count_by_day_map = train_df.groupby('Publication_Day')['Episode_Sentiment'] \
                          .count().to_dict()

# 1b) Applying to train & test
for df in (train_df, test_df):
    df['genre_mean_listen'] = df['Genre'].map(genre_mean_map)
    df['count_by_day']     = df['Publication_Day'].map(count_by_day_map)



test_df[['genre_mean_listen','count_by_day']].isnull().sum()



def feature_eng(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')
    
    return df


train_df = feature_eng(train_df)
test_df = feature_eng(test_df)


def add_engineered_features(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """
    Adds engineered numerical and group-based features to a DataFrame:
      - length_per_ad: Episode_Length_minutes / (Number_of_Ads + eps)
      - host_guest_prod: Host_Popularity_percentage * Guest_Popularity_percentage
      - host_guest_diff: Host_Popularity_percentage - Guest_Popularity_percentage
      - host_guest_ratio: Host_Popularity_percentage / (Guest_Popularity_percentage + eps)
      - genre_mean_listen: mean Listening_Time_minutes per Genre
      - count_by_day: count of episodes per Publication_Day
    
    :param df: pandas DataFrame with required source columns
    :param eps: small constant to avoid division by zero
    :return: new DataFrame with added columns
    """
    df = df.copy()
    # ratio: minutes per ad
    df['length_per_ad'] = df['Episode_Length_minutes'] / (df['Number_of_Ads'] + eps)
    # product: host × guest popularity
    df['host_guest_prod'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    # difference & ratio
    df['host_guest_diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['host_guest_ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + eps)

   

    return df






train_df = add_engineered_features(train_df)
test_df  = add_engineered_features(test_df)


train_df.info()


#Clubbing columns under different categories(for future use)

#ordinal columns
ord_col = train_df.columns[[4,5]].tolist()

#non-ordinal columns
non_ord_col = train_df.columns[[0,2,8]].tolist()

#numeric columns
numeric_col = train_df.columns[[1,3,6,7]].tolist()

#engineered columns
eng_col = train_df.columns[[10,11,12,13,14,15]].tolist()



def numerical_col_analysis(data, numerical_col, target):
    data_clean = data.copy()
    data_clean[numerical_col] = (
        data_clean[numerical_col]
        .replace([np.inf, -np.inf], np.nan)
    )
    
    """
    Comprehensive EDA for numeric columns:
     - histograms + boxplots
     - skewness & kurtosis
     - IQR outlier detection
     - scatter vs target
     - correlation matrix


    :param data: Pandas DataFrame containing the dataset
    :param numerical_col: List of numerical column names
    :param target: Name of the target variable
    """
    #1) univariate distributions & outliers
    
    for feature in numerical_col:
        
        #compute skewness and kurtosis
        skew = data_clean[feature].skew()
        kurt = data_clean[feature].kurtosis()

        #detect the outliers
        Q1,Q3 = np.percentile(data_clean[feature],[25,75])
        IQR = Q3 - Q1
        lower,upper = Q1 - 1.5 * IQR,Q3 + 1.5 * IQR
        n_outlier = ((data_clean[feature]<lower)|(data_clean[feature]>upper)).sum()

        print(f"\n=== {feature} ===")
        print(f" Skewness: {skew:.2f}, Kurtosis: {kurt:.2f}")
        print(f" IQR bounds: ({lower:.2f}, {upper:.2f}), Outliers: {n_outlier}")

         # visualization
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        sns.histplot(data_clean[feature], kde=True, ax=axes[0])
        axes[0].set(title=f"{feature} histogram")
        sns.boxplot(x=data_clean[feature], ax=axes[1])
        axes[1].set(title=f"{feature} boxplot")
        plt.tight_layout()
        plt.show()

    # 2) scatter vs target
    for feature in numerical_col:
         #compute skewness and kurtosis
        skew = data_clean[feature].skew()
        kurt = data_clean[feature].kurtosis()
        print(f"\n=== {feature} ===")
        print(f" Skewness: {skew:.2f}, Kurtosis: {kurt:.2f}")
        if feature == target: 
            continue
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=data_clean[feature], y=data_clean[target], alpha=0.5)
        plt.title(f"{feature} vs {target}")
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

    # 3) correlation matrix
    cols = numerical_col + [target]
    corr = data_clean[cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

        

      


numerical_col_analysis(train_df, numeric_col, "Listening_Time_minutes")


#X_train,Y_train
feature_col = ord_col+non_ord_col+numeric_col+eng_col
feature_col


# Slice X and y from train_df
X_train = train_df[feature_col]
y_train = train_df['Listening_Time_minutes']
X_test = test_df.copy()


param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth':    [3, 5, 7],
    'model__learning_rate':[0.01, 0.1],
    'model__subsample':    [0.8, 1.0]
}


#K-Fold cross-validation 
cv = KFold(n_splits=5, shuffle=True, random_state=42)



pipeline = Pipeline([
    ('feature_eng', FunctionTransformer(feature_eng, validate=False)),
    ('engineer',    FunctionTransformer(add_engineered_features, validate=False)),
    ('imputer',     SimpleImputer(strategy='median')),
    ('model',       XGBRegressor(
                        random_state=42,
                        tree_method='hist'   # faster
                   ))
])


# neg_mse = cross_val_score(
#     pipeline,
#     X_train,
#     y_train,
#     cv=cv,
#     scoring='neg_mean_squared_error'
# )


#GridSearchCV

grid_search = GridSearchCV(
    estimator=pipeline,                        # pipeline 
    param_grid=param_grid,                     # the hyperparameter grid
    scoring='neg_root_mean_squared_error',     # use RMSE as the metric
    cv=cv,                                     # 5-fold CV
    n_jobs=-1,                                 # parallelize across all cores
    verbose=2                                  # show progress
)


# rmse_scores = np.sqrt(-neg_mse)
# print("CV RMSE per fold:", np.round(rmse_scores, 2))
# print("Mean CV RMSE:      ", np.round(rmse_scores.mean(), 2))
# print("Std  CV RMSE:      ", np.round(rmse_scores.std(),  2))


#fit model
grid_search.fit(X_train, y_train)


print("Best params:   ", grid_search.best_params_)
print("Best CV RMSE:  ", -grid_search.best_score_)



#best estimator in the pipeline 
best_pipe = grid_search.best_estimator_
preproc    = best_pipe[:-1]


def preprocess_df(pipe, X):
    X_m = pipe.named_steps['feature_eng'].transform(X)
    X_e = pipe.named_steps['engineer']   .transform(X_m)
    
    # drop into NumPy for the imputer to avoid mismatch of columns
    imp = pipe.named_steps['imputer']
    X_i = imp.transform(X_e.values)
    
    # wrap back into DataFrame using the engineer’s column names
    cols = X_e.columns.tolist()
    return pd.DataFrame(X_i, columns=cols, index=X.index)


# Preprocess train & test
df_imp_train = preprocess_df(best_pipe, X_train)
df_imp_test  = preprocess_df(best_pipe, X_test)


df_imp_train.head()


df_imp_test.head()


y_train.head()


samp_df.head()


# 1) List out the columns from each
train_cols = df_imp_train.columns.tolist()
test_cols  = df_imp_test .columns.tolist()

print("Train cols (n={}):".format(len(train_cols)), train_cols)
print("\nTest  cols (n={}):".format(len(test_cols)),  test_cols)

# 2) Show exactly which names differ
train_set = set(train_cols)
test_set  = set(test_cols)

print("\nIn train not in test:", train_set - test_set)
print("In test  not in train:", test_set  - train_set)


# align test columns to train order
df_imp_test = df_imp_test[df_imp_train.columns]

# now the assertion will pass
assert list(df_imp_train.columns) == list(df_imp_test.columns)
print("Columns are now aligned.")



df_imp_train.shape


xgb_model = best_pipe.named_steps['model']


explainer = sp.TreeExplainer(xgb_model, df_imp_train)



X_sample = df_imp_train.sample(n=1000, random_state=42)



shap_vals_sample = explainer.shap_values(X_sample)
# b) Global importance plots
sp.summary_plot(shap_vals_sample, X_sample, plot_type="bar", max_display=15)
sp.summary_plot(shap_vals_sample, X_sample, plot_type="dot", max_display=15)



# Build a ranking DataFrame from the sample
imp_df = (
    pd.DataFrame({
        'feature':       X_sample.columns,
        'mean_abs_shap': np.abs(shap_vals_sample).mean(axis=0)
    })
    .sort_values('mean_abs_shap', ascending=False)
    .reset_index(drop=True)
)

# Display the top 7 features
top7 = imp_df.head(7)['feature'].tolist()
print("Top 7 features by sampled SHAP:", top7)


print(imp_df.head(15))


preds_full = xgb_model.predict(df_imp_test)


preds_full


# fill preds_full array
samp_df['Listening_Time_minutes'] = preds_full




# save for upload
samp_df.to_csv('my_submission.csv', index=False)
print("Wrote my_submission.csv")







