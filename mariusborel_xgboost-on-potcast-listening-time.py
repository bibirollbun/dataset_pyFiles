import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-dark')
# change default colormap
plt.rcParams['image.cmap'] = 'Dark2'
from termcolor import colored


# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import r2_score, mean_squared_error, roc_auc_score, roc_curve
from sklearn.compose import make_column_transformer
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold, TimeSeriesSplit as TSS)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance, cv

from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
import optuna

from yellowbrick.regressor import ResidualsPlot, PredictionError

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


train_0 = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_0 = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
ext_0 = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')

target = 'Listening_Time_minutes'

train_0.head(3)


for df_name, df in [('train', train_0), ('test', test_0), ('external', ext_0)]:
    nunb_of_duplicates = df.duplicated().sum()
    if nunb_of_duplicates != 0:
        print(f'{df_name} dataset has {nunb_of_duplicates} duplicates.')
    else:
        print(f'{df_name} dataset has no duplicates')


ext_raw_shape = ext_0.shape

ext_0 = ext_0.drop_duplicates()
print(f'The length of the original dataset has change from {ext_raw_shape[0]} to {ext_0.shape[0]}')


plt.figure(figsize=(15, 10))
# Create a GridSpec layout with 2 rows and 3 columns
gs = GridSpec(3, 3)

# First subplot
ax1 = plt.subplot(gs[0, 0])  # Top-left
sns.countplot(data=train_0, x='Publication_Time', ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

# Second subplot
ax2 = plt.subplot(gs[0, 1])  # Top-middle
sns.countplot(data=train_0, x='Publication_Day', ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set_ylabel('')

# Third subplot
ax3 = plt.subplot(gs[0, 2])  # Top-right
sns.countplot(data=train_0, x='Episode_Sentiment', ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
ax3.set_ylabel('')

# Fourth subplot
ax4 = plt.subplot(gs[1, 0])  # Bottom-left
sns.countplot(data=train_0, x='Genre', ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)

# Fifth subplot (spanning bottom-middle and bottom-right)
ax5 = plt.subplot(gs[1, 1:])  # Bottom spanning last two columns
sns.countplot(data=train_0, x='Podcast_Name', ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
ax5.set_ylabel('')

# sixth subplot (spanning bottom)
ax6 = plt.subplot(gs[2, :])  # Bottom spanning
sns.countplot(data=train_0, x='Episode_Title', ax=ax6)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90)
ax6.set_ylabel('')

plt.tight_layout()
plt.show()


from matplotlib.gridspec import GridSpec

plt.figure(figsize=(15, 9))
# Create a GridSpec layout with 2 rows and 3 columns
gs = GridSpec(3, 3)

# First subplot
ax1 = plt.subplot(gs[0, 0])  # Top-left
sns.boxplot(data=train_0, x='Publication_Time', y=target, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, color='red')

# Second subplot
ax2 = plt.subplot(gs[0, 1])  # Top-middle
sns.boxplot(data=train_0, x='Publication_Day', y=target, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, color='darkgreen')
ax2.set_ylabel('')

# Third subplot
ax3 = plt.subplot(gs[0, 2])  # Top-right
sns.boxplot(data=train_0, x='Episode_Sentiment', y=target, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, color='blue')
ax3.set_ylabel('')

# Fourth subplot
ax4 = plt.subplot(gs[1, 0])  # Bottom-left
sns.boxplot(data=train_0, x='Genre', y=target, ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90, color='maroon')

# Fifth subplot (spanning bottom-middle and bottom-right)
ax5 = plt.subplot(gs[1, 1:])  # Bottom spanning last two columns
sns.boxplot(data=train_0, x='Podcast_Name', y=target, ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
ax5.set_ylabel('')

# sixth subplot (spanning bottom)
ax6 = plt.subplot(gs[2, :])  # Bottom spanning
sns.boxplot(data=train_0, x='Episode_Title', y=target, ax=ax6)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90)
ax6.set_ylabel('')

plt.tight_layout()
plt.show()


train_0.select_dtypes(exclude='number'
                     ).nunique().sort_values(ascending=False
                                                             ).to_frame(name='nunmer of uniques'
                                                                       ).style.bar(cmap='RdYlGn_r')


null_count = pd.DataFrame({'train': train_0.isna().sum(), 
                           'test': test_0.isna().sum(), 
                           'external': ext_0.isna().sum()}
                         )
null_count.style.format("{:.0f}").background_gradient(cmap='Reds')


print(f'There are {ext_0[target].isna().sum()} missing targets in the external data. We decided to drop them.')

ext_0 = ext_0.dropna(subset=target)



# Define a function to perform the adversarial validation of two datasets
def adversarial_validation(df_1, df_2, name_1, name_2):
    adv_df_1 = df_1[num_features].copy()
    adv_df_2 = df_2[num_features].copy()


    # label the test and train data with 0 and 1 (it doesn't really matter which is which)
    adv_df_1 = adv_df_1.assign(adv=1)
    adv_df_2 = adv_df_2.assign(adv=0)


    # combine the training and test data into one big dataset
    combined = pd.concat([adv_df_1, adv_df_2], axis=0)

    # Shuffle
    combined = combined.sample(frac=1, random_state=64)

    # perform the binary classification, for example using XGboost
    X_combined = combined.drop('adv', axis=1)
    y_combined = combined.adv


    cv = StratifiedKFold(n_splits = 5,
                        shuffle = True,
                        random_state = 64)
    xgb_model = XGBClassifier(max_depth=3,
                              learning_rate = 0.1,
                              n_estimators = 100,
                              objective = 'binary:logistic',
                              random_state = 64)

    # Get the cross validation scores
    adv_scores = []
    for i, _ in enumerate(cv.split(X_combined, y_combined)):
        X_train, X_valid, y_train, y_valid = train_test_split(X_combined, 
                                                              y_combined, 
                                                              test_size=0.3)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict_proba(X_valid)[:,1]
        score = roc_auc_score(y_valid, y_pred)
        adv_scores.append(score)

#         print(f"Fold {i+1} AUC Score: {score:.5f}")

    #Plot the roc_curve
    mean_auc = np.mean(adv_scores)
    fpr, tpr, _ = roc_curve(y_valid, y_pred)
    plt.plot(fpr, tpr, label = 'roc_curve (AUC = %0.4f)' % mean_auc, color='red')
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray', label = 'Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'roc_curve {name_1} vs {name_2}', weight='bold')
    plt.legend()


num_features = list(test_0.select_dtypes('number'))
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
adversarial_validation(train_0, test_0, 'train', 'test')
plt.subplot(1,2,2)
adversarial_validation(test_0, ext_0, 'train', 'external')


num_feat = test_0.select_dtypes('number').columns.tolist()
print(f'numeric features: {num_feat}\n')
cat_feat = test_0.select_dtypes(exclude='number').columns.tolist()
print(f'categoric features: {cat_feat}')


# def missing_handler(df):
for df in [train_0, test_0, ext_0]:
    for feat in cat_feat:
        df[feat] = df[feat].bfill()
    for feat in num_feat:
        df[feat] = df[feat].fillna(df[feat].median())


def pick_the_train_data(include_ext=False):
    if include_ext:
        train_data = pd.concat([train_0, ext_0], ignore_index=True)
        print('The external data is combined with the train data for training set')
    else:
        train_data = train_0
        print('The external data is left out of training set')
    return train_data


train_data = pick_the_train_data(include_ext=True)

X = train_data.copy()
y = X.pop(target)

X.shape


# class OutlierHandler(BaseEstimator, TransformerMixin):
#     def fit(self, df, y=None):
#         return self
    
#     def transform(self, df):
#         df = df.copy()  # Avoid modifying the original DataFrame
#         for col in df.select_dtypes(include=['number']).columns:
#             Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
#             Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             df[col] = np.clip(df[col], lower_bound, upper_bound)
#         return df


class OutlierHandler(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()  # Avoid modifying the original DataFrame
        for col in df.select_dtypes(include=['number']).columns:
            # Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
            # Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
            # IQR = Q3 - Q1
            lower_bound = ext_0[col].min()
            upper_bound = ext_0[col].max()
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df


class Feature_Eng(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        df['Episode_Sentiment'] = df['Episode_Sentiment'].map({'Positive':1, 'Negative':-1, 'Neutral':0})
        df = df.astype({'Podcast_Name': 'category', 
              'Episode_Title': 'category', 
              'Genre': 'category', 
              'Publication_Day': 'category', 
              'Publication_Time': 'category', 
              'Episode_Sentiment': 'category'})

        df['Episode_numb'] = df['Episode_Title'].apply(lambda x: x.split(' ')[1])
        df['Popularity_Average'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'])/2
        df['Episode_Length/Ads'] = df['Number_of_Ads']/df['Episode_Length_minutes']
        return df


column_trans = make_column_transformer(
    (OneHotEncoder(), cat_feat),
    (RobustScaler(), num_feat), 
    remainder='passthrough', 
    sparse_threshold=0)



prep_pipeline = make_pipeline(OutlierHandler(), Feature_Eng(), column_trans)

prep_pipeline


# Define the objective function
def objective_xgb(trial):
    xgb_param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),  # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),  # L2 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    # Data preprocessing and model initialization
    model = Pipeline([
            ('outlier_handler', OutlierHandler()),
            ('feature_eng', Feature_Eng()),
            ('column_trans', column_trans),
            ('xgb', XGBRegressor(**xgb_param_grid, verbose=True))
        ])
    
    # Ensure X and y are defined externally
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8)
    model.fit(X_train, y_train)

    # Evaluate the model using RMSE
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# Define the function to run the study
def Run_Pass_xgb_study(n_trials=1):
    if n_trials > 1:
        # Create and run the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_xgb, n_trials=n_trials, timeout=36000, show_progress_bar=True)
        best_study_params = study.best_params

        # Print results
        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        # best_study_params = {'n_estimators': 950, 
        #                      'learning_rate': 0.027347897049165393, 
        #                      'max_depth': 15, 
        #                      'subsample': 0.7696489083594584, 
        #                      'colsample_bytree': 0.5727158534273917, 
        #                      'reg_alpha': 9.262199389885755, 
        #                      'reg_lambda': 0.7134001369547716, 
        #                      'min_child_weight': 2}
        
        best_study_params = {'n_estimators': 930, 
                              'learning_rate': 0.031629338917482916, 
                              'max_depth': 15, 
                              'subsample': 0.9468611176708397, 
                              'colsample_bytree': 0.8042585491461298, 
                              'reg_alpha': 9.193697404267928, 
                              'reg_lambda': 2.4499368941014024, 
                              'min_child_weight': 3}

    print(f"Best parameters: {best_study_params}")
    return best_study_params

xgb_best_params = Run_Pass_xgb_study(n_trials=1)


my_spliter = KFold(n_splits=5, shuffle=True)

cv_splits = my_spliter.split(X, y)
scores = []

for f, (train_idx, val_idx) in enumerate(cv_splits, start=1):
    # Initialize the XGBRegressor
    model = Pipeline([
            ('outlier_handler', OutlierHandler()),
            ('feature_eng', Feature_Eng()),
            ('column_trans', column_trans),
            ('xgb', XGBRegressor(**xgb_best_params, verbose=50))
        ])

    
    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]
    
    print(colored('Fold_{}'.format(f), 'red'))
    # Fit the model
    model.fit(X_train, y_train)  # Adjust `early_stopping_rounds` as needed
    
    # Validation predictions and RMSE
    val_pred = model.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, val_pred))
    scores.append(score)
    
    print(colored('RMSE: {:.8f}\n'.format(score), 'red'))

# Final average RMSE
print(colored(f'\nAverage RMSE: {np.mean(scores):.8f} Â± {np.std(scores):.8f}\n', 'green'))


final_model = Pipeline([
            ('outlier_handler', OutlierHandler()),
            ('feature_eng', Feature_Eng()),
            ('column_trans', column_trans),
            ('xgb', XGBRegressor(**xgb_best_params))
        ])

final_model.fit(X, y)


X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=84)

fig, ax = plt.subplots(figsize=(6, 4))
resid = ResidualsPlot(final_model)
resid.fit(X_tr, y_tr)
resid.score(X_va, y_va)
resid.poof();


fig, ax = plt.subplots(figsize=(8, 4))
pred_error = PredictionError(final_model)
pred_error.fit(X_tr, y_tr)
pred_error.score(X_va, y_va)
pred_error.poof();


# Load the sample submission file
sub_0 = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv').copy()

# Predict the target in the test data
sub_0[target] = final_model.predict(test_0).tolist()
# sub_0[target] = test_preds_series

# Safe the csv submission file
sub_0.to_csv('submission.csv', index=False)

# Display the submission file
display(sub_0.head(10))
print('The file is ready for submission!')

