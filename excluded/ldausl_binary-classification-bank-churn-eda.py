# Importing Libraries

import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 

import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from scipy import stats


# Reading .csv data file

train_data = pd.read_csv("/kaggle/input/iriisss/train.csv")
test_data = pd.read_csv("/kaggle/input/iriisss/test.csv")


 # Having a look at the dataset

train_data.head()


test_data.head()


# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape

num_test_rows, num_test_columns = test_data.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")


# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")


# Having a look at the description of all the numerical columns present in the dataset

train_data.describe().T


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

def tf_idf(df, column, n, p):
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=n)
    
    # Transform data
    vectors = vectorizer.fit_transform(df[column])
    
    # Apply TruncatedSVD for dimensionality reduction
    svd = TruncatedSVD(p)
    x_pca = svd.fit_transform(vectors)
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(x_pca)

    # Naming columns in the new DataFrame
    cols = [(column + "_tfidf_" + str(f)) for f in tfidf_df.columns.to_list()]
    tfidf_df.columns = cols
    
    # Reset the index of the DataFrame before concatenation
    df = df.reset_index(drop=True)

    # Concatenate transformed features with original data
    df = pd.concat([df, tfidf_df], axis="columns")
    
    return df


numerical_variables = train_data.select_dtypes(include=['number', 'bool'])
columns_to_check = numerical_variables
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.1)
    Q3 = data[column].quantile(0.9)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    return filtered_data, rows_deleted

rows_deleted_total = 0

for column in columns_to_check:
    train_data, rows_deleted = remove_outliers_iqr(train_data, column)
    rows_deleted_total += rows_deleted
    print(f"Rows deleted for {column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")


#customer_state, customer_area_code, has_voice_mail_plan, voice_mail_message_count
train_data = train_data.drop(columns=['customer_state'])
test_data = test_data.drop(columns=['customer_state'])

def fill_voice_mail_data(df):
    df['voice_mail_message_count'] = df.apply(
        lambda row: 28.84 * row['has_voice_mail_plan'] if pd.isna(row['voice_mail_message_count']) else row['voice_mail_message_count'], axis=1
    )
    df['has_voice_mail_plan'] = df.apply(
        lambda row: 1 if pd.isna(row['has_voice_mail_plan']) and pd.notna(row['voice_mail_message_count']) and row['voice_mail_message_count'] > 0 else 0 
        if pd.isna(row['has_voice_mail_plan']) else row['has_voice_mail_plan'], axis=1
    )
    df['voice_mail_message_count'] = df['voice_mail_message_count'].fillna(0)
    df['has_voice_mail_plan'] = df['has_voice_mail_plan'].map({True: 1, False: 0})
    return df
    
train_data = fill_voice_mail_data(train_data)
test_data = fill_voice_mail_data(test_data)
train_data['customer_area_code'] = train_data['customer_area_code'].str.extract('(\d+)')
test_data['customer_area_code'] = test_data['customer_area_code'].str.extract('(\d+)')
train_data['customer_area_code'] = train_data['customer_area_code'].fillna(train_data['customer_area_code'].mode()[0])
test_data['customer_area_code'] = test_data['customer_area_code'].fillna(test_data['customer_area_code'].mode()[0])

#customer_account_duration
train_data['customer_account_duration'] = train_data['customer_account_duration'].fillna(train_data['customer_account_duration'].mean())
test_data['customer_account_duration'] = test_data['customer_account_duration'].fillna(test_data['customer_account_duration'].mean())

#has_international_plan
mapping = {'False': 0, 'false': 0, 'no': 0, 'No': 0, 'FALSE': 0, 'NO': 0, '0': 0,
           'True': 1, 'true': 1, 'yes': 1, 'Yes': 1, 'TRUE': 1, 'YES': 1, '1': 1}
train_data['has_international_plan'] = train_data['has_international_plan'].replace(mapping).astype('float64')
test_data['has_international_plan'] = test_data['has_international_plan'].replace(mapping).astype('float64')
train_data['has_international_plan'] = train_data['has_international_plan'].fillna(0) #train_data['churn'].map({0: 0, 1: 1})
test_data['has_international_plan'] = test_data['has_international_plan'].fillna(0)

#daytime, evening, nighttime, intl : total_minutes, total_charges
def fill_missing_values(df, minutes_col, charges_col, rate, base_charge=0):
    df[charges_col] = df.apply(
        lambda row: rate * row[minutes_col] + base_charge if pd.isna(row[charges_col]) else row[charges_col], axis=1
    )
    df[minutes_col] = df.apply(
        lambda row: (row[charges_col] - base_charge) / rate if pd.isna(row[minutes_col]) else row[minutes_col], axis=1
    )
    df[minutes_col] = df[minutes_col].fillna(df[minutes_col].median()).astype('float64')
    df[charges_col] = df[charges_col].fillna(df[charges_col].median()).astype('float64')
    
columns_info = [
    ('daytime_total_minutes', 'daytime_total_charges', 0.17, 0.30),
    ('evening_total_minutes', 'evening_total_charges', 0.08, 0.13),
    ('nighttime_total_minutes', 'nighttime_total_charges', 0.04, 0.08),
    ('intl_total_minutes', 'intl_total_charges', 0.27, 0)
]
for minutes_col, charges_col, rate, base_charge in columns_info:
    fill_missing_values(train_data, minutes_col, charges_col, rate, base_charge)
    fill_missing_values(test_data, minutes_col, charges_col, rate, base_charge)
    
#daytime, evening, nighttime, intl : total_calls
call_columns = ['daytime_total_calls','evening_total_calls','nighttime_total_calls','intl_total_calls']
for col in call_columns:
    train_data[col] = train_data[col].fillna(train_data[col].median())
    test_data[col] = test_data[col].fillna(test_data[col].median())

#customer_service_call_count
train_data['customer_service_call_count'] = train_data['customer_service_call_count'].fillna(0)
test_data['customer_service_call_count'] = test_data['customer_service_call_count'].fillna(0)

def feature_engineering(df):
    
    df['daytime_avg_minutes_per_call'] = df['daytime_total_minutes'] / (df['daytime_total_calls'] + 1e-6)
    df['evening_avg_minutes_per_call'] = df['evening_total_minutes'] / (df['evening_total_calls'] + 1e-6)
    df['nighttime_avg_minutes_per_call'] = df['nighttime_total_minutes'] / (df['nighttime_total_calls'] + 1e-6)
    df['intl_avg_minutes_per_call'] = df['intl_total_minutes'] / (df['intl_total_calls'] + 1e-6)
    df['total_minutes'] = (df['daytime_total_minutes'] +df['evening_total_minutes'] +df['nighttime_total_minutes'] +df['intl_total_minutes'])
    df['total_calls'] = (df['daytime_total_calls'] +df['evening_total_calls'] +df['nighttime_total_calls'] +df['intl_total_calls'])
    df['total_charges'] = (df['daytime_total_charges'] +df['evening_total_charges'] +df['nighttime_total_charges'] +df['intl_total_charges'])
    df['charges_per_minute'] = df['total_charges'] / (df['total_minutes'] + 1e-6)
    df['service_calls_per_duration'] = df['customer_service_call_count'] / (df['customer_account_duration'] + 1e-6)

    df['charges_to_account_duration'] = df['total_charges'] / (df['customer_account_duration'] + 1e-6)
    df['charges_per_service_call'] = df['total_charges'] / (df['customer_service_call_count'] + 1e-6)
    df['total_minutes_per_month'] = df['total_minutes'] / (df['customer_account_duration'] + 1e-6)
    df['total_calls_per_month'] = df['total_calls'] / (df['customer_account_duration'] + 1e-6)
    df['total_cost_efficiency'] = df['total_charges'] / (df['total_minutes'] + 1e-6)
    df['day_vs_night_usage'] = df['daytime_total_minutes'] / (df['nighttime_total_minutes'] + 1e-6)
    df['intl_vs_total_usage'] = df['intl_total_minutes'] / (df['total_minutes'] + 1e-6)
    df['service_calls_per_month'] = df['customer_service_call_count'] / (df['customer_account_duration'] + 1e-6)
    df['day_night_charge_ratio'] = df['daytime_total_charges'] / (df['nighttime_total_charges'] + 1e-6)
    df['voice_mail_to_total_calls'] = df['voice_mail_message_count'] / (df['total_calls'] + 1e-6)
    df['intl_calls_ratio'] = df['intl_total_calls'] / (df['total_calls'] + 1e-6)
    df['is_high_usage'] = (df['total_minutes'] > df['total_minutes'].mean()).astype(int)
    df['frequent_service_calls'] = (df['customer_service_call_count'] > df['customer_service_call_count'].mean()).astype(int)

    return df

#median_income = train_data['person_income'].median()
train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)


y = train_data['churn']
id_test = test_data['id']


# Selecting specific columns for encoding
columns_to_encode = ['customer_area_code']
train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]

# Dropping selected columns for scaling
train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)


# Use pandas get_dummies to one-hot encode 'Geography' and 'Gender' in train_data
train_data_encoded = pd.get_dummies(train_data_to_encode, columns=['customer_area_code'], drop_first=True)

# Use pandas get_dummies to one-hot encode 'Geography' and 'Gender' in test_data
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=['customer_area_code'], drop_first=True)


train_data_encoded.head()


test_data_encoded.head()


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training data
minmax_scaler.fit(train_data_to_scale.drop(['churn'], axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['churn'], axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['churn'], axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)


scaled_train_df.head()


scaled_test_df.head()


# Concatenate train datasets
train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)

# Concatenate test datasets
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)


train_data_combined.head()


test_data_combined.head()


submit = True


# Add the 'churn' column back to the scaled training data
train_data_combined['churn'] = train_data['churn'].values


# Select numeric columns
numeric_cols = train_data_combined.select_dtypes(include=['number', 'bool'])

# Calculate the correlation matrix
corr_matrix = numeric_cols.corr()

# Create a heatmap using Seaborn with smaller font size for annotations
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, annot_kws={"size": 8})
plt.title('Correlation Plot of Numeric Columns in train_data_combined')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

X = train_data_combined.drop('churn',axis=1)
y = train_data_combined['churn']

# Function to display feature importance
def display_feature_importance(model, top_n=34,percentage=3, plot=False):
    # Fit the model
    model.fit(X, y, verbose=0)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Calculate threshold based on percentage of the top feature importance
    threshold = percentage / 100 * feature_importance_df.iloc[0]['Importance']
    
    # Select features that meet the threshold
    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
    
    if plot==True:
        # Set seaborn color palette to "viridis"
        sns.set(style="whitegrid", palette="viridis")
    
        # Display or plot the top features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
        plt.title('Feature Importance for {}'.format(type(model).__name__))
        plt.show()
        
        print("Selected Features at threshold {}%; {}".format(percentage,selected_features))
    
    # Add 'churn' to the list of selected features
    selected_features.append('churn')
        
    return selected_features


if submit == False:

    from xgboost import XGBClassifier
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    # List to store AUC scores for each trial percentage
    auc_scores = []

    # List to store selected features for each model and trial percentage
    selected_features_xgb = []
    selected_features_lgb = []
    selected_features_cat = []

    # List of trial percentages
    trial_percentages = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22, 26, 30]

    # Loop over each trial percentage
    for percentage in trial_percentages:
        # Get selected features for each model
        xgb_selected_features = display_feature_importance(XGBClassifier(), percentage=percentage)
        lgb_selected_features = display_feature_importance(lgb.LGBMClassifier(), percentage=percentage)
        cat_selected_features = display_feature_importance(CatBoostClassifier(), percentage=percentage)

        # Append selected features to the respective lists
        selected_features_xgb.append(xgb_selected_features)
        selected_features_lgb.append(lgb_selected_features)
        selected_features_cat.append(cat_selected_features)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit models on training data
        xgb_model = XGBClassifier()
        lgb_model = lgb.LGBMClassifier()
        cat_model = CatBoostClassifier()

        xgb_model.fit(X_train[[feature for feature in xgb_selected_features if feature != 'churn']], y_train, verbose=0)
        lgb_model.fit(X_train[[feature for feature in lgb_selected_features if feature != 'churn']], y_train, verbose=0)
        cat_model.fit(X_train[[feature for feature in cat_selected_features if feature != 'churn']], y_train, verbose=0)

        # Predict probabilities on the test set
        xgb_pred_proba = xgb_model.predict_proba(X_test[[feature for feature in xgb_selected_features if feature != 'churn']])[:, 1]
        lgb_pred_proba = lgb_model.predict_proba(X_test[[feature for feature in lgb_selected_features if feature != 'churn']])[:, 1]
        cat_pred_proba = cat_model.predict_proba(X_test[[feature for feature in cat_selected_features if feature != 'churn']])[:, 1]

        # Calculate AUC scores and append to the list
        auc_xgb = roc_auc_score(y_test, xgb_pred_proba)
        auc_lgb = roc_auc_score(y_test, lgb_pred_proba)
        auc_cat = roc_auc_score(y_test, cat_pred_proba)

        auc_scores.append((auc_xgb, auc_lgb, auc_cat))

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting lines for each model
    plt.plot(trial_percentages, [auc[0] for auc in auc_scores], label='XGB', marker='o')
    plt.plot(trial_percentages, [auc[1] for auc in auc_scores], label='LGB', marker='o')
    plt.plot(trial_percentages, [auc[2] for auc in auc_scores], label='CatBoost', marker='o')

    plt.xlabel('Trial Percentages')
    plt.ylabel('AUC Score')
    plt.title('Model Performance for Different Feature Selection Percentages')
    plt.legend()
    plt.show()


from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_selected_features = display_feature_importance(xgb_model, percentage=0, plot=True)

import lightgbm as lgb
lgb_model = lgb.LGBMClassifier()
lgb_selected_features = display_feature_importance(lgb_model, percentage=2, plot=True)

from catboost import CatBoostClassifier
cat_model = CatBoostClassifier()
cb_selected_features = display_feature_importance(cat_model, percentage=0,  plot=True)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Set a seed for reproducibility
seed = 42

# Initialize all the classification models in the requested format
log_reg = LogisticRegression(random_state=seed, max_iter=1000000)
svc = SVC(random_state=seed, probability=True)
lda = LinearDiscriminantAnalysis()
gnb = GaussianNB()
bnb = BernoulliNB()
knn = KNeighborsClassifier()
gauss = GaussianProcessClassifier(random_state=seed)
rf = RandomForestClassifier(random_state=seed)
et = ExtraTreesClassifier(random_state=seed)
xgb = XGBClassifier(random_state=seed)
lgb = LGBMClassifier(random_state=seed)
dart = LGBMClassifier(random_state=seed, boosting_type='dart')
cb = CatBoostClassifier(random_state=seed, verbose=0)
gb = GradientBoostingClassifier(random_state=seed)
hgb = HistGradientBoostingClassifier(random_state=seed)


if submit==False:

    import optuna
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    # Assuming 'X' is your feature matrix and 'y' is your target variable
    X = train_data_combined[cb_selected_features].drop('churn', axis=1)
    y = train_data_combined['churn']

    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2, log=True),
            'random_state': 42,
            'verbose': 0,
            'eval_metric': 'AUC',
        }

        model = CatBoostClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')


if submit == False:
    import optuna
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    # Assuming 'X' is your feature matrix and 'y' is your target variable
    X = train_data_combined[xgb_selected_features].drop('churn', axis=1)
    y = train_data_combined['churn']

    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'random_state': trial.suggest_categorical('random_state', [42]),
            'tree_method': 'hist',  # Use GPU for training
            'device': 'cuda',
            'eval_metric': 'auc',  # Evaluation metric
            'verbosity': 2,  # Set verbosity to 0 for less output
        }

        model = XGBClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')


if submit == False:
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    # Assuming 'X' is your feature matrix and 'y' is your target variable
    X = train_data_combined[lgb_selected_features].drop('churn', axis=1)
    y = train_data_combined['churn']

    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
        }

        model = lgb.LGBMClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')


import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Updated LGBM Model
lgb_params = {
    'max_depth': 5,
    'min_child_samples': 1,
    'learning_rate': 0.06545162204587239,
    'n_estimators': 927,
    'subsample': 0.981043589736463,
    'colsample_bytree': 0.4120165727598161,
    'reg_alpha': 0.05200005749596763,
    'reg_lambda': 0.970699319910319
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

# Updated XGBoost Model
xgb_params = {
    'max_depth': 5,
    'min_child_weight': 10,
    'learning_rate': 0.06412082210309321,
    'n_estimators': 796,
    'subsample': 0.9631084457336389,
    'colsample_bytree': 0.5501821632318996,
    'random_state': 42
}
xgb_model = XGBClassifier(**xgb_params)

# Updated CatBoost Model
cat_params = {
    'iterations': 839,
    'depth': 5,
    'min_data_in_leaf': 4,
    'learning_rate': 0.08678125515120433,
    'verbose': 0
}
cat_model = CatBoostClassifier(**cat_params)


if submit==False:
    # Define objective function for Optuna to optimize
    X = train_data_combined.drop('churn', axis=1)
    y = train_data_combined['churn']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    xgb_model.fit(X_train[[feature for feature in xgb_selected_features if feature != 'churn']], y_train, verbose=0)
    lgb_model.fit(X_train[[feature for feature in lgb_selected_features if feature != 'churn']], y_train, verbose=0)
    cat_model.fit(X_train[[feature for feature in cb_selected_features if feature != 'churn']], y_train, verbose=0)

    def objective(trial):
        xgb_weight = trial.suggest_uniform('xgb_weight', 0, 1)
        lgb_weight = trial.suggest_uniform('lgb_weight', 0, 1)
        cat_weight = trial.suggest_uniform('cat_weight', 0, 1)

        # Normalize weights to sum up to 1
        total_weight = xgb_weight + lgb_weight + cat_weight
        xgb_weight /= total_weight
        lgb_weight /= total_weight
        cat_weight /= total_weight

        # Ensemble predictions
        ensemble_pred_proba = (
            xgb_weight * xgb_model.predict_proba(X_test[[feature for feature in xgb_selected_features if feature != 'churn']])[:, 1] +
            lgb_weight * lgb_model.predict_proba(X_test[[feature for feature in lgb_selected_features if feature != 'churn']])[:, 1] +
            cat_weight * cat_model.predict_proba(X_test[[feature for feature in cb_selected_features if feature != 'churn']])[:, 1]
        )

        # Assuming y_test is available
        auc_score = roc_auc_score(y_test, ensemble_pred_proba)

        return auc_score

    # Optimize using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)


if submit==False:
    # Get the best weights
    best_weights = study.best_params
    xgb_weight = best_weights['xgb_weight']
    lgb_weight = best_weights['lgb_weight']
    cat_weight = best_weights['cat_weight']
    
    total_weight = xgb_weight + lgb_weight + cat_weight
    xgb_weight /= total_weight
    lgb_weight /= total_weight
    cat_weight /= total_weight
    
    print('xgb_weight: ',xgb_weight)
    print('lgb_weight: ',lgb_weight)
    print('cat_weight: ',cat_weight)


xgb_weight = 0.26421382751952965
lgb_weight = 0.3392368062355378
cat_weight = 0.3965493662449324


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

y = train_data_combined['churn']

xgb_model.fit(train_data_combined[xgb_selected_features].drop('churn', axis=1), y)
xgb_pred_proba = xgb_model.predict_proba(test_data_combined[[feature for feature in xgb_selected_features if feature != 'churn']])[:, 1]

lgb_model.fit(train_data_combined[lgb_selected_features].drop('churn', axis=1), y)
lgb_pred_proba = lgb_model.predict_proba(test_data_combined[[feature for feature in lgb_selected_features if feature != 'churn']])[:, 1]

cat_model.fit(train_data_combined[cb_selected_features].drop('churn', axis=1), y)
cb_pred_proba = cat_model.predict_proba(test_data_combined[[feature for feature in cb_selected_features if feature != 'churn']])[:, 1]

#ensemble_pred_proba = (xgb_pred_proba * 0) + (lgb_pred_proba * 0) + (cb_pred_proba * 1) 
ensemble_pred_proba = (xgb_pred_proba * xgb_weight) + (lgb_pred_proba * lgb_weight) + (cb_pred_proba * cat_weight) 
#ensemble_pred_proba = np.mean([xgb_pred_proba, lgb_pred_proba, cb_pred_proba], axis=0)

# Assuming 'test_data_combined' is the DataFrame for the test set
ensemble_submission_df = pd.DataFrame({
    'id': id_test,
    'churn': ensemble_pred_proba  # Fill in the predicted probabilities
})

# Save the submission DataFrame to a CSV file
ensemble_submission_df.to_csv('submission.csv', index=False)

ensemble_submission_df.head(10)

