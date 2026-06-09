import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression



app_train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
app_train.shape


app_train.head()


app_test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
app_test.shape


app_test.head()


app_train['TARGET'].value_counts()


import pandas as pd

def missing_values(data):
    mis_val = data.isna().sum()
    mis_perc = (mis_val / len(data)) * 100

    mis_table = pd.DataFrame({'Missing Values': mis_val, '% of Total Values': mis_perc})
    
    mis_table = mis_table[mis_table['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False).round(2)

    count_missing = mis_table.shape[0]
    total_columns = data.shape[1]

    print(f"You have {count_missing} columns with missing values out of {total_columns} total columns.")

    return mis_table



miss_values = missing_values(app_train)
miss_values.head(25)



app_train.dtypes.value_counts()



le = LabelEncoder()
cou = 0

for col in app_train:
    if app_train[col].dtype == 'object':
        if len(app_train[col].unique()) <= 2:
            app_train[col] = app_train[col].fillna('Missing')
            app_test[col] = app_test[col].fillna('Missing')

            le.fit(app_train[col])

            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])

            cou += 1

print(f"Num of columns that are label encoded: {cou}")



app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print(f"Train shape after Encoding: {app_train.shape}")
print(f"Test shape after Encoding: {app_test.shape}")



tar = app_train['TARGET']

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
app_train['TARGET'] = tar

print(f"Train shape: {app_train.shape}")
print(f"Test shape: {app_test.shape}")



num_cols = app_train.select_dtypes(include=['number']).columns
num_count = len(num_cols)

print(f"Number of numerical columns: {num_count}")




def plot_boxplots_with_outliers(data, threshold=0.05, figsize=(14, 6)):
    num_cols = data.select_dtypes(include=['number']).columns
    outlier_info = {}

    for col in num_cols:
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outlier_ratio = outliers / len(data)

        if outlier_ratio > threshold:
            outlier_info[col] = outlier_ratio

    if not outlier_info:
        print("No columns with a high number of outliers found.")
        return

    # Sort columns by outlier percentage (descending order)
    sorted_cols = sorted(outlier_info, key=outlier_info.get, reverse=True)

    plt.figure(figsize=figsize)
    sns.boxplot(data=data[sorted_cols])
    plt.xticks(rotation=90)
    plt.title("Boxplots of Features with High Outliers")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.show()

    print("Columns with high outliers (sorted by outlier ratio):")
    for col, ratio in outlier_info.items():
        print(f"\t{col}: {ratio:.2%} of values are outliers")

# Call the function
plot_boxplots_with_outliers(app_train, threshold=0.05)



app_train['DAYS_EMPLOYED'].describe()


app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243

app_train['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace = True)


app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == 365243

app_test['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace = True)



app_train.shape


app_test.shape


corrle = app_train.corr()['TARGET'].sort_values(ascending=False)

print(f"pos Correlations:\n {corrle.head(15)}")
print(f"neg Correlations:\n {corrle.tail(15)}")



app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])


plt.figure(figsize = (12,6))

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target = 0')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target = 1')

plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages by Target Value')
plt.legend()

plt.show()


age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365 

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)



age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups


age_groups = age_data.groupby('YEARS_BINNED').mean()

plt.figure(figsize=(12, 6))
plt.bar(x=age_groups.index.astype(str), height=100 * age_groups['TARGET'])

plt.title('Failure to Repay by Age Group', pad=25)
plt.xlabel('Age Groups (years)')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corr = ext_data.corr()
ext_data_corr.head()


plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(ext_data_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


plt.figure(figsize = (10, 12))

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    plt.subplot(3, 1, i + 1)
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)
    


poly_features_train = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']].copy()
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']].copy()

poly_target = poly_features_train['TARGET'].copy()

poly_features_train = poly_features_train.drop(columns=['TARGET'])

imputer = SimpleImputer(strategy='median')

imputer.fit(poly_features_train)
poly_features_train_imputed = imputer.transform(poly_features_train)
poly_features_test_imputed = imputer.transform(poly_features_test)

feature_names = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
poly_features_train = pd.DataFrame(poly_features_train_imputed, columns=feature_names)
poly_features_test = pd.DataFrame(poly_features_test_imputed, columns=feature_names)



poly_transformer = PolynomialFeatures(degree=3)

poly_transformer.fit(poly_features_train)
poly_features_train_transformed = poly_transformer.transform(poly_features_train)
poly_features_test_transformed = poly_transformer.transform(poly_features_test)

print('Polynomial Features shape:', poly_features_train_transformed.shape)



feature_names = poly_transformer.get_feature_names_out(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
print(feature_names)



# Create a DataFrame for the polynomial features
poly_features_train = pd.DataFrame(poly_features_train_transformed, 
                                   columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                 'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add the target column
poly_features_train['TARGET'] = poly_target

# Find correlations with the target
poly_corrs = poly_features_train.corr()['TARGET'].sort_values()

# Display the most negative and most positive correlations
print(poly_corrs.head(10))
print(poly_corrs.tail(10))



poly_features_train = pd.DataFrame(poly_features_train_transformed, 
                                   columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                 'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features_train['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features_train, on='SK_ID_CURR', how='left')

poly_features_test = pd.DataFrame(poly_features_test_transformed, 
                                  columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                 'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

print('Training data with polynomial features shape:', app_train_poly.shape)
print('Testing data with polynomial features shape:', app_test_poly.shape)



app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']



train = app_train.drop(columns=['TARGET'], errors='ignore')

features = list(train.columns)

test = app_test.copy()

imputer = SimpleImputer(strategy='median')

scaler = MinMaxScaler(feature_range=(0, 1))

train = imputer.fit_transform(train)
test = imputer.transform(test) 

train = scaler.fit_transform(train)
test = scaler.transform(test)

print('Training data shape:', train.shape)
print('Testing data shape:', test.shape)



train_labels = app_train['TARGET']

log_reg = LogisticRegression(C=0.0001, max_iter=1000)

log_reg.fit(train, train_labels)



log_reg_pred = log_reg.predict_proba(test)[:, 1]

submit = app_test[['SK_ID_CURR']].copy() 
submit.loc[:, 'TARGET'] = log_reg_pred
submit.head()



submission_filename = 'logi_reg_baseline.csv'

submit.to_csv(submission_filename, index=False)

print(f'Submission file "{submission_filename}" has been saved successfully!')



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
random_forest.fit(train, train_labels)

feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

predictions = random_forest.predict_proba(test)[:, 1]



submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

submit.to_csv('random_forest_baseline.csv', index=False)



new_poly_feature_names = list(app_train_poly.columns)

new_imputer = SimpleImputer(strategy='median')

new_poly_features = new_imputer.fit_transform(app_train_poly)
new_poly_features_test = new_imputer.transform(app_test_poly)

new_scaler = MinMaxScaler(feature_range=(0, 1))

new_poly_features = new_scaler.fit_transform(new_poly_features)
new_poly_features_test = new_scaler.transform(new_poly_features_test)

new_random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)

new_random_forest.fit(new_poly_features, train_labels)

new_predictions = new_random_forest.predict_proba(new_poly_features_test)[:, 1]



submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = new_predictions

submit.to_csv('random_forest_baseline_engineered.csv', index=False)





