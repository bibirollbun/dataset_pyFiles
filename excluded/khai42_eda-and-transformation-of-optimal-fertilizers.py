import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
%pip -q install git+https://github.com/iseedeep/deeprage.git@main
from deeprage.core import val_pie, val_bar, val_all_hist, compare_columns


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


print('Train data')
display(train.head(3))
print('\nTest data')
display(test.head(3))


print('Train data shape: ',train.shape)
print('\nTrain data columns :',train.columns)
print('\nTest data shape: ',test.shape)
print('\nTest data columns :',test.columns)
print('\nTrain data Null Value: ',train.isnull().sum().sum())
print('\nTest data Null Value: ',test.isnull().sum().sum())
print('\nTrain data Duplicate Value: ',train.duplicated().sum())
print('\nTest data Duplicate Value: ',test.duplicated().sum())


print('Train data Dtype :\n\n',train.dtypes)
print('\nTest data Dtype :\n',test.dtypes)


print('Train data description:')
train.drop(columns=['id']).describe()


def unique_value(df, categorical_columns, dataset_name="Dataset"):
    print("=" * 50)
    print(f"Unique values in {dataset_name} categorical features")
    print("=" * 50)
    for col in categorical_columns:
        unique_vals = sorted(df[col].unique())
        value_counts = df[col].value_counts()
        top_value = value_counts.index[0]
        top_freq = value_counts.iloc[0]
        print(f"{col} Unique values: {len(unique_vals)}")
        print(f"Unique values: {unique_vals}")

train_col = ['Soil Type', 'Crop Type', 'Fertilizer Name']
test_col = ['Soil Type', 'Crop Type']

unique_value(train, train_col, "Train Data")
unique_value(test, test_col, "Test Data")


train['Fertilizer Name'].value_counts()


plt.figure(figsize=(12, 6))
train['Fertilizer Name'].value_counts().plot(kind='bar')
plt.title('Distribution of Fertilizer Labels')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


val_pie(train, 'Soil Type')


val_pie(train, 'Crop Type')


val_pie(train, 'Fertilizer Name')


def plot_covariance_matrix(train_df, num_features):
    cov_matrix = train_df[num_features].cov()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Covariance Matrix of Train Data")
    plt.show()

    return cov_matrix
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
covariance_matrix = plot_covariance_matrix(train, num_features)


from scipy import stats
def hypothesis_testing(train_df, num_features, group_col=None, ref_value=0):
    print("=" * 50)
    print("One-sample t-test")
    print("=" * 50)
    for feature in num_features:
        t_stat, p_value = stats.ttest_1samp(train_df[feature], popmean=ref_value)
        print(f"{feature}: t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}")

    if group_col:
        print("=" * 50)
        print(f"ANOVA: Comparing means of {', '.join(num_features)} across {group_col} groups")
        print("=" * 50)
        for feature in num_features:
            grouped_data = [group[feature].values for name, group in train_df.groupby(group_col)]
            f_stat, p_value = stats.f_oneway(*grouped_data)
            print(f"{feature}: F-statistic = {f_stat:.3f}, p-value = {p_value:.4f}")

num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
hypothesis_testing(train, num_features, group_col='Crop Type', ref_value=30)


from scipy.stats import zscore, norm

def gaussian_and_zscore(train_df, num_features):
    zscore_df = train_df[num_features].apply(zscore)  # Z-score for all numerical features
    
    for feature in num_features:
        plt.figure(figsize=(10, 5))
        
        sns.histplot(train_df[feature], bins=30, kde=False, stat="density", color="skyblue")
        
        mean_val = train_df[feature].mean()
        std_val = train_df[feature].std()
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_val, std_val)
        
        plt.plot(x, p, 'r', linewidth=2)
        plt.title(f"{feature} - Gaussian Distribution")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

        print(f"Z-score summary for {feature}:")
        print(zscore_df[feature].describe())
        print("-" * 40)

    return zscore_df

num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
z_scores = gaussian_and_zscore(train, num_features)


from sklearn.preprocessing import PowerTransformer
def power_transform_features(train_df, num_features, method='yeo-johnson'):
    pt = PowerTransformer(method=method, standardize=True)
    transformed_data = pt.fit_transform(train_df[num_features])
    
    transformed_df = pd.DataFrame(transformed_data, columns=num_features)
    
    for feature in num_features:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        sns.histplot(train_df[feature], bins=30, kde=True, color='skyblue')
        plt.title(f"Original - {feature}")
        plt.subplot(1, 2, 2)
        sns.histplot(transformed_df[feature], bins=30, kde=True, color='green')
        plt.title(f"Power Transformed - {feature}")
        
        plt.tight_layout()
        plt.show()
    
    return transformed_df
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
power_transformed_train = power_transform_features(train, num_features)


def compute_t_scores(train_df, num_features):
    t_scores = pd.DataFrame()

    for feature in num_features:
        mean_val = train_df[feature].mean()
        std_err = train_df[feature].std() / np.sqrt(len(train_df[feature]))
        
        t_scores[feature + '_t_score'] = (train_df[feature] - mean_val) / std_err

        print(f"T-score summary for {feature}:")
        print(t_scores[feature + '_t_score'].describe())
        print("=" * 40)

    return t_scores
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
t_scores_df = compute_t_scores(train, num_features)




