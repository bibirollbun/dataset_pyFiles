import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


!pip install -qq autogluon


!pip install -qq ydata-profiling


!wget https://raw.githubusercontent.com/ezzaddeentru/recipe-popularity-prediction/main/helper_functions.py


from helper_functions import *


!wget https://raw.githubusercontent.com/ezzaddeentru/used-cars-selling-price-estimating---case-study/refs/heads/main/reg_helper_functions.py


from reg_helper_functions import *


TRAIN_PATH = '/kaggle/input/playground-series-s5e2/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e2/test.csv'
TRAIN_EXTRA_PATH = '/kaggle/input/playground-series-s5e2/training_extra.csv' 
SAMPLE_SUB_PATH = '/kaggle/input/playground-series-s5e2/sample_submission.csv'

ORIGINAL_DATA_PATH = '/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv'


df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_extra = pd.read_csv(TRAIN_EXTRA_PATH)
df_sub = pd.read_csv(SAMPLE_SUB_PATH)

df_orig = pd.read_csv(ORIGINAL_DATA_PATH)


df_train.head()


df_train.info()


df_train.shape


df_test.head()


df_test.info()


df_sub.head()


df_orig.head()


df_orig.info()


from ydata_profiling import ProfileReport

df_train_profile = ProfileReport(df_train, title="df_train Profiling Report")
df_train_profile


df_train_profile.to_file("tdf_train_report.html")


df_orig_profile = ProfileReport(df_orig, title="df_orig Profiling Report")
df_orig_profile


df_train.describe()


df_train.describe(include='object')


plot_feature_distributions(df_train)


sns.pairplot(df_train)


print("\nCorrelation Matrix:")
correlation_matrix = df_train.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Distribution of Categorical Features across Price
categorical_features = df_train.select_dtypes(include='object').columns
categorical_features


for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=feature, y='Price', data=df_train)
    plt.title(f'Distribution of Price across {feature}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)


from sklearn.model_selection import train_test_split

X = df_train.drop('Price', axis=1)
y = df_train['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


df_train_t = pd.concat([X_train, y_train], axis=1)
df_train_t.shape


from autogluon.tabular import TabularPredictor

# Take a smaller sample of the training data (e.g., 10%)
# small_train_df = train_df.sample(frac=0.1, random_state=42)

# Initialize the predictor
predictor = TabularPredictor(label='Price')

# Fit the predictor on the small subset
predictor.fit(df_train_t, time_limit=600)



predictor.leaderboard()


y_train_pred = predictor.predict(X_train)
y_test_pred = predictor.predict(X_test)


autogluon_metrics_df = regression_metrics_df(y_train, y_train_pred, y_test, y_test_pred, 'autogluon')
autogluon_metrics_df


plot_residuals(y_train, y_train_pred, 'Training Residuals')


df_test_pred = predictor.predict(df_test)
df_sub['Price'] = df_test_pred
df_sub


df_sub.to_csv('submission.csv', index=False)

