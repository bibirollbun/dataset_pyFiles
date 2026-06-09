!pip install -qq ydata-profiling


!pip install -qq autogluon


!wget https://raw.githubusercontent.com/ezzaddeentru/recipe-popularity-prediction/main/helper_functions.py



from helper_functions import *



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TRAIN_PATH = '/kaggle/input/playground-series-s5e5/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e5/test.csv'
SUB_PATH = '/kaggle/input/playground-series-s5e5/sample_submission.csv'

ORIG_PATH = '/kaggle/input/calories-burnt-prediction/calories.csv'


df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_sub = pd.read_csv(SUB_PATH)
df_orig = pd.read_csv(ORIG_PATH)


print(f'Train shape: {df_train.shape}')
df_train.head()


print(f'Test shape: {df_test.shape}')
df_test.head()


print(f'Original shape: {df_orig.shape}')
df_orig.head()


print(f'Sub shape: {df_sub.shape}')
df_sub.head()


from ydata_profiling import ProfileReport

df_train_profile = ProfileReport(df_train, title="df_train Profiling Report")
df_train_profile


df_train_profile.to_file("df_train_profile.html")


# https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering
from matplotlib import colors
#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])


plot_feature_distributions(df_train)


df_train.drop(columns=['id'], inplace=True)


numerical_features = df_train.select_dtypes(include=np.number).columns
categorical_features = df_train.select_dtypes(exclude=np.number).columns
print(f'Numerical features: {numerical_features}')
print(f'Categorical features: {categorical_features}')


plot_numerical_features(df_train[numerical_features])


numerical_features_without_calories = numerical_features.drop(['Calories'])

# Scatter plots for numerical features vs Calories
num_features = len(numerical_features_without_calories)
cols = 3
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten()

for i, feature in enumerate(numerical_features_without_calories):
    sns.scatterplot(data=df_train, x=feature, y='Calories', ax=axes[i])
    axes[i].set_title(f'Calories vs {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Calories Burned')
    axes[i].grid(True)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
sns.boxplot(data=df_train, x='Sex', y='Calories')
plt.title(f'Calories vs Sex')
plt.xlabel(feature)
plt.ylabel('Calories Burned')
plt.grid(True)
plt.show()


df_train.columns


sns.set()


corr = df_train[numerical_features].corr()
sns.heatmap(corr, annot=True, center=0, cmap="Blues")


from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='Calories')

# Fit the predictor on the small subset
predictor.fit(df_train, time_limit=600*3)


predictor.leaderboard()


X_train = df_train.drop(columns=['Calories'])
y_train = df_train['Calories']


y_train_pred = predictor.predict(X_train)


from sklearn.metrics import mean_squared_error, root_mean_squared_error
rmse = root_mean_squared_error(y_train, y_train_pred)
print(f'RMSE: {rmse}')
mse = mean_squared_error(y_train, y_train_pred)
print(f'MSE: {mse}')


predictions = predictor.predict(df_test)
predictions


df_sub['Calories'] = predictions
df_sub.to_csv('submission.csv', index=False)


df_sub.Calories.hist()


df_train.Calories.hist()




