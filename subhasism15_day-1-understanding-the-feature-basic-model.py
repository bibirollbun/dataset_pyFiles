import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Though we are using seaborn we still require it since plt ensure proper rendering for sns plots. Additionally it easier to build customizable subplots using matplotlib.
import seaborn as sns           # For its aesthetics and ease of creating statisctical plots especially with regards to grouping and aggregation.

sns.set(style='whitegrid')      # Horizontal gridlines on a white background


# Loading data from Kaggle input folders
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Data dimensions - For both Train and Test - Test does not have the target variable - Calories
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print(f"Features in the Training data alongwith the Target variable - Calories: \n {train.columns}")


missing_train= train.isnull().sum()
print(f"Missing and NaN values in train dataset: \n {missing_train}")

missing_test= test.isnull().sum()
print(f"\n \n Missing and NaN values in test dataset: \n {missing_test}")


if train.id.nunique() == len(train):
    print("id has no bearing on the target variable")


train.Sex.value_counts()


plt.figure(figsize=(6, 4))
sns.boxplot(x='Sex', y='Calories', data=train)
plt.title("Calories Burned by Sex")
plt.ylabel("Calories Burned")
plt.show()


sns.violinplot(x='Sex', y='Calories', data=train)
plt.title("Distribution of Calories Burned by Sex")
plt.show()


group_stats = train.groupby('Sex')['Calories'].agg(['mean', 'median', 'count'])
display(group_stats)


from scipy.stats import ttest_ind

male_cals = train[train['Sex'] == 'male']['Calories']
female_cals = train[train['Sex'] == 'female']['Calories']

t_stat, p_value = ttest_ind(male_cals, female_cals, equal_var=False)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")


plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Calories', data=train, alpha=0.3)
plt.title('Calories Burned vs Age')
plt.show()


sns.lmplot(x='Age', y='Calories', data=train, height=5, aspect=1.5)


sns.regplot(x='Age', y='Calories', data=train, lowess=True, scatter_kws={'alpha':0.3})


train['Age_Bin'] = pd.cut(train['Age'], bins=[0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
bin_means = train.groupby('Age_Bin')['Calories'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='Age_Bin', y='Calories', data=bin_means)
plt.title("Average Calories Burned by Age Group")
plt.xticks(rotation=45)
plt.show()


corr = train['Age'].corr(train['Calories'])
print(f"Correlation between Age and Calories: {corr:.3f}")


# Linear Regression

import statsmodels.api as sm

# Define target and feature
X = train[['Age']]
X = sm.add_constant(X)  # adds intercept term
y = train['Calories']

# Fit OLS model
model = sm.OLS(y, X).fit()

# Show summary
print(model.summary())


# Linear Regression

from sklearn.linear_model import LinearRegression

X_age = train[['Age']]
y = train['Calories']

model = LinearRegression()
model.fit(X_age, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")



# POlynomial Regression

train['Age2'] = train['Age'] ** 2

train['Age3'] = train['Age'] ** 3

X_poly = train[['Age', 'Age2', 'Age3']]
X_poly = sm.add_constant(X_poly)
y = train['Calories']

poly_model = sm.OLS(y, X_poly).fit()
print(poly_model.summary())



# POlynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_age, y)

# Plot predicted curve
age_range = np.linspace(train['Age'].min(), train['Age'].max(), 200).reshape(-1, 1)
cal_preds = poly_model.predict(age_range)

plt.figure(figsize=(8, 5))
plt.scatter(train['Age'], train['Calories'], alpha=0.3, label='Observed')
plt.plot(age_range, cal_preds, color='red', label='Polynomial Fit')
plt.title('Polynomial Fit: Age vs Calories Burned')
plt.legend()
plt.show()



plt.figure(figsize=(8, 5))
sns.scatterplot(x='Height', y='Calories', data=train, alpha=0.3)
plt.title('Calories Burned vs Height')
plt.show()


sns.lmplot(x='Height', y='Calories', data=train, height=5, aspect=1.5)


sns.regplot(x='Height', y='Calories', data=train, lowess=True, scatter_kws={'alpha':0.3})



train['Height_Bin'] = pd.cut(train['Height'], bins=30)

# Calculate mean Calories for each bin
bin_means = train.groupby('Height_Bin')['Calories'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x='Height_Bin', y='Calories', data=bin_means)
plt.title("Average Calories Burned by Height Group")
plt.xticks(rotation=45)
plt.show()



corr = train['Height'].corr(train['Calories'])
print(f"Correlation between Height and Calories: {corr:.3f}")


# Linear Regression

import statsmodels.api as sm

# Define target and feature
X = train['Height']
X = sm.add_constant(X)  # adds intercept term
y = train['Calories']

# Fit OLS model
model = sm.OLS(y, X).fit()

# Show summary
print(model.summary())


# Linear Regression

from sklearn.linear_model import LinearRegression

X_age = train[['Height']]
y = train['Calories']

model = LinearRegression()
model.fit(X_age, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")



# POlynomial Regression

train['Height2'] = train['Height'] ** 2

train['Height3'] = train['Height'] ** 3

X_poly = train[['Height', 'Height2', 'Height3']]
X_poly = sm.add_constant(X_poly)
y = train['Calories']

poly_model = sm.OLS(y, X_poly).fit()
print(poly_model.summary())



# POlynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_age, y)

# Plot predicted curve
age_range = np.linspace(train['Height'].min(), train['Height'].max(), 200).reshape(-1, 1)
cal_preds = poly_model.predict(age_range)

plt.figure(figsize=(8, 5))
plt.scatter(train['Height'], train['Calories'], alpha=0.3, label='Observed')
plt.plot(age_range, cal_preds, color='red', label='Polynomial Fit')
plt.title('Polynomial Fit: Height vs Calories Burned')
plt.legend()
plt.show()



plt.figure(figsize=(8, 5))
sns.scatterplot(x='Weight', y='Calories', data=train, alpha=0.3)
plt.title('Calories Burned vs Weight')
plt.show()


sns.lmplot(x='Weight', y='Calories', data=train, height=5, aspect=1.5)


sns.regplot(x='Weight', y='Calories', data=train, lowess=True, scatter_kws={'alpha':0.3})


# Create 10 equal-width bins
train['Weight_Bin'] = pd.cut(train['Weight'], bins=30)

# Calculate mean Calories for each bin
bin_means = train.groupby('Weight_Bin')['Calories'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x='Weight_Bin', y='Calories', data=bin_means)
plt.title("Average Calories Burned by Weight Group")
plt.xticks(rotation=45)
plt.show()



corr = train['Weight'].corr(train['Calories'])
print(f"Correlation between Weight and Calories: {corr:.3f}")


# Linear Regression

import statsmodels.api as sm

# Define target and feature
X = train['Weight']
X = sm.add_constant(X)  # adds intercept term
y = train['Calories']

# Fit OLS model
model = sm.OLS(y, X).fit()

# Show summary
print(model.summary())


# Linear Regression

from sklearn.linear_model import LinearRegression

X_age = train[['Weight']]
y = train['Calories']

model = LinearRegression()
model.fit(X_age, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")



# POlynomial Regression

train['Weight2'] = train['Weight'] ** 2

train['Weight3'] = train['Weight'] ** 3

X_poly = train[['Weight', 'Weight2', 'Weight3']]
X_poly = sm.add_constant(X_poly)
y = train['Calories']

poly_model = sm.OLS(y, X_poly).fit()
print(poly_model.summary())



# POlynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_age, y)

# Plot predicted curve
age_range = np.linspace(train['Weight'].min(), train['Weight'].max(), 200).reshape(-1, 1)
cal_preds = poly_model.predict(age_range)

plt.figure(figsize=(8, 5))
plt.scatter(train['Weight'], train['Calories'], alpha=0.3, label='Observed')
plt.plot(age_range, cal_preds, color='red', label='Polynomial Fit')
plt.title('Polynomial Fit: Weight vs Calories Burned')
plt.legend()
plt.show()



plt.figure(figsize=(8, 5))
sns.scatterplot(x='Duration', y='Calories', data=train, alpha=0.3)
plt.title('Calories Burned vs Duration')
plt.show()


sns.lmplot(x='Duration', y='Calories', data=train, height=5, aspect=1.5)


sns.regplot(x='Duration', y='Calories', data=train, lowess=True, scatter_kws={'alpha':0.3})


corr = train['Duration'].corr(train['Calories'])
print(f"Correlation between Duration and Calories: {corr:.3f}")


# Linear Regression

import statsmodels.api as sm

# Define target and feature
X = train['Duration']
X = sm.add_constant(X)  # adds intercept term
y = train['Calories']

# Fit OLS model
model = sm.OLS(y, X).fit()

# Show summary
print(model.summary())


    # POlynomial Regression

train['Duration2'] = train['Duration'] ** 2

# train['Duration3'] = train['Duration'] ** 3

X_poly = train[['Duration', 'Duration2']]
X_poly = sm.add_constant(X_poly)
y = train['Calories']

poly_model = sm.OLS(y, X_poly).fit()
print(poly_model.summary())



    # POlynomial Regression

train['Duration2'] = train['Duration'] ** 2

train['Duration3'] = train['Duration'] ** 3

X_poly = train[['Duration', 'Duration2', 'Duration3']]
X_poly = sm.add_constant(X_poly)
y = train['Calories']

poly_model = sm.OLS(y, X_poly).fit()
print(poly_model.summary())



plt.figure(figsize=(8, 5))
sns.scatterplot(x='Heart_Rate', y='Calories', data=train, alpha=0.3)
plt.title('Calories Burned vs Heart_Rate')
plt.show()


sns.lmplot(x='Heart_Rate', y='Calories', data=train, height=5, aspect=1.5)


sns.regplot(x = 'Heart_Rate', y = 'Calories', data = train, lowess = True, scatter_kws = {'alpha':0.3})


corr = train['Heart_Rate'].corr(train['Calories'])
print(f"Correlation btwn Heart_Rate and Calories: {corr:.4f}")


import statsmodels.api as sm

X = train[['Heart_Rate']]
X = sm.add_constant(X)
Y = train['Calories']

model = sm.OLS(Y,X).fit()

print(model.summary())


train['Heart_Rate2'] = train['Heart_Rate']**2
# train['Heart_Rate3'] = train['Heart_Rate']**3

X_poly = train[['Age', 'Heart_Rate2']]
X_poly = sm.add_constant(X_poly)

Y = train['Calories']

poly_model = sm.OLS(Y, X_poly).fit()

print(poly_model.summary())


train['Heart_Rate2'] = train['Heart_Rate']**2
train['Heart_Rate3'] = train['Heart_Rate']**3

X_poly = train[['Heart_Rate', 'Heart_Rate2', 'Heart_Rate3']]
X_poly = sm.add_constant(X_poly)

Y = train['Calories']

poly_model = sm.OLS(Y, X_poly).fit()

print(poly_model.summary())


plt.figure(figsize=(8, 5))
sns.scatterplot(x='Body_Temp', y='Calories', data=train, alpha=0.3)
plt.title('Calories Burned vs Body_Temp')
plt.show()


sns.lmplot(x = 'Body_Temp', y = 'Calories', data = train, height = 5, aspect = 1.5)


sns.regplot(x = 'Body_Temp', y = 'Calories', data = train, lowess = True, scatter_kws = {'alpha':0.3})


corr = train['Body_Temp'].corr(train['Calories'])
print(f"Correlation between Body_Temp and Calories: {corr:.4f}")


train['Body_Temp2'] = train['Body_Temp']**2
corr2 = train['Body_Temp2'].corr(train['Calories'])
print(f"Correlation between Body_Temp2 and Calories: {corr:.4f}")

# THis is to improve the understanding on how correlation works


import statsmodels.api as sm

X = train[['Body_Temp']]
X = sm.add_constant(X)

Y = train['Calories']

model = sm.OLS(Y,X).fit()

print(model.summary())


train['Body_Temp2'] = train['Body_Temp']**2
train['Body_Temp3'] = train['Body_Temp']**3

X = train[['Body_Temp', 'Body_Temp2']]
X = sm.add_constant(X)

Y = train['Calories']

model = sm.OLS(Y,X).fit()

print(model.summary())



train['Body_Temp2'] = train['Body_Temp']**2
train['Body_Temp3'] = train['Body_Temp']**3

X = train[['Body_Temp', 'Body_Temp2', 'Body_Temp3']]
X = sm.add_constant(X)

Y = train['Calories']

model = sm.OLS(Y,X).fit()

print(model.summary())


