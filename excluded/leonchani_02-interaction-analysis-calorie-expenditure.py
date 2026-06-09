# Table Manipulation, CalculatinTg
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")

# Set seed for reproducibility
np.random.seed(42)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv') # importing 'train' data
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')   # importing 'test' data


total_memory_bytes = df_train.memory_usage(deep=True).sum()
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes} byte")
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**2):.2f} MB") # megabyte display
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**3):.2f} GB")  # Gigabyte display
display(df_train.info(memory_usage='deep'))

# 'train' data
display(df_train)


total_memory_bytes = df_test.memory_usage(deep=True).sum()
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes} byte")
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**2):.2f} MB") # megabyte display
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**3):.2f} GB")  # Gigabyte display
display(df_test.info(memory_usage='deep'))

# 'test' data
display(df_test)


# Create a list of variables (both numerical and categorical data)
numerical_variables = df_train.select_dtypes(include=['number']).columns
categorical_variables = df_train.select_dtypes(include=['object']).columns

numerical_variables_tmp = pd.Index(numerical_variables[:-1], dtype='object').tolist()
categorical_variables_tmp = pd.Index(categorical_variables, dtype='object').tolist()


# # A scatter plot and linear regression line
# for col in numerical_variables_tmp:
#     plt.figure(figsize = (15,6))
#     sns.scatterplot(x = df_train[col], y = df_train[df_train.columns[-1]], alpha = 0.2)
#     sns.regplot(x = df_train[col], y = df_train["Calories"], scatter = False, color = "red")
#     plt.title(f"{col} vs Calories")
#     plt.show()


# Setting
df_train['Calories'] = df_train['Calories'].apply(lambda x: max(0, x)) # Don't let calories drop below 0

# Setting the objective and explanatory variables
X = df_train[['Duration']]  # Regressors must be 2D arrays
y = df_train['Calories']    # objective variable


# Instantiating and Training a Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Obtaining the parameters of the regression equation
parameter = model.coef_[0]   # Regression coefficient (slope)
intercept = model.intercept_ # # intercept

print(f"Regression Equation: Calories = {parameter:.2f} * Duration + {intercept:.2f}")

# Calculate the predicted value of the regression line
df_train['Calories_pred'] = model.predict(X)


# Plotting scatter plots and regression lines
print(f"Regression Equation: Calories = {parameter:.2f} * Duration + {intercept:.2f}")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Duration', y='Calories', data=df_train, alpha=0.5, s=10)              # Scatter Plots
sns.lineplot(x='Duration', y='Calories_pred', data=df_train, color='red', linewidth=2) # Regression Lines

plt.title('Calories versus Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 7))

# Scatter plot plot (colored by gender)
sns.scatterplot(
    x='Duration',
    y='Calories',
    hue='Sex',     # Color based on 'Sex' column
    data=df_train,
    alpha=0.3,
    s=10
)

# Regression analysis by gender and plotting the regression line
sex_categories = df_train['Sex'].unique()  # Get unique gender categories
colors = {'male': 'blue', 'female': 'red'} # Explicitly set line colors for each gender

print("--- Regression equation by gender ---")

for sex in sex_categories:
    
    # Extract only data for the relevant gender
    df_subset = df_train[df_train['Sex'] == sex].copy()

    # Set explanatory variables and target variables
    X = df_subset[['Duration']]
    y = df_subset['Calories']

    # Training a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients and intercepts of the regression equation
    coefficient = model.coef_[0]
    intercept = model.intercept_

    # Define the range of the regression line (from the minimum to maximum Duration values â€‹â€‹for that gender)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_predicted = model.predict(x_range)

    # Plot the regression line
    plt.plot(
        x_range,
        y_predicted,
        color=colors.get(sex, 'gray'), # Defined color, grey if none
        linewidth=2,
        label=f'{sex} (Calories = {coefficient:.2f} * Duration + {intercept:.2f})' # Show regression equation in legend
    )
    print(f"[{sex}] Calories = {coefficient:.2f} * Duration + {intercept:.2f}")

# graph decoration
plt.title('Calories versus Duration by Sex')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.legend(title='Sex', loc='upper left')
plt.grid(True)
plt.show()


# Train a linear regression model on the entire data set
df_train['Residuals'] = df_train['Calories'] - df_train['Calories_pred']

# Create a residual plot
plt.figure(figsize=(10, 5))

# Duration vs Residuals (Color-coded by sex)
sns.scatterplot(x='Duration', y='Residuals', hue='Sex', data=df_train, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Duration by Sex')

plt.tight_layout()
plt.show()


# Model equation including interaction terms
# C(Sex) indicates that Sex should be treated as a categorical variable
# C(Sex):Duration is the interaction term between Sex and Duration
model_formula = 'Calories ~ Duration + C(Sex) + Duration:C(Sex)'
model = smf.ols(formula=model_formula, data=df_train).fit()
print("\n--- Results of a linear regression model with interaction terms ---")
print(model.summary())

# Check the P-values and coefficients to see the significance of interactions.


plt.figure(figsize=(12, 7))

# Scatter plot plot (colored by gender)
sns.scatterplot(
    x='Heart_Rate',
    y='Calories',
    hue='Sex',     # Color based on 'Sex' column
    data=df_train,
    alpha=0.3,
    s=10
)

# Regression analysis by gender and plotting the regression line
sex_categories = df_train['Sex'].unique()  # Get unique gender categories
colors = {'male': 'blue', 'female': 'red'} # Explicitly set line colors for each gender

print("--- Regression equation by gender ---")

for sex in sex_categories:
    
    # Extract only data for the relevant gender
    df_subset = df_train[df_train['Sex'] == sex].copy()

    # Set explanatory variables and target variables
    X = df_subset[['Heart_Rate']]
    y = df_subset['Calories']

    # Training a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients and intercepts of the regression equation
    coefficient = model.coef_[0]
    intercept = model.intercept_

    # Define the range of the regression line (from the minimum to maximum Duration values â€‹â€‹for that gender)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_predicted = model.predict(x_range)

    # Plot the regression line
    plt.plot(
        x_range,
        y_predicted,
        color=colors.get(sex, 'gray'), # Defined color, grey if none
        linewidth=2,
        label=f'{sex} (Calories = {coefficient:.2f} * Heart_Rate + {intercept:.2f})' # Show regression equation in legend
    )
    print(f"[{sex}] Calories = {coefficient:.2f} * Heart_Rate + {intercept:.2f}")

# graph decoration
plt.title('Calories versus Heart_Rate by Sex')
plt.xlabel('Heart_Rate')
plt.ylabel('Calories')
plt.legend(title='Sex', loc='upper left')
plt.grid(True)
plt.show()


# Create a residual plot
plt.figure(figsize=(10, 5))

# Duration vs Residuals (Color-coded by sex)
sns.scatterplot(x='Heart_Rate', y='Residuals', hue='Sex', data=df_train, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Heart_Rate by Sex')

plt.tight_layout()
plt.show()


# Model equation including interaction terms
# C(Sex) indicates that Sex should be treated as a categorical variable
# C(Sex):Duration is the interaction term between Sex and Duration
model_formula = 'Calories ~ Heart_Rate + C(Sex) + Duration:C(Sex)'
model = smf.ols(formula=model_formula, data=df_train).fit()
print("\n--- Results of a linear regression model with interaction terms ---")
print(model.summary())

# Check the P-values and coefficients to see the significance of interactions.


plt.figure(figsize=(12, 7))

# Scatter plot plot (colored by gender)
sns.scatterplot(
    x='Body_Temp',
    y='Calories',
    hue='Sex',     # Color based on 'Sex' column
    data=df_train,
    alpha=0.3,
    s=10
)

# Regression analysis by gender and plotting the regression line
sex_categories = df_train['Sex'].unique()  # Get unique gender categories
colors = {'male': 'blue', 'female': 'red'} # Explicitly set line colors for each gender

print("--- Regression equation by gender ---")

for sex in sex_categories:
    
    # Extract only data for the relevant gender
    df_subset = df_train[df_train['Sex'] == sex].copy()

    # Set explanatory variables and target variables
    X = df_subset[['Body_Temp']]
    y = df_subset['Calories']

    # Training a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients and intercepts of the regression equation
    coefficient = model.coef_[0]
    intercept = model.intercept_

    # Define the range of the regression line (from the minimum to maximum Duration values â€‹â€‹for that gender)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_predicted = model.predict(x_range)

    # Plot the regression line
    plt.plot(
        x_range,
        y_predicted,
        color=colors.get(sex, 'gray'), # Defined color, grey if none
        linewidth=2,
        label=f'{sex} (Calories = {coefficient:.2f} * Body_Temp + {intercept:.2f})' # Show regression equation in legend
    )
    print(f"[{sex}] Calories = {coefficient:.2f} * Body_Temp + {intercept:.2f}")

# graph decoration
plt.title('Calories versus Body_Temp by Sex')
plt.xlabel('Body_Temp')
plt.ylabel('Calories')
plt.legend(title='Sex', loc='upper left')
plt.grid(True)
plt.show()


# Create a residual plot
plt.figure(figsize=(10, 5))

# Duration vs Residuals (Color-coded by sex)
sns.scatterplot(x='Body_Temp', y='Residuals', hue='Sex', data=df_train, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Body_Temp by Sex')

plt.tight_layout()
plt.show()


# Model equation including interaction terms
# C(Sex) indicates that Sex should be treated as a categorical variable
# C(Sex):Duration is the interaction term between Sex and Duration
model_formula = 'Calories ~ Body_Temp + C(Sex) + Duration:C(Sex)'
model = smf.ols(formula=model_formula, data=df_train).fit()
print("\n--- Results of a linear regression model with interaction terms ---")
print(model.summary())

# Check the P-values and coefficients to see the significance of interactions.

