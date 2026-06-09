# %pip install optuna
# import optuna

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import gaussian_kde

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import DBSCAN
from category_encoders import TargetEncoder

test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col='id')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(df.columns)


categorical_features = ['road_type', 'num_lanes', 'lighting', 'weather', 
                  'road_signs_present', 'public_road', 'time_of_day', 
                  'holiday', 'school_season']

encoder = TargetEncoder(cols=categorical_features)
df[categorical_features] = encoder.fit_transform(df[categorical_features], df['accident_risk'])


# Plot histograms for all numeric features
df[categorical_features].hist(bins=30, figsize=(15, 12), layout=(4, 3))

plt.suptitle("Feature Distributions")
plt.show()


# Calculate the correlation matrix with Pearson
corr = df.corr()

# Calculate the correlation matrix with Kendall
corr_kendall = df.corr(method='kendall')

# Create the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Heatmap of correlation
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=axes[0], cbar_kws={"shrink": .8})
axes[0].set_title('Correlation Matrix Pearson')

# Heatmap of correlation (Kendall)
sns.heatmap(corr_kendall, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=axes[1], cbar_kws={"shrink": .8})
axes[1].set_title('Correlation Matrix Kendall')

# Adjust layout
plt.tight_layout()
plt.show()


main_features = ['speed_limit', 'lighting',
       'weather', 'curvature']
# after cleaning I can see that  'num_reported_accidents' is not representativ


features = ['road_type', 'num_lanes', 'speed_limit', 'lighting',
       'weather', 'road_signs_present', 'public_road', 'time_of_day',
       'holiday', 'school_season', 'num_reported_accidents', 'curvature']
target= 'accident_risk'


# Create a copy of the original DataFrame to avoid modifying it directly
df_cleaned = df.copy()

# Loop through each feature to calculate ranks and filter outliers
for feature in features:
    # Calculate ranks for the current feature
    df_cleaned[f'{feature}_rank'] = df_cleaned[feature].rank()
    
    # Define outlier thresholds (e.g., top/bottom 25%)
    lower_threshold = df_cleaned[f'{feature}_rank'].quantile(0.10)
    upper_threshold = df_cleaned[f'{feature}_rank'].quantile(0.90)
    
    # Filter out outliers based on the current feature's rank
    df_cleaned = df_cleaned[(df_cleaned[f'{feature}_rank'] >= lower_threshold) & 
                            (df_cleaned[f'{feature}_rank'] <= upper_threshold)]

    # Drop the rank column for the current feature
    df_cleaned = df_cleaned.drop(columns=[f'{feature}_rank'])

# df_cleaned now contains the DataFrame with outliers removed based on the specified features
df_cleaned.head()


print(df.columns)


percentage_unclustered = (1-(len(df_cleaned['road_type'])/len(df['road_type']))) * 100
print("Outlier Volume: {:.2f}%".format(percentage_unclustered))


# Calculate the correlation matrix with Pearson
corr_cleaned = df_cleaned.corr()

# Calculate the correlation matrix with Kendall
corr_kendall_cleaned = df_cleaned.corr(method='kendall')

# Create the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Heatmap of correlation
sns.heatmap(corr_cleaned, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=axes[0], cbar_kws={"shrink": .8})
axes[0].set_title('Correlation Matrix Pearson (Cleaned)')

# Heatmap of correlation (Kendall)
sns.heatmap(corr_kendall_cleaned, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=axes[1], cbar_kws={"shrink": .8})
axes[1].set_title('Correlation Matrix Kendall (Cleaned)')

# Adjust layout
plt.tight_layout()
plt.show()


# Sort the data from corr['accident_risk'] in descending order
corr_sorted = corr['accident_risk'].drop('accident_risk').sort_values(ascending=True)
# Reindex corr_cleaned to match the order of corr_sorted
corr_cleaned_sorted = corr_cleaned['accident_risk'].drop('accident_risk').reindex(corr_sorted.index)

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))

# Set positions for the bars
y_pos = np.arange(len(corr_sorted))

# Create horizontal bars for corr_cleaned['accident_risk']
plt.barh(y_pos - 0.2, corr_cleaned_sorted, height=0.4, color='orange', alpha=0.5, label='corr_cleaned[accident_risk]')
# Create horizontal bars for corr['accident_risk']
plt.barh(y_pos + 0.2, corr_sorted, height=0.4, color='blue', alpha=0.7, label='corr[accident_risk]')

# Set y-ticks to the correct labels
plt.yticks(y_pos, corr_sorted.index)

# Set titles and labels
plt.title('Comparison of Accident Risk Pearson Correlation')
plt.xlabel('Accident Risk')
plt.ylabel('Feature')
plt.legend()

# Show the grid
plt.grid(axis='x')

# Adjust the layout
plt.tight_layout()

# Save the plot
plt.savefig("horizontal_bar_chart_correlation_pearson.png")

# Show the plot
plt.show()


# Sort the data from corr['accident_risk'] in descending order
corr_kendall_sorted = corr_kendall['accident_risk'].drop('accident_risk').sort_values(ascending=True)
# Reindex corr_cleaned to match the order of corr_sorted
corr_kendall_cleaned_sorted = corr_kendall_cleaned['accident_risk'].drop('accident_risk').reindex(corr_sorted.index)

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))

# Set positions for the bars
y_pos = np.arange(len(corr_kendall_sorted))

# Create horizontal bars for corr_cleaned['accident_risk']
plt.barh(y_pos - 0.2, corr_kendall_cleaned_sorted, height=0.4, color='orange', alpha=0.5, label='corr_cleaned[accident_risk]')
# Create horizontal bars for corr['accident_risk']
plt.barh(y_pos + 0.2, corr_kendall_sorted, height=0.4, color='blue', alpha=0.7, label='corr[accident_risk]')

# Set y-ticks to the correct labels
plt.yticks(y_pos, corr_sorted.index)

# Set titles and labels
plt.title('Comparison of Accident Risk Kendall Correlation')
plt.xlabel('Accident Risk')
plt.ylabel('Features')
plt.legend()

# Show the grid
plt.grid(axis='x')

# Adjust the layout
plt.tight_layout()

# Save the plot
plt.savefig("horizontal_bar_chart_correlation_kendal.png")

# Show the plot
plt.show()


# Set up the grid for the plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
axes = axes.flatten()

for i, feature in enumerate(main_features):
    # Create the hexbin plot
    hb = axes[i].hexbin(df_cleaned['accident_risk'], df_cleaned[feature], gridsize=30, cmap='coolwarm', mincnt=1)

    # Add color bar for density
    plt.colorbar(hb, ax=axes[i], label='Point Density')

    # Set titles and labels
    axes[i].set_title(f'Residuals vs. Accident Risk for {feature}')
    axes[i].set_xlabel('Accident Risk')
    axes[i].set_ylabel(feature)
    axes[i].grid()

# Hide the last subplot if there are only 5 features
if len(main_features) < len(axes):
    axes[len(main_features)].set_visible(False)

# Adjust layout
plt.tight_layout()
plt.show()


# Crear la figura y los ejes
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 filas, 2 columnas
axs = axs.flatten()  # Aplanar el array de ejes para facilitar el acceso

for i, feature in enumerate(main_features):
    # Ajustar el modelo de regresiÃ³n lineal
    X = df_cleaned[feature]
    y = df_cleaned['accident_risk']
    X = sm.add_constant(X)  # Agregar el tÃ©rmino de intercepciÃ³n

    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # Crear el grÃ¡fico Q-Q en el eje correspondiente
    sm.qqplot(residuals, line='s', ax=axs[i])
    axs[i].set_title(f'Q-Q Plot of Residuals for {feature}')
    axs[i].grid()

# Ajustar el layout
plt.tight_layout()
plt.show()


from scipy.stats import norm
# Clean NA data
data = df_cleaned['accident_risk'].dropna()

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Create the histogram
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True, stat='density', color='blue', alpha=0.6)

# Create a range of values for the PDF
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
pdf = norm.pdf(x, mean, std_dev)

# Plot the PDF
plt.plot(x, pdf, 'r', linewidth=2)
plt.title('Histogram of Adjusted Normal DistribuciÃ³n [Accident Risk]')
plt.xlabel('Values')
plt.ylabel('Density')
plt.grid()
plt.show()


# Assuming df_cleaned is your DataFrame and target is the column of interest
stat, p_value = stats.shapiro(df_cleaned[target])

alpha = 0.05  # significance level

# Print the results
if p_value > alpha:
    print("The data is normally distributed (fail to reject H0).")
else:
    print("The data is not normally distributed (reject H0).")


def randomGaus(df, target_field):
    # df is your DataFrame and 'target_field' is the column of interest
    # 1. Calculate mean and variance
    mean = df[target_field].mean()
    std_dev = df[target_field].std()

    # 2. Calculate the IQR to identify outliers
    Q1 = df[target_field].quantile(0.25)
    Q3 = df[target_field].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df = df[(df[target_field] >= lower_bound) & (df[target_field] <= upper_bound)]

    # 3. Plot the histogram of the observed distribution
    plt.figure(figsize=(10, 6))
    bins = 30  # You can adjust the number of bins here
    hist, bin_edges = np.histogram(df[target_field], bins=bins, density=True)  # Get the histogram
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Calculate bin centers

    # 4. Calculate the density of the normal distribution in the same bins
    pdf_normal = norm.pdf(bin_centers, mean, std_dev)

    # Plot the histogram and the normal curve as histograms
    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.5, color='blue', label='Observed Distribution')
    plt.bar(bin_centers, pdf_normal, width=bin_edges[1] - bin_edges[0], alpha=0.5, color='orange', label='Normal Distribution')
    plt.title('Comparison of Observed Distribution and Normal Distribution')
    plt.xlabel(f"{target_field}")
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # 5. Create a DataFrame with the difference in counts
    observed_counts = hist * (len(df) * (bin_edges[1] - bin_edges[0]))  # Count of observed records
    normal_counts = np.round(pdf_normal * (len(df) * (bin_edges[1] - bin_edges[0])))  # Count of normal records, rounded
    difference_counts = observed_counts - normal_counts  # Difference in counts

    difference_df = pd.DataFrame({
        'Bin Center': bin_centers,
        'Observed Records': observed_counts,
        'Normal Records': normal_counts,
        'Difference Records': difference_counts
    })

    print(difference_df)

    # 7. Adjust records according to difference_counts
    for i, count in enumerate(difference_counts):
        if count > 0:  # If there are more observed records
            to_remove = int(count)  # Number of records to remove
            bin_mask = (df[target_field] >= bin_edges[i]) & (df[target_field] < bin_edges[i + 1])  # Mask for the current bin
            if df[bin_mask].shape[0] > 0:  # Ensure there are records to remove
                indices_to_remove = df[bin_mask].sample(n=min(to_remove, df[bin_mask].shape[0])).index  # Remove randomly
                df = df.drop(indices_to_remove)
        elif count < 0:  # If there are fewer observed records
            to_duplicate = -int(count)  # Number of records to duplicate
            bin_mask = (df[target_field] >= bin_edges[i]) & (df[target_field] < bin_edges[i + 1])  # Mask for the current bin
            if df[bin_mask].shape[0] > 0:  # Ensure there are records to duplicate
                indices_to_duplicate = df[bin_mask].sample(n=min(to_duplicate, df[bin_mask].shape[0]), replace=True).index  # Duplicate randomly
                df = pd.concat([df, df.loc[indices_to_duplicate]])

    # Now df should have a distribution closer to normal
    return df


df_cleaned=randomGaus(df_cleaned,target)


# OTHER TRANSFORMATIONS [EN RESUMEN TODO CON SKLEARN]
# df_cleaned['accident_risk_transformed'] = np.log(df_cleaned['accident_risk'] + 1) #--> Logaritmica
# df_cleaned['accident_risk_transformed'], lambda_value_risk = stats.yeojohnson(df_cleaned['accident_risk'])
# df_cleaned['accident_risk_transformed'] = np.sqrt(df_cleaned['accident_risk'] + 1) #--> EstandarizaciÃ³n z= Ïƒ / yâˆ’Î¼
# df_cleaned['accident_risk_transformed'], lambda_value_risk = stats.boxcox(df_cleaned['accident_risk']) #-->Box Cox INSERT A TRY
# df_cleaned['accident_risk_transformed'] = df_cleaned['accident_risk'] - (0.044981739716115535-0.13208733938099307*df_cleaned['accident_risk']) #--> Lineal Reg Residuos
#-->LamberW


# Prepare features and target
X = df_cleaned[main_features]
y = df_cleaned[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el escalador
scaler = StandardScaler()

# Escalar las caracterÃ­sticas
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ðŸ§· In the following lines, 
# I will show the code used to calculate the best 
# hyperparameters for the model using Optuna:

# ðŸ§®
# def objective(trial):
#     # Suggest hyperparameters
#     n_estimators = trial.suggest_int('n_estimators', 50, 200)
#     max_depth = trial.suggest_int('max_depth', 5, 50)
#     min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
#     min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
#     bootstrap = trial.suggest_categorical('bootstrap', [True, False])

#     # Create the model
#     model = ExtraTreesRegressor(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         bootstrap=bootstrap,
#         random_state=42
#     )

#     # Train the model
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Calculate the RMSE
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     return rmse

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# print("Best hyperparameters:", study.best_params)
# print("Best RMSE:", study.best_value)


# Initialize and train the Random Forest model
model_ET = ExtraTreesRegressor(n_estimators=182,
                            max_depth=13,
                            min_samples_split=6,
                            min_samples_leaf=1,
                            criterion='squared_error',  # MSE
                            bootstrap=True,              # Bagging
                            random_state=42)
model_ET.fit(X_train, y_train)

# Make predictions
predictions_ET = model_ET.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions_ET))
print("Root Mean Squared Error (RMSE): {:.5f}".format(rmse))


# Initialize and train the Random Forest model
model_RF = RandomForestRegressor(n_estimators=195, max_depth=11, min_samples_split=7, random_state=42)
model_RF.fit(X_train, y_train)

# Make predictions
predictions_RF = model_RF.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions_RF))
print("Root Mean Squared Error (RMSE): {:.5f}".format(rmse))



import matplotlib.pyplot as plt

# Assume you already have your residuals calculated
# residuals_ET and residuals_RF should be defined beforehand
residuals_ET = predictions_ET - y_test
residuals_RF = predictions_RF - y_test  # Make sure to define this

# Create a figure with two subplots in one row
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# First plot: Histogram of Residuals Extra Trees
axs[0].hist(residuals_ET, bins=30, color='blue', alpha=0.7)
axs[0].set_title('Histogram of Residuals Extra Trees')
axs[0].set_xlabel('Residuals')
axs[0].set_ylabel('Frequency')
axs[0].grid()

# Second plot: Histogram of Residuals Linear Regression
axs[1].hist(residuals_RF, bins=30, color='orange', alpha=0.7)
axs[1].set_title('Histogram of Residuals Random Forest')
axs[1].set_xlabel('Residuals')
axs[1].set_ylabel('Frequency')
axs[1].grid()

# Adjust layout
plt.tight_layout()
plt.show()


# Calculate point density for ET
xy_ET = np.vstack([y_test, residuals_ET])
z_ET = gaussian_kde(xy_ET)(xy_ET)  # Kernel Density Estimation for ET

# Calculate point density for RF (assuming residuals_RF is defined)
xy_RF = np.vstack([y_test, residuals_RF])
z_RF = gaussian_kde(xy_RF)(xy_RF)  # Kernel Density Estimation for LR

# Create the figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# First subplot: Residuals Et
scatter_ET = axs[0].scatter(y_test, residuals_ET, c=z_ET, cmap='coolwarm', alpha=0.7)
axs[0].axhline(0, color='red', linestyle='--')
coefficients_ET = np.polyfit(y_test, residuals_ET, 1)
slope_ET, intercept_ET = coefficients_ET
trend_line_ET = np.polyval(coefficients_ET, y_test)
axs[0].plot(y_test, trend_line_ET, color='orange', linestyle='-', label='Trend Line')
axs[0].set_title('Residuals ET vs. Actual Values with Density Color Map')
axs[0].set_xlabel('Actual Values ET (y_test)')
axs[0].set_ylabel('Residuals ET')
axs[0].legend()
axs[0].grid()

# Add the colorbar for ET after the scatter plot
cbar_ET = plt.colorbar(scatter_ET, ax=axs[0], orientation='vertical')
cbar_ET.set_label('Point Density')

# Second subplot: Residuals RF
scatter_RF = axs[1].scatter(y_test, residuals_RF, c=z_RF, cmap='coolwarm', alpha=0.7)
axs[1].axhline(0, color='red', linestyle='--')
coefficients_RF = np.polyfit(y_test, residuals_RF, 1)
slope_RF, intercept_RF = coefficients_RF
trend_line_RF = np.polyval(coefficients_RF, y_test)
axs[1].plot(y_test, trend_line_RF, color='orange', linestyle='-', label='Trend Line')
axs[1].set_title('Residuals RF vs. Actual Values with Density Color Map')
axs[1].set_xlabel('Actual Values RF (y_test)')
axs[1].set_ylabel('Residuals RF')
axs[1].legend()
axs[1].grid()

# Add the colorbar for RF after the scatter plot
cbar_LR = plt.colorbar(scatter_RF, ax=axs[1], orientation='vertical')
cbar_LR.set_label('Point Density')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig("residual_Analysis.png")

# Show the plot
plt.show()

# Print trend line parameters for both
print(f'Slope of the ET trend line: {slope_ET}')
print(f'Intercept of the ET trend line: {intercept_ET}')
print(f'Slope of the RF trend line: {slope_RF}')
print(f'Intercept of the RF trend line: {intercept_RF}')


# Calculate residuals
residuals = predictions_ET - y_test

# Create hexbin plot
plt.figure(figsize=(10, 6))
plt.hexbin(y_test, residuals, gridsize=30, cmap='coolwarm', mincnt=1)
plt.colorbar(label='Counts')

# Add horizontal line at residuals = 0
plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel('y_combined_test')
plt.ylabel('Residuals (predictions_combined - y_test)')
plt.title('Hexbin Plot of Residuals vs y_test')
plt.show()



y_test_arr = np.array(y_test)  # Actual test values
predictions_ET_arr = np.array(predictions_ET)  # Predictions from the model

# Define ranges
#ranges = [(0, 0.2), (0.2, 0.6), (0.6, 1)]
ranges = [(i/5, (i+1)/5) for i in range(4)] #NOT ERRORS OVER y_combined_test>0.8
rmse_values = []
mode_values = []  # List to store mode values
errors_by_range = []

# Calculate RMSE, mean for each range, and collect errors
for lower, upper in ranges:
    mask = (y_test_arr >= lower) & (y_test_arr < upper)
    if np.any(mask):  # Check if there are any values in this range
        rmse = np.sqrt(mean_squared_error(y_test_arr[mask], predictions_ET_arr[mask]))
        rmse_values.append(rmse)
        
        # Collect errors for the current range
        errors = y_test_arr[mask] - predictions_ET_arr[mask]
        errors_by_range.append(errors)

        # Kernel Density Estimation for mode calculation
        kde = stats.gaussian_kde(errors)
        x = np.linspace(min(errors), max(errors), 100)
        mode_value = x[np.argmax(kde(x))]  # Estimate mode
        mode_values.append(mode_value)

# Calculate mean RMSE for the overall predictions
mean_rmse = np.sqrt(mean_squared_error(y_test_arr, predictions_ET_arr))

# Calculate overall errors for mode
overall_errors = y_test_arr - predictions_ET_arr
kde_overall = stats.gaussian_kde(overall_errors)
x_overall = np.linspace(min(overall_errors), max(overall_errors), 100)
overall_mode = x_overall[np.argmax(kde_overall(x_overall))]  # Overall mode

# Create DataFrame for RMSE and mode values
df_RSME = pd.DataFrame({
    'Range': [f"{lower:.1f} to {upper:.1f}" for lower, upper in ranges],
    'RMSE': rmse_values,
    'Estimated Mode of Errors': mode_values
})

# Print results with mode instead of mean of target field
for i, (lower, upper) in enumerate(ranges):
    print(f"RMSE for range {lower} to {upper}: {rmse_values[i]:.5f} - Estimated Mode of Errors: {mode_values[i]:.5f}")

print(f"Mean RMSE: {mean_rmse:.5f} - Overall Estimated Mode of Errors: {overall_mode:.5f}")

# Plot histograms of errors for each range
plt.figure(figsize=(12, 8))
for i, (lower, upper) in enumerate(ranges):
    plt.subplot(len(ranges), 1, i + 1)
    plt.hist(errors_by_range[i], bins=30, alpha=0.5, color='blue', edgecolor='black', density=True)
    
   # Kernel Density Estimation
    kde = stats.gaussian_kde(errors_by_range[i])
    x = np.linspace(min(errors_by_range[i]), max(errors_by_range[i]), 100)
    plt.plot(x, kde(x), color='green', label='KDE')
    
    # Estimated mode line
    plt.axvline(mode_values[i], color='orange', linestyle='dashed', linewidth=1, label='Estimated Mode')

    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Line at zero error
    plt.title(f'Histogram of Prediction Errors for Range {lower} to {upper}')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()


# Create low and high DataFrames based on the target variable
df_cleaned_low = df[df[target] < 0.2]
df_cleaned_up = df[df[target] > 0.6]

# Split both DataFrames into train and test sets
X_low = df_cleaned_low[main_features]
y_low = df_cleaned_low[target]

X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.2, random_state=42)

X_up = df_cleaned_up[main_features]
y_up = df_cleaned_up[target]

X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(X_up, y_up, test_size=0.2, random_state=42)

# Train the low model
model_ET_low = RandomForestRegressor()
model_ET_low.fit(X_train_low, y_train_low)

# Train the high model
model_ET_up = RandomForestRegressor()
model_ET_up.fit(X_train_up, y_train_up)


# Initialize an array to store the final predictions
final_predictions = np.zeros(predictions_ET.shape)

# Making combined predictions based on conditions
for i, pred in enumerate(predictions_ET):
    if pred < 0.2:
        final_predictions[i] = model_ET_low.predict(X_test[i:i+1])[0]  # Predict using ET_low
    elif pred > 0.6:
        final_predictions[i] = model_ET_up.predict(X_test[i:i+1])[0]  # Predict using ET_up
    else:
        final_predictions[i] = pred  # Use the prediction from the general model

# Calculate RMSE for the final predictions
rmse_final = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"RMSE for combined predictions: {rmse_final:.5f}")


y_test_arr = np.array(y_test)  # Actual test values
predictions_ET_arr = np.array(final_predictions)  # Predictions from the model

# Define ranges
#ranges = [(0, 0.2), (0.2, 0.6), (0.6, 1)]
ranges = [(i/5, (i+1)/5) for i in range(4)] #NOT ERRORS OVER y_combined_test>0.8
rmse_values = []
mode_values = []  # List to store mode values
errors_by_range = []

# Calculate RMSE, mean for each range, and collect errors
for lower, upper in ranges:
    mask = (y_test_arr >= lower) & (y_test_arr < upper)
    if np.any(mask):  # Check if there are any values in this range
        rmse = np.sqrt(mean_squared_error(y_test_arr[mask], predictions_ET_arr[mask]))
        rmse_values.append(rmse)
        
        # Collect errors for the current range
        errors = y_test_arr[mask] - predictions_ET_arr[mask]
        errors_by_range.append(errors)

        # Kernel Density Estimation for mode calculation
        kde = stats.gaussian_kde(errors)
        x = np.linspace(min(errors), max(errors), 100)
        mode_value = x[np.argmax(kde(x))]  # Estimate mode
        mode_values.append(mode_value)

# Calculate mean RMSE for the overall predictions
mean_rmse = np.sqrt(mean_squared_error(y_test_arr, predictions_ET_arr))

# Calculate overall errors for mode
overall_errors = y_test_arr - predictions_ET_arr
kde_overall = stats.gaussian_kde(overall_errors)
x_overall = np.linspace(min(overall_errors), max(overall_errors), 100)
overall_mode = x_overall[np.argmax(kde_overall(x_overall))]  # Overall mode

# Add new_RMSE column to df_RSME
df_RSME['new_RMSE'] = rmse_values[:len(df_RSME)]  # Ensure the length matches

print(df_RSME[['RMSE','new_RMSE']])


# Transform the new data using the fitted encoder
test[categorical_features] = encoder.transform(test[categorical_features])


X_submit = test[main_features]
predictions_ET_sumbit = model_ET.predict(X_submit)

# Initialize an array to store the y_pred_submit
y_pred_submit = np.zeros(predictions_ET_sumbit.shape)

# Making combined predictions based on conditions
for i, pred in enumerate(predictions_ET_sumbit):
    if pred < 0.2:
        y_pred_submit[i] = model_ET_low.predict(X_submit[i:i+1])[0]  # Predict using ET_low
    elif pred > 0.6:
        y_pred_submit[i] = model_ET_up.predict(X_submit[i:i+1])[0]  # Predict using ET_up
    else:
        y_pred_submit[i] = pred  # Use the prediction from the general model

df_sub[target] = y_pred_submit
df_sub.to_csv('test_rangeET.csv', index=False)
df_sub.head()

