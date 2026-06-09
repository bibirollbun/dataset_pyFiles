# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').drop(columns=['id'])
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv').drop(columns=['id'])
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Verificar que no queden valores NaN
# Show only columns with missing values
print("Columnas con NaN en train:\n", train.isna().sum()[train.isna().sum() > 0])
print("\nColumnas con NaN en test:\n", test.isna().sum()[test.isna().sum() > 0])
test['winddirection'] = test['winddirection'].fillna(train['winddirection'].median())
print('*******' * 3)
print(train.columns)
print(train.sample(10).T)
print('test:')
print(len(test))
print('sample:')
print(len(sample))


# Aggregate functions
rainfall = pd.DataFrame(train['rainfall'].value_counts()).reset_index()
rainfall = rainfall.sort_values(by='rainfall')
rainfall.head()


import matplotlib.pyplot as plt
import seaborn as sns


def bar_plot(data, x:str, y:str):
    # Set plot style
    sns.set_style("whitegrid")
    
    # Create the bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x=x, y=y, palette='Blues', edgecolor='black')
    
    # Labels and title
    plt.xlabel(f"{x}", fontsize=12)
    plt.ylabel(f"{y}", fontsize=12)
    plt.title("Rainfall Count Distribution", fontsize=14, fontweight='bold')
    
    # Show exact count values on bars
    for i, val in enumerate(data['count']):
        plt.text(i, val + 20, str(val), ha='center', fontsize=8, fontweight='bold')
    
    plt.show()
bar_plot(rainfall, x='rainfall', y='count')


# Aggregate functions
days = pd.DataFrame(train['day'].value_counts()).reset_index()
days = days.sort_values(by='day')
print(days.sample(10))
print(days.describe().T)
# Limit to the top 10 values (optional, to avoid clutter)
days_top = days.sample(10)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    days_top["count"], 
    labels=days_top["day"], 
    autopct='%1.1f%%', 
    colors=sns.color_palette("Blues"), 
    startangle=140,
    wedgeprops={'edgecolor': 'black'}
)

# Add title
plt.title("Top 10 days Count Distribution", fontsize=14, fontweight='bold')
plt.show()


g = 9.81  # Acceleration due to gravity in m/s^2
density_water_vapor = 0.6  # kg/m^3 (density of water vapor)

# Function to calculate the mass of the cloud
def calculate_mass(cloud_cover):
    # Assuming cloud_cover is a percentage (0 to 100)
    volume = 1e9  # 1 km^3 in m^3
    return cloud_cover / 100 * volume * density_water_vapor

# Function to calculate the velocity of the cloud
def calculate_velocity(wind_speed):
    # Convert wind speed from km/h to m/s
    return wind_speed * 1000 / 3600

# Function to calculate the Lagrangian
def calculate_lagrangian(row):
    # Extract values from the row
    cloud_cover = row['cloud']
    wind_speed = row['windspeed']
    height = 1000  # Assume a constant height of 1000 m for clouds

    # Calculate mass and velocity
    mass = calculate_mass(cloud_cover)
    velocity = calculate_velocity(wind_speed)

    # Calculate kinetic energy
    KE = 0.5 * mass * velocity**2

    # Calculate potential energy
    PE = mass * g * height

    # Calculate Lagrangian
    L = KE - PE
    return L

train['lagrangian'] = train.apply(calculate_lagrangian, axis=1)

test['lagrangian'] = test.apply(calculate_lagrangian, axis=1)



print(train['lagrangian'].sample(5)) 


# Define relative humidity
def relative_humidity(dewpoint, temparature):
    # Compute the equation
    HR = (np.exp((17.625 * dewpoint) / (243.04 + dewpoint)) / np.exp((17.625 * temparature) / (243.04 + temparature)))
    return HR

# extract all values that are more than 1.0 in HR  cause theya are naturaly impossible 
train['HR'] = relative_humidity(train['dewpoint'], train['temparature']) * 100
test['HR'] = relative_humidity(test['dewpoint'], test['temparature']) * 100

train['cloud_covered'] = train['cloud'] * train['sunshine'] 
test['cloud_covered'] = test['cloud'] * test['sunshine'] 

train['cloud_pressure'] = (train['cloud'] / train['pressure']) * 100 
test['cloud_pressure'] = (test['cloud']  / test['pressure'] ) * 100

train['cloud_covered_by_pressure']= (train['cloud_covered'] / train['pressure']) * 100
test['cloud_covered_by_pressure']= (test['cloud_covered'] / test['pressure']) * 100

train['cloud_direction'] = (train['cloud_covered'] / (train['winddirection'] * train['windspeed'])) * 10
test['cloud_direction'] = (test['cloud_covered'] /  (test['winddirection'] * test['windspeed'])) * 10

train['cloud_sparsity'] = train['cloud_covered'] * train['cloud_direction']
test['cloud_sparsity'] = test['cloud_covered'] * test['cloud_direction']

train['dif_temp'] = train['maxtemp'] - train['mintemp']
test['dif_temp'] = test['maxtemp'] - test['mintemp']

train['cloud_sparsity_by_max_temp_over_pressure'] = (train['cloud_sparsity']  * train['maxtemp']) / (1 + train['cloud_covered_by_pressure'])
test['cloud_sparsity_by_max_temp_over_pressure'] = (test['cloud_sparsity'] * test['maxtemp']) / (1+test['cloud_covered_by_pressure'])

print('test:')
print(len(test))
print('sample:')
print(len(sample))
train.sample(5)


train.describe().T


rainfall = pd.DataFrame(train['rainfall'].value_counts()).reset_index()
rainfall = rainfall.sort_values(by='rainfall')
print(rainfall.head())
bar_plot(rainfall, x='rainfall', y='count')


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Definir variables predictoras y objetivo
X = train.drop(columns=['rainfall'])
y = train['rainfall']

# Create and fit the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# Get feature importance
feature_importance = rf_model.feature_importances_

# Create a dataframe with features and their importance scores
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(feature_importance_df['feature'], feature_importance_df['importance'], alpha=0.8)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance from Random Forest')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print the feature importance scores
print("\nFeature Importance Scores:")
print(feature_importance_df)



train_no_rain = train[train['rainfall'] == 0.0]
train_rain = train [train['rainfall'] == 1.0]


def plot_comparison_sns(df1, df2, title1="No Rain", title2="Rain"):
    df1['Rain'] = title1
    df2['Rain'] = title2
    combined_df = pd.concat([df1, df2])

    combined_df = combined_df.replace([np.inf, -np.inf], np.nan) # Handle infinite values
    combined_df = combined_df.sample(frac=0.2) # Downsample if needed

    # Melt the DataFrame to long format for FacetGrid
    melted_df = combined_df.melt(id_vars=['day', 'Rain', 'rainfall'], var_name='Variable', value_name='Value')


    g = sns.FacetGrid(melted_df, col='Variable', hue='Rain', col_wrap=5, sharey=False, height=3) # Adjust col_wrap as needed
    g.map(sns.lineplot, 'day', 'Value', marker='o')
    g.add_legend()
    plt.show()

    g = sns.FacetGrid(melted_df, col="Variable", hue="Rain", col_wrap=3, sharey=False, height=3)
    g.map(sns.histplot, 'Value', kde=False, bins=20, element='step') # or bins='auto'
    g.add_legend()
    plt.show()


plot_comparison_sns(train_no_rain.copy(), train_rain.copy())


import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


# Assuming your data is in a pandas DataFrame called 'train'
# Separate features (X) and target (y)
X = train.drop(columns=['rainfall'])
y = train['rainfall']
y = (y > 0.1).astype(int) 

# Define the number of folds for k-fold cross-validation
n_folds = 7  # Increased to 10 folds

# Apply SMOTE to the entire training set
smote = SMOTE(random_state=42)
X = X.astype(np.float32)
y = y.astype(np.int32)

X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature selection
selector = SelectKBest(f_classif, k=15)  # Select top 10 features
X_selected = selector.fit_transform(X_resampled, y_resampled)

# Define the objective function for SVM optimization with regularization
def objective(trial):
    params = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),  # Wider range, lower minimum
        'gamma': trial.suggest_float('gamma', 1e-6, 1e2, log=True),  # Wider range, lower minimum
        'kernel': trial.suggest_categorical('kernel', ['rbf']),  # Removed 'linear' as it often overfits
    }
    model = SVC(probability=True, random_state=42, **params)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auc_scores = []

    for train_index, val_index in skf.split(X_selected, y_resampled):
        X_train_fold, X_val_fold = X_selected[train_index], X_selected[val_index]
        y_train_fold, y_val_fold = y_resampled.iloc[train_index], y_resampled.iloc[val_index]

        # Feature scaling
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        auc_scores.append(roc_auc_score(y_val_fold, y_pred_proba))

    return np.mean(auc_scores)

# Optimize SVM using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Increased trials

# Get the best parameters
best_params = study.best_params
print("Best parameters:", best_params)
print("Best AUC-ROC score:", study.best_value)

# Train final model with the best parameters
scaler = StandardScaler()
X_selected_scaled = scaler.fit_transform(X_selected)

final_svm_model = SVC(probability=True, random_state=42, **best_params)
final_svm_model.fit(X_selected_scaled, y_resampled)

# Try other models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_selected_scaled, y_resampled)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_selected_scaled, y_resampled)

# Function to evaluate models
def evaluate_model(model, X, y):
    y_pred_proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred_proba)

# Evaluate all models
print("SVM AUC-ROC:", evaluate_model(final_svm_model, X_selected_scaled, y_resampled))
print("Random Forest AUC-ROC:", evaluate_model(rf_model, X_selected_scaled, y_resampled))
print("Gradient Boosting AUC-ROC:", evaluate_model(gb_model, X_selected_scaled, y_resampled))


test_selected = selector.transform(test)  # Seleccionar las mismas características
test_scaled = scaler.transform(test_selected)  # Aplicar el mismo escalado
y_pred_proba_svm = final_svm_model.predict(test_scaled)
y_prob_proba_svm = final_svm_model.predict_proba(test_scaled)
y_pred_rf = rf_model.predict(test_scaled)
y_pred_gb = gb_model.predict(test_scaled)
test['y_pred_proba_svm'] = y_pred_proba_svm
test['y_pred_rf'] = y_pred_rf
test['y_pred_gb']= y_pred_gb


test.sample(15)


print(test['y_pred_rf'])




import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
 # Aplicar el mismo escalado
# After training the final model, create predictions for plotting ROC
y_pred_proba = final_svm_model.predict_proba(X_selected_scaled)[:, 1]
rf_pred = rf_model.predict_proba(X_selected_scaled)[:,1]
gb_pred = gb_model.predict_proba(X_selected_scaled)[:,1]
def plot_auc(y_pred_proba):    
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_resampled, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    # Print the AUC score
    print(f'AUC-ROC Score : {roc_auc:.4f}')
    # Create the ROC plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
plot_auc(y_pred_proba)
plot_auc(rf_pred)
plot_auc(gb_pred)


sample['rainfall'] = y_pred_proba_svm

rainfall_sample = pd.DataFrame(sample['rainfall'].value_counts()).reset_index()
rainfall_sample = rainfall_sample.sort_values(by='rainfall')
print(sample.sample(3))
bar_plot(rainfall_sample, x='rainfall', y='count')
sample.to_csv('svm_optuna.csv', index = False)


sample['rainfall'] = y_pred_rf

rainfall_sample = pd.DataFrame(sample['rainfall'].value_counts()).reset_index()
rainfall_sample = rainfall_sample.sort_values(by='rainfall')
print(sample.sample(3))
bar_plot(rainfall_sample, x='rainfall', y='count')
sample.to_csv('svm_optuna.csv', index = False)


sample['rainfall'] = y_pred_gb

rainfall_sample = pd.DataFrame(sample['rainfall'].value_counts()).reset_index()
rainfall_sample = rainfall_sample.sort_values(by='rainfall')
print(sample.sample(3))
bar_plot(rainfall_sample, x='rainfall', y='count')
sample.to_csv('all_model_optuna.csv', index = False)








