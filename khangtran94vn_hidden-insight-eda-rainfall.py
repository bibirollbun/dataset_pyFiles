import numpy as np 
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score,f1_score

import shap

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train = train.set_index('id')
test = test.set_index('id')
train


def check_df(df,df_name):
    print('***** Checking Dataframe {} *****'.format(df_name))
    if df.duplicated().sum() != 0:
        print('{} dataset have {} duplicated rows.'.format(df_name,df.duplicated().sum()))
    else:
        print('{} dataset have zero duplicated rows.'.format(df_name))
    if df.isnull().sum().sum() != 0:
        print('{} dataset have {} missing values'.format(df_name,df.isnull().sum().sum()))
    else:
        print('{} dataset have zero missing values'.format(df_name))
    print('_'*50)
    print()
    
check_df(train,'Train')
check_df(test,'Test')


#### Only one missing value therefore, I fill forward 
test = test.fillna(method='ffill')

# Define the function to add cyclical encoding
def transform_day(df):
    """
    Converts a day column (1-365) into corresponding month, quarter, and cyclical encoding.
    """
    # Convert day number to a datetime object (assuming non-leap year)
    df['date'] = pd.to_datetime(df['day'], format='%j', errors='coerce')
    
    # Extract month and quarter
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Drop intermediate 'date' column
    df.drop(columns=['date'], inplace=True)

    # Add cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    
    return df

# Apply transformations
train = transform_day(train)
test = transform_day(test)

# Choose a high-contrast colormap
bright_cmap = plt.cm.rainbow

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot for train data
sns.scatterplot(x='day_sin', y='day_cos', data=train, ax=ax[0], hue='quarter',
                palette=bright_cmap, edgecolor='black', linewidth=0.3)
ax[0].set_title('Cyclical Day Train Data, hue = Quarter')

# Scatter plot for test data
sns.scatterplot(x='day_sin', y='day_cos', data=test, ax=ax[1], hue='quarter',
                palette=bright_cmap, edgecolor='black', linewidth=0.3)
ax[1].set_title('Cyclical Day Test Data, hue = Quarter')

# Add markers and labels for Day 1 and Day 365
for i, ax_i in enumerate(ax):
    day1 = train[train['day'] == 1]
    day365 = train[train['day'] == 365]
    
    ax_i.scatter(day1['day_sin'], day1['day_cos'], color='red', s=100, edgecolor='black', label='Day 1', zorder=5)
    ax_i.text(day1['day_sin'].values[0] + 0.02, day1['day_cos'].values[0], "Day 1", color='red', fontsize=10, fontweight='bold')
    
    ax_i.legend()

plt.tight_layout()
plt.show()


# Function to convert wind direction (degrees) to cardinal direction
def wind_direction_to_cardinal(degrees):
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']  # Include 'N' again to handle 360Â°
    index = int(np.round(degrees / 45.0))  # Divide by 45Â° to get index
    return directions[index]

train['Cardinal_Direction'] = train['winddirection'].apply(wind_direction_to_cardinal)
test['Cardinal_Direction'] = test['winddirection'].apply(wind_direction_to_cardinal)

### Function for relationship between temperatue
def temp(df):
    df['diff_temp'] = np.abs(df['maxtemp'] - df['mintemp'])
    df = df.drop(columns = ['temparature','mintemp'],axis = 1)
    return df
train = temp(train)
test = temp(test)


# Select only numerical features
corr = train[['pressure','maxtemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed','rainfall']].corr(method='pearson')

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create the heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', mask=mask)

# Add title
ax.set_title('Pearson Correlation Heatmap')

plt.show()


col = ['pressure','maxtemp','diff_temp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed','rainfall']
for i in col[:-1]:
    fig, ax = plt.subplots(1,4,figsize=(16,5))
    sns.boxplot(y=i,data=train,ax = ax[0],x='rainfall')
    sns.violinplot(y=i,data=train,ax = ax[1],x='rainfall')
    sns.histplot(x=i,data=train,ax = ax[2],bins=10)
    sns.histplot(x=i,data=train,ax = ax[3],hue='rainfall',bins=10,kde=True)
    fig.suptitle('Analysis of Feature: {} vs Rainfall'.format(i))
    plt.tight_layout()
    plt.show()


# Compute normalized counts (100% stacked)
month_rainfall = train.groupby('month')['rainfall'].value_counts(normalize=True).unstack()
quarter_rainfall = train.groupby('quarter')['rainfall'].value_counts(normalize=True).unstack()

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 100% stacked column chart for month
month_rainfall.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax[0])
ax[0].set_title('100% Stacked Column Chart - Rainfall by Month')
ax[0].set_ylabel('Proportion')
ax[0].set_xlabel('Month')

# Plot 100% stacked column chart for quarter
quarter_rainfall.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax[1])
ax[1].set_title('100% Stacked Column Chart - Rainfall by Quarter')
ax[1].set_ylabel('Proportion')
ax[1].set_xlabel('Quarter')

ax[0].tick_params(axis='x', rotation=0)  
ax[1].tick_params(axis='x', rotation=0)  

# Add annotations for each bar segment as whole percentages
for a in ax:
    for container in a.containers:
        labels = [f'{int(v.get_height() * 100)}%' if v.get_height() > 0 else '' for v in container]
        a.bar_label(container, labels=labels, label_type='center', fontsize=10, color='white')

# Adjust layout
plt.tight_layout()
plt.show()


# Define cardinal directions in order
cardinal_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# Get rainfall probability by direction
direction_rainfall = train.groupby('Cardinal_Direction')['rainfall'].value_counts(normalize=True).unstack()[1]  # Only for rainfall=1
direction_rainfall = direction_rainfall.reindex(cardinal_order, fill_value=0)  # Ensure correct order

# Get average wind speed by direction
wind_speed_avg = train.groupby('Cardinal_Direction')['windspeed'].mean().reindex(cardinal_order)

# Convert to numpy arrays
angles = np.linspace(0, 2 * np.pi, len(cardinal_order), endpoint=False).tolist()  # Angles for directions
angles += angles[:1]  # Close the circle

rainfall_values = direction_rainfall.values.tolist()
rainfall_values += rainfall_values[:1]  # Close the circle

wind_values = wind_speed_avg.values.tolist()
wind_values += wind_values[:1]  # Close the circle

# Create radar chart
fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

# 1ï¸�âƒ£ Rainfall Radar Chart
ax[0].plot(angles, rainfall_values, marker='o', label='Rainfall Probability', color='b')
ax[0].fill(angles, rainfall_values, color='b', alpha=0.3)
ax[0].set_xticks(angles[:-1])
ax[0].set_xticklabels(cardinal_order)
ax[0].set_title("Rainfall Probability by Wind Direction")
ax[0].set_ylim(0, 1)  # Set maximum to 1


# Add annotations
for angle, value, label in zip(angles, rainfall_values, cardinal_order):
    ax[0].annotate(f"{value:.0%}", xy=(angle, value), xytext=(5, 5),
                   textcoords="offset points", ha='center', fontsize=10, color='black')

# 2ï¸�âƒ£ Wind Speed Radar Chart
ax[1].plot(angles, wind_values, marker='o', label='Avg Wind Speed', color='r')
ax[1].fill(angles, wind_values, color='r', alpha=0.3)
ax[1].set_xticks(angles[:-1])
ax[1].set_xticklabels(cardinal_order)
ax[1].set_title("Average Wind Speed by Wind Direction")
ax[1].set_ylim(0, 30)  # Set maximum to 30


# Add annotations
for angle, value, label in zip(angles, wind_values, cardinal_order):
    ax[1].annotate(f"{value:.1f}", xy=(angle, value), xytext=(5, 5),
                   textcoords="offset points", ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()


# Convert to a long-format DataFrame
corr_df = (
    corr.where(~mask)  # Keep only lower triangle
    .stack()  # Convert to long format
    .reset_index()
)

# Rename columns
corr_df.columns = ['Feature A', 'Feature B', 'Correlation']

# Filter where |correlation| > 0.7
strong_corr = corr_df[abs(corr_df['Correlation']) > 0.6]
strong_corr


fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # Change to 5 subplots

# First 4 strong correlations
for ax, (_, row) in zip(axes[:4], strong_corr.head(4).iterrows()):
    sns.scatterplot(x=train[row['Feature A']], y=train[row['Feature B']], hue=train['rainfall'], ax=ax)
    ax.set_xlabel(row['Feature A'])
    ax.set_ylabel(row['Feature B'])
    ax.set_title(f"{row['Feature A']} vs {row['Feature B']}")

# Add the 5th scatterplot for 'cloud' vs 'humidity'
sns.scatterplot(x='cloud', y='humidity', data=train, hue='rainfall', ax=axes[4])
axes[4].set_title("cloud vs humidity")
axes[4].set_xlabel("cloud")
axes[4].set_ylabel("humidity")

plt.tight_layout()
plt.show()



def new_features(df):
    df['dewpoint_spread'] = df['maxtemp'] - df['dewpoint']
    df['cloud_sunshine'] = df['cloud'] / (df['sunshine'] + 1)
    df['humidity_index'] = df['dewpoint'] / df['maxtemp']
    df['cloud_humidity'] = (df['cloud'] * df['humidity'])
    df['High_Cloud_Cover'] = (df['cloud'] > 60).astype(int)
    df['High_Humidity'] = (df['humidity'] > 75).astype(int)
    return df

train = new_features(train)
test = new_features(test)
train.groupby('rainfall')[train.columns[-6:]].agg('mean')


for i in train.columns[-6:-2]:
    fig, ax = plt.subplots(1,4,figsize=(16,5))
    sns.boxplot(y=i,data=train,ax = ax[0],x='rainfall')
    sns.violinplot(y=i,data=train,ax = ax[1],x='rainfall')
    sns.histplot(x=i,data=train,ax = ax[2],bins=10)
    sns.histplot(x=i,data=train,ax = ax[3],hue='rainfall',bins=10,kde=True)
    fig.suptitle('Analysis of Feature: {} vs Rainfall'.format(i))
    plt.tight_layout()
    plt.show()


features = ['pressure', 'maxtemp', 'dewpoint']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns

# First row: Boxplots
for ax, feature in zip(axes[0], features):
    sns.boxplot(y=feature, x='Cardinal_Direction', data=train, hue='rainfall', ax=ax)
    ax.set_title(f"Boxplot of {feature}")

# Second row: Violin plots
for ax, feature in zip(axes[1], features):
    sns.violinplot(y=feature, x='Cardinal_Direction', data=train, hue='rainfall', ax=ax, split=True)
    ax.set_title(f"Violin Plot of {feature}")

plt.tight_layout()
plt.show() 


def bin_meteo_features_df(df):
    """
    Bin meteorological features in a DataFrame into geographical categories.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns 'pressure', 'maxtemp', 'dewpoint'
    
    Returns:
        pandas.DataFrame: DataFrame with added columns for binned categories
    """
    
    # Define bin edges and labels for each feature
    # Pressure
    pressure_bins = [999, 1005, 1013, 1020, 1028, 1035]
    pressure_labels = [
        "Very Low",
        "Low",
        "Normal",
        "High",
        "Very High"
    ]
    
    # Maximum Temperature
    maxtemp_bins = [min(np.min(train['maxtemp']), np.min(test['maxtemp'])),
    15, 20, 25, 30,
    max(np.max(train['maxtemp']), np.max(test['maxtemp']))]
    maxtemp_labels = [
        "Cool",
        "Mild",
        "Warm",
        "Hot",
        "Very Hot"
    ]
    
    # Dewpoint
    dewpoint_bins = [min(np.min(train['dewpoint']), np.min(test['dewpoint'])),
    5, 12, 20,
    max(np.max(train['dewpoint']), np.max(test['dewpoint']))]
    dewpoint_labels = [
        "Dry",
        "Moderate",
        "Humid",
        "Very Humid"
    ]
    
    # Create new columns with binned categories
    df['pressure_bin'] = pd.cut(
        df['pressure'],
        bins=pressure_bins,
        labels=pressure_labels,
        include_lowest=True,
        right=True
    )
    
    df['maxtemp_bin'] = pd.cut(
        df['maxtemp'],
        bins=maxtemp_bins,
        labels=maxtemp_labels,
        include_lowest=True,
        right=True
    )
    
    df['dewpoint_bin'] = pd.cut(
        df['dewpoint'],
        bins=dewpoint_bins,
        labels=dewpoint_labels,
        include_lowest=True,
        right=True
    )
    
    return df

train = bin_meteo_features_df(train)
test = bin_meteo_features_df(test)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns layout

for ax, feature in zip(axes, ['pressure_bin', 'maxtemp_bin', 'dewpoint_bin']):
    # Compute rainfall probability
    rainfall_distribution = (
        train.groupby([feature])['rainfall']
        .value_counts(normalize=True)
        .unstack()
    ) * 100  # Convert to percentage

    # Remove columns where all values are zero
    rainfall_distribution = rainfall_distribution.loc[:, (rainfall_distribution != 0).any()]

    # Plot
    rainfall_distribution.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)

    # Add annotations (rounded to whole numbers)
    for bars in ax.containers:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero values
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_y() + height / 2, 
                    f"{int(round(height))}%",  
                    ha='center', va='center', color='white', fontsize=10
                )

    # Labels & title
    ax.set_ylabel("Rainfall Probability (%)")
    ax.set_xlabel(f"{feature}")
    ax.set_title(f"Rainfall Probability ({feature})")
    ax.legend(title="Rainfall", labels=['No Rain', 'Rain'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()


day = train.groupby('day')['rainfall'].value_counts(normalize=True).to_frame().reset_index()
rains_day = day[day['proportion'] == 1]['day'].nunique()
rain_data = pd.DataFrame({
    "Category": ["Always", "Inconsistent"],
    "Count": [rains_day, 365 - rains_day]
})

train['records_always_rain'] = train['day'].isin(day[day['proportion'] == 1]['day'])
# Create subplots with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Calculate all rain days vs non-all rain days
day = train.groupby('day')['rainfall'].value_counts(normalize=True).to_frame().reset_index()
rains_day = day[day['proportion'] == 1]['day'].nunique()
rain_data = pd.DataFrame({
    "Category": ["Always", "Inconsistent"],
    "Count": [rains_day, 365 - rains_day]
})

# --- First subplot: Countplot of rainfall values ---
sns.countplot(data=train, x="rainfall", ax=axes[0])
axes[0].set_title("Rainfall Count in Train Dataset")
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Rainfall Amount")

# Add integer annotations for first plot
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}',  
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10)

# --- Second subplot: All Rain Days vs Non-All Rain Days ---
sns.barplot(data=rain_data, x="Category", y="Count", ax=axes[1])
axes[1].set_title("Record Rain Days in Train Dataset")
axes[1].set_ylabel("Number of Days")
axes[1].set_xlabel("Category")

# Add integer annotations for second plot
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}',  
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10)

# --- Third subplot: Scatter plot with rain day legend ---
sns.scatterplot(
    x='day_sin', y='day_cos', data=train, ax=axes[2],
    hue='records_always_rain', 
    palette={True: 'orange', False: 'gray'},  # Ensure True = orange
    edgecolor='black', linewidth=0.3
)

handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(handles, ["Inconsistent", "Always"], title="Record Rain Days")
axes[2].set_title("Record Rain Days during Year in Train Dataset")

plt.tight_layout()
plt.show()

# Get the list of "Always rain" days from the train DataFrame
always_rain_days = train.loc[train['records_always_rain'], 'day'].unique()

# Create the records_always_test column in the test DataFrame
test['records_always_rain'] = test['day'].isin(always_rain_days)


# Get the column at index 9
col_to_move = train.columns[9]

# Create a new column order: all columns except the one at index 9, then add it at the end
new_column_order = train.columns[:9].tolist() + train.columns[10:].tolist() + [col_to_move]

# Reorder the DataFrame
train = train[new_column_order]

# Map True to 1, False to 0
train['records_always_rain'] = train['records_always_rain'].map({True: 1, False: 0})
test['records_always_rain'] = test['records_always_rain'].map({True: 1, False: 0})

# Drop columns id vs day
train = train.reset_index().drop(columns=['id','day','Cardinal_Direction'])
test = test.reset_index().drop(columns=['id','day','Cardinal_Direction'])


# Define X,y
X = train.drop(columns='rainfall')
y = train['rainfall']
X_test = test.copy()
# Train-test-split with stratify = y because imbalance dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)


def encode_ordinal_columns(X_train, X_val, X_test):
    # Define category orders
    pressure_order = ["Very Low", "Low", "Normal", "High", "Very High"]
    maxtemp_order = ["Cool", "Mild", "Warm", "Hot", "Very Hot"]
    dewpoint_order = ["Dry", "Moderate", "Humid", "Very Humid"]

    # Set up the encoder
    encoder = OrdinalEncoder(categories=[pressure_order, maxtemp_order, dewpoint_order],
                             handle_unknown='use_encoded_value', unknown_value=-1)
    columns_to_encode = ['pressure_bin', 'maxtemp_bin', 'dewpoint_bin']
    
    # Create deep copies to avoid modifying original data
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()

    # Ensure columns are strings before encoding
    for df in [X_train_enc, X_val_enc, X_test_enc]:
        df[columns_to_encode] = df[columns_to_encode].astype(str)
    
    # Fit and transform on training data
    X_train_enc[columns_to_encode] = encoder.fit_transform(X_train_enc[columns_to_encode]).astype(int)
    
    # Transform validation and test data
    X_val_enc[columns_to_encode] = encoder.transform(X_val_enc[columns_to_encode]).astype(int)
    X_test_enc[columns_to_encode] = encoder.transform(X_test_enc[columns_to_encode]).astype(int)
    
    return X_train_enc, X_val_enc, X_test_enc

# Get the encoded versions
X_train_encode, X_val_encode, X_test_encode = encode_ordinal_columns(X_train, X_val, X_test)


# === Model Definition and Training ===
# Define XGBoost model with best hyperparameters
XGBoost_Model1 = xgb.XGBClassifier(
    n_estimators=841,
    learning_rate=0.19075484915736696,
    max_depth=4,
    min_child_weight=10,
    gamma=3.9247497193983323,
    subsample=0.563934561965989,
    colsample_bytree=0.6586125002673843,
    scale_pos_weight=9.40032400372528,
    objective='binary:logistic',
    random_state=42,
    early_stopping_rounds=10
)

# Train the model with validation set
XGBoost_Model1.fit(
    X_train_encode, y_train,
    eval_set=[(X_val_encode, y_val)],
    verbose=10
)

# === ROC Curve Plot ===
# Predict probabilities for the positive class
y_prob = XGBoost_Model1.predict_proba(X_val_encode)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# === Feature Importance Plot ===
# Get feature importance scores
feature_importance = XGBoost_Model1.feature_importances_

# Create DataFrame and sort by importance
importance_df = pd.DataFrame({
    'Feature': X_train_encode.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Plot feature importance with annotations
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()

# Add annotations to bars
for bar in bars:
    plt.text(
        bar.get_width() + 0.005, 
        bar.get_y() + bar.get_height() / 2, 
        f'{bar.get_width():.2f}', 
        ha='left', 
        va='center'
    )
plt.show()

# === SHAP Analysis ===
# Create SHAP explainer and compute SHAP values
explainer = shap.Explainer(XGBoost_Model1, X_train_encode)
shap_values = explainer(X_val_encode)

# Generate SHAP summary plot
shap.summary_plot(shap_values, X_val_encode)


# === Predict Probabilities for the Positive Class ===
y_prob = XGBoost_Model1.predict_proba(X_val_encode)[:, 1]

# === Predict the class labels ===
y_pred = XGBoost_Model1.predict(X_val_encode)

# === 1. Precision-Recall Curve ===
# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
average_precision = average_precision_score(y_val, y_prob)

# === 2. Confusion Matrix ===
# Compute confusion matrix
cm = confusion_matrix(y_val, y_pred)

# === 3. Classification Report ===
# Generate classification report

print(classification_report(y_val, y_pred))

class_report = classification_report(y_val, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# === Create Subplots (1,3) for Precision-Recall, Confusion Matrix, and Classification Report ===
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# === Plot Precision-Recall Curve ===
axs[0].plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
axs[0].set_xlabel('Recall')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision-Recall Curve')
axs[0].legend(loc='lower left')
axs[0].grid(True)

# === Plot Confusion Matrix as Heatmap ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axs[1])
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')
axs[1].set_title('Confusion Matrix')

# === Plot Classification Report as Heatmap ===
sns.heatmap(class_report_df.iloc[:-1, :-1].astype(float), annot=True, cmap='Blues', fmt='.2f', cbar=True, ax=axs[2])
axs[2].set_xlabel('Metrics')
axs[2].set_ylabel('Classes')
axs[2].set_title('Classification Report')

# Adjust layout and show plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# Define the new XGBoost model with updated hyperparameters
XGBoost_Model2 = xgb.XGBClassifier(
    max_depth=9,
    learning_rate=0.16103988822353865,
    n_estimators=154,
    min_child_weight=9,
    subsample=0.8775694757347743,
    colsample_bytree=0.8227587533995413,
    gamma=4.957387406466892,
    scale_pos_weight=0.6437042215765173,
    objective='binary:logistic',
    random_state=42,early_stopping_rounds=10
)

# Fit the model
XGBoost_Model2.fit(X_train_encode, y_train, eval_set=[(X_val_encode, y_val)],  verbose=10)

# Make predictions and get probabilities
y_pred = XGBoost_Model2.predict(X_val_encode)
y_prob = XGBoost_Model2.predict_proba(X_val_encode)[:, 1]

# --- 1. Plot ROC AUC Curve ---
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve for XGBoost_model2")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- 2. Feature Importance with Annotations ---
feature_importance = XGBoost_Model2.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train_encode.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightcoral')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Model 2 Feature Importance (with Annotations)')
plt.gca().invert_yaxis()

for bar in bars:
    plt.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f'{bar.get_width():.2f}',
        ha='left',
        va='center'
    )

plt.show()

# --- 3. SHAP Analysis ---
explainer = shap.Explainer(XGBoost_Model2, X_train_encode)
shap_values = explainer(X_val_encode)

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_val_encode, show=False)

plt.tight_layout()
plt.show()



# === Predict Probabilities for the Positive Class ===
y_prob = XGBoost_Model2.predict_proba(X_val_encode)[:, 1]

# === Predict the class labels ===
y_pred = XGBoost_Model2.predict(X_val_encode)

# === 1. Precision-Recall Curve ===
# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
average_precision = average_precision_score(y_val, y_prob)

# === 2. Confusion Matrix ===
# Compute confusion matrix
cm = confusion_matrix(y_val, y_pred)

# === 3. Classification Report ===
# Generate classification report

print(classification_report(y_val, y_pred))

class_report = classification_report(y_val, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# === Create Subplots (1,3) for Precision-Recall, Confusion Matrix, and Classification Report ===
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# === Plot Precision-Recall Curve ===
axs[0].plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
axs[0].set_xlabel('Recall')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision-Recall Curve')
axs[0].legend(loc='lower left')
axs[0].grid(True)

# === Plot Confusion Matrix as Heatmap ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axs[1])
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')
axs[1].set_title('Confusion Matrix')

# === Plot Classification Report as Heatmap ===
sns.heatmap(class_report_df.iloc[:-1, :-1].astype(float), annot=True, cmap='Blues', fmt='.2f', cbar=True, ax=axs[2])
axs[2].set_xlabel('Metrics')
axs[2].set_ylabel('Classes')
axs[2].set_title('Classification Report')

# Adjust layout and show plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# Define model with provided hyperparameters
params = {
    'learning_rate': 0.2145,
    'num_leaves': 268,
    'max_depth': 9,
    'min_data_in_leaf': 147,
    'lambda_l1': 0.0228,
    'lambda_l2': 0.1334,
    'feature_fraction': 0.9970,
    'bagging_fraction': 0.7052,
    'bagging_freq': 5,
    'scale_pos_weight': 1.7692,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1
}

# Train the model
LightGBM_Model1 = lgb.LGBMClassifier(**params)
LightGBM_Model1.fit(X_train_encode, y_train)

# Evaluate the model on validation data
y_val_pred_proba = LightGBM_Model1.predict_proba(X_val_encode)[:, 1]

# Plot ROC-AUC curve
fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Feature Importance
plt.figure(figsize=(10, 8))
lgb.plot_importance(LightGBM_Model1, importance_type='gain', max_num_features=20)
plt.title('Feature Importance (by Gain)')
plt.show()

# SHAP values with TreeExplainer
explainer = shap.TreeExplainer(LightGBM_Model1)
shap_values = explainer.shap_values(X_val_encode)

# SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], X_val_encode)
plt.show()


# === Predict Probabilities for the Positive Class ===
y_prob = LightGBM_Model1.predict_proba(X_val_encode)[:, 1]

# === Predict the class labels ===
y_pred = LightGBM_Model1.predict(X_val_encode)

# === 1. Precision-Recall Curve ===
# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
average_precision = average_precision_score(y_val, y_prob)

# === 2. Confusion Matrix ===
# Compute confusion matrix
cm = confusion_matrix(y_val, y_pred)

# === 3. Classification Report ===
# Generate classification report

print(classification_report(y_val, y_pred))

class_report = classification_report(y_val, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# === Create Subplots (1,3) for Precision-Recall, Confusion Matrix, and Classification Report ===
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# === Plot Precision-Recall Curve ===
axs[0].plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
axs[0].set_xlabel('Recall')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision-Recall Curve')
axs[0].legend(loc='lower left')
axs[0].grid(True)

# === Plot Confusion Matrix as Heatmap ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axs[1])
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')
axs[1].set_title('Confusion Matrix')

# === Plot Classification Report as Heatmap ===
sns.heatmap(class_report_df.iloc[:-1, :-1].astype(float), annot=True, cmap='Blues', fmt='.2f', cbar=True, ax=axs[2])
axs[2].set_xlabel('Metrics')
axs[2].set_ylabel('Classes')
axs[2].set_title('Classification Report')

# Adjust layout and show plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


params2 = {
    'num_leaves': 134,
    'max_depth': 3,
    'learning_rate': 0.0785,
    'n_estimators': 859,
    'min_child_samples': 10,
    'subsample': 0.7037,
    'colsample_bytree': 0.7543,
    'reg_alpha': 0.8292,
    'reg_lambda': 2.7462,
    'scale_pos_weight': 1.0266,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1
}

# Train Model 2
LightGBM_Model2 = lgb.LGBMClassifier(**params2)
LightGBM_Model2.fit(X_train_encode, y_train)

# Evaluate Model 2
y_val_pred_proba2 = LightGBM_Model2.predict_proba(X_val_encode)[:, 1]

# Plot ROC-AUC curve for Model 2
fpr2, tpr2, _ = roc_curve(y_val, y_val_pred_proba2)
roc_auc2 = auc(fpr2, tpr2)
plt.figure(figsize=(8, 6))
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'ROC curve (area = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve (Model 2)')
plt.legend(loc='lower right')
plt.show()

# Plot Feature Importance for Model 2
plt.figure(figsize=(10, 8))
lgb.plot_importance(LightGBM_Model2, importance_type='gain', max_num_features=20)
plt.title('Feature Importance (Model 2 - by Gain)')
plt.show()

# SHAP values with TreeExplainer for Model 2
explainer2 = shap.TreeExplainer(LightGBM_Model2)
shap_values2 = explainer2.shap_values(X_val_encode)

# SHAP summary plot for Model 2
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values2[1], X_val_encode)
plt.show()



# === Predict Probabilities for the Positive Class ===
y_prob = LightGBM_Model2.predict_proba(X_val_encode)[:, 1]

# === Predict the class labels ===
y_pred = LightGBM_Model2.predict(X_val_encode)

# === 1. Precision-Recall Curve ===
# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
average_precision = average_precision_score(y_val, y_prob)

# === 2. Confusion Matrix ===
# Compute confusion matrix
cm = confusion_matrix(y_val, y_pred)

# === 3. Classification Report ===
# Generate classification report

print(classification_report(y_val, y_pred))

class_report = classification_report(y_val, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# === Create Subplots (1,3) for Precision-Recall, Confusion Matrix, and Classification Report ===
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# === Plot Precision-Recall Curve ===
axs[0].plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
axs[0].set_xlabel('Recall')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision-Recall Curve')
axs[0].legend(loc='lower left')
axs[0].grid(True)

# === Plot Confusion Matrix as Heatmap ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axs[1])
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')
axs[1].set_title('Confusion Matrix')

# === Plot Classification Report as Heatmap ===
sns.heatmap(class_report_df.iloc[:-1, :-1].astype(float), annot=True, cmap='Blues', fmt='.2f', cbar=True, ax=axs[2])
axs[2].set_xlabel('Metrics')
axs[2].set_ylabel('Classes')
axs[2].set_title('Classification Report')

# Adjust layout and show plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# Define the new CatBoost model with updated hyperparameters
categorical_columns = ['month', 'quarter', 'High_Cloud_Cover', 'High_Humidity', 'pressure_bin', 'maxtemp_bin', 'dewpoint_bin', 'records_always_rain']


Catboost_Model1 = catboost.CatBoostClassifier(
    iterations=377,
    depth=6,
    learning_rate=0.02085400276118919,
    l2_leaf_reg=0.0010737999495730064,
    subsample=0.7666661318336279,
    colsample_bylevel=0.7922396264594407,
    scale_pos_weight=5.909987581971301,
    objective='Logloss',  # Binary classification objective
    random_state=42,
    cat_features=categorical_columns,  # Specify categorical columns
    eval_metric='AUC',  # Evaluate using ROC AUC
    loss_function='Logloss',  # Binary classification loss
)

# Fit the model
Catboost_Model1.fit(X_train_encode, y_train, eval_set=[(X_val_encode, y_val)], verbose=10)

# Make predictions and get probabilities
y_pred = Catboost_Model1.predict(X_val_encode)
y_prob = Catboost_Model1.predict_proba(X_val_encode)[:, 1]

# --- 1. Plot ROC AUC Curve ---
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve for CatBoost_Model1")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- 2. Feature Importance with Annotations ---
feature_importance = Catboost_Model1.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': X_train_encode.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightcoral')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('CatBoost Model 1 Feature Importance (with Annotations)')
plt.gca().invert_yaxis()

for bar in bars:
    plt.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f'{bar.get_width():.2f}',
        ha='left',
        va='center'
    )

plt.show()

# --- 3. SHAP Analysis ---
# Use TreeExplainer specifically for tree-based models like CatBoost
explainer = shap.TreeExplainer(Catboost_Model1)

# Get SHAP values
shap_values = explainer.shap_values(X_val_encode)

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_val_encode, show=False)

plt.tight_layout()
plt.show()



# === Predict Probabilities for the Positive Class ===
y_prob = Catboost_Model1.predict_proba(X_val_encode)[:, 1]

# === Predict the class labels ===
y_pred = Catboost_Model1.predict(X_val_encode)

# === 1. Precision-Recall Curve ===
# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
average_precision = average_precision_score(y_val, y_prob)

# === 2. Confusion Matrix ===
# Compute confusion matrix
cm = confusion_matrix(y_val, y_pred)

# === 3. Classification Report ===
# Generate classification report

print(classification_report(y_val, y_pred))

class_report = classification_report(y_val, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# === Create Subplots (1,3) for Precision-Recall, Confusion Matrix, and Classification Report ===
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# === Plot Precision-Recall Curve ===
axs[0].plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
axs[0].set_xlabel('Recall')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision-Recall Curve')
axs[0].legend(loc='lower left')
axs[0].grid(True)

# === Plot Confusion Matrix as Heatmap ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axs[1])
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')
axs[1].set_title('Confusion Matrix')

# === Plot Classification Report as Heatmap ===
sns.heatmap(class_report_df.iloc[:-1, :-1].astype(float), annot=True, cmap='Blues', fmt='.2f', cbar=True, ax=axs[2])
axs[2].set_xlabel('Metrics')
axs[2].set_ylabel('Classes')
axs[2].set_title('Classification Report')

# Adjust layout and show plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# === CatBoost Model 2 ===
Catboost_Model2 = catboost.CatBoostClassifier(
    iterations=820,
    depth=4,
    learning_rate=0.07513008764479123,
    l2_leaf_reg=6.498786889814566,
    border_count=77,
    bagging_temperature=0.22666365214770046,
    random_strength=0.9026811563785527,
    scale_pos_weight=3.195725564080475,
    objective='Logloss',  # Binary classification objective
    random_state=42,
    cat_features=categorical_columns,  # Specify categorical columns
    eval_metric='AUC',  # Evaluate using ROC AUC
    loss_function='Logloss',  # Binary classification loss
)

# Fit the CatBoost_Model2
Catboost_Model2.fit(X_train_encode, y_train, eval_set=[(X_val_encode, y_val)], verbose=10)

# Make predictions and get probabilities for CatBoost_Model2
y_pred_2 = Catboost_Model2.predict(X_val_encode)
y_prob_2 = Catboost_Model2.predict_proba(X_val_encode)[:, 1]

# --- 1. Plot ROC AUC Curve for CatBoost_Model2 ---
fpr, tpr, _ = roc_curve(y_val, y_prob_2)
roc_auc_2 = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc_2:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve for CatBoost_Model2")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- 2. Feature Importance with Annotations for CatBoost_Model2 ---
feature_importance_2 = Catboost_Model2.get_feature_importance()
importance_df_2 = pd.DataFrame({
    'Feature': X_train_encode.columns,
    'Importance': feature_importance_2
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
bars_2 = plt.barh(importance_df_2['Feature'], importance_df_2['Importance'], color='lightcoral')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('CatBoost Model 2 Feature Importance (with Annotations)')
plt.gca().invert_yaxis()

for bar in bars_2:
    plt.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f'{bar.get_width():.2f}',
        ha='left',
        va='center'
    )

plt.show()

# --- 3. SHAP Analysis for CatBoost_Model2 ---
explainer_2 = shap.TreeExplainer(Catboost_Model2)
shap_values_2 = explainer_2.shap_values(X_val_encode)

# SHAP Summary Plot for CatBoost_Model2
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_2, X_val_encode, show=False)

plt.tight_layout()
plt.show()


# List of models
models = [
    ('XGBoost_Model1', XGBoost_Model1),
    ('XGBoost_Model2', XGBoost_Model2),
    ('LightGBM_Model1', LightGBM_Model1),
    ('LightGBM_Model2', LightGBM_Model2),
    ('Catboost_Model1', Catboost_Model1),
    ('Catboost_Model2', Catboost_Model2)
]

# Function to calculate optimal threshold based on F1-score
def get_optimal_threshold_f1(model, X_val_encode, y_val):
    y_prob = model.predict_proba(X_val_encode)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
    return optimal_threshold_f1, np.max(f1_scores)

# Function to calculate optimal threshold based on Youden's J statistic
def get_optimal_threshold_j(model, X_val_encode, y_val):
    y_prob = model.predict_proba(X_val_encode)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    youden_j = tpr - fpr
    optimal_threshold_j = thresholds[np.argmax(youden_j)]
    return optimal_threshold_j, np.max(youden_j)

# Dictionaries to store the optimal thresholds for each model
optimal_thresholds_f1 = {}
optimal_thresholds_j = {}

# Calculate the optimal thresholds and scores for each model
for model_name, model in models:
    # Optimal threshold for F1-score
    optimal_threshold_f1, max_f1 = get_optimal_threshold_f1(model, X_val_encode, y_val)
    optimal_thresholds_f1[model_name] = (optimal_threshold_f1, max_f1)

    # Optimal threshold for Youden's J statistic
    optimal_threshold_j, max_j = get_optimal_threshold_j(model, X_val_encode, y_val)
    optimal_thresholds_j[model_name] = (optimal_threshold_j, max_j)

# Prepare the plot
plt.figure(figsize=(15, 18))

# Subplot 1: ROC AUC Curve
plt.subplot(3, 2, 1)
for model_name, model in models:
    y_prob = model.predict_proba(X_val_encode)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Models')
plt.legend(loc='lower right')

# Subplot 2: Precision-Recall Curve
plt.subplot(3, 2, 2)
for model_name, model in models:
    y_prob = model.predict_proba(X_val_encode)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    average_precision = average_precision_score(y_val, y_prob)
    plt.plot(recall, precision, label=f'{model_name} (AP = {average_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for All Models')
plt.legend(loc='lower left')

# Subplot 3: F1 Scores
plt.subplot(3, 2, 3)
f1_scores = {model: score[1] for model, score in optimal_thresholds_f1.items()}
bars = plt.bar(f1_scores.keys(), f1_scores.values(), color='skyblue')
plt.ylim(0.9, 1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('F1 Scores of 6 Models Using Optimal Threshold')
plt.xticks(rotation=45, ha='right')

# Subplot 4: Youden's J Statistic
plt.subplot(3, 2, 4)
youden_j_scores = {model: score[1] for model, score in optimal_thresholds_j.items()}
bars = plt.bar(youden_j_scores.keys(), youden_j_scores.values(), color='lightgreen')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)
plt.xlabel('Model')
plt.ylabel("Youden's J Statistic")
plt.title("Youden's J Statistic for 6 Models")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')

# Subplot 5: F1-score Threshold Values
plt.subplot(3, 2, 5)
plt.bar(f1_scores.keys(), [optimal_thresholds_f1[m][0] for m in f1_scores.keys()], color='lightcoral')
plt.ylabel("Optimal Threshold")
plt.title("Thresholds Based on F1-Score")
for i, (model, value) in enumerate(zip(f1_scores.keys(), [optimal_thresholds_f1[m][0] for m in f1_scores.keys()])):
    plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')

# Subplot 6: Youden's J Threshold Values
plt.subplot(3, 2, 6)
plt.bar(youden_j_scores.keys(), [optimal_thresholds_j[m][0] for m in youden_j_scores.keys()], color='lightblue')
plt.ylabel("Optimal Threshold")
plt.title("Thresholds Based on Youden's J")
for i, (model, value) in enumerate(zip(youden_j_scores.keys(), [optimal_thresholds_j[m][0] for m in youden_j_scores.keys()])):
    plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()



# probs = XGBoost_Model1.predict_proba(X_val_encode)[:, 1]  # Get probabilities for class 1
# plt.hist(probs[y_val == 1], bins=50, alpha=0.5, label='Class 1')
# plt.hist(probs[y_val == 0], bins=50, alpha=0.5, label='Class 0')
# plt.legend()
# plt.show()


# from sklearn.metrics import precision_score, recall_score, f1_score

# best_threshold = 0.5
# best_f1 = 0

# for threshold in [i / 100 for i in range(1, 100)]:
#     preds = (probs >= threshold).astype(int)
#     f1 = f1_score(y_val, preds)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold

# print(f"Best Threshold: {best_threshold}, Best F1 Score: {best_f1:.2f}")



# precision, recall, thresholds = precision_recall_curve(y_val, probs)

# plt.plot(thresholds, precision[:-1], label="Precision")
# plt.plot(thresholds, recall[:-1], label="Recall")
# plt.xlabel("Threshold")
# plt.legend()
# plt.show()




# fpr, tpr, thresholds = roc_curve(y_val, probs)
# youden = tpr - fpr
# optimal_threshold = thresholds[np.argmax(youden)]

# print(f"Optimal Threshold using Youden's J: {optimal_threshold}")



### Choose ML model
ML = Catboost_Model2
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sample['rainfall'] = ML.predict_proba(X_test_encode)[:,1]
sample.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created!")


sample

