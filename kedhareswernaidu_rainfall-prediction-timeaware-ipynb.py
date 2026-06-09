pip install windrose


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_data.head()


train_data.columns


train_data.info()


train_data.shape


train_data.describe().T


train_data.isnull().sum()


# Drop 'id' column it's not needed
df = train_data.drop(columns=['id'])

# List of numerical columns (ignoring 'id')
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Set a rainfall theme using Seaborn
sns.set_theme(style="whitegrid", palette="cool")  # Cool shades for a rainfall vibe

# Set up the plotting grid
plt.figure(figsize=(15, 20))  # Adjust size to fit all plots
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color='steelblue')  # Using steelblue for rainfall
    plt.title(f'Distribution of {col}', fontsize=12, color='navy')
    plt.xlabel(col, fontsize=10, color='darkblue')  # Axis labels with darker blue
    plt.ylabel('Frequency', fontsize=10, color='darkblue')

plt.tight_layout()
plt.show()


## For Better Outlier Detection
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#d3dfe5", "axes.edgecolor": "#96a5ab"})

plt.figure(figsize=(16, 22))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(
        data=df,
        x=col,
        color='#4682b4',
        linewidth=2,
        flierprops={'marker': 'o', 'markerfacecolor': '#4169e1', 'markeredgecolor': '#27408b', 'markersize': 5}
    )
    plt.title(f'{col} Distribution', fontsize=14, color='#27408b', weight='bold')
    plt.xlabel('', fontsize=11, color='#1c3c5a')
    plt.grid(color='#a9c0ce', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()


# Calculate correlation matrix
correlation_matrix = df.corr()

# Set up a rainfall-inspired theme
sns.set_theme(style="whitegrid", palette="cool")

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True, 
    fmt=".2f",
    cmap="Blues",
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Correlation Matrix', fontsize=16, color='navy', weight='bold')
plt.show()


# Group data by Rainfall (0 or 1)
grouped_data = df.groupby('rainfall').mean()

# Display summary statistics
print("Mean Values for Each Group:")
grouped_data


sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#d3dfe5", "axes.edgecolor": "#96a5ab"})

variables_to_plot = df.columns.drop(['rainfall', 'day'])

for col in variables_to_plot:
    g = sns.FacetGrid(df, col="rainfall", height=4, aspect=1.5, palette="Blues")
    g.map(sns.histplot, col, kde=True, color='#4682b4')
    g.set_titles("Rainfall = {col_name}")
    g.fig.suptitle(f"{col} Distribution by Rainfall", fontsize=16, color='#27408b', weight='bold', y=1.05)
    g.set_axis_labels(x_var=col, y_var='Frequency')
    g.fig.subplots_adjust(top=0.85)
    plt.show()


# Dropping 'id' and 'day'
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns.drop(['id','day'])

# Creating a pair plot
sns.set_theme(style="whitegrid")
pair_plot = sns.pairplot(
    df[numerical_columns],
    hue="rainfall",  # Color by rainfall (0 = No rain, 1 = Rain)
    palette="cool",  # Rain-themed palette
    diag_kind="kde",  # Kernel Density Estimate on the diagonal
    markers=["o", "s"],  # Different markers for groups
    plot_kws={"alpha": 0.7, "s": 50}  # Transparency and marker size
)

pair_plot.fig.suptitle("Pair Plot for All Variables", y=1.02, fontsize=16, color="navy")
plt.show()


df['date'] = pd.to_datetime('2021-01-01') + pd.to_timedelta(df['day'] - 1, unit='D')
## We were provided 365 days so - i believe it is from same year


# Select only numeric columns (excluding 'id')
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# If you don't want to include 'day' in the aggregation, remove it:
numeric_cols = [col for col in numeric_cols if col not in ['day']]

# Now, group by date and take the mean of numeric columns
daily_avg = df.groupby('date')[numeric_cols].mean().reset_index()

daily_avg.head()


plot_vars = ['pressure', 'maxtemp', 'temparature', 'mintemp', 
             'dewpoint', 'humidity', 'cloud', 'sunshine', 
             'winddirection', 'windspeed']


sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#d3e4f0", "axes.edgecolor": "#7eaed6"})

fig, axes = plt.subplots(nrows=len(plot_vars), ncols=1, figsize=(14, 3 * len(plot_vars)), sharex=True)

for ax, var in zip(axes, plot_vars):
    # Plot the data points
    ax.plot(daily_avg['date'], daily_avg[var], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=6)

    # Add a trend line by calculating the line of best fit
    x = np.arange(len(daily_avg['date']))
    y = daily_avg[var]
    z = np.polyfit(x, y, 1)  # Linear fit (degree 1)
    trend = np.poly1d(z)

    # Plot the trend line
    ax.plot(daily_avg['date'], trend(x), color='darkorange', linestyle='--', linewidth=1.5, label='Trend Line')

    ax.set_title(f"Average seasonal{var}", fontsize=12, color='navy', weight='bold')
    ax.set_ylabel(var, fontsize=10, color='#27408b')
    ax.grid(color='#a9c0ce', linestyle='--', linewidth=0.7)
    ax.tick_params(axis='x', rotation=45, labelsize=9, colors='#1c3c5a')
    ax.tick_params(axis='y', labelsize=9, colors='#1c3c5a')
    ax.legend(fontsize=8)  # Add legend for the trend line

plt.xlabel("Date", fontsize=12, color='darkblue')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.suptitle("Seasonal Trends", y=1.02, fontsize=16, color='#27408b', weight='bold')
plt.show()


train_data.head()


# Scatter plot: Wind Direction vs Wind Speed
plt.figure(figsize=(8, 6))
sns.scatterplot(x='winddirection', y='windspeed', data=train_data, s=100, color='steelblue')
plt.xlabel("Wind Direction (degrees)")
plt.ylabel("Wind Speed (units)")
plt.title("Scatter Plot of Wind Direction vs Wind Speed")
plt.grid(True)
plt.show()

# Distribution plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(train_data['winddirection'], bins=10, kde=True, color='mediumseagreen')
plt.xlabel("Wind Direction (degrees)")
plt.title("Distribution of Wind Direction")

plt.subplot(1, 2, 2)
sns.histplot(train_data['windspeed'], bins=10, kde=True, color='coral')
plt.xlabel("Wind Speed (units)")
plt.title("Distribution of Wind Speed")

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8, 6))
ax = WindroseAxes.from_ax(fig=fig)
ax.patch.set_facecolor("#d3e4f0")  # Light blue background for a rainfall theme

# Plot the wind rose with a blue colormap and defined bins
ax.bar(df['winddirection'], df['windspeed'], normed=True, opening=0.8,
       edgecolor='white', bins=[0, 10, 20, 30, 40, 50], cmap=plt.cm.Blues)

# Set the title with styling
ax.set_title("Wind Rose: Wind Direction and Speed", color='navy', fontsize=16, weight='bold')

# Create the legend with a light background and navy text
legend = ax.legend(title="Wind Speed (units)", fontsize=10, title_fontsize=12,
                   facecolor="#f0f8ff", edgecolor='navy')
plt.setp(legend.get_texts(), color='navy')  # Ensure legend text is visible

# Reposition the legend outside the plot area to avoid overlap
ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), ncol=3, frameon=True, fancybox=True, shadow=True)

plt.show()


#Cyclical Encoding for â€œdayâ€�

# For the train data
train_data['sin_day'] = np.sin(2 * np.pi * train_data['day'] / 365)
train_data['cos_day'] = np.cos(2 * np.pi * train_data['day'] / 365)

# For the test data
test_data['sin_day'] = np.sin(2 * np.pi * test_data['day'] / 365)
test_data['cos_day'] = np.cos(2 * np.pi * test_data['day'] / 365)


features = [
    'pressure', 'temparature', 'dewpoint', 'humidity',
    'cloud', 'sin_day', 'cos_day'
]


X = train_data[features]
y = train_data['rainfall']


tscv = TimeSeriesSplit(n_splits=5)


# Loop through each split, train and evaluate the model
fold = 1
for train_index, val_index in tscv.split(X):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]
    
    model = RandomForestClassifier(n_estimators=180, random_state=42)
    model.fit(X_train_cv, y_train_cv)
    
    y_val_pred = model.predict(X_val_cv)
    print(f"Fold {fold} Classification Report:")
    print(classification_report(y_val_cv, y_val_pred))
    fold += 1


from sklearn.impute import SimpleImputer
X_test = test_data[features]

# Handle missing values by filling them with median (recommended for RF)
imputer = SimpleImputer(strategy="median")
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=features)

# Check feature order consistency
X_test = X_test[model.feature_names_in_] 


# Make predictions
rainfall_probs = model.predict_proba(X_test)[:, 1]  # Get probability of rain


# Prepare submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': rainfall_probs
})

# Save as CSV
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file saved successfully!")


submission = pd.read_csv('submission.csv')
submission.head(10)




