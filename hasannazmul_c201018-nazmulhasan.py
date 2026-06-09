import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Data preprocessing for training data
# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Define features and target variable
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler  

# Select only numerical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Step 2: Initialize your transformer (e.g., StandardScaler)
scaler = StandardScaler()

# Step 3: Fit and transform only the numerical columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Result: Only numerical columns are transformed, categorical columns remain unchanged
print(X_train)
# print(X_test)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.5f}")


# Load the test dataset
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")

# Preprocess the test dataset
# Encode categorical variables
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

# Select features for prediction
X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# Handle missing values in test data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Scaling
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Make predictions
solution['satisfaction'] = classifier.predict(X_test)

# Map predictions back to original labels
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])


# Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("Submission.csv", index=False)


solution.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv')


df.head()


# Missing Values of each column
missing_values = df.isnull().sum()
print("Missing values in each column: ")
print(missing_values)


# How does Customer Satisfaction vary by class?

# Create cross-tabulation
class_satisfaction = pd.crosstab(df['Class'], df['satisfaction'], 
                                normalize='index') * 100

# Plot styling
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
colors = ['#F44336', '#4CAF50']  # Green for satisfied, red for neutral or dissatisfied

# Create stacked bar plot
class_satisfaction.plot(kind='bar', 
                        stacked=True, 
                        color=colors,
                        edgecolor='black',
                        width=0.7)

# Customize plot
plt.title('Customer Satisfaction Distribution by Travel Class', 
          fontsize=14, pad=20)
plt.xlabel('Travel Class', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Satisfaction', 
           bbox_to_anchor=(1.05, 1), 
           loc='upper left')

# Add percentage labels
for n, x in enumerate([*class_satisfaction.index.values]):
    for (proportion, y_loc) in zip(class_satisfaction.loc[x],
                                  class_satisfaction.loc[x].cumsum()):
        plt.text(x=n - 0.17,
                y=(y_loc - proportion) + (proportion / 2),
                s=f'{proportion:.1f}%',
                color="white",
                fontsize=10,
                fontweight='bold')

plt.tight_layout()
plt.show()


# What's the relationship between flight distance and satisfaction?

data = {
    'Flight Distance': df['Flight Distance'],
    'satisfaction': df['satisfaction']
}

# Set style
plt.style.use('seaborn-v0_8-pastel')
plt.figure(figsize=(12, 6))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot
sns.boxplot(x='satisfaction', y='Flight Distance', data=data, 
            palette={'satisfied': '#2ecc71', 'neutral or dissatisfied': '#e74c3c'},
            width=0.5, ax=ax1)
ax1.set_title('Flight Distance by Satisfaction (Boxplot)', fontsize=14, pad=15)
ax1.set_xlabel('Satisfaction Level', fontsize=12)
ax1.set_ylabel('Flight Distance (miles)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Add median labels
medians = df.groupby('satisfaction')['Flight Distance'].median()
for xtick in ax1.get_xticks():
    ax1.text(xtick, medians[xtick]+50, f'Median: {medians[xtick]:.0f}', 
             horizontalalignment='center', 
             fontsize=10, 
             color='black',
             weight='semibold')

# Violin plot
sns.violinplot(x='satisfaction', y='Flight Distance', data=df, 
               palette={'satisfied': '#2ecc71', 'neutral or dissatisfied': '#e74c3c'},
               inner='quartile', bw=0.2, ax=ax2)
ax2.set_title('Flight Distance by Satisfaction (Violin Plot)', fontsize=14, pad=15)
ax2.set_xlabel('Satisfaction Level', fontsize=12)
ax2.set_ylabel('Flight Distance (miles)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.suptitle('Distribution Analysis: Flight Distance vs Passenger Satisfaction', 
             y=1.05, fontsize=16, weight='bold')
plt.show()


# How do service ratings (WiFi, food, etc.) differ between satisfied vs dissatisfied customers?

from math import pi

data = {
    'satisfaction': df['satisfaction'],
    'Inflight wifi service': df['Inflight wifi service'],
    'Food and drink': df['Food and drink'],
    'Online boarding': df['Online boarding'],
    'Seat comfort': df['Seat comfort'],
    'Inflight entertainment': df['Inflight entertainment'],
    'On-board service': df['On-board service']
}

data = pd.DataFrame(data)

# Select service metrics (all 0-5 rating columns except satisfaction)
service_metrics = data.columns[1:]

# Calculate average ratings by satisfaction group
avg_ratings = data.groupby('satisfaction')[service_metrics].mean().reset_index()

# Number of variables we're plotting
categories = service_metrics
N = len(categories)

# Create radar chart
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Calculate angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the loop

# Plot for each satisfaction group
for i, satisfaction in enumerate(avg_ratings['satisfaction']):
    values = avg_ratings.loc[avg_ratings['satisfaction'] == satisfaction, categories].values.flatten().tolist()
    values += values[:1]  # Complete the loop
    
    color = '#2ecc71' if satisfaction == 'satisfied' else '#e74c3c'
    linestyle = '-' if satisfaction == 'satisfied' else '--'
    
    ax.plot(angles, values, color=color, linewidth=2, linestyle=linestyle, label=satisfaction)
    ax.fill(angles, values, color=color, alpha=0.25)

# Add labels
plt.xticks(angles[:-1], categories, color='black', size=12)
ax.set_rlabel_position(30)
plt.yticks([1,2,3,4,5], ["1","2","3","4","5"], color="grey", size=10)
plt.ylim(0,5)

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Average Service Ratings by Passenger Satisfaction', size=15, y=1.1)

# Adjust layout
plt.tight_layout()
plt.show()


# Does age distribution differ between loyal and disloyal customers?

import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Create DataFrame from your data
data = pd.DataFrame({
    'Age': df['Age'],
    'Customer Type': df['Customer Type']
})

# Filter out unrealistic ages (0-100)
data = data[(data['Age'] > 0) & (data['Age'] < 100)]

# Set modern seaborn style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 6))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Overlapping Histograms
# First verify the actual customer type values in your data
print("Unique customer types:", data['Customer Type'].unique())

# Use exact values from your data in the palette
customer_palette = {
    'Loyal Customer': '#3498db', 
    'disloyal Customer': '#e74c3c'  # Note: adjust case to match your data
}

sns.histplot(
    data=data, 
    x='Age', 
    hue='Customer Type', 
    palette=customer_palette,
    bins=20, 
    kde=True, 
    alpha=0.6, 
    ax=ax1, 
    common_norm=False
)

ax1.set_title('Age Distribution by Customer Type\n(Overlapping Histograms)', fontsize=14, pad=15)
ax1.set_xlabel('Age', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.legend(title='Customer Type', title_fontsize=12)

# 2. Side-by-Side Boxplots
sns.boxplot(
    data=data, 
    x='Customer Type', 
    y='Age',
    palette=customer_palette,
    width=0.5, 
    ax=ax2, 
    showmeans=True,
    meanprops={
        "marker": "o", 
        "markerfacecolor": "white", 
        "markeredgecolor": "black"
    }
)

# Add annotations
medians = data.groupby('Customer Type')['Age'].median()
for i, customer_type in enumerate(medians.index):
    ax2.text(
        i, 
        medians.loc[customer_type] + 1, 
        f'Median: {medians.loc[customer_type]:.1f}',
        horizontalalignment='center', 
        fontsize=11, 
        weight='bold'
    )

ax2.set_title('Age Distribution by Customer Type\n(Side-by-Side Boxplots)', fontsize=14, pad=15)
ax2.set_xlabel('Customer Type', fontsize=12)
ax2.set_ylabel('Age', fontsize=12)

# Main title
plt.suptitle('Age Distribution Analysis by Customer Loyalty', y=1.02, fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


# How does departure delay affect arrival delay?

from scipy import stats

data = df.copy()
numeric_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
data = data.dropna(subset=numeric_cols)  # Remove rows with missing values

# Filter extreme delays (optional)
max_delay_threshold = 300  # Adjust as needed
data = data[(data['Departure Delay in Minutes'] <= max_delay_threshold) & 
            (data['Arrival Delay in Minutes'] <= max_delay_threshold)]

# Set style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))

# Create scatter plot with trend line
ax = sns.regplot(
    data=data,
    x='Departure Delay in Minutes',
    y='Arrival Delay in Minutes',
    scatter_kws={'alpha': 0.5, 'color': '#3498db', 's': 20},  # Added point size
    line_kws={'color': '#e74c3c', 'linewidth': 2},
    ci=95  # 95% confidence interval
)

# Calculate and display correlation - using the filtered 'data' not original 'df'
corr, p_value = stats.pearsonr(data['Departure Delay in Minutes'], 
                               data['Arrival Delay in Minutes'])
plt.text(
    0.05, 0.9, 
    f'Pearson r = {corr:.2f}\n(p-value = {p_value:.1g})',  # Improved formatting
    transform=ax.transAxes,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
    fontsize=11
)

# Add reference line for perfect correlation (y = x)
# FIXED: Get max value from both columns correctly
max_delay = max(data['Departure Delay in Minutes'].max(), 
               data['Arrival Delay in Minutes'].max())
ref_line = np.linspace(0, max_delay, 100)
plt.plot(ref_line, ref_line, '--', color='gray', alpha=0.5, label='Perfect correlation (y = x)')

# Add density contours for high-density areas
sns.kdeplot(
    data=data,
    x='Departure Delay in Minutes',
    y='Arrival Delay in Minutes',
    levels=5,
    color='black',
    alpha=0.5,
    linewidths=0.8,
    linestyles='-'
)

# Customize plot
plt.title('Impact of Departure Delays on Arrival Delays', fontsize=16, pad=20)
plt.xlabel('Departure Delay (minutes)', fontsize=12)
plt.ylabel('Arrival Delay (minutes)', fontsize=12)
plt.xlim(0, max_delay * 1.05)
plt.ylim(0, max_delay * 1.05)
plt.legend(loc='lower right')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


# Which service aspect has the strongest correlation with overall satisfaction?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DATA LOADING WITH ROBUST CLEANING
try:
    # Create dataframe with explicit numeric conversion
    data = pd.DataFrame({
        'satisfaction': df['satisfaction'],
        'Inflight wifi service': pd.to_numeric(df['Inflight wifi service'], errors='coerce'),
        'Food and drink': pd.to_numeric(df['Food and drink'], errors='coerce'),
        'Online boarding': pd.to_numeric(df['Online boarding'], errors='coerce'),
        'Seat comfort': pd.to_numeric(df['Seat comfort'], errors='coerce'),
        'Inflight entertainment': pd.to_numeric(df['Inflight entertainment'], errors='coerce'),
        'On-board service': pd.to_numeric(df['On-board service'], errors='coerce'),
        'Leg room service': pd.to_numeric(df['Leg room service'], errors='coerce')
    })
    
    # 2. SATISFACTION ENCODING WITH VALIDATION
    valid_satisfaction = {'satisfied', 'neutral or dissatisfied'}
    if not set(data['satisfaction'].unique()).issubset(valid_satisfaction):
        invalid = set(data['satisfaction'].unique()) - valid_satisfaction
        raise ValueError(f"Invalid satisfaction values: {invalid}")
    
    data['satisfaction_num'] = data['satisfaction'].map({
        'satisfied': 1, 
        'neutral or dissatisfied': 0
    })
    
    # 3. DATA SANITY CHECK
    print("Data Quality Report:")
    print(f"Original rows: {len(data)}")
    print("Missing values per column:")
    print(data.isna().sum())
    
    data_clean = data.dropna()
    print(f"\nRows after cleaning: {len(data_clean)}")
    
    if len(data_clean) < 10:
        raise ValueError("Insufficient data after cleaning (min 10 rows required)")
    
    # 4. CORRELATION MATRIX WITH SAFETY CHECKS
    numeric_cols = data_clean.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = data_clean[numeric_cols].corr()
    
    if corr_matrix.isna().any().any():
        raise ValueError("NaN values in correlation matrix - check input data")
    
    # 5. VISUALIZATION WITH ERROR-FREE COLORMAPPING
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Correlation Coefficient'},
        mask=np.triu(np.ones_like(corr_matrix))  # Mask upper triangle
    )
    
    # Highlight the satisfaction row and strongest correlation
    satisfaction_index = corr_matrix.columns.get_loc('satisfaction_num')
    strongest_corr = corr_matrix['satisfaction_num'].abs().nlargest(2).iloc[1]
    strongest_index = corr_matrix.columns.get_loc(strongest_corr.name)
    
    ax.add_patch(plt.Rectangle((0, satisfaction_index), len(corr_matrix), 1, 
                             fill=False, edgecolor='gold', lw=3))
    ax.add_patch(plt.Rectangle((strongest_index, satisfaction_index), 1, 1,
                             fill=False, edgecolor='lime', lw=3))
    
    plt.title('Service Rating Correlations with Passenger Satisfaction\n' +
             f"Strongest: {strongest_corr.name} (r={strongest_corr:.2f})", 
             pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure to avoid display issues
    plt.savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    from IPython.display import Image
    display(Image(filename='correlation_heatmap.png'))
    
    print("\nHeatmap generated successfully!")
    print(f"Strongest correlation with satisfaction: {strongest_corr.name} (r = {strongest_corr:.2f})")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("\nTROUBLESHOOTING GUIDE:")
    print("1. Check for missing values: df.isna().sum()")
    print("2. Verify satisfaction values: df['satisfaction'].unique()")
    print("3. Inspect rating distributions:")
    print(df[['Inflight wifi service', 'Food and drink']].describe())
    
    if 'data' in locals():
        print("\nCLEANED DATA STATS:")
        print(data_clean.describe())


# How does satisfaction vary by travel purpose (business/personal)?

# Create DataFrame
data = pd.DataFrame({
    'Type of Travel': df['Type of Travel'],
    'satisfaction': df['satisfaction']
})

# Set style
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(14, 6))

# Get unique travel types (sorted Business first)
travel_types = sorted(data['Type of Travel'].unique(), reverse=True)

# Create gridspec for better layout control
gs = fig.add_gridspec(1, len(travel_types) + 1, width_ratios=[1]*len(travel_types) + [0.2])

# Custom colors
colors = {
    'satisfied': '#2ecc71',  # Green
    'neutral or dissatisfied': '#e74c3c'  # Red
}

# Create subplots
axes = []
for i in range(len(travel_types)):
    axes.append(fig.add_subplot(gs[0, i]))

for i, travel_type in enumerate(travel_types):
    # Filter and calculate
    subset = data[data['Type of Travel'] == travel_type]
    counts = subset['satisfaction'].value_counts()
    percentages = counts / counts.sum() * 100
    
    # Create donut chart
    wedges, _, autotexts = axes[i].pie(
        percentages,
        colors=[colors[x] for x in percentages.index],
        autopct=lambda p: f'{p:.1f}%',
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
        pctdistance=0.85
    )
    
    # Add center text
    centre_circle = plt.Circle((0,0), 0.3, color='white')
    axes[i].add_artist(centre_circle)
    axes[i].text(0, 0, f"{percentages.get('satisfied', 0):.1f}%", 
                ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Add title
    axes[i].set_title(f'{travel_type} Travel\n(n={len(subset):,})', 
                     pad=20, fontsize=14, fontweight='semibold')

# Add legend in the dedicated grid space
legend_ax = fig.add_subplot(gs[0, -1])
legend_ax.axis('off')
legend_elements = [
    plt.Rectangle((0,0),1,1, fc=colors['satisfied']),
    plt.Rectangle((0,0),1,1, fc=colors['neutral or dissatisfied'])
]
legend = legend_ax.legend(
    legend_elements,
    ['Satisfied', 'Neutral/Dissatisfied'],
    title='Satisfaction Level',
    loc='center',
    frameon=False
)
plt.setp(legend.get_title(), fontweight='bold')

# Main title
fig.suptitle('Passenger Satisfaction by Travel Purpose', 
             y=1.05, fontsize=16, fontweight='bold')

# No tight_layout needed due to gridspec
plt.show()


# What's the distribution of service ratings across different demographics?

# Load and prepare data
data = pd.DataFrame({
    'Gender': df['Gender'],
    'Customer Type': df['Customer Type'],
    'Class': df['Class'],
    'Inflight WiFi': df['Inflight wifi service'],
    'Food/Drink': df['Food and drink'],
    'Seat Comfort': df['Seat comfort'],
    'Inflight Entertainment': df['Inflight entertainment']
})

# Clean and validate data
numeric_cols = ['Inflight WiFi', 'Food/Drink', 'Seat Comfort', 'Inflight Entertainment']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Melt data for faceting
melted = data.melt(
    id_vars=['Gender', 'Customer Type', 'Class'],
    value_vars=numeric_cols,
    var_name='Service',
    value_name='Rating'
)

# Set modern seaborn style (fixes deprecation warning)
sns.set_theme(style="whitegrid", font_scale=0.9)
palette = {'Male': '#3498db', 'Female': '#e74c3c'}

# Create figure with GridSpec for perfect layout
fig = plt.figure(figsize=(14, 10), layout="constrained")

# Create facet grid with layout control
g = sns.FacetGrid(
    melted,
    row='Service',
    col='Customer Type',
    hue='Gender',
    palette=palette,
    height=3,
    aspect=1.3,
    margin_titles=True,
    despine=False
)

# Map barplots with improved formatting
g.map(
    sns.barplot,
    'Class',
    'Rating',
    order=['Business', 'Eco Plus', 'Eco'],
    errorbar='sd',  # Show standard deviation
    alpha=0.85,
    linewidth=1,
    edgecolor='w'
)

# Customize labels and titles
g.set_titles(col_template="{col_name} Customers", row_template="{row_name}")
g.set_axis_labels("Travel Class", "Average Rating (1-5)")
g.set(ylim=(0, 5.5))  # Consistent y-axis across all plots

# Add legend outside
g.add_legend(
    title='Gender',
    bbox_to_anchor=(1.05, 0.5),
    frameon=True,
    edgecolor='none',
    title_fontsize='11'
)

# Add value labels with improved positioning
for ax in g.axes.flat:
    ax.grid(True, axis='y', alpha=0.3)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}",
            (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

# Main title with proper positioning
fig.suptitle('Service Ratings by Customer Type, Travel Class and Gender\n',
             y=1.02, fontsize=14, fontweight='bold')

plt.show()


# Which combination of factors best predicts customer loyalty?

from sklearn.tree import plot_tree

# Prepare data
df_loyalty = df[['satisfaction', 'Type of Travel', 'Class', 'Flight Distance', 
                'Inflight wifi service', 'Customer Type']].copy()

# Encode categorical variables
le = LabelEncoder()
cat_cols = ['satisfaction', 'Type of Travel', 'Class', 'Customer Type']
for col in cat_cols:
    df_loyalty[col] = le.fit_transform(df_loyalty[col])

# Build decision tree
X = df_loyalty.drop('Customer Type', axis=1)
y = df_loyalty['Customer Type']
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50)
model.fit(X, y)

# Visualize
plt.figure(figsize=(20,10))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=['Disloyal', 'Loyal'], 
          filled=True,
          proportion=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Paths to Customer Loyalty", pad=20, fontsize=14)
plt.show()

