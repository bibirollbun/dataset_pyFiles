import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.info()


def clean_data(df):
    # Handle boolean columns
    bool_cols = ['Drained_after_socializing']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0, True: 1, False: 0})
    
    # Convert infinity values to NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


train = clean_data(train)
test = clean_data(test)


# Exclude non-feature columns
exclude_cols = ['id', 'Personality', 'target_num']  # Remove any non-feature columns
features = [col for col in train.columns if col not in exclude_cols]

# Select only numerical features
numerical_features = train[features].select_dtypes(include=['number']).columns.tolist()

print(f"Numerical features to plot: {numerical_features}")


n_cols = 3
n_rows = (len(numerical_features) + n_cols - 1) // n_cols  # Round up division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten()  # Flatten for easy iteration

for i, col in enumerate(numerical_features):
    try:
        # Plot train distribution
        sns.kdeplot(train[col].dropna(), ax=axes[i], 
                   color='#3498db', fill=True, label='Train')
        
        # Plot test distribution
        sns.kdeplot(test[col].dropna(), ax=axes[i], 
                   color='#e74c3c', fill=True, alpha=0.5, label='Test')
        
        # Customize plot
        axes[i].set_title(f'{col} Distribution', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].legend()
        
        # Remove top and right spines
        sns.despine(ax=axes[i])
        
    except Exception as e:
        print(f"Could not plot {col}: {str(e)}")
        axes[i].set_visible(False)

# Hide any empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.suptitle('Numerical Feature Distributions: Train vs Test', 
             fontsize=16, y=1.02)
plt.show()




categorical_features = train[features].select_dtypes(exclude=['number']).columns.tolist()

if categorical_features:
    print(f"\nCategorical features found: {categorical_features}")
    # Similar plotting logic for categorical features can be added here
else:
    print("\nNo categorical features found.")


plt.figure(figsize=(10,6))
ax = sns.countplot(x='Personality', data=train, palette='viridis')
plt.title('Personality Distribution (Extrovert 74% vs Introvert 26%)', fontsize=16)
for p in ax.patches:
    ax.annotate(f'{p.get_height()/len(train)*100:.1f}%', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0,10), 
                textcoords='offset points', fontsize=12)
plt.show()


plt.figure(figsize=(12,6))
missing = train.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
sns.barplot(x=missing.values, y=missing.index, palette='rocket')
plt.title('Missing Values Analysis', fontsize=16)
plt.xlabel('Missing Count')
plt.show()


train['target_num'] = train['Personality'].map({'Extrovert':1, 'Introvert':0})



# Select only numerical columns for correlation
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns

# Calculate correlation matrix only for numerical columns
corr = train[numerical_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up matplotlib figure
plt.figure(figsize=(16,12))

# Generate heatmap
sns.heatmap(corr, 
            mask=mask, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0, 
            linewidths=0.5,
            vmin=-1, vmax=1)

plt.title('Numerical Features Correlation Matrix', fontsize=20)
plt.show()


# Check missing values
print(train.isnull().sum())

# Histograms for numeric features
numeric_cols = ['Time_spent_Alone','Stage_fear','Going_outside','Friends_circle_size','Post_frequency']
train[numeric_cols].hist(bins=20, figsize=(12,8))
plt.suptitle("Histograms of Numeric Features")
plt.show()





# Countplots for boolean/categorical features
fig, axs = plt.subplots(1,2, figsize=(10,4))
sns.countplot(x='Social_event_attendance', data=train, ax=axs[0])
sns.countplot(x='Drained_after_socializing', data=train, ax=axs[1])
axs[0].set_title('Social Event Attendance')
axs[1].set_title('Drained After Socializing')
plt.show()


# Boxplot: Time_spent_Alone by Personality
sns.boxplot(x='Personality', y='Time_spent_Alone', data=train)
plt.title("Time Spent Alone by Personality")
plt.show()


# Boxplot: Friends_circle_size by Personality
sns.boxplot(x='Personality', y='Friends_circle_size', data=train)
plt.title("Friends Circle Size by Personality")
plt.show()




