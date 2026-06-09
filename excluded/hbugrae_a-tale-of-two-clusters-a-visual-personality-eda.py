import pandas as pd 
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)

test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


# Display concise information about the datasets
print("--- Training Set Information ---")
train.info()
print("\n" + "="*60 + "\n")
print("--- Test Set Information ---")
test.info()


# Display descriptive statistics for the numerical columns
print("--- Training Set Descriptive Statistics ---")
print(train.describe())
print("\n" + "="*114 + "\n")
print("--- Test Set Descriptive Statistics ---")
print(test.describe())


# Analyze the distribution of the target variable
print("--- Target Variable (Personality) Distribution ---")
print(train['Personality'].value_counts())
print("\n" + "="*50 + "\n")
print("--- Target Variable (Personality) Proportional Distribution ---")
print(train['Personality'].value_counts(normalize=True))


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Select only numerical columns (excluding 'id')
numerical_features = train.select_dtypes(include=['float64', 'int64']).drop('id', axis=1).columns

# Set up the figure and axes
plt.figure(figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features (Histograms)', fontsize=16)

# Plot a histogram for each numerical feature
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    # kde=True adds a Kernel Density Estimate curve for a smoother distribution shape
    sns.histplot(data=train, x=col, kde=True, bins=20) 
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Set up the figure and axes
plt.figure(figsize=(18, 12))
plt.suptitle('Distribution of Numerical Features by Personality Type (Box Plots)', fontsize=16)

# Plot a box plot for each numerical feature
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    # Set the order of the x-axis categories for consistent visualization
    sns.boxplot(data=train, x='Personality', y=col, order=['Introvert', 'Extrovert']) 
    plt.title(col)
    plt.xlabel('Personality')
    plt.ylabel('Value')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Define the categorical features to be analyzed
categorical_features = ['Stage_fear', 'Drained_after_socializing']

# Set up the figure and axes
plt.figure(figsize=(12, 5))
plt.suptitle('Distribution of Categorical Features by Personality Type', fontsize=16)

# Plot a count plot for each categorical feature
for i, col in enumerate(categorical_features):
    plt.subplot(1, 2, i + 1)
    # Use hue='Personality' to create separate bars for each class
    sns.countplot(data=train, x=col, hue='Personality', order=['No', 'Yes'], hue_order=['Introvert', 'Extrovert'])
    plt.title(col)
    plt.xlabel('Response')
    plt.ylabel('Count of Individuals')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()


# Calculate the correlation matrix for numerical features
correlation_matrix = train[numerical_features].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
# annot=True: Displays the correlation values on the map
# cmap='coolwarm': Sets the color palette (warm for positive, cool for negative correlations)
# fmt='.2f': Formats the numbers to two decimal places
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)


plt.title('Correlation Matrix of Numerical Features', fontsize=15)
plt.show()


# Create a Pair Plot for a comprehensive overview
# The 'hue' argument colors the data points by personality type.
# corner=True plots only the lower triangle of the matrix to avoid redundancy and improve performance.
print("Generating Pair Plot...")
sns.pairplot(train.drop('id', axis=1), hue='Personality', corner=True, palette={'Introvert': 'blue', 'Extrovert': 'red'})
plt.suptitle('Pair Plot of Features by Personality Type', y=1.02) 
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import numpy as np

# Create a copy of the original train set to avoid modifying it
train_processed = train.copy()

# --- 1. Data Preprocessing ---

# Convert categorical features to numerical format
train_processed['Stage_fear'] = train_processed['Stage_fear'].map({'No': 0, 'Yes': 1})
train_processed['Drained_after_socializing'] = train_processed['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

# Impute missing values
# Use median for numerical, mode for categorical
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Separate column types
num_cols = train_processed.select_dtypes(include=np.number).drop(['id'], axis=1).columns
cat_cols = ['Stage_fear', 'Drained_after_socializing']

train_processed[num_cols] = num_imputer.fit_transform(train_processed[num_cols])
train_processed[cat_cols] = cat_imputer.fit_transform(train_processed[cat_cols])

# Define features (X) and target (y)
X = train_processed.drop(['id', 'Personality'], axis=1)
y = train['Personality'] # Get the original labels

# Scale the data (important for distance-based algorithms like t-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- 2. Applying the t-SNE Model ---
print("Calculating t-SNE...")
# t-SNE is used to visualize high-dimensional data in 2D or 3D
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
print("Calculation complete.")

# --- 3. Visualizing the Results ---
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE_1', 'TSNE_2'])
tsne_df['Personality'] = y

plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="TSNE_1", y="TSNE_2",
    hue="Personality",
    palette={'Introvert': 'blue', 'Extrovert': 'red'},
    data=tsne_df,
    legend="full",
    alpha=0.6
)
plt.title('t-SNE Visualization of All Features', fontsize=16)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# --- 1. Data Preprocessing & PCA Application ---
train_processed = train.copy()
train_processed['Stage_fear'] = train_processed['Stage_fear'].map({'No': 0, 'Yes': 1})
train_processed['Drained_after_socializing'] = train_processed['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_cols = train_processed.select_dtypes(include=np.number).drop(['id'], axis=1).columns
cat_cols = ['Stage_fear', 'Drained_after_socializing']

train_processed[num_cols] = num_imputer.fit_transform(train_processed[num_cols])
train_processed[cat_cols] = cat_imputer.fit_transform(train_processed[cat_cols])

X = train_processed.drop(['id', 'Personality'], axis=1)
y = train['Personality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce data to 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['Personality'] = y



# --- 2. Create and Save the Animation ---
print("\nCreating the animation and saving as a GIF...")

fig_anim = plt.figure(figsize=(12, 10))
ax_anim = fig_anim.add_subplot(111, projection='3d')

colors = {'Introvert': 'blue', 'Extrovert': 'red'}
ax_anim.scatter(
    pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
    c=pca_df['Personality'].map(colors), alpha=0.6
)

ax_anim.set_xlabel('Principal Component 1 (PC1)', fontsize=12)
ax_anim.set_ylabel('Principal Component 2 (PC2)', fontsize=12)
ax_anim.set_zlabel('Principal Component 3 (PC3)', fontsize=12)
ax_anim.set_title('3D PCA Visualization (Animation)', fontsize=16)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Extrovert', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Introvert', markerfacecolor='blue', markersize=10)
]
ax_anim.legend(handles=legend_elements, title="Personality")

# Rotation function for animation
def rotate(angle):
    ax_anim.view_init(elev=30, azim=angle)

ani = animation.FuncAnimation(fig_anim, rotate, frames=np.arange(0, 360, 30), interval=500)

ani.save('pca_3d_animation_final.gif', writer='pillow', fps=4)
plt.close(fig_anim)

print("Animation successfully saved as 'pca_3d_animation_final.gif'.")




