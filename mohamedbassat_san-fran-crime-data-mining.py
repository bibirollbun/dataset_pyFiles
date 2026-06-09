!pip install scikit-learn-extra
!pip install -U imbalanced-learn
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score




data_dir = '/kaggle/input/sf-crime'

df = pd.read_csv(os.path.join(data_dir, 'train.csv.zip'))


df.shape


df.head()


df.info()




df.Dates = pd.to_datetime(df.Dates)
# Extract components 
df['Year'] = df.Dates.dt.year
df['Month'] = df.Dates.dt.month
df['Hour'] = df.Dates.dt.hour

# === Year and Month Bar Plots ===
fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# Crime by Year
year_counts = df['Year'].value_counts().sort_index()
sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax[0])
ax[0].set_title('Number of Crimes by Year')
ax[0].tick_params(axis='x', labelrotation=90)

# Crime by Month
month_counts = df['Month'].value_counts().sort_index()
sns.barplot(x=month_counts.index, y=month_counts.values, ax=ax[1])
ax[1].set_title('Number of Crimes by Month')
ax[1].tick_params(axis='x', labelrotation=90)

plt.show()

# === Hour Bar Plot ===
hour_counts = df['Hour'].value_counts().sort_index()
plt.figure(figsize=(8, 4))
sns.barplot(x=hour_counts.index, y=hour_counts.values)
plt.title('Number of Crimes by Hour')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# === Day of Week Plot ===
dow_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
dow_counts = df['DayOfWeek'].value_counts().reindex(dow_order)
plt.figure(figsize=(8, 4))
sns.barplot(x=dow_counts.index, y=dow_counts.values)
plt.title('Number of Crimes by Day of Week')
plt.ylim([100000, 140000])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



# Precompute the counts
district_counts = df.PdDistrict.value_counts()
print(district_counts) 

fig = plt.figure(figsize=(8, 4))
sns.barplot(
    x=district_counts.index,
    y=district_counts.values,
)
plt.title('Number of Crimes by Police District')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



df.Descript


df.Resolution.value_counts()



import pandas as pd
import matplotlib.pyplot as plt

# Count the occurrences of each resolution
resolution_counts = df['Resolution'].value_counts()

# Define a threshold: only keep resolutions with more than X cases
threshold = 10000  # You can adjust this value
top_resolutions = resolution_counts[resolution_counts > threshold]

# Add "Other" category for the rest
other_count = resolution_counts[resolution_counts <= threshold].sum()
top_resolutions["Other"] = other_count

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_resolutions, labels=top_resolutions.index, autopct='%1.1f%%', startangle=140)
plt.title('Resolutions of Reported Cases (Grouped)')
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.tight_layout()
plt.show()



# Check for duplicate rows 
num_dupes = df.duplicated().sum()
print(f"Number of duplicate rows: {num_dupes}")



#drop them 
df = df.drop_duplicates()


print("New shape:", df.shape)


df.isnull().sum()


((df == 'none') | (df == 'nan')).sum()




numeric_cols = ['X', 'Y']

for col in numeric_cols:
    # 1. Compute Q1 and Q3
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # 2. Compute IQR
    IQR = Q3 - Q1

    # 3. Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Report
    print(f"{col}: Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
    print(f"    Outliers < {lower_bound:.3f} or > {upper_bound:.3f}\n")

    # 4. Flag outliers
    is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
    print(f"    Found {is_outlier.sum()} outliers in '{col}'\n")

    # 5. Boxplot
    plt.figure(figsize=(6, 2))
    plt.boxplot(df[col].dropna(), whis=1.5, vert=False)
    plt.title(f"Boxplot of {col} (whis=1.5·IQR)")
    plt.xlabel(col)
    plt.show()



# 1. Compute Q1, Q3 and IQR for X and Y
Q1 = df[['X', 'Y']].quantile(0.25)
Q3 = df[['X', 'Y']].quantile(0.75)
IQR = Q3 - Q1

# 2. Compute the 1.5·IQR bounds
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 3. Drop rows where X or Y are outside their bounds
mask = (
    df['X'].between(lower['X'], upper['X']) &
    df['Y'].between(lower['Y'], upper['Y'])
)
df = df[mask]  # Apply the mask to filter out outliers

# 4. Sanity check
print("New df shape:", df.shape)
print("X: min =", df['X'].min(), "max =", df['X'].max())
print("Y: min =", df['Y'].min(), "max =", df['Y'].max())
print("Any X outliers left?", not df['X'].between(lower['X'], upper['X']).all())
print("Any Y outliers left?", not df['Y'].between(lower['Y'], upper['Y']).all())


df.describe()


df.head()


df.info()


def feature_engineering(df):
    
    # Ensure Dates is datetime
    if 'Dates' in df.columns:
        df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
        # Create date features
        #df['n_days']     = (df['Dates'] - df['Dates'].min()).dt.days
        df['Day']        = df['Dates'].dt.day
        df['DayOfWeek']  = df['Dates'].dt.weekday
        df['Month']      = df['Dates'].dt.month
        df['Year']       = df['Dates'].dt.year
        df['Hour']       = df['Dates'].dt.hour
        df['Minute']     = df['Dates'].dt.minute
        df.drop(columns=['Dates'], inplace=True)
    else:
        print("Warning: 'Dates' column not found. Skipping date features.")

    # Address feature
    if 'Address' in df.columns:
        df['Block'] = df['Address'].str.contains('block', case=False).astype(int)
        df.drop(columns=['Address'], inplace=True)
    else:
        print("Warning: 'Address' column not found. Skipping address features.")

    # Drop text columns
    for col in ['Descript', 'Resolution']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df

df=feature_engineering(df)





df.info()


# Select only numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numeric data
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# View result
print(df.head())



# Initialize label encoders
category_encoder = LabelEncoder()
district_encoder = LabelEncoder()

# Apply label encoding to the two object columns
df['Category'] = category_encoder.fit_transform(df['Category'])
df['PdDistrict'] = district_encoder.fit_transform(df['PdDistrict'])


def cluster_data(df, features=None, k_range=range(2, 11), max_samples=10000, 
                random_state=42, verbose=True):
    # Select features for clustering
    if features is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = numeric_cols
        if verbose:
            print(f"Using all numeric features: {features}")
    else:
        # Verify all specified features exist
        missing = [col for col in features if col not in df.columns]
        if missing:
            print(f"Warning: Features {missing} not found in dataframe. Using available features only.")
            features = [col for col in features if col in df.columns]
            
    if not features:
        print("Error: No valid features found for clustering.")
        return df, None, None
    
# Prepare data
    data = df[features].copy()
    
    
    
    
    # Sample if dataset is large
    sample_size = min(len(df), max_samples)
    samples = df.sample(n=sample_size, random_state=random_state)
    
    # Evaluate different k values
    scores = []
    labels_dict = {}
    
    if verbose:
        print(f"Evaluating {len(k_range)} different cluster counts...")
    
    for k in k_range:
        if verbose:
            print(f"Testing k={k}...", end=" ")
        
        kmedoids = KMedoids(n_clusters=k, random_state=random_state, metric='euclidean')
        labels = kmedoids.fit_predict(samples)
        labels_dict[k] = labels
        medoids_dict = {k: kmedoids} if k > 1 else {k: None}
        
        # Calculate silhouette score (only if we have more than 1 cluster)
        if k > 1:
            score = silhouette_score(samples, labels)
            scores.append(score)
            if verbose:
                print(f"Silhouette score: {score:.4f}")
        else:
            scores.append(0)
            if verbose:
                print("Skipped silhouette score (k=1)")
    
    # Find optimal k (skip k=1 if it's in the range)
    valid_scores = scores.copy()
    if 1 in k_range:
        valid_scores[k_range.index(1)] = -1  # Replace score for k=1 with -1
        
    best_k = k_range[valid_scores.index(max(valid_scores))]
    
    # Apply best clustering
    if verbose:
        print(f"\nOptimal number of clusters: {best_k} (Silhouette score: {max(valid_scores):.4f})")
    
    # Add cluster labels to original dataframe for the samples that were used
    df_clustered = df.copy()
    df_clustered.loc[samples.index, 'Cluster'] = labels_dict[best_k]
    
    # Get the best model
    best_kmedoids = KMedoids(n_clusters=best_k, random_state=random_state).fit(samples)
    
    # Handle any rows that weren't in the sample
    if len(df_clustered) > sample_size:
        # Assign remaining points to nearest medoid center
        remaining_indices = df_clustered.index.difference(samples.index)
        remaining_data = df.loc[remaining_indices]
        
        # Predict clusters for remaining points
        remaining_labels = best_kmedoids.predict(remaining_data)
        df_clustered.loc[remaining_indices, 'Cluster'] = remaining_labels
    
    return df_clustered, best_k, scores




def plot_silhouette_scores(k_range, scores):
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'o-', markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs. Number of Clusters', fontsize=14)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.show()




def visualize_clusters_2d(df, x_col, y_col, cluster_col='Cluster', figsize=(12, 10)):

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        df[x_col], 
        df[y_col],
        c=df[cluster_col], 
        cmap='tab10', 
        alpha=0.7,
        s=30
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'KMedoids Clustering Results (k={df[cluster_col].nunique()})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()





def analyze_clusters(df, cluster_col='Cluster', features=None):
  
    # Select features for analysis
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != cluster_col]
    
    # Calculate cluster statistics
    print(f"Cluster distribution:")
    print(df[cluster_col].value_counts().sort_index())
    print("\nCluster characteristics (means):")
    
    # Group by cluster and calculate means
    means = df.groupby(cluster_col)[features].mean()
    return means


# Example usage:
X = df.drop(columns=['Category'])
Y = df['Category']

# For all numeric features:
df_clustered, best_k, scores = cluster_data(X)

# For specific features:
#selected_features = ['X', 'Y', 'feature1', 'feature2', ...]
#df_clustered, best_k, scores = cluster_data(df, features=selected_features)

# Plot silhouette scores
plot_silhouette_scores(range(2, 11), scores)

# Visualize clusters in 2D
visualize_clusters_2d(df_clustered, 'X', 'Y')

# Analyze cluster characteristics
cluster_means = analyze_clusters(df_clustered)
print(cluster_means)

df_clustered.info()


df_clustered['Category'] = df['Category']

df=df_clustered


def apply_smote(df, target_column='Category', random_state=42):
   
    from imblearn.over_sampling import SMOTE
    import pandas as pd
    
    # Check if SMOTE is installed
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("Please install imbalanced-learn: pip install imbalanced-learn")
        return None, None
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Display original class distribution
    print("Original class distribution:")
    print(y.value_counts())
    print(f"Original shape: {X.shape}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Convert back to DataFrame/Series to maintain column names
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=target_column)
    
    # Display new class distribution
    print("\nBalanced class distribution after SMOTE:")
    print(y_resampled.value_counts())
    print(f"New shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled
X_balanced, y_balanced = apply_smote(df, target_column='Category')

# If you want to get a complete balanced DataFrame
df = pd.concat([X_balanced, y_balanced], axis=1)


df.info()


#Histograms of X (Longitude) and Y (Latitude)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.histplot(df['X'], bins=50, ax=ax[0])
ax[0].set_title('Distribution of Longitude (X)')
sns.histplot(df['Y'], bins=50, ax=ax[1])
ax[1].set_title('Distribution of Latitude (Y)')
plt.tight_layout()
plt.show()





df.info()


# 1. Select only numeric columns
numeric_cols = df.select_dtypes(include='number')

# 2. Compute correlation matrix
corr = numeric_cols.corr()

# 3. Plot heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,        # show correlation coefficients
    fmt=".2f",         # 2 decimal places
    cmap="coolwarm",   # diverging color palette
    square=True,       # make cells square
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Matrix of Numeric Features (df)")
plt.tight_layout()
plt.show()



df_sampled = df.sample(1000000, random_state=42)

# Separate features and target
X = df_sampled.drop(columns=['Category'])
y = df_sampled['Category']
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42, stratify=y)


# import optuna
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score

# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 150),
#         'max_depth': trial.suggest_int('max_depth', 5, 20),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
#     }

#     model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
#     score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
#     return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20)  # Keep trials low to avoid OOM

# print("Best parameters:", study.best_params)



# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', s=30, alpha=0.7)
plt.title('PCA Projection of Crime Categories')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



print("Explained variance ratio:", pca.explained_variance_ratio_)



X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.05, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42,stratify=y)

n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
max_components = min(n_features, n_classes - 1)

print("Maximum allowed LDA components:", max_components)

lda = LinearDiscriminantAnalysis(n_components=max_components)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test_lda)
print(classification_report(y_test, y_pred))



from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train_lda, y_train)

y_pred = model.predict(X_test_lda)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))





