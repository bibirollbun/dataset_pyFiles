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


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, mutual_info_score
from scipy.stats import spearmanr
import zipfile
import os

os.makedirs("/kaggle/working/telstra-data", exist_ok=True)

def unzip_csv_files():
    """
    Unzips specified CSV files from the input directory to the working directory.
    """
    # List of zip files to be extracted
    zip_files = [
        'train.csv.zip',
        'test.csv.zip',
        'event_type.csv.zip',
        'log_feature.csv.zip',
        'resource_type.csv.zip',
        'severity_type.csv.zip',
        'sample_submission.csv.zip'
    ]
    
    # Iterate over each zip file and extract its contents
    for zip_file in zip_files:
        input_zip_path = f"/kaggle/input/telstra-recruiting-network/{zip_file}"  # Path to input zip file
        output_dir = "/kaggle/working/telstra-data"  # Directory where files will be extracted
        
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)  # Extract all contents of the zip file to the output directory



print("Unzipping files...")
unzip_csv_files()  # Call the function to unzip files
print("All files unzipped successfully!")




data_dir = "/kaggle/working/telstra-data"

# Read each CSV file into a pandas DataFrame
train         = pd.read_csv(f"{data_dir}/train.csv")
test          = pd.read_csv(f"{data_dir}/test.csv")
event_type    = pd.read_csv(f"{data_dir}/event_type.csv")
log_feature   = pd.read_csv(f"{data_dir}/log_feature.csv")
resource_type = pd.read_csv(f"{data_dir}/resource_type.csv")
severity_type = pd.read_csv(f"{data_dir}/severity_type.csv")



def create_time_features(log_feature):
    """
    Creates aggregated time-based features from the log_feature DataFrame.
    
    Parameters:
    - log_feature (pd.DataFrame): DataFrame containing log feature data.
    
    Returns:
    - pd.DataFrame: DataFrame with aggregated time-based features.
    """
    # Sort log_feature DataFrame by 'id' to ensure chronological order
    log_feature = log_feature.sort_values(["id"])
    
    # Aggregate count and number of unique events per 'id'
    time_stats = (
        log_feature.groupby("id").agg({"log_feature": ["count", "nunique"]}).fillna(0)
    )
    time_stats.columns = ["event_count", "unique_events"]  # Rename columns for clarity
    
    # Calculate event density as the ratio of event_count to unique_events
    time_stats["event_density"] = time_stats["event_count"] / (
        time_stats["unique_events"] + 1  # Add 1 to avoid division by zero
    )
    
    # Create a list of events per 'id'
    event_sequences = log_feature.groupby("id")["log_feature"].apply(list)
    
    # Calculate the number of event changes in the sequence for each 'id'
    event_changes = event_sequences.apply(
        lambda x: sum(1 for i in range(len(x) - 1) if x[i] != x[i + 1])
    )
    
    # Add event_changes and change_ratio to time_stats
    time_stats["event_changes"] = event_changes
    time_stats["change_ratio"] = time_stats["event_changes"] / (
        time_stats["event_count"] + 1  # Add 1 to avoid division by zero
    )
    
    return time_stats  # Return the aggregated time-based features


# One-hot encode 'event_type' and aggregate by 'id'
event_type_pivot = pd.get_dummies(event_type["event_type"])
event_type_agg = event_type_pivot.set_index(event_type["id"]).groupby("id").sum()
event_type_agg.columns = [f"event_type_{i}" for i in range(len(event_type_agg.columns))]  # Rename columns

# One-hot encode 'log_feature' and aggregate by 'id'
log_feature_pivot = pd.get_dummies(log_feature["log_feature"])
log_feature_agg = log_feature_pivot.set_index(log_feature["id"]).groupby("id").sum()
log_feature_agg.columns = [
    f"log_feature_{i}" for i in range(len(log_feature_agg.columns))
]  # Rename columns

# Aggregate statistical features from 'log_feature' by 'id'
volume_stats = (
    log_feature.groupby("id")["volume"]
    .agg(["mean", "std", "min", "max", "sum"])
    .fillna(0)  # Fill NaN values with 0
)

# Create time-based aggregated features
time_stats = create_time_features(log_feature)

# One-hot encode 'resource_type' and aggregate by 'id'
resource_type_pivot = pd.get_dummies(resource_type["resource_type"])
resource_type_agg = (
    resource_type_pivot.set_index(resource_type["id"]).groupby("id").sum()
)
resource_type_agg.columns = [
    f"resource_type_{i}" for i in range(len(resource_type_agg.columns))
]  # Rename columns

# One-hot encode 'severity_type' and aggregate by 'id'
severity_type_pivot = pd.get_dummies(severity_type["severity_type"])
severity_type_agg = (
    severity_type_pivot.set_index(severity_type["id"]).groupby("id").sum()
)
severity_type_agg.columns = [
    f"severity_type_{i}" for i in range(len(severity_type_agg.columns))
]  # Rename columns



# Initialize LabelEncoder for the 'location' feature
le = LabelEncoder()

# Concatenate 'location' from both train and test to fit the encoder
all_locations = pd.concat([train["location"], test["location"]])

# Fit the LabelEncoder on the combined 'location' data
le.fit(all_locations)



# Create a temporary DataFrame by concatenating event and resource type aggregates
temp_train = pd.concat(
    [
        event_type_agg.reindex(train["id"], fill_value=0),  # Align with training 'id's
        resource_type_agg.reindex(train["id"], fill_value=0),  # Align with training 'id's
    ],
    axis=1,
)

# Add the target variable 'fault_severity' to the temporary DataFrame
temp_train["fault_severity"] = train["fault_severity"].values

# Identify event and resource type columns
event_cols = [col for col in temp_train.columns if col.startswith("event_type_")]
resource_cols = [col for col in temp_train.columns if col.startswith("resource_type_")]

def get_mi_score(feature, target):
    """
    Calculates the mutual information score between a feature and the target.
    
    Parameters:
    - feature (array-like): Feature values.
    - target (array-like): Target variable.
    
    Returns:
    - float: Mutual information score.
    """
    return mutual_info_score(feature, target)

# Calculate mutual information scores for event type features
event_mi = [
    (col, get_mi_score(temp_train[col], temp_train["fault_severity"]))
    for col in event_cols
]

# Calculate mutual information scores for resource type features
resource_mi = [
    (col, get_mi_score(temp_train[col], temp_train["fault_severity"]))
    for col in resource_cols
]

# Select features with mutual information scores above the median
event_median_mi = np.median([score for _, score in event_mi])
resource_median_mi = np.median([score for _, score in resource_mi])

# List of selected event type features
selected_event_features = [col for col, score in event_mi if score > event_median_mi]

# List of selected resource type features
selected_resource_features = [
    col for col, score in resource_mi if score > resource_median_mi
]


def prepare_data(df):
    """
    Prepares the data by encoding categorical features and merging aggregated features.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame (train or test).
    
    Returns:
    - pd.DataFrame: Prepared feature set ready for modeling.
    """
    df_copy = df.copy()
    
    # Encode 'location' feature using the previously fitted LabelEncoder
    df_copy["location_encoded"] = le.transform(df_copy["location"])
    
    # Set 'id' as the index for merging with aggregated features
    df_copy = df_copy.set_index("id")
    
    # Drop the original 'location' column as it's now encoded
    df_copy = df_copy.drop("location", axis=1)

    # List of DataFrames containing different feature sets to be merged
    feature_dfs = [
        df_copy,
        event_type_agg.reindex(df_copy.index, fill_value=0),      # Merge event type features
        log_feature_agg.reindex(df_copy.index, fill_value=0),     # Merge log feature counts
        volume_stats.reindex(df_copy.index, fill_value=0),        # Merge volume statistics
        resource_type_agg.reindex(df_copy.index, fill_value=0),   # Merge resource type features
        severity_type_agg.reindex(df_copy.index, fill_value=0),   # Merge severity type features
        time_stats.reindex(df_copy.index, fill_value=0),          # Merge time-based features
    ]

    # Initialize a dictionary to store interaction features
    interaction_features = {}
    
    # Create interaction features between selected event and resource type features
    for event_feat in selected_event_features:
        event_values = event_type_agg[event_feat].reindex(df_copy.index, fill_value=0)
        for resource_feat in selected_resource_features:
            resource_values = resource_type_agg[resource_feat].reindex(df_copy.index, fill_value=0)
            # Interaction feature is the product of event and resource feature values
            interaction_features[f"interaction_{event_feat}_{resource_feat}"] = event_values * resource_values
    
    # If any interaction features were created, add them to the feature list
    if interaction_features:
        feature_dfs.append(pd.DataFrame(interaction_features))
    
    # Concatenate all feature DataFrames horizontally
    return pd.concat(feature_dfs, axis=1)

# Prepare feature sets for training and testing
X_train = prepare_data(train)  # Features for training
X_test = prepare_data(test)    # Features for testing

# Extract the target variable 'fault_severity' from training data
y = X_train["fault_severity"]

# Remove the target variable from the feature set
X_train = X_train.drop(["fault_severity"], axis=1)

# Initialize the LightGBM classifier with specified parameters

model = LGBMClassifier(
    objective="multiclass",          # Multi-class classification objective
    num_class=3,                     # Number of classes
    n_estimators=100,                # Number of trees
    learning_rate=0.05,              # Learning rate
    random_state=42,                 # Seed for reproducibility
    force_row_wise=True,             # Address potential warnings by enforcing row-wise data
    verbose=-1                       # Suppress warnings and verbose output
)

# Initialize StratifiedKFold with 5 splits to maintain class distribution across folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize list to store validation log loss scores
val_scores = []

# Initialize array to store averaged test set predictions
test_preds = np.zeros((len(X_test), 3))  # Assuming 3 classes

# Iterate over each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"Training fold {fold + 1}/5...")
    
    # Split data into training and validation sets based on indices
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Train the LightGBM model on the current fold's training data
    model.fit(X_train_fold, y_train_fold)  # Removed verbose parameter from fit()
    
    # Predict probabilities on the validation set
    val_pred = model.predict_proba(X_val_fold)
    
    # Calculate log loss for the current fold
    val_score = log_loss(y_val_fold, val_pred)
    
    # Append the log loss score to the list of validation scores
    val_scores.append(val_score)
    
    # Print the log loss score for the current fold
    print(f"Fold {fold + 1} validation log loss: {val_score:.4f}")

    # Accumulate the test set predictions by averaging over folds
    test_preds += model.predict_proba(X_test) / 5  # Divide by number of folds to average


print(
    f"\nMean validation log loss: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})"
)

# Create a DataFrame for submission with predicted probabilities
submission = pd.DataFrame(
    test_preds, columns=["predict_0", "predict_1", "predict_2"], index=X_test.index
)

# Set the 'id' as the index name
submission.index.name = "id"

# Save the submission DataFrame to a CSV file without the index
submission.to_csv("/kaggle/working/submission.csv")


df = pd.read_csv("/kaggle/working/telstra-data/train.csv")


df.head()


df['fault_severity'].unique()


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

# Create a graph
G = nx.Graph()

# Add nodes with a 'type' attribute to distinguish IDs and Locations
# This is useful for specifying node sets in bipartite layouts
id_nodes = df['id'].unique()
location_nodes = df['location'].unique()

# Add nodes to the graph, assigning a 'type' attribute
G.add_nodes_from(id_nodes, type='id')
G.add_nodes_from(location_nodes, type='location')

# Add edges based on the DataFrame rows
# Each row represents a connection between an 'id' and a 'location'
# We also add 'fault_severity' as an edge attribute
for index, row in df.iterrows():
    G.add_edge(row['id'], row['location'], fault_severity=row['fault_severity'])

# Separate nodes by type for applying a bipartite layout
nodes_id = {n for n, d in G.nodes(data=True) if d['type'] == 'id'}
nodes_location = {n for n, d in G.nodes(data=True) if d['type'] == 'location'}

# Use a bipartite layout for clear separation of ID and Location nodes
# Check if the graph is bipartite before applying bipartite_layout
if nx.is_bipartite(G):
    # Apply the bipartite layout, positioning ID nodes in one partition
    pos = nx.bipartite_layout(G, nodes_id)
else:
    # Fallback to a general layout like spring_layout if the graph is not strictly bipartite
    # (e.g., if there were direct connections between IDs or locations in the data)
    print("Warning: Graph is not strictly bipartite. Using spring_layout instead.")
    pos = nx.spring_layout(G) # Other options: nx.kamada_kawai_layout(G), nx.circular_layout(G)


# Get edge fault severities for coloring the edges
# We need a list of fault severities in the same order as the edges are drawn
edge_list = list(G.edges(data=True))
edge_fault_severity = [data['fault_severity'] for u, v, data in edge_list]

# Create a figure and axes for the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Draw the ID nodes
nx.draw_networkx_nodes(G, pos, nodelist=list(nodes_id), node_color='skyblue', label='ID', node_size=300, ax=ax)
# Draw the Location nodes
nx.draw_networkx_nodes(G, pos, nodelist=list(nodes_location), node_color='lightgreen', label='Location', node_size=300, ax=ax)

# Draw the edges, coloring them based on 'fault_severity'
# We use a colormap ('viridis' in this case) to map severity values to colors
cmap = cm.viridis # Choose a colormap (e.g., 'plasma', 'inferno', 'cividis')
norm = colors.Normalize(vmin=min(edge_fault_severity), vmax=max(edge_fault_severity)) # Normalize severity values for the colormap

# Draw edges on the main axes, passing the list of fault severities to edge_color
# The 'edges' variable will store the LineCollection object, which is needed for the color bar
edges = nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_fault_severity,
                               width=1.0, edge_cmap=cmap, ax=ax)


# Draw labels for the nodes
# We create a dictionary mapping node IDs to their labels (the node value itself)
all_nodes = list(G.nodes())
labels = {node: node for node in all_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)

# Add a color bar to the plot to indicate the mapping of colors to fault severity values
# We use the LineCollection object ('edges') returned by draw_networkx_edges as the mappable
if edges is not None:
    # Create a new axes specifically for the color bar to control its position and size
    # The arguments are [left, bottom, width, height] in figure coordinates (0 to 1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) # Adjust these values to position the color bar

    # Create the color bar using the LineCollection object and the dedicated color bar axes
    cbar = fig.colorbar(edges, cax=cbar_ax, label='Fault Severity')

# Set the title of the plot
ax.set_title("Network Visualization of IDs and Locations with Fault Severity")
# Add a legend to distinguish between ID and Location nodes
ax.legend()
# Turn off the axes
ax.axis('off')
# Display the plot
plt.show()



import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

subset_ids = [14121, 9320, 10001]
subset_df = df[df['id'].isin(subset_ids)].copy()

if subset_df.empty:
    print("The selected subset is empty.")
else:
    # Create Graph from Subset
    G_subset = nx.Graph()
    id_nodes = subset_df['id'].unique()
    location_nodes = subset_df['location'].unique()
    G_subset.add_nodes_from(id_nodes, type='id')
    G_subset.add_nodes_from(location_nodes, type='location')

    for index, row in subset_df.iterrows():
        G_subset.add_edge(row['id'], row['location'], fault_severity=row['fault_severity'])

    # Bipartite Layout
    nodes_id_subset = {n for n, d in G_subset.nodes(data=True) if d['type'] == 'id'}
    if nx.is_bipartite(G_subset):
        pos_subset = nx.bipartite_layout(G_subset, nodes_id_subset)
    else:
        pos_subset = nx.spring_layout(G_subset)

    # Edge Colors
    edge_list_subset = list(G_subset.edges(data=True))
    edge_fault_severity_subset = [data['fault_severity'] for u, v, data in edge_list_subset]

    # Plotting
    fig_subset, ax_subset = plt.subplots(1, 1, figsize=(10, 6))

    nx.draw_networkx_nodes(G_subset, pos_subset, nodelist=list(id_nodes), node_color='skyblue', label='ID', node_size=400, ax=ax_subset)
    nx.draw_networkx_nodes(G_subset, pos_subset, nodelist=list(location_nodes), node_color='lightgreen', label='Location', node_size=400, ax=ax_subset)

    cmap_subset = cm.viridis
    if len(set(edge_fault_severity_subset)) > 1:
        norm_subset = colors.Normalize(vmin=min(edge_fault_severity_subset), vmax=max(edge_fault_severity_subset))
    else:
        norm_subset = colors.BoundaryNorm([min(edge_fault_severity_subset)-0.5, max(edge_fault_severity_subset)+0.5], cmap_subset.N)

    edges_subset = nx.draw_networkx_edges(G_subset, pos_subset, edgelist=edge_list_subset, edge_color=edge_fault_severity_subset,
                                          width=1.5, edge_cmap=cmap_subset, ax=ax_subset)

    # Labels
    labels_subset = {node: node for node in G_subset.nodes()}
    nx.draw_networkx_labels(G_subset, pos_subset, labels=labels_subset, font_size=9, font_weight='bold', ax=ax_subset)

    # Color Bar
    if edges_subset is not None and len(set(edge_fault_severity_subset)) > 1:
        cbar_ax_subset = fig_subset.add_axes([0.85, 0.15, 0.03, 0.7])
        cbar_subset = fig_subset.colorbar(edges_subset, cax=cbar_ax_subset, label='Fault Severity', norm=norm_subset)
        cbar_subset.set_ticks(list(sorted(set(edge_fault_severity_subset))))
    elif len(set(edge_fault_severity_subset)) <= 1:
        print("Only one fault severity level in the subset, color bar not added.")

    ax_subset.set_title("Network Visualization for Subset of Data")
    ax_subset.legend()
    ax_subset.axis('off')
    plt.show()



from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Disable Grid Search by default
if True:
    # Load the data
    df = pd.read_csv("/kaggle/working/telstra-data/train.csv")
    
    # Handle categorical variables
    le = LabelEncoder()
    df['location'] = le.fit_transform(df['location'])
    
    # Separate features and target
    X = df.drop('fault_severity', axis=1)
    y = df['fault_severity']
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Define parameter grid for LightGBM
    param_grid = {
        # Basic parameters
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [-1],  # -1 means no limit
        'num_leaves': [31, 63, 127],
        
        # Sampling parameters
        'subsample': [0.6, 0.8, 1.0],  # fraction of samples used for training
        'subsample_freq': [0, 1],  # frequency of subsample
        'colsample_bytree': [0.6, 0.8, 1.0],  # fraction of features used per tree
    }
    
    # Initialize base model
    base_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=42,
        force_row_wise=True,
        verbose=-1
    )
    
    # Initialize GridSearchCV
    print("Starting Grid Search...")
    halving_search = HalvingGridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_log_loss',
        factor=3,
        min_resources='smallest',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the halving search
    print("Fitting Halving Grid Search...")
    halving_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("\nBest parameters found:")
    print(halving_search.best_params_)
    print(f"\nBest cross-validation score: {-halving_search.best_score_:.4f}")
    
    # Use best model for predictions
    best_model = halving_search.best_estimator_
    
    # Make predictions with the best model
    print("\nMaking predictions with best model...")
    test_preds = best_model.predict_proba(X_test)
    
    # Create submission file
    submission = pd.DataFrame(
        test_preds, 
        columns=["predict_0", "predict_1", "predict_2"], 
        index=X_test.index
    )
    submission.index.name = "id"
    
    # Save the best parameters to a file
    with open('/kaggle/working/best_params.txt', 'w') as f:
        f.write("Best LightGBM Parameters:\n")
        for param, value in halving_search.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest CV Score (Log Loss): {-halving_search.best_score_:.4f}")
    
    # Print feature importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importances to file
    feature_importance.to_csv('/kaggle/working/feature_importances.csv', index=False)


import pandas as pd
import numpy as np

# Create a DataFrame from the test predictions
# The columns correspond to the predicted fault severities (0, 1, 2)
predictions_df = pd.DataFrame(
    test_preds,
    columns=["predict_0", "predict_1", "predict_2"],
    index=X_test.index # Use the original index from X_test to link back to IDs
)

# Add 'id' and 'location' from X_test to the predictions DataFrame
# Ensure the index aligns correctly
predictions_df['id'] = X_test['id']
predictions_df['location'] = X_test['location']

# Determine the predicted class (fault severity) for each instance
# This is the class with the highest predicted probability
predictions_df['predicted_severity'] = predictions_df[["predict_0", "predict_1", "predict_2"]].idxmax(axis=1).str.replace('predict_', '').astype(int)

# Filter for instances where the predicted severity indicates a fault (severity > 0)
faulty_predictions_df = predictions_df[predictions_df['predicted_severity'] > 0].copy()

# For each faulty prediction, get the probability of the predicted fault class
# We use .apply() to dynamically select the probability column based on 'predicted_severity'
# Explicitly cast 'predicted_severity' to int to avoid KeyError
faulty_predictions_df['predicted_probability'] = faulty_predictions_df.apply(
    lambda row: row[f'predict_{int(row["predicted_severity"])}'], axis=1 # Added int() here
)

# Sort the faulty predictions by their predicted probability in descending order
top_faulty_predictions = faulty_predictions_df.sort_values(
    by='predicted_probability', ascending=False
)

# Select the top 10 predictions
top_10_faults = top_faulty_predictions.head(10)

# Display the top 10 predicted faults
print("Top 10 Predicted Faults (ID, Location, Predicted Severity, Probability):")
print(top_10_faults[['id', 'location', 'predicted_severity', 'predicted_probability']])

# Optional: Save the top 10 faults to a CSV file
# top_10_faults[['id', 'location', 'predicted_severity', 'predicted_probability']].to_csv('/kaggle/working/top_10_predicted_faults.csv', index=False)



import pandas as pd
# Filter for rows with fault_severity > 0
faulty_entries = df[df['fault_severity'] > 0]
# Count faults per location
location_fault_counts = faulty_entries['location'].value_counts()

# 2. Define a threshold for problematic locations
fault_threshold = 2 # Example threshold: locations with 2 or more faults are problematic

# Identify locations that exceed the fault threshold
problematic_locations = location_fault_counts[location_fault_counts >= fault_threshold].index.tolist()

# 3. Identify potentially faulty id-location pairs based on the heuristic
# Initialize a list to store predicted faulty pairs
predicted_faulty_pairs = []

# Iterate through the original dataframe
for index, row in df.iterrows():
    # Check if the location of the current row is in the list of problematic locations
    if row['location'] in problematic_locations:
        # If it is, consider this id-location pair as potentially faulty by the heuristic
        # We can assign a 'heuristic_severity' based on the simple rule (e.g., 1)
        predicted_faulty_pairs.append({
            'id': row['id'],
            'location': row['location'],
            'heuristic_severity': 1 # Assign a severity based on the heuristic
        })

# Convert the results to a DataFrame for easier viewing
predicted_faults_df = pd.DataFrame(predicted_faulty_pairs)

# Display the predicted faulty pairs based on the heuristic
print("Predicted Faulty Pairs based on Simple Heuristic:")
print(predicted_faults_df.head()) # Print head as the list can be long



import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt # Included for potential visualization, though not the primary output

# 1. Filter the DataFrame to include only rows with fault_severity > 0
faulty_df = df[df['fault_severity'] > 0].copy()

if faulty_df.empty:
    print("No historical faults found in the data to build the faulty graph.")
else:
    # 2. Create a graph from the faulty data
    G_faulty = nx.from_pandas_edgelist(
        faulty_df,
        source='id',
        target='location',
        edge_attr='fault_severity' # Keep fault severity as an edge attribute
    )

    # 3. Calculate Degree Centrality for all nodes in the faulty graph
    # Degree centrality measures the number of connections a node has
    degree_centrality = nx.degree_centrality(G_faulty)

    # 4. Extract Degree Centrality specifically for Location nodes
    # We need to know which nodes are locations. We can identify them from the original faulty_df.
    location_nodes_in_faulty_graph = faulty_df['location'].unique()

    location_centralities = {
        location: degree_centrality[location]
        for location in location_nodes_in_faulty_graph if location in degree_centrality # Ensure location is in the graph nodes
    }

    # 5. Sort locations by their Degree Centrality in descending order
    sorted_locations_by_centrality = sorted(
        location_centralities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # 6. Select the top N locations based on centrality
    num_top_locations = 5 # You can adjust this number
    top_locations = [location for location, centrality in sorted_locations_by_centrality[:num_top_locations]]

    print(f"Top {num_top_locations} locations by Degree Centrality in the faulty network: {top_locations}")

    # 7. Identify potentially faulty id-location pairs from the *original* dataframe
    # These are all pairs in the original data that involve the top faulty locations
    predicted_faulty_pairs_graph = df[df['location'].isin(top_locations)].copy()

    # Optional: Add the location's centrality score to the output DataFrame for context
    centrality_map = dict(sorted_locations_by_centrality)
    predicted_faulty_pairs_graph['location_faulty_centrality'] = predicted_faulty_pairs_graph['location'].map(centrality_map)

    # Sort the results by location centrality (and potentially by original fault severity if available)
    predicted_faulty_pairs_graph = predicted_faulty_pairs_graph.sort_values(
        by=['location_faulty_centrality', 'fault_severity'],
        ascending=[False, False]
    )


    # Display the predicted faulty pairs based on the graph algorithm
    print("\nPotentially Faulty Pairs based on Graph Algorithm (Top Locations by Faulty Network Centrality):")
    # Display relevant columns. Include original fault_severity for comparison if needed.
    print(predicted_faulty_pairs_graph[['id', 'location', 'fault_severity', 'location_faulty_centrality']].head(10)) # Display top 10 pairs from this list


