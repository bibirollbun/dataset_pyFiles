import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import polars as pl
import kaggle_evaluation.cmi_inference_server
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


train_pl = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo_pl = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
train = train_pl.to_pandas()
train_demo = train_demo_pl.to_pandas()


train.head(2)


train.describe()


gesture_counts = train['gesture'].value_counts()

# Calculate percentages
percentages = (gesture_counts / gesture_counts.sum()) * 100

# Create labels with both value counts and percentages
labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(gesture_counts.index, gesture_counts.values, percentages)]

# Create the pie chart
plt.figure(figsize=(8, 8)) # Set figure size for better readability
plt.pie(gesture_counts, labels=labels, startangle=160, counterclock=False, wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Gestures')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Donut Chart
fig, ax = plt.subplots(figsize=(8, 8))

# Calculate percentages for labels (optional, can also use autopct)
orientation_counts = train["orientation"].value_counts().head()
total_count = orientation_counts.sum()
labels = [f'{label} ({count})' for label, count in orientation_counts.items()]

wedges, texts, autotexts = ax.pie(
    orientation_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors,
    pctdistance=0.85, # Distance of the percentage labels from the center
    wedgeprops=dict(width=0.3, edgecolor='w'), # Create the donut hole
    labels=labels
)

# Draw a circle in the middle to make it a donut chart
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title("Subject's orientation (Donut Chart)")
ax.axis('equal')
plt.show()


# Vertical Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
orientation_counts.sort_values(ascending=False).plot( # Often good to sort descending for vertical bars
    kind="bar", color='purple', ax=ax, title="Target (Vertical Bar)"
)
ax.set_xlabel("Target")
ax.set_ylabel("Count")
plt.xticks(rotation=20, ha='right') # Rotate x-axis labels for better readability if they overlap
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# Donut Chart
fig, ax = plt.subplots(figsize=(8, 8))

# Calculate percentages for labels (optional, can also use autopct)
sequence_type_counts = train["sequence_type"].value_counts().head()
total_count = sequence_type_counts.sum()
labels = [f'{label} ({count})' for label, count in sequence_type_counts.items()]

wedges, texts, autotexts = ax.pie(
    sequence_type_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors,
    pctdistance=0.85, # Distance of the percentage labels from the center
    wedgeprops=dict(width=0.3, edgecolor='w'), # Create the donut hole
    labels=labels
)

# Draw a circle in the middle to make it a donut chart
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title("Target vs Non-target (Donut Chart)")
ax.axis('equal')
plt.show()


# Vertical Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
sequence_type_counts.sort_values(ascending=False).plot( # Often good to sort descending for vertical bars
    kind="bar", color='lightgreen', ax=ax, title="Target vs Non Target (Vertical Bar)"
)
ax.set_xlabel("Target")
ax.set_ylabel("Count")
plt.xticks(rotation=20, ha='right') # Rotate x-axis labels for better readability if they overlap
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# Donut Chart
fig, ax = plt.subplots(figsize=(8, 8))

# Calculate percentages for labels (optional, can also use autopct)
behavior_counts = train["behavior"].value_counts().head()
total_count = behavior_counts.sum()
labels = [f'{label} ({count})' for label, count in behavior_counts.items()]

wedges, texts, autotexts = ax.pie(
    behavior_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors,
    pctdistance=0.85, # Distance of the percentage labels from the center
    wedgeprops=dict(width=0.3, edgecolor='w'), # Create the donut hole
    labels=labels
)

# Draw a circle in the middle to make it a donut chart
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title("Subject's Behavior During the Sequence (Donut Chart)")
ax.axis('equal')
plt.show()


# Vertical Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
behavior_counts.sort_values(ascending=False).plot( # Often good to sort descending for vertical bars
    kind="bar", color='skyblue', ax=ax, title="Subject's Behavior During the Sequence (Vertical Bar)"
)
ax.set_xlabel("Behavior")
ax.set_ylabel("Count")
plt.xticks(rotation=20, ha='right') # Rotate x-axis labels for better readability if they overlap
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# correlation matrix
plt.figure(figsize = (8, 4))
sns.heatmap(data = train_demo.corr(numeric_only = True), linewidth = 0.5, annot = True, linecolor = "black")
plt.title('Train Data Demographic Correlation Heatmap')
plt.show()


# Get value counts for gender
vc = train_demo['sex'].value_counts()

# Extract counts (assuming 'count' column doesn't exist)
counts = vc.values

# Create labels (assuming 'count' column doesn't exist)
labels = ['Male', 'Female']  # Assuming 'Male' maps to '1' and 'Female' to '0'

# Plot the pie chart
plt.pie(counts, labels=labels, autopct="%1.1f%%")  # Add percentages to pie slices
plt.title('Gender')
plt.show()


#'train' is DataFrame
_, axs = plt.subplots(2, 1, sharex=True)

for sex in range(2):
  ax = axs.ravel()[sex]

  # Filter by sex using boolean indexing
  sex_filter = train_demo['sex'] == sex
  vc = train_demo[sex_filter]['age'].value_counts()  # Access column after filtering

  # Plot the bar chart
  ax.bar(vc.index, vc.values, color=['lightblue', 'coral'][sex], label=['Male', 'Female'][sex])
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_ylabel('count')
  ax.legend()

plt.suptitle('Age distribution with respect to Gender')
axs.ravel()[1].set_xlabel('years')
plt.show()



# Initialize the 3D plot
fig = plt.figure(figsize=(10, 10)) # Adjust figure size for better visibility
ax = fig.add_subplot(111, projection='3d')

# Get unique values from the 'gesture' column for coloring
unique_gestures = train['gesture'].unique()

# Use a colormap to get distinct colors for each gesture type
# matplotlib.colormaps is the recommended way to access colormaps now
colors = plt.colormaps['tab10'].resampled(len(unique_gestures)) # 'tab10' is good for categorical data

# Iterate through unique gestures and plot them with different colors
for i, gesture_type in enumerate(unique_gestures):
    subset = train[train['gesture'] == gesture_type]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'],
               color=colors(i),    # Assign a unique color from the colormap
               label=gesture_type, # Label for the legend
               alpha=0.7,          # Opacity (matches your Plotly opacity)
               s=2,
               rasterized=True)                # Size of points (similar to Plotly's size_max=1, adjusted for visibility)

# Set labels and title
ax.set_xlabel('acc_x', fontsize=14)
ax.set_ylabel('acc_y', fontsize=14)
ax.set_zlabel('acc_z', fontsize=14)
ax.set_title('Linear acceleration along X, Y, Z axes', fontsize=16)

# Add a legend to differentiate 'gesture' types
ax.legend(title='Gesture Type', fontsize=10, title_fontsize=12, loc='upper right')

# Set tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# --- Mimic Plotly's 'plotly_dark' template ---
# Set the background color of the plot
ax.set_facecolor('black')

# Make the pane background transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Set the pane edge colors
ax.xaxis.pane.edgecolor = 'grey'
ax.yaxis.pane.edgecolor = 'grey'
ax.zaxis.pane.edgecolor = 'grey'

# Set the actual pane face colors to transparent
ax.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))

# Set grid lines color
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()



# Initialize the 3D plot
fig = plt.figure(figsize=(10, 10)) # Adjust figure size for better visibility
ax = fig.add_subplot(111, projection='3d')

# Get unique values from 'sequence_type' for coloring
unique_sequence_types = train['sequence_type'].unique()

# --- ADDRESSING DEPRECATION WARNING 1: get_cmap ---
# Use matplotlib.colormaps[name] instead of plt.cm.get_cmap(name)
colors = plt.colormaps['viridis'].resampled(len(unique_sequence_types))

# Iterate through unique sequence types and plot them with different colors
for i, seq_type in enumerate(unique_sequence_types):
    subset = train[train['sequence_type'] == seq_type]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'],
               color=colors(i), # Assign a unique color
               label=seq_type,  # Label for the legend
               alpha=0.7,       # Opacity
               s=10)            # Size of points (similar to Plotly's size_max)

# Set labels and title
ax.set_xlabel('acc_x', fontsize=14)
ax.set_ylabel('acc_y', fontsize=14)
ax.set_zlabel('acc_z', fontsize=14)
ax.set_title('Linear acceleration along X, Y, Z axes', fontsize=16)

# Add a legend to differentiate 'sequence_type'
ax.legend(title='Sequence Type', fontsize=12, title_fontsize=14)

# Set tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Set background color to dark and adjust grid/pane colors
ax.set_facecolor('black')

# --- ADDRESSING DEPRECATION WARNINGS 2, 3, 4: w_xaxis, w_yaxis, w_zaxis ---
# Use .xaxis, .yaxis, .zaxis directly for panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.edgecolor = 'grey'
ax.yaxis.pane.edgecolor = 'grey'
ax.zaxis.pane.edgecolor = 'grey'

# Set pane colors to transparent for a truly dark effect
# These are now accessed directly through .xaxis.pane.set_facecolor
ax.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))

ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()



fig = plt.figure(figsize=(14, 8)) # Adjust figure size as needed
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D line
# For color, matplotlib's plot3D doesn't directly support a continuous 'color'
# argument like px.line_3d. You'd typically use a colormap and map 'rot_w' to it.
# This example uses a simple line color. For continuous color, you'd need
# to segment your data or use scatter3D with c=train['rot_w'].
# ax.plot(train['rot_x'], train['rot_y'], train['rot_z'], color='skyblue', linewidth=0.5)

# If you want to color by 'rot_w' with a gradient, it's often better to use scatter3D
# if a continuous line isn't strictly necessary, or segment the line.
# For a true line plot with continuous color, it's more complex in matplotlib.
# Example for scatter3D with color:
sc = ax.scatter(train['rot_x'], train['rot_y'], train['rot_z'], c=train['rot_w'], cmap='viridis', s=1)
fig.colorbar(sc, ax=ax, label='rot_w')


ax.set_xlabel('rot_x')
ax.set_ylabel('rot_y')
ax.set_zlabel('rot_z')
ax.set_title('3D Path of Gyroscope Rotation Over Time')

# Adjust font sizes if needed (similar to update_layout in Plotly)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.zaxis.label.set_size(14)
ax.title.set_size(16)

plt.show()


# Melt the DataFrame for easier plotting
df_melted = train.melt(id_vars='sequence_type', 
                    value_vars=['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
                    var_name='sensor', 
                    value_name='temperature')

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='sequence_type', y='temperature', hue='sensor', data=df_melted)
plt.title('Thermopile Sensor Readings by Sequence Type')
plt.xlabel('Sequence Type')
plt.ylabel('Temperature (Â°C)')
plt.legend(title='Sensor')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Group by sequence_type and get mean values
df_grouped = train.groupby('sequence_type')[['thm_1','thm_2','thm_3','thm_4','thm_5']].mean()

categories = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for idx, row in df_grouped.iterrows():
    values = row.values.tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, marker='o', label=idx)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.legend(title="Sequence Type")
plt.title("Radar Chart of Average Thermopile Readings by Sequence Type")
plt.show()


# Melt the DataFrame for easier plotting
df_melted = train.melt(id_vars='gesture', 
                    value_vars=['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
                    var_name='sensor', 
                    value_name='temperature')

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='gesture', y='temperature', hue='sensor', data=df_melted)
plt.title('Thermopile Sensor Readings by Gesture')
plt.xlabel('Gesture')
plt.ylabel('Temperature (Â°C)')
plt.legend(title='Sensor')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Melt the DataFrame for easier plotting
df_melted = train.melt(id_vars='behavior', 
                    value_vars=['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
                    var_name='sensor', 
                    value_name='temperature')

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='behavior', y='temperature', hue='sensor', data=df_melted)
plt.title('Thermopile Sensor Readings by Behavior')
plt.xlabel('Behavior')
plt.ylabel('Temperature (Â°C)')
plt.legend(title='Sensor')
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


def null_percent(df):
    per=((df.isnull().sum()/len(df))*100).round(2)
    return per[per>0]

print("Nan Values in Train data")
print(null_percent(train))


print(train_pl.shape)

# Drop these columns from training data
train = train_pl.drop(['phase', 'orientation', 'behavior', 'sequence_type'])
print(train.shape)

data = train.join(train_demo_pl,on="subject",how="left")
print(data.shape)


train.head(2)


def feature_engineering(data:pl.DataFrame, non_sensor_cols: list):
    
    # All numeric sensor columns (everything except id, demo, target)
    stat_cols = [
        c for c in data.columns
        if c not in non_sensor_cols + ["sequence_id", "row_id","sequence_counter","subject"]
    ]
    
    # Build aggregation expressions
    agg_exprs = []
    
    # full-stats bundle for sensor columns
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            # pl.col(c).mode().list.first().alias(f"{c}_mode"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).first().alias(f"{c}_first"),
            pl.col(c).last().alias(f"{c}_last"),
            pl.col(c).quantile(0.25, "nearest").alias(f"{c}_t25"),
            pl.col(c).quantile(0.75, "nearest").alias(f"{c}_t75"),
            (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_delta"),
            pl.corr("sequence_counter", c).alias(f"{c}_corr_time"),
            pl.col(c).diff().mean().alias(f"{c}_diff_mean"),
            pl.col(c).diff().std().alias(f"{c}_diff_std"),
            pl.col(c).skew().alias(f"{c}_skew"),
            pl.col(c).kurtosis().alias(f"{c}_kurt"),
            pl.col(c).diff().abs().gt(0).sum().alias(f"{c}_n_changes")
        ]
        agg_exprs += [
            pl.when(pl.col("sequence_counter") < 0.1 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg1_mean"),
            pl.when(pl.col("sequence_counter") > 0.9 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg3_mean"),
        ]
    
    # first() for demographics and target
    agg_exprs += [
        pl.col(c).first().alias(c) for c in non_sensor_cols
    ]
    
    # Group-by and aggregate
    cleaned_data = (
        data
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
    )
    return cleaned_data



train_demographic_target_cols = [
    "adult_child", "age", "sex", "handedness", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm", "gesture"
    ]
cleaned_data = feature_engineering(data,train_demographic_target_cols)
cleaned_data.shape


# Assume cleaned_data is already a Polars DataFrame
target_col = "gesture"

# --- Convert Polars DataFrame to Pandas only if needed ---
# CatBoost does not yet fully support Polars directly
df = cleaned_data.to_pandas()

# --- Define X and y properly ---
X = df.drop(columns=[target_col, "sequence_id"])  # Feature matrix
y = df[target_col].values # Target

# Encode target if needed
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Set up stratified KFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_acc, fold_f1 = [], []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    final_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.1,
        loss_function='MultiClass',
        task_type="GPU",
        devices='0',
        verbose=False
    )

    final_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    y_pred = final_model.predict(X_val).ravel()

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")

    fold_acc.append(acc)
    fold_f1.append(f1)
    print(f"Fold {fold}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}")

print("\n======  5-Fold Summary  ======")
print(f"Accuracy:  mean={np.mean(fold_acc):.4f}  std={np.std(fold_acc):.4f}")
print(f"Macro-F1 : mean={np.mean(fold_f1):.4f}  std={np.std(fold_f1):.4f}")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    data = sequence.join(demographics,on="subject",how="left")
    test_demographic_cols = [
        "adult_child", "age", "sex", "handedness", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"
    ]
    cleaned_data = feature_engineering(data, test_demographic_cols)
    pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])
    predictions = final_model.predict(pdf).ravel()
    predictions = le.inverse_transform(predictions)
    return predictions[0]


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

