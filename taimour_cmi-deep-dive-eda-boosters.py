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
from sklearn.metrics import accuracy_score, f1_score
import joblib
from scipy.spatial.transform import Rotation as R


train_pl = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo_pl = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
train = train_pl.to_pandas()
train_demo = train_demo_pl.to_pandas()


train.info()


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


le = joblib.load('/kaggle/input/cmi-voting-classifier-ensemble-training/Target_LabelEncoder.joblib')
robustscaler = joblib.load('/kaggle/input/cmi-voting-classifier-ensemble-training/robustscaler.joblib')
model_cat =  joblib.load('/kaggle/input/cmi-voting-classifier-ensemble-training/model_cat.joblib')
model_lgb =  joblib.load('/kaggle/input/cmi-voting-classifier-ensemble-training/model_lgb.joblib')
model_xgb =  joblib.load('/kaggle/input/cmi-voting-classifier-ensemble-training/model_xgb.joblib')
meta_model =  joblib.load('/kaggle/input/cmi-voting-classifier-ensemble-training/meta_model.joblib')


def remove_gravity_from_acc(acc_data, rot_data):

    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
            
    return angular_vel

def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0 # Ğ˜Ğ»Ğ¸ np.nan, Ğ² Ğ·Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ğ¾Ñ‚ Ğ¶ĞµĞ»Ğ°ĞµĞ¼Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
            continue
        try:
            # Converting quaternions to Rotation objects
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Calculating the angular distance: 2 * arccos(|real(p * q*)|)
            # where q* is the conjugate of quaternion q
            # In scipy.spatial.transform.Rotation, r1.inv() * r2 gives the relative rotation.
            # The angle of this relative rotation is the angular distance.
            relative_rotation = r1.inv() * r2
            
            # The angle of the rotation vector corresponds to the angular distance
            # The norm of the rotation vector is the angle in radians
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # In case of invalid quaternions
            pass
            
    return angular_dist



def feature_engineering_imu(data:pl.DataFrame):
    data = data.to_pandas()
    data['acc_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
    data['rot_angle'] = 2 * np.arccos(data['rot_w'].clip(-1, 1))
    data['acc_mag_jerk'] = data.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    data['rot_angle_vel'] = data.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    linear_accel_list = []
    for _, group in data.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    
    df_linear_accel = pd.concat(linear_accel_list)
    data = pd.concat([data, df_linear_accel], axis=1)
    data['linear_acc_mag'] = np.sqrt(data['linear_acc_x']**2 + data['linear_acc_y']**2 + data['linear_acc_z']**2)
    data['linear_acc_mag_jerk'] = data.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    angular_vel_list = []
    for _, group in data.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
    
    df_angular_vel = pd.concat(angular_vel_list)
    data = pd.concat([data, df_angular_vel], axis=1)
    
    print("  Calculating angular distance between successive quaternions...")
    angular_distance_list = []
    for _, group in data.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
    
    df_angular_distance = pd.concat(angular_distance_list)
    data = pd.concat([data, df_angular_distance], axis=1)
    data = pl.from_pandas(data)
    return data


def feature_engineering_stat(data:pl.DataFrame):
    non_sensor_cols = []
    if "gesture" in data.columns:
        non_sensor_cols = ["gesture"]
        
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


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # data =sequence
    data = sequence.join(demographics,on="subject",how="left")
    # print(data.schema)
    data = feature_engineering_imu(data)
    cleaned_data = feature_engineering_stat(data)

    pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])

    pdf_scaled = robustscaler.transform(pdf)  # Shape: (1, n_features)
    
    # Get predictions from base models (probabilities)
    p_cat = model_cat.predict_proba(pdf_scaled)  # (1, num_classes)
    p_lgb = model_lgb.predict_proba(pdf_scaled)
    p_xgb = model_xgb.predict_proba(pdf_scaled)
    
    # Stack probabilities for meta-model
    meta_features = np.hstack([p_cat, p_lgb, p_xgb])  # (1, 3*num_classes)
    
    # Final prediction using meta-model
    y_pred_encoded = meta_model.predict(meta_features)  # Array of length 1
    class_label = le.inverse_transform(y_pred_encoded)[0]
    return class_label


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

