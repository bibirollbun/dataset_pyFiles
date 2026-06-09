# Import libraries
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft
from scipy.stats import kruskal, zscore
from sklearn.ensemble import RandomForestClassifier

# Set professional plotting style
plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Arial'
custom_palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
sns.set_palette(custom_palette)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)



# Load data with progress bar
def load_data():
    data_files = {
        'train': '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv',
        'test': '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
        'train_demo': '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv',
        'test_demo': '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
    }
    data = {}
    with tqdm(total=len(data_files), desc="Loading Data") as pbar:
        for name, file in data_files.items():
            try:
                data[name] = pd.read_csv(file)
                pbar.update(1)
            except FileNotFoundError:
                print(f"Error: {file} not found. Please check the file path.")
                return None
    return data

data = load_data()
if data is None:
    raise FileNotFoundError("Data loading failed.")
train, test, demo_train, demo_test = data['train'], data['test'], data['train_demo'], data['test_demo']

# Display basic dataset info
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("Train Demographics Shape:", demo_train.shape)
print("Test Demographics Shape:", demo_test.shape)


def plot_gesture_distribution(df):

    gesture_counts = df['gesture'].value_counts().sort_values(ascending=False)
    gesture_percentages = gesture_counts / gesture_counts.sum() * 100

    # Bar plot with counts
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=gesture_counts.index, y=gesture_counts.values, 
                     order=gesture_counts.index, palette='viridis')
    plt.title('Gesture Distribution in Training Data\n(Log Scale)', fontsize=16, pad=20)
    plt.xlabel('Gesture Type', fontsize=12)
    plt.ylabel('Count (log scale)', fontsize=12)
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    
    # Add count annotations
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()

    # Percentage plot
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=gesture_percentages.index, y=gesture_percentages.values, 
                     order=gesture_counts.index, palette='viridis')
    plt.title('Gesture Distribution (% of Total)', fontsize=16)
    plt.xlabel('Gesture Type', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage annotations
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()

plot_gesture_distribution(train)


def plot_sensor_distributions(df, sensor_type, cols_per_row=3):
    sensor_cols = [col for col in df.columns if sensor_type in col]
    if not sensor_cols:
        print(f"No columns found for {sensor_type}.")
        return
    
    n_cols = len(sensor_cols)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row
    
    plt.figure(figsize=(18, 4 * n_rows))
    for i, col in enumerate(sensor_cols, 1):
        plt.subplot(n_rows, cols_per_row, i)
        sns.kdeplot(df[col].dropna(), fill=True, alpha=0.7, color=custom_palette[i % len(custom_palette)])
        plt.title(f'Distribution of {col}', fontsize=10)
        plt.xlabel('Value')
        plt.ylabel('Density')
    plt.tight_layout()
    plt.suptitle(f'{sensor_type.upper()} Sensor Distributions', y=1.02, fontsize=16)
    plt.show()

    # Summary statistics
    print(f"\nSummary Statistics for {sensor_type} Sensors:")
    print(df[sensor_cols].describe())

# Plot accelerometer and rotation distributions
plot_sensor_distributions(train, 'acc_')
plot_sensor_distributions(train, 'rot_')

# Correlation heatmap
sensor_cols = [col for col in train.columns if 'acc_' in col or 'rot_' in col]
if sensor_cols:
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[sensor_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Sensor Features', fontsize=16)
    plt.tight_layout()
    plt.show()


def create_interactive_sequence(df, sequence_id):
    pio.renderers.default = 'iframe'
    seq_data = df[df['sequence_id'] == sequence_id].copy()
    if seq_data.empty:
        print(f"Sequence {sequence_id} not found.")
        return
    
    gesture = seq_data['gesture'].iloc[0]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add acceleration traces
    for axis, color in zip(['x', 'y', 'z'], ['#e74c3c', '#3498db', '#2ecc71']):
        fig.add_trace(
            go.Scatter(
                x=seq_data['sequence_counter'],
                y=seq_data[f'acc_{axis}'],
                name=f'Acc {axis.upper()}',
                line=dict(color=color),
                opacity=0.8
            ),
            secondary_y=False
        )
    
    # Add behavior phases as background
    behaviors = seq_data['behavior'].unique()
    colors = px.colors.qualitative.Plotly
    for i, behavior in enumerate(behaviors):
        behavior_data = seq_data[seq_data['behavior'] == behavior]
        fig.add_trace(
            go.Scatter(
                x=behavior_data['sequence_counter'],
                y=[0] * len(behavior_data),
                mode='markers',
                marker=dict(size=15, color=colors[i % len(colors)], opacity=0.2),
                name=behavior,
                showlegend=True
            ),
            secondary_y=True
        )
    
    # Add rotation traces
    for axis, color in zip(['x', 'y', 'z'], ['#e67e22', '#16a085', '#c0392b']):
        fig.add_trace(
            go.Scatter(
                x=seq_data['sequence_counter'],
                y=seq_data[f'rot_{axis}'],
                name=f'Rot {axis.upper()}',
                line=dict(color=color, dash='dot'),
                opacity=0.6
            ),
            secondary_y=False
        )
    
    fig.update_layout(
        title=f'Sequence {sequence_id} - {gesture}<br><sup>Interactive Sensor Timeline (Sequence Counter = Time Steps)</sup>',
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    fig.update_yaxes(title_text="Sensor Values", secondary_y=False)
    fig.update_yaxes(title_text="Behavior Phases", secondary_y=True)
    fig.show()

    # FFT analysis for one sensor
    def plot_fft(data, sensor='acc_x'):
        fft_vals = np.abs(fft(data[sensor].dropna()))[:len(data)//2]
        freqs = np.linspace(0, 0.5, len(fft_vals))
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, fft_vals, color=custom_palette[0])
        plt.title(f'FFT of {sensor} for Sequence {sequence_id}')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
    
    plot_fft(seq_data, 'acc_x')

# Plot for a sample sequence
sample_seq = train['sequence_id'].unique()[10]
create_interactive_sequence(train, sample_seq)


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np

# Reuse custom palette
custom_palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
sns.set_palette(custom_palette)

def plot_missing_data(df, top_n=10):
    # Calculate missing data percentages
    missing_data = df.isna().mean() * 100
    missing_data = missing_data[missing_data > 0]  # Filter columns with missing values
    
    if missing_data.empty:
        print("No missing data found.")
        return
    
    # --- 1. Grouped Bar Plot by Sensor Type ---
    # Define sensor types based on column prefixes
    sensor_types = ['acc_', 'rot_', 'thermopile_', 'tof_']  
    missing_by_type = {}
    for stype in sensor_types:
        cols = [col for col in missing_data.index if stype in col]
        if cols:
            missing_by_type[stype.replace('_', '')] = missing_data[cols].mean()
    
    if missing_by_type:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(missing_by_type.keys()), y=list(missing_by_type.values()))
        plt.title('Average Missing Data by Sensor Type', fontsize=16)
        plt.xlabel('Sensor Type', fontsize=12)
        plt.ylabel('Missing (%)', fontsize=12)
        # Add annotations
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points', fontsize=9)
        plt.tight_layout()
        plt.show()

    # --- 2. Top-N Features Bar Plot ---
    top_missing = missing_data.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=top_missing.index, y=top_missing.values, palette='magma')
    plt.title(f'Top {top_n} Features with Highest Missing Data', fontsize=16)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Missing (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    # Add annotations
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=9)
    plt.tight_layout()
    plt.show()

    # --- 3. Heatmap of Missing Data by Gesture ---
    sensor_cols = [col for col in df.columns if any(stype in col for stype in sensor_types)]
    if sensor_cols:
        missing_by_gesture = df.groupby('gesture')[sensor_cols].apply(lambda x: x.isna().mean() * 100)
        plt.figure(figsize=(12, 8))
        sns.heatmap(missing_by_gesture, cmap='magma', annot=True, fmt='.1f', cbar_kws={'label': 'Missing (%)'})
        plt.title('Missing Data (%) by Gesture and Feature', fontsize=16)
        plt.xlabel('Feature')
        plt.ylabel('Gesture')
        plt.tight_layout()
        plt.show()

    # --- 5. Summary Table ---
    print("\nSummary of Missing Data (%):")
    print(missing_data.to_frame(name='Missing (%)').sort_values(by='Missing (%)', ascending=False))


plot_missing_data(train)


def plot_demographic_insights(train_df, demo_df):
    # Merge train and demographic data
    merged = train_df.merge(demo_df, on='subject', how='left')
    
    # Age distribution by gender
    plt.figure(figsize=(10, 6))
    sns.histplot(data=demo_df, x='age', hue='sex', multiple='stack', palette='Set2')
    plt.title('Age Distribution by Gender', fontsize=16)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    # Gesture distribution by gender
    gesture_by_gender = merged.groupby(['sex', 'gesture']).size().unstack().fillna(0)
    gesture_by_gender.plot(kind='bar', stacked=True, figsize=(10, 6), color=custom_palette)
    plt.title('Gesture Distribution by Gender', fontsize=16)
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Gesture', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_demographic_insights(train, demo_train)


from scipy.stats import kruskal
import seaborn as sns
import matplotlib.pyplot as plt

def plot_gesture_duration(df):
    # Compute gesture counts for ordering
    gesture_counts = df['gesture'].value_counts().sort_values(ascending=False)
    gesture_durations = df.groupby(['gesture', 'sequence_id']).size().reset_index(name='duration')
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=gesture_durations, x='gesture', y='duration', 
                order=gesture_counts.index, palette='viridis', showfliers=False)
    plt.title('Gesture Duration Distribution (Without Outliers)', fontsize=16)
    plt.xlabel('Gesture')
    plt.ylabel('Duration (frames)')
    plt.xticks(rotation=45, ha='right')
    
    # Add median labels
    medians = gesture_durations.groupby('gesture')['duration'].median().loc[gesture_counts.index]
    for i, median in enumerate(medians):
        plt.text(i, median + 5, f"Median: {median}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical test for duration differences
    groups = [gesture_durations[gesture_durations['gesture'] == g]['duration'] for g in gesture_counts.index]
    stat, p = kruskal(*groups)
    print(f"Kruskal-Wallis Test for Duration Differences: Statistic={stat:.2f}, p-value={p:.4f}")
    
plot_gesture_duration(train)


def compare_train_test(train_df, test_df):
    sensor_cols = [col for col in train_df.columns if 'acc_' in col or 'rot_' in col]
    for sensor in sensor_cols[:3]:  # Limit to first 3 for brevity
        plt.figure(figsize=(10, 6))
        sns.kdeplot(train_df[sensor].dropna(), label='Train', color=custom_palette[0])
        sns.kdeplot(test_df[sensor].dropna(), label='Test', color=custom_palette[1])
        plt.title(f'Train vs. Test: {sensor} Distribution', fontsize=16)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()

compare_train_test(train, test)


def plot_feature_importance(df):
    sensor_cols = [col for col in df.columns if 'acc_' in col or 'rot_' in col]
    X = df[sensor_cols].fillna(0)  # Simple imputation for demo
    y = df['gesture']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=sensor_cols).sort_values()
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color=custom_palette[2])
    plt.title('Feature Importance (Random Forest)', fontsize=16)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

plot_feature_importance(train)


from scipy.stats import zscore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def detect_outliers(df, z_threshold=3):
    # Define sensor columns
    sensor_cols = [col for col in df.columns if 'acc_' in col or 'rot_' in col]
    if not sensor_cols:
        print("No sensor columns found (acc_ or rot_).")
        return
    
    # Initialize dictionary to store outlier counts
    outlier_counts = {}
    
    # Clean data and compute Z-scores
    for col in sensor_cols:
        # Drop NaN and inf values for this column
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            outlier_counts[col] = 0
            continue
        
        # Check for zero variance
        if data.std() == 0:
            print(f"Warning: Column {col} has zero variance. Skipping outlier detection.")
            outlier_counts[col] = 0
            continue
        
        # Compute Z-scores
        z_scores = zscore(data, nan_policy='omit')
        # Count outliers (|Z| > z_threshold)
        outlier_counts[col] = np.sum(np.abs(z_scores) > z_threshold)
    
    # Print outlier counts
    print("\nOutliers per Sensor (Z-score > {}):".format(z_threshold))
    print(pd.Series(outlier_counts))
    
    # Visualize outliers for acc_x (if available)
    if 'acc_x' in sensor_cols and 'sequence_counter' in df.columns:
        # Create a mask for outliers
        data = df['acc_x'].replace([np.inf, -np.inf], np.nan)
        if data.std() > 0:
            z_scores = zscore(data.dropna(), nan_policy='omit')
            # Align Z-scores with original data
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask[data.dropna().index] = np.abs(z_scores) > z_threshold
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df['sequence_counter'], y=df['acc_x'], 
                            hue=outlier_mask, palette={False: custom_palette[0], True: custom_palette[1]})
            plt.title('Outliers in acc_x (Z-score > {})'.format(z_threshold), fontsize=16)
            plt.xlabel('Sequence Counter')
            plt.ylabel('acc_x')
            plt.tight_layout()
            plt.show()
        else:
            print("Warning: Cannot visualize acc_x due to zero variance or insufficient data.")

detect_outliers(train)




