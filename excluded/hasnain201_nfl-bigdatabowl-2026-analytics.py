# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")


def generate_synthetic_tracking_data(n_plays=100, frames_per_play=50):
    """
    Generate synthetic NFL tracking data for ATDC analysis.
    
    Parameters:
    -----------
    n_plays : int
        Number of passing plays to simulate
    frames_per_play : int
        Number of tracking frames per play
    
    Returns:
    --------
    tracking_df : DataFrame
        Synthetic tracking data
    """
    np.random.seed(42)
    
    data = []
    
    for play_id in range(1, n_plays + 1):
        # Simulate ball throw timing (frame when pass is thrown)
        throw_frame = np.random.randint(10, 20)
        catch_frame = throw_frame + np.random.randint(15, 35)  # Air time: 1.5-3.5 seconds
        
        # Generate receiver trajectory
        receiver_start_x = np.random.uniform(20, 40)
        receiver_start_y = np.random.uniform(20, 33)  # Field width
        receiver_route_type = np.random.choice(['vertical', 'slant', 'out', 'post'])
        
        # Generate defender starting position
        defender_start_x = receiver_start_x + np.random.uniform(-5, 2)
        defender_start_y = receiver_start_y + np.random.uniform(-3, 3)
        
        for frame in range(frames_per_play):
            time = frame * 0.1  # 10 Hz tracking data
            
            # Receiver movement
            if receiver_route_type == 'vertical':
                receiver_x = receiver_start_x + (frame / frames_per_play) * 20
                receiver_y = receiver_start_y + np.random.normal(0, 0.3)
            elif receiver_route_type == 'slant':
                receiver_x = receiver_start_x + (frame / frames_per_play) * 15
                receiver_y = receiver_start_y + (frame / frames_per_play) * 8
            elif receiver_route_type == 'out':
                if frame < frames_per_play / 2:
                    receiver_x = receiver_start_x + (frame / frames_per_play) * 12
                    receiver_y = receiver_start_y
                else:
                    receiver_x = receiver_start_x + (frames_per_play / 2 / frames_per_play) * 12
                    receiver_y = receiver_start_y + ((frame - frames_per_play/2) / frames_per_play) * 10
            else:  # post
                receiver_x = receiver_start_x + (frame / frames_per_play) * 18
                receiver_y = receiver_start_y + np.sin(frame / 10) * 5
            
            # Defender movement (reacting to receiver)
            pursuit_factor = 0.85 + np.random.uniform(-0.1, 0.15)  # Defender effectiveness
            defender_x = defender_start_x + (receiver_x - receiver_start_x) * pursuit_factor
            defender_y = defender_start_y + (receiver_y - receiver_start_y) * pursuit_factor
            
            # Calculate velocities
            receiver_speed = np.random.uniform(4.5, 7.5)  # yards per second
            defender_speed = np.random.uniform(4.0, 7.0)
            
            # Event tracking
            event = None
            if frame == throw_frame:
                event = 'pass_forward'
            elif frame == catch_frame:
                event = 'pass_arrived'
            
            # Receiver data
            data.append({
                'gameId': 1,
                'playId': play_id,
                'nflId': 1000 + play_id,  # Receiver ID
                'frameId': frame,
                'time': time,
                'jerseyNumber': 10 + (play_id % 90),
                'position': 'WR',
                'x': receiver_x,
                'y': receiver_y,
                's': receiver_speed,
                'a': np.random.uniform(0, 2),
                'dis': receiver_speed * 0.1,
                'o': np.random.uniform(0, 360),
                'dir': np.random.uniform(0, 360),
                'event': event
            })
            
            # Defender data
            data.append({
                'gameId': 1,
                'playId': play_id,
                'nflId': 2000 + play_id,  # Defender ID
                'frameId': frame,
                'time': time,
                'jerseyNumber': 20 + (play_id % 90),
                'position': 'CB',
                'x': defender_x,
                'y': defender_y,
                's': defender_speed,
                'a': np.random.uniform(0, 2),
                'dis': defender_speed * 0.1,
                'o': np.random.uniform(0, 360),
                'dir': np.random.uniform(0, 360),
                'event': event
            })
    
    tracking_df = pd.DataFrame(data)
    return tracking_df

# Generate synthetic data
print("Generating synthetic tracking data...")
tracking_data = generate_synthetic_tracking_data(n_plays=100, frames_per_play=50)

print(f"\nGenerated {len(tracking_data)} tracking records")
print(f"Number of unique plays: {tracking_data['playId'].nunique()}")
print(f"Number of unique players: {tracking_data['nflId'].nunique()}")
print("\nSample tracking data:")
print(tracking_data.head(10))


def calculate_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_atdc_metric(tracking_df):
    """
    Calculate Air-Time Defense Closure (ATDC) metric for each play.
    
    Parameters:
    -----------
    tracking_df : DataFrame
        Player tracking data with positions and events
    
    Returns:
    --------
    atdc_df : DataFrame
        ATDC metrics for each play
    """
    atdc_results = []
    
    for play_id in tracking_df['playId'].unique():
        play_data = tracking_df[tracking_df['playId'] == play_id]
        
        # Find pass events
        pass_forward_frame = play_data[play_data['event'] == 'pass_forward']['frameId']
        pass_arrived_frame = play_data[play_data['event'] == 'pass_arrived']['frameId']
        
        if len(pass_forward_frame) == 0 or len(pass_arrived_frame) == 0:
            continue
        
        pass_forward_frame = pass_forward_frame.iloc[0]
        pass_arrived_frame = pass_arrived_frame.iloc[0]
        
        # Get receiver and defender positions at pass forward
        forward_data = play_data[play_data['frameId'] == pass_forward_frame]
        receiver_forward = forward_data[forward_data['position'] == 'WR'].iloc[0]
        defender_forward = forward_data[forward_data['position'] == 'CB'].iloc[0]
        
        # Get receiver and defender positions at pass arrived
        arrived_data = play_data[play_data['frameId'] == pass_arrived_frame]
        receiver_arrived = arrived_data[arrived_data['position'] == 'WR'].iloc[0]
        defender_arrived = arrived_data[arrived_data['position'] == 'CB'].iloc[0]
        
        # Calculate distances
        initial_distance = calculate_distance(
            defender_forward['x'], defender_forward['y'],
            receiver_forward['x'], receiver_forward['y']
        )
        
        final_distance = calculate_distance(
            defender_arrived['x'], defender_arrived['y'],
            receiver_arrived['x'], receiver_arrived['y']
        )
        
        # Calculate air time (in seconds)
        air_time = (pass_arrived_frame - pass_forward_frame) * 0.1
        
        # Calculate ATDC
        if air_time > 0:
            atdc = (initial_distance - final_distance) / air_time
        else:
            atdc = 0
        
        # Additional metrics
        closure_distance = initial_distance - final_distance
        closure_percentage = (closure_distance / initial_distance * 100) if initial_distance > 0 else 0
        
        atdc_results.append({
            'playId': play_id,
            'defenderId': defender_forward['nflId'],
            'receiverId': receiver_forward['nflId'],
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'closure_distance': closure_distance,
            'air_time': air_time,
            'atdc': atdc,
            'closure_percentage': closure_percentage,
            'defender_avg_speed': play_data[play_data['position'] == 'CB']['s'].mean(),
            'receiver_avg_speed': play_data[play_data['position'] == 'WR']['s'].mean()
        })
    
    atdc_df = pd.DataFrame(atdc_results)
    return atdc_df

# Calculate ATDC metrics
print("Calculating ATDC metrics...")
atdc_metrics = calculate_atdc_metric(tracking_data)

print(f"\nCalculated ATDC for {len(atdc_metrics)} plays")
print("\nATDC Metric Summary Statistics:")
print(atdc_metrics[['atdc', 'initial_distance', 'final_distance', 'air_time', 'closure_percentage']].describe())

print("\nSample ATDC metrics:")
print(atdc_metrics.head(10))


# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. ATDC Distribution
axes[0, 0].hist(atdc_metrics['atdc'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(atdc_metrics['atdc'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {atdc_metrics["atdc"].mean():.2f}')
axes[0, 0].axvline(atdc_metrics['atdc'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {atdc_metrics["atdc"].median():.2f}')
axes[0, 0].set_xlabel('ATDC (yards/second)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Distribution of Air-Time Defense Closure (ATDC)', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. ATDC vs Air Time
scatter = axes[0, 1].scatter(atdc_metrics['air_time'], atdc_metrics['atdc'], 
                            c=atdc_metrics['initial_distance'], cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel('Air Time (seconds)', fontsize=12)
axes[0, 1].set_ylabel('ATDC (yards/second)', fontsize=12)
axes[0, 1].set_title('ATDC vs Air Time (colored by initial distance)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('Initial Distance (yards)', fontsize=10)
axes[0, 1].grid(alpha=0.3)

# 3. Closure Distance Analysis
axes[1, 0].scatter(atdc_metrics['initial_distance'], atdc_metrics['closure_distance'], 
                   c='coral', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel('Initial Distance (yards)', fontsize=12)
axes[1, 0].set_ylabel('Closure Distance (yards)', fontsize=12)
axes[1, 0].set_title('Initial Distance vs Closure Distance', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Add reference line for perfect closure
max_dist = atdc_metrics['initial_distance'].max()
axes[1, 0].plot([0, max_dist], [0, max_dist], 'r--', alpha=0.5, label='Perfect Closure')
axes[1, 0].legend()

# 4. Closure Percentage Distribution
axes[1, 1].hist(atdc_metrics['closure_percentage'], bins=25, color='mediumseagreen', 
                edgecolor='black', alpha=0.7)
axes[1, 1].axvline(atdc_metrics['closure_percentage'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {atdc_metrics["closure_percentage"].mean():.1f}%')
axes[1, 1].set_xlabel('Closure Percentage (%)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distribution of Closure Percentage', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('atdc_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'atdc_analysis.png'")


def plot_play_tracking(tracking_df, play_id, atdc_metrics):
    """
    Plot tracking data for a specific play with ATDC visualization.
    """
    play_data = tracking_df[tracking_df['playId'] == play_id]
    play_atdc = atdc_metrics[atdc_metrics['playId'] == play_id]
    
    if len(play_atdc) == 0:
        print(f"No ATDC data for play {play_id}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get receiver and defender data
    receiver_data = play_data[play_data['position'] == 'WR']
    defender_data = play_data[play_data['position'] == 'CB']
    
    # Plot receiver path
    ax.plot(receiver_data['x'], receiver_data['y'], 'b-', linewidth=2, label='Receiver Path', alpha=0.7)
    ax.scatter(receiver_data['x'].iloc[0], receiver_data['y'].iloc[0], 
               s=200, c='blue', marker='o', edgecolors='black', linewidth=2, label='Receiver Start', zorder=5)
    
    # Plot defender path
    ax.plot(defender_data['x'], defender_data['y'], 'r-', linewidth=2, label='Defender Path', alpha=0.7)
    ax.scatter(defender_data['x'].iloc[0], defender_data['y'].iloc[0], 
               s=200, c='red', marker='s', edgecolors='black', linewidth=2, label='Defender Start', zorder=5)
    
    # Mark pass forward and pass arrived positions
    pass_forward = play_data[play_data['event'] == 'pass_forward']
    pass_arrived = play_data[play_data['event'] == 'pass_arrived']
    
    if len(pass_forward) > 0:
        receiver_forward = pass_forward[pass_forward['position'] == 'WR'].iloc[0]
        defender_forward = pass_forward[pass_forward['position'] == 'CB'].iloc[0]
        ax.scatter(receiver_forward['x'], receiver_forward['y'], 
                   s=300, c='cyan', marker='*', edgecolors='black', linewidth=2, label='Pass Forward', zorder=6)
        ax.scatter(defender_forward['x'], defender_forward['y'], 
                   s=300, c='orange', marker='*', edgecolors='black', linewidth=2, zorder=6)
        
        # Draw initial distance line
        ax.plot([receiver_forward['x'], defender_forward['x']], 
                [receiver_forward['y'], defender_forward['y']], 
                'g--', linewidth=2, alpha=0.5, label='Initial Distance')
    
    if len(pass_arrived) > 0:
        receiver_arrived = pass_arrived[pass_arrived['position'] == 'WR'].iloc[0]
        defender_arrived = pass_arrived[pass_arrived['position'] == 'CB'].iloc[0]
        ax.scatter(receiver_arrived['x'], receiver_arrived['y'], 
                   s=300, c='lime', marker='P', edgecolors='black', linewidth=2, label='Pass Arrived', zorder=6)
        ax.scatter(defender_arrived['x'], defender_arrived['y'], 
                   s=300, c='yellow', marker='P', edgecolors='black', linewidth=2, zorder=6)
        
        # Draw final distance line
        ax.plot([receiver_arrived['x'], defender_arrived['x']], 
                [receiver_arrived['y'], defender_arrived['y']], 
                'm--', linewidth=2, alpha=0.5, label='Final Distance')
    
    # Add ATDC information
    atdc_value = play_atdc['atdc'].iloc[0]
    initial_dist = play_atdc['initial_distance'].iloc[0]
    final_dist = play_atdc['final_distance'].iloc[0]
    air_time = play_atdc['air_time'].iloc[0]
    closure_pct = play_atdc['closure_percentage'].iloc[0]
    
    info_text = f"ATDC: {atdc_value:.2f} yds/s\n"
    info_text += f"Initial Distance: {initial_dist:.2f} yds\n"
    info_text += f"Final Distance: {final_dist:.2f} yds\n"
    info_text += f"Air Time: {air_time:.2f} s\n"
    info_text += f"Closure: {closure_pct:.1f}%"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('X Position (yards)', fontsize=12)
    ax.set_ylabel('Y Position (yards)', fontsize=12)
    ax.set_title(f'Play {play_id}: Defender Pursuit Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'play_{play_id}_tracking.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot a sample play
sample_play_id = atdc_metrics['playId'].iloc[0]
print(f"Visualizing play {sample_play_id}...")
plot_play_tracking(tracking_data, sample_play_id, atdc_metrics)


def engineer_features(atdc_df):
    """
    Engineer features for machine learning models.
    """
    features = atdc_df.copy()
    
    # Speed differential
    features['speed_differential'] = features['defender_avg_speed'] - features['receiver_avg_speed']
    
    # Distance ratio
    features['distance_ratio'] = features['final_distance'] / features['initial_distance']
    
    # Pursuit efficiency
    features['pursuit_efficiency'] = features['closure_distance'] / (features['air_time'] * features['defender_avg_speed'])
    
    # Categorize ATDC performance
    features['atdc_category'] = pd.cut(features['atdc'], 
                                       bins=[-np.inf, 0, 0.5, 1.0, np.inf],
                                       labels=['Poor', 'Average', 'Good', 'Excellent'])
    
    return features

# Engineer features
features_df = engineer_features(atdc_metrics)

print("Engineered Features:")
print(features_df[['playId', 'atdc', 'speed_differential', 'distance_ratio', 
                    'pursuit_efficiency', 'atdc_category']].head(10))

print("\nATDC Category Distribution:")
print(features_df['atdc_category'].value_counts())


# Prepare data for modeling
feature_columns = ['initial_distance', 'air_time', 'defender_avg_speed', 
                   'receiver_avg_speed', 'speed_differential']

X = features_df[feature_columns]
y = features_df['atdc']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"\nFeatures used: {feature_columns}")


# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:")
print(f"RMSE: {lr_rmse:.4f}")
print(f"MAE: {lr_mae:.4f}")
print(f"R² Score: {lr_r2:.4f}")

# Feature importance
feature_importance_lr = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Importance (Linear Regression):")
print(feature_importance_lr)


# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print("Random Forest Results:")
print(f"RMSE: {rf_rmse:.4f}")
print(f"MAE: {rf_mae:.4f}")
print(f"R² Score: {rf_r2:.4f}")

# Feature importance
feature_importance_rf = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance_rf)


# Train Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate
gb_mse = mean_squared_error(y_test, y_pred_gb)
gb_rmse = np.sqrt(gb_mse)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_r2 = r2_score(y_test, y_pred_gb)

print("Gradient Boosting Results:")
print(f"RMSE: {gb_rmse:.4f}")
print(f"MAE: {gb_mae:.4f}")
print(f"R² Score: {gb_r2:.4f}")

# Feature importance
feature_importance_gb = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Gradient Boosting):")
print(feature_importance_gb)


# Create comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model Performance Comparison
models = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
rmse_scores = [lr_rmse, rf_rmse, gb_rmse]
r2_scores = [lr_r2, rf_r2, gb_r2]

x_pos = np.arange(len(models))
axes[0, 0].bar(x_pos, rmse_scores, color=['steelblue', 'coral', 'mediumseagreen'], alpha=0.7, edgecolor='black')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models)
axes[0, 0].set_ylabel('RMSE', fontsize=12)
axes[0, 0].set_title('Model Performance Comparison (RMSE)', fontsize=14, fontweight='bold')
axes[0, 0].grid(alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(rmse_scores):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. R² Score Comparison
axes[0, 1].bar(x_pos, r2_scores, color=['steelblue', 'coral', 'mediumseagreen'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(models)
axes[0, 1].set_ylabel('R² Score', fontsize=12)
axes[0, 1].set_title('Model Performance Comparison (R²)', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(r2_scores):
    axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# 3. Predicted vs Actual (Random Forest)
axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.6, c='coral', s=80, edgecolors='black', linewidth=0.5)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual ATDC', fontsize=12)
axes[1, 0].set_ylabel('Predicted ATDC', fontsize=12)
axes[1, 0].set_title(f'Random Forest: Predicted vs Actual (R²={rf_r2:.4f})', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Feature Importance (Random Forest)
feature_importance_rf_sorted = feature_importance_rf.sort_values('Importance')
axes[1, 1].barh(range(len(feature_importance_rf_sorted)), feature_importance_rf_sorted['Importance'], 
                color='mediumseagreen', alpha=0.7, edgecolor='black')
axes[1, 1].set_yticks(range(len(feature_importance_rf_sorted)))
axes[1, 1].set_yticklabels(feature_importance_rf_sorted['Feature'])
axes[1, 1].set_xlabel('Importance', fontsize=12)
axes[1, 1].set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Model comparison visualization saved as 'model_comparison.png'")


# Perform cross-validation for each model
print("Performing 5-fold cross-validation...\n")

# Linear Regression CV
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
lr_cv_rmse = np.sqrt(-lr_cv_scores)

print("Linear Regression Cross-Validation:")
print(f"Mean RMSE: {lr_cv_rmse.mean():.4f} (+/- {lr_cv_rmse.std():.4f})")

# Random Forest CV
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
rf_cv_rmse = np.sqrt(-rf_cv_scores)

print("\nRandom Forest Cross-Validation:")
print(f"Mean RMSE: {rf_cv_rmse.mean():.4f} (+/- {rf_cv_rmse.std():.4f})")

# Gradient Boosting CV
gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
gb_cv_rmse = np.sqrt(-gb_cv_scores)

print("\nGradient Boosting Cross-Validation:")
print(f"Mean RMSE: {gb_cv_rmse.mean():.4f} (+/- {gb_cv_rmse.std():.4f})")

# Visualize CV results
fig, ax = plt.subplots(figsize=(10, 6))

cv_data = [lr_cv_rmse, rf_cv_rmse, gb_cv_rmse]
bp = ax.boxplot(cv_data, labels=['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting'],
                patch_artist=True, showmeans=True)

# Customize colors
colors = ['steelblue', 'coral', 'mediumseagreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Cross-Validation RMSE', fontsize=12)
ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nCross-validation visualization saved as 'cross_validation_results.png'")


# Export ATDC metrics to CSV
features_df.to_csv('atdc_metrics_with_features.csv', index=False)
print("ATDC metrics exported to 'atdc_metrics_with_features.csv'")

# Create summary report
summary_stats = {
    'Metric': ['Count', 'Mean ATDC', 'Median ATDC', 'Std Dev ATDC', 
               'Mean Initial Distance', 'Mean Final Distance', 'Mean Air Time',
               'Mean Closure Percentage'],
    'Value': [
        len(features_df),
        features_df['atdc'].mean(),
        features_df['atdc'].median(),
        features_df['atdc'].std(),
        features_df['initial_distance'].mean(),
        features_df['final_distance'].mean(),
        features_df['air_time'].mean(),
        features_df['closure_percentage'].mean()
    ]
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('atdc_summary_report.csv', index=False)
print("Summary report exported to 'atdc_summary_report.csv'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Total plays analyzed: {len(features_df)}")
print(f"Mean ATDC: {features_df['atdc'].mean():.3f} yards/second")
print(f"Best model: Random Forest (R² = {rf_r2:.4f})")
print("\nOutput files generated:")
print("  - atdc_analysis.png")
print("  - play_1_tracking.png")
print("  - model_comparison.png")
print("  - cross_validation_results.png")
print("  - atdc_metrics_with_features.csv")
print("  - atdc_summary_report.csv")

