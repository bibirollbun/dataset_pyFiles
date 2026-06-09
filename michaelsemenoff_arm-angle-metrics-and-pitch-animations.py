import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/nwds-k/train.csv')

train.head()
train.info()


# filter train so that column 'strikes' == 2
train_2 = train[train['strikes'] == 2]

#drop rows with missing values
train_2 = train_2.dropna()  


def generate_new_metrics(df):
    """
    Generate new physics-related pitching metrics that incorporate 'arm_angle'.
    The input dataframe must contain the following columns:
    'sz_top', 'sz_bot', 'pfx_x', 'pfx_z', 'arm_angle', 
    'release_speed', 'release_pos_x', 'release_extension', 
    'release_pos_z', 'release_spin_rate', 'spin_axis'
    
    Returns the dataframe with new feature columns added.
    """
    # Calculate the horizontal component of the release speed, influenced by arm angle.
    df['velocity_horizontal'] = df['release_speed'] * np.cos(np.radians(df['arm_angle']))
    
    # Calculate the vertical component of the release speed.
    df['velocity_vertical'] = df['release_speed'] * np.sin(np.radians(df['arm_angle']))
    
    # Create a metric that adjusts spin rate based on the arm angle.
    df['spin_effect'] = df['release_spin_rate'] * np.sin(np.radians(df['arm_angle']))
    
    # Create a metric that combines release extension and arm angle to capture effective pitch extension.
    df['extension_control'] = df['release_extension'] * np.cos(np.radians(df['arm_angle']))
    
    # Create a metric that adjusts the strike zone based on the vertical distance between sz_top and sz_bot.
    df['strike_zone_adjustment'] = (df['sz_top'] - df['sz_bot']) * np.sin(np.radians(df['arm_angle']))
    
    return df

train_df = generate_new_metrics(train_2)
train_df.head()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
%matplotlib inline
plt.style.use('default')  # Use default style for consistent display
warnings.filterwarnings('ignore')  # Suppress deprecation warnings

# Set style using newer seaborn API
sns.set_theme(style="whitegrid")

# Get descriptive statistics
stats = train_df[['velocity_horizontal', 'velocity_vertical', 'spin_effect', 
                  'extension_control', 'strike_zone_adjustment']].describe()
print("Descriptive Statistics:")
print(stats)

# Create figure with subplots and adjust spacing
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Distribution of New Baseball Metrics by Strike/Ball', fontsize=16, y=1.02)

# Plot 1: Velocity components with colorbar
scatter1 = axes[0,0].scatter(train_df['velocity_horizontal'], 
                           train_df['velocity_vertical'],
                           c=train_df['is_strike'], 
                           alpha=0.5, 
                           cmap='coolwarm')
axes[0,0].set_xlabel('Horizontal Velocity (mph)')
axes[0,0].set_ylabel('Vertical Velocity (mph)')
axes[0,0].set_title('Velocity Components')
fig.colorbar(scatter1, ax=axes[0,0], label='Is Strike')

# Plot 2: Spin Effect vs Extension Control with colorbar
scatter2 = axes[0,1].scatter(train_df['extension_control'], 
                           train_df['spin_effect'],
                           c=train_df['is_strike'], 
                           alpha=0.5, 
                           cmap='coolwarm')
axes[0,1].set_xlabel('Extension Control')
axes[0,1].set_ylabel('Spin Effect')
axes[0,1].set_title('Spin Effect vs Extension Control')
fig.colorbar(scatter2, ax=axes[0,1], label='Is Strike')

# Plot 3: Strike Zone Adjustment Distribution
sns.kdeplot(data=train_df, 
            x='strike_zone_adjustment', 
            hue='is_strike',
            ax=axes[1,0],
            common_norm=False)
axes[1,0].set_title('Strike Zone Adjustment Distribution')
axes[1,0].legend(title='Is Strike', labels=['Ball', 'Strike'])

# Plot 4: Box plots
metrics = ['velocity_horizontal', 'velocity_vertical', 'spin_effect']
box_data = []
labels = []
for metric in metrics:
    box_data.append(train_df[train_df['is_strike']==1][metric])
    box_data.append(train_df[train_df['is_strike']==0][metric])
    labels.extend([f'{metric}\nStrike', f'{metric}\nBall'])
    
axes[1,1].boxplot(box_data, labels=labels)
axes[1,1].set_title('Metrics Distribution by Outcome')
axes[1,1].tick_params(axis='x', rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
plt.close()  # Clean up memory


# Configure matplotlib for inline plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arc
from IPython.display import HTML
%matplotlib inline

def plot_pitch_trajectory(df, row_idx):
    """
    Creates a 3D animation of pitch trajectory with three fixed views side by side
    """
    pitch = df.iloc[row_idx]
    
    # MLB dimensions
    MOUND_TO_PLATE = 60.5
    MOUND_HEIGHT = 0.417
    MOUND_RADIUS = 5
    PLATE_WIDTH = 1.42
    
    # Create single figure with three views
    fig = plt.figure(figsize=(20, 6))
    
    # Create three different viewing angles
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Set fixed views
    ax1.view_init(elev=0, azim=-90)    # Side view
    ax2.view_init(elev=90, azim=-90)   # Top view
    ax3.view_init(elev=20, azim=-120)  # Behind plate view
    
    axes = [ax1, ax2, ax3]
    titles = ['Side View', 'Top View', 'Behind Plate View']
    
    for ax, title in zip(axes, titles):
        # Draw pitcher's mound as semicircle
        theta = np.linspace(0, np.pi, 50)  # Semicircle
        x = MOUND_RADIUS * np.cos(theta)
        y = np.zeros_like(theta)
        z = MOUND_RADIUS * np.sin(theta)
        ax.plot(x, y, z, 'k-', alpha=0.3)
        
        # Draw home plate
        plate_coords = np.array([
            [-PLATE_WIDTH/2, MOUND_TO_PLATE, 0],
            [PLATE_WIDTH/2, MOUND_TO_PLATE, 0],
            [0, MOUND_TO_PLATE + PLATE_WIDTH/2, 0],
            [-PLATE_WIDTH/2, MOUND_TO_PLATE, 0]
        ])
        ax.plot(plate_coords[:,0], plate_coords[:,1], plate_coords[:,2], 'k-')
        
        # Draw strike zone
        sz_top = pitch.sz_top
        sz_bot = pitch.sz_bot
        zone_corners = np.array([
            [-PLATE_WIDTH/2, MOUND_TO_PLATE, sz_bot],
            [PLATE_WIDTH/2, MOUND_TO_PLATE, sz_bot],
            [PLATE_WIDTH/2, MOUND_TO_PLATE, sz_top],
            [-PLATE_WIDTH/2, MOUND_TO_PLATE, sz_top],
            [-PLATE_WIDTH/2, MOUND_TO_PLATE, sz_bot]
        ])
        ax.plot(zone_corners[:,0], zone_corners[:,1], zone_corners[:,2], 'r--', alpha=0.5)
        
        # Set labels and limits
        ax.set_xlabel('X (ft)')
        ax.set_ylabel('Y (ft)')
        ax.set_zlabel('Z (ft)')
        ax.set_title(f'{title}\n{pitch.pitch_name}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, MOUND_TO_PLATE + 5)
        ax.set_zlim(0, 8)
    
    # Calculate trajectory
    t = np.linspace(0, 0.5, 40)
    g = 32.174
    v0 = pitch.release_speed * 1.467
    x0 = pitch.release_pos_x
    z0 = pitch.release_pos_z
    
    x = x0 + pitch.pfx_x * t
    y = t * v0 * np.cos(np.radians(pitch.release_pos_z))
    z = z0 + pitch.pfx_z * t - 0.5 * g * t**2
    
    # Initialize trajectories and balls
    trajectories = []
    balls = []
    for ax in axes:
        traj, = ax.plot(x, y, z, 'b-', alpha=0.6)
        ball = ax.plot([x[0]], [y[0]], [z[0]], 'o', 
                      color='white', markeredgecolor='red', 
                      markersize=20)[0]
        trajectories.append(traj)
        balls.append(ball)
    
    def update(frame):
        for ball in balls:
            ball.set_data_3d([x[frame]], [y[frame]], [z[frame]])
        return balls
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(t),
                        interval=50, blit=True)
    
    # Add pitch information as text box instead of suptitle
    text_str = (f"Pitch: {pitch.pitch_name}\n"
                f"Speed: {pitch.release_speed:.1f} mph\n"
                f"Spin Rate: {pitch.release_spin_rate:.0f} rpm")
    fig.text(0.02, 0.98, text_str,
             transform=fig.transFigure,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Close any existing figures to prevent duplicates
    plt.close()
    
    return HTML(anim.to_jshtml())

# Create and display single animation with three views
anim = plot_pitch_trajectory(train_df, 0) #change index from 0 to view other pitches
display(anim)

