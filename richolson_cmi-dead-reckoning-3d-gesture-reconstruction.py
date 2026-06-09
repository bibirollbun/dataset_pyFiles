!pip install ahrs -q

import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation as R

# Madgwick filter (only needed if gyro present & quaternions absent)
from ahrs.filters import Madgwick

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

kaggle_kernel_run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if kaggle_kernel_run_type != 'Interactive': 
    pio.renderers.default = "notebook"
print("Pio renderer set to:", pio.renderers.default)


raw = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")


ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ORI_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def integrate_imu_sequence(imu, *, verbose: bool = False, enable_filtering: bool = True, use_orientation: bool = True):
    """
    Integrate IMU data using raw AC acceleration signal with optional orientation support.
    
    Args:
        imu (pd.DataFrame): IMU data for a single sequence.
        verbose (bool, optional): If True, print diagnostic information.
                                  Defaults to False.
        enable_filtering (bool, optional): If True, apply high-pass filtering.
                                         If False, use raw integration (will drift).
                                         Defaults to True.
        use_orientation (bool, optional): If True, use quaternion data to transform
                                        to world frame and remove gravity.
                                    
    Returns
    -------
    dict
        Dictionary containing position, velocity, acceleration, and metadata.
    """
    
    # === CONFIGURATION PARAMETERS ===
    # High-pass filter settings (only used if enable_filtering=True)
    HIGHPASS_CUTOFF_HZ = 3.0    # Frequency cutoff in Hz
    
    FILTER_ORDER = 1            # Butterworth filter order
                                # - Order 1: Gentle rolloff, minimal phase distortion
                                # - Order 2: Steeper rolloff, good balance (recommended)
                                # - Order 3+: Very steep rolloff, may introduce artifacts
    
    MIN_SAMPLES_FOR_FILTER = 10 # Minimum number of samples required to apply filtering
                                # - Filters need sufficient data to work properly
                                # - Increase if you get filter instability warnings
                                # - Decrease only if working with very short sequences
    
    DEFAULT_SAMPLE_RATE = 50    # Assumed sample rate in Hz if timestamps unavailable
                                # - Common IMU rates: 25, 50, 100, 200, 1000 Hz
                                # - Should match your actual IMU sampling rate
    
    # Calculate time step
    if "timestamp" in imu.columns:
        imu["timestamp"] = imu["timestamp"].astype(float) * 1e-6  # Âµs â†’ s
        dt = np.median(imu["timestamp"].diff().dropna())
    else:
        dt = 1 / DEFAULT_SAMPLE_RATE
    if verbose:
        print(f"Î”t â‰ˆ {dt:.4f} s  (~{1/dt:.1f} Hz)")
    
    # Get raw acceleration data (AC signal only)
    acc_raw = imu[ACC_COLS].values
    
    # === OPTIONAL ORIENTATION PROCESSING ===
    if use_orientation and all(col in imu.columns for col in ORI_COLS):
        quaternions = imu[ORI_COLS].values  # [w, x, y, z]
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)  # Normalize
        
        # Transform to world frame and remove gravity
        gravity_world = np.array([0, 0, 9.81])
        acc_processed = np.zeros_like(acc_raw)
        
        for i in range(len(acc_raw)):
            R = quaternion_to_rotation_matrix(quaternions[i])
            acc_processed[i] = R.T @ acc_raw[i] - gravity_world
        
        if verbose:
            print("Applied orientation transformation and gravity removal")
    else:
        acc_processed = acc_raw
        if verbose:
            print("Using raw body-frame acceleration (no orientation processing)")
    
    if verbose:
        print("Acceleration statistics:")
        print(f"  Mean: {acc_processed.mean(axis=0)}")
        print(f"  Std:  {acc_processed.std(axis=0)}")
        print(f"  Range: X=[{acc_processed[:,0].min():.2f}, {acc_processed[:,0].max():.2f}]")
        print(f"         Y=[{acc_processed[:,1].min():.2f}, {acc_processed[:,1].max():.2f}]")
        print(f"         Z=[{acc_processed[:,2].min():.2f}, {acc_processed[:,2].max():.2f}]")
    
    # === FILTERING SECTION ===
    if enable_filtering:
        # High-pass filter: Remove low-frequency drift components
        if len(acc_processed) > MIN_SAMPLES_FOR_FILTER:
            nyquist = 0.5 / dt
            normal_cutoff = HIGHPASS_CUTOFF_HZ / nyquist
            
            # Design Butterworth high-pass filter
            b_hp, a_hp = butter(FILTER_ORDER, normal_cutoff, btype="high", analog=False)
            
            # Apply high-pass filter to acceleration data
            acc_filtered = np.zeros_like(acc_processed)
            for axis in range(3):
                acc_filtered[:, axis] = filtfilt(b_hp, a_hp, acc_processed[:, axis])
            
            if verbose:
                print(f"Applied {HIGHPASS_CUTOFF_HZ} Hz high-pass filter (order {FILTER_ORDER}) to acceleration")
        else:
            acc_filtered = acc_processed
            if verbose:
                print(f"Skipped filtering (only {len(acc_processed)} samples, need >{MIN_SAMPLES_FOR_FILTER})")
    else:
        acc_filtered = acc_processed
        if verbose:
            print("Filtering disabled - using processed acceleration data")
    
    # === INTEGRATION SECTION ===
    # First integration: acceleration â†’ velocity
    vel = np.vstack((np.zeros(3), cumulative_trapezoid(acc_filtered, dx=dt, axis=0)))
    
    # Apply same high-pass filter to velocity if filtering is enabled
    if enable_filtering and len(vel) > MIN_SAMPLES_FOR_FILTER:
        vel_filtered = np.zeros_like(vel)
        for axis in range(3):
            vel_filtered[:, axis] = filtfilt(b_hp, a_hp, vel[:, axis])
        vel = vel_filtered
        
        if verbose:
            print(f"Applied {HIGHPASS_CUTOFF_HZ} Hz high-pass filter to velocity")
    elif verbose and enable_filtering:
        print(f"Skipped velocity filtering (only {len(vel)} samples)")
    elif verbose:
        print("Velocity filtering disabled")
    
    # Second integration: velocity â†’ position
    pos = np.vstack((np.zeros(3), cumulative_trapezoid(vel, dx=dt, axis=0)))
    
    # Trajectory centering: Remove DC offset from position
    pos_centered = pos - pos.mean(axis=0)
    
    if verbose:
        print("Final position range:")
        print(f"  X: {pos_centered[:,0].min():.3f} to {pos_centered[:,0].max():.3f} m")
        print(f"  Y: {pos_centered[:,1].min():.3f} to {pos_centered[:,1].max():.3f} m")
        print(f"  Z: {pos_centered[:,2].min():.3f} to {pos_centered[:,2].max():.3f} m")
    
    # Map behaviors to integer IDs
    behaviors = imu["behavior"].astype("category")
    imu["beh_id"] = behaviors.cat.codes
    
    return {
        "pos": pos_centered,
        "vel": vel,
        "acc_world": acc_filtered,  # Processed acceleration (raw, or world-frame with gravity removed)
        "imu": imu,
        "dt": dt,
        "gravity_estimate": 9.81 if use_orientation else None,
        "filtering_enabled": enable_filtering,
        "orientation_used": use_orientation and all(col in imu.columns for col in ORI_COLS),
    }


def create_trajectory_animation(integration_results, title=None, gesture_only=True):
    """
    Create a memory-efficient animated 3D trajectory plot from IMU integration results.
    
    Args:
        integration_results: dict from integrate_imu_sequence()
        gesture_only: if True, only plot data where behavior == "Performs gesture"
    
    Returns:
        plotly Figure object
    """
    pos = integration_results['pos']
    imu = integration_results['imu']
    
    # Filter to gesture behaviors only
    if gesture_only and 'behavior' in imu.columns:
        gesture_mask = imu['behavior'] == "Performs gesture"
        if gesture_mask.any():
            pos = pos[gesture_mask]
            imu = imu[gesture_mask].reset_index(drop=True)
    
    # Get quaternion data (keeping for potential future use)
    q_wxyz = imu[ORI_COLS].values
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]  # wxyz â†’ xyzw
    rotations = R.from_quat(q_xyzw)
    
    # Calculate axis ranges with minimal padding
    mins, maxs = pos.min(axis=0), pos.max(axis=0)
    span = maxs - mins
    pad = 0.05 * span + 1e-3
    x_rng = [mins[0]-pad[0], maxs[0]+pad[0]]
    y_rng = [mins[1]-pad[1], maxs[1]+pad[1]]
    z_rng = [mins[2]-pad[2], maxs[2]+pad[2]]
    
    # Set up camera
    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=0.8))
    
    # Color setup - "Performs gesture" always gets blue (ID 0)
    behaviors = imu["behavior"].astype("category")
    unique_behaviors = list(behaviors.cat.categories)
    
    # Create behavior-to-ID mapping with "Performs gesture" always ID 0 (blue)
    behavior_to_id = {}
    
    # Always assign "Performs gesture" as ID 0 (blue)
    if "Performs gesture" in unique_behaviors:
        behavior_to_id["Performs gesture"] = 0
    
    # Get non-gesture behaviors in order of first appearance
    non_gesture_behaviors = []
    seen_behaviors = set()
    
    for behavior in imu["behavior"]:
        if behavior != "Performs gesture" and behavior not in seen_behaviors:
            non_gesture_behaviors.append(behavior)
            seen_behaviors.add(behavior)
    
    # Assign IDs to non-gesture behaviors starting from 1
    for i, behavior in enumerate(non_gesture_behaviors):
        behavior_to_id[behavior] = i + 1
    
    # Map behaviors to IDs
    beh_ids = behaviors.map(behavior_to_id).values
    
    # Convert to numpy array to avoid categorical issues
    beh_ids = np.array(beh_ids)
    
    # Define colors: blue for gesture, red for first non-gesture, green for second non-gesture
    behavior_colors = ["#1f77b4", "#ff0000", "#00ff00"]  # blue, red, green
    # Add more colors if needed
    while len(behavior_colors) < len(unique_behaviors):
        behavior_colors.extend(["#ff8c00", "#8a2be2", "#00ffff"])  # orange, purple, cyan
    
    # Build colorscale
    colorscale = []
    num_behaviors = len(unique_behaviors)
    for i in range(num_behaviors):
        color = behavior_colors[i]
        if num_behaviors == 1:
            colorscale = [[0, color], [1, color]]
        else:
            frac_start = i / num_behaviors
            frac_end = (i + 1) / num_behaviors
            colorscale.extend([[frac_start, color], [frac_end, color]])
    
    # MEMORY OPTIMIZATION: Reduce number of frames significantly
    # Use fewer frames but with smoother transitions
    max_frames = min(150, len(pos))  # Cap at 150 frames maximum
    frame_indices = np.linspace(0, len(pos)-1, max_frames, dtype=int)
    
    def create_frame_data(end_idx):
        """Create data for frame ending at end_idx"""
        # Only store what's needed for this frame
        return [
            go.Scatter3d(
                x=pos[:end_idx+1, 0], 
                y=pos[:end_idx+1, 1], 
                z=pos[:end_idx+1, 2],
                mode="lines",
                line=dict(
                    width=4, 
                    color=beh_ids[:end_idx+1], 
                    colorscale=colorscale, 
                    cmin=0, 
                    cmax=num_behaviors-1
                ),
                showlegend=False,
                # Memory optimization: don't store unnecessary properties
                hoverinfo='skip'
            )
        ]
    
    # Create frames with reduced memory footprint
    frames = []
    for idx in frame_indices:
        frame = go.Frame(
            data=create_frame_data(idx),
            layout=dict(scene=dict(
                xaxis=dict(range=x_rng, autorange=False),
                yaxis=dict(range=y_rng, autorange=False), 
                zaxis=dict(range=z_rng, autorange=False),
                aspectmode="cube"
            )),
            # Memory optimization: don't store frame names
            name=None
        )
        frames.append(frame)
    
    # Initial figure - show complete trajectory
    initial_data = [go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="lines",
        line=dict(width=4, color=beh_ids, colorscale=colorscale, cmin=0, cmax=num_behaviors-1),
        showlegend=False,
        hoverinfo='skip'  # Disable hover to save memory
    )]
    
    fig = go.Figure(
        data=initial_data,
        frames=frames,
        layout=go.Layout(
            width=900, height=700, autosize=False,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            scene=dict(
                xaxis=dict(range=x_rng, autorange=False, showticklabels=False, title="", 
                          showgrid=True, gridcolor="lightgray", gridwidth=1),
                yaxis=dict(range=y_rng, autorange=False, showticklabels=False, title="",
                          showgrid=True, gridcolor="lightgray", gridwidth=1),
                zaxis=dict(range=z_rng, autorange=False, showticklabels=False, title="",
                          showgrid=True, gridcolor="lightgray", gridwidth=1),
                aspectmode="cube", camera=camera
            ),
            dragmode="orbit",
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", 
                         args=[None, {"frame": {"duration": 50, "redraw": True}, "transition": {"duration": 20}}]),
                    dict(label="Pause", method="animate", 
                         args=[[None], {"frame": {"duration": 0, "redraw": False}}])
                ],
                showactive=False, x=0.02, y=0.98
            )]
        )
    )
    
    return fig


# Count samples by gesture and orientation
counts = raw.groupby(['gesture', 'orientation']).size().reset_index(name='samples')

# Get unique lists for numbering
unique_gestures = sorted(counts['gesture'].unique())
unique_orientations = sorted(counts['orientation'].unique())

print("=== GESTURE-ORIENTATION SAMPLE COUNTS ===")

# List orientations once
print("\nORIENTATIONS:")
for i, orientation in enumerate(unique_orientations, 1):
    print(f"  {i}. {orientation}")

print("\nGESTURES (AVAIL. ORIENTATIONS):")
for i, gesture in enumerate(unique_gestures, 1):
    # Get which orientations are present for this gesture
    present_orientations = []
    for ori_num, orientation in enumerate(unique_orientations, 1):
        gesture_data = counts[(counts['gesture'] == gesture) & (counts['orientation'] == orientation)]
        if len(gesture_data) > 0:  # Only include if there are samples
            present_orientations.append(str(ori_num))
    
    print(f"  {gesture}: {', '.join(present_orientations)}")


def visualize_by_index(gesture_name, orientation=None, samples=1, gesture_only=True):
    """
    Visualize sequences by gesture and orientation indices.
        """    
    # Filter by gesture, and optionally by orientation
    filter_func = lambda x: x['gesture'].iloc[0] == gesture_name
    if orientation is not None:
        orientation_name = unique_orientations[orientation - 1]
        filter_func = lambda x: x['gesture'].iloc[0] == gesture_name and x['orientation'].iloc[0] == orientation_name
    
    matches = raw.groupby('sequence_id').filter(filter_func)
    sequence_ids = matches['sequence_id'].unique()
    
    if len(sequence_ids) == 0:
        print("â�Œ No matching sequences found")
        return []
    
    # If no specific orientation requested, pick the first available orientation
    if orientation is None:
        # Find the first available orientation for this gesture
        first_seq_id = sequence_ids[0]
        chosen_orientation = raw[raw.sequence_id == first_seq_id]['orientation'].iloc[0]
        
        # Filter to only sequences with this orientation
        filter_func = lambda x: x['gesture'].iloc[0] == gesture_name and x['orientation'].iloc[0] == chosen_orientation
        matches = raw.groupby('sequence_id').filter(filter_func)
        sequence_ids = matches['sequence_id'].unique()
        search_desc = f"{gesture_name} + {chosen_orientation}"
    else:
        search_desc = f"{gesture_name} + {orientation_name}"
    
    # Take the requested number of samples
    selected_sequences = sequence_ids[:samples]
    
    # Display header
    gesture_filter = "Gestures Only" if gesture_only else "Sequence Includes Transitions"
    print(f"\n{'='*80}")
    print(f"ğŸ�¯ {search_desc.upper()}")
    print(f"ğŸ“Š Showing {len(selected_sequences)} sequences ({gesture_filter})")
    print(f"{'='*80}")
        
    figures = []
    for i, seq_id in enumerate(selected_sequences, 1):
        imu = raw[raw.sequence_id == seq_id].copy().reset_index(drop=True)
        results = integrate_imu_sequence(imu)
        actual_orientation = imu['orientation'].iloc[0]
        title = f"{gesture_name} - {actual_orientation} (#{i}, ID:{seq_id})"
        fig = create_trajectory_animation(results, title, gesture_only=gesture_only)
        figures.append(fig)
        fig.show(config=dict(responsive=False))


visualize_by_index("Glasses on/off", samples=3, gesture_only=False)


visualize_by_index("Write name in air", orientation=2, samples=3, gesture_only=True)


visualize_by_index("Forehead - pull hairline", samples=3, gesture_only=False)


visualize_by_index("Wave hello", samples=3, gesture_only=False)


visualize_by_index("Pull air toward your face", samples=3, gesture_only=False)

