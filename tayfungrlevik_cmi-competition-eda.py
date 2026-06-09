# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install pykalman
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import animation
from IPython.display import HTML
import warnings
from matplotlib.animation import FuncAnimation
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
df_subjects=pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
grouped = df.groupby("sequence_id")
print(df["sequence_id"].unique())


def apply_kalman_filter_to_sequence(seq, columns):
    seq_filtered = seq.copy()
    
    for col in columns:
        data = seq_filtered[col].values
        mask = np.isnan(data)

        if mask.sum() == 0:
            continue  # Eksik veri yoksa geç

        # Kalman filtresi tanımı
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        data_filled = kf.em(data, n_iter=10).smooth(data)[0]
        seq_filtered[col] = data_filled

    return seq_filtered

def compute_gravity_vector_from_quaternion(qx, qy, qz, qw):
    # Quaternion'dan gravity vektörü çıkarımı
    gravity_x = 2 * (qx * qz - qw * qy)
    gravity_y = 2 * (qw * qx + qy * qz)
    gravity_z = qw**2 - qx**2 - qy**2 + qz**2
    return np.array([gravity_x, gravity_y, gravity_z])
def remove_gravity_from_acceleration(seq):
    acc_no_gravity = []

    for _, row in seq.iterrows():
        gravity = compute_gravity_vector_from_quaternion(
            row["rot_x"], row["rot_y"],
            row["rot_z"], row["rot_w"]
        )
        acc = np.array([row["acc_x"], row["acc_y"], row["acc_z"]])
        acc_corrected = acc - gravity
        acc_no_gravity.append(acc_corrected)

    acc_no_gravity = np.array(acc_no_gravity)
    seq["acc_x_nograv"] = acc_no_gravity[:, 0]
    seq["acc_y_nograv"] = acc_no_gravity[:, 1]
    seq["acc_z_nograv"] = acc_no_gravity[:, 2]
    return seq


def remove_gravity_from_acceleration(seq):
    acc = seq[["acc_x", "acc_y", "acc_z"]].values
    quat = seq[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    rotation = R.from_quat(quat)

    # Yerçekimi vektörü (dünya koordinatlarında [0, 0, -9.81])
    gravity_world = np.array([9.81, 0, 0])
    gravity_sensor = rotation.apply(gravity_world, inverse=True)

    # Gerçek ivme = ölçülen ivme - yerçekimi
    acc_corrected = acc - gravity_sensor
    seq["acc_x_nograv"], seq["acc_y_nograv"], seq["acc_z_nograv"] = acc_corrected.T
    return seq
def estimate_position_xyz_from_seq(seq, dt=0.01):
    # Gravity'den arındırılmış ivme verileri
    acc_x = seq["acc_x_nograv"].values
    acc_y = seq["acc_y_nograv"].values
    acc_z = seq["acc_z_nograv"].values

    # Hız ve konum entegrasyonu
    velocity_x = np.cumsum(acc_x * dt)
    velocity_y = np.cumsum(acc_y * dt)
    velocity_z = np.cumsum(acc_z * dt)

    position_x = np.cumsum(velocity_x * dt)
    position_y = np.cumsum(velocity_y * dt)
    position_z = np.cumsum(velocity_z * dt)

    return position_x, position_y, position_z

def animate_combined_sequence(seq, pixels, interval=100):
    warnings.filterwarnings("ignore")  

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    def update(i):
        axs[0].clear()
        axs[1].clear()

        # IMU ivme
        axs[0].plot(seq["sequence_counter"][:i], seq["acc_x"][:i], label="acc_x")
        axs[0].plot(seq["sequence_counter"][:i], seq["acc_y"][:i], label="acc_y")
        axs[0].plot(seq["sequence_counter"][:i], seq["acc_z"][:i], label="acc_z")
        axs[0].legend(); axs[0].set_title(f"IMU İvme (Frame {i})")

        # ToF grid
        grid = np.array(seq.iloc[i][pixels].fillna(0).astype(float)).reshape(8, 8)
        sns.heatmap(grid, cmap="viridis", cbar=False, ax=axs[1])
        axs[1].set_title("ToF Mesafe Heatmap")

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(seq)), interval=interval)
    return HTML(ani.to_jshtml())
def animate_trajectory_3d_kaggle(x_vals, y_vals, z_vals):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Sensor Trajectory (Gravity Removed)", fontsize=14)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    margin = 0.1
    ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
    ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)
    ax.set_zlim(min(z_vals) - margin, max(z_vals) + margin)

    trajectory, = ax.plot([], [], [], lw=2, color='royalblue')
    point = ax.scatter([], [], [], color='red', s=50)

    def init():
        trajectory.set_data([], [])
        trajectory.set_3d_properties([])
        point._offsets3d = ([], [], [])
        return trajectory, point

    def update(frame):
        trajectory.set_data(x_vals[:frame], y_vals[:frame])
        trajectory.set_3d_properties(z_vals[:frame])
        point._offsets3d = ([x_vals[frame]], [y_vals[frame]], [z_vals[frame]])
        return trajectory, point

    ani = FuncAnimation(fig, update, frames=len(x_vals), init_func=init,
                        interval=40, blit=False)

    return HTML(ani.to_jshtml())
def extract_features_from_sequence(seq):
    features = {}

    # Quaternion → Euler dönüşümü
    if all(col in seq.columns for col in ["quat_w", "quat_x", "quat_y", "quat_z"]):
        quat = seq[["quat_w", "quat_x", "quat_y", "quat_z"]].fillna(0).values
        euler = R.from_quat(quat).as_euler("xyz", degrees=False)
        seq["euler_x"], seq["euler_y"], seq["euler_z"] = euler.T

        for axis in ["euler_x", "euler_y", "euler_z"]:
            values = seq[axis].values
            features[f"{axis}_mean"] = np.mean(values)
            features[f"{axis}_std"] = np.std(values)
            features[f"{axis}_max"] = np.max(values)
            features[f"{axis}_min"] = np.min(values)
            features[f"{axis}_energy"] = np.sum(values ** 2)

    # Acceleration öznitelikleri
    for col in ["acc_x", "acc_y", "acc_z"]:
        if col in seq.columns:
            values = seq[col].values
            features[f"{col}_mean"] = np.mean(values)
            features[f"{col}_std"] = np.std(values)
            features[f"{col}_max"] = np.max(values)
            features[f"{col}_min"] = np.min(values)
            features[f"{col}_energy"] = np.sum(values ** 2)

    # ToF öznitelikleri (örnek: tof_1 sensörü)
    tof_cols = [f"tof_1_v{i}" for i in range(64) if f"tof_1_v{i}" in seq.columns]
    if len(tof_cols) == 64:
        tof_matrix = seq[tof_cols].fillna(0).astype(float).values
        avg_grid = np.mean(tof_matrix, axis=0).reshape(8, 8)
        features["tof_mean_center"] = avg_grid[3:5, 3:5].mean()
        features["tof_entropy"] = -np.sum((avg_grid / avg_grid.sum()) * np.log1p(avg_grid / avg_grid.sum()))

    # Hareket süresi
    if "sequence_counter" in seq.columns:
        features["duration"] = seq["sequence_counter"].max() - seq["sequence_counter"].min()

    return pd.Series(features)


def apply_kalman_filter_to_sequence(seq, columns):
    seq_filtered = seq.copy()
    for col in columns:
        if col in seq.columns:
            values = seq[col].values
            mask = np.isnan(values)
            if np.all(mask):  # tamamen eksikse atla
                continue
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            values_filled = values.copy()
            values_filled[mask] = 0  # NaN'leri geçici olarak sıfırla
            filtered_state_means, _ = kf.smooth(values_filled.reshape(-1, 1))
            seq_filtered[col] = filtered_state_means.ravel()
    return seq_filtered
imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
tof_cols = [f"tof_1_v{i}" for i in range(64)]

def preprocess_sequence(seq):
    cols_to_filter = [col for col in imu_cols + tof_cols if col in seq.columns]
    return apply_kalman_filter_to_sequence(seq, cols_to_filter)



valid_id = df["sequence_id"].unique()[0]  # first seq.
seq = grouped.get_group(valid_id)

plt.plot(seq["sequence_counter"], seq["acc_x"], label="acc_x")
plt.plot(seq["sequence_counter"], seq["acc_y"], label="acc_y")
plt.plot(seq["sequence_counter"], seq["acc_z"], label="acc_z")
plt.legend(); plt.title("Acc/time series")






rot = R.from_quat(seq[["rot_x", "rot_y", "rot_z", "rot_w"]])
euler = rot.as_euler("xyz", degrees=True)
plt.plot(seq["sequence_counter"], euler[:, 0], label="Roll")
plt.plot(seq["sequence_counter"], euler[:, 1], label="Pitch")
plt.plot(seq["sequence_counter"], euler[:, 2], label="Yaw")
plt.legend(); plt.title("IMU Rotations (Euler)")



warnings.filterwarnings("ignore")


valid_id = np.random.choice(df["sequence_id"].unique())


seq = df[df["sequence_id"] == valid_id]
seq = remove_gravity_from_acceleration(seq)
x_vals, y_vals, z_vals = estimate_position_xyz_from_seq(seq)
animate_trajectory_3d_kaggle(x_vals, y_vals, z_vals)





pixels = [f"tof_1_v{i}" for i in range(64)]
grid = np.array(seq.iloc[50][pixels].astype(float)).reshape(8, 8)
sns.heatmap(grid, cmap="viridis", cbar=True)
plt.title("ToF Sensor 1 – Distance Heatmap")
plt.show()




fig, ax = plt.subplots()

def update(i):
    grid = np.array(seq.iloc[i][pixels].astype(float)).reshape(8, 8)
    ax.clear()
    sns.heatmap(grid, cmap="viridis", cbar=False, ax=ax)

ani = animation.FuncAnimation(fig, update, frames=range(0, len(seq)), interval=100)

HTML(ani.to_jshtml())



fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# IMU ivme
axs[0].plot(seq["sequence_counter"], seq["acc_x"], label="acc_x")
axs[0].plot(seq["sequence_counter"], seq["acc_y"], label="acc_y")
axs[0].plot(seq["sequence_counter"], seq["acc_z"], label="acc_z")
axs[0].legend(); axs[0].set_title("IMU Acc/time")

# ToF grid (örnek zaman noktası: 50)
grid = np.array(seq.iloc[50][pixels].astype(float)).reshape(8, 8)
sns.heatmap(grid, cmap="viridis", ax=axs[1], cbar=True)
axs[1].set_title("ToF Sensor – Distance Heatmap")



pixels = [f"tof_1_v{i}" for i in range(64)]
valid_id = df["sequence_id"].unique()[0]
seq = df[df["sequence_id"] == valid_id]

animate_combined_sequence(seq, pixels)


