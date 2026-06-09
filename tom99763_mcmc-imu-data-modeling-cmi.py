import os
import sys
import pandas as pd
import numpy as np
sys.path.append('/kaggle/input/cmi-modules-tom')
from feature_engineering import *
from scipy.spatial.transform import Rotation as R, Slerp
import pyro
from pyro.distributions import Normal, Uniform
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance, MCMC, NUTS
import argparse
from configs_IMUonly_exp1 import *
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import eigh


g = 9.81  # gravity
dt = 0.005
num_samples = 300


def quaternion_to_rotation_matrix(quat):
    """Convert (T, 4) quaternion to rotation matrix (T, 3, 3)."""
    rotations = R.from_quat(quat).as_matrix()
    return torch.tensor(rotations, dtype=torch.float32)

def rotate_acceleration(acc_local, quat):
    """Rotate local acceleration to global frame using quaternions."""
    R_t = quaternion_to_rotation_matrix(quat)  # (T, 3, 3)
    acc_global = torch.einsum("tij,tj->ti", R_t, acc_local)
    return acc_global

def kinematic_integration(acc_global, dt):
    """Integrate acceleration to get velocity and position."""
    v = torch.cumsum(acc_global * dt, dim=0)
    p = torch.cumsum(v * dt, dim=0)
    return v, p

# Pyro model
def imu_model(acc_obs, quat_obs, angvel_obs, dt=0.005):
    T = acc_obs.shape[0]

    # Priors
    bias = pyro.sample("acc_bias", dist.Normal(torch.zeros(3), 0.1 * torch.ones(3)))
    noise_scale = pyro.sample("noise_scale", dist.HalfNormal(0.1))

    # Rotate acceleration to global frame
    acc_corrected = acc_obs - bias
    acc_global = rotate_acceleration(acc_corrected, quat_obs)

    # Gravity correction
    acc_global = acc_global - torch.tensor([0, 0, g])
    v, p = kinematic_integration(acc_global, dt)

    # Observation likelihood (we observe positions here as proxy)
    for t in range(T):
        pyro.sample(f"obs_{t}", dist.Normal(p[t], noise_scale).to_event(1), obs=None)


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')


seq_id = df.sequence_id.unique()[5]


df[df.sequence_id == seq_id].shape


df_seq = df[df.sequence_id == seq_id].copy().groupby('sequence_id', group_keys=False).apply(lambda x: compute_combined_features(x, CFG))


acc_obs = torch.tensor(df_seq[['acc_x', 'acc_y', 'acc_z']].values, dtype=torch.float32)
quat_obs = torch.tensor(df_seq[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values)
angvel_obs = torch.tensor(df_seq[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']].values)


torch.manual_seed(42)
nuts_kernel = NUTS(imu_model)
mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=100)
mcmc.run(acc_obs, quat_obs, angvel_obs, dt)
posterior_samples = mcmc.get_samples()


num_samples = posterior_samples["obs_0"].shape[0]
T = len([k for k in posterior_samples if k.startswith("obs_")])
positions = torch.stack(
    [posterior_samples[f"obs_{t}"] for t in range(T)], dim=1)


positions = torch.stack([posterior_samples[f"obs_{t}"] for t in range(T)], dim=1)
positions_np = positions.numpy()
mean_pos = positions_np.mean(axis=0)  # shape (T, 3)


num_samples, T, _ = positions_np.shape

# Prepare long-form DataFrame for ridge plots
def make_df_for_dim(dim, name):
    return pd.DataFrame({
        "position": positions_np[:, :, dim].flatten(),
        "time": np.repeat(np.arange(T), num_samples),
        "axis": name
    })

df_x = make_df_for_dim(0, "x")
df_y = make_df_for_dim(1, "y")
df_z = make_df_for_dim(2, "z")
df_all = pd.concat([df_x, df_y, df_z], ignore_index=True)


fig = plt.figure(figsize=(14, 10))

# --- Main 3D trajectory plot ---
ax3d = fig.add_subplot(2, 2, 1, projection='3d')
mean_pos = positions.mean(0).numpy()

ax3d.plot(mean_pos[:, 0], mean_pos[:, 1], mean_pos[:, 2], color='blue', lw=2)
ax3d.set_title("Mean 3D Trajectory")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")

# --- Ridge plot helper ---
def plot_ridges(axis_name, subplot_idx):
    df_axis = df_all[df_all["axis"] == axis_name]
    ax = fig.add_subplot(2, 2, subplot_idx)
    sns.set(style="white")
    g = sns.FacetGrid(df_axis, row="time", aspect=15, height=0.25)
    g.map(sns.kdeplot, "position", fill=True, alpha=1)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel=f"{axis_name.upper()} Position")
    g.despine(bottom=True, left=True)
    plt.close(g.fig)  # Avoid duplicated output
    ax.imshow(g.fig.canvas.buffer_rgba())
    ax.axis("off")

# NOTE: FacetGrid + subplot combination is tricky in one figure.
# Better alternative:
# --- Violin plots for x, y, z per time ---
for i, axis_name in enumerate(["x", "y", "z"]):
    ax = fig.add_subplot(2, 2, i + 2)
    df_axis = df_all[df_all["axis"] == axis_name]
    sns.violinplot(x="time", y="position", data=df_axis, ax=ax, scale="width", inner="quartile")
    ax.set_title(f"Posterior {axis_name.upper()} Distribution")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"{axis_name.upper()} Position")
    ax.set_xticks([])

plt.tight_layout()
plt.show()


def plot_cov_ellipsoid(ax, mean, cov, n_std=2.0, resolution=20, alpha=0.2, color='red'):
    """Plot a 3D ellipsoid representing covariance."""
    # Eigenvalues (radii) and eigenvectors (rotation)
    eigvals, eigvecs = eigh(cov)
    radii = n_std * np.sqrt(eigvals)
    
    # Create a unit sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack((x, y, z), axis=-1)  # shape (res, res, 3)
    
    # Transform sphere to ellipsoid
    ellipsoid = sphere @ np.diag(radii) @ eigvecs.T + mean
    
    # Convert to Poly3D and add to plot
    ax.plot_surface(
        ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
        rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot mean trajectory
ax.plot(mean_pos[:, 0], mean_pos[:, 1], mean_pos[:, 2], lw=2, color='blue', label="Mean Trajectory")

# Add ellipsoids at selected time steps
skip = max(1, T // 30)
for t in range(0, T, skip):
    cloud = positions_np[:, t, :]  # shape (N, 3)
    mean_t = cloud.mean(axis=0)
    cov_t = np.cov(cloud.T)

    plot_cov_ellipsoid(ax, mean_t, cov_t, n_std=2.0, alpha=0.01, color='red')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory with Posterior Covariance Ellipsoids")
ax.legend()
plt.tight_layout()
plt.show()




