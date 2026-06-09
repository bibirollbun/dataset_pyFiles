!pip uninstall scipy -y -q
!pip install scipy==1.10.0 -q


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.integrate import cumtrapz


class KalmanFilter1D:
    def __init__(self, dt, process_var, meas_var):
        self.dt = dt
        self.A = np.array([[1, dt, 0.5 * dt**2],
                           [0, 1, dt],
                           [0, 0, 1]])
        self.H = np.array([[1, 0, 0]])
        self.Q = process_var * np.eye(3)
        self.R = np.array([[meas_var]])
        self.x = np.zeros((3, 1)) # (pos, vel, acc)
        self.P = np.eye(3)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.array([[z]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P

    def get_position(self):
        return self.x[0, 0]


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
unique_seqs = df['sequence_id'].unique()


np.random.seed(42)
sample_ids = np.random.choice(unique_seqs, size=9, replace=False)


fig = plt.figure(figsize=(18, 18))
for i, seq_id in enumerate(sample_ids):
    df_seq = df[df['sequence_id'] == seq_id].reset_index(drop=True)
    gesture = df_seq['gesture'].iloc[0]

    acc = df_seq[['acc_x', 'acc_y', 'acc_z']].values
    quat = df_seq[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values

    # bias = np.mean(acc[:50], axis=0)
    # acc -= bias

    rot = R.from_quat(quat[:, [1, 2, 3, 0]]) # (x, y, z, w)
    acc_world = rot.apply(acc)

    g = np.array([0, 0, 9.81])
    acc_world -= g

    dt = 1 / 50 # 50 hz

    vel = cumtrapz(acc_world, dx=dt, initial=0, axis=0)
    pos_raw = cumtrapz(vel, dx=dt, initial=0, axis=0)

    kf_x = KalmanFilter1D(dt, process_var=1e-4, meas_var=1e-2)
    kf_y = KalmanFilter1D(dt, process_var=1e-4, meas_var=1e-2)
    kf_z = KalmanFilter1D(dt, process_var=1e-4, meas_var=1e-2)

    pos_kf = []
    for t in range(pos_raw.shape[0]):
        kf_x.predict()
        kf_y.predict()
        kf_z.predict()

        kf_x.update(pos_raw[t, 0])
        kf_y.update(pos_raw[t, 1])
        kf_z.update(pos_raw[t, 2])

        pos_kf.append([kf_x.get_position(), kf_y.get_position(), kf_z.get_position()])

    pos_kf = np.array(pos_kf)

    ax = fig.add_subplot(3, 3, i + 1, projection='3d')
    ax.plot(pos_raw[:, 0], pos_raw[:, 1], pos_raw[:, 2], color='blue', alpha=0.5, label='raw integration')
    ax.plot(pos_kf[:, 0], pos_kf[:, 1], pos_kf[:, 2], color='red', label='kalman filtered')
    ax.scatter(pos_raw[0, 0], pos_raw[0, 1], pos_raw[0, 2], color='green', label='start')
    ax.scatter(pos_raw[-1, 0], pos_raw[-1, 1], pos_raw[-1, 2], color='blue')
    ax.set_title(f'{seq_id=}, {gesture=}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

plt.tight_layout()
plt.show()

