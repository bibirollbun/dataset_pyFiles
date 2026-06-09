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


# =====================================
# PS-2 FINAL WORKING CODE (NO PATH ERROR)
# =====================================

import numpy as np
import pandas as pd
import os

# -----------------------------
# AUTO FIND road_profiles.csv
# -----------------------------
csv_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "road_profiles.csv" in files:
        csv_path = os.path.join(root, "road_profiles.csv")
        break

if csv_path is None:
    raise FileNotFoundError("road_profiles.csv NOT FOUND in /kaggle/input")

print("FOUND DATASET AT:", csv_path)

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv(csv_path)
time = data["t"].values

# -----------------------------
# PARAMETERS
# -----------------------------
m_s = 290.0
m_u = 59.0
k_s = 16000.0
k_t = 190000.0

c_min = 300.0
c_max = 3500.0

fs = 200.0
dt = 1 / fs
T = 20.0
N = int(T * fs)

DELAY = 4  # actuator delay steps

# -----------------------------
# UTILITY
# -----------------------------
def RMS(x):
    return np.sqrt(np.mean(x ** 2))

# -----------------------------
# CONTROLLER (LEGAL)
# -----------------------------
def controller(a_s, a_u):
    c = 900 * abs(a_s) + 250 * abs(a_u)
    return np.clip(c, c_min, c_max)

# -----------------------------
# SIMULATION
# -----------------------------
def simulate(road):
    z_s = np.zeros(N)
    z_u = np.zeros(N)
    v_s = np.zeros(N)
    v_u = np.zeros(N)

    a_s = np.zeros(N)
    a_u = np.zeros(N)

    buffer = [c_min] * DELAY

    for k in range(1, N):
        c_cmd = controller(a_s[k-1], a_u[k-1])
        buffer.append(c_cmd)
        c = buffer.pop(0)

        a_s[k] = (-k_s * (z_s[k-1] - z_u[k-1])
                  - c * (v_s[k-1] - v_u[k-1])) / m_s

        a_u[k] = ( k_s * (z_s[k-1] - z_u[k-1])
                  + c * (v_s[k-1] - v_u[k-1])
                  - k_t * (z_u[k-1] - road[k])) / m_u

        v_s[k] = v_s[k-1] + a_s[k] * dt
        v_u[k] = v_u[k-1] + a_u[k] * dt

        z_s[k] = z_s[k-1] + v_s[k] * dt
        z_u[k] = z_u[k-1] + v_u[k] * dt

    return z_s, a_s

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(z_s, a_s):
    z_rel = z_s - z_s[0]

    rms_zs = RMS(z_rel)
    max_zs = np.max(np.abs(z_rel))

    jerk = np.zeros_like(a_s)
    jerk[1:] = (a_s[1:] - a_s[:-1]) / dt

    rms_jerk = RMS(jerk)
    jerk_max = np.max(np.abs(jerk))

    comfort = (
        0.5 * rms_zs +
        max_zs +
        0.5 * rms_jerk +
        jerk_max
    )

    return rms_zs, max_zs, rms_jerk, comfort

# -----------------------------
# RUN ALL PROFILES
# -----------------------------
rows = []

for i in range(1, 6):
    road = data[f"profile_{i}"].values
    z_s, a_s = simulate(road)

    rms_zs, max_zs, rms_jerk, comfort = compute_metrics(z_s, a_s)

    rows.append([
        f"profile_{i}",
        rms_zs,
        max_zs,
        rms_jerk,
        comfort
    ])

# -----------------------------
# SAVE SUBMISSION
# -----------------------------
submission = pd.DataFrame(
    rows,
    columns=["profile", "rms_zs", "max_zs", "rms_jerk", "comfort_score"]
)

submission.to_csv("submission.csv", index=False)

print("submission.csv GENERATED SUCCESSFULLY")
submission

