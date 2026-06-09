import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

DATASET_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"

df = pd.read_csv(DATASET_PATH)

sequence_ids = df["sequence_id"].unique()

chosen_sequence_id = random.choice(sequence_ids)
print(f"randomly chose sequence: {chosen_sequence_id}")

filtered_df = df[df['sequence_id'] == chosen_sequence_id]


RELEVANT_COLUMNS = ["sequence_id", "sequence_counter", "acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
filtered_df = filtered_df[RELEVANT_COLUMNS]
filtered_df.head()


plt.figure(figsize=(10, 5))

plt.plot(filtered_df['sequence_counter'], filtered_df['acc_x'], label='acc_x')
plt.plot(filtered_df['sequence_counter'], filtered_df['acc_y'], label='acc_y')
plt.plot(filtered_df['sequence_counter'], filtered_df['acc_z'], label='acc_z')

plt.xlabel('Sequence Counter')
plt.ylabel('Device-Relative Acceleration')
plt.title(f'Device-Relative Acceleration Over Time\n{chosen_sequence_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(filtered_df['sequence_counter'], filtered_df['rot_w'], label='rot_w')
plt.plot(filtered_df['sequence_counter'], filtered_df['rot_x'], label='rot_x')
plt.plot(filtered_df['sequence_counter'], filtered_df['rot_y'], label='rot_y')
plt.plot(filtered_df['sequence_counter'], filtered_df['rot_z'], label='rot_z')

plt.xlabel('Sequence Counter')
plt.ylabel('World-Relative Orientation')
plt.ylim([-1, 1])
plt.title(f'World-Relative Orientation Over Time\n{chosen_sequence_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def quaternion_conjugate(q):
    """Conjugate of a quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vector_by_quaternion(v, q):
    """Rotate 3D vector v by quaternion q."""
    v_quat = np.array([0, *v])
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return rotated[1:]  # return only x, y, z


# Apply to all rows
world_acc = filtered_df.apply(
    lambda row: rotate_vector_by_quaternion(
        [row['acc_x'], row['acc_y'], row['acc_z']],
        [row['rot_w'], row['rot_x'], row['rot_y'], row['rot_z']]
    ),
    axis=1, result_type='expand'
)

# Assign to new columns
filtered_df[['world_acc_x', 'world_acc_y', 'world_acc_z']] = world_acc
filtered_df.head()


plt.figure(figsize=(10, 5))

plt.plot(filtered_df['sequence_counter'], filtered_df['world_acc_x'], label='world_acc_x')
plt.plot(filtered_df['sequence_counter'], filtered_df['world_acc_y'], label='world_acc_y')
plt.plot(filtered_df['sequence_counter'], filtered_df['world_acc_z'], label='world_acc_z')

plt.xlabel('Sequence Counter')
plt.ylabel('World-Relative Acceleration')
plt.title(f'World-Relative Acceleration Over Time\n{chosen_sequence_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


ACCOUNT_FOR_GRAVITY = True
ALWAYS_SUBTRACT = False and ACCOUNT_FOR_GRAVITY
ALWAYS_ADD = False and ACCOUNT_FOR_GRAVITY

z_axis_avg_accel = filtered_df['world_acc_z'].mean()
print(f"avg. z_axis accel: {z_axis_avg_accel:.3f}")

if ACCOUNT_FOR_GRAVITY and (ALWAYS_SUBTRACT or (z_axis_avg_accel > 6)):
    filtered_df['world_acc_z'] -= 9.81 # subtracting the magnitude of gravity from z axis accel
    
if ACCOUNT_FOR_GRAVITY and (ALWAYS_ADD or (z_axis_avg_accel < -6)):
    filtered_df['world_acc_z'] += 9.81 # adding the magnitude of gravity to z axis accel


plt.figure(figsize=(10, 5))

plt.plot(filtered_df['sequence_counter'], filtered_df['world_acc_x'], label='world_acc_x')
plt.plot(filtered_df['sequence_counter'], filtered_df['world_acc_y'], label='world_acc_y')
plt.plot(filtered_df['sequence_counter'], filtered_df['world_acc_z'], label='world_acc_z')

plt.xlabel('Sequence Counter')
plt.ylabel('World-Relative Acceleration')
plt.title(f'World-Relative Acceleration Over Time (No Gravity) \n{chosen_sequence_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def rotate_vector_back_to_device(v_world, q):
    """Rotate a world-frame vector back into the device frame using the inverse quaternion."""
    q_conj = quaternion_conjugate(q)
    v_quat = np.array([0, *v_world])
    rotated = quaternion_multiply(quaternion_multiply(q_conj, v_quat), q)
    return rotated[1:]  # return x, y, z


# Apply to each row
device_acc = filtered_df.apply(
    lambda row: rotate_vector_back_to_device(
        [row['world_acc_x'], row['world_acc_y'], row['world_acc_z']],
        [row['rot_w'], row['rot_x'], row['rot_y'], row['rot_z']]
    ),
    axis=1, result_type='expand'
)

# Save to new columns
filtered_df[['device_acc_x', 'device_acc_y', 'device_acc_z']] = device_acc


plt.figure(figsize=(10, 5))

plt.plot(filtered_df['sequence_counter'], filtered_df['device_acc_x'], label='device_acc_x')
plt.plot(filtered_df['sequence_counter'], filtered_df['device_acc_y'], label='device_acc_y')
plt.plot(filtered_df['sequence_counter'], filtered_df['device_acc_z'], label='device_acc_z')

plt.xlabel('Sequence Counter')
plt.ylabel('Device-Relative Acceleration')
plt.title(f'Device-Relative Acceleration Over Time (No Gravity) \n{chosen_sequence_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

