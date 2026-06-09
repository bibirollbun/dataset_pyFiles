import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# loading data
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')


# Displaying data
train_df.shape


train_df.sequence_type.value_counts(normalize=True)


train_df.head(100)


train_df.behavior.value_counts()


train_df.columns[:50]


train_df[['acc_x', 'acc_y','acc_z', 'rot_w', 'rot_x', 'rot_y']].head()


train_df.orientation.unique()


train_df.gesture.unique()


train_df.loc[train_df.sequence_id == 'SEQ_000007',:].head(1)


train_df.loc[(train_df.gesture == 'Cheek - pinch skin') & (train_df.orientation=='Seated Lean Non Dom - FACE DOWN'), 'sequence_id'].unique()


# Choosing couple of gestures
cheek_pinch_gest_exmp_1 = (
    train_df
    .loc[
    (train_df.sequence_id == 'SEQ_000007') & 
    (train_df.phase=='Gesture'),
    ['rot_w', 'rot_x', 'rot_y', 'thm_1', 'thm_2', 'thm_3','thm_4', 'thm_5']]
)
cheek_pinch_gest_exmp_1['avg_temp'] = cheek_pinch_gest_exmp_1[['thm_1', 'thm_2', 'thm_3','thm_4', 'thm_5']].median(axis=1)

cheek_pinch_gest_exmp_2 = (
    train_df
    .loc[
    (train_df.sequence_id == 'SEQ_010026') & 
    (train_df.phase=='Gesture'),
    ['rot_w', 'rot_x', 'rot_y', 'thm_1', 'thm_2', 'thm_3','thm_4', 'thm_5']]
)
cheek_pinch_gest_exmp_2['avg_temp'] = cheek_pinch_gest_exmp_2[['thm_1', 'thm_2', 'thm_3','thm_4', 'thm_5']].median(axis=1)


cheek_pinch_gest_exmp = pd.concat([cheek_pinch_gest_exmp_1, cheek_pinch_gest_exmp_2])


cheek_pinch_gest_exmp.head()


# plotting
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

x = cheek_pinch_gest_exmp_1['rot_x'].values
y = cheek_pinch_gest_exmp_1['rot_y'].values
w = cheek_pinch_gest_exmp_1['rot_w'].values
t = cheek_pinch_gest_exmp_1['avg_temp'].values

points = np.array([x, y, w]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(t.min(), t.max())
colors = cm.plasma(norm(t))

lc = Line3DCollection(segments, cmap='plasma', norm=norm)
lc.set_array(t)
lc.set_linewidth(2)

# Crear figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(lc)

# Ajustar límites
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(w.min(), w.max())

# Etiquetas
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(lc, ax=ax, label='Temperature')

plt.title("3D trajectory colored by temperature")
plt.tight_layout()
plt.show()


# plotting
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

x = cheek_pinch_gest_exmp_2['rot_x'].values
y = cheek_pinch_gest_exmp_2['rot_y'].values
w = cheek_pinch_gest_exmp_2['rot_w'].values
t = cheek_pinch_gest_exmp_2['avg_temp'].values

points = np.array([x, y, w]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(t.min(), t.max())
colors = cm.plasma(norm(t))

lc = Line3DCollection(segments, cmap='plasma', norm=norm)
lc.set_array(t)
lc.set_linewidth(2)

# Crear figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(lc)

# Ajustar límites
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(w.min(), w.max())

# Etiquetas
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(lc, ax=ax, label='Temperature')

plt.title("3D trajectory colored by temperature")
plt.tight_layout()
plt.show()




