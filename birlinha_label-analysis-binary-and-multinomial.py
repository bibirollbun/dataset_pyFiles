import pandas as pd
bs_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"

df_labels_train = pd.read_csv(bs_path + "train_labels.csv")


df_labels_train.head()


df_labels_train['y'] = df_labels_train['Motor axis 0'].apply(lambda x: 0 if x == -1 else 1)


count_label_type = df_labels_train[['tomo_id','y']].rename(columns={'tomo_id':'count'}) \
        .groupby(['count','y']).count().reset_index().groupby('y').count().T


display(count_label_type.head()) # sum of 0 and 1 counts is n of tomographies

print("size of tomography samples: "+ str(len(df_labels_train.groupby('tomo_id').count())))


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')


# make data
wedge_sizes = count_label_type.loc['count'].to_list()
lbs = count_label_type.loc['count'].index.to_list()


colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(wedge_sizes)))

# plot
fig, ax = plt.subplots()
ax.pie(wedge_sizes, colors=colors, radius=3, center=(4, 4),
       labels= lbs,
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.title("sample per motor presence")
plt.show()


import numpy as np

col = "Number of motors"

count_label_type = (
    df_labels_train[['tomo_id', col]]
    .rename(columns={'tomo_id': 'count'})
    .groupby(['count', col]).count()
    .reset_index()
    .groupby(col).count()
    .T
)

# Apply log to the existing 'count' row
count_label_type.loc['log_count'] = np.log(count_label_type.loc['count'])


count_label_type


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')


# make data
wedge_sizes = count_label_type.loc['count'].to_list()
lbs = count_label_type.loc['count'].index.to_list()


colors = plt.get_cmap('Reds')(np.linspace(0.2, 0.7, len(wedge_sizes)))

# plot
fig, ax = plt.subplots()
ax.pie(wedge_sizes, colors=colors, radius=3, center=(4, 4),
       labels= lbs,
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.title("samples per number of motors")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')


# make data
log_wedge_sizes = count_label_type.loc['log_count'].to_list()
lbs = count_label_type.loc['count'].index.to_list()


colors = plt.get_cmap('Reds')(np.linspace(0.2, 0.7, len(log_wedge_sizes)))

# plot
fig, ax = plt.subplots()
ax.pie(log_wedge_sizes, colors=colors, radius=3, center=(4, 4),
       labels= lbs,
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.title("samples per number of motors log")
plt.show()


df_labels_train.head()


df_tomo = (
    df_labels_train[['tomo_id', 'Number of motors', 'Array shape (axis 0)',
                     'Array shape (axis 1)', 'Array shape (axis 2)','Voxel spacing']]
    .rename(columns={
        "Array shape (axis 0)": "z_max",
        "Array shape (axis 1)": "y_max",
        "Array shape (axis 2)": "x_max",
        "Voxel spacing":"voxel_spacing"
    })
    .groupby(['tomo_id', 'z_max', 'y_max', 'x_max', 'voxel_spacing'])
    .count()
    .reset_index()
    .set_index('tomo_id')
)


df_tomo.head()


df_tomo['cube_size'] = df_tomo.apply(lambda x: x['z_max'] * x['y_max'] * x['x_max'], axis=1)


df_tomo.head()


def sum_flagellar_motor_size(number_of_motors):
    mean_flagellar_motor_size = 45 # 45 nm == 45 voxel spaces
    # 45 nanometers in diameter
    # in 3 dimensions: 45 * 45 * 45
    dim_ = mean_flagellar_motor_size ** 3
    return dim_ * number_of_motors

df_tomo['cube_voxels'] = df_tomo.apply \
            (lambda x: (x['voxel_spacing']** 3) * x['cube_size'], axis=1)

df_tomo['flagellar_motors_sum_size'] = \
    df_tomo.apply(lambda x: sum_flagellar_motor_size(x['Number of motors']), axis=1)


df_tomo.head()


df = df_tomo[['cube_voxels','flagellar_motors_sum_size']] \
    .rename(columns={'flagellar_motors_sum_size':'motors³'}) \
    .sum()

x = df.to_list()
y = df.index.to_list()

pe_ = x[1] * 100 / x[0]

y2 = []

for l in y:
    if l == 'motors³':
        l = 'motors³_'+str(float(np.round(pe_,6)))
    y2.append(l)
print(x,y2)


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')


# make data
wedge_sizes = x
lbs = y2

colors = plt.get_cmap('Greys')(np.linspace(0.2, 0.7, len(wedge_sizes)))

# plot
fig, ax = plt.subplots()
ax.pie(wedge_sizes, colors=colors, radius=3, center=(4, 4),
       labels= lbs,
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()

