import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
display(train_df.head(2))
display(train_dem_df.head(2))
display(test_df.head(2))
display(test_dem_df.head(2))
excluded_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')

def filtered_describe(df, name):
    filtered_cols = [col for col in df.columns 
                     if not col.startswith(excluded_prefixes) and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f'\n➡️ Description of numerical columns in {name}')
    return df[filtered_cols].describe().T.style.background_gradient(cmap='viridis')

display(filtered_describe(train_df, "train_df"))
display(filtered_describe(test_df, "test_df"))
display(filtered_describe(train_dem_df, "train_dem_df"))
display(filtered_describe(test_dem_df, "test_dem_df"))
excluded_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')
sensor_cols = [col for col in train_df.columns if not col.startswith(excluded_prefixes)]

missing_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    '[TRAIN] Missing Count': train_df[sensor_cols].isnull().sum().values,
    '[TRAIN] Missing %': (train_df[sensor_cols].isnull().sum().values / len(train_df)) * 100
})

unique_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    'Unique Values [TRAIN]': train_df[sensor_cols].nunique().values
})

dtypes_sensor = pd.DataFrame({
    'Feature': sensor_cols,
    'Data Type': train_df[sensor_cols].dtypes.values
})

sensor_cols_test = [col for col in test_df.columns if col in sensor_cols]

missing_sensor_test = pd.DataFrame({
    'Feature': sensor_cols_test,
    '[TEST] Missing Count': test_df[sensor_cols_test].isnull().sum().values,
    '[TEST] Missing %': (test_df[sensor_cols_test].isnull().sum().values / len(test_df)) * 100
})

unique_sensor_test = pd.DataFrame({
    'Feature': sensor_cols_test,
    'Unique Values [TEST]': test_df[sensor_cols_test].nunique().values
})

sensor_summary = missing_sensor_train.merge(missing_sensor_test, on='Feature', how='left')
sensor_summary = sensor_summary.merge(unique_sensor_train, on='Feature', how='left')
sensor_summary = sensor_summary.merge(unique_sensor_test, on='Feature', how='left')
sensor_summary = sensor_summary.merge(dtypes_sensor, on='Feature', how='left')

styled_df = sensor_summary.fillna(0) 

styled_df.style.background_gradient(cmap='viridis')
dem_cols = train_dem_df.columns

missing_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    '[TRAIN DEMO] Missing Count': train_dem_df[dem_cols].isnull().sum().values,
    '[TRAIN DEMO] Missing %': (train_dem_df[dem_cols].isnull().sum().values / len(train_dem_df)) * 100
})

missing_demo_test = pd.DataFrame({
    'Feature': dem_cols,
    '[TEST DEMO] Missing Count': test_dem_df[dem_cols].isnull().sum().values,
    '[TEST DEMO] Missing %': (test_dem_df[dem_cols].isnull().sum().values / len(test_dem_df)) * 100
})

unique_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    'Unique Values [TRAIN DEMO]': train_dem_df[dem_cols].nunique().values
})

unique_demo_test = pd.DataFrame({
    'Feature': dem_cols,
    'Unique Values [TEST DEMO]': test_dem_df[dem_cols].nunique().values
})

dtypes_demo = pd.DataFrame({
    'Feature': dem_cols,
    'Data Type': train_dem_df[dem_cols].dtypes.values
})

demo_summary = (
    missing_demo_train
    .merge(missing_demo_test, on='Feature', how='left')
    .merge(unique_demo_train, on='Feature', how='left')
    .merge(unique_demo_test, on='Feature', how='left')
    .merge(dtypes_demo, on='Feature', how='left')
)

demo_summary.style.background_gradient(cmap='viridis')
acc_cols = [col for col in train_df.columns if col.startswith('acc_')]
rot_cols = [col for col in train_df.columns if col.startswith('rot_')]
thm_cols = [col for col in train_df.columns if col.startswith('thm_')]
tof_cols = [col for col in train_df.columns if col.startswith('tof_')]
import pandas as pd

acc_cols = [col for col in train_df.columns if col.startswith('acc_')]
rot_cols = [col for col in train_df.columns if col.startswith('rot_')]
thm_cols = [col for col in train_df.columns if col.startswith('thm_')]
tof_cols = [col for col in train_df.columns if col.startswith('tof_')]

def sensor_summary(df, cols, name, dataset_name):
    summary = pd.DataFrame({
        'Feature': cols,
        f'{dataset_name} Missing %': df[cols].isnull().mean().values * 100,
        f'{dataset_name} Min': df[cols].min().values,
        f'{dataset_name} Max': df[cols].max().values,
        f'{dataset_name} Mean': df[cols].mean().values,
        f'{dataset_name} Std': df[cols].std().values
    })
    summary.insert(0, 'Sensor', name)
    return summary

def combined_sensor_summary(train_df, test_df, sensor_cols_dict):
    all_train = []
    all_test = []

    for sensor_name, cols in sensor_cols_dict.items():
        all_train.append(sensor_summary(train_df, cols, sensor_name, 'Train'))
        all_test.append(sensor_summary(test_df, cols, sensor_name, 'Test'))

    train_summary = pd.concat(all_train, ignore_index=True)
    test_summary = pd.concat(all_test, ignore_index=True)

    merged = pd.merge(train_summary, test_summary, on=['Sensor', 'Feature'], how='outer')
    return merged

sensor_cols_dict = {
    'acc': acc_cols,
    'rot': rot_cols,
    'thm': thm_cols,
    'tof': tof_cols
}

sensor_comparison = combined_sensor_summary(train_df, test_df, sensor_cols_dict)

summary_by_group = sensor_comparison.groupby('Sensor').mean(numeric_only=True)

summary_by_group.style.background_gradient(cmap='viridis')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

sensor_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')

categorical_columns = ['phase', 'behavior', 'orientation', 'sequence_type',
                       'adult_child', 'sex', 'handedness']

target_columns = ['gesture']

categorical_tracker = categorical_columns.copy()


def get_numerical_variables(df, excluded_prefixes, excluded_columns):
    return [col for col in df.columns 
            if pd.api.types.is_numeric_dtype(df[col])
            and not col.startswith(excluded_prefixes)
            and col not in excluded_columns]

train_main = train_df.copy()
test_main = test_df.copy()
train_demo = train_dem_df.copy()
test_demo = test_dem_df.copy()

train_main['Dataset'] = 'Train'
test_main['Dataset'] = 'Test'
train_demo['Dataset'] = 'Train'
test_demo['Dataset'] = 'Test'

main_data = pd.concat([train_main, test_main], axis=0)
demo_data = pd.concat([train_demo, test_demo], axis=0)

main_numeric_vars = get_numerical_variables(main_data, sensor_prefixes, categorical_columns)
demo_numeric_vars = get_numerical_variables(demo_data, (), categorical_columns)

def create_variable_plots(variable, dataset_label, data):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f" Box Plot for {variable} — {dataset_label}")

    plt.subplot(1, 2, 2)
    for label, color in zip(data['Dataset'].unique(), custom_palette):
        subset = data[data['Dataset'] == label]
        sns.histplot(data=subset, x=variable, kde=True, bins=30, label=label, color=color)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f" Histogram for {variable} — {dataset_label}")
    plt.legend()

    plt.tight_layout()
    plt.show()

for var in main_numeric_vars:
    create_variable_plots(var, "Context Features", main_data)

for var in demo_numeric_vars:
    create_variable_plots(var, "Demographic Features", demo_data)

train_main.drop('Dataset', axis=1, inplace=True)
test_main.drop('Dataset', axis=1, inplace=True)
train_demo.drop('Dataset', axis=1, inplace=True)
test_demo.drop('Dataset', axis=1, inplace=True)
import numpy as np
import pandas as pd

train_temp = train_df.copy()
test_temp  = test_df.copy()

train_temp['acc_mag'] = np.sqrt(
    train_temp['acc_x']**2 + train_temp['acc_y']**2 + train_temp['acc_z']**2
)
test_temp['acc_mag'] = np.sqrt(
    test_temp['acc_x']**2 + test_temp['acc_y']**2 + test_temp['acc_z']**2
)

train_temp['rot_angle'] = 2 * np.arccos(train_temp['rot_w'].clip(-1,1))
test_temp['rot_angle']  = 2 * np.arccos(test_temp['rot_w'].clip(-1,1))

acc_agg_funcs = {
    'acc_mag': ['mean', 'std', 'max']
}
train_acc_summary = train_temp.groupby('sequence_id').agg(acc_agg_funcs)
test_acc_summary  = test_temp.groupby('sequence_id').agg(acc_agg_funcs)

train_acc_summary.columns = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]
test_acc_summary.columns  = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]

rot_agg_funcs = {
    'rot_angle': ['mean', 'std', 'max']
}
train_rot_summary = train_temp.groupby('sequence_id').agg(rot_agg_funcs)
test_rot_summary  = test_temp.groupby('sequence_id').agg(rot_agg_funcs)

train_rot_summary.columns = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]
test_rot_summary.columns  = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]

thm_cols = [f"thm_{i}" for i in range(1, 6)]

thm_agg_funcs = {col: ['mean', 'std'] for col in thm_cols}

train_thm_summary = train_temp.groupby('sequence_id').agg(thm_agg_funcs)
test_thm_summary  = test_temp.groupby('sequence_id').agg(thm_agg_funcs)

flattened_thm_cols = []
for sensor in thm_cols:
    for stat in ['mean','std']:
        flattened_thm_cols.append(f"{sensor}_{stat}")

train_thm_summary.columns = flattened_thm_cols
test_thm_summary.columns  = flattened_thm_cols



def compute_tof_sequence_summary(df):

    seq_summaries = {}

    for i in range(1, 6):
        tof_cols = [f"tof_{i}_v{pix}" for pix in range(64)]
        ts_grid = df[tof_cols].replace(-1, np.nan).astype(float)
        df[f"tof_{i}_mean_at_ts"] = ts_grid.mean(axis=1)
    
    agg_dict = {f"tof_{i}_mean_at_ts": ['mean','std'] for i in range(1, 6)}
    summary = df.groupby('sequence_id').agg(agg_dict)
    flat_cols = [f"tof_{i}_{stat}" for i in range(1, 6) for stat in ['mean','std']]
    summary.columns = flat_cols
    return summary

train_tof_summary = compute_tof_sequence_summary(train_temp)
test_tof_summary  = compute_tof_sequence_summary(test_temp)

train_sensor_summary = (
    train_acc_summary
    .join(train_rot_summary, how='outer')
    .join(train_thm_summary, how='outer')
    .join(train_tof_summary, how='outer')
)

test_sensor_summary = (
    test_acc_summary
    .join(test_rot_summary, how='outer')
    .join(test_thm_summary, how='outer')
    .join(test_tof_summary, how='outer')
)

train_sensor_summary['Dataset'] = 'Train'
test_sensor_summary['Dataset']  = 'Test'

combined_sensor_summary = pd.concat(
    [train_sensor_summary, test_sensor_summary],
    axis=0
).reset_index(drop=True)
import seaborn as sns
import matplotlib.pyplot as plt

custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

def create_sensor_summary_plots(variable, dataset_label, data):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x=variable, y="Dataset", palette=custom_palette[:2])
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable} — {dataset_label}")

    plt.subplot(1, 2, 2)
    for label, color in zip(['Train', 'Test'], custom_palette[:2]):
        subset = data[data['Dataset'] == label]
        sns.histplot(data=subset, x=variable, kde=True, bins=30, label=label, color=color)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} — {dataset_label}")
    plt.legend()

    plt.tight_layout()
    plt.show()

sensor_summary_vars = [col for col in combined_sensor_summary.columns if col != 'Dataset']

for var in sensor_summary_vars:
    create_sensor_summary_plots(var, "Per‐Sequence Sensor Summaries", combined_sensor_summary)
    


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779',
                     '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
custom_palette = ['#3498db', '#e74c3c']  # Train, Test

categorical_variables = [col for col in categorical_tracker if col != 'gesture']

train_main = train_df.copy()
test_main = test_df.copy()
train_main['dataset'] = 'train'
test_main['dataset'] = 'test'
main_combined = pd.concat([train_main, test_main], axis=0)

train_demo = train_dem_df.copy()
test_demo = test_dem_df.copy()
train_demo['dataset'] = 'train'
test_demo['dataset'] = 'test'
demo_combined = pd.concat([train_demo, test_demo], axis=0)

def create_categorical_plots(variable, data, source_name):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plt.subplot(1, 2, 1)
    value_counts = data[variable].value_counts()

    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts.copy()
    filtered_values[value_counts < threshold] = 0
    filtered_values = filtered_values[filtered_values > 0]
    other_count = value_counts.sum() - filtered_values.sum()
    if other_count > 0:
        filtered_values['Other'] = other_count

    wedges, texts, autotexts = plt.pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=pie_chart_palette[:len(filtered_values)],
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if p > 5 else 0 for p in filtered_values],
        textprops={'fontsize': 10}
    )

    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {variable} — {source_name}", width=50)))
    plt.legend(filtered_values.index, loc="upper left", bbox_to_anchor=(1, 1))

    plt.subplot(1, 2, 2)
    sns.countplot(
        data=data,
        y=variable,
        hue='dataset',
        palette=custom_palette,
        alpha=0.85
    )
    plt.ylabel(variable)
    plt.xlabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Countplot for {variable} — {source_name}", width=50)))
    plt.tight_layout()
    plt.show()

for var in categorical_variables:
    if var in main_combined.columns:
        create_categorical_plots(var, main_combined, "Context Features")

for var in categorical_variables:
    if var in demo_combined.columns:
        create_categorical_plots(var, demo_combined, "Demographic Features")

train_main.drop('dataset', axis=1, inplace=True)
test_main.drop('dataset', axis=1, inplace=True)
train_demo.drop('dataset', axis=1, inplace=True)
test_demo.drop('dataset', axis=1, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779',
                     '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
custom_palette = ['#3498db', '#e74c3c']  

target_columns = ['gesture']

train_main = train_df.copy()
test_main = test_df.copy()
train_main['dataset'] = 'train'
test_main['dataset'] = 'test'
main_combined = pd.concat([train_main, test_main], axis=0)

def create_target_plots(variable, data, source_name):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plt.subplot(1, 2, 1)
    value_counts = data[variable].value_counts()

    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts.copy()
    filtered_values[value_counts < threshold] = 0
    filtered_values = filtered_values[filtered_values > 0]
    other_count = value_counts.sum() - filtered_values.sum()
    if other_count > 0:
        filtered_values['Other'] = other_count

    wedges, texts, autotexts = plt.pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=pie_chart_palette[:len(filtered_values)],
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if p > 5 else 0 for p in filtered_values],
        textprops={'fontsize': 10}
    )

    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {variable} — {source_name}", width=50)))
    plt.legend(filtered_values.index, loc="upper left", bbox_to_anchor=(1, 1))

    plt.subplot(1, 2, 2)
    sns.countplot(
        data=data,
        y=variable,
        hue='dataset',
        palette=custom_palette,
        alpha=0.85
    )
    plt.ylabel(variable)
    plt.xlabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Countplot for {variable} — {source_name}", width=50)))
    plt.tight_layout()
    plt.show()

for var in target_columns:
    if var in main_combined.columns:
        create_target_plots(var, main_combined, "Target Variable")

train_main.drop('dataset', axis=1, inplace=True)
test_main.drop('dataset', axis=1, inplace=True)


import numpy as np
import matplotlib.pyplot as plt

subject_id = train_df['subject'].unique()[0]
subj_df = train_df[train_df['subject'] == subject_id].copy()

subj_df['acc_mag'] = np.sqrt(
    subj_df['acc_x']**2 + subj_df['acc_y']**2 + subj_df['acc_z']**2
)
subj_df['rot_w_clipped'] = subj_df['rot_w'].clip(-1, 1)
subj_df['rot_angle'] = 2 * np.arccos(subj_df['rot_w_clipped'])


gesture_to_seq = subj_df.groupby('gesture')['sequence_id'].first().to_dict()
seq_ids = list(gesture_to_seq.values())

n = len(seq_ids)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False, sharey=False)
axes = axes.flatten()

for i, seq in enumerate(seq_ids):
    ax = axes[i]
    seq_df = subj_df[subj_df['sequence_id'] == seq].sort_values('sequence_counter')
    times = seq_df['sequence_counter']

    ax.plot(times, seq_df['acc_mag'],
            label='Acceleration Magnitude', color='tab:blue', linewidth=1.5)
    ax.plot(times, seq_df['rot_angle'],
            label='Rotation Angle (rad)', color='tab:orange', linewidth=1.5)

    used_labels = set()
    for phase_label, color in [('Transition', 'lightgray'),
                               ('Gesture', 'lightcoral')]:
        mask = seq_df['phase'] == phase_label
        if mask.any():
            idxs = seq_df.index[mask]
            breaks = np.where(np.diff(idxs) != 1)[0]
            spans = []
            start_idx = idxs[0]
            for b in breaks:
                end_idx = idxs[b]
                spans.append((start_idx, end_idx))
                start_idx = idxs[b + 1]
            spans.append((start_idx, idxs[-1]))
            for (start_i, end_i) in spans:
                t0 = seq_df.loc[start_i, 'sequence_counter']
                t1 = seq_df.loc[end_i, 'sequence_counter']
                label_arg = phase_label if phase_label not in used_labels else None
                ax.axvspan(t0, t1, color=color, alpha=0.3, label=label_arg)
                used_labels.add(phase_label)

    gesture_name = seq_df['gesture'].iloc[0]
    seq_type = seq_df['sequence_type'].iloc[0]
    ax.set_title(f"Subject {subject_id} – Seq {seq} – {gesture_name} ({seq_type})",
                 fontsize=10)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)


for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

therm_df = subj_df.copy()

thm_cols = [f'thm_{i}' for i in range(1, 6)]
therm_df['thm_mean'] = therm_df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False, sharey=False)
axes = axes.flatten()

for i, seq in enumerate(seq_ids):
    ax = axes[i]
    seq_thm = therm_df[therm_df['sequence_id'] == seq].sort_values('sequence_counter')
    times = seq_thm['sequence_counter']
      for col in thm_cols:
        ax.plot(times, seq_thm[col], label=col, linewidth=1)
    ax.plot(times, seq_thm['thm_mean'], label='Average Temperature', color='black', linewidth=2.5)
    for phase_label, color in [('Transition', 'lightgray'), ('Gesture', 'lightcoral')]:
        mask = seq_thm['phase'] == phase_label
        if mask.any():
            idxs = seq_thm.index[mask]
            breaks = np.where(np.diff(idxs) != 1)[0]
            spans = []
            start_idx = idxs[0]
            for b in breaks:
                end_idx = idxs[b]
                spans.append((start_idx, end_idx))
                start_idx = idxs[b + 1]
            spans.append((start_idx, idxs[-1]))
            for (start_i, end_i) in spans:
                t0 = seq_thm.loc[start_i, 'sequence_counter']
                t1 = seq_thm.loc[end_i, 'sequence_counter']
                ax.axvspan(t0, t1, color=color, alpha=0.3, label=phase_label)
    gesture_name = seq_thm['gesture'].iloc[0]
    seq_type = seq_thm['sequence_type'].iloc[0]
    ax.set_title(f"Subject {subject_id} – Seq {seq} – {gesture_name} ({seq_type})", fontsize=10)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Temperature (°C)", fontsize=9)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

tof_df = subj_df.copy()

mean_cols = []
for i_sensor in range(1, 6):
    pixel_cols = [f'tof_{i_sensor}_v{pix}' for pix in range(64)]
    tof_df[f'tof_{i_sensor}_mean'] = tof_df[pixel_cols].replace(-1, np.nan).mean(axis=1)
    mean_cols.append(f'tof_{i_sensor}_mean')

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False, sharey=False)
axes = axes.flatten()

for i, seq in enumerate(seq_ids):
    ax = axes[i]
    seq_tof = tof_df[tof_df['sequence_id'] == seq].sort_values('sequence_counter')
    times = seq_tof['sequence_counter']

    for col in mean_cols:
        ax.plot(times, seq_tof[col], label=col, linewidth=1)

    for phase_label, color in [('Transition', 'lightgray'), ('Gesture', 'lightcoral')]:
        mask = seq_tof['phase'] == phase_label
        if mask.any():
            idxs = seq_tof.index[mask]
            breaks = np.where(np.diff(idxs) != 1)[0]
            spans = []
            start_idx = idxs[0]
            for b in breaks:
                end_idx = idxs[b]
                spans.append((start_idx, end_idx))
                start_idx = idxs[b + 1]
            spans.append((start_idx, idxs[-1]))
            for (start_i, end_i) in spans:
                t0 = seq_tof.loc[start_i, 'sequence_counter']
                t1 = seq_tof.loc[end_i, 'sequence_counter']
                ax.axvspan(t0, t1, color=color, alpha=0.3, label=phase_label)

    gesture_name = seq_tof['gesture'].iloc[0]
    seq_type     = seq_tof['sequence_type'].iloc[0]
    ax.set_title(f"Subject {subject_id} – Seq {seq} – {gesture_name} ({seq_type})", fontsize=10)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Mean ToF Distance", fontsize=9)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

train_df = train_df.merge(
    train_dem_df,
    on="subject",
    how="left"

gesture_to_plot = "Write name on leg"

df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()

left_group  = df_gesture[df_gesture["handedness"] == 0]
right_group = df_gesture[df_gesture["handedness"] == 1]

if left_group["sequence_id"].nunique() == 0:
    raise ValueError("No left‐handed example of that gesture found. Choose a different gesture.")

left_seq  = left_group["sequence_id"].unique()[0]
right_seq = right_group["sequence_id"].unique()[0]

df_left  = train_df[(train_df["sequence_id"] == left_seq)].sort_values("sequence_counter")
df_right = train_df[(train_df["sequence_id"] == right_seq)].sort_values("sequence_counter")

for df in (df_left, df_right):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Left‐handed (Subj: {df_left['subject'].iloc[0]})",
     f"Right‐handed (Subj: {df_right['subject'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/s²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acceleration Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Handedness: Acceleration Magnitude – Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Left‐handed (Subj: {df_left['subject'].iloc[0]})",
     f"Right‐handed (Subj: {df_right['subject'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rotation Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Handedness: Rotation Angle – Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


for df in (df_left, df_right):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Left‐handed (Subj: {df_left['subject'].iloc[0]})",
     f"Right‐handed (Subj: {df_right['subject'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="x-small", ncol=2)
    ax.grid(True)
plt.suptitle("Handedness: Thermopile – Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


for df in (df_left, df_right):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Left‐handed (Subj: {df_left['subject'].iloc[0]})",
     f"Right‐handed (Subj: {df_right['subject'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Handedness: ToF Mean Distance – Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



import numpy as np
import matplotlib.pyplot as plt

gesture_to_plot = "Neck - pinch skin"

df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()
child_group = df_gesture[df_gesture["adult_child"] == 0]
adult_group = df_gesture[df_gesture["adult_child"] == 1]

if (child_group["sequence_id"].nunique() == 0) or (adult_group["sequence_id"].nunique() == 0):
    raise ValueError("Insufficient examples in one of the groups. Pick another gesture.")

child_seq = child_group["sequence_id"].unique()[0]
adult_seq = adult_group["sequence_id"].unique()[0]

df_child = train_df[train_df["sequence_id"] == child_seq].sort_values("sequence_counter")
df_adult = train_df[train_df["sequence_id"] == adult_seq].sort_values("sequence_counter")

for df in (df_child, df_adult):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/s²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acc Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Adult vs. Child: Acceleration Magnitude", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rot Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Adult vs. Child: Rotation Angle", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


for df in (df_child, df_adult):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
plt.suptitle("Adult vs. Child: Thermopile", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

for df in (df_child, df_adult):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Adult vs. Child: ToF Mean Distance", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



gesture_to_plot = "Forehead - scratch"

df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()
female_group = df_gesture[df_gesture["sex"] == 0]
male_group   = df_gesture[df_gesture["sex"] == 1]

if (female_group["sequence_id"].nunique() == 0) or (male_group["sequence_id"].nunique() == 0):
    raise ValueError("Not enough examples for each sex. Try a different gesture.")

female_seq = female_group["sequence_id"].unique()[0]
male_seq   = male_group["sequence_id"].unique()[0]

df_fem = train_df[train_df["sequence_id"] == female_seq].sort_values("sequence_counter")
df_male= train_df[train_df["sequence_id"] == male_seq].sort_values("sequence_counter")

for df in (df_fem, df_male):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/s²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acceleration Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Sex: Acceleration Magnitude – Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rotation Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Sex: Rotation Angle – Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


for df in (df_fem, df_male):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    times = df_seq["sequence_counter"]
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
plt.suptitle("Sex: Thermopile – Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


for df in (df_fem, df_male):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Sex: ToF Mean Distance – Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



gesture_to_plot = "Eyelash - pull hair"

df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()
median_shoulder = df_gesture["shoulder_to_wrist_cm"].median()

short_group = df_gesture[df_gesture["shoulder_to_wrist_cm"] <= median_shoulder]
long_group  = df_gesture[df_gesture["shoulder_to_wrist_cm"]  > median_shoulder]

if (short_group["sequence_id"].nunique() == 0) or (long_group["sequence_id"].nunique() == 0):
    raise ValueError("Not enough examples in one of the length‐groups. Try another gesture or adjust threshold.")

short_seq = short_group["sequence_id"].unique()[0]
long_seq  = long_group["sequence_id"].unique()[0]

df_short = train_df[train_df["sequence_id"] == short_seq].sort_values("sequence_counter")
df_long  = train_df[train_df["sequence_id"] == long_seq].sort_values("sequence_counter")

for df in (df_short, df_long):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulder‐Wrist ≤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulder‐Wrist > {median_shoulder:.1f} cm)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/s²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acceleration Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Arm Length: Acceleration Magnitude – Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulder‐Wrist ≤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulder‐Wrist > {median_shoulder:.1f} cm)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rotation Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Arm Length: Rotation Angle – Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


for df in (df_short, df_long):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulder‐Wrist ≤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulder‐Wrist > {median_shoulder:.1f} cm)"]
):
    times = df_seq["sequence_counter"]
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
plt.suptitle("Arm Length: Thermopile – Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


for df in (df_short, df_long):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulder‐Wrist ≤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulder‐Wrist > {median_shoulder:.1f} cm)"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Arm Length: ToF Mean Distance – Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

combined_main = pd.concat([train_df, test_df], axis=0)
combined_demo = pd.concat([train_dem_df, test_dem_df], axis=0)

target_variable = 'gesture'

sensor_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')
excluded_columns = [target_variable]

def get_numerical_columns(df, exclude_prefixes, excluded_cols):
    return [col for col in df.columns 
            if pd.api.types.is_numeric_dtype(df[col]) 
            and not col.startswith(exclude_prefixes)
            and col not in excluded_cols]

main_vars = get_numerical_columns(combined_main, sensor_prefixes, excluded_columns)
demo_vars = get_numerical_columns(combined_demo, (), excluded_columns)

combined_data = pd.concat([
    combined_main[main_vars].reset_index(drop=True),
    combined_demo[demo_vars].reset_index(drop=True)
], axis=1)

corr_all = combined_data.corr()
mask_all = np.triu(np.ones_like(corr_all, dtype=bool))

plt.figure(figsize=(18, 10))
ax = sns.heatmap(
    corr_all, mask=mask_all, cmap='viridis', annot=True, 
    square=False, linewidths=.5, annot_kws={"size": 12}
)
plt.title('Correlation Heatmap — Combined (Demographic + Main) Data', fontsize=16)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder

train_corr_df = train_df.copy()
train_corr_df[target_variable] = LabelEncoder().fit_transform(train_corr_df[target_variable])

train_main_vars = get_numerical_columns(train_corr_df, sensor_prefixes, excluded_columns)

train_demo_vars = get_numerical_columns(train_dem_df, (), excluded_columns)

train_all_corr = pd.concat([train_corr_df[train_main_vars], train_dem_df[train_demo_vars],
                            train_corr_df[[target_variable]]], axis=1)

corr_target_only = train_all_corr.corr()[[target_variable]].T

plt.figure(figsize=(12, 3))
sns.heatmap(corr_target_only, cmap='viridis', annot=True, linewidths=0.5, cbar=False, annot_kws={"size": 10})
plt.title("Correlation with Target (gesture) — Train Data", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import kaggle_evaluation.cmi_inference_server



def mixup_data(x, y, alpha=0.2):
    """
    Return mixed inputs and mixed targets (one-hot) for mixup.
    x: Tensor of shape (batch_size, features, seq_len)
    y: Tensor of shape (batch_size, num_classes)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

class SequenceDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X: np.ndarray of shape (n_samples, features, seq_len)
        y: np.ndarray of shape (n_samples, num_classes) or None for test
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]



print("Loading datasets...")
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

label_encoder = LabelEncoder()
train_df['gesture'] = label_encoder.fit_transform(train_df['gesture'].astype(str))
gesture_classes = label_encoder.classes_
np.save('gesture_classes.npy', gesture_classes)

excluded_cols = {
    'gesture', 'sequence_type', 'behavior', 'orientation',
    'row_id', 'subject', 'phase',
    'sequence_id', 'sequence_counter'
}
all_feature_cols = [c for c in train_df.columns if c not in excluded_cols]

imu_cols = [c for c in all_feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
tof_thm_cols = [c for c in all_feature_cols if c.startswith('thm_') or c.startswith('tof_')]

feature_cols = imu_cols + tof_thm_cols
imu_dim = len(imu_cols)
tof_thm_dim = len(tof_thm_cols)
print(f"IMU features: {imu_dim}, TOF/Thermal features: {tof_thm_dim}, Total features: {len(feature_cols)}")

nan_counts = train_df[feature_cols].isna().sum().sum()
print("Total NaNs in train features:", nan_counts)

print("Fitting StandardScaler on train data...")
all_values = train_df[feature_cols].ffill().bfill().fillna(0).values
scaler = StandardScaler().fit(all_values)
joblib.dump(scaler, 'global_scaler.pkl')

print("Building sequences...")
sequences = train_df.groupby('sequence_id')
X_list = []
lengths = []
y_list = []

for i, (seq_id, seq) in enumerate(sequences):
    seq_data = seq[feature_cols].ffill().bfill().fillna(0).values
    scaled = scaler.transform(seq_data)
    X_list.append(scaled)
    lengths.append(scaled.shape[0])
    y_list.append(seq['gesture'].iloc[0])
    if i % 500 == 0 and i > 0:
        print(f"  Processed {i} sequences...")

pad_len = int(np.percentile(lengths, 90))
print(f"Pad/truncate all sequences to length {pad_len} (90th percentile).")
np.save("sequence_maxlen.npy", pad_len)

print("Padding/truncating sequences...")
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
X = keras_pad_sequences(
    X_list,
    maxlen=pad_len,
    dtype='float32',
    padding='post',
    truncating='post'
)  

y = np.array(y_list)  

num_classes = len(np.unique(y))
y_cat = np.eye(num_classes)[y] 

X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)
print("Train/Val shapes:", X_train_np.shape, X_val_np.shape, y_train_np.shape, y_val_np.shape)

X_train_np = np.transpose(X_train_np, (0, 2, 1))
X_val_np = np.transpose(X_val_np, (0, 2, 1))

labels_train = np.argmax(y_train_np, axis=1)
class_weights_values = compute_class_weight('balanced',
                                            classes=np.unique(labels_train),
                                            y=labels_train)
class_weights = torch.tensor(class_weights_values, dtype=torch.float)



batch_size = 128

train_dataset = SequenceDataset(X_train_np, y_train_np)
val_dataset = SequenceDataset(X_val_np, y_val_np)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       
        se = x.mean(dim=2)                     
        se = self.relu(self.fc1(se))            
        se = self.sigmoid(self.fc2(se))         
        se = se.unsqueeze(2)                    
        return x * se                           

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout_rate=0.3):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels, reduction=8)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        shortcut = self.shortcut(x)                               
        out = self.conv1(x)                                          
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)                                        
        out = self.bn2(out)

        out = self.se(out)                                           

        out = out + shortcut                                         
        out = self.relu(out)

        out = self.pool(out)                                        
        out = self.dropout(out)
        return out

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.score_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        scores = torch.tanh(self.score_fc(x))            
        scores = scores.squeeze(2)                       
        weights = F.softmax(scores, dim=1)              
        weights = weights.unsqueeze(2)                   
        weighted = x * weights                          
        context = weighted.sum(dim=1)                    
        return context

class TwoBranchHARModel(nn.Module):
    def __init__(self, total_features, imu_dim, tof_thm_dim, pad_len, num_classes, wd=1e-4):
        super(TwoBranchHARModel, self).__init__()
        self.resblock1 = ResidualSEBlock(imu_dim, 64, kernel_size=3, pool_size=2, dropout_rate=0.3)
        self.resblock2 = ResidualSEBlock(64, 128, kernel_size=5, pool_size=2, dropout_rate=0.3)

        self.conv1_ttf = nn.Conv1d(tof_thm_dim, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_ttf = nn.BatchNorm1d(64)
        self.pool1_ttf = nn.MaxPool1d(kernel_size=2)
        self.drop1_ttf = nn.Dropout(0.3)

        self.conv2_ttf = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2_ttf = nn.BatchNorm1d(128)
        self.pool2_ttf = nn.MaxPool1d(kernel_size=2)
        self.drop2_ttf = nn.Dropout(0.3)

        reduced_len = pad_len // 4
        merged_channels = 128 + 128  

        self.lstm = nn.LSTM(
            input_size=merged_channels,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.drop_lstm = nn.Dropout(0.4)

        self.attention = Attention(input_dim=256)

        self.fc1 = nn.Linear(256, 256, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop_fc2 = nn.Dropout(0.3)

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x_imu = x[:, :imu_dim, :]          
        x_ttf = x[:, imu_dim:, :]           

        b1 = self.resblock1(x_imu)          
        b1 = self.resblock2(b1)            

        b2 = self.conv1_ttf(x_ttf)          
        b2 = self.bn1_ttf(b2)
        b2 = F.relu(b2)
        b2 = self.pool1_ttf(b2)             
        b2 = self.drop1_ttf(b2)

        b2 = self.conv2_ttf(b2)             
        b2 = self.bn2_ttf(b2)
        b2 = F.relu(b2)
        b2 = self.pool2_ttf(b2)             
        b2 = self.drop2_ttf(b2)

        merged = torch.cat([b1, b2], dim=1)  

        merged = merged.permute(0, 2, 1)

        lstm_out, _ = self.lstm(merged)       
        lstm_out = self.drop_lstm(lstm_out)   

        context = self.attention(lstm_out)    

        x = self.fc1(context)                
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)

        x = self.fc2(x)                       
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)

        out = self.out(x)                     
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_shape = (len(feature_cols), pad_len)
model = TwoBranchHARModel(
    total_features=len(feature_cols),
    imu_dim=imu_dim,
    tof_thm_dim=tof_thm_dim,
    pad_len=pad_len,
    num_classes=num_classes,
    wd=1e-4
).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {count_parameters(model)}")

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5 * steps_per_epoch,
    T_mult=2,
    eta_min=1e-5
)

def soft_cross_entropy(pred, soft_targets):
    """
    pred: (batch, num_classes) raw scores (no softmax)
    soft_targets: (batch, num_classes) probabilities
    """
    log_probs = F.log_softmax(pred, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
    return loss

patience = 10
best_val_loss = np.inf
epochs_no_improve = 0
num_epochs = 100


print("Starting training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)           
        batch_y = batch_y.to(device)           

        # Apply mixup
        mixed_x, mixed_y = mixup_data(batch_x, batch_y, alpha=0.2)

        optimizer.zero_grad()
        outputs = model(mixed_x)               
        loss = soft_cross_entropy(outputs, mixed_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = soft_cross_entropy(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best model.")
            model.load_state_dict(best_model_state)
            break

torch.save(best_model_state, "gesture_two_branch_mixup_pytorch.pth")
print("Model training complete and saved as gesture_two_branch_mixup_pytorch.pth")

gesture_classes = np.load("gesture_classes.npy", allow_pickle=True)
pad_len = int(np.load("sequence_maxlen.npy", allow_pickle=True))
scaler = joblib.load('global_scaler.pkl')

model = TwoBranchHARModel(
    total_features=len(feature_cols),
    imu_dim=imu_dim,
    tof_thm_dim=tof_thm_dim,
    pad_len=pad_len,
    num_classes=num_classes,
    wd=1e-4
).to(device)
state_dict = torch.load("gesture_two_branch_mixup_pytorch.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

def preprocess_sequence(df_seq: pd.DataFrame):
    """
    Process a single sequence DataFrame (pandas):
    - Forward/backward fill missing
    - Scale using loaded scaler
    - Pad/truncate to pad_len
    - Return torch.Tensor of shape (1, features, seq_len)
    """
    data = df_seq[feature_cols].ffill().bfill().fillna(0).values
    scaled = scaler.transform(data)  
    # Pad/truncate
    padded = keras_pad_sequences(
        [scaled],
        maxlen=pad_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )[0]  
  
    tensor = torch.from_numpy(padded.T).unsqueeze(0).float()  # (1, features, pad_len)
    return tensor

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle evaluation API will call this for each sequence.
    sequence: polars DataFrame for a single sequence
    demographics: unused in this model
    Returns: predicted gesture string
    """
    df_seq = sequence.to_pandas()
    x_tensor = preprocess_sequence(df_seq).to(device)
    with torch.no_grad():
        outputs = model(x_tensor)                  # (1, num_classes)
        pred_idx = torch.argmax(outputs, dim=1).item()
    return str(gesture_classes[pred_idx])




inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

