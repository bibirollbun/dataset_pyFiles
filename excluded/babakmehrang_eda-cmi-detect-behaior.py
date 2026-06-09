import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os


df= pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
df.shape


df.iloc[0:3,0:21]#max_col_num: 341


df.describe()


print("phase:",df['phase'].unique(),'\n ', 30 * '--') 
print("sequence_type:",df['sequence_type'].unique(),'\n ', 30 * '--')
print("orientation:",df['orientation'].unique(),'\n ', 30 * '--')
print("behavior:",df['behavior'].unique(),'\n ', 30 * '--')
print("gesture:",df['gesture'].unique(),'\n ', 30 * '--')
print("subject:",df['subject'].unique(),'\n ', 30 * '--')

#print(df.nunique()[:22])


df['sequence_counter'].plot(kind='hist', bins=700)


df['sequence_counter'][df['sequence_counter']<50].plot(kind='hist', bins=50)


#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
print(df.isnull().sum())


df_sequence_lengths = df.groupby("gesture").size()
df_sequence_lengths.sort_index().plot(kind='bar')
print(df_sequence_lengths.sort_index())


# Step 1: Count number of unique sequences per gesture
gesture_seq_counts = df.groupby('gesture')['sequence_id'].nunique()
# Step 2: Plot histogram (bar chart)
gesture_seq_counts.sort_values().plot(kind='barh', figsize=(10,6))
plt.xlabel("Number of Unique Sequences")
plt.ylabel("Gesture")
plt.title("Number of Unique Sequences per Gesture")
plt.tight_layout()
plt.show()



for seq_id, group in df.groupby('sequence_id'):
    if(len(group['gesture'].unique() )> 1):
        print(f"\n--- Sequence ID: {seq_id} ---")
        print(group['gesture'].unique())


for seq_id, group in df.groupby('sequence_id'):
    if(len(group['orientation'].unique() )> 1):
        print(f"\n--- Sequence ID: {seq_id} ---")
        print(group['orientation'].unique())


for seq_id, group in df.groupby('sequence_id'):
    l=len(group['behavior'].unique() )
    if(l< 3):
        print(f"\n--- Sequence ID: {seq_id} ---")
        print(len(group['behavior'].unique()))


# انتخاب 10 sequence_id اول
sequence_ids = df['sequence_id'].unique()[:10]

# رنگ‌های پس‌زمینه برای behavior
colors = ['#d0f5cb', '#f8fa96', '#faafac', '#fff0b3', '#f0e6ff', '#d9e3f0', '#e6ffe6']

for sequence_id in sequence_ids:
    df_seq = df[df['sequence_id'] == sequence_id].copy()

    # اگر داده‌ای برای این sequence_id وجود نداشت، رد شو
    if df_seq.empty:
        continue

    # گرفتن اطلاعات اولیه
    gesture = df_seq['gesture'].iloc[0]
    last_step = df_seq['sequence_counter'].iloc[-1]

    # تعیین مرزهای رفتارها
    beh_idx = df_seq.groupby('behavior')['sequence_counter'].first().sort_values()
    boundaries = list(beh_idx.values) + [last_step]
    behaviors = list(beh_idx.index)

    # رسم نمودار
    plt.figure(figsize=(12, 5))

    # رنگ‌آمیزی نواحی براساس behavior
    for i in range(len(behaviors)):
        plt.axvspan(boundaries[i], boundaries[i + 1],
                    facecolor=colors[i % len(colors)],
                    alpha=0.3, label=f'{behaviors[i]}')

    # رسم داده‌ها
    plt.plot(df_seq['sequence_counter'], df_seq['acc_x'], label='acc_x', color='blue')
    plt.plot(df_seq['sequence_counter'], df_seq['acc_y'], label='acc_y', color='green')
    plt.plot(df_seq['sequence_counter'], df_seq['acc_z'], label='acc_z', color='red')

    # اطلاعات نمودار
    plt.xlabel('Time Step (sequence_counter)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'Acceleration Over Time - {sequence_id} - {gesture}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# انتخاب 10 sequence_id اول
sequence_ids = df['sequence_id'].unique()[:10]

# رنگ‌های پس‌زمینه برای behavior
colors = ['#d0f5cb', '#f8fa96', '#faafac', '#fff0b3', '#f0e6ff', '#d9e3f0', '#e6ffe6']

for sequence_id in sequence_ids:
    df_seq = df[df['sequence_id'] == sequence_id].copy()

    # اگر داده‌ای برای این sequence_id وجود نداشت، رد شو
    if df_seq.empty:
        continue

    # گرفتن اطلاعات اولیه
    gesture = df_seq['gesture'].iloc[0]
    last_step = df_seq['sequence_counter'].iloc[-1]

    # تعیین مرزهای رفتارها
    beh_idx = df_seq.groupby('behavior')['sequence_counter'].first().sort_values()
    boundaries = list(beh_idx.values) + [last_step]
    behaviors = list(beh_idx.index)

    # رسم نمودار
    plt.figure(figsize=(12, 5))

    # رنگ‌آمیزی نواحی براساس behavior
    for i in range(len(behaviors)):
        plt.axvspan(boundaries[i], boundaries[i + 1],
                    facecolor=colors[i % len(colors)],
                    alpha=0.3, label=f'{behaviors[i]}')

    # رسم داده‌ها
    plt.plot(df_seq['sequence_counter'], df_seq['rot_x'], label='rot_x', color='blue')
    plt.plot(df_seq['sequence_counter'], df_seq['rot_y'], label='rot_y', color='green')
    plt.plot(df_seq['sequence_counter'], df_seq['rot_z'], label='rot_z', color='red')
    plt.plot(df_seq['sequence_counter'], df_seq['rot_w'], label='rot_w', color='brown')

    # اطلاعات نمودار
    plt.xlabel('Time Step (sequence_counter)')
    plt.ylabel('orientation')
    plt.title(f'orientation Over Time - {sequence_id} - {gesture}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




# select sequence_ids
sequence_ids = df['sequence_id'].unique()[:10]

# back ground behavior
colors = ['#d0f5cb', '#f8fa96', '#faafac', '#fff0b3', '#f0e6ff', '#d9e3f0', '#e6ffe6']

for sequence_id in sequence_ids:
    df_seq = df[df['sequence_id'] == sequence_id].copy()

    # اگر داده‌ای برای این sequence_id وجود نداشت، رد شو
    if df_seq.empty:
        continue

    # get primery infos
    gesture = df_seq['gesture'].iloc[0]
    last_step = df_seq['sequence_counter'].iloc[-1]

    # تعیین مرزهای رفتارها
    beh_idx = df_seq.groupby('behavior')['sequence_counter'].first().sort_values()
    boundaries = list(beh_idx.values) + [last_step]
    behaviors = list(beh_idx.index)

    # رسم نمودار
    plt.figure(figsize=(12, 5))

    # رنگ‌آمیزی نواحی براساس behavior
    for i in range(len(behaviors)):
        plt.axvspan(boundaries[i], boundaries[i + 1],
                    facecolor=colors[i % len(colors)],
                    alpha=0.3, label=f'{behaviors[i]}')

    # رسم داده‌ها
    plt.plot(df_seq['sequence_counter'], df_seq['thm_1'], label='thm_1', color='blue')
    plt.plot(df_seq['sequence_counter'], df_seq['thm_2'], label='thm_2', color='green')
    plt.plot(df_seq['sequence_counter'], df_seq['thm_3'], label='thm_3', color='red')
    plt.plot(df_seq['sequence_counter'], df_seq['thm_4'], label='thm_4', color='brown')
    plt.plot(df_seq['sequence_counter'], df_seq['thm_5'], label='thm_5', color='brown')

    # اطلاعات نمودار
    plt.xlabel('Time Step (sequence_counter)')
    plt.ylabel('temperature')
    plt.title(f'temperature Over Time - {sequence_id} - {gesture}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# انتخاب 10 sequence_id اول
sequence_ids = df['sequence_id'].unique()[:10]

# رنگ‌های پس‌زمینه برای رفتارها
colors = ['#d0f5cb', '#f8fa96', '#faafac', '#fff0b3', '#f0e6ff', '#d9e3f0', '#e6ffe6']

for sequence_id in sequence_ids:
    df_seq = df[df['sequence_id'] == sequence_id].copy()
    if df_seq.empty:
        continue

    # محاسبه میانگین TOF برای هر سنسور
    for sensor in range(1, 6):
        cols = [f"tof_{sensor}_v{i}" for i in range(64)]
        df_seq[f"tof_{sensor}_mean"] = df_seq[cols].replace(-1, np.nan).mean(axis=1)

    # گرفتن gesture و مرزهای رفتار
    gesture = df_seq['gesture'].iloc[0]
    last_step = df_seq['sequence_counter'].iloc[-1]
    beh_idx = df_seq.groupby('behavior')['sequence_counter'].first().sort_values()
    boundaries = list(beh_idx.values) + [last_step]
    behaviors = list(beh_idx.index)

    # رسم نمودار
    plt.figure(figsize=(12, 5))

    # رنگی کردن نواحی رفتار
    for i in range(len(behaviors)):
        plt.axvspan(boundaries[i], boundaries[i + 1],
                    facecolor=colors[i % len(colors)],
                    alpha=0.3, label=f'{behaviors[i]}')

    # رسم منحنی میانگین TOF
    for sensor in range(1, 6):
        plt.plot(df_seq['sequence_counter'],
                 df_seq[f"tof_{sensor}_mean"],
                 label=f"TOF {sensor}")

    # عنوان و تنظیمات نهایی
    plt.xlabel('Time Step (sequence_counter)')
    plt.ylabel('Mean TOF Value')
    plt.title(f"Mean TOF Sensor Values Over Time - {sequence_id} - {gesture}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


