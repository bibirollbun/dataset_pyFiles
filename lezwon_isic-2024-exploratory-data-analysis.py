import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


image_path = "/kaggle/input/isic-2024-challenge/train-image/image"
column_types = {
    'iddx_5': 'str',
    'mel_mitotic_index': 'str'
}

df = pd.read_csv('/kaggle/input/isic-2024-challenge/train-metadata.csv', dtype=column_types)
df_test = pd.read_csv('/kaggle/input/isic-2024-challenge/test-metadata.csv')




def open_image(isic_id):
    img_path = f"{image_path}/{isic_id}.jpg"
    return Image.open(img_path)


def thicken_border(subplot):
    subplot.spines['top'].set_color('red')
    subplot.spines['right'].set_color('red')
    subplot.spines['bottom'].set_color('red')
    subplot.spines['left'].set_color('red')
    subplot.spines['top'].set_linewidth(4)
    subplot.spines['right'].set_linewidth(4)
    subplot.spines['bottom'].set_linewidth(4)
    subplot.spines['left'].set_linewidth(4)


def plot_images(df, n=5, columns=None, title=None):
    # main title
    plt.suptitle(title, fontsize=20)


    fig, ax = plt.subplots(1, n, figsize=(20, 20))
    for i in range(n):
        img = open_image(df.iloc[i]["isic_id"])
        
        # Check if target is 1 thicken subplot border and make it red
        if df.iloc[i]['target'] == 1:
            thicken_border(ax[i])
        
        ax[i].imshow(img)
        ax[i].set_title(df.iloc[i]['isic_id'])

        if columns:
            # show at the bottom
            xlabel = "\n".join([f"{col}: {df.iloc[i][col]}" for col in columns])
            ax[i].set_xlabel(xlabel)

    plt.show()


def plot_high_low(df, column, additional_columns=None):
    # Descending
    filtered = df.sort_values(by=column, ascending=False).head(5)
    plot_images(filtered, columns=[column] + (additional_columns or []))

    # Ascending
    filtered = df.sort_values(by=column, ascending=True).head(5)
    plot_images(filtered, columns=[column] + (additional_columns or []))


def plot_person(df, patient_id):
    patient_df = df[df.patient_id == patient_id]
    
    fig, ax1 = plt.subplots(figsize=(10, 15))
    
    # Scatter plot
    one_patient_0 = patient_df[patient_df.target == 0]
    one_patient_1 = patient_df[patient_df.target == 1]

    ax1.scatter(one_patient_0['tbp_lv_x'], one_patient_0['tbp_lv_y'], s=one_patient_0['tbp_lv_z'], alpha=0.5, c='green', label='Benign')
    ax1.scatter(one_patient_1['tbp_lv_x'], one_patient_1['tbp_lv_y'], s=one_patient_1['tbp_lv_z'], alpha=0.5, c='red', label='Malignant')

    ax1.set_xlabel('tbp_lv_x')
    ax1.set_ylabel('tbp_lv_y')
    ax1.set_title('2D Scatter Plot of tbp_lv_x vs tbp_lv_y with tbp_lv_z as Size')
    ax1.legend()
    plt.show()



def plot_scatter(df, x_col, y_col, target_col):
    # Plotting 

    # Separate the data based on target
    df_target_0 = df[df[target_col] == 0]
    df_target_1 = df[df[target_col] == 1]

    plt.figure(figsize=(10, 6))

    # Plot target 0 (green) first
    plt.scatter(df_target_0[y_col], df_target_0[x_col], c='green', alpha=0.5, label='Target 0')

    # Plot target 1 (red) on top
    plt.scatter(df_target_1[y_col], df_target_1[x_col], c='red', alpha=0.5, label='Target 1')

    plt.xlabel(y_col)
    plt.ylabel(x_col)
    plt.title(f'{x_col} vs {y_col} with Target as Green and Red')
    plt.legend()
    plt.show()



def plot_person_with_details(df, patient_id):
    patient_df = df[df.patient_id == patient_id]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    
    # Scatter plot
    one_patient_0 = patient_df[patient_df.target == 0]
    one_patient_1 = patient_df[patient_df.target == 1]

    ax1.scatter(one_patient_0['tbp_lv_x'], one_patient_0['tbp_lv_y'], s=one_patient_0['tbp_lv_z'], alpha=0.5, c='green', label='Benign')
    ax1.scatter(one_patient_1['tbp_lv_x'], one_patient_1['tbp_lv_y'], s=one_patient_1['tbp_lv_z'], alpha=0.5, c='red', label='Malignant')

    ax1.set_xlabel('tbp_lv_x')
    ax1.set_ylabel('tbp_lv_y')
    ax1.set_title('2D Scatter Plot of tbp_lv_x vs tbp_lv_y with tbp_lv_z as Size')
    ax1.legend()

    # Details plot
    details = patient_df.iloc[0][['isic_id', 'age_approx', 'sex', 'patient_id']].reset_index()
    details.columns = ['Field', 'Value']
    ax2.axis('off')
    table = ax2.table(cellText=details.values, colLabels=details.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.show()


from sklearn.preprocessing import MinMaxScaler

def plot_all(df):
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['normalized_z'] = scaler.fit_transform(df[['tbp_lv_z']])

    plt.figure(figsize=(6, 10))

    one_patient_0 = df[df.target == 0]
    one_patient_1 = df[df.target == 1]

    plt.scatter(one_patient_0['tbp_lv_x'], one_patient_0['tbp_lv_y'], s=one_patient_0['normalized_z'], alpha=0.5, c='green', label='Benign')
    plt.scatter(one_patient_1['tbp_lv_x'], one_patient_1['tbp_lv_y'], s=one_patient_1['normalized_z'], alpha=0.5, c='red', label='Malignant')


    plt.xlabel('tbp_lv_x')
    plt.ylabel('tbp_lv_y')
    plt.title('Scatter Plot of tbp_lv_x vs tbp_lv_y')
    plt.show()


def offset_coords(df):
    patient_ids = df[df.tbp_lv_y < 0].patient_id
    to_offset = df[df.patient_id.isin(patient_ids)].copy()
    to_offset['tbp_lv_y'] = to_offset['tbp_lv_y'] - to_offset['tbp_lv_y'].min()
    df.update(to_offset)
    return df


pd.set_option('display.max_columns', None)


df.head(5)


plot_images(df[df.target == 1], 5, title='Malignant')


plot_images(df[df.target == 0], 5, title='Benign')


df['target'].hist()
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.title('Histogram of Target')
plt.xticks([0, 1])
plt.show()


df['sex'].value_counts().plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Sex Distribution')
plt.show()


df['age_approx'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Age')
plt.ylabel('Counts')
plt.title('Histogram of Age')
plt.show()


df.groupby(['age_approx', 'sex']).size().unstack().plot(kind='bar', stacked=False, figsize=(20, 10))
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age and Gender Distribution')
plt.show()


plot_person(df, 'IP_1959239')


plot_person_with_details(df, 'IP_1959239')


# some of the coordinates are negative, we need to offset them
df = offset_coords(df)


plot_all(df)


plot_high_low(df[df.target == 0], 'tbp_lv_radial_color_std_max', ['target', 'sex', 'age_approx'])


plot_high_low(df[df.target == 1], 'tbp_lv_radial_color_std_max', ['target', 'sex', 'age_approx'])


plot_high_low(df[df.target == 0], 'tbp_lv_symm_2axis', ['target', 'sex', 'age_approx'])


plot_high_low(df[df.target == 1], 'tbp_lv_symm_2axis', ['target', 'sex', 'age_approx'])


# Group tbp_lv_areaMM2 by bins of 10
df_filtered = df[df.target == 1].copy()
bins = range(0, int(df_filtered['tbp_lv_areaMM2'].max()) + 10, 10)
df_filtered['tbp_lv_areaMM2_binned'] = pd.cut(df_filtered['tbp_lv_areaMM2'], bins)

# Draw histogram
df_filtered['tbp_lv_areaMM2_binned'].value_counts().sort_index().plot(kind='bar', figsize=(20, 5))
plt.xlabel('tbp_lv_areaMM2 Bins', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Histogram of tbp_lv_areaMM2 Grouped by Bins of 10', fontsize=16)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


plot_scatter(df, 'age_approx', 'tbp_lv_areaMM2', 'target')


plot_scatter(df, 'age_approx', 'tbp_lv_deltaB', 'target')


plot_scatter(df, 'age_approx', 'tbp_lv_deltaA', 'target')


plot_scatter(df, 'age_approx', 'tbp_lv_stdLExt', 'target')




