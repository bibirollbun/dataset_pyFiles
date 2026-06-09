# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ------------------------------
# 1. بارگذاری فایل‌های CSV و تعریف مسیرها
# ------------------------------
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

train_df      = pd.read_csv(os.path.join(train_path, 'train.csv'))
label_df      = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
train_desc_df = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))
test_desc_df  = pd.read_csv(os.path.join(train_path, 'test_series_descriptions.csv'))
sub           = pd.read_csv(os.path.join(train_path, 'sample_submission.csv'))



# ------------------------------
# 2. تابع برای تولید مسیرهای تصاویر
# ------------------------------
def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        if os.path.exists(series_dir):
            images = os.listdir(series_dir)
            image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

train_image_paths = generate_image_paths(train_desc_df, os.path.join(train_path, 'train_images'))
test_image_paths  = generate_image_paths(test_desc_df, os.path.join(train_path, 'test_images'))

print("نمونه مسیر از تصاویر train:", train_image_paths[2])
print("تعداد ردیف‌های train_desc:", len(train_desc_df))
print("تعداد تصاویر train:", len(train_image_paths))



# ------------------------------
# 3. تغییر شکل داده‌های train و ادغام دیتا فریم‌ها
# ------------------------------
def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    for column, value in row.items():
        if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
            parts = column.split('_')
            condition = ' '.join([word.capitalize() for word in parts[:-2]])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(value)
    return pd.DataFrame(data)

new_train_df = pd.concat([reshape_row(row) for _, row in train_df.iterrows()], ignore_index=True)

merged_df = pd.merge(new_train_df, label_df, on=['study_id', 'condition', 'level'], how='inner')
final_merged_df = pd.merge(merged_df, train_desc_df, on=['series_id','study_id'], how='inner')

final_merged_df['row_id'] = (final_merged_df['study_id'].astype(str) + '_' +
                               final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
                               final_merged_df['level'].str.lower().str.replace('/', '_'))

final_merged_df['image_path'] = (os.path.join(train_path, 'train_images') + '/' +
                                 final_merged_df['study_id'].astype(str) + '/' +
                                 final_merged_df['series_id'].astype(str) + '/' +
                                 final_merged_df['instance_number'].astype(str) + '.dcm')

# تغییر برچسب severity به حروف کوچک
final_merged_df['severity'] = final_merged_df['severity'].map({
    'Normal/Mild': 'normal_mild',
    'Moderate': 'moderate',
    'Severe': 'severe'
})

# فیلتر کردن ردیف‌هایی که مسیر تصویر موجود است
def check_exists(path):
    return os.path.exists(path)
final_merged_df = final_merged_df[final_merged_df['image_path'].apply(check_exists)]

# نگاشت برچسب‌ها به اعداد صحیح
severity_map = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
final_merged_df['severity'] = final_merged_df['severity'].map(severity_map)

# استفاده از final_merged_df به عنوان داده‌های آموزشی
train_data = final_merged_df.copy()
train_data = train_data.dropna()   # حذف ردیف‌های دارای NaN



# ------------------------------
# 4. توابع بارگذاری و پیش‌پردازش تصاویر DICOM
# ------------------------------
def load_dicom_image(path):
    """
    تابعی برای بارگذاری تصویر DICOM
    """
    path = path.numpy().decode('utf-8')  # تبدیل EagerTensor به رشته
    ds = pydicom.dcmread(path)
    data = ds.pixel_array.astype(np.float32)
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def load_and_preprocess(path, label=None):
    """
    - بارگذاری تصویر DICOM با استفاده از tf.py_function
    - افزودن بعد کانال (برای تصاویر خاکستری)
    - تغییر اندازه به 224x224، تبدیل از grayscale به RGB و نرمال‌سازی به [0, 1]
    """
    image = tf.py_function(func=lambda p: load_dicom_image(p), inp=[path], Tout=tf.uint8)
    image.set_shape([None, None])
    image = tf.expand_dims(image, axis=-1)  # تبدیل (ارتفاع, عرض) به (ارتفاع, عرض, 1)
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    else:
        return image, label



# ------------------------------
# 5. ایجاد دیتاست‌های TensorFlow برای هر سری توضیحی
# ------------------------------
def create_datasets(df, series_description, batch_size=8):
    filtered_df = df[df['series_description'] == series_description]
    if filtered_df.empty:
        raise ValueError(f"داده‌ای برای سری توضیحی: {series_description} پیدا نشد.")
    train_df_part, val_df_part = train_test_split(filtered_df, test_size=0.2, random_state=42)
    
    train_paths = train_df_part['image_path'].values
    train_labels = train_df_part['severity'].values
    val_paths = val_df_part['image_path'].values
    val_labels = val_df_part['severity'].values
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(lambda p, l: load_and_preprocess(p, l),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(train_df_part)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(lambda p, l: load_and_preprocess(p, l),
                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, len(train_df_part), len(val_df_part)

# ایجاد دیتاست‌ها برای سه سری توضیحی
train_ds_t1, val_ds_t1, len_train_t1, len_val_t1 = create_datasets(train_data, 'Sagittal T1', batch_size=8)
train_ds_t2, val_ds_t2, len_train_t2, len_val_t2 = create_datasets(train_data, 'Axial T2', batch_size=8)
train_ds_t2stir, val_ds_t2stir, len_train_t2stir, len_val_t2stir = create_datasets(train_data, 'Sagittal T2/STIR', batch_size=8)



# ------------------------------
# 6. تعریف مدل VGG16 با استفاده از Keras (TensorFlow)
# ------------------------------
def create_vgg16_model(num_classes=3):
    base_model = tf.keras.applications.VGG16(include_top=False,
                                             input_shape=(224, 224, 3),
                                             weights='imagenet')
    base_model.trainable = False
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

# ایجاد سه مدل مجزا برای سری‌های مختلف
model_t1 = create_vgg16_model(num_classes=3)
model_t2 = create_vgg16_model(num_classes=3)
model_t2stir = create_vgg16_model(num_classes=3)

model_t1.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
model_t2.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
model_t2stir.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])


# ------------------------------
# 7. آموزش مدل‌ها
# ------------------------------
es_callback_t1 = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
ckpt_callback_t1 = callbacks.ModelCheckpoint('best_model_t1.keras', monitor='val_accuracy', save_best_only=True)

es_callback_t2 = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
ckpt_callback_t2 = callbacks.ModelCheckpoint('best_model_t2.keras', monitor='val_accuracy', save_best_only=True)

es_callback_t2stir = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
ckpt_callback_t2stir = callbacks.ModelCheckpoint('best_model_t2stir.keras', monitor='val_accuracy', save_best_only=True)

print("آموزش مدل Sagittal T1")
history_t1 = model_t1.fit(train_ds_t1, epochs=10, validation_data=val_ds_t1,
                          callbacks=[es_callback_t1, ckpt_callback_t1])

print("آموزش مدل Axial T2")
history_t2 = model_t2.fit(train_ds_t2, epochs=10, validation_data=val_ds_t2,
                          callbacks=[es_callback_t2, ckpt_callback_t2])

print("آموزش مدل Sagittal T2/STIR")
history_t2stir = model_t2stir.fit(train_ds_t2stir, epochs=10, validation_data=val_ds_t2stir,
                                  callbacks=[es_callback_t2stir, ckpt_callback_t2stir])



# ------------------------------
# 8. رسم نمودارهای تاریخچه آموزش و محاسبه Confusion Matrix برای هر مدل
# ------------------------------
def plot_history_and_confusion(model, history, val_ds, model_name):
    # رسم نمودار loss و accuracy
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # محاسبه پیش‌بینی‌ها روی داده‌های اعتبارسنجی
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    print(f'Classification Report for {model_name}:\n', classification_report(y_true, y_pred))

# رسم نمودارها و confusion matrix برای هر مدل
plot_history_and_confusion(model_t1, history_t1, val_ds_t1, "VGG16 - Sagittal T1")
plot_history_and_confusion(model_t2, history_t2, val_ds_t2, "VGG16 - Axial T2")
plot_history_and_confusion(model_t2stir, history_t2stir, val_ds_t2stir, "VGG16 - Sagittal T2/STIR")



# ------------------------------
# 9. پیش‌بینی روی داده‌های Test و ایجاد سابمیشن
# ------------------------------
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

expanded_rows = []
for index, row in test_desc_df.iterrows():
    study_id = row['study_id']
    series_id = row['series_id']
    series_description = row['series_description']
    series_path = os.path.join(train_path, 'test_images', str(study_id), str(series_id))
    if os.path.exists(series_path):
        image_files = [os.path.join(series_path, f) for f in os.listdir(series_path)
                       if os.path.isfile(os.path.join(series_path, f))]
        conditions = condition_mapping.get(series_description, {})
        if isinstance(conditions, str):
            conditions = {'left': conditions, 'right': conditions}
        for side, condition in conditions.items():
            for image_path in image_files:
                expanded_rows.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'series_description': series_description,
                    'image_path': image_path,
                    'condition': condition,
                    'row_id': f"{study_id}_{condition}"
                })

expanded_test_desc = pd.DataFrame(expanded_rows)

# به‌روزرسانی row_id با اضافه کردن سطح (level)
levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
def update_row_id(row, levels):
    level = levels[row.name % len(levels)]
    return f"{row['study_id']}_{row['condition']}_{level}"

expanded_test_desc['row_id'] = expanded_test_desc.apply(lambda row: update_row_id(row, levels), axis=1)

# ایجاد دیتاست test (بدون برچسب)
test_paths = expanded_test_desc['image_path'].values
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(lambda p: load_and_preprocess(p),
                      num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(1)

# دیکشنری مدل‌ها بر اساس سری توضیحی
models_dict = {
    'Sagittal T1': model_t1,
    'Axial T2': model_t2,
    'Sagittal T2/STIR': model_t2stir
}

normal_mild_probs = []
moderate_probs = []
severe_probs = []
predictions_list = []

for i, batch in enumerate(tqdm(test_ds)):
    series_description = expanded_test_desc.iloc[i]['series_description']
    model_used = models_dict.get(series_description, None)
    if model_used is None:
        normal_mild_probs.append(None)
        moderate_probs.append(None)
        severe_probs.append(None)
        predictions_list.append(None)
    else:
        preds = model_used.predict(batch)
        preds = preds[0]
        normal_mild_probs.append(preds[0])
        moderate_probs.append(preds[1])
        severe_probs.append(preds[2])
        predictions_list.append(preds)

expanded_test_desc['normal_mild'] = normal_mild_probs
expanded_test_desc['moderate'] = moderate_probs
expanded_test_desc['severe'] = severe_probs

submission_df = expanded_test_desc[["row_id", "normal_mild", "moderate", "severe"]]
grouped_submission = submission_df.groupby('row_id').max().reset_index()

sub[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']]
sub.to_csv("/kaggle/working/submission.csv", index=False)

print("نمونه سابمیشن:")
print(sub.head())


# ------------------------------
# 10. ذخیره مدل‌ها
# ------------------------------
model_t1.save("Vgg16_t1.keras")
model_t2.save("Vgg16_t2.keras")
model_t2stir.save("Vgg16_t2stir.keras")








