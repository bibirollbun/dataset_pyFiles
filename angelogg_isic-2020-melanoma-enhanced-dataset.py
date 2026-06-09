import os
import math
import tensorflow as tf
import pandas as pd

def process_single_tfrecord(input_file, output_file, metadata_df, training):
    """
    Reads one TFRecord (input_file), parses it, merges metadata from metadata_df,
    then writes an enhanced TFRecord (output_file).
    """
    # --- 1) Parse function ---
    def parse_tfrecord(example):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_name': tf.io.FixedLenFeature([], tf.string),
        }
        if training:
            features['target'] = tf.io.FixedLenFeature([], tf.int64)
        return tf.io.parse_single_example(example, features)

    # --- 2) Function to add metadata via py_function ---
    # def add_metadata(image, image_name, target):
    #     """Python-side function to look up the row in metadata_df."""
    #     image_name_str = image_name.numpy().decode('utf-8')
    #     row = metadata_df[metadata_df['image_name'] == image_name_str]
    #     if not row.empty:
    #         sex_val = row['sex'].values[0]
    #         sex_val = 'unknown' if pd.isna(sex_val) else sex_val

    #         age_val = row['age_approx'].values[0]
    #         age_val = -1 if pd.isna(age_val) else age_val

    #         patient_val = row['patient_id'].values[0]
    #         patient_val = 'unknown' if pd.isna(patient_val) else patient_val

    #         anatom_val = row['anatom_site_general_challenge'].values[0]
    #         anatom_val = 'unknown' if pd.isna(anatom_val) else anatom_val
    #     else:
    #         sex_val = 'unknown'
    #         age_val = -1
    #         patient_val = 'unknown'
    #         anatom_val = 'unknown'
        
    #     # Convert to tf Tensors
    #     sex_t = tf.convert_to_tensor(sex_val, dtype=tf.string)
    #     age_t = tf.convert_to_tensor(age_val, dtype=tf.int64)
    #     pid_t = tf.convert_to_tensor(patient_val, dtype=tf.string)
    #     site_t = tf.convert_to_tensor(anatom_val, dtype=tf.string)
    #     return sex_t, age_t, pid_t, site_t
    def add_metadata_py(image, image_name):
        image_name_str = image_name.numpy().decode('utf-8')
        row = metadata_df[metadata_df['image_name'] == image_name_str]
        if not row.empty:
            sex_val = row['sex'].values[0]
            sex_val = 'unknown' if pd.isna(sex_val) else sex_val
            age_val = row['age_approx'].values[0]
            age_val = -1 if pd.isna(age_val) else age_val
            patient_val = row['patient_id'].values[0]
            patient_val = 'unknown' if pd.isna(patient_val) else patient_val
            anatom_val = row['anatom_site_general_challenge'].values[0]
            anatom_val = 'unknown' if pd.isna(anatom_val) else anatom_val
        else:
            sex_val = 'unknown'
            age_val = -1
            patient_val = 'unknown'
            anatom_val = 'unknown'

        return (tf.convert_to_tensor(sex_val, tf.string),
                tf.convert_to_tensor(age_val, tf.int64),
                tf.convert_to_tensor(patient_val, tf.string),
                tf.convert_to_tensor(anatom_val, tf.string))

    # --- 3) Combine original record with new metadata ---
    def combine_records(record, metadata):
        sex, age_approx, patient_id, anatom_site = metadata
        record['sex'] = sex
        record['age_approx'] = age_approx
        record['patient_id'] = patient_id
        record['anatom_site_general_challenge'] = anatom_site
        return record

    # --- 4) Serialize combined record back into TFRecord ---
    def serialize_example(record):
        feature = {
            # 'target': tf.train.Feature(
            #     int64_list=tf.train.Int64List(value=[record['target'].numpy()])
            # ),
            'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[record['image'].numpy()])
            ),
            'image_name': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[record['image_name'].numpy()])
            ),
            'sex': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[record['sex'].numpy()])
            ),
            'age_approx': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[record['age_approx'].numpy()])
            ),
            'patient_id': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[record['patient_id'].numpy()])
            ),
            'anatom_site_general_challenge': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[record['anatom_site_general_challenge'].numpy()])
            ),
        }
        if training:
            feature['target'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[record['target'].numpy()]))

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # # --- 5) Build the dataset pipeline for this file ---
    raw_dataset = tf.data.TFRecordDataset([input_file])
    parsed_dataset = raw_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # # Add metadata via py_function
    # tfrecords_with_metadata = parsed_dataset.map(
    #     lambda rec: tf.py_function(
    #         add_metadata,
    #         [rec['image'], rec['image_name'], rec['target']],
    #         Tout=(tf.string, tf.int64, tf.string, tf.string)
    #     )
    # )
    # final_dataset = tf.data.Dataset.zip((parsed_dataset, tfrecords_with_metadata)) \
    #                                .map(combine_records)

    # # --- 6) Materialize and write out ---
    # with tf.io.TFRecordWriter(output_file) as writer:
    #     for record in final_dataset:
    #         example = serialize_example(record)
    #         writer.write(example)

    # Dynamically call py_function with or without target
    if training:
        def pyfn(record):
            return tf.py_function(add_metadata_py, [record['image'], record['image_name']],
                                  [tf.string, tf.int64, tf.string, tf.string])
    else:
        def pyfn(record):
            return tf.py_function(add_metadata_py, [record['image'], record['image_name']],
                                  [tf.string, tf.int64, tf.string, tf.string])

    metadata_dataset = parsed_dataset.map(pyfn, num_parallel_calls=tf.data.AUTOTUNE)
    final_dataset = tf.data.Dataset.zip((parsed_dataset, metadata_dataset)) \
                                   .map(combine_records, num_parallel_calls=tf.data.AUTOTUNE)

    with tf.io.TFRecordWriter(output_file) as writer:
        for record in final_dataset:
            serialized = serialize_example(record)
            writer.write(serialized)



INPUT_DIR = '/kaggle/input/siim-isic-melanoma-classification/'

print('Training dataset:')
metadata_df = pd.read_csv(INPUT_DIR + "train.csv")

# All input TFRecord files
tfrecord_files = tf.io.gfile.glob(INPUT_DIR + 'tfrecords/train*.tfrec')
print(f"Found {len(tfrecord_files)} tfrecords:", tfrecord_files)

# Output directory for enhanced files
output_dir = './enhanced_dataset'
tf.io.gfile.makedirs(output_dir)

# Loop over each input file => produce 1 output file
for infile in sorted(tfrecord_files):
    # e.g. infile = "/kaggle/input/.../train00-2071.tfrec"
    base_name = os.path.basename(infile)
    # e.g. "train00-2071.tfrec" => "train00-2071-enhanced.tfrecord"
    out_name = base_name.replace('.tfrec', '-enhanced.tfrec')
    outfile = os.path.join(output_dir, out_name)
    
    print(f"Processing: {infile} => {outfile}")
    process_single_tfrecord(infile, outfile, metadata_df, training=True)

print('Finished Training dataset.')

print('Test dataset:')
metadata_df = pd.read_csv(INPUT_DIR + "test.csv")
# All input TFRecord files
tfrecord_files = tf.io.gfile.glob(INPUT_DIR + 'tfrecords/test*.tfrec')
print(f"Found {len(tfrecord_files)} tfrecords:", tfrecord_files)

# Loop over each input file => produce 1 output file
for infile in sorted(tfrecord_files):
    base_name = os.path.basename(infile)
    # e.g. "test00-687.tfrec" => "test00-687-enhanced.tfrecord"
    out_name = base_name.replace('.tfrec', '-enhanced.tfrec')
    outfile = os.path.join(output_dir, out_name)
    
    print(f"Processing: {infile} => {outfile}")
    process_single_tfrecord(infile, outfile, metadata_df, training=False)



def parse_enhanced_tfrecord(example):
    features = {
        'target': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'sex': tf.io.FixedLenFeature([], tf.string),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'patient_id': tf.io.FixedLenFeature([], tf.string),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example, features)

enhanced_file = './enhanced_dataset/train01-2071-enhanced.tfrec'
debug_dataset = tf.data.TFRecordDataset(enhanced_file).map(parse_enhanced_tfrecord)

import numpy as np

# Number of records to validate (to avoid printing everything for huge files)
n_samples_to_check = 10

for i, record in enumerate(debug_dataset.take(n_samples_to_check)):
    # Extract the Tensor values in Python
    image_name_str = record['image_name'].numpy().decode('utf-8')
    sex_str = record['sex'].numpy().decode('utf-8')
    age_approx_val = record['age_approx'].numpy()
    patient_id_str = record['patient_id'].numpy().decode('utf-8')
    anatom_site_str = record['anatom_site_general_challenge'].numpy().decode('utf-8')

    # Lookup the corresponding row in the dataframe
    row = metadata_df[metadata_df['image_name'] == image_name_str]
    
    # Print a summary for cross-check
    print(f"--- Record #{i} ---")
    print("image_name:", image_name_str)
    
    if row.empty:
        print("  >> Not found in metadata_df. The TFRecord shows:")
        print(f"     sex={sex_str}, age_approx={age_approx_val}, "
              f"patient_id={patient_id_str}, anatom_site={anatom_site_str}")
    else:
        # There's exactly one row or more, but we assume 1
        row = row.iloc[0]
        
        # In your code, if 'sex' is NaN, you stored 'unknown'
        df_sex = row['sex'] if not pd.isna(row['sex']) else 'unknown'
        df_age = row['age_approx'] if not pd.isna(row['age_approx']) else -1
        df_pid = row['patient_id'] if not pd.isna(row['patient_id']) else 'unknown'
        df_site = row['anatom_site_general_challenge'] if not pd.isna(row['anatom_site_general_challenge']) else 'unknown'
        
        # Convert them all to strings for easy printing/comparison
        df_sex = str(df_sex)
        df_age = str(int(df_age))  # or just str(df_age)
        df_pid = str(df_pid)
        df_site = str(df_site)
        
        # Compare
        match_sex = (sex_str == df_sex)
        match_age = (str(age_approx_val) == df_age)
        match_pid = (patient_id_str == df_pid)
        match_site = (anatom_site_str == df_site)
        
        # Print out the comparisons
        print("  TFRecord vs DataFrame:")
        print(f"    sex: {sex_str} vs {df_sex}  -> {'OK' if match_sex else 'MISMATCH!'}")
        print(f"    age_approx: {age_approx_val} vs {df_age}  -> {'OK' if match_age else 'MISMATCH!'}")
        print(f"    patient_id: {patient_id_str} vs {df_pid}  -> {'OK' if match_pid else 'MISMATCH!'}")
        print(f"    site: {anatom_site_str} vs {df_site}  -> {'OK' if match_site else 'MISMATCH!'}")

    print()


