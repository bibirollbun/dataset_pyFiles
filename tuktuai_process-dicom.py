!git clone https://github.com/Tuprott991/SoftAI---DataForLife---MedSightAI


cd SoftAI---DataForLife---MedSightAI/MedSight3


!git pull


!python preprocess_dicom.py --input_dir "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test" --output_dir "/kaggle/working/test" --csv_file "/kaggle/input/vindr-cxr-physionet/image_labels_test.csv" --size 224 --num_workers 4


# import pydicom

# ds = pydicom.dcmread("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/000434271f63a053c4128a0ba6352c7f.dicom")
# print("Rows:", ds.Rows)
# print("Columns:", ds.Columns)
# print("Bits Stored:", ds.BitsStored)
# print("PhotometricInterpretation:", ds.PhotometricInterpretation)




