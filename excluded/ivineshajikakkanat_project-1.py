import os

# DATA_DIR = "/kaggle/input/malware-classification/"

print(os.listdir("/kaggle/input/malware-classification/"))


!apt-get install -y p7zip-full



!mkdir -p /kaggle/working/sampleData
!7z e /kaggle/input/malware-classification/dataSample.7z -o/kaggle/working/sampleData


print(os.listdir("/kaggle/working/sampleData")[:10])



import pandas as pd

labels = pd.read_csv("/kaggle/input/malware-classification/trainLabels.csv")
print(labels.head())



import os

bytes_files = [f for f in os.listdir("/kaggle/working/sampleData") if f.endswith(".bytes")]
print("Found", len(bytes_files), "bytes files")
print(bytes_files[:5])



sample_path = "/kaggle/working/sampleData/" + bytes_files[0]

with open(sample_path, "r") as f:
    lines = f.readlines()[:10]  # read first 10 lines

for l in lines:
    print(l.strip())

