print("Sooo what")


import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


Total_Train = train.shape[0]
Total_Sample = sample.shape[0]
Total_Test = test.shape[0]
print("Total Baris kolom Train ", Total_Train - 1)
print("Total Baris kolom Sample ", Total_Sample - 1)
print("Total Baris kolom Test ", Total_Test - 1)


c_train = train.columns.tolist()
c_test = test.columns.tolist()
c_sample = sample.columns.tolist()

print("Columns Train :", c_train)
print("\n")
print("Columns Test:", c_test)
print("\n")
print("Columns Sample :", c_sample)


print("=== Missing Values ===")
print(train.isnull().sum())


print("\n=== Duplicate Rows ===")
print("Jumlah duplikat di train:", train.duplicated().sum())
print("Jumlah duplikat di test:", test.duplicated().sum())


print("\n=== Range tiap fitur ===")
for col in train.columns:
    if train[col].dtype != 'object':
        print(f"{col}: min={train[col].min()}, max={train[col].max()}")


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.hist(train['BeatsPerMinute'], bins=50)
plt.title("Distribusi Beats Per Minute")
plt.xlabel("BPM")
plt.ylabel("Jumlah")


plt.subplot(2, 1, 2)
plt.boxplot(train['BeatsPerMinute'], vert=False)
plt.title("Boxplot Beats Per Minute")
plt.xlabel("BPM")
plt.tight_layout()
plt.show()

