import os
import pandas as pd

csv_dir = "/kaggle/input/csv-files"
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

total_rows = 0
unique_images = set()

for fn in csv_files:
    path = os.path.join(csv_dir, fn)
    df = pd.read_csv(path)
    # count all rows (each is one crop)
    n = len(df)
    total_rows += n
    
    # also track unique (study, series, instance) triples
    unique_images.update(
        zip(df['study_id'], df['series_id'], df['instance_number'])
    )
    
    print(f"{fn}: {n} crops")

print(f"\nTotal crops across all conditions: {total_rows}")
print(f"Total unique image slices:       {len(unique_images)}")



import os
import pandas as pd

# Directory containing the three condition CSVs
csv_dir = "/kaggle/input/csv-files"
csv_files = [os.path.join(csv_dir, f) 
             for f in os.listdir(csv_dir) if f.endswith(".csv")]

# Read and concatenate all rows
all_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Total number of crops (rows)
total_crops = len(all_df)

# Count by severity
severity_counts = all_df['score'].value_counts()

print(f"Total crops with x,y coordinates: {total_crops}\n")
print("Counts by severity:")
print(severity_counts.to_string())



import os
import pandas as pd

csv_dir = "/kaggle/input/csv-files"
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

print("Counts per condition (rows = number of x,y crops):\n")
for fn in csv_files:
    path = os.path.join(csv_dir, fn)
    df = pd.read_csv(path)
    counts = df['score'].value_counts()
    total = len(df)
    print(f"{fn}: {total} total")
    for severity, cnt in counts.items():
        print(f"  {severity}: {cnt}")
    print()



import os
import pandas as pd

# Path to CSVs (adjust to your Kaggle input folder)
csv_dir = "/kaggle/input/csv-files"
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]

# Combine all into one DataFrame
all_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

print("Total crops:", len(all_df))
print("Unique slices:", all_df[['study_id','series_id','instance_number']].drop_duplicates().shape[0])
all_df.head()



condition_counts = all_df.groupby(['condition', 'score']).size().unstack(fill_value=0)
print(condition_counts)



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.countplot(data=all_df, x='score', order=['Normal/Mild','Moderate','Severe'], palette="Set2")
plt.title("Severity Distribution Across All Conditions")
plt.show()



condition_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10,6),
    colormap="viridis"
)
plt.title("Condition × Severity Distribution")
plt.ylabel("Number of crops")
plt.show()



all_df['score'].value_counts().plot(
    kind="pie",
    autopct='%1.1f%%',
    startangle=90,
    colors=["#66c2a5","#fc8d62","#8da0cb"]
)
plt.ylabel("")
plt.title("Overall Severity Class Distribution")
plt.show()





