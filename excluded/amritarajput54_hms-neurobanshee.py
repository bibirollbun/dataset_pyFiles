# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, array
from pyspark.sql.types import DoubleType, ArrayType, StringType
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EEG Analysis") \
    .getOrCreate()

# Path to data
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'
FILE_PATH = BASE_PATH + 'train_eegs/1000913311.parquet'



# # Shutdown Spark
# spark.stop()


df_eeg = spark.read.parquet(FILE_PATH)
df_eeg


 df_eeg.show(5)



 df = spark.read.csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv", 
                     header=True, inferSchema=True)

# Display the first few rows
df.show(5)


# if "eval" in FLAGS:
#     import os

#     # Set the environment variable
#     os.environ["PYSPARK_PIN_THREAD"] = "False"
#     # spark.builder.config("spark.jars.packages", "org.mlflow.mlflow-spark")
#     import mlflow

#     # mlflow.set_tracking_uri("http://127.0.0.0:5000")
#     mlflow.set_tracking_uri("http://localhost:5000")
#     mlflow.autolog()




 # Extract column names
columns = df.columns
TARGETS = columns[-6:]

# Print shape (row count and column count)
print("Train shape:", (df.count(), len(columns)))

# Display target column names
print("Target Labels:", TARGETS)



from pyspark.sql.functions import col

# Count the number of occurrences of each EEG pattern
for target in TARGETS:
    df.groupBy(target).count().orderBy(col("count").desc()).show()



from pyspark.sql.functions import first, min, max, sum, col

# Select the first spectrogram_id and earliest spectrogram_label_offset_seconds for each eeg_id
train = df.groupBy("eeg_id").agg(
    first("spectrogram_id").alias("spec_id"),
    min("spectrogram_label_offset_seconds").alias("min")
)

# Find the latest spectrogram_label_offset_seconds
tmp = df.groupBy("eeg_id").agg(max("spectrogram_label_offset_seconds").alias("max"))
train = train.join(tmp, on="eeg_id", how="left")



tmp = df.groupBy("eeg_id").agg(first("patient_id").alias("patient_id"))
train = train.join(tmp, on="eeg_id", how="left")



target_agg = df.groupBy("eeg_id").agg(
    *[sum(col(t)).alias(t) for t in TARGETS]  # Sum votes for each target label
)

train = train.join(target_agg, on="eeg_id", how="left")


train.show()


train = df
train.show()


from pyspark.sql.functions import sum as spark_sum

from pyspark.sql.functions import monotonically_increasing_id


vote_cols = [col for col in train.columns if '_vote' in col] 
print("vote cols:", vote_cols)
colss=["eeg_id", "spectrogram_id", "patient_id"]

# Group by eeg_id, spectrogram_id, patient_id and sum the vote columns
train_group = train.groupBy("eeg_id", "spectrogram_id", "patient_id")\
                   .agg(*[spark_sum(col).alias(col) for col in vote_cols]).withColumn("index_column", monotonically_increasing_id())

train_group.show(7)



from pyspark.sql.functions import sum as spark_sum, monotonically_increasing_id

# Add an index column if needed (but not necessary for groupBy)
train = train.withColumn("index_column", monotonically_increasing_id())

# Define vote columns
vote_cols = [col for col in train.columns if '_vote' in col]
print("Vote cols:", vote_cols)

# Group by eeg_id, spectrogram_id, patient_id and sum the vote columns
train_group = train.groupBy("eeg_id", "spectrogram_id", "patient_id")\
                   .agg(*[spark_sum(col).alias(col) for col in vote_cols])

# Show the first 7 rows
train_group.show(7)






from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import numpy, pandas
import pyspark.pandas as ps
import numpy as np
ps.set_option('compute.ops_on_diff_frames', True)

def categorize_votes(row):
    # compute max and sum
    col_names = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    max_vote = row[col_names].max()
    total_votes = row[col_names].sum()
 
    percentage = max_vote / total_votes * 100

    high_agreement_threshold = 70
    equal_splitting_threshold = 40

    if percentage >= high_agreement_threshold:
        return 'idealized'
    elif row['other_vote'] / total_votes >= 0.4 and percentage >= equal_splitting_threshold:
        return 'proto'
    elif row['other_vote'] == 0 and percentage >= equal_splitting_threshold:
        return 'edge'
    else:
        return 'undecided'


 
train_group= ps.DataFrame(train_group)
train_group['pattern'] = train_group.apply(categorize_votes, axis=1)
train_group.head(7)




train_group[train_group.eeg_id==722738444]


train[train.eeg_id==722738444].show()


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, min
import os

# Initialize Spark session 

# Define path to spectrogram parquet files
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"

# List all parquet files
files = [f for f in os.listdir(PATH) if f.endswith(".parquet")]

print(f"There are {len(files)} spectrogram parquet files.")



spectrogram_id = 789577333

# read in the data
spec_base_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
spec_data = spark.read.parquet(spec_base_path + str(spectrogram_id) + ".parquet")
 
print((spec_data.count(), len(spec_data.columns)))
spec_data.show(2)


from pprint import pprint

# number of spectrograms for each category
N = 5

# Define the categories
categories = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

# Filter DataFrame to only include "idealized" pattern - using correct pandas-on-spark syntax
idealized_df = train_group[train_group["pattern"] == "idealized"].reset_index(drop=True)

# Initialize empty dictionary to store results
spec_dict = {}

# For each category, find the top N rows by vote value
for category in categories:
    # Sort by the category column and get top N rows
    top_rows = idealized_df.sort_values(by=category, ascending=False).head(N)
    
    # Get the spectrogram_ids as a numpy array
    spec_dict[category] = top_rows["spectrogram_id"].to_numpy()

pprint(spec_dict)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")  # Dark theme for contrast

def plot_spectrograms_by_category(spectrogram_ids, category, cmap="bone"):
    # Base path for spectrograms
    spec_base_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    
    # Read spectrogram data and handle NaN
    spec_data = [pd.read_parquet(spec_base_path + str(id) + ".parquet").fillna(0) for id in spectrogram_ids]  

    # Avoid log(0) issues
    epsilon = 1e-10  

    # ğŸ”¥ Make the plots **bigger**
    fig, axes = plt.subplots(1, len(spectrogram_ids), figsize=(30, 10), sharey=False)  # Bigger size
    plt.suptitle(f"{category}", fontsize=24, fontweight="bold", color="black")

    if len(spectrogram_ids) == 1:
        axes = [axes]  # Ensure it's always iterable

    for i, ax in enumerate(axes):
        # Apply log transformation safely
        log_spec_data = np.log(np.maximum(spec_data[i].T, epsilon))  # Avoid NaN & log(0)
        
        # Plot spectrogram with **bigger size**
        img = ax.imshow(log_spec_data, cmap=cmap, aspect="auto")

        # Set labels and title
        ax.set_title(f'ID {spectrogram_ids[i]}', fontsize=18, fontweight="bold", color="black")
        ax.set_xlabel("Time", fontsize=18, fontweight="bold", color="black")
        ax.set_ylabel("Frequency (Hz)", fontsize=18, fontweight="bold", color="black")

        # Set ticks color
        ax.tick_params(axis='both', labelsize=14, colors="black")
        axes[i].tick_params(axis='both', which='both', labelsize=10)

        # Add color bar for each spectrogram
        cbar = fig.colorbar(img, ax=ax, fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=14, colors="black")

    plt.tight_layout()  # Ensure labels are not cropped
    plt.subplots_adjust(top=0.85)
    plt.show()

# Plot spectrograms for each category using high-contrast color maps
dark_color_maps = ["bone", "gray", "afmhot", "gist_heat", "cividis", "inferno"]
for i, (key, values) in enumerate(spec_dict.items()):
    plot_spectrograms_by_category(spectrogram_ids=values, category=key, cmap=dark_color_maps[i % len(dark_color_maps)])
 


 
import os
import gc
import wandb
import random
import math
from glob import glob
from tqdm import tqdm
from time import time
from pprint import pprint
import warnings
import pandas as pd
import numpy as np
from scipy.signal import spectrogram

# visuals
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt



 spect_data = np.load("/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True).item()



print(spect_data[319287046])
print(spect_data[319287046].shape)



import pandas as pd
# Step 3: Extract the NumPy array from the dictionary
spectrogram_array = list(spect_data.values())[0]  # Extract the first value (NumPy array) from the dictionary

# Step 4: Load column names from the sample Parquet file using PySpark
sample_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/1000086677.parquet"
feature_col_names = spark.read.parquet(sample_path).columns[1:]  # Skip the first column (e.g., 'time' or 'index')

# # Step 5: Convert the NumPy array to a Pandas DataFrame with the extracted column names
# pdf = pd.DataFrame(spectrogram_array, columns=feature_col_names)

# # Step 6: Convert the Pandas DataFrame to a PySpark DataFrame
# ppdf = spark.createDataFrame(pdf)

# Step 7: Show the DataFrame
# df.show(1)



fe_data = {}

# Iterate over spectrogram data with tqdm for progress tracking
for spect_id, data in tqdm(spect_data.items()):  # âœ… Correct usage
    fe_data[spect_id] = {}
    
    for k, feature in enumerate(feature_col_names):
        fe_data[spect_id][f"{feature}_mean"] = data[:, k].mean()
        fe_data[spect_id][f"{feature}_min"] = data[:, k].min()
        fe_data[spect_id][f"{feature}_max"] = data[:, k].max()
        fe_data[spect_id][f"{feature}_std"] = data[:, k].std()


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, first
from pyspark.ml.feature import StringIndexer
import pandas as pd
# Step 2: Convert fe_data dictionary to a PySpark DataFrame
fe_data_df = spark.createDataFrame(pd.DataFrame.from_dict(fe_data, orient='index').reset_index())

# Step 3: Load target labels from the train DataFrame
# Assuming `train` is already a PySpark DataFrame
# If not, convert it to a PySpark DataFrame:
# train = spark.createDataFrame(train)

# Group by spectrogram_id and get the first expert_consensus value
target_df = train.groupBy("spectrogram_id").agg(first("expert_consensus").alias("expert_consensus"))

# Step 4: Encode expert_consensus from string to numerical values
# Use StringIndexer to encode the expert_consensus column
string_indexer = StringIndexer(inputCol="expert_consensus", outputCol="expert_consensus_encoded")
target_df = string_indexer.fit(target_df).transform(target_df)

# Step 5: Merge fe_data_df with target_df on spectrogram_id
final_df = fe_data_df.join(target_df, fe_data_df["index"] == target_df["spectrogram_id"], how="inner")

# # Step 6: Show the final DataFrame
# final_df.show()


final_df = final_df.drop("expert_consensus")
print(final_df.count(), len(final_df.columns))


# final_df.write.csv("/kaggle/working/final_df", header=True)




# final_df.to_parquet("/kaggle/working/final_df.parquet")




# Step 2: Display the first 10 and last 10 column names
col_names = final_df.columns

print("First 10 column names:")
print(col_names[:10])

print("\nLast 10 column names:")
print(col_names[-10:])

# # Step 3: Display 'expert_consensus' and 'expert_consensus_encoded' for the first 5 rows
# final_df.select("expert_consensus", "expert_consensus_encoded").show(5)


# # Select distinct pairs of expert_consensus and expert_consensus_encoded
# mapping_df = final_df.select("expert_consensus", "expert_consensus_encoded").distinct()

# # Sort by expert_consensus_encoded for better readability
# mapping_df = mapping_df.orderBy("expert_consensus_encoded")

# # Show the mapping
# mapping_df.show(truncate=False)


index_value = 319287046
filtered_df = final_df.filter(col("index") == index_value)

# Step 3: Select and display the required columns
result = filtered_df.select("index", "spectrogram_id", "expert_consensus", "expert_consensus_encoded")

# Step 4: Show the result
result.show(truncate=False)


pip install bigdl-spark3  # Install the latest release version

!pip install bigdl --upgrade
# Let's check what modules are available in bigdl.dllib.keras
import bigdl.dllib.keras as keras
print(dir(keras))

# Let's also check what's in the bigdl.dllib package
import bigdl.dllib as dllib
print(dir(dllib))
!pip install pyspark==3.3.0 --force-reinstall

!python --version
!pip show pyspark
!pip show bigdl-dllib



from bigdl.dllib.nncontext import init_nncontext
from bigdl.dllib.feature.image import *
from bigdl.dllib.keras.models import Model
from bigdl.dllib.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape
from bigdl.dllib.optim.optimizer import Adam, MaxIteration
from bigdl.dllib.keras.objectives import CategoricalCrossEntropy # Use this instead
from pyspark.sql import SparkSession
from bigdl.dllib.utils.common import Sample


# # spark = SparkSession.builder.appName("EEG_Classification").getOrCreate()
# sc = init_nncontext()

# # ğŸ”¹ Convert Spark DataFrame to BigDL FeatureSet
# def df_to_featureset(df, feature_cols, label_col):
#     rdd = df.rdd.map(lambda row: 
#         Sample.from_ndarray(
#             row[feature_cols], 
#             row[label_col]
#         )
#     )
#     return rdd
 
# # Exclude index, expert_consensus, and expert_consensus_encoded columns
# feature_cols = [col for col in final_df.columns if col not in ["index", "expert_consensus", "expert_consensus_encoded"]]
# label_col = "expert_consensus_encoded"  # Use the encoded label column

# train_rdd = df_to_featureset(final_df, feature_cols, label_col)
 
# input_layer = Input(shape=(len(feature_cols),))  # EEG features
# x = Reshape((1, len(feature_cols), 1))(input_layer)  # Reshape for CNN

# x = Conv2D(32, kernel_size=(1,3), activation="relu", padding="same")(x)
# x = MaxPooling2D(pool_size=(1,2))(x)
# x = Flatten()(x)

# x = Reshape((1, -1))(x)  # Reshape for LSTM
# x = LSTM(64, return_sequences=False)(x)
# output_layer = Dense(3, activation="softmax")(x)  # 3-class classification (adjust based on your data)

# model = Model(input_layer, output_layer)
 
# optimizer = Adam(learningrate=0.001)
# model.compile(optimizer=optimizer, loss=ClassNLLCriterion())
 
# model.fit(train_rdd, batch_size=32, nb_epoch=10)
  


# print("Train shape:", (train.count(), len(train.columns)), "\n")  # PySpark equivalent of shape
# print("Unique eeg_ids: ", train.select("eeg_id").distinct().count())
# print(train.groupBy("eeg_id").count().describe().show(), "\n")
# print("Unique spectrogram_ids: ", train.select("spec_id").distinct().count())
# print("Unique patient_ids: ", train.select("patient_id").distinct().count(), "\n")

















 








 


 


 







