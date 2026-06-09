### Loading Relevant Packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


current_directory = os.getcwd()
print(current_directory)


path = "/kaggle/input/stanford-rna-3d-folding/"
os.chdir(path)


train_sequence_raw = pd.read_csv("train_sequences.csv")
train_sequence_raw.describe()


train_sequence_raw.head()


### The key in the training data seems to be the sequence of nitrogenous bases

### Let's focus on that for now.


sequence_train_raw = train_sequence_raw["sequence"]
sequence_train_raw[0]


sequence_train_length = sequence_train_raw.apply(lambda x: len(x))
print(np.mean(sequence_train_length))
print(np.median(sequence_train_length))


### The mean length is about 162 nitrogenous bases.

### The median length is about 40 nitrogenous bases.

### The distribution is very skewed. Let's have a look.


# Generate a Histogram of the RNA Chain Length measured in number of nitrogenous bases

# This histogram has all of the lenghts included

data = sequence_train_length

# Define colors for the histogram bars
colors = plt.cm.plasma(np.linspace(0, 1, 15))  # 15 colors from 'plasma' colormap

# Create the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# 'bins' controls the number of bars
n, bins, patches = ax.hist(data, bins=30, edgecolor='black')

# Color each bar individually
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)

# Set y-axis to log scale
ax.set_yscale('log')

# Fancy styling
ax.set_title('Histogram (Log Scale)', fontsize=18)
ax.set_xlabel('Value', fontsize=14)
ax.set_ylabel('Frequency (log scale)', fontsize=14)
ax.grid(True, which="both", linestyle='--', alpha=0.6)

plt.show()



# Generate a Histogram of the RNA Chain Length measured in number of nitrogenous bases

# This histogram has only lengths of 1,000 or less

data = [x for x in sequence_train_length if x <= 1000]

# Define colors for the histogram bars
colors = plt.cm.plasma(np.linspace(0, 1, 15))  # 15 colors from 'plasma' colormap

# Create the histogram
fig, ax = plt.subplots(figsize=(12, 6))

# 'bins' controls the number of bars
n, bins, patches = ax.hist(data, bins=30, edgecolor='black')

# Color each bar individually
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)

# Set y-axis to log scale
ax.set_yscale('log')

# Fancy styling
ax.set_title('Histogram (Log Scale)', fontsize=18)
ax.set_xlabel('Value', fontsize=14)
ax.set_ylabel('Frequency (log scale)', fontsize=14)
ax.grid(True, which="both", linestyle='--', alpha=0.6)

plt.show()



### Let's do some statistics on the occurences of nitrogenous bases


# I am going to concatenate all of the sequences together using this format

# I will ignore the ends for now and treat everything as a long continous chain to run some statisics.

sequence_train_concatenated = sequence_train_raw[0] + sequence_train_raw[1]
sequence_train_concatenated


# Here is the concatenation of all of the training set

sequence_train_concat = list()
for i in sequence_train_raw:
    sequence_train_concat += i


# Check what are the unique elements in the concatenation

set(sequence_train_concat)


### I observe that there are the four nitrogenous bases (adenine, guanine, uracil and cytosine)

### However, there are also '-' or 'X' which occur and I will intentionally ignore for now


sequence_train_concat_df = pd.DataFrame(sequence_train_concat)
sequence_train_concat_df.describe()


sequence_train_concat_df.head()


### Let's now do some statistics on the four nitrogenous bases (adenine, guanine, uracil and cytosine)


from collections import Counter

# Example DataFrame
df = sequence_train_concat_df

# Function to count letters in a text
def count_letters(text):
    return Counter(c for c in text if c.isalpha())  # only letters

# Apply the function to the whole column
letter_counts = df.apply(count_letters)

# Sum up all the counters
total_counts = sum(letter_counts, Counter())

# Convert to a DataFrame
letter_df = pd.DataFrame.from_dict(total_counts, orient='index', columns=['count'])

# Pivot-style: Sort by letter
letter_df_1L = letter_df.sort_values(by='count', ascending=False)

print(letter_df_1L)



### What is the probability for each of the nitrogenous base (adenine, guanine, uracil and cytosine)


prob_letter_1L = 100 * letter_df/(sum(letter_df_1L["count"]))
prob_letter_1L = prob_letter_1L.sort_values(by='count', ascending=False)
prob_letter_1L


## So Guanine is the most frequent and Uracil is the least frequent.


### Let's have a histogram for this

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Example DataFrame (with fewer letters for demonstration)
df = sequence_train_concat_df

# Function to count letters
def count_letters(text):
    return Counter(c for c in text if c.isalpha())

# Count letters
letter_counts = df.apply(count_letters)
total_counts = sum(letter_counts, Counter())
letter_df = pd.DataFrame.from_dict(total_counts, orient='index', columns=['count'])
letter_df = letter_df.sort_values(by='count', ascending=False)

# Define 4 distinct colors manually
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

# Assign colors cyclically
bar_colors = [colors[i % len(colors)] for i in range(len(letter_df))]

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(letter_df.index, letter_df['count'], color=bar_colors, edgecolor='black')

# Styling
ax.set_xlabel('Nitrogenous Base', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('1-Letter Frequency Histogram', fontsize=18)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top
for bar in bars:
    height = bar.get_height()


plt.tight_layout()
plt.show()



### Let's go up in complexity and try to do some statistics on 2 consecutive nitrogenous bases or 2-letters


### Let's first contatenate the entire dataset to build a 2-Letter dataset

sequence_train_concat_2 = [sequence_train_concat[i] + sequence_train_concat[i+1] for i in range(len(sequence_train_concat) - 1)]
sequence_train_concat_2[:10]


### Let's run some statistics on the 2 consecutive nitrogenous bases or 2-letters dataset

from collections import Counter

# Example DataFrame
df = pd.DataFrame(sequence_train_concat_2)

# Function to count letters in a text
def count_letters(text):
    return Counter(c for c in text if c.isalpha())  # only letters

# Apply the function to the whole column
letter_counts = df.apply(count_letters)

# Sum up all the counters
total_counts = sum(letter_counts, Counter())

# Convert to a DataFrame
letter_df = pd.DataFrame.from_dict(total_counts, orient='index', columns=['count'])

# Pivot-style: Sort by letter
letter_df = letter_df.sort_index()

letter_df = letter_df.sort_values(by='count', ascending=False)

# Pivot-style: Sort by letter
letter_df_L2 = letter_df.sort_values(by='count', ascending=False)

print(letter_df_L2)



## Here are the probabilities for the 2 consecutive nitrogenous bases or 2-letters dataset

prob_letter_df_L2 = 100 * letter_df/(sum(letter_df_L2["count"]))
prob_letter_df_L2


## So the most common pattern is to have 2 consecutive guanines.

## The next most common pattern is to have a guanine next to a cytosine. 


### Let's generate a Histogram for 2 consecutive nitrogenous bases or 2-letters dataset


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Example DataFrame (with fewer letters for demonstration)
df = pd.DataFrame(sequence_train_concat_2)

# Function to count letters
def count_letters(text):
    return Counter(c for c in text if c.isalpha())

# Count letters
letter_counts = df.apply(count_letters)
total_counts = sum(letter_counts, Counter())
letter_df = pd.DataFrame.from_dict(total_counts, orient='index', columns=['count'])
letter_df = letter_df.sort_index()

# Define 4 distinct colors manually
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

# Assign colors cyclically
bar_colors = [colors[i % len(colors)] for i in range(len(letter_df))]

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(letter_df.index, letter_df['count'], color=bar_colors, edgecolor='black')

# Styling
ax.set_xlabel('2 Consecutive Nitrogenous Bases', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('2 Letter Frequency Histogram', fontsize=18)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top
for bar in bars:
    height = bar.get_height()


plt.tight_layout()
plt.show()



### Let's go up in complexity and try to do some statistics on 3 consecutive nitrogenous bases or 3-letters


## Let's build the 3-letter dataset using the same approach

sequence_train_concat_3 = [sequence_train_concat[i] + sequence_train_concat[i+1] + sequence_train_concat[i+2] for i in range(len(sequence_train_concat) - 2)]
sequence_train_concat_3[:10]


### Let's do the counting

from collections import Counter

# Example DataFrame
df = pd.DataFrame(sequence_train_concat_3)

# Function to count letters in a text
def count_letters(text):
    return Counter(c for c in text if c.isalpha())  # only letters

# Apply the function to the whole column
letter_counts = df.apply(count_letters)

# Sum up all the counters
total_counts = sum(letter_counts, Counter())

# Convert to a DataFrame
letter_df_3L = pd.DataFrame.from_dict(total_counts, orient='index', columns=['count'])

# Pivot-style: Sort by letter
letter_df_3L = letter_df_3L.sort_values(by='count', ascending=False)

pd.set_option('display.max_rows', None) 

print(letter_df_3L)



### Let's calculate the probabilities

prob_letter_df_L3 = 100 * letter_df_3L/(sum(letter_df_3L["count"]))

prob_letter_df_L3 = prob_letter_df_L3.sort_values(by='count', ascending=False)

prob_letter_df_L3


## So the most common pattern is to have 3 consecutive guanines.

## The next most common pattern is to have a cytosine next to 2 consecutive guanines.

## The next most common patter is to have two consecutive cytosine next to one guanine.


### Let's generate a histogram for the 3 consecutive nitrogenous bases or 3-letters

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Example DataFrame (with fewer letters for demonstration)
df = pd.DataFrame(sequence_train_concat_3)

# Function to count letters
def count_letters(text):
    return Counter(c for c in text if c.isalpha())

# Count letters
letter_counts = df.apply(count_letters)
total_counts = sum(letter_counts, Counter())
letter_df = pd.DataFrame.from_dict(total_counts, orient='index', columns=['count'])
letter_df = letter_df.sort_index()

# Define 4 distinct colors manually
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

# Assign colors cyclically
bar_colors = [colors[i % len(colors)] for i in range(len(letter_df))]

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))

bars = ax.bar(letter_df.index, letter_df['count'], color=bar_colors, edgecolor='black')

# Styling
ax.set_xlabel('3 consecutive nitrogenous bases', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('3_Letter Frequency Histogram', fontsize=18)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top
for bar in bars:
    height = bar.get_height()

plt.xticks(rotation=45)


plt.tight_layout()
plt.show()



### Let's build the necessary dataset for this

L2_Letters = [x[0] + x[1] for x in sequence_train_concat_3]

L_first_Letter = [x[0] for x in sequence_train_concat_3]

L_last_Letter = [x[1] for x in sequence_train_concat_3]

L_3_label = [x[2] for x in sequence_train_concat_3]


df_2_Letters = pd.DataFrame(L2_Letters)
df_2_Letters.columns = ["feature_1"]

df_L_first_Letter = pd.DataFrame(L_first_Letter)
df_L_first_Letter.columns = ["feature_2"]

df_L_last_Letter = pd.DataFrame(L_last_Letter)
df_L_last_Letter.columns = ["feature_3"]

df_3_label = pd.DataFrame(L_3_label)
df_3_label.columns = ["label"]

df = pd.concat([df_2_Letters, df_L_first_Letter, df_L_last_Letter,  df_3_label], axis = 1)
df.describe()


### Now that we have the right dataset

### Let's build a simple machine learning model using Naive Bayes


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Example dataset with categorical features

# Separate features and target
X = df[['feature_1', 'feature_2', 'feature_3']]  # input features
y = df['label']                   # target label

# Encode the categorical features
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train Categorical Naive Bayes model
model = CategoricalNB()
model.fit(X_train, y_train)

# Optional: evaluate model
y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))



### The model accuracy is low. 


### Let's look at the confusion matrix

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay


_ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)




### Let's see how this model compares against always predicting 'G", the most common one.

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data
X = df[['feature_1', 'feature_2', 'feature_3']]  # input features
y = df['label']                   # target label

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create a baseline model
baseline_model = DummyClassifier(strategy="most_frequent")

# Train the model
baseline_model.fit(X_train, y_train)

# Predict
y_pred = baseline_model.predict(X_test)

# Evaluate
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))




### The Baseline model with the Guanine prediction is very similar to the Naive Bayes model


### Let's look at the confusion matrix

### Let's look at the confusion matrix

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay


_ = ConfusionMatrixDisplay.from_estimator(baseline_model, X_test, y_test)




### Let's try the same approach with xgboost


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Example dataset with categorical features and categorical target

# Separate features and target

X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['label']

# Encode categorical features
feature_encoder = OrdinalEncoder()
X_encoded = feature_encoder.fit_transform(X)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Optional: evaluate model
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_test_decoded = label_encoder.inverse_transform(y_test)

print("Test accuracy:", accuracy_score(y_test_decoded, y_pred))




### The Baseline model with the Guanine prediction is very similar to the XGBoost


### Let's look at the confusion matrix

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay


_ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)




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

