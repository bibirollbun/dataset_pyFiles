# pip install catboost


import numpy as np
import pandas as pd 
import math

from collections import Counter

from catboost import CatBoostClassifier, Pool


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score

# plotting

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# remove warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# add original dataset
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


# join original dataset 
train = pd.concat([train, original], axis=0, ignore_index=True)


train.shape


train.info()


train.describe().T


train.head()


# define numerical and categorical features

features_numerical = [
    'Temparature', 
    'Humidity', 
    'Moisture', 
    'Nitrogen', 
    'Potassium', 
    'Phosphorous'
]

features_categorical = [
    'Soil Type',
    'Crop Type'
]



train_counts_df = train['Fertilizer Name'].value_counts().reset_index()

train_counts_df.columns = ['Fertilizer Name', 'Count']


train_counts_df


# Plot Fertilizer Name 
plt.figure(figsize=(12, 6))
sns.countplot(data = train, 
              x = 'Fertilizer Name', 
              order = train['Fertilizer Name'].value_counts().index,
              palette='Set2'
             )

plt.title('Fertilizer Count (ordered)')

plt.ylabel('Count')
plt.xlabel('Fertilizer Name')

# plt.tight_layout()
plt.show()


# Select color palette
palette_name = 'rocket'

    

for feature in features_categorical:
    
    # value counts for the current feature
    counts = train[feature].value_counts()

    # color palette extended to length of 'counts'
    colors = sns.color_palette(palette_name, n_colors=len(counts))

    # Create a new figure for each plot
    plt.figure(figsize=(10, 6))

    # Plot bar chart
    plt.bar(counts.index, counts.values, color = colors)

    # Add titles and labels for clarity
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right") 
    plt.tight_layout() 

    plt.show()

    # Print unique and missing values for the current feature
    
    print(f"Number of Unique {feature}: {train[feature].nunique()}")


# define plots
# 3 plots in a row
num_cols = 3
num_rows = math.ceil(len(features_numerical) / num_cols)


fig, axes = plt.subplots(num_rows, 
                         num_cols, 
                         figsize=(6*num_cols, 4*num_rows))
axes = axes.flatten() 


colors = sns.color_palette("husl", 
                           n_colors=num_rows * num_cols)


# One violin plot per subplot
for i, col in enumerate(features_numerical[1:]):
    sns.violinplot(data=train, 
                   x=col, 
                   ax=axes[i], 
                   color = colors[i]
                  )
    axes[i].set_title(f"Distribution: {col}")
    axes[i].tick_params(axis='x', rotation=0)

# remove p
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# define label encoder
le = LabelEncoder()


# Encode target


train['Fertilizer Encoded'] = le.fit_transform(train['Fertilizer Name'])


train.head()


# Create X and Y columns



features = [col for col in train.columns if col not in ['Fertilizer Name', 'Fertilizer Encoded']]
X = train[features]
Y = train['Fertilizer Encoded']


X.head()


# define custom metrics - MAP @ K (where k = 3)

def map_at_k(y_true, y_prob, k=3):

    top_k_indices = np.argsort(y_prob, axis=1)[:, -k:][:, ::-1]
    
    map_scores = []
    for i, true_label in enumerate(y_true):

        relevance = [1 if pred == true_label else 0 for pred in top_k_indices[i]]
        
        # Calculate AP for this sample
        if sum(relevance) == 0:
            map_scores.append(0.0)
        else:
            precision_at_j = []
            for j in range(len(relevance)):
                if relevance[j] == 1:
                    precision_at_j.append(sum(relevance[:j+1]) / (j+1))
            map_scores.append(np.mean(precision_at_j) if precision_at_j else 0.0)
    
    return np.mean(map_scores)



class MAP3Evaluator:
    def get_final_error(self, error, weight):
        return error
    
    def is_max_optimal(self):
        return True
    
    def evaluate(self, approxes, target, weight):
        # Convert CatBoost outputs to probabilities
        y_prob = np.array(approxes).T
        y_prob = np.exp(y_prob) / np.sum(np.exp(y_prob), axis=1, keepdims=True)
        
        y_true = np.array(target, dtype=int)
        score = map_at_k(y_true, y_prob, k=3)
        return score, 0




# Define CatBoost Model


model = CatBoostClassifier(
    iterations = 1500,
    learning_rate = 0.02,
    depth = 6,
    objective = 'MultiClass', 
    # eval_metric = MAP3Evaluator(),
    eval_metric = 'Accuracy',
    random_seed = 42,
    early_stopping_rounds = 50,
    cat_features = features_categorical,
    verbose = 250
)


def evaluate_map3(model, X_test, y_test):
    # Get prediction probabilities
    y_prob = model.predict_proba(X_test)
    
    # Calculate MAP@3
    map3_score = map_at_k(y_test, y_prob, k=3)
    

    
    print(f"MAP@3 Score: {map3_score:.4f}")

    
    return map3_score


# fit CatBoost model

model_catboost = model.fit(X, Y)


# Evaluate with MAP@3
map3_score = evaluate_map3(model, X, Y)


map3_score


# Prepare test set in the same way


# X_test = pd.get_dummies(test[features_numerical + features_categorical])

X_test = test.reindex(columns = X.columns, fill_value=0)  # match columns


prediction = model_catboost.predict_proba(X_test)


prediction


# function argsort finds order of probabilities.  As it goes from smallest to highest, we need to take the last three 
# (in reverse order)

top_3 = []
for line in prediction:

    top_3.append(np.argsort(line)[-1:-4:-1])  # taking last three values in reverse order


# Convert back to fertilizer names
preds = [' '.join(le.inverse_transform(row)) for row in top_3]


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': preds
})
submission.to_csv('submission.csv', index=False)

