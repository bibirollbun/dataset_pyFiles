!pip install Unidecode


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import random
import re
from collections import Counter
from unidecode import unidecode

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# SVC + OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# Word2Vec
from gensim.models import Word2Vec

# FastText
from gensim.models import FastText

# Metrics
from sklearn.metrics import accuracy_score

# Plotting
from sklearn.metrics import classification_report, confusion_matrix

from nltk.stem import WordNetLemmatizer

file_path = '/kaggle/input/whats-cooking-kernels-only/'


MANUAL_SEED = 48


train = pd.read_json(file_path + 'train.json')
test = pd.read_json(file_path + 'test.json')


#We add a new column with the counting to help with the analysis
train['num_ingredients'] = train['ingredients'].apply(len).tolist()


recipe_df = train.copy()
test_df = test.copy()


recipe_df.info()


recipe_df.head()


recipe_df.cuisine.unique()


import random

def random_colors(number_of_colors):
    colors = []
    for i in range(number_of_colors):
        colors.append('#{}'.format(''.join(random.choice('0123456789A') for j in range(6))))
    return colors

random_colors(3)


recipe_df['cuisine'].value_counts().plot.bar(
    color=random_colors(recipe_df['cuisine'].unique().size),
    figsize=(16,6)
)


#We can get the ingredients without duplication like this:
unique_ingredients = list(set(ingredient for ingredients in recipe_df['ingredients'] for ingredient in ingredients))

#We can get the ingredients with duplication like this:
raw_ingredients = [ingredient for ingredients in recipe_df['ingredients'] for ingredient in ingredients]


#To get the max quantity of ingredients used in a recipe
print('Max quanity of ingredients:', recipe_df['ingredients'].str.len().max())
print(max(recipe_df['ingredients'].apply(len).tolist()))

#To get the min quantity of ingredients used in a recipe
print('Min quanity of ingredients:', recipe_df['ingredients'].str.len().min())
print(min(recipe_df['ingredients'].apply(len).tolist()))

#We create a new feature with the number of ingredients used on each recipe
recipe_df['num_ingredients'] = recipe_df['ingredients'].apply(len).tolist()


plt.figure(figsize=(16,6))
sns.displot(recipe_df['num_ingredients'], kde=False, bins=60)


plt.figure(figsize=(16,6))
sns.countplot(x='num_ingredients', data=recipe_df)


long_recipe = recipe_df.loc[recipe_df['num_ingredients']>30]
print('Number of recipes with too many ingredients:',len(long_recipe))
print('-------------')
print('Number of long recipes per cuisine:')
long_recipe.cuisine.value_counts()


short_recipe = recipe_df.loc[recipe_df['num_ingredients']<=2]
print('Number of recipes with low ingredients:',len(short_recipe))
print('----------')
print('Number of short recipes per cuisine:')
short_recipe['cuisine'].value_counts()


[ingredient for ingredient in raw_ingredients if len(ingredient) <= 2]




plt.figure(figsize=(20,8))
sns.boxplot(x='cuisine', y='num_ingredients', data=recipe_df)


import re

special_chr_ingredients = [ingredient for ingredient in unique_ingredients if len(re.findall('[^a-zA-Z0-9- áéíóú]',ingredient)) != 0]
special_chr_ingredients[:5]


from collections import Counter


top_ingredients = Counter(raw_ingredients)
top_ingredients.most_common(20)


top_ingredients_df = pd.DataFrame(top_ingredients.most_common(20),columns=['ingredient','ocurrences'])
top_ingredients_df

plt.figure(figsize=(6,5))
sns.barplot(x='ocurrences',y='ingredient',data=top_ingredients_df)


cuisine_ingredients = recipe_df.groupby('cuisine').ingredients.agg(
    ingredients = lambda x: list(set(x.sum())) #By using aggregated function we can sum not duplicated values
).reset_index() #by default the index is the group by column, with this a new index is created

#we can count not duplicated values
cuisine_ingredients['ing_count'] = cuisine_ingredients.ingredients.apply(len)


cuisine_ingredients = cuisine_ingredients.sort_values(by='ing_count',ascending=False)
cuisine_ingredients.head(5)


plt.figure(figsize=(20,6))
sns.barplot(x='cuisine', y='ing_count', data=cuisine_ingredients)


def top_exclusive_ing_x_cuisine(df,cuisine,n):
    
    #Getting all ingredientes of the requested cuisine
    cuisine_all_ing_ls = []

    for ingredients in df.loc[df['cuisine']==cuisine]['ingredients']:
        cuisine_all_ing_ls += ingredients

    
    #Getting a DataFrame with the ingredients and the ocurrences
    cuisine_all_ing_counter = Counter(cuisine_all_ing_ls)

    cuisine_all_ing = pd.DataFrame.from_dict(cuisine_all_ing_counter,orient='index').reset_index()
    cuisine_all_ing = cuisine_all_ing.rename(columns={'index':'ingredients',0:'ing_count'})

    
    #Getting all the ingredientes from other cuisines without duplicate
    other_cuisin_ing = []

    for ingredients in train.loc[train['cuisine']!='mexican']['ingredients']:
        other_cuisin_ing += ingredients

    other_cuisin_ing = list(set(other_cuisin_ing))

    
    #Deleting the ingredientes from other cuisines that are in the requested cuisine
    cuisine_all_ing = cuisine_all_ing.loc[~cuisine_all_ing['ingredients'].isin(other_cuisin_ing)]

    #Sorting to return the dataframe
    top_cuisine_ingredients = cuisine_all_ing.sort_values(by='ing_count',ascending=False).copy()
    
    return top_cuisine_ingredients[:n]


top_exclusive_ing_x_cuisine(train,'mexican',10)


print('initial number of records:',len(recipe_df))
recipe_df = recipe_df.loc[recipe_df['num_ingredients']>2]
recipe_df = recipe_df.loc[recipe_df['num_ingredients']<60]
print('final number of records:',len(recipe_df))


def adjust_ingredients(ingredients):

    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-',' ')

    words = []
    for word in ingredients_text.split():
        word = re.sub('[0-9]','',word)
        word = re.sub(r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b','',word)
        #review "and", "&"
        #review ","
        #review known product labels
        #review countries and locations
        
        if len(word) > 2:
            word = unidecode(word)
            words.append(word)
  
    return ' '.join(words)


recipe_df['ingredients_str'] = recipe_df['ingredients'].apply(adjust_ingredients)
recipe_df.head()


def data_split(data,target):
    train_set, valid_set, train_target, valid_target = train_test_split(
        data,
        target,
        test_size = 0.2, #Validation split
        stratify = target,
        random_state = MANUAL_SEED
    )

    return train_set, valid_set, train_target, valid_target 


def get_accuracy(output,y):
    accuracy = accuracy_score(y, output)
    return accuracy


def confusion_plot(target,prediction,target_classes):
    print(classification_report(target, prediction, target_names=target_classes))

    conf_matrix = confusion_matrix(target, prediction)
    plt.figure(figsize=(8,6))
    sns.heatmap(data=conf_matrix, xticklabels=target_classes, yticklabels=target_classes, cmap='Blues', annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Split Data --------------------------------------------------
train_df, valid_df, _, _ = data_split(recipe_df,recipe_df['cuisine'])
train_df = train_df.copy()

# Vectorizer --------------------------------------------------
vectorizer = TfidfVectorizer(sublinear_tf=True)
#vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['ingredients_str'].values)
X_train = X_train.toarray()

# Label Encoder ------------------------------------------------
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['cuisine'].values)
y_train = np.array(y_train,copy=True)

# Model --------------------------------------------------------
# Suppor Vector Classification (Binary Class)
# This is used for non-linear classification
#classifier = LinearSVC()
classifier = SVC(
    C=100, # penalty parameter
    kernel='rbf', # kernel type
    degree = 3, # default value
    gamma = 1, # Kernel coefficient
    coef0 = 1, # The default value is 0.0
    shrinking = True, # uses shrinking heuristics
    tol = 0.001, # stop criteria tolerance
    probability = False, # Probability estimates
    cache_size = 200, # 200 MB cache size
    class_weight = None, # all classes are treated equally
    verbose = False, # print the logs
    max_iter = -1, # -1 means no limit
    decision_function_shape = 'ovr',  # By default uses ovo (one vs one) but could explecity be assigned ovr (one vs rest
    random_state = None    
)

# OneVsRestClassifier (a.k.a. OvR or One-vs-All)
# Multi class Classifier: Is added to SVC to support multiclass behavior
model = OneVsRestClassifier(classifier , n_jobs=4)


model.fit(X_train,y_train)



X_valid = vectorizer.transform(valid_df['ingredients_str'])
y_valid = label_encoder.transform(valid_df['cuisine'])

y_valid_pred = model.predict(X_valid)


accuracy = get_accuracy(y_valid_pred,y_valid)

print('Validation Accuracy: {}'.format(accuracy))


confusion_plot(y_valid, y_valid_pred,target_classes = label_encoder.classes_)

