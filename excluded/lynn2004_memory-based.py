# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import datetime as dt

# Data Viz 
import seaborn as sns
import matplotlib.pyplot as plt


# Similarity calculation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# settings
pd.options.display.max_rows = 100
pd.options.display.max_columns = None

# Math
import math

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# import the tables
train = pd.read_csv(filepath_or_buffer='/kaggle/input/santander-product-recommendation/train_ver2.csv.zip')
#test = pd.read_csv(filepath_or_buffer='/kaggle/input/santander-product-recommendation/test_ver2.csv.zip')


## Data Profiling
train.info()
train.describe(include = 'all').T


# Check missing values
train.isnull().sum()/train.shape[0] * 100


## Inspect the data sample
train.sample(5)


# 1) Feature Name Transformation
col_names = {"ncodpers":"cust_id", "ind_empleado":"emp_index","pais_residencia":"residence",
            "sexo":"sex","fecha_alta":"first_date","ind_nuevo":"new_cust","antiguedad":"seniority",
            "indrel":"is_primary","ult_fec_cli_1t":"last_primary_date","indrel_1mes":"cust_type",
            "tiprel_1mes":"cust_rel_type","indresi":"residence_index","indext":"foreigner_index",
            "conyuemp":"spouse_index","canal_entrada":"channel","cod_prov":"province","nomprov":"province_name",
            "ind_actividad_cliente":"active_index","renta":"income","segmento":"segment"}

train.rename(col_names, axis = 1, inplace = True)



# 2) Data Type Conversion
# Convert the features into their intuitive types
train.age = pd.to_numeric(train.age, errors='coerce')
train.income = pd.to_numeric(train.income, errors='coerce')
train.seniority = pd.to_numeric(train.seniority, errors='coerce')
train.first_date = pd.to_datetime(train.first_date, errors = 'coerce')
train['fecha_dato'] = pd.to_datetime(train['fecha_dato'])




# Drop the last primary date and spouse index fields given over 99% missing values
train.drop(['last_primary_date','spouse_index'], axis = 1, inplace = True)



train.isna().mean()



# ğŸ“Œ Version 1: No EDA, Delete All Null Rows Except Income, Impute Income by Median (Grouped by Residence)

# âœ… Step 1: Copy train dataset into v1
v1 = train.copy()

# âœ… Step 2: Delete all rows having any null values, except 'income'
v1 = v1.dropna(subset=[col for col in v1.columns if col != 'income'])

# âœ… Step 3: Impute 'income' using median, grouped by 'residence'
v1['income'] = v1.groupby('province')['income'].transform(lambda x: x.fillna(x.median()))

# ğŸ”� Check result
print(v1.isna().sum())  # Should show no NaNs



v1.head()



def split_dataset(dataset):
    # Step 1: Filter data to include only January - May 2016
    cf_v1 = dataset[dataset['fecha_dato'].dt.year == 2016]

    # Step 2: Split into train (before 28/04/2016) and validation (only 28/04/2016)
    train_set = cf_v1[cf_v1['fecha_dato'] < '2016-04-28']
    eval_set = cf_v1[cf_v1['fecha_dato'] == '2016-04-28']

    # Return the train and validation sets
    return train_set, eval_set



v1_train, v1_eval =split_dataset(v1)
v1_train.shape
v1_eval.head()




def user_item_maker(train_set):
    """
    Creates a user-item matrix from the training set.

    Parameters:
    - train_set: The training dataset containing user interactions with products.

    Returns:
    - A DataFrame representing the user-item matrix.
    """
    # Select relevant columns for the user-item matrix
    train_set = train_set.iloc[:, [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]]

    # Create the user-item matrix
    user_item_matrix = train_set.groupby("cust_id").sum()
    user_item_matrix = user_item_matrix.fillna(0)
    user_item_matrix.index = user_item_matrix.index.astype('string')

    return user_item_matrix



user_item=user_item_maker(v1_train)
user_item.head()




def calculate_user_similarity(user_item_matrix, top_n_users=10000):
    """
    Calculates the user similarity matrix using cosine similarity.

    Parameters:
    - user_item_matrix: The user-item matrix.
    - top_n_users: The number of top users to consider for similarity calculation.

    Returns:
    - A DataFrame representing the user similarity matrix.
    """
    # Select the top N users for similarity calculation
    df_cf_user = user_item_matrix[:top_n_users].copy()

    # Calculate cosine similarity for users
    similarity_matrix = pd.DataFrame(
        cosine_similarity(df_cf_user),
        index=df_cf_user.index.astype('string'),
        columns=df_cf_user.index.astype('string')
    )

    return similarity_matrix


user_sim=calculate_user_similarity(user_item, top_n_users=10000)
user_sim.head()


def user_based_recommender(cust_id, sim_rate, top_n, similarity_matrix, user_item_matrix):
    """
    User-based recommendation model.

    Parameters:
    - cust_id: The customer ID for whom to recommend products.
    - sim_rate: Minimum similarity rate for other customers to be included.
    - top_n: The number of top products to recommend.
    - similarity_matrix: The cosine similarity matrix of users.
    - user_item_matrix: The user-item matrix containing product interactions.

    Returns:
    - A Series of top-N recommended products with predicted purchasing volumes.
    """
    cust_id = str(cust_id)

    # Find similar users with a similarity score above the threshold
    sim_cust = similarity_matrix.loc[
        (similarity_matrix[cust_id] >= sim_rate) & (similarity_matrix[cust_id] < 1), cust_id
    ]

    # Get products interacted by the target user
    products = user_item_matrix.loc[cust_id, :]

    # Find unpurchased products (assuming 0 = not purchased)
    unpurchased_products = products[products == 0].index

    # Create a new matrix with similar users and unpurchased products
    new_matrix = user_item_matrix.loc[sim_cust.index, unpurchased_products].copy()

    # Add similarity scores to the matrix
    new_matrix['sim_score'] = sim_cust.loc[new_matrix.index]

    # Apply similarity weight to each product
    for product in unpurchased_products:
        new_matrix[product] *= new_matrix['sim_score']

    # Calculate the predicted purchasing amount based on weighted sum
    top_item = new_matrix.drop(columns=['sim_score']).sum() / new_matrix['sim_score'].sum()

    # Sort to get top-N recommendations
    top_item = top_item.sort_values(ascending=False)[:top_n]

    # Filter out products with 0 predicted interest
    top_item = top_item[top_item > 0]

    # Show message if no recommendations are found
    if top_item.empty:
        print("There is no recommendation for the customer")

    return top_item

# Example usage:
# Assuming `similarity_matrix` and `user_item_matrix` are already defined
# recommendations = user_based_recommender(cust_id, sim_rate, top_n, similarity_matrix, user_item_matrix)



user_based_recommender(cust_id = 15929, sim_rate = 0.70, top_n = 10, similarity_matrix=user_sim, user_item_matrix= user_item)


# Testing Case 2: cust_id "16117", similarity score min 0.80, top 5 products
user_based_recommender(cust_id = 16117, sim_rate = 0.70, top_n = 10, similarity_matrix=user_sim, user_item_matrix= user_item)


# Testing Case 3: cust_id "16117", similarity score min 0.80, top 5 products



def calculate_item_similarity(user_item_matrix, top_n_users=10000):
    """
# 1. Based on the User-Item Matrix above, use Cosine Similarity to calculate the item similarities and build the similarity matrix
# Given the memory limitation (RAM), only using the top 10,000 customers' into to build the model. 
df_cf_item = df_cf.copy()
df_cf_item = df_cf_item[:10000].T

df_cf_item = pd.DataFrame(cosine_similarity(df_cf_item), index = df_cf_item.index.astype('string'), columns = df_cf_item.index.astype('string'))
df_cf_item.shape
    """
    # Select the top N users and transpose the matrix for item similarity calculation    df_cf_item = user_item_matrix.copy()
    df_cf_item = user_item_matrix[:top_n_users].T
    # Calculate cosine similarity for items
    similarity_matrix = pd.DataFrame(
        cosine_similarity(df_cf_item),
        index=df_cf_item.index.astype('string'),
        columns=df_cf_item.index.astype('string')
    )

    return similarity_matrix


item_sim = calculate_item_similarity(user_item, top_n_users=10000)
item_sim.shape


def item_based_recommender(cust_id, sim_rate, top_n, similarity_matrix, user_item_matrix):
    """
    Item-based recommendation model.

    Parameters:
    - cust_id: The customer ID for whom to recommend products.
    - sim_rate: Minimum similarity rate for other items to be included.
    - top_n: The number of top products to recommend.
    - item_similarity_matrix: The cosine similarity matrix of items.
    - user_item_matrix: The user-item matrix containing product interactions.

    Returns:
    - A Series of top-N recommended products with predicted purchasing volumes.
    """
    cust_id = str(cust_id)

    # Get the products the target user has interacted with
    user_products = user_item_matrix.loc[cust_id, :]

    # Find purchased products (assuming non-zero values indicate purchased products)
    purchased_products = user_products[user_products > 0].index

    # Check if the customer has purchased any product
    if purchased_products.empty:
        return pd.Series(dtype=float)  # Return an empty Series

    # Initialize a dictionary to store the predicted purchasing volumes
    predicted_volumes = {}

    # Loop through each purchased product to find similar items
    for product in purchased_products:
        # Get similar items with a similarity score above the threshold
        sim_items = similarity_matrix.loc[
            (similarity_matrix[product] >= sim_rate) & (similarity_matrix[product] < 1), product
        ]

        # Calculate the predicted purchasing volume for each similar item
        for sim_item, sim_score in sim_items.items():
            if sim_item not in purchased_products:  # Exclude already purchased items
                if sim_item in predicted_volumes:
                    predicted_volumes[sim_item] += sim_score * user_products[product]
                else:
                    predicted_volumes[sim_item] = sim_score * user_products[product]

    # Convert the predicted volumes to a Series and sort to get top-N recommendations
    top_item = pd.Series(predicted_volumes).sort_values(ascending=False)[:top_n]

    # Filter out products with 0 predicted interest
    top_item = top_item[top_item > 0]

    # Show message if no recommendations are found
    if top_item.empty:
        print("There is no recommendation for the customer")

    return top_item

# Example usage:
# Assuming `item_similarity_matrix` and `user_item_matrix` are already defined
# recommendations = item_based_recommender(cust_id, sim_rate, top_n, item_similarity_matrix, user_item_matrix)



user_item['ind_cco_fin_ult1']




item_based_recommender(cust_id = "16117", sim_rate = 0.0, top_n = 10, similarity_matrix=item_sim, user_item_matrix= user_item)












import pandas as pd

def generate_predictions(recommender_function, user_item_matrix, similarity_matrix, sim_rate=0.7, top_n=1, num_users=1000):
    """
    Generates predictions using a specified recommender function.

    Parameters:
    - recommender_function: The recommender function to use for predictions.
    - user_item_matrix: The user-item matrix containing user interactions with products.
    - similarity_matrix: The similarity matrix (item-based or user-based) used by the recommender function.
    - sim_rate: Minimum similarity rate for recommendations.
    - top_n: Number of top recommendations to generate.
    - num_users: Number of users to generate predictions for.

    Returns:
    - A DataFrame with user IDs and their predicted items.
    """
    # Initialize a list to store predictions
    predictions = []

    # Loop through the specified number of users in the user-item matrix
    for user in user_item_matrix.index[:num_users]:
        # Generate prediction using the specified recommender function
        prediction = recommender_function(
            cust_id=user,
            sim_rate=sim_rate,
            top_n=top_n,
            similarity_matrix=similarity_matrix,
            user_item_matrix=user_item_matrix
        )
        predictions.append((user, prediction))  # Store (user_id, predicted_item)

    # Convert to DataFrame for better analysis
    df_predictions = pd.DataFrame(predictions, columns=["user_id", "predicted_item"])

    return df_predictions



generate_predictions(item_based_recommender, user_item, sim_rate=0, top_n=1, num_users=1000,similarity_matrix=item_sim)



def calculate_map(merged_df):
    total_ap = 0
    num_users = len(merged_df)

    for _, row in merged_df.iterrows():
        actual_purchases = row['products_purchased']
        predicted_item = row['predicted_item']
        precision_at_1 = 1 if predicted_item in actual_purchases else 0
        total_ap += precision_at_1

    return total_ap / num_users if num_users > 0 else 0





df_predictions=generate_predictions(item_based_recommender, user_item, sim_rate=0, top_n=1, num_users=1000,similarity_matrix=item_sim)

# Melt the DataFrame to long format
product_columns = [col for col in v1_eval.columns if col.startswith('ind_')]
df_melted = v1_eval.melt(id_vars=['cust_id'], value_vars=product_columns, var_name='product', value_name='purchased')

# Filter rows where the product was purchased
df_purchased = df_melted[df_melted['purchased'] == 1]

# Group by cust_id and aggregate product names into a list
df_actual = df_purchased.groupby('cust_id')['product'].apply(list).reset_index()

# Rename columns to match the desired format
df_actual.columns = ['cust_id', 'products_purchased']

# Display the resulting DataFrame
print(df_predictions.head())






# Make sure that 2 tables are similar before merging
df_predictions['predicted_item'] = df_predictions['predicted_item'].astype(str)
df_predictions['predicted_item'] = df_predictions['predicted_item'].str.split().str[0]
df_predictions.head()
df_predictions['user_id'] = df_predictions['user_id'].astype(int)
df_actual['cust_id'] = df_actual['cust_id'].astype(int)
merged_df = df_predictions.merge(df_actual, left_on='user_id', right_on='cust_id', how='inner')





map_score = calculate_map(merged_df)
print(f"MAP Score: {map_score:.4f}")

