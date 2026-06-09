import os

# Define required files
required_files = {
    "sample_submission.csv": None,
    "articles.csv": None,
    "transactions_train.csv": None,
    "customers.csv": None
}

# Search for files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename in required_files:  # Check only required files
            required_files[filename] = os.path.join(dirname, filename)
        if all(required_files.values()):  # Stop early if all found
            break

# Print found files
for name, path in required_files.items():
    if path:
        print(path)
    else:
        print(f"Missing: {name}")



# Standard library imports
import sys
import warnings
import time
import os
import copy
import gc
import re
import random
import pickle
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint

# Third-party imports
import numpy as np
import cupy as cp
import pandas as pd
import cudf
from cuml.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import preprocessing

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    def __init__(self, random_state=2025, n_clusters=12):
        self.random_state = random_state
        self.n_clusters = n_clusters
        
        self.news_frequency_mapping = {
            np.nan: 0, 
            'None': 0, 
            'NONE': 0,
            'Monthly': 1, 
            'Regularly': 2
        }
        
        self.member_status_mapping = {
            np.nan: 0,
            'PRE-CREATE': 1,
            'ACTIVE': 2,
            'LEFT CLUB': -1
        }

    def preprocess_customer_data(self, customer_df, columns_to_drop=['postal_code']):
        """Preprocess customer data by handling missing values and encoding categorical variables."""
        # Convert to cuDF if input is pandas DataFrame
        if isinstance(customer_df, pd.DataFrame):
            processed_df = cudf.from_pandas(customer_df)
        else:
            processed_df = customer_df.copy()
            
        # Drop specified columns
        if any(col in processed_df.columns for col in columns_to_drop):
            processed_df = processed_df.drop(columns=[col for col in columns_to_drop if col in processed_df.columns])
        
        # Handle categorical and missing values
        if 'fashion_news_frequency' in processed_df.columns:
            processed_df['fashion_news_frequency'] = (
                processed_df['fashion_news_frequency']
                .astype('str')  # Convert to string to handle NaN values
                .replace('NONE', 'None')
                .map(self.news_frequency_mapping)
                .fillna(0)
            )
            
        if 'club_member_status' in processed_df.columns:
            processed_df['club_member_status'] = (
                processed_df['club_member_status']
                .astype('str')
                .map(self.member_status_mapping)
                .fillna(0)
            )
            
        if 'age' in processed_df.columns:
            processed_df['age'] = processed_df['age'].fillna(-1)
            
        # Handle binary indicators
        for col in ['FN', 'Active']:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(0)
        
        return processed_df

    def create_customer_segments(self, df, id_columns, feature_columns, normalization='StandardScaler', return_normalized=False):
        """Create customer segments using KMeans clustering."""
        # Ensure all feature columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Convert features to cupy array for sklearn
        X = df[feature_columns].to_pandas().values
        
        # Apply normalization if specified
        if normalization == 'StandardScaler':
            normalizer = preprocessing.StandardScaler()
            X = normalizer.fit_transform(X)
        elif normalization == 'minMax':
            normalizer = preprocessing.MinMaxScaler()
            X = normalizer.fit_transform(X)
        
        print(f'Normalization Method: {normalization}')
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(X)
        
        print(f'Clustering Distortion: {kmeans.inertia_:.2f}')
        

        # Add predictions to original dataframe
        cluster_assignments = cudf.Series(kmeans.labels_, name='cluster_id')
        result_df = df.copy()
        result_df['cluster_id'] = cluster_assignments
        
        if return_normalized:
            norm_features_df = cudf.DataFrame(X, columns=feature_columns)
            print("\n=== Normalized Features Summary ===")
            print(norm_features_df.describe())
            return result_df, norm_features_df
            
        return result_df



class PurchaseAnalyzer:
    def __init__(self, input_path, date_range=('2020-09-01', '2020-09-21')):
        self.input_path = Path(input_path)
        self.date_range = date_range
        
    def get_recent_transactions(self):
        """Load and filter recent transactions using cuDF."""
        transactions_path = self.input_path / 'transactions_train.csv'
        if not transactions_path.exists():
            raise FileNotFoundError(f"Transactions file not found at {transactions_path}")

        transactions_df = cudf.read_csv(
            transactions_path,
            usecols=['t_dat', 'customer_id', 'article_id'],
            dtype={
                'article_id': 'int32',
                't_dat': 'str',
                'customer_id': 'str'
            }
        )
        
        # Convert date and filter range
        transactions_df['t_dat'] = cudf.to_datetime(transactions_df['t_dat'])
        mask = (transactions_df['t_dat'] >= self.date_range[0]) & (transactions_df['t_dat'] <= self.date_range[1])
        return transactions_df[mask]
    
    def analyze_cluster_preferences(self, recent_transactions, customer_segments, top_n=100):
        """Analyze product preferences for each customer cluster."""
        # Ensure required columns exist
        required_cols = ['customer_id', 'cluster_id']
        if not all(col in customer_segments.columns for col in required_cols):
            raise ValueError(f"Missing required columns in customer_segments: {required_cols}")

        # Merge transactions with cluster assignments
        merged_df = recent_transactions.merge(
            customer_segments[['customer_id', 'cluster_id']], 
            on='customer_id', 
            how='inner'
        )
        
        # Calculate purchase counts by cluster and article
        purchase_counts = (
            merged_df.groupby(['cluster_id', 'article_id'])
            .size()
            .reset_index()
            .rename(columns={0: 'purchase_count'})
        )
        
        purchase_counts = purchase_counts.to_pandas()
        
        # Get top products for each cluster
        cluster_preferences = {}
        for cluster_id in purchase_counts['cluster_id'].unique():
            cluster_purchases = purchase_counts[purchase_counts['cluster_id'] == cluster_id]
            top_products = cluster_purchases.nlargest(top_n, 'purchase_count')['article_id'].tolist()
            cluster_preferences[cluster_id] = top_products
            
        # Calculate similarity matrix
        similarity_df = pd.DataFrame([cluster_preferences]).T.rename(columns={0: 'top_products'})
        for cluster_id in similarity_df.index:
            similarity_df[cluster_id] = [
                len(set(similarity_df.at[cluster_id, 'top_products']) & 
                    set(similarity_df.at[x, 'top_products'])) / top_n 
                for x in similarity_df.index
            ]
            
        return similarity_df.drop(columns='top_products')

    def generate_recommendations(self, customer_segments, recent_transactions, n_recommendations=12):
        """Generate product recommendations for each customer segment."""
        if len(recent_transactions) == 0:
            raise ValueError("No transactions found in the specified date range")

        last_date = recent_transactions['t_dat'].max()
        
        # Calculate days since last purchase for weighting
        recent_transactions = recent_transactions.copy()
        recent_transactions['days_since'] = (
            last_date - recent_transactions['t_dat']
        ).dt.days
        
        # Convert days_since to cupy array for GPU calculations
        days_since_array = cp.asarray(recent_transactions['days_since'].values.astype('float32'))
        
        # Weight calculation parameters
        a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
        
        # Calculate weights using cupy operations
        weights = (
            a / cp.sqrt(days_since_array) + 
            b * cp.exp(-c * days_since_array) - 
            d
        )
        
        # Convert weights back to cudf Series and clip negative values
        recent_transactions['weight'] = cudf.Series(
            cp.asnumpy(weights), 
            index=recent_transactions.index
        ).clip(lower=0)
        
        # Calculate weighted purchase counts
        weighted_counts = (
            recent_transactions
            .groupby(['customer_id', 'article_id'])
            ['weight']
            .sum()
            .reset_index()
        )
        
        # Rank products for each customer
        weighted_counts['rank'] = weighted_counts.groupby('customer_id')['weight'].rank(
            method='dense',
            ascending=False
        )
        
        # Filter top N recommendations
        recommendations = weighted_counts[weighted_counts['rank'] <= n_recommendations]
        
        # Clean up GPU memory
        del days_since_array, weights
        cp.get_default_memory_pool().free_all_blocks()
        
        return recommendations




class RecommendationEngine:
    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.articles_df = None
        self.load_article_data()
        
    def load_article_data(self):
        """Load and preprocess article data."""
        articles_path = self.input_path / 'articles.csv'
        if not articles_path.exists():
            raise FileNotFoundError(f"Articles file not found at {articles_path}")
            
        self.articles_df = pd.read_csv(articles_path)
        
    def get_user_recommendations(self, customer_id, segmented_customers, recent_transactions, 
                               n_recommendations=12, include_article_details=True):
        """
        Generate personalized recommendations for a specific user.
        
        Args:
            customer_id (str): The customer ID to generate recommendations for
            segmented_customers (cudf.DataFrame): DataFrame with customer segments
            recent_transactions (cudf.DataFrame): Recent transaction data
            n_recommendations (int): Number of recommendations to generate
            include_article_details (bool): Whether to include article details in results
        """
        # Get user's cluster
        user_cluster = segmented_customers[
            segmented_customers['customer_id'] == customer_id
        ]['cluster_id'].iloc[0]
        
        # Get cluster-based recommendations
        cluster_transactions = recent_transactions.merge(
            segmented_customers[segmented_customers['cluster_id'] == user_cluster][['customer_id']],
            on='customer_id',
            how='inner'
        )
        
        # Get user's recent purchases
        user_purchases = set(
            recent_transactions[
                recent_transactions['customer_id'] == customer_id
            ]['article_id'].to_pandas().tolist()
        )
        
        # Calculate article popularity within cluster
        article_scores = (
            cluster_transactions
            .groupby('article_id')
            .size()
            .reset_index(name='popularity_score')
        )
        
        # Convert to pandas for easier processing
        article_scores = article_scores.to_pandas()
        
        # Remove articles user has already purchased
        article_scores = article_scores[
            ~article_scores['article_id'].isin(user_purchases)
        ]
        
        # Sort by popularity and get top recommendations
        recommendations = article_scores.nlargest(n_recommendations, 'popularity_score')
        
        if include_article_details and self.articles_df is not None:
            recommendations = recommendations.merge(
                self.articles_df,
                on='article_id',
                how='left'
            )
            
        return recommendations


def main():
    INPUT_PATH = Path('../input/h-and-m-personalized-fashion-recommendations')
    
    # Initialize classes
    segmentation = CustomerSegmentation(random_state=2025, n_clusters=12)
    analyzer = PurchaseAnalyzer(input_path=INPUT_PATH)
    recommendation_engine = RecommendationEngine(input_path=INPUT_PATH)  # Add recommendation engine
    
    try:
        # Load and preprocess customer data
        customers_path = INPUT_PATH / 'customers.csv'
        if not customers_path.exists():
            raise FileNotFoundError(f"Customers file not found at {customers_path}")
            
        customers_df = cudf.read_csv(customers_path)
        processed_customers = segmentation.preprocess_customer_data(customers_df)
        
        # Create customer segments
        feature_cols = ['club_member_status', 'fashion_news_frequency', 'age', 'FN', 'Active']
        segmented_customers = segmentation.create_customer_segments(
            processed_customers,
            id_columns=['customer_id'],
            feature_columns=feature_cols
        )
        
        # Analyze recent transactions
        recent_transactions = analyzer.get_recent_transactions()
        
        # Generate recommendations
        recommendations = analyzer.generate_recommendations(
            segmented_customers,
            recent_transactions
        )
        
        # Analyze cluster similarities
        cluster_similarity = analyzer.analyze_cluster_preferences(
            recent_transactions,
            segmented_customers
        )
        
        # Visualize results
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_similarity, annot=True, cbar=False)
        plt.title('Cluster Purchase Pattern Similarity')
        plt.show()

         # Generate recommendations for a sample user
        sample_user_id = segmented_customers['customer_id'].iloc[0]
        print(f"\nGenerating recommendations for user: {sample_user_id}")
        
        user_recommendations = recommendation_engine.get_user_recommendations(
            sample_user_id,
            segmented_customers,
            recent_transactions
        )
        
        print("\nTop Recommended Articles:")
        print(user_recommendations[['article_id', 'popularity_score', 'product_type_name']].head())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

