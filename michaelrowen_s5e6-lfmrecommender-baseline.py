!pip install lightfm


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder


df_rec = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
additional_df_rec = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
test_df_rec = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


full_df_rec = pd.concat([df_rec.drop('id', axis=1), additional_df_rec], ignore_index=True)

print(f"Training data shape: {full_df_rec.shape}")
print(f"Test data shape: {test_df_rec.shape}")
print(f"Unique fertilizers: {full_df_rec['Fertilizer Name'].nunique()}")

fert_names = full_df_rec['Fertilizer Name'].unique()
fert_to_id = {name: i for i, name in enumerate(fert_names)}
id_to_fert = {i: name for i, name in enumerate(fert_names)}

print(f"Fertilizer types: {list(fert_names)}")


class LFMRecommender:
    
    def __init__(self, loss='warp', learning_rate=0.001, no_components=100):
        self.model = LightFM(
            loss=loss, 
            learning_rate=learning_rate, 
            no_components=no_components, 
            random_state=42
        )
        self.dataset = LightFMDataset()
        
        self._setup_feature_bins()
    
    def _setup_feature_bins(self):
        self.temp_bins = np.linspace(0, 50, 10)  
        self.humidity_bins = np.linspace(0, 100, 10)
        self.moisture_bins = np.linspace(0, 100, 10)
        self.npk_bins = np.linspace(0, 100, 10)
    
    def _bin_features_vectorized(self, df):
        df_copy = df.copy()
        
        df_copy['temp_bin'] = np.digitize(df_copy['Temparature'], self.temp_bins)
        df_copy['humidity_bin'] = np.digitize(df_copy['Humidity'], self.humidity_bins)
        df_copy['moisture_bin'] = np.digitize(df_copy['Moisture'], self.moisture_bins)
        df_copy['n_bin'] = np.digitize(df_copy['Nitrogen'], self.npk_bins)
        df_copy['p_bin'] = np.digitize(df_copy['Phosphorous'], self.npk_bins)
        df_copy['k_bin'] = np.digitize(df_copy['Potassium'], self.npk_bins)
        
        df_copy['user_id'] = (df_copy['temp_bin'].astype(str) + '_' + 
                             df_copy['humidity_bin'].astype(str) + '_' + 
                             df_copy['Soil Type'] + '_' + 
                             df_copy['Crop Type'])
        
        return df_copy
    
    def prepare_lightfm_data(self, df):
        df_processed = self._bin_features_vectorized(df)
        
        user_features = []
        unique_users = df_processed['user_id'].unique()
        
        for user_id in unique_users:
            user_data = df_processed[df_processed['user_id'] == user_id].iloc[0]
            features = [
                f"temp_{user_data['temp_bin']}", 
                f"hum_{user_data['humidity_bin']}", 
                f"soil_{user_data['Soil Type']}", 
                f"crop_{user_data['Crop Type']}"
            ]
            user_features.append((user_id, features))
        
        fert_stats = df_processed.groupby('Fertilizer Name')[['Nitrogen', 'Potassium', 'Phosphorous']].mean()
        item_features = []
        
        for fert_name, stats in fert_stats.iterrows():
            n_bin = np.digitize(stats['Nitrogen'], self.npk_bins)
            p_bin = np.digitize(stats['Phosphorous'], self.npk_bins)
            k_bin = np.digitize(stats['Potassium'], self.npk_bins)
            
            features = [f"fert_N_{n_bin}", f"fert_P_{p_bin}", f"fert_K_{k_bin}"]
            item_features.append((fert_name, features))
        
        interactions = [(row['user_id'], row['Fertilizer Name']) for _, row in df_processed.iterrows()]
        
        return interactions, user_features, item_features, df_processed
    
    def fit(self, df):
        """Optimized training"""
        print("Preparing data...")
        interactions, user_features, item_features, self.df_processed = self.prepare_lightfm_data(df)
        
        print("Building dataset...")
        self.dataset.fit(
            users=[interaction[0] for interaction in interactions],
            items=[interaction[1] for interaction in interactions],
            user_features=[feature for user_feat in user_features for feature in user_feat[1]],
            item_features=[feature for item_feat in item_features for feature in item_feat[1]]
        )
        
        user_features_dict = {user_id: features for user_id, features in user_features}
        item_features_dict = {item_id: features for item_id, features in item_features}
        
        self.user_features_matrix = self.dataset.build_user_features(
            [(user_id, features) for user_id, features in user_features_dict.items()]
        )
        
        self.item_features_matrix = self.dataset.build_item_features(
            [(item_id, features) for item_id, features in item_features_dict.items()]
        )
        
        interactions_matrix = self.dataset.build_interactions(interactions)
        
        print("Training model...")
        self.model.fit(
            interactions_matrix[0],
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            epochs=100,  
            num_threads=2,
            verbose=False
        )
        
        self.item_features_dict = item_features_dict
        return self
    
    def recommend(self, test_df, top_k=3):
        """Generate recommendations efficiently"""
        test_processed = self._bin_features_vectorized(test_df)
        recommendations = []
        all_items = list(self.item_features_dict.keys())
        
        for _, sample in test_processed.iterrows():
            user_id = f"{sample['temp_bin']}_{sample['humidity_bin']}_{sample['Soil Type']}_{sample['Crop Type']}"
            
            try:
                user_internal_id = self.dataset.mapping()[0][user_id]
                item_internal_ids = [self.dataset.mapping()[2][item] for item in all_items if item in self.dataset.mapping()[2]]
                
                scores = self.model.predict(
                    user_internal_id,
                    item_internal_ids,
                    user_features=self.user_features_matrix,
                    item_features=self.item_features_matrix
                )
                
                item_scores = list(zip([item for item in all_items if item in self.dataset.mapping()[2]], scores))
                item_scores.sort(key=lambda x: x[1], reverse=True)
                top_items = [item[0] for item in item_scores[:top_k]]
                
            except KeyError:
                top_items = list(self.item_features_dict.keys())[:top_k]
            
            recommendations.append(top_items)
        
        return recommendations

print("\nTraining LightFM Recommender...")
lightfm_recommender = LFMRecommender(loss='warp', no_components=100)
lightfm_recommender.fit(full_df_rec)

sample_test = test_df_rec.drop('id', axis=1).head(5)
lightfm_recommendations = lightfm_recommender.recommend(sample_test, top_k=3)

print("LightFM Recommendations:")
for i, rec in enumerate(lightfm_recommendations):
    print(f"Sample {i+1}: {rec}")


class SimplifiedEnsemble:
    
    def __init__(self, weights=None):
        if weights is None:
            weights = {'lightfm': 0.5, 'popularity': 0.5}
        self.weights = weights
        
    def fit(self, df):
        print("Training Ensemble...")
        
        print("Training LightFM...")
        self.lightfm = LFMRecommender(loss='warp', no_components=100)
        self.lightfm.fit(df)
        
        print("Computing popularity baseline...")
        self.popularity_scores = df['Fertilizer Name'].value_counts(normalize=True)
        
        return self
    
    def recommend(self, test_df, top_k=3):
        print(f"Generating recommendations for {len(test_df)} samples...")
        
        lightfm_recs = self.lightfm.recommend(test_df, top_k=6)
        
        ensemble_recommendations = []
        pop_items = list(self.popularity_scores.head(6).index)
        
        for i in range(len(test_df)):
            scores = {}
            
            for j, fert in enumerate(lightfm_recs[i]):
                scores[fert] = scores.get(fert, 0) + self.weights['lightfm'] * (6 - j)

            for j, fert in enumerate(pop_items):
                scores[fert] = scores.get(fert, 0) + self.weights['popularity'] * (6 - j)
            
            sorted_fertilizers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_fertilizers = [fert[0] for fert in sorted_fertilizers[:top_k]]
            
            ensemble_recommendations.append(top_fertilizers)
        
        return ensemble_recommendations


print("="*50)
print("RECOMMENDER ENSEMBLE")
print("="*50)

ensemble = SimplifiedEnsemble()
ensemble.fit(full_df_rec)

print("\nGenerating final recommendations for test set...")
final_recommendations = ensemble.recommend(test_df_rec.drop('id', axis=1), top_k=3)

# Prepare submission
test_ids = test_df_rec['id'].values
submission_recommendations = []

for recs in final_recommendations:
    submission_recommendations.append(' '.join(recs))

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_recommendations
})

print(f"Submission shape: {submission_df.shape}")
print("Sample recommendations:")
print(submission_df.head(10))

# Save submission
submission_df.to_csv('submission.csv', index=False)
print("\nsubmission saved!")

