from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import seaborn as sns

sns.set()


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col = 0)
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col=0)


train[['Fertilizer Name']].value_counts().plot(kind='bar')
plt.ylabel("Count")
plt.show()


train.describe()


# Approximate nutrient values based on Indian agriculture guidelines and FAO/ICAR recommendations (for SFI weights)
# (Note: These may not be fully relevant or optimal for the model, but including them could introduce useful variation.)

crop_weights = {
    'Wheat': [6, 3, 2],
    'Paddy': [2, 1, 1],
    'Maize': [6, 3, 2],
    'Barley': [6, 3, 2],
    'Cotton': [2, 1, 1],
    'Sugarcane': [2, 1, 1],
    'Millets': [6, 3, 2],
    'Pulses': [1, 2, 1],
    'Ground Nuts': [5, 8, 10],
    'Tobacco': [2, 1, 2],
    'Oil seeds': [2, 3, 2],
}


class FeaturesEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, weights_dict=None):
        self.weights = weights_dict or {}
        self.new_columns = ['N_ratio', 'P_ratio', 'K_ratio', 'SFI']
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Nothing to fit
        return self

    def transform(self, X):
        X = X.copy()
        new_values = []

        for index, row in X.iterrows():
            N = row.get('Nitrogen', 0)
            P = row.get('Phosphorous', 0)
            K = row.get('Potassium', 0)
            crop = row.get('Crop Type')

            weights = self.weights.get(crop, [1, 1, 1])
            SFI = (weights[0]*N + weights[1]*P + weights[2]*K) / sum(weights)

            vals = [x for x in [N, P, K] if x > 0]
            if len(vals) == 0:
                ratios = [0, 0, 0]
            else:
                s = min(vals)
                ratios = [round(N/s), round(P/s), round(K/s)]

            new_values.append(ratios + [SFI])

        new_df = pd.DataFrame(new_values, columns=self.new_columns, index=X.index)
        X = pd.concat([X, new_df], axis=1)

        # Add more derived features
        X['SMI'] = X['Humidity'] / X['Temparature']
        X['Evaporation'] = X['Temparature'] * (1 - X['Humidity'] / 100)
        X['N_P_ratio'] = X['Nitrogen'] / (X['Phosphorous'] + 1)
        X['P_K_ratio'] = X['Phosphorous'] / (X['Potassium'] + 1)
        X['Soil_stress_index'] = (X['Temparature'] * X['Humidity']) / (X['Moisture'] + 1)
        X['N_level'] = pd.cut(X['Nitrogen'], bins=[0, 10, 20, 30, 50], labels=[1, 2, 3, 4]).astype(int)
        X['Moisture_level'] = pd.cut(X["Moisture"], bins=[20, 30, 40, 50, 60, 70], labels=[1, 2, 3, 4, 5]).astype(int)
        self.feature_names_ = X.columns
        return X

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            raise ValueError("Call fit or transform first.")
        return self.feature_names_


X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']

X_fe = FeaturesEngineer(weights_dict=crop_weights).fit_transform(X)
X_fe_test = FeaturesEngineer(weights_dict=crop_weights).transform(test)


# Example column sets
categorical_cols = ['Soil Type', 'Crop Type']
numerical_cols = [col for col in X_fe.columns if col not in categorical_cols]

# Transformers
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
num_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
                                                ('c', cat_transformer, categorical_cols),
                                                ('n', num_transformer, numerical_cols)
                                 ])

# Full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])



main_transformer = pipeline.fit(X_fe)


## train test transform

new_train_df = pd.DataFrame(main_transformer.transform(X_fe),
                            columns=main_transformer.get_feature_names_out(),
                            index=train.index)

new_test_df = pd.DataFrame(main_transformer.transform(X_fe_test),
                            columns=main_transformer.get_feature_names_out(),
                            index=test.index)


new_train_df.describe()


score1 = []

# Try different k values
for i in range(1, 20):
    kmeans = MiniBatchKMeans(n_clusters=i+1, batch_size=50000, random_state=42)
    kmeans.fit(new_train_df)
    score1.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 20), score1, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('')
plt.grid(True)
plt.show()   


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Add, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import tensorflow as tf


# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


# # model layers
# input_layer = Input(shape=(new_train_df.shape[1],))

# # encoded
# x = Dense(7, activation='relu')(input_layer) 
# x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
# x = Dense(400, activation='relu', kernel_initializer='glorot_uniform')(x)
# x = Dense(800, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=0.0000001))(x)
# encoded = Dense(4, activation='relu', kernel_initializer='glorot_uniform')(x)

# # decoded
# x = Dense(800, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=0.0000001))(encoded)
# x = Dense(400, activation='relu', kernel_initializer='glorot_uniform')(x)
# x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
# decoded = Dense(new_train_df.shape[1], kernel_initializer='glorot_uniform')(x)


# autoencoder=Model(input_layer, decoded)
# encoder = Model(input_layer, encoded)
# autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# autoencoder.fit(
#                 new_train_df,
#                 new_train_df,
#                 epochs=60,
#                 batch_size=192,
#                 verbose=1,
#                 shuffle=False,
#                 validation_split=0.2,
#                 callbacks=[early_stop]
#                 )

# autoencoder.summary()


#To optimize runtime efficiency within the Kaggle notebook, the model â¬†ï¸� was pre-trained and saved in advance, 
# allowing for direct loading during execution.
autoencoder2 = load_model('/kaggle/input/autoencoder_keras_4_fertiliser/keras/default/1/autoencoder_model.keras')
encoder2 = load_model('/kaggle/input/encoder_keras_4_fertilzer/keras/default/1/encoder_model.keras', compile=False)


new_train_df.shape


encoded_df = encoder2.predict(new_train_df)
encoded_df.shape


encoded_test_df = encoder2.predict(new_test_df)
encoded_test_df.shape


score2 = []
for i in range(1, 20):
  cluster = MiniBatchKMeans(n_clusters=i+1, batch_size=50000, random_state=42)
  cluster.fit(encoded_df)
  score2.append(cluster.inertia_)


plt.plot(range(1, 20), score1, marker='o', linestyle='--', color='blue')
plt.plot(range(1, 20), score2, marker='o', linestyle='-', color='red')

plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Method')
plt.legend(['initial score', 'after autoencoder',])

for i, txt in enumerate(range(1, 20)):
    plt.annotate(txt, (range(1, 20)[i], score2[i]))
    plt.annotate(txt, (range(1, 20)[i], score1[i]))

plt.show()


from sklearn.metrics import silhouette_samples, silhouette_score

scores = []

for k in range(3, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(encoded_df)
    score = silhouette_score(encoded_df, labels, sample_size=10000, random_state=42)
    scores.append(score)
    print(f"k={k}, Silhouette Score={score:.4f}")   


max_score = max(scores)
best_k_num = scores.index(max_score) + 3   
print(f"Best score is {max_score:.4f} for {best_k_num} clusters")


kmeans=KMeans(best_k_num, random_state=42)
kmeans.fit(encoded_df)
labels=kmeans.labels_


pca = PCA(3, random_state=42)
principal_comp = pca.fit_transform(encoded_df)
pca_df = pd.DataFrame(data=principal_comp, columns= ['pca1', 'pca2', 'pca3'])
pca_clusters = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 8))
unique_clusters = sorted(pca_clusters['cluster'].unique())
palette = sns.color_palette("tab10", n_colors=len(unique_clusters))
cluster_to_color = {label: color for label, color in zip(unique_clusters, palette)}
colors = pca_clusters['cluster'].map(cluster_to_color)

# --- Left: 2D PCA Plot ---
ax1 = fig.add_subplot(1, 2, 1)
sns.scatterplot( x='pca1', y='pca2', hue='cluster', palette=cluster_to_color, data=pca_clusters,  alpha=0.6, ax=ax1)
ax1.set_title('2D PCA Scatter Plot')
ax1.set_xlabel('PCA 1')
ax1.set_ylabel('PCA 2')
ax1.legend(title="Cluster")

# --- Right: 3D PCA Plot ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scatter = ax2.scatter(pca_clusters['pca1'], pca_clusters['pca2'], pca_clusters['pca3'],
                      c=colors, s=50, alpha=0.5)

legend1 = ax2.legend(*scatter.legend_elements(), title="Cluster")
ax2.add_artist(legend1)

ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')
ax2.set_zlabel('PCA 3')
ax2.set_title('3D PCA Scatter Plot')

plt.tight_layout()
plt.show()


test_labels = kmeans.predict(encoded_test_df)
test_labels



encoded_train_data = pd.DataFrame(encoded_df, 
                                  columns = [f'encoded_{i}' for i in range(encoded_df.shape[1])],
                                  index=train.index)
encoded_test_data = pd.DataFrame(encoded_test_df, 
                                 columns = [f'encoded_{i}' for i in range(encoded_test_df.shape[1])],
                                 index=test.index)

encoded_train_data['clusters'] = labels
encoded_test_data['clusters'] = test_labels


encoded_train_data.describe()


class MultiTargetEncoder:
    """
    Multi-class target encoder using out-of-fold smoothing strategy.

    For each target class, encodes a categorical column by replacing each category 
    with a smoothed version of the likelihood that the target is equal to that class.
    This is done in an out-of-fold manner to reduce target leakage.

    Parameters
    ----------
    column : str
        The name of the categorical column to encode.

    n_splits : int, default=5
        Number of stratified folds used for out-of-fold encoding.

    smoothing : float, default=10
        Smoothing effect to balance category means with global mean.
        Higher values mean stronger regularization toward the global mean.

    Attributes
    ----------
    class_means_ : dict of dict
        Dictionary mapping each class label to a dictionary of category-to-encoded-value mappings.

    global_means_ : dict
        Dictionary mapping each class label to the global mean of the binary target (y == class).
    """
    
    def __init__(self, column, n_splits=5, smoothing=10):
        self.column = column
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.class_means_ = {}  
        self.global_means_ = {}  

    def fit_transform(self, X, y):
        X = X.copy()
        X_new = pd.DataFrame(index=X.index)
        classes = np.unique(y)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for cls in classes:
            col_name = f"{self.column}_te_class_{cls}"
            encoded = np.zeros(len(X))
            y_binary = (y == cls).astype(int)

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train_bin = y_binary.iloc[train_idx]

                means = y_train_bin.groupby(X_train[self.column]).mean()
                counts = X_train[self.column].value_counts()
                global_mean = y_train_bin.mean()

                smoothed = (means * counts + global_mean * self.smoothing) / (counts + self.smoothing)
                X_val_map = X_val[self.column].map(smoothed).fillna(global_mean)
                encoded[val_idx] = X_val_map.values

            X_new[col_name] = encoded

            # Store smoothed means and global mean for test
            y_binary_full = (y == cls).astype(int)
            full_means = y_binary_full.groupby(X[self.column]).mean()
            full_counts = X[self.column].value_counts()
            global_mean = y_binary_full.mean()

            smoothed_full = (full_means * full_counts + global_mean * self.smoothing) / (full_counts + self.smoothing)

            self.class_means_[cls] = smoothed_full.to_dict()
            self.global_means_[cls] = global_mean

        return X_new

    def transform(self, X):
        X = X.copy()
        X_new = pd.DataFrame(index=X.index)

        for cls, means in self.class_means_.items():
            col_name = f"{self.column}_te_class_{cls}"
            global_mean = self.global_means_[cls]
            encoded = X[self.column].map(means).fillna(global_mean)
            X_new[col_name] = encoded

        return X_new



# Generate a target-encoded feature using the interaction between Soil Type and Crop Type

X_fe['soil_crop'] = X_fe['Soil Type'] + X_fe['Crop Type']
target_encoder = MultiTargetEncoder(column='soil_crop', n_splits=5)
X_train_te_encoded = target_encoder.fit_transform(X_fe, y)

# encode test
X_fe_test['soil_crop'] = X_fe_test['Soil Type'] + X_fe_test['Crop Type']
X_test_te_encoded = target_encoder.transform(X_fe_test)


X_train_te_encoded.describe()


# Concat encoded data, clusters, original data

X = pd.concat([new_train_df.drop([col for col in new_train_df.columns if 'n__' in col], axis=1), 
               X_fe.drop(['Soil Type', 'Crop Type', 'soil_crop'], axis=1), 
               encoded_train_data, X_train_te_encoded], 
               axis=1)

X_test = pd.concat([new_test_df.drop([col for col in new_test_df.columns if 'n__' in col], axis=1), 
                    X_fe_test.drop(['Soil Type', 'Crop Type', 'soil_crop'], axis=1),
                    encoded_test_data, X_test_te_encoded], 
                    axis=1)


X_small, y_small = resample(X, y, n_samples=100_000, random_state=42)

importances = mutual_info_classif(X_small, y_small, random_state=42, n_neighbors=5)


mi_scores = pd.Series(importances, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=True)  

# Step 3: Plot
plt.figure(figsize=(10, 10))
mi_scores.plot(kind='barh', color='blue', alpha=0.6)
plt.title('Feature Importance via Mutual Information (Classification)')
plt.xlabel('Mutual Information Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


scaler = MinMaxScaler()
cols_scale = [col for col in X_small.columns if 'c__' not in col]
X_small_scaled = pd.DataFrame(scaler.fit_transform(X_small[cols_scale]), columns=cols_scale)
X_mi = pd.concat([X_small_scaled, X_small[[col for col in X_small.columns if 'c__' in col]].reset_index(drop=True, inplace=True)], axis=1)


from sklearn.feature_selection import mutual_info_regression

def compute_mutual_info_matrix(X):
    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            mi = mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j])
            mi_matrix[i, j] = mi[0]  
            mi_matrix[j, i] = mi[0]
    return mi_matrix

mi_matrix = compute_mutual_info_matrix(X_mi.values)


df_mi = pd.DataFrame(mi_matrix, columns=X_mi.columns, index=X_mi.columns)
plt.figure(figsize=(12,8))
mask = np.triu(np.ones_like(mi_matrix, dtype=bool))
sns.heatmap(df_mi, annot=True, fmt='.1f', mask=mask, cmap='flare')
plt.title('Mutual Information Analysis of Independent Features')


from typing import Sequence, Tuple, List
from numpy.typing import NDArray


def mapk(y_true: Sequence[int], y_pred_proba: NDArray, k: int = 3) -> float:
    """
    Computes Mean Average Precision at K (MAP@K).

    Parameters:
    - y_true: array-like of shape (n_samples,), true class labels.
    - y_pred_proba: array-like of shape (n_samples, n_classes), predicted class probabilities.
    - k: top-K classes to consider.

    Returns:
    - float: MAP@k score.
    """
    score = 0.0
    # Get top-k predicted class indices per row
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]

    for true, top_preds in zip(y_true, top_k_indices):
        if true in top_preds:
            rank = list(top_preds).index(true)
            score += 1.0 / (rank + 1)

    return score / len(y_true)


def test_model(model: ClassifierMixin, 
               X_df: pd.DataFrame, 
               y_df: pd.Series, 
               n_splits: int,
               print_mapk: bool = False) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """
    Trains and evaluates a classification model using Stratified K-Fold cross-validation.

    Parameters:
    - model (ClassifierMixin): A scikit-learn compatible classification model with `fit`, `predict`, and `predict_proba` methods.
    - X_df (pd.DataFrame): Feature matrix.
    - y_df (pd.Series): Target labels.
    - n_splits (int): Number of folds for cross-validation.

    Returns:
    - Tuple containing:
        - all_true (List[int]): Ground truth labels across all folds.
        - all_pred (List[int]): Out-of-fold predictions.
        - all_proba (List[np.ndarray]): Out-of-fold predicted probabilities.
    """

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_true = []
    all_pred = []
    all_proba = []
    all_idx = []

    for train_idx, test_idx in kf.split(X_df, y_df):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_df.iloc[train_idx], y_df.iloc[test_idx]
    
        # Fit model
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_proba.extend(y_proba)
        all_idx.extend(test_idx)
        
        # Print results
        print(f'{str(model)[:20]} confusion matrix: \n{confusion_matrix(y_test, y_pred)}')
        if print_mapk:
            print(f"{str(model)[:10]} MAP@3 - {mapk(all_true, all_proba, k=3):.3f}")


    return all_true, all_pred, all_proba, all_idx, model


le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y))


y_true, y_pred, y_proba, idx, model_clf1 = test_model(model=LGBMClassifier(verbose=0),
                                                      X_df = X, 
                                                      y_df = y_encoded, 
                                                      n_splits=5, 
                                                      print_mapk=True)
# Overall metrics for oof predictions
print(f"Main Model MAP@3 - {mapk(y_true, y_proba, k=3):.3f}")


import shap
explainer = shap.TreeExplainer(model_clf1)
shap_values = explainer.shap_values(X.iloc[:150])
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X.iloc[:150], plot_type="bar", max_display=X.shape[1], show=False)

# Add title and adjust layout
plt.title(f"SHAP Summary Plot for Model: {type(model_clf1).__name__}", fontsize=14)
plt.tight_layout()
plt.show()


from lime.lime_tabular import LimeTabularExplainer

row_num = 1

# Create explainer
explainer = LimeTabularExplainer(
    training_data=X.values if hasattr(X, "values") else X,
    feature_names=X.columns.tolist() if hasattr(X, "columns") else [f'feature_{i}' for i in range(X.shape[1])],
    class_names=model_clf1.classes_.tolist(),  
    mode='classification'              
)

# Explain one instance
exp = explainer.explain_instance(X.iloc[row_num], model_clf1.predict_proba, top_labels=1)

# Plot 
try:
    class_idx = exp.top_labels[0] 
    fig = exp.as_pyplot_figure(label=class_idx)  
    plt.title(f"LIME Explanation for ID{row_num}:\n prediction - {le.inverse_transform([class_idx])[0]},\ny_true - {y.iloc[row_num]}")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plotting failed:", e)


# Based on SHAP analysis, we will eliminate features with low importance scores.
drop_cols = [c for c in X.columns if 'c__' in c] + ['Moisture_level', "N_level", 'clusters']
X_final = X.drop(drop_cols, axis=1)


X_final.columns


skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

stacked_clf1 = StackingClassifier(estimators=[('xgb', XGBClassifier(verbosity=0,  
                                                                    n_estimators=250, 
                                                                    learning_rate=0.05, 
                                                                    max_depth=4,
                                                                    subsample=0.8,  
                                                                    colsample_bytree=0.8,
                                                                    use_label_encoder=False, 
                                                                    random_state=42)), 
                                              ('lgb', LGBMClassifier(verbose=0,  
                                                                     n_estimators=250,  
                                                                     learning_rate=0.05,
                                                                     max_depth=-1,  
                                                                     num_leaves=31, 
                                                                     subsample=0.8,
                                                                     colsample_bytree=0.8,  
                                                                     class_weight='balanced',  
                                                                     random_state=42)),
                                              ('cat', CatBoostClassifier(verbose=0, 
                                                                         n_estimators=250,  
                                                                         learning_rate=0.05, 
                                                                         random_state=42)),
                                              ('rf', RandomForestClassifier(n_estimators=200, 
                                                                            max_depth=None,
                                                                            min_samples_leaf=3,
                                                                            max_features='sqrt',
                                                                            class_weight='balanced'))
                                              ],
                                 final_estimator=LGBMClassifier(verbose=0,  
                                                                n_estimators=280,  
                                                                learning_rate=0.05),
                                 cv = skf,
                                 passthrough=True)

# stacked_clf1.fit(X_final, y_encoded)   ## kaggle score - 0.32635


from scipy.optimize import minimize
from scipy.special import softmax

class TemperatureScaler:
    """
    Temperature scaling for model calibration.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        def loss_fn(temp):
            temp = temp[0]
            scaled_logits = logits / temp
            probs = softmax(scaled_logits, axis=1)
            N = len(y_true)
            log_probs = -np.log(probs[np.arange(N), y_true] + 1e-15)
            return np.mean(log_probs)

        res = minimize(loss_fn, x0=[1.0], bounds=[(0.5, 5.0)])
        self.temperature = res.x[0]

    def transform(self, logits):
        return softmax(logits / self.temperature, axis=1)

    def fit_transform(self, logits, y_true):
        self.fit(logits, y_true)
        return self.transform(logits)


class OvRWrapper(BaseEstimator, ClassifierMixin):
    """
    One-vs-Rest wrapper
    """
    
    def __init__(self, base_model_class, n_classes=7, **model_params):
        self.n_classes = n_classes
        self.base_model_class = base_model_class
        self.model_params = model_params
        self.models = []
        self.classes_ = None


    def fit(self, X, y):
        self.models = []
        self.classes_ = np.unique(y)
    
        for cls in range(self.n_classes):
            y_bin = (y == cls).astype(int)
            base_model = self.base_model_class(**self.model_params)         
            base_model.fit(X, y_bin)
            self.models.append(base_model)
    
        return self

    
    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], self.n_classes))

        for cls, model in enumerate(self.models):
            if model is None:
                probs[:, cls] = 0.0
            else:
                probs[:, cls] = model.predict_proba(X)[:, 1]

        return probs

    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    
    def get_params(self, deep=True):
        return {
            'base_model_class': self.base_model_class,
            'n_classes': self.n_classes,
            **self.model_params
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'base_model_class':
                self.base_model_class = value
            elif key == 'n_classes':
                self.n_classes = value
            else:
                self.model_params[key] = value
        return self


def train_ovr(X, y, model, n_splits=10, temp_scale = False, **params):
    """
    Train OvR models with/without temp scaling and return OOF predictions.
    """
    
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((len(y), n_classes))
    oof_y = np.zeros(len(y), dtype=int)
    
    scaler_temps = [] 

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\n=== Fold {fold + 1} ===')

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y[val_idx]

        print(f"y_train unique classes: {np.unique(y_train)}")

        ovr_model = OvRWrapper(model, n_classes=n_classes, **params)
        ovr_model.fit(X_train, y_train)

        # Raw OOF Predictions (probabilities)
        raw_pred = ovr_model.predict_proba(X_val)

        # Temperature Scaling
        if temp_scale:
            temp_scaler = TemperatureScaler()
            scaled_probs = temp_scaler.fit_transform(np.log(raw_pred + 1e-15), y_val)
            scaler_temps.append(temp_scaler.temperature)
            predictions = scaled_probs

           
        else:
            predictions = raw_pred
            
        # Save raw probabilities
        oof[val_idx] = predictions
        oof_y[val_idx] = y_val
        
        # MAP@3 
        top3 = np.argsort(raw_pred, axis=1)[:, ::-1][:, :3]
        top3_list = [list(row) for row in top3]
        score = mapk(y_val, top3_list)

        print(f" MAP@3: {score:.4f}")
    

    # Create DataFrame for stacking
    model_name = (
        model.__class__.__name__
        if not isinstance(model, type)
        else model.__name__
    )
    results = pd.DataFrame(oof, columns=[f'{model_name}_class_{i}' for i in range(n_classes)])
    results['y_true'] = oof_y

    return results, scaler_temps


numerical_features = [col for col in X_final.columns if 'c__' not in col]
lr_preprocessor = ColumnTransformer(transformers=[('scale_num', StandardScaler(), numerical_features)])
lr_ovr_params = {'solver': 'saga', 
                 'max_iter': 2000, 
                 'class_weight': 'balanced', 
                 'penalty': 'l1', 
                 'C': 0.0930467} # on scaled data

def make_lr_pipeline(**params):
    return Pipeline([
        ('pre', lr_preprocessor),
        ('lr', LogisticRegression(**params))
    ])


results_lr, scaler_temps_lr = train_ovr(X_final, y_encoded, make_lr_pipeline, n_splits=4, **lr_ovr_params)


# X_test_final = X_test[X_final.columns]
# clf_final = stacked_clf1
# test_proba = clf_final.predict_proba(X_test_final) 


# fert_labels = le.inverse_transform(np.arange(len(clf_final.classes_)))
# top_test_indices = np.argsort(test_proba, axis=1)[:, ::-1][:, :3]
# top_preds = [[fert_labels[i] for i in row] for row in top_test_indices]
# fertilizer_names = [' '.join(row) for row in top_preds]

# final_df = pd.DataFrame(data=fertilizer_names,
#                        columns=['Fertilizer Name'],
#                        index = test.index)


# final_df.to_csv('submission2.csv')

