import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

import optuna


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pd.set_option('display.float_format', '{:.1f}'.format)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


class dfCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean podcast listening dataset:
    - Caps extreme ad counts.
    - Adjusts episode lengths.
    - Fixes percentages > 100.
    - Ensures logical consistency between listening time and episode length.
    - Fills missing values using group means by Podcast_Name and Episode_Title.
    """
    
    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """No fitting required."""
        return self

    def transform(self, X):
        X = X.copy()

        # outlier filtering
        numeric_cols = X.select_dtypes(include='number').columns
        thresholds = X[numeric_cols].quantile(0.99)
        no_nans = X[numeric_cols].notna().all(axis=1)
        within_thresholds = (X[numeric_cols] <= thresholds).all(axis=1)
        df_filtered = X[no_nans & within_thresholds].copy()


        # find group (nooutlier) means
        group_cols = ['Podcast_Name', 'Publication_Time', 'Genre']
        group_means = df_filtered.groupby(group_cols)[numeric_cols].mean().reset_index()
        group_means = group_means.rename(columns={col: f"{col}_mean" for col in numeric_cols})
        X = X.merge(group_means, how='left', on=group_cols)
        
        # Fill with group mean first, then global mean as fallback
        for col in numeric_cols:
            mean_col = f"{col}_mean"
            X[col] = X[col].fillna(X[mean_col])
            X[col] = X[col].fillna(X[col].mean())  # fallback to overall mean
        
        X.drop(columns=[f"{col}_mean" for col in numeric_cols], inplace=True)
        
        # Cap Number_of_Ads
        if 'Number_of_Ads' in X.columns:
            X.loc[X['Number_of_Ads'] > 3, 'Number_of_Ads'] = 1
    
        # Adjust Episode_Length_minutes > 121
        if 'Episode_Length_minutes' in X.columns:
            ep_group = ['Genre', 'Podcast_Name']
            if all(col in X.columns for col in ep_group):
                # Create a group mean DataFrame
                ep_means_df = (
                    df_filtered.groupby(ep_group)['Episode_Length_minutes']
                    .mean()
                    .reset_index()
                    .rename(columns={'Episode_Length_minutes': 'Episode_Length_mean'})
                )
        
                # Merge the means into X
                X = X.merge(ep_means_df, how='left', on=ep_group)
        
                # Replace long episode lengths with group mean
                mask = X['Episode_Length_minutes'] > 121
                X.loc[mask, 'Episode_Length_minutes'] = X.loc[mask, 'Episode_Length_mean']
                X.loc[mask, 'Episode_Length_minutes'] = X['Episode_Length_minutes'].fillna(X['Episode_Length_minutes'].mean())
        
                # Drop the temp column
                X.drop(columns='Episode_Length_mean', inplace=True)
            
        # Fix popularity percentage > 100
        for col in ['Host_Popularity_percentage', 'Guest_Popularity_percentage']:
            if col in X.columns:
                mask = X[col] > 100
                if mask.any():
                    X.loc[mask, col] = X.loc[~mask, col].mean()
    
        if 'id' in X.columns:
            X = X.set_index('id')

        self.feature_names_ = X.columns
           
        return X

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            raise ValueError("You must call fit or fit_transform before getting feature names.")
        return self.feature_names_



class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer for handling categorical features using:
    - One-Hot Encoding for low-cardinality (<11)
    - Frequency Encoding for high-cardinality (>70)
    - Target Encoding for intermediate cardinality (11 to 70)
    """
    def __init__(self, target_name=None, smooth_factor=1):
        self.target_name = target_name
        self.smooth_factor = smooth_factor
        self.encoders = {}  
        self.freq_maps = {}
        self.one_hot_encoders = {}
        self.feature_names_ = None 

    def fit(self, X, y=None):
        X = X.copy()
        if self.target_name is None and y is None:
            raise ValueError("Either 'target_name' or 'y' must be provided for target encoding.")

        if y is None:
            y = X[self.target_name]

        cat_df = X.select_dtypes(include=['object', 'category'])

        for col in cat_df.columns:
            nunique = cat_df[col].nunique()

            if nunique < 11:
                encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
                encoder.fit(cat_df[[col]])
                self.one_hot_encoders[col] = encoder

            elif nunique > 70:
                freq_map = cat_df[col].value_counts().to_dict()
                self.freq_maps[col] = freq_map

            else:
                target_encoder = TargetEncoder(cols=[col], smoothing=self.smooth_factor)
                target_encoder.fit(cat_df[[col]], y)
                self.encoders[col] = target_encoder

        return self

    def transform(self, X):
        X = X.copy()
        final_df = pd.DataFrame(index=X.index)
        cat_df = X.select_dtypes(include=['object', 'category'])

        for col in cat_df.columns:
            if col in self.one_hot_encoders:
                encoder = self.one_hot_encoders[col]
                encoded_array = encoder.transform(cat_df[[col]])
                encoded_cols = encoder.get_feature_names_out([col])
                encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=cat_df.index)
                final_df = final_df.join(encoded_df)

            elif col in self.freq_maps:
                freq_map = self.freq_maps[col]
                final_df[col + '_freq'] = cat_df[col].map(freq_map)

            elif col in self.encoders:
                encoder = self.encoders[col]
                encoded_df = encoder.transform(cat_df[[col]])
                final_df = final_df.join(encoded_df)

        # Preserve numeric columns except the target
        numeric_df = X.select_dtypes(include=['int', 'float']).drop(columns=[self.target_name], errors='ignore')
        final_df = final_df.join(numeric_df)

        # Save feature names
        self.feature_names_ = final_df.columns.tolist()

        return final_df

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            raise ValueError("You must call fit or fit_transform before getting feature names.")
        return self.feature_names_



class AddClusterLabels(BaseEstimator, TransformerMixin):
    """
    Adds KMeans cluster labels to the dataset using One-Hot Encoding.
    - Automatically handles DataFrame or NumPy array inputs.
    - One-hot encoded cluster labels are appended to the original features.
    """
    def __init__(self, n_clusters=10, one_hot=True, drop_first=True):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        self.one_hot = one_hot
        self.drop_first = drop_first
        self.encoder = None
        self.feature_names_in_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        labels = self.kmeans.labels_.reshape(-1, 1)

        if self.one_hot:
            self.encoder = OneHotEncoder(sparse=False, drop='first' if self.drop_first else None)
            self.encoder.fit(labels)

        # extract feature names from the previous step in the pipeline
        if hasattr(self, 'previous_step') and hasattr(self.previous_step, 'get_feature_names_out'):
            self.feature_names_in_ = self.previous_step.get_feature_names_out()
        else:
            # If no previous step or no feature names from previous step, fallback to default
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]

        # Generate cluster feature names
        if self.one_hot:
            cluster_names = self.encoder.get_feature_names_out(['cluster'])
        else:
            cluster_names = [f'cluster_label']

        self.feature_names_ = self.feature_names_in_ + list(cluster_names)

        return self

    def transform(self, X):
        labels = self.kmeans.predict(X).reshape(-1, 1)
        if self.one_hot:
            cluster_features = self.encoder.transform(labels)
        else:
            cluster_features = labels  # Keep single column

        return np.hstack([X, cluster_features])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_  # fallback to original feature names

        # Generate final feature names
        cluster_names = (
            self.encoder.get_feature_names_out(['cluster']) if self.one_hot
            else [f'cluster_label']
        )
        
        return list(input_features) + list(cluster_names)


class AddPrincipalComponents(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        # Apply PCA and append the components as new columns
        pca_components = self.pca.transform(X)
        return np.hstack([X, pca_components])


def objective(trial, X_train, y_train, params, model):
    try:
        model = model(**params)
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)  

        rmse_scores = cross_val_score(
            model, X_train, y_train, 
            cv=kfold, 
            scoring='neg_root_mean_squared_error'
        )
        mean_rmse = -rmse_scores.mean()
        return mean_rmse

    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise TrialPruned()


numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage',
'Number_of_Ads', 'Podcast_Name', 'Episode_Title_freq'] 

scale_numeric = ColumnTransformer(transformers=[('scale_num', StandardScaler(), numeric_cols)],
                                  remainder='passthrough')


preprocess_base =  Pipeline(steps=[('cleaner', dfCleaner()),
                                   ('category_transform', SmartCategoricalEncoder(target_name='Listening_Time_minutes', smooth_factor=5)),
                                   ('scale_numeric', scale_numeric)
                                  ])


preprocess_to_PCA =  Pipeline(steps=[('cleaner', dfCleaner()),
                                     ('category_transform', SmartCategoricalEncoder(target_name='Listening_Time_minutes', smooth_factor=5)),
                                     ('scale_numeric', scale_numeric),
                                     ('pca', PCA(n_components=25)),
                                     ])


preprocess_with_clusters =  Pipeline(steps=[('cleaner', dfCleaner()),
                                     ('category_transform', SmartCategoricalEncoder(target_name='Listening_Time_minutes', smooth_factor=5)),
                                     ('scale_numeric', scale_numeric),
                                     ('cluster', AddClusterLabels(n_clusters=12, one_hot=True, drop_first=True)),
                                     ])


df_transformed = preprocess_base.fit_transform(train_df)
column_names = preprocess_base.get_feature_names_out() 
clear_col = [col.replace('scale_num__', 'scaled_').replace('remainder__', '') for col in column_names]
df_transformed = pd.DataFrame(data = df_transformed, columns = clear_col)
df_transformed.head(5)



## Elbow Method
wcss = []  # Within-cluster sum of squares
K_range = range(1, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(df_transformed)
    wcss.append(kmeans.inertia_)

# 4. Plot the Elbow
plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()



n_comp = 25

pca = PCA(n_components=n_comp)
pr_comp = pca.fit_transform(df_transformed)

pca_df = pd.DataFrame(data= pr_comp, columns=[f'pca_{i}' for i in range(n_comp)])
        
        
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(exp_var)+1), exp_var, marker='o', label='Individual Explained Variance')
plt.plot(range(1, len(cum_var)+1), cum_var, marker='s', label='Cumulative Explained Variance', color='green')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by PCA Components')
plt.legend()
plt.grid(True)
plt.show()


for i, (var, cum) in enumerate(zip(exp_var, cum_var), 1):
    print(f"PC{i}: {var:.4f}  |  Cumulative: {cum:.2f}")


# Turn features to PC

df_PCA = pd.DataFrame(preprocess_to_PCA.fit_transform(train_df), 
                         columns=preprocess_to_PCA.get_feature_names_out(),  
                         index=train_df.index)


# Turn features to scaled version and add clusters

df_clusters = pd.DataFrame(preprocess_with_clusters.fit_transform(train_df), 
                          columns=preprocess_with_clusters.get_feature_names_out(),  
                          index=train_df.index)  


# Train/test split 

train_idx, test_idx = train_test_split(df_PCA.index, test_size=0.1, random_state=42)


# for PCA df
X_train_pc, X_valid_pc = df_PCA.loc[train_idx], df_PCA.loc[test_idx]
y_train_pc, y_valid_pc = train_df['Listening_Time_minutes'].loc[train_idx], train_df['Listening_Time_minutes'].loc[test_idx]


# for clustered df
X_train_c, X_valid_c = df_clusters.loc[train_idx], df_clusters.loc[test_idx]
y_train_c, y_valid_c = train_df['Listening_Time_minutes'].loc[train_idx], train_df['Listening_Time_minutes'].loc[test_idx]


# for test_df (submission)

test_data_PCA = pd.DataFrame(preprocess_to_PCA.transform(test_df), 
                             columns=preprocess_to_PCA.get_feature_names_out(),  
                             index=test_df.index) 

test_data_clusters = pd.DataFrame(preprocess_with_clusters.transform(test_df), 
                                  columns=preprocess_with_clusters.get_feature_names_out(),  
                                  index=test_df.index) 


# find best params for model

def define_EN_params(trial: object) -> object:
        params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-6, 1e+1),
        'l1_ratio': trial.suggest_uniform('l1_ratio', 0.0, 1.0),
        'random_state': 42
        }
        return params


# study_EN = optuna.create_study(direction='minimize')
    
# # # Define the objective function inline, using the current dataset version
# study_EN.optimize(lambda trial: objective(trial, 
#                                            X_train=X_train_pc,
#                                            y_train=y_train_pc,
#                                            params=define_EN_params(trial),
#                                            model=ElasticNet
#                                            ),
#                                  n_trials=100
#                   )


# best_en_params = study_EN.best_trial.params
# best_en_params 


# best_EN_param = study_EN.best_trial.params

best_EN_param_for_pc = {'alpha': 0.0007671856836737196, 'l1_ratio': 0.9329328104912228}
best_EN_param_for_df_with_clusters = {'alpha': 0.005253015057271132, 'l1_ratio': 0.9965909461616541}



#fit model
model = ElasticNet(**best_EN_param_for_pc)
en_model = model.fit(X_train_pc, y_train_pc)

# train test predictions
train_en_predict = en_model.predict(X_train_pc)
test_en_predict = en_model.predict(X_valid_pc)

print(f'rmse for valid - {np.sqrt(mean_squared_error(y_valid_pc, test_en_predict))}')

# rmse for pca - 13.2939
# rmse for scaled with clusters - 13.35398


# residuals count

train_en_resid = y_train_pc - train_en_predict
test_en_resid = y_valid_pc - test_en_predict


train_en_resid.plot(kind='hist', alpha=0.5, color='blue', label='Train Residuals')
test_en_resid.plot(kind='hist', alpha=0.5, color='red', label='Test Residuals')
plt.title('Residuals Distribution')
plt.show()


# coef_df = pd.DataFrame({
#     'Feature': [f'pca_{i}' for i in range(25)],
#     'Coefficient': en_model.coef_
# })
# coef_df.sort_values('Coefficient', ascending=True)


# kfold = KFold(n_splits=3, shuffle=True, random_state=42)  
# rmse_scores = cross_val_score(en_model, df, train_df['Listening_Time_minutes'], cv=kfold, scoring='neg_root_mean_squared_error')
# mean_rmse = -rmse_scores.mean()
# mean_rmse


final_en_prediction = en_model.predict(test_data_PCA)


submission_df = pd.DataFrame(index=test_df.id, columns=['Listening_Time_minutes'], data = final_en_prediction)
submission_df.to_csv('submission1.csv')


def define_XGB_params(trial: object) -> object:
        params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 350),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1
        }
        return params


# study_XGB = optuna.create_study(direction='minimize')
    
# # Define the objective function inline, using the current dataset version
# study_XGB.optimize(lambda trial: objective(trial, 
#                                            X_train=X_train_pc,
#                                            y_train=train_en_resid,
#                                            params=define_XGB_params(trial),
#                                            model=XGBRegressor,
#                                            ),
#                                  n_trials=100
#                   )


# best_XGB_params = study_XGB.best_trial.params
# best_XGB_params 


best_XGB_params_for_PC =  {'n_estimators': 180, 
                           'max_depth': 13, 
                           'learning_rate': 0.031294078471867944, 
                           'subsample': 0.8961233425857253, 
                           'colsample_bytree': 0.8029783192873106}

best_XGB_params_for_PC_resid =  {'n_estimators': 320, 
                                 'max_depth': 15, 
                                 'learning_rate': 0.014596228675714619, 
                                 'subsample': 0.8676207459115856, 
                                 'colsample_bytree': 0.7971027502303012}

best_XGB_for_df_with_clusters = {'n_estimators': 266,
                                 'max_depth': 14,
                                 'learning_rate': 0.0337948950616333,
                                 'subsample': 0.8340874974199307,
                                 'colsample_bytree': 0.7846367146534168}


# fit model
XGB_model = XGBRegressor(**best_XGB_for_df_with_clusters)      
XGB_model.fit(X_train_c, train_en_resid)

# predict residuals
train_resid_predict = XGB_model.predict(X_train_c)
test_resid_predict = XGB_model.predict(X_valid_c)

#count final prediction
final_train_predict = train_en_predict + train_resid_predict
final_test_predict = test_en_predict + test_resid_predict

#count rmse
print(f'rmse for train - {np.sqrt(mean_squared_error(y_train_c, final_train_predict))}')
print(f'rmse for valid - {np.sqrt(mean_squared_error(y_valid_c, final_test_predict))}')


final_resid_predict = XGB_model.predict(test_data_clusters)
main_predict = final_en_prediction + final_resid_predict


main_predict


submission_df = pd.DataFrame(index=test_df.id, columns=['Listening_Time_minutes'], data = main_predict)
submission_df.to_csv('submission1.csv')


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Add, BatchNormalization, LayerNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


## NN model 1
## listen_time prediction

input_layer = Input(shape=(X_train_c.shape[1],))
x = BatchNormalization()(input_layer)

x = Dense(64)(x)
x = LayerNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

# Residual block
res1 = Dense(64)(x)
res1 = LayerNormalization()(res1)
res1 = Activation('relu')(res1)
res1 = Dropout(0.4)(res1)

x = Add()([x, res1])
x = LayerNormalization()(x)  

# Further layers
x = Dense(128)(x)
x = LayerNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)

# Build the model
model_nn_1 = Model(inputs=input_layer, outputs=output)

# Compile the model
model_nn_1.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Set early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=6, 
                           restore_best_weights=True,
                           min_delta=0.005) 



# Train model_nn_1
history = model_nn_1.fit(X_train_c, y_train_c, 
                       validation_data=(X_valid_c, y_valid_c), 
                       epochs=50, 
                       batch_size=128, 
                       callbacks=[early_stop],
                       verbose=1)

# Predict on test set
y_pred = model_nn_1.predict(X_valid_c)
rmse = np.sqrt(mean_squared_error(y_valid_c, y_pred))
print("Test RMSE:", rmse)


history_df = pd.DataFrame(history.history)
history_df.loc[1:, ['loss', 'val_loss']].plot()


y_train_predict = model_nn_1.predict(X_train_c)


## train residuals
y_train_true = y_train_pc.values    # converts to numpy
y_train_pred = y_train_predict.flatten()

train_residuals = y_train_true - y_train_pred


# valid residuals 

y_valid_true = y_valid_pc.values    # converts to numpy
y_valid_pred = y_pred.flatten()

valid_residuals =  y_valid_true - y_valid_pred 


## nn model 2 ResNet-style 
## residuals predicion

# Input Layer
input_layer = Input(shape=(X_train_c.shape[1],))
x = BatchNormalization()(input_layer)

# First Dense block
x = Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
x = LayerNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

# Residual Block 1
res1 = Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
res1 = LayerNormalization()(res1)
res1 = Activation('relu')(res1)
res1 = Dropout(0.4)(res1)

x = Add()([x, res1])
x = LayerNormalization()(x)  # Normalize after add

# Residual Block 2
res2 = Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
res2 = LayerNormalization()(res2)
res2 = Activation('relu')(res2)
res2 = Dropout(0.4)(res2)

x = Add()([x, res2])
x = LayerNormalization()(x)

# Residual Block 3
res3 = Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
res3 = LayerNormalization()(res3)
res3 = Activation('relu')(res3)
res3 = Dropout(0.3)(res3)

x = Add()([x, res3])
x = LayerNormalization()(x)

# Final layers
x = Dense(32, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(16, activation='relu')(x)

# Output Layer
output = Dense(1)(x)

# Build model
model_nn_2 = Model(inputs=input_layer, outputs=output)

# Compile
model_nn_2.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=6, 
                           restore_best_weights=True,
                           min_delta=0.005)


history_resid = model_nn_2.fit(X_train_c, train_residuals, 
                       validation_data=(X_valid_c, valid_residuals), 
                       epochs=70, 
                       batch_size=128, 
                       callbacks=[early_stop],
                       verbose=1)

# Predict on test set
resid_pred = model_nn_2.predict(X_valid_c)
rmse = np.sqrt(mean_squared_error(y_valid_c, resid_pred))
print("Test RMSE:", rmse)


history_resid_df = pd.DataFrame(history_resid.history)
history_resid_df.loc[2:, ['loss', 'val_loss']].plot()


final_valid_pred =  y_valid_pred + resid_pred.flatten()
rmse = np.sqrt(mean_squared_error(y_valid_pc, final_valid_pred))
print("Test RMSE:", rmse)


# predict from clustered test df
predictions_nn1_pca = model_nn_1.predict(test_data_clusters)
residuals_nn2_cluster = model_nn_2.predict(test_data_clusters)
final_pred = predictions_nn1_pca + residuals_nn2_cluster


final_pred



submission_df = pd.DataFrame(index=test_df.id, columns=['Listening_Time_minutes'], data = final_pred)
submission_df.to_csv('submission2.csv')

