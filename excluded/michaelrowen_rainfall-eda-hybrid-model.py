import pandas as pd
import numpy as np

TEST_PATH = "/kaggle/input/playground-series-s5e3/test.csv"
TRAIN_PATH = "/kaggle/input/playground-series-s5e3/train.csv"
df = pd.read_csv(TRAIN_PATH)
train_df = df.copy()
test_df = pd.read_csv(TEST_PATH)
train_df.head()


print("train info: ")
train_df.info()
print("=" * 50 + "\ntest info: ")
test_df.info()


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].median())
train_df.describe()


test_df.describe()


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

seasonal_freqs = {
    'annual': 365,
    'semi-annual': 183,
    'quarterly': 92,
    'half-quarterly': 46,
    'monthly': 30,
    'half-monthly': 15
    }
def add_seasonality(in_df):
    for key, val in seasonal_freqs.items():
        in_df[key] = (in_df['id'] // val) + 1

add_seasonality(train_df)
add_seasonality(test_df)

train_df['annual'].describe()


fig, axes = plt.subplots(11, 2, figsize=(16, 45))
plt.subplots_adjust(hspace=0.4)


features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']


for i, feature in enumerate(features):
    # Box plot on the left
    sns.boxplot(x='annual', y=feature, data=train_df, ax=axes[i, 0])
    axes[i, 0].set_title(f'{feature} by Year (Box Plot)')
    axes[i, 0].set_xlabel('Year')
    axes[i, 0].set_ylabel(feature.capitalize())
    
    # Histogram with KDE on the right
    sns.histplot(data=train_df, x=feature, kde=True, hue='rainfall', ax=axes[i, 1])
    axes[i, 1].set_title(f'{feature} Distribution (Histogram with Density)')
    axes[i, 1].set_xlabel(feature.capitalize())
    axes[i, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


import matplotlib.colors as mcolors

n_features = len(features)
n_cols = 3  
n_rows = (n_features + n_cols - 1) // n_cols  


fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))

if n_rows == 1:
    axes = axes.reshape(1, -1)

years = sorted(train_df['annual'].unique())
colors = list(mcolors.TABLEAU_COLORS)[:len(years)]
year_colors = dict(zip(years, colors))

for i, feature in enumerate(features):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    ax.set_title(f'Distribution of {feature} across years', fontsize=14)
    
    for year in years:
        year_data = train_df[train_df['annual'] == year][feature]
        sns.kdeplot(year_data, label=f'Year {year}', 
                     ax=ax, color=year_colors[year], alpha=0.6)
    
    ax.legend()
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')

if n_features % n_cols != 0:
    for j in range(n_features % n_cols, n_cols):
        fig.delaxes(axes[n_rows-1, j])

plt.tight_layout()
plt.show()


seasons = seasonal_freqs.keys()
n_seasons = len(seasons)
n_cols = 3  
n_rows = (n_seasons + n_cols - 1) // n_cols  


fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)


for i, season in enumerate(seasons):
    uniques = sorted(train_df[season].unique())
    colors = list(mcolors.TABLEAU_COLORS)[:len(uniques)] 
    
    unique_colors = dict(zip(uniques, colors))
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    ax.set_title(f'Distribution of {season} RainFall', fontsize=14)
    
    for unique in uniques:
        color = unique_colors[unique] if unique < len(list(mcolors.TABLEAU_COLORS)) else None
        unique_data = train_df[train_df[season] == unique]["rainfall"]
        sns.kdeplot(unique_data, label=f'{season} {unique}', 
                     ax=ax, color=color, alpha=0.6)
    
    ax.set_xlabel("RainFall")
    ax.set_ylabel('Frequency')

if n_seasons % n_cols != 0:
    for j in range(n_features % n_cols, n_cols):
        fig.delaxes(axes[n_rows-1, j])

plt.tight_layout()
plt.show()
    


plt.figure(figsize=(18, 15))
features.remove("rainfall")
n_features = len(features)
n_cols = 2
n_rows = (n_features + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()

for i, feature in enumerate(features):
    if i < len(axes):
        ax = axes[i]
        sns.kdeplot(
            data=train_df, x=feature, hue="rainfall", 
            fill=True, common_norm=False, 
            palette=['red', 'blue'], alpha=0.5,
            ax=ax
        )
        ax.set_title(f'Distribution of {feature} by Target')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


def feature_engineering(df):
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_deviation'] = df['temparature'] - df['avg_temp']
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']
    # df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
    # df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    # df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    # df.drop(columns=['wind_dir_rad'], inplace=True)
    df['humidity_temp'] = df['humidity'] * df['temparature']
    # df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    # df['season_cloud_trend'] = df['cloud'] * df['quarterly']
    # df["cloud_humidity/pressure"] = (df["cloud"] * df["humidity"]) / df["pressure"]
    # df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1)
    # df["cloud_windspeed"] = df["cloud"] * df["windspeed"] # ***
    # df["cloud_to_humidity"] = df["cloud"] / df["humidity"]

    for c in ['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 
              'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']:
        for gap in [1]:
            df[c+f"_shift{gap}"]=df[c].shift(gap)
            df[c+f"_diff{gap}"]=df[c].diff(gap)



from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

X = train_df.drop(columns=['id', 'day','rainfall'])
test_dataset = test_df.drop(columns=['id', 'day'])
feature_engineering(X), feature_engineering(test_dataset)
y = train_df['rainfall']

scaler = MinMaxScaler()
X_scaled= scaler.fit_transform(X)
test_scaled = scaler.transform(test_dataset)


FOLD = 5
skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=42)
split = skf.split(X_scaled, y)
X.info()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

class HybridModel:
    def __init__(self, shape):
        self.input_shape = shape
        self.nn_model = None
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        self.optimizer = Adam(learning_rate=0.001)
        self.xgb_params = {
            'objective': 'binary:logistic',   
            'eval_metric': 'auc',             
            'max_depth': 5,                   
            'learning_rate': 0.02,            
            'n_estimators': 320,              
            
            # Regularization parameters
            'reg_alpha': 0.01,                 
            'reg_lambda': 0.01,                
            'min_child_weight': 3.1,            
            'subsample': 0.8,                 
            'colsample_bytree': 0.9,          
        
            'scale_pos_weight': 3, 
            
            'tree_method': 'hist',            
            'random_state': 42               
        }
        self.xgb_model = XGBClassifier(
            **self.xgb_params            
        )
        self.build_nn_model()
    def build_nn_model(self):
        self.nn_model = Sequential([
            Dense(128, activation='relu', 
                  kernel_initializer='he_normal', 
                  input_shape=(self.input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu', 
                  kernel_initializer='he_normal', 
                  input_shape=(self.input_shape,)),
            Dropout(0.3),
            Dense(32, activation='relu', 
                  kernel_initializer='he_normal'),
            Dropout(0.2),
            Dense(16, activation='relu', 
                  kernel_initializer='he_normal'),
            Dense(1, activation='sigmoid')  
        ])
        self.nn_model.compile(optimizer=self.optimizer, loss='binary_crossentropy', 
                           metrics=['accuracy'])
        # return self.model.summary()
    def get_interaction_feature_importance(self, interaction_features):
        return pd.DataFrame({
            'features': interaction_features,
            "importance": self.xgb_model.feature_importances_
        })
    def fit(self, X_train, y_train, trend_indices, interaction_indices):
    
        self.xgb_model.fit(X_train[:, interaction_indices], y_train)
        
        self.nn_model.fit(X_train[:, trend_indices], y_train, 
                  epochs=200, batch_size=32, 
                  callbacks=[self.early_stopping], 
                  verbose=0)
        
    def predict(self, test_X, trend_indices, interaction_indices):
        return (self.nn_model.predict(test_X[:, trend_indices]).flatten() + 
                self.xgb_model.predict_proba(test_X[:, interaction_indices])[:, 1]) / 2.



fold_auc = []
trend_features = ["annual", "quarterly", "monthly", 
                        "half-monthly",'semi-annual', 'half-quarterly']

interaction_features = list(set(X.columns.to_list()).difference(set(trend_features)))


all_feature_names = X.columns.to_list()
interaction_indices = [all_feature_names.index(feature) for feature in interaction_features]
trend_indices = [all_feature_names.index(feature) for feature in trend_features]


for fold, (train_idx, val_idx) in enumerate(split):
    print(f"fold {fold+1} in progress")
    print("=" * 70)
    model = HybridModel(len(trend_indices))
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train, trend_indices, interaction_indices)
    y_pred = model.predict(X_val, trend_indices, interaction_indices)
    
    auc = roc_auc_score(y_val, y_pred)
    fold_auc.append(auc)


print(f"Mean {FOLD} fold MSE: {np.mean(fold_auc)}" )


model = HybridModel(len(trend_indices))
model.fit(X_scaled, y, trend_indices, interaction_indices)
result_y = model.predict(test_scaled, trend_indices, interaction_indices)


interaction_df = model.get_interaction_feature_importance(interaction_features)
interaction_df = interaction_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 10))
sns.barplot(x='importance', y='features', data=interaction_df)
plt.title(f"Interaction Features Importance")
plt.show()


sub = pd.DataFrame({
    'id': test_df.id,
    'rainfall': result_y
})
sub.to_csv("submission.csv", index=False)
sub.head()

