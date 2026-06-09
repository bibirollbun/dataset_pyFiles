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


import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


print("Shape of Train Data", train_df.shape)


display(train_df.head(10))


train_df.info()


train_df.nunique()


train_df.describe().transpose()


plt.figure(figsize=(12, 6))
soil_counts = train_df['Soil Type'].value_counts()

# Plot barplot
ax = sns.barplot(x=soil_counts.index, y=soil_counts.values, palette="viridis")

plt.title("Distribution of Soil Types ", fontsize=15)
plt.xlabel("Soil Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add percentage labels on top of each bar
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(14, 7))
crop_counts = train_df['Crop Type'].value_counts()

# Plot barplot
ax = sns.barplot(x=crop_counts.index, y=crop_counts.values, palette="magma")
plt.title("Distribution of Crop Types", fontsize=15)
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add percentage labels
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(14, 7))  # Wider figure for more categories
fert_counts = train_df['Fertilizer Name'].value_counts()

# Plot barplot
ax = sns.barplot(x=fert_counts.index, y=fert_counts.values, palette="plasma")
plt.title("Distribution of Fertilizer Names", fontsize=15)
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=90)  # Rotate 90Â° if labels overlap

# Add percentage labels
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',
            ha='center', va='center', fontsize=9)  # Smaller font for tight spaces

plt.tight_layout()  # Prevent label cutoff
plt.show()


plt.figure(figsize=(10, 20))
pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name']).plot(kind='bar', stacked=False, colormap='viridis')
plt.title('Fertilizer Preference by Soil Type', fontsize=16)
plt.xlabel('Soil Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 20))
pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name']).plot(kind='bar', stacked=False, colormap='viridis')
plt.title('Fertilizer Preference by Crop Type', fontsize=16)
plt.xlabel('Crop Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Compare average N-P-K levels per Soil
train_df.groupby('Soil Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().plot(kind='bar', figsize=(14, 6))
plt.title('Average Nutrient Levels by Soil Type')
plt.show()


# Compare average N-P-K levels per crop
train_df.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().plot(kind='bar', figsize=(14, 6))
plt.title('Average Nutrient Levels by Crop Type')
plt.show()


# Heatmap: Crop vs. Fertilizer (Counts)
cross_tab = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])
plt.figure(figsize=(16, 8))
sns.heatmap(cross_tab, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Crop-Fertilizer Frequency', fontsize=16)
plt.xticks(rotation=45)
plt.show()


numerical_df = train_df.select_dtypes(include=['int64', 'float64'])


numerical_df.columns


# from scipy import stats
# from itertools import combinations
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Get all pairs of numerical columns
# column_pairs = combinations(['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'], 2)

# # Set style
# sns.set(style="whitegrid")

# # Loop through each pair and plot
# for col1, col2 in column_pairs:
#     # Create figure
#     plt.figure(figsize=(10, 6))
    
#     # Scatter plot with regression line
#     sns.regplot(x=col1, y=col2, data=numerical_df, scatter_kws={'alpha':0.6})
    
#     # Calculate statistics
#     corr_coef, p_value = stats.pearsonr(numerical_df[col1].dropna(), numerical_df[col2].dropna())
#     slope, intercept, _, _, _ = stats.linregress(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    
#     # Add statistics to plot
#     stats_text = (f"Pearson r = {corr_coef:.2f}\n"
#                   f"p-value = {p_value:.4f}\n"
#                   f"Regression: y = {slope:.2f}x + {intercept:.2f}")
    
#     plt.gcf().text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
#                    bbox=dict(facecolor='white', alpha=0.8))
    
#     # Titles and labels
#     plt.title(f'{col1} vs {col2}', fontsize=14)
#     plt.xlabel(col1, fontsize=12)
#     plt.ylabel(col2, fontsize=12)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Automated interpretation
#     abs_r = abs(corr_coef)
    
#     # Interpret Pearson r
#     if abs_r >= 0.8:
#         strength = "very strong"
#     elif abs_r >= 0.6:
#         strength = "strong"
#     elif abs_r >= 0.4:
#         strength = "moderate"
#     elif abs_r >= 0.2:
#         strength = "weak"
#     else:
#         strength = "very weak or no"
    
#     direction = "positive" if corr_coef > 0 else "negative" if corr_coef < 0 else "no"
    
#     # Interpret p-value
#     if p_value < 0.001:
#         sig_text = "highly statistically significant (p < 0.001)"
#     elif p_value < 0.05:
#         sig_text = "statistically significant (p < 0.05)"
#     else:
#         sig_text = "not statistically significant (p â‰¥ 0.05)"
    
#     # Print interpretation
#     print(f"\nInterpretation for {col1} vs {col2}:")
#     print(f"- {strength} {direction} linear relationship")
#     print(f"- The correlation is {sig_text}\n")
#     print("-" * 60)  # Separator line


corr = abs(numerical_df.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,8))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'Blues', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
plt.xticks(rotation = 50)   # Aesthetic purposes
plt.yticks(rotation = 20)   # Aesthetic purposes
plt.show()


from scipy.stats import skew  # For skewness calculation

# Set up subplots
n_cols = 3  # Number of columns in the grid
n_rows = (len(numerical_df.columns) // n_cols) + 1

# Create a figure with subplots
plt.figure(figsize=(15, 5 * n_rows))  # Adjust size as needed

# Loop through numerical columns and plot KDE + skewness
for i, column in enumerate(numerical_df.columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.kdeplot(data=numerical_df, x=column, fill=True)
    
    # Calculate skewness
    skewness = skew(numerical_df[column].dropna())  # Handle NaN if needed
    skew_text = f'Skewness: {skewness:.2f}'
    
    # Add skewness as text in the plot
    plt.text(0.05, 0.9, skew_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'KDE of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Plot box plots
plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_df.columns, 1):
    plt.subplot(2, 4, i)  # Adjust subplot grid as needed
    sns.boxplot(data=train_df, y=feature, color='skyblue')
    plt.title(f'Box Plot of {feature}')
    plt.tight_layout()
plt.show()


# Rename the column 
train_df = train_df.rename(columns={'Temparature': 'Temperature'})
print(train_df.columns) 


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class FertilizerEnsemblePredictor:
    def __init__(self):
        self.xgb_models = []
        self.lgbm_models = []
        self.label_encoder = None
        self.cat_features = ['Soil Type', 'Crop Type']
        self.num_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
        self.preprocessor = None

    def feature_engineering(self, df):
        df = df.copy()
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1)
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1)
        df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
        df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
        df['Soil_Freq'] = df['Soil Type'].map(df['Soil Type'].value_counts(normalize=True))
        df['Crop_Freq'] = df['Crop Type'].map(df['Crop Type'].value_counts(normalize=True))
        return df

    def fit_xgb_lgbm(self, df, n_splits=3):
        df = df.copy()
        df = self.feature_engineering(df)

        X = df.drop(columns=['Fertilizer Name', 'id'])
        y = df['Fertilizer Name']

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.num_features + ['N_K_ratio', 'P_K_ratio', 'Temp_Moisture', 'Humidity_Moisture', 'Soil_Freq', 'Crop_Freq']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)
        ])
        X_preprocessed = self.preprocessor.fit_transform(X)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_preprocessed, y_encoded)):
            print(f"ğŸ”� Fold {fold + 1}")
            X_fold, y_fold = X_preprocessed[train_idx], y_encoded[train_idx]
            X_train, X_val, y_train, y_val = train_test_split(
                X_fold, y_fold, test_size=0.15, random_state=42, stratify=y_fold
            )

            # XGBoost model
            xgb_model = XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                learning_rate=0.045,
                max_depth=7,
                n_estimators=1200,
                subsample=0.8,
                colsample_bytree=0.6,
                colsample_bylevel=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            self.xgb_models.append(xgb_model)

            # LightGBM model
            lgbm_model = LGBMClassifier(
                objective='multiclass',
                num_class=num_classes,
                learning_rate=0.045,
                max_depth=7,
                n_estimators=1200,
                subsample=0.8,
                colsample_bytree=0.6,
                verbosity=-1
            )
            lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50)]
            )
            self.lgbm_models.append(lgbm_model)

            # Fold-level MAP@3
            val_proba = (xgb_model.predict_proba(X_val) + lgbm_model.predict_proba(X_val)) / 2
            top3 = np.argsort(val_proba, axis=1)[:, -3:][:, ::-1]
            def apk(actual, predicted, k=3):
                if actual in predicted[:k]:
                    return 1.0 / (predicted[:k].tolist().index(actual) + 1)
                return 0.0
            fold_score = np.mean([apk(a, p) for a, p in zip(y_val, top3)])
            print(f"ğŸ“Š Fold MAP@3: {fold_score:.5f}")

    def predict_top3(self, df):
        df = df.copy()
        df = self.feature_engineering(df)
        X = df.drop(columns=['id'], errors='ignore')
        X_preprocessed = self.preprocessor.transform(X)

        probas = np.zeros((X_preprocessed.shape[0], len(self.label_encoder.classes_)))
        for model in self.xgb_models + self.lgbm_models:
            probas += model.predict_proba(X_preprocessed)
        probas /= (len(self.xgb_models) + len(self.lgbm_models))

        top3 = np.argsort(probas, axis=1)[:, -3:][:, ::-1]
        top3_labels = self.label_encoder.inverse_transform(top3.ravel()).reshape(top3.shape)
        return top3_labels

    def evaluate_map3(self, df):
        df = df.copy()
        df = self.feature_engineering(df)
        X = df.drop(columns=['Fertilizer Name', 'id'])
        y_true = df['Fertilizer Name']
        y_encoded = self.label_encoder.transform(y_true)
        X_preprocessed = self.preprocessor.transform(X)

        probas = np.zeros((X_preprocessed.shape[0], len(self.label_encoder.classes_)))
        for model in self.xgb_models + self.lgbm_models:
            probas += model.predict_proba(X_preprocessed)
        probas /= (len(self.xgb_models) + len(self.lgbm_models))

        top3 = np.argsort(probas, axis=1)[:, -3:][:, ::-1]

        def apk(actual, predicted, k=3):
            if actual in predicted[:k]:
                return 1.0 / (predicted[:k].tolist().index(actual) + 1)
            return 0.0

        scores = [apk(a, p) for a, p in zip(y_encoded, top3)]
        return np.mean(scores)

    def predict_test_and_submit(self, test_df, filename="submission.csv"):
        test_df = test_df.copy()
        ids = test_df['id'].values
        top3 = self.predict_top3(test_df)

        submission = pd.DataFrame({
            'id': ids,
            'Fertilizer Name': [' '.join(row) for row in top3]
        })
        submission.to_csv(filename, index=False)
        print(f"âœ… Submission file saved to {filename}")

    def save(self, path_prefix='fertilizer_ensemble_model'):
        for i, model in enumerate(self.xgb_models):
            joblib.dump(model, f'{path_prefix}_xgb_fold{i}.pkl')
        for i, model in enumerate(self.lgbm_models):
            joblib.dump(model, f'{path_prefix}_lgbm_fold{i}.pkl')
        joblib.dump(self.label_encoder, f'{path_prefix}_label_encoder.pkl')
        joblib.dump(self.preprocessor, f'{path_prefix}_preprocessor.pkl')

    def load(self, path_prefix='fertilizer_ensemble_model', n_models=3):
        self.xgb_models = [joblib.load(f'{path_prefix}_xgb_fold{i}.pkl') for i in range(n_models)]
        self.lgbm_models = [joblib.load(f'{path_prefix}_lgbm_fold{i}.pkl') for i in range(n_models)]
        self.label_encoder = joblib.load(f'{path_prefix}_label_encoder.pkl')
        self.preprocessor = joblib.load(f'{path_prefix}_preprocessor.pkl')


# Train
predictor = FertilizerEnsemblePredictor()
predictor.fit_xgb_lgbm(train_df, n_splits=3)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


test_df = test_df.rename(columns={'Temparature': 'Temperature'})
print(test_df.columns)


predictor.predict_test_and_submit(test_df)

