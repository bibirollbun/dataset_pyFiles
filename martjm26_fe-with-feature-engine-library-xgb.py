!pip install feature_engine


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


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").drop(columns=["id"])
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv").drop(columns=["id"])
display(train.head(), train.info(), train.describe().T)
print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")


from feature_engine.creation import (
    MathFeatures, 
    CyclicalFeatures
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    DecisionTreeDiscretiser
)
from feature_engine.encoding import (
    OneHotEncoder,
    OrdinalEncoder,
    CountFrequencyEncoder
)
from feature_engine.transformation import (
    LogTransformer,
    PowerTransformer,
    YeoJohnsonTransformer
)
from feature_engine.selection import (
    DropCorrelatedFeatures,
    SelectByShuffling
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_agricultural_features(df):
    """
    Create domain-specific features for fertilizer prediction using Feature-engine
    """
    
    # 1. MATHEMATICAL COMBINATIONS - NPK ratios and interactions
    math_combiner = MathFeatures(
        variables=['Nitrogen', 'Potassium', 'Phosphorous'],
        func=['sum', 'prod', 'mean', 'std'],
        new_variables_names=['NPK_sum', 'NPK_product', 'NPK_mean', 'NPK_std']
    )
    
    # 2. RATIO FEATURES - Will be created manually using pandas operations
    
    # 3. POLYNOMIAL FEATURES - Will be created manually using pandas operations
    
    # 4. DISCRETIZATION - Create categorical versions of continuous variables
    # This helps capture non-linear patterns and interactions
    temp_discretizer = EqualFrequencyDiscretiser(
        variables=['Temparature'], 
        q=5,  # Create 5 temperature ranges
    )
    
    humidity_discretizer = EqualFrequencyDiscretiser(
        variables=['Humidity'], 
        q=4,  # Create 4 humidity levels
    )
    
    npk_discretizer = EqualFrequencyDiscretiser(
        variables=['Nitrogen', 'Potassium', 'Phosphorous'],
        q=4,  # Create 4 levels for each nutrient (low, medium-low, medium-high, high)
    )
    
    # 5. ENVIRONMENTAL STRESS INDICATORS - Will be created manually
    
    # 6. NUTRIENT DEFICIENCY INDICATORS
    # Create features indicating potential nutrient imbalances
    nutrient_balance = MathFeatures(
        variables=['Nitrogen', 'Potassium', 'Phosphorous'],
        func=['log', 'log', 'log'],
        new_variables_names=['log_Nitrogen', 'log_Potassium', 'log_Phosphorous']
    )
    
    # Apply transformations
    df_transformed = df.copy()
    
    # Mathematical combinations
    df_transformed = math_combiner.fit_transform(df_transformed)
    
    # Create ratio features manually
    df_transformed['N_K_ratio'] = df_transformed['Nitrogen'] / (df_transformed['Potassium'] + 1)  # Add 1 to avoid division by zero
    df_transformed['N_P_ratio'] = df_transformed['Nitrogen'] / (df_transformed['Phosphorous'] + 1)
    df_transformed['K_P_ratio'] = df_transformed['Potassium'] / (df_transformed['Phosphorous'] + 1)
    df_transformed['Temp_Humidity_ratio'] = df_transformed['Temparature'] / (df_transformed['Humidity'] + 1)
    df_transformed['Humidity_Moisture_ratio'] = df_transformed['Humidity'] / (df_transformed['Moisture'] + 1)
    
    # Create polynomial features manually
    df_transformed['Temp_squared'] = df_transformed['Temparature'] ** 2
    df_transformed['Humidity_squared'] = df_transformed['Humidity'] ** 2
    df_transformed['Moisture_squared'] = df_transformed['Moisture'] ** 2
    
    # Environmental stress indicators - manual creation
    df_transformed['Temp_Humidity_diff'] = df_transformed['Temparature'] - df_transformed['Humidity']
    df_transformed['Moisture_Humidity_diff'] = df_transformed['Moisture'] - df_transformed['Humidity']
    
    # Handle log transformation (add small constant to avoid log(0))
    for col in ['Nitrogen', 'Potassium', 'Phosphorous']:
        df_transformed[f'log_{col}'] = np.log(df_transformed[col] + 1)
    
    # Discretization
    df_transformed = temp_discretizer.fit_transform(df_transformed)
    df_transformed = humidity_discretizer.fit_transform(df_transformed)
    df_transformed = npk_discretizer.fit_transform(df_transformed)
    
    return df_transformed, {
        'math_combiner': math_combiner,
        'temp_discretizer': temp_discretizer,
        'humidity_discretizer': humidity_discretizer,
        'npk_discretizer': npk_discretizer
    }


def create_encoding_pipeline(df):
    """
    Create encoding pipeline for categorical variables
    """
    
    # Count frequency encoding for high-cardinality categoricals
    # This is often better than one-hot for many categories
    freq_encoder = CountFrequencyEncoder(
        variables=['Soil Type', 'Crop Type'],
    )
    
    # Check which discretized features actually exist in the dataframe
    discretized_features = []
    potential_discretized = ['Temp_range', 'Humidity_level', 'N_level', 'K_level', 'P_level']
    
    for feature in potential_discretized:
        if feature in df.columns:
            discretized_features.append(feature)
    
    # Only create one-hot encoder if discretized features exist
    if discretized_features:
        onehot_encoder = OneHotEncoder(
            variables=discretized_features,
            drop_last=True  # Avoid multicollinearity
        )
        
        return Pipeline([
            ('frequency_encoding', freq_encoder),
            ('onehot_encoding', onehot_encoder)
        ])
    else:
        # If no discretized features, just use frequency encoding
        return Pipeline([
            ('frequency_encoding', freq_encoder)
        ])


def create_feature_selection_pipeline():
    """
    Create feature selection pipeline to remove redundant features
    """
    
    # Remove highly correlated features
    correlation_filter = DropCorrelatedFeatures(
        threshold=0.95,
        method='pearson'
    )
    
    return correlation_filter


def create_complete_feature_pipeline():
    """
    Create a complete feature engineering pipeline
    """
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('correlation_filter', DropCorrelatedFeatures(threshold=0.95))
    ])
    
    return pipeline


# Create new features
train_with_features, transformers = create_agricultural_features(train)


# Create encoding pipeline
encoding_pipeline = create_encoding_pipeline(train_with_features)


train_with_features.info()


# Apply encoding
train_encoded = encoding_pipeline.fit_transform(train_with_features)


# Feature selection
#selector = create_feature_selection_pipeline()
#train_final = selector.fit_transform(train_encoded)
train_final = train_encoded


print(f"Original features: {train.shape[1]}")
print(f"After feature engineering: {train_encoded.shape[1]}")
#print(f"After feature selection: {train_final.shape[1]}")

# New features created:
new_features = [
    'NPK_sum', 'NPK_product', 'NPK_mean', 'NPK_std',
    'N_K_ratio', 'N_P_ratio', 'K_P_ratio', 'Temp_Humidity_ratio', 'Humidity_Moisture_ratio',
    'Temp_squared', 'Humidity_squared', 'Moisture_squared',
    'Temp_Humidity_diff', 'Moisture_Humidity_diff',
    'log_Nitrogen', 'log_Potassium', 'log_Phosphorous'
]


display(train_final.info(), train_final.head(), train_final.shape)


# Create new features
test_with_features, transformers = create_agricultural_features(test)

# Create encoding pipeline
encoding_pipeline = create_encoding_pipeline(test_with_features)


test_with_features.info()


# Apply encoding
test_encoded = encoding_pipeline.fit_transform(test_with_features)

# Feature selection
# selector = create_feature_selection_pipeline()
# test_final = selector.fit_transform(test_encoded)
test_final = test_encoded


test_final.info()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_final["Fertilizer Name"] = le.fit_transform(train["Fertilizer Name"])
train_final.head()


X = train_final.drop(columns=["Fertilizer Name"])
y = train_final["Fertilizer Name"]
X_test = test_final


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

FOLDS = 5
#skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape = (len(train_final) ,y.nunique()))
pred_prob = np.zeros(shape = (len(test_final),y.nunique()))

xgb_model = XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state=13,
    enable_categorical=True,
    tree_method='hist',     
    device='cuda'           
)

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    xgb_model.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 0)
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob +=xgb_model.predict_proba(X_test)

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]  
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"âœ… FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")


'''
WITH FEATURE SELECTION
############### 1 ###############
âœ… FOLD 1: MAP@3 Score: 0.32721
############### 2 ###############
âœ… FOLD 2: MAP@3 Score: 0.32837
############### 3 ###############
âœ… FOLD 3: MAP@3 Score: 0.32767
############### 4 ###############
âœ… FOLD 4: MAP@3 Score: 0.32684
############### 5 ###############
âœ… FOLD 5: MAP@3 Score: 0.32645


WITHOUT FEATURE SELECTION
############### 1 ###############
âœ… FOLD 1: MAP@3 Score: 0.32731
############### 2 ###############
âœ… FOLD 2: MAP@3 Score: 0.32789
############### 3 ###############
âœ… FOLD 3: MAP@3 Score: 0.32787
############### 4 ###############
âœ… FOLD 4: MAP@3 Score: 0.32679
############### 5 ###############
âœ… FOLD 5: MAP@3 Score: 0.32780
'''


from xgboost import plot_importance
from matplotlib import pyplot

plot_importance(xgb_model)
pyplot.show()


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")




