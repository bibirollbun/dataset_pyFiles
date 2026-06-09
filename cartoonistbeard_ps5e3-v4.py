MODEL = "XGB"
V = 3


import pandas as pd
import numpy as np


from data_analysis import load_data  # Import the class


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.drop(columns=['id'],inplace=True)
test.drop(columns=['id'],inplace=True)


display(train.head())
display(test.head())


def FE(data):
    df = data.copy()

    # Check for missing columns
    required_cols = ['day', 'maxtemp', 'mintemp', 'temparature', 'winddirection', 'windspeed',
                     'dewpoint', 'humidity', 'cloud', 'sunshine']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Convert day to datetime
    df['day'] = pd.to_datetime(df['day'], errors='coerce')

    # Time-based Features
    df['month'] = df['day'].dt.month.fillna(-1).astype("category")
    df['week'] = df['day'].dt.dayofweek.fillna(-1).astype("category")
    df['is_weekend'] = df['week'].isin([5, 6]).astype(int)
    df['season'] = df['month'].apply(lambda x: 'Spring' if 0 <= x <= 3 else 
                                              'Summer' if 3 < x <= 5 else 
                                              'Autumn' if 6 < x <= 11 else 'Winter')

    # **Temperature-Based Features**
    df['temp_gap'] = df['maxtemp'] - df['mintemp']
    df['temp_avg'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_dev'] = df['temparature'] - df['temp_avg']
    
    # Temperature Variability
    df['temp_variability'] = df['temp_gap'] / df['temp_avg']

    # Wind Features
    df['wind_direction_label'] = df['winddirection'].apply(lambda x: 'NE' if 0 <= x <= 75 else 
                                                                     'SE' if 75 < x <= 150 else 
                                                                     'SW' if 150 < x <= 225 else 'NW')
    df['wind_cos'] = np.cos(np.radians(df['winddirection'].fillna(0)))
    df['wind_sin'] = np.sin(np.radians(df['winddirection'].fillna(0)))
    df['wind_x'] = df['windspeed'].fillna(0) * df['wind_cos']
    df['wind_y'] = df['windspeed'].fillna(0) * df['wind_sin']
    
    # Wind Intensity
    df['wind_intensity'] = np.sqrt(df['wind_x']**2 + df['wind_y']**2)

    # Humidity and Dewpoint Interactions
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']
    df['Humidity_Dewpoint_Diff'] = df['humidity'] - df['dewpoint']
    df['High_Humidity_Flag'] = (df['Humidity_Dewpoint_Diff'] >= 90).astype(int)

    # **Dewpoint-Based Features**
    df['dewpoint_variation'] = df['dewpoint_depression'] / df['humidity']
    
    # Cloud coverage features
    df['cloud_category'] = pd.cut(df['cloud'], bins=[0, 30, 70, 100], labels=['Clear', 'Partly Cloudy', 'Overcast'])
    
    # Sunshine-related features
    df['inverse_sunshine'] = 1 / (df['sunshine'].replace(0, 0.001))
    df['daylight_ratio'] = df['sunshine'] / 24
    df['sun_cloud_ratio'] = df['sunshine'] / (df['cloud'].replace(0, 0.001))

    # **Cloud & Sunshine Interaction**
    df['cloud_sun_interaction'] = df['cloud'] * df['sunshine']
    
    # **Rainfall Probability Estimators**
    df['humidity_rainfall_factor'] = df['humidity'] * df['dewpoint']
    df['temp_humidity_factor'] = df['temp_avg'] * df['humidity']
    df['wind_rainfall_factor'] = df['wind_intensity'] * df['cloud']

    # Feature Interactions
    df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    df['dewpoint_humidity_interaction'] = df['dewpoint'] * df['humidity']
    df['wind_cloud_interaction'] = df['windspeed'] * df['cloud']
    
    # Drop Original Day Column
    df.drop(columns='day', inplace=True)

    return df


train = FE(train)
test = FE(test)


display(train.head())
display(test.head())


train_loader = load_data(file_df = train)
test_loader = load_data(file_df = test)


train_loader.summarize()


a_df = train_loader.feature_target_dependence(target_col="rainfall")
a_df


def analyze_feature_dependence(df):
    """
    Analyzes the p-values in the feature-target dependence summary
    and categorizes features based on significance.

    Parameters:
    df (pd.DataFrame): DataFrame containing feature dependence results.

    Returns:
    dict: Dictionary containing categorized features.
    """
    # Convert p-value to numeric (handling errors)
    df['p-value'] = pd.to_numeric(df['p-value'], errors='coerce')
    
    # Define significance categories
    strong_dependence = df[df['p-value'] <= 0.01]['Feature'].tolist()
    moderate_dependence = df[(df['p-value'] > 0.01) & (df['p-value'] <= 0.05)]['Feature'].tolist()
    weak_dependence = df[(df['p-value'] > 0.05) & (df['p-value'] <= 0.1)]['Feature'].tolist()
    no_dependence = df[df['p-value'] > 0.1]['Feature'].tolist()
    missing_data = df[df['p-value'].isna()]['Feature'].tolist()
    
    # Create a summary dictionary
    summary = {
        "Strong Dependence (p ≤ 0.01)": strong_dependence,
        "Moderate Dependence (0.01 < p ≤ 0.05)": moderate_dependence,
        "Weak Dependence (0.05 < p ≤ 0.1)": weak_dependence,
        "No Dependence (p > 0.1)": no_dependence,
        "Missing/Invalid Data": missing_data
    }
    
    return summary


summary_results = analyze_feature_dependence(a_df.data if hasattr(a_df, 'data') else a_df)
for category, features in summary_results.items():
    print(f"{category}: {features}")


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
#from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


target = 'rainfall'
Features = summary_results['Strong Dependence (p ≤ 0.01)'] + summary_results['Moderate Dependence (0.01 < p ≤ 0.05)']
print(f"We have {len(Features)} Features")
print(Features)


num_cols = [col for col in Features if train[col].dtype in ['float64','int64']]
cat_cols = [col for col in Features if train[col].dtype in ['O','category']]


models = {
    "Logistic Regression": LogisticRegression(random_state=42,max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(random_state=42, max_iter=100, hidden_layer_sizes=(10)),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=6, enable_categorical=True),
    "CatBoost": CatBoostClassifier(random_state=42, iterations=100, learning_rate=0.14, depth=6, verbose=0),
    "Extra_Trees": ExtraTreesClassifier(random_state=42)
}


def train_and_evaluate_models(models, X, y, test_size=0.2, random_state=42):
    """
    Trains each model in the dictionary and returns ROC AUC scores.
    
    Parameters:
    models (dict): Dictionary of models
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target vector
    test_size (float): Proportion of dataset for testing
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary containing model names and their respective ROC AUC scores.
    """
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Define preprocessor for models that need explicit encoding
    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ], remainder='passthrough')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Dictionary to store scores
    scores = {}

    for name, model in models.items():
        print(f"Training {name}...")

        if name == "CatBoost":
            # Ensure categorical columns are correctly assigned
            cat_features = [X_train.columns.get_loc(col) for col in categorical_cols]  # Get column indices
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            model.fit(X_train_processed, y_train, cat_features=cat_features)
        
        elif name == "XGBoost":
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            X_train_processed[categorical_cols] = X_train_processed[categorical_cols].astype("category")
            X_test_processed[categorical_cols] = X_test_processed[categorical_cols].astype("category")
            model.fit(X_train_processed, y_train)

        else:
            # Apply OneHotEncoding for other models
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            model.fit(X_train_processed, y_train)

        # Get probabilities for ROC AUC
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]  
        else:
            y_pred_proba = model.decision_function(X_test_processed)  

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        scores[name] = roc_auc
        print(f"{name} ROC AUC: {roc_auc:.4f}")

    return scores


scores = train_and_evaluate_models(models,train.drop(columns=['rainfall']),train['rainfall'],) 


scores = train_and_evaluate_models(models,train.drop(columns=['rainfall']),train['rainfall']) 


import gc


def model_trainer(model, X, y, test, n_splits=10, random_state=42):
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    roc_aucs = []
    test_pred = np.zeros(len(test))
    oof_train_preds = np.zeros(len(y))

    model_name = model[-1].__class__.__name__ if isinstance(model, Pipeline) else model.__class__.__name__
    
    print("="*72)
    print(f"Training {model_name}")
    print("="*72,sep='\n')
    for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        try:
            y_pred_proba = model_clone.predict_proba(X_test)[:,1]
            test_pred += model_clone.predict_proba(test)[:, 1]
        except:
            y_pred_proba = model_clone.predict(X_test)
            test_pred += model_clone.predict(test)
        oof_train_preds[test_idx] = y_pred_proba
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        roc_aucs.append(roc_auc)
        print(f"Fold {fold+1} --> ROC_AUC Score: {roc_auc:.6f}")
        
        del model_clone, X_train, X_test, y_train, y_test
        gc.collect()

    print(f"\nAverage Fold ROC_AUC Score: {np.mean(roc_aucs):.6f} \xb1 {np.std(roc_aucs):.6f}\n")
    return test_pred/skfold.get_n_splits(), oof_train_preds


def convert_to_string(df):
    df_cat = df.copy()
    df_cat = df_cat.fillna(0)
    for col in features:
        df_cat[col] = df_cat[col].astype('string')
    return df_cat


skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


X = train.drop(target, axis=1)
y = train[target].ravel()


train_preds = {}
test_preds = {}


X_xgb = X.copy()
X_xgb[categorical_features] = X_xgb[categorical_features].astype('category')

test_xgb = test.copy()
test_xgb[categorical_features] = test_xgb[categorical_features].astype('category')

oof_preds = []
oof_aucs = []
oof_train_preds = np.zeros(len(y))

