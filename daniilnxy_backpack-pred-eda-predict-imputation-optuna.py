import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import torch as t
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor  
from optuna.pruners import SuccessiveHalvingPruner
import xgboost as xgb
warnings.filterwarnings("ignore", category=FutureWarning)



train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv').drop(columns=['id'], axis=1)
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv').drop(columns=['id'], axis=1)
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_df.head()


train_df.duplicated().value_counts()


train_df.isna().sum()


train_df.nunique()


train_df.dtypes


train_df.shape


test_df.head()


test_df.duplicated().value_counts()


test_df.info()


test_df.isna().sum()


test_df.shape


sub_df


sub_df.shape


def count_plot_graphic(data, column):
    fig_size = (6,6)
    
    plt.figure(figsize=fig_size)
    sns.set_style('whitegrid')
    sns.set_palette('pastel')
    
    ax = sns.countplot(x=column, data=data)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.title(f'Count of every {column} for a backpack')
    plt.show()


def pie_plot(data, column):
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    
    counts = data[column].value_counts(dropna=True)
    colors = sns.color_palette('pastel', n_colors=len(counts))
    

    plt.rcParams['font.size'] = 8  
    
    ax.pie(
        counts, 
        labels=counts.index, 
        colors=colors, 
        autopct='%.2f%%', 
        startangle=140,
        pctdistance=0.8, 
        labeldistance=1.05  
    )
    
    ax.set_title(f'Pie for "{column}" column', fontsize=12)
    ax.set_aspect("equal")
    
    plt.margins(0.01) 
    
    plt.show()


def boxplot_graphic(data, x_col, y_col):
    sns.color_palette('pastel')
    sns.boxplot(data=train_df,
                x=x_col,
                y=y_col,
                )
    plt.show()


def hist_graphic(data, x_col, color):
    cleaned_data = data[[x_col]].replace([np.inf, -np.inf], np.nan)
    
    sns.histplot(data=cleaned_data, x=x_col, kde=True, color=color)
    
    plt.title(f'Histogram for {x_col}')
    plt.show()



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def missing_percentage(data):
    missing_data = data.isnull().mean() * 100
    return missing_data

def plot_missing_percentage(data):
    missing_data = missing_percentage(data)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=missing_data.index, y=missing_data.values, palette='pastel')
    
    for i, value in enumerate(missing_data.values):
        ax.text(i, value + 1, f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.title('Percentage of Missing Values for Each Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45) 

    plt.ylim(0, missing_data.max() + 10)  
    plt.show()



object_type_columns = train_df.select_dtypes(include=['object']).columns
object_type_columns


for col in object_type_columns:
    count_plot_graphic(train_df, col)


float_type_columns = train_df.select_dtypes(include=['float64']).columns

pastel_colors = sns.color_palette('pastel', len(float_type_columns))

for i, float_col in enumerate(float_type_columns):
    hist_graphic(train_df, float_col, pastel_colors[i])



for object_col in object_type_columns:
    pie_plot(train_df, object_col)


float_type_columns = train_df.select_dtypes(include=['float64']).columns
y_col = 'Brand'


for x_col in float_type_columns:
    boxplot_graphic(train_df, x_col, y_col)


df_encoded = pd.get_dummies(train_df)
correlation_matrix = df_encoded.corr()
price_corr = correlation_matrix['Price'].sort_values(ascending=False)[1:]


plt.figure(figsize=(8, 6))
sns.barplot(y=price_corr.index, x=price_corr.values, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Feature correlation with price")
plt.show()


plot_missing_percentage(train_df)


object_type_columns = test_df.select_dtypes(include=['object']).columns


for object_col in object_type_columns:
    count_plot_graphic(test_df, object_col)


float_type_columns = test_df.select_dtypes(include=['float64']).columns
pastel_colors = sns.color_palette('pastel', len(float_type_columns))

for i, float_col in enumerate(float_type_columns):
    hist_graphic(test_df, float_col, pastel_colors[i])


for object_col in object_type_columns:
    pie_plot(test_df, object_col)


float_type_columns = test_df.select_dtypes(include=['float64']).columns
y_col = 'Brand'
for x_col in float_type_columns:
    boxplot_graphic(test_df, x_col, y_col)


plot_missing_percentage(test_df)


class MissingValueImputer:
    def __init__(self, df, candidate_features, top_n=3, mi_threshold=0.01, sample_size=0.5):
        self.df = df.copy()
        self.df_original = df.copy() 
        self.candidate_features = candidate_features
        self.top_n = top_n
        self.mi_threshold = mi_threshold
        self.sample_size = sample_size
        self.log = []


    def select_best_predictors(self, target_col):
        data = self.df_original[self.df_original[target_col].notna()].sample(frac=self.sample_size, random_state=42)

        if data[target_col].dtype in ['int64', 'float64']:
            is_numeric_target = True
            y = pd.qcut(data[target_col], q=5, labels=False, duplicates='drop')  
        else:
            is_numeric_target = False
            y = LabelEncoder().fit_transform(data[target_col].astype(str)) 

        mi_scores = {}

        for feature in self.candidate_features:
            if feature == target_col:
                continue

            x_feature = data[feature].copy()

        
            if x_feature.dtype == 'object':
                x_feature.fillna(x_feature.mode()[0], inplace=True)
                x_feature = LabelEncoder().fit_transform(x_feature.astype(str))
            else:
                x_feature.fillna(x_feature.median(), inplace=True)
                x_feature = x_feature.to_numpy()

            if is_numeric_target:
                mi = mutual_info_regression(x_feature.reshape(-1, 1), y)
            else:
                mi = mutual_info_classif(x_feature.reshape(-1, 1), y, discrete_features=True)

            mi_scores[feature] = mi[0]

      
        sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        
    
        print(f"ğŸ”� Top predictors for {target_col}:")
        for feature, score in sorted_features[:3]:
            print(f"  - {feature}: {score:.4f}")

        best_features = [feat for feat, score in sorted_features if score >= self.mi_threshold][:self.top_n]

        return best_features

    def impute_missing_values(self, col):
        print(f"\nğŸ”¹ Starting imputation for {col}...")
   
        if col == 'Weight Capacity (kg)':
            median_value = self.df_original[col].median()
            self.df[col].fillna(median_value, inplace=True)
            self.log.append(f"{col} filled with median: '{median_value}'")
            print(f"â�Œ No good predictors found for {col}. Filling missing values with median: '{median_value}'")
            return None, None

        best_features = self.select_best_predictors(col)

        if not best_features:
            mode_value = self.df_original[col].mode()[0]
            self.df[col].fillna(mode_value, inplace=True)
            print(f"â�Œ No good predictors found for {col}. Filling missing values with mode: '{mode_value}'")
            self.log.append(f"{col} filled with mode: '{mode_value}'")
            return None, None

        print(f"âœ… Best predictors for {col}: {best_features}")
        
        known_data = self.df[self.df[col].notna()].copy()
        unknown_data = self.df[self.df[col].isna()].copy()
        
        target_encoder = LabelEncoder()
        target_feature = f'{col}_encoded'
        known_data[target_feature] = target_encoder.fit_transform(known_data[col])
        
        for predictor in best_features:
            if known_data[predictor].dtype == 'object':
                mode_value = self.df_original[predictor].mode()[0]
                known_data[predictor].fillna(mode_value, inplace=True)
                unknown_data[predictor].fillna(mode_value, inplace=True)
            else:
                median_value = self.df_original[predictor].median()
                known_data[predictor].fillna(median_value, inplace=True)
                unknown_data[predictor].fillna(median_value, inplace=True)
        
        encoders = {}
        for predictor in best_features:
            if known_data[predictor].dtype == 'object':
                enc = LabelEncoder()
                known_data[predictor] = enc.fit_transform(known_data[predictor])
                unknown_data[predictor] = enc.transform(unknown_data[predictor])
                encoders[predictor] = enc
        
        X = known_data[best_features]
        y = known_data[target_feature]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_val, y_val)
        print(f"ğŸ“Š Validation accuracy for {col}: {accuracy:.2f}")

        predicted_encoded = model.predict(unknown_data[best_features])
        unknown_data[target_feature] = predicted_encoded
        unknown_data[col] = target_encoder.inverse_transform(predicted_encoded)
        
        self.df.update(unknown_data)
        self.log.append(f"âœ… {col} imputed (accuracy: {accuracy:.2f})")

        return col, accuracy

    def impute_all_missing(self):
        missing_columns = [col for col in self.df.columns if self.df[col].isna().sum() > 0]

        print(f"\nğŸ”§ Starting imputation for all missing values. Found {len(missing_columns)} columns with missing values.")

        for col in missing_columns:
            self.impute_missing_values(col)
        
        print("\nğŸ“œ Imputation log:")
        for log_entry in self.log:
            print(log_entry)

        return self.df




from sklearn.feature_selection import mutual_info_classif
candidate_features = ['Style', 'Material', 'Size', 'Color', 
                      'Laptop Compartment', 'Waterproof', 'Compartments']
imputer = MissingValueImputer(train_df, candidate_features, top_n=3, mi_threshold=0.01, sample_size=0.1)

train_df = imputer.impute_all_missing()


candidate_features = ['Style', 'Material', 'Size', 'Color', 
                      'Laptop Compartment', 'Waterproof', 'Compartments']
imputer = MissingValueImputer(test_df, candidate_features, top_n=3, mi_threshold=0.01, sample_size=0.1)

test_df = imputer.impute_all_missing()



def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    outliers = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)  
        Q3 = df[col].quantile(0.75)  
        IQR = Q3 - Q1  

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    return outliers

outliers = detect_outliers(train_df)

for feature, data in outliers.items():
    if len(data) > 0:
        print(f"\nğŸš¨ Outliers in column: {feature}")
        print(data[[feature]])
    else:
        print(f"\nâœ… No outliers in column: {feature}")

train_df_cleaned = train_df.copy()

for col in train_df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = train_df_cleaned[col].quantile(0.25)
    Q3 = train_df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_df_cleaned = train_df_cleaned[(train_df_cleaned[col] >= lower_bound) & (train_df_cleaned[col] <= upper_bound)]

print(f"\nOriginal data shape: {train_df.shape}")
print(f"Data shape after outlier removal: {train_df_cleaned.shape}")



def cramers_v(cat1, cat2):
    confusion_matrix = pd.crosstab(cat1, cat2)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def correlation_check(df, target_col='Price', threshold=0.8):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    corr_matrix = df[numerical_cols].corr()

    target_corr = corr_matrix[target_col]
    print(f"ğŸ“Š Correlation of numerical features with {target_col}:\n", target_corr)

    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    
    print(f"\n ğŸš¨ Highly correlated numerical features (correlation > {threshold}):\n", correlated_features)

    cat_correlation = {}
    for col in categorical_cols:
        cat_correlation[col] = cramers_v(df[col], df[target_col])

    cat_correlation = pd.Series(cat_correlation).sort_values(ascending=False)
    print(f"\nğŸ“ˆ Categorical feature correlation with {target_col} (CramÃ©r's V):\n", cat_correlation)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

    df_dropped_corr = df.drop(columns=correlated_features)

    return df_dropped_corr

train_df = correlation_check(train_df, target_col='Price')



def feature_engineering_func(df):
    df['Size_Style_Interaction'] = df['Size'] + "_" + df['Style']
    df['Waterproof_Laptop_Compartment'] = df['Waterproof'].astype(str) + "_" + df['Laptop Compartment'].astype(str)
    brand_popularity = df['Brand'].value_counts()
    df['Brand_Popularity'] = df['Brand'].map(brand_popularity)
    return df


train_df = feature_engineering_func(train_df)
test_df = feature_engineering_func(test_df)


X = train_df.drop(columns=['Price'])
y= train_df['Price']


from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2,
                                                   random_state=42)


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def data_transformation_scaling(X_train, X_val, X_test):
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore')) 
    ])
    
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())                
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ]
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)  
    X_test_transformed = preprocessor.transform(X_test)  

    return X_train_transformed, X_val_transformed, X_test_transformed



 X_train_transformed, X_val_transformed, X_test_transformed = data_transformation_scaling(X_train, X_val, test_df)


X_train_transformed.toarray()


def hyperparameter_optimization(
    model_class,         
    param_sampler,   
    X_train, y_train,     
    X_val, y_val,       
    fit_params=None,    
    n_trials=30,          
    direction='minimize' 
):
    if fit_params is None:
        fit_params = {}

    def objective(trial):
        params = param_sampler(trial)
        model = model_class(**params)
        eval_set = [(X_val, y_val)]

        model.fit(X_train, y_train, eval_set=eval_set, verbose=False, **fit_params)
        
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
        trial.report(rmse, step=1)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return rmse

    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(direction=direction, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    print(f"  RMSE: {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = study.best_trial.params
    best_model = model_class(**best_params)

    if model_class == CatBoostRegressor:
        eval_set = [(X_val, y_val)]
    elif model_class == xgb.XGBRegressor:
        eval_set = [(X_val, y_val)]
    else:
        eval_set = None

    best_model.fit(X_train, y_train, eval_set=eval_set, verbose=False, **fit_params)
    
    return best_model, best_params, study


def catboost_param_sampler(trial):
    return {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': 2000,
        'depth': trial.suggest_int('depth', 3, 10),
        'random_strength': 0,
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'task_type': 'GPU',  
        'random_seed': 42,
        'verbose': False
    }


fit_params = {
    'early_stopping_rounds': 100,
}



catboost_model, catboost_params, cat_study = hyperparameter_optimization(
    model_class=CatBoostRegressor,
    param_sampler=catboost_param_sampler,
    X_train=X_train_transformed,
    y_train=y_train,
    X_val=X_val_transformed,
    y_val=y_val,
    fit_params=fit_params,
    n_trials=40
)



def xgboost_param_sampler(trial):
    return {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': 2000,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'random_state': 42,
        'tree_method': 'hist',  
        'device': 'cuda', 
    }

fit_params = {
    'early_stopping_rounds': 100,
}

xgboost_model, xgboost_params, xgboost_study = hyperparameter_optimization(
    model_class=XGBRegressor,
    param_sampler=xgboost_param_sampler,
    X_train=X_train_transformed,
    y_train=y_train,
    X_val=X_val_transformed,
    y_val=y_val,
    fit_params=fit_params,
    n_trials=20
)



y_test = catboost_model.predict(X_test_transformed)

sub_df['Price'] = y_test
sub_df


sub_df.to_csv('submission.csv', index=False)

