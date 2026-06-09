import pandas as pd
import seaborn as sns


df = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')


df.head()


sns.kdeplot(df.Price)


def rmse(y_true, y_pred):
    return np.sqrt(np.sum(((y_true-y_pred)**2)/len(y_true)))


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')


X_train = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
# X_train = df
X_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
subm = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

X_train_original = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
X_train.drop('id', axis=1, inplace=True)
X_test.drop('id', axis=1, inplace=True)



X_train


def add_is_missing_row(df, col):
    """Adds a binary column indicating missing values."""
    df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    return df

# Define imputation strategies
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)", "Compartments"]

# Fill categorical missing values with mode (most frequent value)
for col in categorical_features:
    X_train = add_is_missing_row(X_train, col)
    mode_value = X_train[col].mode()
    if not mode_value.empty:
        X_train[col].fillna(mode_value[0], inplace=True)

    X_test = add_is_missing_row(X_test, col)
    mode_value_test = X_test[col].mode()
    if not mode_value_test.empty:
        X_test[col].fillna(mode_value_test[0], inplace=True)

# Fill numerical missing values with median
for col in numerical_features:
    X_train = add_is_missing_row(X_train, col)
    X_train[col].fillna(X_train[col].mean(), inplace=True)

    X_test = add_is_missing_row(X_test, col)
    X_test[col].fillna(X_test[col].mean(), inplace=True)


X_train





def perform_feature_engineering(df):
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Num'] = df['Size'].map(size_mapping)
    df['Compartments_per_Size'] = df['Compartments'] / df['Size_Num']    
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments'] 
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 'Large' else 0)

    df['Brand_Material'] = df['Brand']+'_'+df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Brand_Style'] = df['Brand']+ '_' + df['Style']
    df['Brand_Color'] = df['Brand'] + '_' + df['Color']



    # df['Weight Capacity (lb)'] = df['Weight Capacity (kg)']/0.45359237
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)']/df['Weight Capacity (kg)'].max()
    return df


X_train_fe = perform_feature_engineering(X_train.copy())
X_test_fe = perform_feature_engineering(X_test.copy())


X_train_fe


cat_features = X_train_fe.select_dtypes(include='object').columns.tolist()
num_features = X_train_fe.select_dtypes(exclude='object').columns.tolist()


y_train = X_train_fe.Price
X_train_fe.drop('Price', axis=1, inplace=True)


from cuml.preprocessing import TargetEncoder
X_train_encoded_df = X_train_fe.copy()
X_test_encoded_df = X_test_fe.copy()


categorical_columns = X_train_encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_columns += ['Weight Capacity (kg)']
TE = TargetEncoder(n_folds=20, smooth=25, split_method='random', stat='mean')
for c in categorical_columns:
    print(c)
    X_train_encoded_df[c] = TE.fit_transform(X_train_encoded_df[c], y_train)
    X_test_encoded_df[c] = TE.transform(X_test_encoded_df[c])
# s = np.sqrt(np.mean( (train.Price-train.pred)**2.0 ) )
# print(f"Validation RSME using Target Encode Weight Capacity = {s}")


import numpy as np

def add_features(df):
    # Ratio features
    df['Weight_Ratio_Material'] = df['Weight Capacity (kg)'] / df['Material']
    df['Weight_Ratio_Brand'] = df['Weight Capacity (kg)'] / df['Brand']
    df['Weight_Ratio_Size'] = df['Weight Capacity (kg)'] / df['Size']
    df['Weight_Ratio_Style'] = df['Weight Capacity (kg)'] / df['Style']
    df['Weight_Ratio_Color'] = df['Weight Capacity (kg)'] / df['Color']
    
    df['Weight_Brand_Material_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Material']
    df['Weight_Brand_Size_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Size']
    df['Weight_Brand_Style_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Style']
    df['Weight_Brand_Color_Ratio'] = df['Weight Capacity (kg)'] / df['Brand_Color']

    # Sum features
    df['Weight_Plus_Size'] = df['Weight Capacity (kg)'] + df['Size']
    df['Brand_Material_Sum'] = df['Brand'] + df['Material']
    df['Weight_Plus_Brand'] = df['Weight Capacity (kg)'] + df['Brand']

    # Difference features
    df['Weight_Minus_Size'] = df['Weight Capacity (kg)'] - df['Size']
    df['Brand_Material_Diff'] = df['Brand'] - df['Material']
    df['Weight_Minus_Brand'] = df['Weight Capacity (kg)'] - df['Brand']

    # Product features
    df['Weight_Size_Product'] = df['Weight Capacity (kg)'] * df['Size']
    df['Brand_Material_Product'] = df['Brand'] * df['Material']
    df['Weight_Material_Product'] = df['Brand']*df['Weight Capacity (kg)']
    df['Weight_Product_Brand'] = df['Weight Capacity (kg)'] * df['Brand']

    # Power & Root features
    df['Weight_Squared'] = df['Weight Capacity (kg)'] ** 2
    df['Weight_Sqrt'] = np.sqrt(df['Weight Capacity (kg)'])

    return df

X_train_encoded_df = add_features(X_train_encoded_df)
X_test_encoded_df = add_features(X_test_encoded_df)



X_train_encoded_df


import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = X_train_encoded_df

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA to reduce dimensionality before applying t-SNE
# We reduce to 50 dimensions (you can adjust this based on the dataset)
pca = PCA(n_components=20)
data_pca = pca.fit_transform(data_scaled)

# Apply t-SNE with a smaller subset of the data for efficiency
# Sample 10,000 rows for t-SNE to speed up the process (or more if needed)
sampled_data = data_pca

# Apply t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=300)
data_tsne = tsne.fit_transform(sampled_data)

# Visualize the t-SNE output
plt.figure(figsize=(8, 6))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='blue', marker='o', alpha=0.5)
plt.title('t-SNE Visualization of Denoised Data (Sampled)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()



data_tsne
pd.DataFrame(data_tsne).to_csv('denoised_train.csv')


data_tsne.shape


y_train[:10000]


y_train.fillna(y_train.mean(), inplace=True)


# Ğ�Ğ°Ğ²Ñ‡Ğ°Ğ½Ğ½Ñ� Ğ»Ñ–Ğ½Ñ–Ğ¹Ğ½Ğ¾Ñ— Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ–Ñ—
from sklearn.linear_model import LinearRegression

X_denoised = data_tsne
regressor = LinearRegression()
regressor.fit(data_tsne, y_train)
y_pred = regressor.predict(X_denoised)

# Ğ�Ñ†Ñ–Ğ½ĞºĞ° Ğ¼Ğ¾Ğ´ĞµĞ»Ñ–
mse = rmse(y_train, y_pred)
print(f"MSE Ğ¿Ñ–Ñ�Ğ»Ñ� Ğ´ĞµĞ½Ğ¾Ğ¹Ğ·Ğ¸Ğ½Ğ³Ñƒ: {mse:.4f}")


test=tsne.fit_transform(X_test_encoded_df)


pd.DataFrame(test).to_csv('denoised_test.csv')





from sklearn.model_selection import train_test_split


x_train, x_val, y_train_model, y_val = train_test_split(X_denoised, y_train)


p = param = {'n_estimators': 350, 'max_depth': 3, 'min_child_weight': 4, 'learning_rate': 0.08594446209873564, 'subsample': 0.8398992680035604, 'colsample_bytree': 0.9529221921014095, 'reg_alpha': 1.1463897658647415, 'reg_lambda': 6.708469240337878, 'gamma': 0.010410503602230198}
import xgboost 
xb = xgboost.XGBRegressor(verbose=1, **p)
xb.fit(x_train, y_train_model)


rmse(y_val, xb.predict(x_val))


from sklearn.model_selection import cross_val_score


def objective_cat(trial):
    # params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 20, 500),
    #     'max_depth': int(trial.suggest_float('max_depth', 1, 100, log=True)),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    #     'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    #     'gamma': trial.suggest_float('gamma', 0, 2),
    #     'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
    #     'subsample': trial.suggest_float('subsample', 0, 1)
    # }

    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000), 
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),  
        'depth': trial.suggest_int('depth', 1, 16),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),  
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'task_type': 'GPU',
        'verbose': 0
    }

    
    cat_model = CatBoostRegressor(**params)

    return -1*cross_val_score(cat_model, x_train, y_train_model, n_jobs=1, cv=3, scoring='neg_root_mean_squared_error').mean()

# study = optuna.create_study(direction='minimize')
# study.optimize(objective_cat, n_trials=100)
# trial = study.best_trial
# param = trial.params


param = {'depth': 6, 'learning_rate': 0.025187969747436947, 'iterations': 636, 'l2_leaf_reg': 5.910298555905153, 'bagging_temperature': 0.06752939348809489}


# param = {'iterations': 950, 'early_stopping_rounds': 11, 'depth': 4, 'l2_leaf_reg': 0.31610651106415355, 'learning_rate': 0.01}
# param = {'iterations': 872, 'early_stopping_rounds': 27, 'depth': 4, 'l2_leaf_reg': 1.4430910951858595, 'learning_rate': 0.06200462154677792}


pd.DataFrame(X_denoised)


model = CatBoostRegressor(verbose=100, **param)


model.fit(x_train, y_train_model)


val_predicted = model.predict(x_val)
rmse(y_val, val_predicted)





# import eli5
# from eli5.sklearn import PermutationImportance

# perm = PermutationImportance(model, random_state=1).fit(x_val, y_val)
# eli5.show_weights(perm, feature_names = x_val.columns.tolist())


import matplotlib.pyplot as plt
import seaborn as sns
def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()
    return fi_df
    
fe_imp = plot_feature_importance(model.get_feature_importance(),X_train_encoded_df.columns,'CATBOOST')


fe_selection = fe_imp[:3].feature_names
fe_selection


model.fit(x_train[fe_selection], y_train_model)


val_predicted = model.predict(x_val[fe_selection])
rmse(y_val, val_predicted)


submission = pd.DataFrame({
    'id': subm.id,
    'Price': model.predict(test)
})
submission.to_csv('submission.csv', index=False)
print('Submission Done')




