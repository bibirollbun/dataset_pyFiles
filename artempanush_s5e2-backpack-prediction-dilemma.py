import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


train = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv")
train.head()


test = pd.read_csv(r"/kaggle/input/playground-series-s5e2/test.csv")
test.head()


train.info()


train.shape


train.isnull().sum()


isn = train[train.isnull().any(axis=1)]
isn.head()


brand_gr = train.groupby(['Brand', 'Material'])['id'].agg('count')
brand_gr


brand_gr = train.groupby(['Brand', 'Style'])['Weight Capacity (kg)'].agg('mean')
brand_gr


train.columns


cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']


def countplots(df, cols, plots_per_row=2):
    
    """
    Plots countplots for multiple categorical columns, arranging them in rows.
    """

    num_cols = len(cols)
    num_rows = (num_cols + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(10, 4 * num_rows)) # Adjust figsize as needed
    axes = axes.flatten() # Flatten the axes array for easier indexing

    for i, col in enumerate(cols):
        sns.countplot(x=col, data=df, ax=axes[i], palette="ch:s=.25,rot=-.25")
        axes[i].set_title(f"Countplot of {col}")
        axes[i].tick_params(axis='x', rotation=35) # Rotate x-axis labels for readability

    # # Remove empty subplots if necessary
    # for i in range(num_cols, num_rows * plots_per_row):
    #     fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


countplots(train, cols, plots_per_row=2)


def boxplots(df, cols):

    num_cols = len(cols)
    fig, axes = plt.subplots(nrows=num_cols, ncols=2, figsize=(10, 4 * num_cols))

    if num_cols == 1:
       axes = [axes]

    for i, col in enumerate(cols):

        palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors=len(df[col].unique()))
        
        sns.boxplot(x=col, y='Weight Capacity (kg)', data=df, palette=palette, linewidth=1.2, fliersize=5, whis=1.5, ax=axes[i][0])
        axes[i][0].set_title(f'Boxplot of Weight Capacity by {col}')
        axes[i][0].set_ylabel('Weight Capacity (kg)')

        sns.boxplot(x=col, y='Price', data=df, palette=palette, linewidth=1.2, fliersize=5, whis=1.5, ax=axes[i][1])
        axes[i][1].set_title(f'Boxplot of Price by {col}')
        axes[i][1].set_ylabel('Price')
      
    plt.tight_layout()
    plt.show()


boxplots(train, cols)


def complete_feature(df):

    '''fill the missing values and create new features'''

    df = df.copy()

    for col in ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']:
        df[col] = df[col].fillna('No Info')

    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median())

    # df['color_brand'] = df['Color'] + "_" + df['Brand']
    # df['color_material'] = df['Color'] + "_" + df['Material']

    return df


#apply the fucntion

train = complete_feature(train)
test = complete_feature(test)


categorical = train.select_dtypes(include=['object', 'category']).columns


#encoding

oe = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
train[categorical] = oe.fit_transform(train[categorical])
test[categorical] = oe.transform(test[categorical])


# for col in ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
#        'Waterproof', 'Style', 'Color']:
#     train[col] = train[col].fillna('No Info')
#     test[col] = test[col].fillna('No Info')


# train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median())
# test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median())                                                                   


def density_heatmap(df, col1, col2, bins=10, num_ticks=10):
    
    """
    Creates a density heatmap for two numerical columns
    """

    heatmap_data, x_edges, y_edges = np.histogram2d(df[col1], df[col2], bins=bins)
    heatmap_data = np.transpose(heatmap_data)

    x_tick_positions = np.linspace(0, heatmap_data.shape[1] - 1, num_ticks)
    x_tick_labels = [f"{x:.0f}" for x in np.linspace(df[col1].min(), df[col1].max(), num_ticks)]

    y_tick_positions = np.linspace(heatmap_data.shape[0] - 1, 0, num_ticks)
    y_tick_labels = [f"{y:.0f}" for y in np.linspace(df[col2].min(), df[col2].max(), num_ticks)]

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu")

    # Set tick positions and labels
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"Density Heatmap of {col1} vs. {col2}")
    plt.show()


density_heatmap(train, 'Weight Capacity (kg)', 'Price', bins=10, num_ticks = 10)


#create the scatterplot

plt.figure(figsize=(7, 7))
sns.regplot(x='Weight Capacity (kg)', y='Price', data=train, color='skyblue',
           scatter_kws={'color': '#007bff', 's': 100, 'alpha': 0.8, 'edgecolor': 'black'},
           line_kws={'color': '#dc3545', 'linewidth': 3})
plt.title('The dependence of price on the weight capacity')
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.show()



# categorical = train.select_dtypes(include=['object', 'category']).columns


# #encoding

# oe = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
# train[categorical] = oe.fit_transform(train[categorical])
# test[categorical] = oe.transform(test[categorical])


def heatmap_plot(df, title_name):

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(14, 8))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidth = .3, cmap='Blues', annot=True, annot_kws={"fontsize":6})
    plt.title(title_name)
    plt.show()

heatmap_plot(train, 'Correlation of the Train Dataset')


# choose columns for building model

X_col = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']
y_col = 'Price'


X_train, X_valid, y_train, y_valid = train_test_split(train[X_col], train[y_col], test_size=0.2, random_state=42)


# X_col1 = ['Weight Capacity (kg)']
# X_col2 = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
#        'Waterproof', 'Style', 'Color']
# X_train1, X_valid1, y_train1, y_valid1 = train_test_split(train[X_col1], train[y_col], test_size=0.2, random_state=42)
# X_train2, X_valid2, y_train2, y_valid2 = train_test_split(train[X_col2], train[y_col], test_size=0.2, random_state=42)


def modelling(model):
    
    ''' Fit the model, make predictions and calculate rmse '''

    model.fit(X_train, y_train)
    model_predict = model.predict(X_valid)
    model_rmse = np.sqrt(mean_squared_error(y_valid, model_predict))
        
    return model_rmse, model_predict


models_predictions = []
models_names = []


def validation_modelling(model):

    """
    Performs K-fold cross-validation for a given model.
    """
    
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    scores = []

    for train_idx, val_idx in kf.split(train[X_col], train[y_col]):

        X_train_fold, X_valid_fold = train[X_col].iloc[train_idx], train[X_col].iloc[val_idx]
        y_train_fold, y_valid_fold = train[y_col].iloc[train_idx], train[y_col].iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold, eval_set = [(X_valid_fold, y_valid_fold)])

        model_predict = model.predict(X_valid_fold)
        rmse = np.sqrt(mean_squared_error(y_valid_fold, model_predict))
        scores.append(rmse)

    return np.mean(scores), model_predict
    


cat_params = {
    'learning_rate': 0.05,  # Model's learning step size.
    'l2_leaf_reg': 7,        # L2 regularization to prevent overfitting.
    'depth': 6,             # Maximum tree depth.
    'iterations': 1000,      # Number of boosting rounds.
    'loss_function': 'RMSE',   # Training loss function.
    'eval_metric': 'RMSE',   # Metric for early stopping.
    'random_seed': 42,        # Seed for reproducibility.
    'od_type': "Iter",      # Overfitting detection type.
    'od_wait': 100          # Iterations to wait after best result.
}

cat_model = CatBoostRegressor(**cat_params, verbose=False)


models_names.append('CAT')
rmse_cat, pred_cat = modelling(cat_model)
models_predictions.append(rmse_cat)
rmse_cat


models_names.append('CATFOLD')
rmse_cat_fold, pred_cat_fold = validation_modelling(cat_model)
models_predictions.append(rmse_cat_fold)
pred_cat_fold


cat_fi = cat_model.get_feature_importance(prettified=True)


plt.figure(figsize=(16, 5))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_cat, color='steelblue')
plt.xlabel('Y_valid', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title('Pred vs Valid for Catboost')

plt.subplot(1,2,2)
sns.barplot(x='Importances', y='Feature Id', data=cat_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.yticks(fontsize=8)
plt.title('Feature Importance', fontsize=16)

plt.tight_layout()
plt.show()


lgbm_params = {
    'learning_rate': 0.01,         # Step size shrinkage to prevent overfitting.
    'max_depth': 17,                # Maximum depth of a tree.
    'subsample': 0.66,              # Fraction of samples used for training each tree.
    'colsample_bytree': 0.63,       # Fraction of features used for building each tree.
    'reg_alpha': 9,                 # L1 regularization term on weights.
    'reg_lambda': 4.5,               # L2 regularization term on weights.
    'n_jobs': -1                   # Use all available cores for parallel processing.
}

lgbm_model = LGBMRegressor(**lgbm_params, verbose=-1)


models_names.append('LGBM')
rmse_lgbm, pred_lgbm = modelling(lgbm_model)
models_predictions.append(rmse_lgbm)
rmse_lgbm


models_names.append('LGBMFOLD')
rmse_lgbm_fold, pred_lgbm_fold = validation_modelling(lgbm_model)
models_predictions.append(rmse_lgbm_fold)
pred_lgbm_fold


lgbm_fi = pd.DataFrame({'Feature Id':X_col, 'LGBM_Importances':lgbm_model.feature_importances_})
lgbm_fi = lgbm_fi.sort_values(by='LGBM_Importances', ascending=False)


plt.figure(figsize=(15, 6))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_lgbm, color='steelblue')
plt.xlabel('Y_valid', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.title('Pred vs Valid for LightGBM')

plt.subplot(1,2,2)
sns.barplot(x='LGBM_Importances', y='Feature Id', data=lgbm_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance', fontsize=16)

plt.tight_layout()
plt.show()


xgb_params = {
    "objective": "reg:squarederror",  # Regression objective using squared error.
    "n_estimators": 100,            # Number of boosted trees to fit.
    "learning_rate": 0.05,           # Step size shrinkage to prevent overfitting.
    "max_depth": 6,                  # Maximum depth of each tree.
    "subsample": 0.8,                # Fraction of training data used for each tree.
    "colsample_bytree": 0.8,         # Fraction of features used for each tree.
    "random_state": 42,              # Seed for reproducibility.
    "min_child_weight": 3, # Minimum sum of instance weight (hessian) needed in a child.
}

xgb_model = XGBRegressor(**xgb_params)


models_names.append('XGB')
rmse_xgb, pred_xgb = modelling(xgb_model)
models_predictions.append(rmse_xgb)
rmse_xgb


# models_names.append('XGBFOLD')
# rmse_xgb_fold, pred_xgb_fold = validation_modelling(xgb_model)
# models_predictions.append(rmse_xgb_fold)
# rmse_xgb_fold


xgb_fi = pd.DataFrame({'Feature Id':X_col, 'XGB_Importances':xgb_model.feature_importances_})
xgb_fi = xgb_fi.sort_values(by='XGB_Importances', ascending=False)


plt.figure(figsize=(15, 6))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_xgb, color='steelblue')
plt.xlabel('Y_valid', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.title('Pred vs Valid for XGBBoost')

plt.subplot(1,2,2)
sns.barplot(x='XGB_Importances', y='Feature Id', data=xgb_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance', fontsize=16)

plt.tight_layout()
plt.show()


models = pd.DataFrame({'model':models_names, 'prediction':models_predictions})
models


display(test)


idd = test['id']


scores = cat_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'Price':scores})
submission.to_csv('sub_cat_1ver.csv', index=False)
submission


scores2 = lgbm_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'Price':scores2})
submission.to_csv('sub_lgbm_1ver.csv', index=False)
submission


# scores4 = lgbm_model1.predict(test[X_col1])
# submission = pd.DataFrame({'id':idd, 'num_sold':scores4})
# submission.to_csv('sub_lgbm1_1ver.csv', index=False)
# submission


# scores6 = lgbm_model2.predict(test[X_col2])
# submission = pd.DataFrame({'id':idd, 'num_sold':scores6})
# submission.to_csv('sub_lgbm2_1ver.csv', index=False)
# submission


scores3 = xgb_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'Price':scores3})
submission.to_csv('sub_xgb_1ver.csv', index=False)
submission

