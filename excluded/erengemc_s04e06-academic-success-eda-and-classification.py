# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set the style of matplotlib
%matplotlib inline
plt.style.use('fivethirtyeight')


# Load training and testing datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv', index_col='id')
train


# Check if there are any missing values
train.isna().sum().sort_values(ascending=False)


# Check if there are duplicate rows
train.duplicated().sum()


# View the general information of the training dataset
train.info()


# View the statistical description of training dataset
train.describe().T


# Store the names of feature columns
initial_features = list(test.columns)
initial_features


# Print the number of unique values for each column
for col in train.columns:
    print(f'{col} has {train[col].nunique()} values')


# Classify columns for better visualization
# Categorical columns: if the number of unique values is 8 or fewer
cat_cols = [col for col in train.columns if train[col].nunique() <= 8]
# Numerical columns: if the number of unique values is 9 or more
num_cols = [col for col in train.columns if train[col].nunique() >= 9]


len(cat_cols)


len(num_cols)


# Target distribution
# Set the figure size and create a count plot
plt.figure(figsize=(10, 8))
ax = sns.countplot(x='Target', data=train, palette='pastel')

# Add labels to each bar in the plot
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 3, f'{int(p.get_height())}', ha="center")

plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Distribution')
plt.show()


# Distribution of categorical variables
plt.figure(figsize=(18, 24))
plotnumber = 1

for col in cat_cols:
    if plotnumber <= len(cat_cols):
        ax = plt.subplot(4, 3, plotnumber)
        sns.countplot(x=train[col], data=train, palette='pastel')
        
        # Add labels to each bar in the plot
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 3, f'{int(p.get_height())}', ha="center")
        
        plt.xlabel(col)
        # plt.xticks(rotation=45)
        plt.xlabel(col)
        
    plotnumber += 1

plt.suptitle('Distribution of Categorical Variables', fontsize=40, y=1)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 24))
plotnumber = 1

# Loop through each column
for col in cat_cols:
    if plotnumber <= len(cat_cols):
        plt.subplot(4, 3, plotnumber)
        ax = sns.countplot(x=train[col], hue=train['Target'], palette='bright')
        
    plotnumber += 1

plt.suptitle('Distribution of Categorical Variables by Target', fontsize=40, y=1)
plt.tight_layout()
plt.show()


# Distribution of numeric variables
plt.figure(figsize=(18, 40))
plotnumber = 1

for column in num_cols:
    if plotnumber <= len(num_cols):
        ax = plt.subplot(9, 3, plotnumber)
        sns.kdeplot(train[column], color='deepskyblue', fill=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
        plt.xlabel(column)
        ax.grid(False)
        
    plotnumber += 1

plt.suptitle('Distribution of Numeric Variables', fontsize=40, y=1)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 135))
plotnumber = 1

for col in num_cols:
    if plotnumber <= len(num_cols):
        
        ax1 = plt.subplot(len(num_cols), 2, 2 * plotnumber - 1)
        sns.kdeplot(train[col], color='salmon', fill=True)
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
        ax1.set_xlabel(col)
        ax1.grid(False)
        
        ax2 = plt.subplot(len(num_cols), 2, 2 * plotnumber)
        sns.boxplot(y=train[col], color='salmon', width=0.6, linewidth=1)
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
        ax2.set_xlabel(col)
        ax2.set_ylabel('')
        ax2.grid(False)

    plotnumber += 1

plt.suptitle('Distribution of Numeric Variables', fontsize=40, y=1)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder

categories = ['dropout', 'enrolled', 'graduate']
label_encoder = LabelEncoder()

# Convert categorical 'Target' labels to numeric values using LabelEncoder
train['Target'] = label_encoder.fit_transform(train['Target'])


# Correlation matrix
plt.figure(figsize=(21, 18))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.1f', linewidths=2, linecolor='lightgrey')
plt.suptitle('Correlation Matrix', fontsize=40, y=1)
plt.show()


# Split the features and target variable
X_train = train[initial_features]
y_train = train['Target']
X_test = test[initial_features]


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def cross_validate_model(model, X_train, y_train, params, n_splits=10):
    """
    Performs K-Fold cross-validation for a given model, returns the last model and average validation accuracy.

    Parameters:
        model: Machine learning model class (e.g., RandomForestClassifier)
        X_train: Training feature dataset
        y_train: Training target dataset
        params: Dictionary of parameters to initialize the model
        n_splits: Number of folds for cross-validation (default: 10)

    Returns:
        last_model: The last trained model instance
        average_val_accuracy: Average validation accuracy over all folds
    """
    # Initialize variables
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    val_scores = []

    # Cross-validation loop
    for fold, (train_ind, valid_ind) in enumerate(cv.split(X_train)):
        # Data splitting
        X_fold_train = X_train.iloc[train_ind]
        y_fold_train = y_train.iloc[train_ind]
        X_val = X_train.iloc[valid_ind]
        y_val = y_train.iloc[valid_ind]
        
        # Model initialization and training
        clf = model(**params)
        clf.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        y_pred_trn = clf.predict(X_fold_train)
        y_pred_val = clf.predict(X_val)
        train_acc = accuracy_score(y_fold_train, y_pred_trn)
        val_acc = accuracy_score(y_val, y_pred_val)
        print(f"Fold: {fold}, Train Accuracy: {train_acc:.5f}, Val Accuracy: {val_acc:.5f}")
        print("-" * 50)
        
        # Accumulate validation scores
        val_scores.append(val_acc)

    # Calculate the average validation score
    average_val_accuracy = np.mean(val_scores)
    print("Average Validation Accuracy:", average_val_accuracy)

    return clf, average_val_accuracy


from sklearn.ensemble import RandomForestClassifier

print('Random Forest Cross-Validation Results:\n')
rf_model, rf_mean_accuracy = cross_validate_model(RandomForestClassifier, X_train, y_train, params={})


# Predict the test set and reverse the label encoding
rf_preds = rf_model.predict(X_test)
rf_preds_labels = label_encoder.inverse_transform(rf_preds)

# Save the predictions to a CSV file
rf_result = pd.DataFrame(X_test.index)
rf_result['Target'] = rf_preds_labels
rf_result.to_csv('result_rf.csv', index=False)
rf_result


from sklearn.ensemble import AdaBoostClassifier

print('AdaBoost Cross-Validation Results:\n')
ada_model, ada_mean_accuracy = cross_validate_model(AdaBoostClassifier, X_train, y_train, params={})


# Predict the test set and reverse the label encoding
ada_preds = ada_model.predict(X_test)
ada_preds_labels = label_encoder.inverse_transform(ada_preds)

# Save the predictions to a CSV file
ada_result = pd.DataFrame(X_test.index)
ada_result['Target'] = ada_preds_labels
ada_result.to_csv('result_ada.csv', index=False)
ada_result


from sklearn.ensemble import GradientBoostingClassifier

print('Gradient Boosting Cross-Validation Results:\n')
gb_model, gb_mean_accuracy = cross_validate_model(GradientBoostingClassifier, X_train, y_train, params={})


# Predict the test set and reverse the label encoding
gb_preds = gb_model.predict(X_test)
gb_preds_labels = label_encoder.inverse_transform(gb_preds)

# Save the predictions to a CSV file
gb_result = pd.DataFrame(X_test.index)
gb_result['Target'] = gb_preds_labels
gb_result.to_csv('result_gb.csv', index=False)
gb_result


from xgboost import XGBClassifier

print('XGBoost Cross-Validation Results:\n')
xgb_model, xgb_mean_accuracy = cross_validate_model(XGBClassifier, X_train, y_train, params={})


# Predict the test set and reverse the label encoding
xgb_preds = xgb_model.predict(X_test)
xgb_preds_labels = label_encoder.inverse_transform(xgb_preds)

# Save the predictions to a CSV file
xgb_result = pd.DataFrame(X_test.index)
xgb_result['Target'] = xgb_preds_labels
xgb_result.to_csv('result_xgb.csv', index=False)
xgb_result


from catboost import CatBoostClassifier

cat_params = {
    'verbose': 0,                       # Silent mode
}

print('CatBoost Cross-Validation Results:\n')
cat_model, cat_mean_accuracy = cross_validate_model(CatBoostClassifier, X_train, y_train, cat_params)


# Predict the test set and reverse the label encoding
cat_preds = cat_model.predict(X_test)
cat_preds_labels = label_encoder.inverse_transform(cat_preds)

# Save the predictions to a CSV file
cat_result = pd.DataFrame(X_test.index)
cat_result['Target'] = cat_preds_labels
cat_result.to_csv('result_cat.csv', index=False)
cat_result


from lightgbm import LGBMClassifier

lgb_params = {
    'verbose': -1,                    # Set to -1 for silent mode, no process information printed
}

print('LightGBM Cross-Validation Results:\n')
lgb_model, lgb_mean_accuracy = cross_validate_model(LGBMClassifier, X_train, y_train, lgb_params)


# Predict the test set and reverse the label encoding
lgb_preds = lgb_model.predict(X_test)
lgb_preds_labels = label_encoder.inverse_transform(lgb_preds)

# Save the predictions to a CSV file
lgb_result = pd.DataFrame(X_test.index)
lgb_result['Target'] = lgb_preds_labels
lgb_result.to_csv('result_lgb.csv', index=False)
lgb_result


accuracy = pd.DataFrame({
    'Model': ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'CatBoost', 'LightGBM'],
    'Score': [rf_mean_accuracy, ada_mean_accuracy, gb_mean_accuracy,
              xgb_mean_accuracy, cat_mean_accuracy, lgb_mean_accuracy]
})

accuracy_sorted = accuracy.sort_values(by='Score', ascending=False)
accuracy_sorted


fig = px.bar(data_frame=accuracy_sorted, x='Score', y='Model', color='Score',
             title='Accuracy Comparison', text='Score')
fig.update_layout(width=600, height=500)
fig.show()


def plot_feature_importances(model, model_name, color_scale='Reds', dataframe=None):
    """
    Plots feature importances of a fitted random forest model.

    Parameters:
    model (RandomForest model): The trained random forest model.
    color_scale (str): Color scale for the plot.
    dataframe (pd.DataFrame): DataFrame used to train the model. Must not be None.

    Returns:
    Plotly Figure: A plot showing feature importances.
    """
    if dataframe is None:
        raise ValueError("Dataframe cannot be None and must contain the feature names.")

    # Extracting feature importances and sorting them
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = dataframe.columns

    # Creating a DataFrame for the importances
    feature_importances = pd.DataFrame({
        'Feature': feature_names[indices],
        'Importance': importances[indices]
    })

    # Plotting the feature importances
    fig = px.bar(feature_importances.sort_values('Importance', ascending=True), 
                 x='Importance', 
                 y='Feature',
                 title=f"Feature Importances in {model_name}",
                 labels={'Importance': 'Importance', 'Feature': 'Feature'},
                 height=1400,
                 color='Importance',
                 color_continuous_scale=color_scale)

    fig.update_layout(xaxis_title='Importance', yaxis_title='Feature')

    return fig


# Feature importance in random forest
model_name = 'Random Forest'
fig = plot_feature_importances(rf_model, model_name, 'Picnic', X_train)
fig.show()


# Feature importance in AdaBoost
model_name = 'AdaBoost'
fig = plot_feature_importances(ada_model, model_name, 'Rainbow', X_train)
fig.show()


# Feature importance in GBM
model_name = 'Gradient Boosting'
fig = plot_feature_importances(gb_model, model_name, 'HSV', X_train)
fig.show()


# Feature importance in XGBoost
model_name = 'XGBoost'
fig = plot_feature_importances(xgb_model, model_name, 'Bluered', X_train)
fig.show()


# Feature importance in CatBoost
model_name = 'CatBoost'
fig = plot_feature_importances(cat_model, model_name, 'Temps', X_train)
fig.show()


# Feature importance in LightGBM
model_name = 'LightGBM'
fig = plot_feature_importances(lgb_model, model_name, 'Reds', X_train)
fig.show()

