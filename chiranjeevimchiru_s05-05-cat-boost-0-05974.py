import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import optuna
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")



# Function to display missing values and data types
def check_missing_and_dtypes(df, name="Dataset"):
    print(f"\n{name} Info:")
    print("-" * 60)
    nulls = df.isnull().sum()
    dtypes = df.dtypes
    summary = pd.DataFrame({
        "Data Type": dtypes,
        "Missing Values": nulls,
        "Missing (%)": (nulls / len(df)) * 100
    })
    print(summary)
    return summary
train_info = check_missing_and_dtypes(train, "Train Set")
test_info = check_missing_and_dtypes(test, "Test Set")


#univariate distribution of the features.
col = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
colors = sb.color_palette("Set2", len(col))

fig, axes = plt.subplots(3, 3, figsize=(12, 8))  
axes = axes.flatten()

for i in range(len(col)):
    sb.histplot(train[col[i]], kde=True, ax=axes[i], color=colors[i])
    axes[i].set_ylabel("")
sb.countplot(x='Sex', data=train, ax=axes[len(col)])
axes[len(col)].set_ylabel("")

for j in range(len(col) + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#relationship between numeric features and calories burned
col = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

fig, axes = plt.subplots(2,3, figsize = (8, 6) )

axes = axes.flatten()

for i in range(len(axes)):
    if i < len(col):
        sb.scatterplot(x = col[i], y = 'Calories', data = train, ax = axes[i])
        axes[i].set_ylabel("")
    else:
        fig.delaxes(axes[i])
plt.tight_layout()
plt.show()


#create categorical grouping for age
bins = [20, 35, 55, 80]
labels = ['Young Adults', 'Middle Age', 'Senior']

train['Age_Grp'] = pd.cut(train['Age'], bins = bins, labels = labels, right = True, include_lowest = True)
#test['Age_Grp'] = pd.cut(test['Age'], bins = bins, labels = labels, right = True, include_lowest = True)


#bivariate relationship between age grp and calories
sb.boxplot(x='Age_Grp', y='Calories', data=train)
plt.title('Calories distribution by Age Group')
plt.show()


g = sb.FacetGrid(train, col="Age_Grp", row="Sex", hue="Heart_Rate", palette="viridis", height=4)
g.map(sb.scatterplot, "Duration", "Calories")

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Calories vs Duration by Age Group and Sex (Heart Rate as color)')

plt.tight_layout()
plt.show()


import numpy as np
def feature_engineering(df):
    # Body Mass Index (BMI)
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    
    # Intensity index
    df['Intensity_Index'] = df['Heart_Rate'] / df['Duration']
    
    # Log transformations 
    df['Age'] = np.log1p(df['Age'])
    df['Body_Temp'] = np.log1p(df['Body_Temp'])

    # BMR
    df['BMR'] = (
        10 * df['Weight'] + 
        6.25 * df['Height'] - 
        5 * df['Age'] + 
        np.where(df['Sex'] == 'male', 5, -161)
    )

    # Interaction features
    df['HR_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']
    df['Metabolic_Load'] = df['Heart_Rate'] * df['Body_Temp'] * df['Duration']  # Same as HR_Duration_Temp
    df['Age_Duration'] = df['Age'] * df['Duration']
    df['BMI_HR'] = df['BMI'] * df['Heart_Rate']
    df['Age_Body_Temp'] = df['Age'] * df['Body_Temp']
    df['Duration_Body_Temp'] = df['Duration'] * df['Body_Temp']
    df['BMI_Body_Temp'] = df['BMI'] * df['Body_Temp']
    df['Age_Duration_Temp'] = df['Age'] * df['Duration'] * df['Body_Temp']

    # Log transform Calories only for training set
    if 'Calories' in df.columns:
        df['Calories'] = np.log1p(df['Calories'])

    return df
    
# Apply to both datasets
train = feature_engineering(train)
test = feature_engineering(test)



train['Sex'] = train['Sex'].map({'male': 1, 'female': 0}).astype(int)
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0}).astype(int)


test_df = test.drop(['id'], axis = 1)

x = train.drop(['Age_Grp', 'Calories', 'id'], axis = 1)
y = train['Calories']


#split x and y into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 4)


## Define custom RMSLE metric
def rmsle(y_true, y_pred):
    preds = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(preds, y_val))

# Define the Optuna objective function
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 7000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 6, 16),  # for SymmetricTree
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log = True),
        'grow_policy': 'SymmetricTree',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 34,
        'early_stopping_rounds': 100,
        'use_best_model': True,
        'verbose': 0
    }

   
    train_pool = Pool(data=x_train, label=y_train)
    valid_pool = Pool(data=x_val, label=y_val)

    
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool)

    
    preds = model.predict(valid_pool)
    score = rmsle(y_val, preds)
    return score


# Create and run the Optuna study
#study = optuna.create_study(direction='minimize') #The goal is to minimize the objective function (e.g., RMSLE, RMSE, loss).
#study.optimize(objective, n_trials=20)

# Print best results
#print("Best RMSLE:", study.best_value)
#print("Best params:")
#for key, value in study.best_params.items():
#print(f"  {key}: {value}")


# Train CatBoost with the best parameters obtained from the Optuna hyperparameter tuning above.
cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.05,
    l2_leaf_reg=3,
    grow_policy='SymmetricTree',
    loss_function='RMSE',
    eval_metric='RMSE',
    depth=10,
    random_seed=34,
    early_stopping_rounds=100,
    task_type='CPU',  # Change to 'GPU' if available
    verbose=100
)
cat_model.fit(x_train, y_train)


cat_pred = cat_model.predict(x_val)

cat_pred = np.expm1(cat_pred)
y_val = np.expm1(y_val)

rmsle = np.sqrt(mean_squared_log_error(cat_pred, y_val))
rmsle


feature_importance_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': cat_model.get_feature_importance()
}).sort_values(by='Importance', ascending=False)

# Display all features
print(feature_importance_df.to_string(index=False))


import seaborn as sns
import matplotlib.pyplot as plt

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance_df,
    x='Importance',
    y='Feature',
    palette='viridis'  # Optional: Choose any color palette
)

# Add labels and title
plt.title('Feature Importances from CatBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Improve layout
plt.tight_layout()
plt.show()
for index, value in enumerate(feature_importance_df['Importance']):
    plt.text(value, index, f'{value:.2f}', va='center')


top_features = feature_importance_df.head(15)

sns.barplot(
    data=top_features,
    x='Importance',
    y='Feature',
    palette='coolwarm'
)



# Make final predictions on the test dataset 
final_pred = cat_model.predict(test_df)

# Convert back from log scale
final_pred = np.expm1(final_pred)


final_pred = np.maximum(0, final_pred)


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': final_pred
})
submission.to_csv('submission.csv', index=False)
submission.head()

