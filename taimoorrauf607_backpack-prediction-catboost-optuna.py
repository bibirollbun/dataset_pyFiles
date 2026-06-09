import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostRegressor,Pool,cv
from sklearn.metrics import mean_squared_error
import optuna
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
# train= pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Drop the 'id' column as it is not useful for modeling
train = train.drop('id', axis=1)


# store categorical features
cat_features = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


# only use df for my expolatory data analysis(EDA)
df = train
df.head()


plt.figure(figsize=(15, 5))
plt.subplot(121)
sns.histplot(df["Weight Capacity (kg)"], bins=20, kde=True)
plt.title("Distribution of Weight",fontweight='bold')
plt.xlabel("Weight")
plt.ylabel("Frequency")

# Boxplot
plt.subplot(122)
sns.boxplot(x=df["Weight Capacity (kg)"])
plt.title("Boxplot of Weight",fontweight='bold')
plt.show()



plt.figure(figsize=(15, 5))
plt.subplot(121)
sns.histplot(df["Price"], bins=20,kde=True)
plt.title("Distribution of Price",fontweight='bold')
plt.xlabel("Price")
plt.ylabel("Frequency")

# Boxplot
plt.subplot(122)
sns.boxplot(x=df["Price"])
plt.title("Boxplot of Price",fontweight='bold')
plt.show()


# # Count plot for categorical features
# for col in df[cat_features].columns:
#     plt.figure(figsize=(8, 4))
#     sns.countplot(x=df[col], order=df[col].value_counts().index)
#     plt.title(f"Distribution of Different {col}")
#     plt.xticks(rotation=45)
#     plt.show()



# Boxplot of Weight by Style
plt.figure(figsize=(10, 5))
sns.boxplot(x="Style", y="Weight Capacity (kg)", hue="Color", data=df, palette=df['Color'].unique().tolist()[0:6])
plt.title("Weight Distribution by Style and color", fontweight='bold')
plt.xticks(rotation=45, fontweight='bold')
plt.show()


# # Boxplot of Weight by different columns and hue is color
# for col in df.columns[0:7]:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=col, y="Weight Capacity (kg)", hue="Color", data=df, palette=df['Color'].unique().tolist()[0:6])
#     plt.title(f" Distribution by {col} and color", fontweight='bold')
#     plt.xticks(rotation=45, fontweight='bold')
#     plt.show()


# Crosstab for categorical relationships
cross_tab = pd.crosstab(df["Brand"], df["Waterproof"])
sns.heatmap(cross_tab, annot=True, cmap="coolwarm", fmt="d")
plt.title("Laptop Compartment vs Waterproof")
plt.show()



# Crosstab for categorical relationships
cross_tab = pd.crosstab(df["Brand"], df["Laptop Compartment"])
sns.heatmap(cross_tab, annot=True, cmap="coolwarm", fmt="d")
plt.title(" Brand vs Laptop Compartment")
plt.show()


sns.pairplot(df, hue="Style")
plt.show()


# Fill missing values in categorical columns with "Missing" and convert to string type
train[cat_features] = train[cat_features].fillna("Missing").astype(str)  

# Fill missing values in the numerical column 'Weight Capacity (kg)' with its mean value
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean())  


# Define feature set (X) and target variable (y)
X = train.drop('Price', axis=1)  # Drop 'Price' column to create feature set
y = train['Price']  # 'Price' is the target variable

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create CatBoost Pool objects for efficient training with categorical features
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)


# # Best hyperparameters for the CatBoost model (optimized values by Optuna)
best_params = {
    'iterations': 700,  # Number of boosting iterations
    'depth': 7,  # Tree depth (complexity of the model)
    'learning_rate': 0.15123333,  # Step size for weight updates
    'l2_leaf_reg': 5.735312101321457,  # Regularization parameter to prevent overfitting
    'random_strength': 2.8798502237206745  # Adds randomness to the learning process
}


# Initialize CatBoost Regressor model with the best parameters
model = CatBoostRegressor(**best_params)
# Train the model using the training dataset with early stopping
model.fit(train_pool, early_stopping_rounds=50, verbose=False)


# Predict the target variable on the test dataset
y_pred = model.predict(test_pool)


# Compute the Root Mean Squared Error (RMSE) to evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root_mean_squared_error:",rmse)


# Get feature importance scores from the trained model
feature_importance = model.get_feature_importance(train_pool)

# Convert the feature importance array to a list for easier interpretation
importance = feature_importance.tolist()


# create a feature importance dataframe 
impotance_df =pd.DataFrame({"Features":train.drop(['Price'],axis=1).columns,"importance":importance}).sort_values(by='importance')


# Plotting feature importance
plt.figure(figsize=(15,6))
colors = sns.color_palette("plasma", len(impotance_df))
# horizontal bar chart with colors based on importance
plt.barh(impotance_df.iloc[:,0], impotance_df.iloc[:,1], color=[colors[i] for i in range(len(impotance_df))])
plt.xlabel('Feature Importance',fontweight='bold')
plt.title('CatBoost Feature Importance',fontsize=20,fontweight='bold')
plt.xlabel("Importance Score", fontsize=20,fontweight='bold')
plt.ylabel("Features", fontsize=20,fontweight='bold')
# Add importance scores as text on bars
for i, v in enumerate(impotance_df["importance"]):
    plt.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=15, fontweight="bold", color='black')  # Display values with two decimal points

# Set the x-axis limit slightly beyond the max importance score for better visualization
plt.xlim(0, max(impotance_df["importance"]) * 1.1)
plt.yticks(fontsize=15, fontweight="bold", )
plt.tight_layout()

plt.show()



# # Hyperparameter tunning using Optuna 
# def objective(trial):  # objective function for optuna study
#     params = {
#         'iterations': trial.suggest_int('iterations', 600, 1000),
#         'depth':trial.suggest_int('depth', 5, 8),
#         'learning_rate':trial.suggest_float('learning_rate', 0.1,0.3),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.5,6),
#         'random_strength': trial.suggest_float('random_strength', 2, 3),
#         'loss_function': 'RMSE',
#     }
    
#     # Train CatBoost Model
#     model = CatBoostRegressor(**params)
#     model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50, verbose=False)

#     # Predict on validation set
#     y_pred = model.predict(X_test)

#     # Calculate RMSE
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
#     return rmse  # Optuna minimizes this RMSE

# # Run Optuna Optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=10)


# study.best_params


def preprocess(train):
    # Fill missing values in categorical columns with "Missing" and convert to string type
    train[cat_features] = train[cat_features].fillna("Missing").astype(str)  

    # Fill missing values in the numerical column 'Weight Capacity (kg)' with its mean value
    train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean())  

    train = train.drop('id',axis=1)
    return train

test = preprocess(test)


# Predict the target variable on the test dataset
test_pred = model.predict(test)


test_pred


id = [i for i in range(300000,300000+test.shape[0])]


submission =pd.DataFrame({'id':id,'Price':test_pred})


submission.to_csv('submission_file.csv')
print("file is submited as submision_file.csv")

