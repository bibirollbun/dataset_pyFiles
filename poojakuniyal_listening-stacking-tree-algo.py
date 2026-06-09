import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBRegressor 


sample_df= pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sample_df.head() 


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
train_df.head() 


train_df.info()


train_df.describe()


train_df.describe(include='object')


train_df.drop(columns='id', axis=1,inplace=True) 


sns.set_style('darkgrid')
missing_percentage = train_df.isna().sum() / train_df.shape[0] *100
plt.figure(figsize=(10,6))
missing_percentage.plot(kind='bar', color='skyblue')
plt.title('(Train) Percentage of Missing Values Per Column')
plt.xlabel('Columns')
plt.ylabel('Percentage ( %)')
plt.xticks(rotation=75)
plt.show();


test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_df.tail()


test_id = test_df.id
test_id.head(3)


test_df.drop(columns='id', axis=1,inplace=True)



missing_percentage = test_df.isna().sum() / test_df.shape[0] * 100

plt.figure(figsize=(10, 6))
missing_percentage.plot(kind='bar', color='skyblue')
plt.title('(Test) Percentage of Missing Values Per Column')
plt.xlabel('Columns')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=75)
plt.show()


train_df.dtypes


plt.figure(figsize=(8, 6))
plt.hexbin(train_df['Guest_Popularity_percentage'], train_df['Listening_Time_minutes'], gridsize=30, cmap='Blues')
plt.colorbar(label='Density')
plt.title('Hexbin Plot Dependent vs Independent Variables')
plt.xlabel('Guest_Popularity_percentage')
plt.ylabel('Listening_Time_minutes')
plt.show()



# Create the scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Episode_Length_minutes', y='Listening_Time_minutes', hue='Episode_Sentiment', data=train_df, alpha=0.7)
plt.title('Scatter Plot of Independent and dependent Variables with a tinge o')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.legend(title='Episode_Sentiment')
plt.show()


# Bar plot

plt.figure(figsize=(8, 5))
sns.barplot(x='Publication_Time', y='Listening_Time_minutes', data=train_df, palette="viridis") 
plt.title('Bar Plot for Publication_Time & Listening_Time_minutes', fontsize=14) 
plt.xlabel("Publication Time", fontsize=12)  
plt.ylabel("Listening Time (minutes)", fontsize=12)  
plt.xticks(rotation=45, fontsize=10)  
plt.tight_layout()
plt.show()


# Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x='Genre', y='Listening_Time_minutes', data=train_df)
plt.title('Box Plot')
plt.xticks(rotation=75)
plt.show()



crosstab_result = pd.crosstab(train_df.Podcast_Name, train_df.Episode_Sentiment)

# Create a heatmap
plt.figure(figsize=(12, 8))  
sns.heatmap(crosstab_result, annot=True, fmt="d", cmap="YlGnBu", cbar=True, 
            annot_kws={"size": 12})  # Increase font size for numbers inside heatmap
plt.title("Crosstab: Podcast Name vs Episode Sentiment", fontsize=16)  # Larger title font
plt.xlabel("Episode Sentiment", fontsize=14)  # Larger X-axis label
plt.ylabel("Podcast Name", fontsize=14)  # Larger Y-axis label
plt.xticks(rotation=45, fontsize=12)  #  for X-axis ticks
plt.yticks(fontsize=12)  # Adjust size for Y-axis ticks
plt.tight_layout()
plt.show()


plt.hist(train_df.Guest_Popularity_percentage, bins=30, color='pink');


!pip install squarify
import squarify



# Calculate frequencies
value_counts = train_df['Podcast_Name'].value_counts()
sizes = value_counts.values[:20]  # Show only the top 20 for readability

plt.figure(figsize=(12, 8))
squarify.plot(sizes=sizes, label=value_counts.index[:20], pad=True)
plt.title(f"Treemap of Podcast_Name (Top 20)")
plt.axis("off")
plt.show()


numerical = train_df.select_dtypes(exclude='object')
num_vars = [i for i in numerical if i not in ['Listening_Time_minutes']]

categorical = train_df.select_dtypes(include='object')
cat_vars = [i for i in categorical]
num_vars, cat_vars


from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


X = train_df[num_vars + cat_vars]
y = train_df.Listening_Time_minutes


X.shape, y.shape


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)


# Defining transformers for numerical and categorical columns

numerical_transformers = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean'))])

categorical_transformer = Pipeline(steps=[
    ('encode', OrdinalEncoder())
])
    

# Combining transformers in a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformers, num_vars),
    ('cat', categorical_transformer, cat_vars)
])



test_df.columns


# Pre-process the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
test_df_transformed = preprocessor.transform(test_df)


feature_names = list(num_vars)+list(cat_vars)
feature_names


# Recreate DataFrame with transformed data and original column names
X_train = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test = pd.DataFrame(X_test_transformed, columns=feature_names)
test_df = pd.DataFrame(test_df_transformed, columns=feature_names)


X_train.head(3)


X_test.head(3)


reg1 = RandomForestRegressor(**{'n_estimators': 100,
 'min_samples_split': 5,
 'min_samples_leaf': 5,
 'max_features': 5,
 'max_depth': 10,
 'criterion': 'squared_error',
 'bootstrap': True})

reg2 = ExtraTreesRegressor(n_estimators=700, 
    min_samples_split=5, 
    min_samples_leaf=20, 
    max_features=35, 
    max_depth=15, 
    bootstrap=True)

reg3 = GradientBoostingRegressor()

reg4 = XGBRegressor(n_estimators=500,learning_rate = 0.01, objective='reg:squarederror',
                     max_depth= 4, min_child_weight= 5,gamma=10,
                       subsample = 0.9, colsample_bytree = 0.9, colsample_bylevel= 0.9)

reg5 = XGBRegressor()


Algos = [reg1,reg2,reg3,reg4,reg5]


rows = X_train.shape[0]
rows


# creating a dataframe layer1 : filling values in step by step
# creating one empty df and then using kfold cv filling these values

layer1 = pd.DataFrame({'reg'+ str(i): np.zeros(rows) for i in range(len(Algos) +1)})


print(layer1.shape)
layer1.head()


from sklearn.model_selection import KFold





kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5-Fold stacking
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nFold {fold}")
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    
    X_val_fold = X_train.iloc[val_idx]
    
    for i, model in enumerate(Algos):
        print(f"  Training base model {i+1}")
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        layer1.iloc[val_idx, i] = preds



layer1


layer1 = layer1.drop(columns=['reg5']) # added it by mistake in above code


rows = X_test.shape[0]
layer2_test = pd.DataFrame({'reg'+ str(i): np.zeros(rows) for i in range(len(Algos))})

for i, reg in enumerate(Algos):
    print(f"Training base regressor {i+1} on full training data")
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    layer2_test.iloc[:, i] = preds



print(layer2_test.shape)
layer2_test.head()



# Train meta-model on layer1 (from KFold training predictions)
meta_reg = ExtraTreesRegressor()
meta_reg.fit(layer1, y_train)

# Predict using meta-model on test layer
final_preds = meta_reg.predict(layer2_test)

# Evaluate
mse = mean_squared_error(y_test, final_preds)
print(f"Stacked model MSE on test data: {mse:.4f}")



Root_mse = np.sqrt(mse)
print('Root Mean Squared Error is : ', Root_mse)


# Step 1: Prepare Layer2 features for final test_df
rows = test_df.shape[0]
layer2_final_test = pd.DataFrame({'reg'+ str(i): np.zeros(rows) for i in range(len(Algos))})

for i, reg in enumerate(Algos):
    print(f"Retraining base model {i+1} on full training data")
    reg.fit(X_train, y_train)
    preds = reg.predict(test_df)
    layer2_final_test.iloc[:, i] = preds

# Step 2: Use trained meta-model to predict
final_submission_preds = meta_reg.predict(layer2_final_test)



 # Build submission file
submission = pd.DataFrame({
    'id': test_id, 
    'Listening_Time_minutes': final_submission_preds
})
submission.head(10)


# Save to CSV
submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")





