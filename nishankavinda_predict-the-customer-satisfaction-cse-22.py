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


data = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/train_dataset.csv')
test = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/test_dataset.csv')


data


desc = pd.DataFrame(index=data.columns.to_list())
desc['type'] = data.dtypes
desc['count'] = data.count()
desc['nunique'] = data.nunique()
desc['null'] = data.isnull().sum()
desc


data['Gender'].unique()


data.describe()


data.info(memory_usage='deep')


def date_time(df):
    date_columns = ['Date_Registered', 'payment_datetime', 'purchased_datetime', 
                    'released_date', 'estimated_delivery_date', 'received_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

data = date_time(data)
test= date_time(test)


duplicates = data[data.drop(columns=['id']).duplicated()]
duplicates


# Remove rows that are duplicates when excluding the 'id' column
data = data[~data.drop(columns=['id']).duplicated()]


data.isnull().sum()


data = data[data['age'] > 0]


data.fillna({
    'Received_tier_discount_percentage': 0,
    'Received_card_discount_percentage': 0,
    'Received_coupon_discount_percentage': 0,
    'loyalty_tier': -1
}, inplace=True)

test.fillna({
    'Received_tier_discount_percentage': 0,
    'Received_card_discount_percentage': 0,
    'Received_coupon_discount_percentage': 0,
    'loyalty_tier': -1
}, inplace=True)


df = pd.concat([data,test], axis =0)


df = df.drop(columns=['tracking_number', 'order_id'])


# Optimizing data types for df
df['id'] = df['id'].astype('Int32')
df['age'] = df['age'].astype('Int8')
df['loyalty_points_redeemed'] = df['loyalty_points_redeemed'].astype('Int16')
df['Received_coupon_discount_percentage'] = df['Received_coupon_discount_percentage'].astype('Int16')
df['Product_value'] = df['Product_value'].astype('Int32')
df['final_payment'] = df['final_payment'].astype('float32')
df['loyalty_tier'] = df['loyalty_tier'].astype('float32')
df['Received_tier_discount_percentage'] = df['Received_tier_discount_percentage'].astype('float32')
df['Received_card_discount_percentage'] = df['Received_card_discount_percentage'].astype('float32')


df['shipping_method'] = pd.Categorical(df['shipping_method'], categories=['standard', 'express'], ordered=True)
df['shipping_method'] = df['shipping_method'].cat.codes

df['customer_experience'] = pd.Categorical(df['customer_experience'], categories=['bad', 'neutral','good'], ordered=True)
df['customer_experience'] = df['customer_experience'].cat.codes


df


from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_and_add(df, column):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[[column]])
    encoded_columns = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out([column]))
    encoded_columns.index = df.index
    df = pd.concat([df, encoded_columns], axis=1)
    df = df.drop(columns=[column])
    return df

columns_to_encode = ['Gender', 'Is_current_loyalty_program_member', 'purchase_medium']

for col in columns_to_encode:
    df = one_hot_encode_and_add(df, col)


df


df.count()


data = df[df['customer_experience'] != -1]
test = df[df['customer_experience'] == -1]
test=test.drop(columns=['customer_experience'])


data.isnull().sum()


# data.hist(figsize=(20, 15),  
#                 bins=30,           
#                 grid=True,        
#                 rwidth=0.9) 


import seaborn as sns
import matplotlib.pyplot as plt

columns_to_plot = data.select_dtypes(include=['number']).columns

num_cols = 4
num_rows = int(np.ceil(len(columns_to_plot) / num_cols))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4 * num_rows)) 

axes = axes.flatten()

for i, column in enumerate(columns_to_plot):
    sns.boxplot(y=data[column], ax=axes[i])
    axes[i].set_title(f"{column}")
    axes[i].grid(False)

plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

columns_to_plot = test.select_dtypes(include=['number']).columns

num_cols = 4
num_rows = int(np.ceil(len(columns_to_plot) / num_cols))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4 * num_rows)) 

axes = axes.flatten()

for i, column in enumerate(columns_to_plot):
    sns.boxplot(y=test[column], ax=axes[i])
    axes[i].set_title(f"{column}")
    axes[i].grid(False)

plt.tight_layout()
plt.show()


outlier_count = data[data['Product_value'] > 10000].shape[0]
print("Outlier count:", outlier_count)


data = data[data['Product_value'] <= 10000]
data = data[data['Received_card_discount_percentage'] <= 5]
data = data[data['Received_coupon_discount_percentage'] <= 5]


def process_dataframe(df):

    # Financial Features
    # Discount amount features
    df['total_discount_percentage'] = df['Received_tier_discount_percentage'] + df['Received_card_discount_percentage'] + df['Received_coupon_discount_percentage']
    df['Total_Discount_Amount'] = df['Product_value'] * ((df['total_discount_percentage']) / 100)
  
    df['discount_amount_ratio'] = df['Total_Discount_Amount'] / (df['Product_value'] + 1e-9)
    df['high_discount_order'] = (df['Total_Discount_Amount'] > df['Total_Discount_Amount'].median()).astype(int)
    
    # shipping cost calculation
    df['shipping_cost'] = df['final_payment'] - (df['Product_value'] - df['Total_Discount_Amount'])
    df['shipping_cost_ratio'] = df['shipping_cost'] / df['Product_value']
    
    # Price sensitivity metrics
    df['price_tier'] = pd.qcut(df['Product_value'], q=5, labels=[0,1,2,3,4])
    df['discount_types_used'] = (df[['Received_tier_discount_percentage', 
                                   'Received_card_discount_percentage', 
                                   'Received_coupon_discount_percentage']] > 0).sum(axis=1)
    
    # Loyalty Features
    df['loyalty_engagement_score'] = df['loyalty_points_redeemed'] / df['Product_value']
    
    # Temporal Features
    # Purchase timing
    df['purchase_hour'] = df['purchased_datetime'].dt.hour
    df['purchase_day_of_week'] = df['purchased_datetime'].dt.dayofweek
    df['is_weekend_purchase'] = df['purchase_day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['purchase_hour'] >= 9) & (df['purchase_hour'] <= 17)).astype(int)
    df['purchase_month'] = df['purchased_datetime'].dt.month
    df['purchase_quarter'] = df['purchased_datetime'].dt.quarter
    
    # Delivery Features
    df['receive_day_of_week'] = df['received_date'].dt.dayofweek
    df['processing_days'] = (df['released_date'] - df['purchased_datetime']).dt.days
    df['delivery_days'] = (df['received_date'] - df['released_date']).dt.days
    df['total_order_days'] = (df['received_date'] - df['purchased_datetime']).dt.days
    df['delivery_delay'] = (df['received_date'] - df['estimated_delivery_date']).dt.days
    df['is_delayed'] = (df['delivery_delay'] > 0).astype(int)
    
    # Customer Features
    df['customer_tenure_days'] = (df['purchased_datetime'] - df['Date_Registered']).dt.days
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 25, 35, 50, 65], 
                            labels=[0, 1, 2, 3])
    
    # Purchase History Features
    df['is_first_purchase'] = (df.groupby('user_id')['purchased_datetime'].cumcount() == 0).astype(int)
    df['purchase_count'] = df.groupby('user_id')['transaction_id'].transform('count')
    
    # Product Features
    df['product_category_encoded'] = pd.factorize(df['product_category'])[0]
    
    return df


# Apply the function to both dataframes
data = process_dataframe(data)
test = process_dataframe(test)


data =data.drop(columns=['user_id','transaction_id','product_category','payment_method'])
test =test.drop(columns=['user_id','transaction_id','product_category','payment_method'])


# Check for missing values in each column of X
print(data.isnull().sum())


# numerical_columns = df.select_dtypes(include=int).columns.to_list()

# plt.figure(figsize=(16, 4))
# for i, col in enumerate(numerical_columns):
#     plt.subplot(8, 4, i+1)
#     sns.countplot(data=data, x=col, hue='customer_experience')
# plt.tight_layout()
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Extract numerical columns excluding `customer_experience`
numerical_columns = data.select_dtypes(include=['number']).columns

# Create box plots for each numerical column
plt.figure(figsize=(16, 6 * len(numerical_columns)))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 3, i)
    sns.boxplot(data=data, x='customer_experience', y=col, palette="muted")
    plt.title(f'{col} by Customer Experience', fontsize=14)
    plt.xlabel('Customer Experience', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.tight_layout()

plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Select columns to plot
columns_to_plot =  data.select_dtypes(include=['number']).columns


# Create a copy of the dataframe to avoid modifying the original
plot_data = data.copy()

# Convert customer_experience to float64
plot_data['customer_experience'] = plot_data['customer_experience'].astype('float64')

# Convert all columns to plot to float64
for col in columns_to_plot:
    plot_data[col] = plot_data[col].astype('float64')

# Create violin plots
plt.figure(figsize=(16, 6 * len(columns_to_plot)))

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(len(columns_to_plot), 1, i)
    sns.violinplot(data=plot_data, 
                  x='customer_experience', 
                  y=col,
                  palette="muted")
    plt.title(f'{col} by Customer Experience', fontsize=14)
    plt.xlabel('Customer Experience', fontsize=12)
    plt.ylabel(col, fontsize=12)
    
plt.tight_layout()
plt.show()


# Calculate the correlation matrix for all columns
cor_mat = data.corr(method="pearson")

# Extract the correlation values with 'customer_experience'
customer_experience_corr = cor_mat['customer_experience']

# Create a new DataFrame that includes the correlation with 'customer_experience' for all other columns
cor_mat_customer_experience = customer_experience_corr.drop('customer_experience')  # Drop the self-correlation

# Plot the correlation with 'customer_experience'
plt.figure(figsize=(10, 8))
sns.barplot(x=cor_mat_customer_experience.index, y=cor_mat_customer_experience.values, palette="coolwarm")
plt.xticks(rotation=45, ha='right')
plt.title("Correlation with Customer Experience")
plt.ylabel("Correlation")
plt.show()



# Sort the correlation values by absolute value in descending order
sorted_customer_experience_corr = cor_mat_customer_experience.abs().sort_values(ascending=False)

# Print the sorted correlation values
print("Correlation with Customer Experience (sorted by magnitude):")
print(sorted_customer_experience_corr)

# Plot the sorted correlation
plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_customer_experience_corr.index, y=sorted_customer_experience_corr.values, palette="coolwarm")
plt.xticks(rotation=45, ha='right')
plt.title("Correlation with Customer Experience (Sorted by Magnitude)")
plt.ylabel("Correlation")
plt.show()



# data.columns.to_list()


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    X = X.copy()
    X = X.dropna()  
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


data.columns


X = data.drop(['id','customer_experience','Date_Registered', 'payment_datetime', 'purchased_datetime', 
                    'released_date', 'estimated_delivery_date', 'received_date'], axis=1)
y = data['customer_experience']



mi_scores = make_mi_scores(X, y)

print(mi_scores)
# print(mi_scores.tail(20))  

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
# plot_mi_scores(mi_scores.tail(20)) 


selected_features=['loyalty_points_redeemed',
       'loyalty_tier', 'Received_tier_discount_percentage',
       'Received_card_discount_percentage',
       'Received_coupon_discount_percentage', 'Product_value',
        'final_payment','shipping_method', 'Gender_F', 'Gender_M',
       'Gender_O', 'Is_current_loyalty_program_member_NO',
       'Is_current_loyalty_program_member_YES', 'purchase_medium_in-store',
       'purchase_medium_online', 'total_discount_percentage',
       'Total_Discount_Amount', 'discount_amount_ratio', 'high_discount_order',
       'shipping_cost', 'shipping_cost_ratio', 'price_tier',
       'discount_types_used', 'loyalty_engagement_score', 'purchase_hour',
       'purchase_day_of_week', 'is_weekend_purchase', 'is_business_hours',
       'purchase_month', 'purchase_quarter', 'receive_day_of_week',
       'processing_days', 'delivery_days', 'total_order_days',
       'delivery_delay', 'is_delayed', 'customer_tenure_days', 'age_group',
       'is_first_purchase', 'purchase_count', 'product_category_encoded']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create feature set
X = data[selected_features].copy()

# Target variable
y = data['customer_experience']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

# Define models with optimized parameters
models = {
    'XGBoost': XGBClassifier(
        objective='multi:softmax', 
        num_class=3,
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss'
    ),
    'LightGBM': LGBMClassifier(
        objective='multiclass', 
        num_class=3,
        learning_rate=0.1,
        n_estimators=90,
        num_leaves=64,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42
    )
    # , 'Catboost' : CatBoostClassifier(
    #     loss_function='MultiClass', 
    #     random_state=42, 
    #     verbose=0
    # ),
    # 'MLP': MLPClassifier(
    #     random_state=42
    # ),
    # 'RandomForest': RandomForestClassifier(
    #     n_estimators=200,
    #     max_depth=10,
    #     min_samples_split=5,
    #     min_samples_leaf=2,
    #     random_state=42
    # ),
    # 'GradientBoosting': GradientBoostingClassifier(
    #     learning_rate=0.1,
    #     n_estimators=200,
    #     max_depth=5,
    #     min_samples_split=5,
    #     random_state=42
    # )
}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Print performance metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Print feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n{model_name} Feature Importances:")
        print(importances)
    elif hasattr(model, 'coef_'):
        # For models like Logistic Regression
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(model.coef_[0])  # Using absolute values for coefficient importance
        }).sort_values('importance', ascending=False)
        print(f"\n{model_name} Feature Importances (based on coefficients):")
        print(importances)

# Train and evaluate all models
for model_name, model in models.items():
    evaluate_model(model, X_train, X_test, y_train, y_test, model_name)


# from lightgbm import LGBMClassifier
# from sklearn.model_selection import RandomizedSearchCV

# # Define the parameter grid
# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'n_estimators': [50, 90, 120, 150],
#     'num_leaves': [31, 64, 128, 256],
#     'feature_fraction': [0.6, 0.7, 0.8, 0.9],
#     'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
#     'lambda_l1': [0.0, 0.1, 0.5, 1.0],
#     'lambda_l2': [0.0, 0.1, 0.5, 1.0],
#     'random_state': [42]
# }

# # Initialize the LightGBM classifier
# lgbm = LGBMClassifier(objective='multiclass', num_class=3)

# # Initialize RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=lgbm,
#     param_distributions=param_grid,
#     n_iter=50,  # Number of parameter combinations to try
#     scoring='accuracy',
#     cv=5,  # 5-fold cross-validation
#     verbose=1,
#     random_state=42,
#     n_jobs=-1  # Use all available processors
# )

# # Fit RandomizedSearchCV
# random_search.fit(X_train, y_train)

# # Print the best parameters and score
# print("Best Parameters:", random_search.best_params_)
# print("Best Accuracy:", random_search.best_score_)



# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# # Define base models (including LightGBM)

# base_models = [
#         ('xgb', XGBClassifier(
#             learning_rate=0.05,
#             n_estimators=300,
#             max_depth=6,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42,
#             n_jobs=-1
#         )),
#         ('lgb', LGBMClassifier(
#             learning_rate=0.05,
#             n_estimators=300,
#             num_leaves=31,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42,
#             n_jobs=-1
#         )),
#         ('rf', RandomForestClassifier(
#             n_estimators=300,
#             max_depth=10,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=42,
#             n_jobs=-1
#         ))
#     ]

# # Meta-model
# # meta_model = LogisticRegression(max_iter=1000)
# meta_model = LogisticRegression(max_iter=1000, penalty='l2', C=0.1) 
# # Meta model
# # meta_model = XGBClassifier(
# #     learning_rate=0.03,
# #     n_estimators=150,
# #     max_depth=4,
# #     subsample=0.8,
# #     colsample_bytree=0.8,
# #     random_state=42,
# #     n_jobs=-1
# # )

# # Create StackingClassifier
# stacking_clf = StackingClassifier(
#     estimators=base_models,
#     final_estimator=meta_model,
#     cv=5  # Cross-validation for better meta-model training
# )


# # Train StackingClassifier
# print("\nTraining StackingClassifier...")
# stacking_clf.fit(X_train, y_train)
# y_pred_stacking = stacking_clf.predict(X_test)

# # Evaluate performance
# print(f"Accuracy: {accuracy_score(y_test, y_pred_stacking):.4f}")
# print(f"F1 Score: {f1_score(y_test, y_pred_stacking, average='weighted'):.4f}")
# print("\nClassification Report:\n", classification_report(y_test, y_pred_stacking))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_stacking))



# final_model =  XGBClassifier(learning_rate=0.1,n_estimators=200,max_depth=5,random_state=42,eval_metric='mlogloss')
final_model = LGBMClassifier(
        objective='multiclass', 
        num_class=3,
        learning_rate=0.1,
        n_estimators=90,
        num_leaves=64,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42
    )

final_model.fit(X, y)



test = test[selected_features]
predictions = final_model.predict(test)
predictions


submission = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/sample_submission.csv')
submission['customer_experience'] = predictions


submission['customer_experience'] = submission['customer_experience'].replace(0,'bad')
submission['customer_experience'] = submission['customer_experience'].replace(1,'neutral')
submission['customer_experience'] = submission['customer_experience'].replace(2,'good')


submission.to_csv("submission.csv", index=False)

