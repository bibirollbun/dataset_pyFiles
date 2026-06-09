import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns, plotly.express as px

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


target_column = (set(train_df.columns) - set(test_df.columns)).pop()

print(f"Target column: {target_column}\nData type: {train_df[target_column].dtype}")


print(f"The Rows Train dataset contains : {train_df.shape[0]}\nThe Columns dataset contains : {train_df.shape[1]}")
print("-"*50)
print(f"The Rows Test dataset contains : {test_df.shape[0]}\nThe Columns dataset contains : {test_df.shape[1]}")


print(train_df.columns)
print("-"*50)
print(test_df.columns)


def generate_null_analysis(df):
    count = df.isnull().sum()
    percen = count / len(df) * 100
    
    df_null = pd.DataFrame({
        'column name': df.columns,
        'total count': count,
        'percentage': percen
    })
    
    df_null.reset_index(drop = True, inplace = True)
    df_null_sorted = df_null.sort_values(by = 'percentage', ascending = False)
    df_filtered = df_null_sorted[df_null_sorted['percentage'] > 0]
    df_filtered.reset_index(drop = True, inplace = True)
    
    return df_filtered

df_filtered_train = generate_null_analysis(train_df)
df_filtered_test = generate_null_analysis(test_df)

def style_null_analysis(df):
    return df.style.background_gradient(cmap = 'YlOrRd', subset = ['percentage', 'total count'])

df_filtered_train_styled = style_null_analysis(df_filtered_train)
df_filtered_test_styled = style_null_analysis(df_filtered_test)


display(df_filtered_train_styled)


display(df_filtered_test_styled)


display(train_df.head(1))
display(test_df.head(1))


train_df[train_df[['Guest_Popularity_percentage', 'Episode_Length_minutes']].isnull().all(axis=1)].head()


train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mean(), inplace=True)  
test_df['Number_of_Ads'].fillna(test_df['Number_of_Ads'].mean(), inplace=True)  


train_df.describe()


filtered_df = train_df[(train_df['Podcast_Name'] == 'Mystery Matters') & (train_df['Episode_Title'] == 'Episode 98') & (train_df['Publication_Time'] == 'Night')]


display(filtered_df.head())
avg_episode_length = filtered_df['Episode_Length_minutes'].mean()
avg_guest_popularity = filtered_df['Guest_Popularity_percentage'].mean()

print(f"Average Episode Length: {avg_episode_length:.2f} minutes")
print(f"Average Guest Popularity: {avg_guest_popularity:.2f}%")


group_cols = ['Podcast_Name', 'Episode_Title', 'Publication_Time']

train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(
    train_df.groupby(group_cols)['Episode_Length_minutes'].transform('mean')
)

train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(
    train_df.groupby(group_cols)['Guest_Popularity_percentage'].transform('mean')
)

test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(
    test_df.groupby(group_cols)['Episode_Length_minutes'].transform('mean')
)

test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(
    test_df.groupby(group_cols)['Guest_Popularity_percentage'].transform('mean')
)


test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].mean(), inplace = True)


train_df.head(1)


train_df['Genre'].value_counts()


print("Skewness:", train_df['Listening_Time_minutes'].skew())


corr_matrix = train_df.select_dtypes(include=['int64', 'float64']).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="Greens", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


from sklearn.preprocessing import LabelEncoder

# Copy to avoid changing the original DataFrames (optional)
train_encoded = train_df.copy()
test_encoded = test_df.copy()

# Detect categorical columns
categorical_cols = train_encoded.select_dtypes(include=['object', 'category']).columns

# Dictionary to store encoders for each column
encoders = {}

# Fit encoders on train, then transform train and test
for col in categorical_cols:
    le = LabelEncoder()
    
    # Combine train and test values to handle unseen categories in test
    combined_data = pd.concat([train_encoded[col], test_encoded[col]], axis=0).astype(str).fillna('Unknown')
    
    le.fit(combined_data)
    
    # Save the encoder
    encoders[col] = le
    
    # Fill and transform
    train_encoded[col] = le.transform(train_encoded[col].astype(str).fillna('Unknown'))
    test_encoded[col] = le.transform(test_encoded[col].astype(str).fillna('Unknown'))



from sklearn.model_selection import train_test_split

# Define the target column
target_column = 'Listening_Time_minutes'

# Split into X and y
X = train_encoded.drop(columns=[target_column])
y = train_encoded[target_column]

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42  # 80/20 split with reproducibility
)



from sklearn.preprocessing import StandardScaler

test_ids = test_encoded['id'].copy()

X_train = X_train.drop(columns=['id'])
X_val = X_val.drop(columns=['id'])
X_test = test_encoded.drop(columns=['id'], errors='ignore')

# Step 2: Normalize the data
scaler = StandardScaler()

# Fit on training data only, then transform train, val, and test
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)





from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)

y_val_pred = lr_model.predict(X_val_scaled)

mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation R² Score: {r2:.4f}")



y_val_pred = lr_model.predict(X_val_scaled)

mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation R² Score: {r2:.4f}")



X_test_scaled = np.where(np.isnan(X_test_scaled), np.nanmean(X_test_scaled, axis=0), X_test_scaled)


y_test_pred = lr_model.predict(X_test_scaled)


# Ensure you still have access to the original test data to get 'id'
submission = pd.DataFrame({
    'id': test_encoded['id'], 
    'Listening_Time_minutes': y_test_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("✅ Submission file 'submission.csv' created successfully.")



































