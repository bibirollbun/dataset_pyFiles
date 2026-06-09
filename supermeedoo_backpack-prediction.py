import pandas as pd
filename = "/kaggle/input/playground-series-s5e2/train.csv"
df = pd.read_csv(filename)

X = df[['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof','Style','Color','Weight Capacity (kg)']]
y = df['Price']
df.head(5)


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# ØªØ­Ø¯ÙŠØ¯ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ù�Ø¦ÙˆÙŠØ©
categorical_features = ['Brand', 'Material', 'Size', 'Compartments', 
                        'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Ø¥Ù†Ø´Ø§Ø¡ OneHotEncoder Ù…Ø¹ Ø§Ø³ØªØ®Ø¯Ø§Ù… Ø§Ù„Ø¨Ø±Ø§Ù…ÙŠØªØ± Ø§Ù„Ø¬Ø¯ÙŠØ¯
encoder = OneHotEncoder(drop='first', sparse_output=False)

# ØªØ·Ø¨ÙŠÙ‚ Ø§Ù„ØªØ±Ù…ÙŠØ²
encoded_data = encoder.fit_transform(X[categorical_features])

# ØªØ­ÙˆÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ù…Ø´Ù�Ø±Ø© Ø¥Ù„Ù‰ DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

# Ø¯Ù…Ø¬ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ù…Ø´Ù�Ø±Ø© Ù…Ø¹ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ø¹Ø¯Ø¯ÙŠØ© Ø§Ù„Ù…ØªØ¨Ù‚ÙŠØ©
X_encoded = pd.concat([X.drop(columns=categorical_features).reset_index(drop=True), encoded_df], axis=1)

# Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 5 ØµÙ�ÙˆÙ� Ù„Ù„ØªØ£ÙƒØ¯
print(X_encoded.head(5))



from sklearn.preprocessing import MinMaxScaler

# Ø¥Ù†Ø´Ø§Ø¡ ÙƒØ§Ø¦Ù† MinMaxScaler Ù„ØªØ·Ø¨ÙŠØ¹ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¨ÙŠÙ† 0 Ùˆ 1
scaler = MinMaxScaler()

# ØªØ·Ø¨ÙŠÙ‚ Ø§Ù„ØªØ·Ø¨ÙŠØ¹ Ø¹Ù„Ù‰ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ù…Ø´Ù�Ø±Ø©
X_normalized = scaler.fit_transform(X_encoded)

# ØªØ­ÙˆÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ù…Ø·Ø¨ÙˆØ¹Ø© Ø¥Ù„Ù‰ DataFrame Ù„Ø³Ù‡ÙˆÙ„Ø© Ø§Ù„Ø¹Ø±Ø¶
X_normalized_df = pd.DataFrame(X_normalized, columns=X_encoded.columns)

# Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 5 ØµÙ�ÙˆÙ� Ø¨Ø¹Ø¯ Ø§Ù„ØªØ·Ø¨ÙŠØ¹
print(X_normalized_df.head())



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# 0ï¸�âƒ£ ØªØ¬Ù‡ÙŠØ² Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
features = ['Brand', 'Material', 'Size', 'Compartments', 
            'Laptop Compartment', 'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']
X = df[features].copy()  # âœ… Ù†Ø³Ø® Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ù„ØªØ¬Ù†Ø¨ Ø£ÙŠ ØªØ£Ø«ÙŠØ± Ø¹Ù„Ù‰ Ø§Ù„Ø£ØµÙ„
y = df['Price']

# 1ï¸�âƒ£ ØªØ¹ÙˆÙŠØ¶ Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ù…Ù�Ù‚ÙˆØ¯Ø©
# Ù„Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ù�Ø¦ÙˆÙŠØ©: Ø§Ø³ØªØ¨Ø¯Ø§Ù„ Ø¨Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ø£ÙƒØ«Ø± ØªÙƒØ±Ø§Ø±Ø§Ù‹
# Ù„Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ø¹Ø¯Ø¯ÙŠØ©: Ø§Ø³ØªØ¨Ø¯Ø§Ù„ Ø¨Ø§Ù„Ù…ØªÙˆØ³Ø·
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_features = ['Compartments', 'Weight Capacity (kg)']

# Ù…Ø¹Ø§Ù„Ø¬ Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ù�Ø¦ÙˆÙŠØ©
cat_imputer = SimpleImputer(strategy='most_frequent')
X.loc[:, categorical_features] = cat_imputer.fit_transform(X[categorical_features])

# Ù…Ø¹Ø§Ù„Ø¬ Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ø¹Ø¯Ø¯ÙŠØ©
num_imputer = SimpleImputer(strategy='mean')
X.loc[:, numerical_features] = num_imputer.fit_transform(X[numerical_features])

# 2ï¸�âƒ£ ØªØ±Ù…ÙŠØ² Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ù�Ø¦ÙˆÙŠØ© Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… OrdinalEncoder
encoder = OrdinalEncoder()
X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

# 3ï¸�âƒ£ ØªÙ‚Ø³ÙŠÙ… Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¥Ù„Ù‰ ØªØ¯Ø±ÙŠØ¨ ÙˆØ§Ø®ØªØ¨Ø§Ø±
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4ï¸�âƒ£ ØªØ·Ø¨ÙŠØ¹ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5ï¸�âƒ£ Ø¥Ù†Ø´Ø§Ø¡ Ù†Ù…ÙˆØ°Ø¬ KNN
k = 20  # ğŸ‘ˆ ÙŠÙ…ÙƒÙ†Ùƒ ØªØºÙŠÙŠØ± Ø¹Ø¯Ø¯ Ø§Ù„Ø¬ÙŠØ±Ø§Ù† Ù‡Ù†Ø§
knn = KNeighborsRegressor(n_neighbors=k)

# 6ï¸�âƒ£ ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
knn.fit(X_train_scaled, y_train)

# 7ï¸�âƒ£ Ø§Ù„ØªÙ†Ø¨Ø¤ Ø¨Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±ÙŠØ©
y_pred = knn.predict(X_test_scaled)

# 8ï¸�âƒ£ Ø§Ù„ØªÙ‚ÙŠÙŠÙ…
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"K = {k}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")



import pandas as pd

# ØªØ­ÙˆÙŠÙ„ X_train_scaled Ø¥Ù„Ù‰ DataFrame Ø¥Ø°Ø§ ÙƒØ§Ù† Ù…ØµÙ�ÙˆÙ�Ø© NumPy
if not isinstance(X_train_scaled, pd.DataFrame):
    X_train_scaled = pd.DataFrame(X_train_scaled)

# Ø§Ù„ØªØ£ÙƒØ¯ Ù…Ù† ØªØ·Ø§Ø¨Ù‚ Ø§Ù„Ø£Ø¨Ø¹Ø§Ø¯ Ø¨ÙŠÙ† X Ùˆ y
min_length = min(len(X_train_scaled), len(y_train))
X_train_synced = X_train_scaled.iloc[:min_length]
y_train_synced = y_train.iloc[:min_length]

# Ø¥Ø¹Ø§Ø¯Ø© ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø¨Ø¹Ø¯ Ø§Ù„Ù…Ø²Ø§Ù…Ù†Ø©
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_synced, y_train_synced)

# Ø§Ù„ØªÙ†Ø¨Ø¤
y_pred_rf = rf.predict(X_test_scaled)

# Ø§Ù„ØªÙ‚ÙŠÙŠÙ…
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error (MSE): {mse_rf}")
print(f"Random Forest R-squared (R2): {r2_rf}")



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ø¥Ù†Ø´Ø§Ø¡ Ù†Ù…ÙˆØ°Ø¬ XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)

# ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø¨Ø¹Ø¯ Ù…Ø²Ø§Ù…Ù†Ø© Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
xgb.fit(X_train_synced, y_train_synced)

# Ø§Ù„ØªÙ†Ø¨Ø¤
y_pred_xgb = xgb.predict(X_test_scaled)

# Ø§Ù„ØªÙ‚ÙŠÙŠÙ…
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Mean Squared Error (MSE): {mse_xgb}")
print(f"XGBoost R-squared (R2): {r2_xgb}")



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ø¥Ù†Ø´Ø§Ø¡ Ù†Ù…ÙˆØ°Ø¬ Ø§Ù„Ø§Ù†Ø­Ø¯Ø§Ø± Ø§Ù„Ø®Ø·ÙŠ
lr = LinearRegression()

# ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ù…ØªØ²Ø§Ù…Ù†Ø©
lr.fit(X_train_synced, y_train_synced)

# Ø§Ù„ØªÙ†Ø¨Ø¤ Ø¨Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±ÙŠØ©
y_pred_lr = lr.predict(X_test_scaled)

# Ø§Ù„ØªÙ‚ÙŠÙŠÙ…
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression R2: {r2_lr}")



from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ø¥Ù†Ø´Ø§Ø¡ TargetEncoder
encoder = TargetEncoder(cols=['Brand'])

# ØªØ­ÙˆÙŠÙ„ Ø§Ù„Ø¹Ù…ÙˆØ¯ Brand Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… Target Encoding
X_brand_encoded = encoder.fit_transform(df[['Brand']], df['Price'])
y = df['Price']

# ØªÙ‚Ø³ÙŠÙ… Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
X_train, X_test, y_train, y_test = train_test_split(X_brand_encoded, y, test_size=0.2, random_state=42)

# ØªØ¯Ø±ÙŠØ¨ Ù†Ù…ÙˆØ°Ø¬ Ø¨Ø³ÙŠØ· Ø¨Ø¹Ø¯ Ø§Ù„ØªØ±Ù…ÙŠØ²
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)

# Ø§Ù„ØªÙ†Ø¨Ø¤
y_pred = xgb.predict(X_test)

# Ø§Ù„ØªÙ‚ÙŠÙŠÙ…
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost with Target Encoding MSE: {mse}")
print(f"XGBoost with Target Encoding R2: {r2}")





