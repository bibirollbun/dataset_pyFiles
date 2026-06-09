import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df = df_train.copy()
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


df.head()


df_test.head()


def data_overview(df):
    from IPython.display import display, Markdown

    def print_md_title(title, emoji):
        display(Markdown(f"**{emoji} {title}**"))
        print("=" * 35)
        
    print()
    print_md_title("Duplicate Rows Check", "ğŸ—ƒï¸�")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        print("âœ… No duplicate rows found.")
    else:
        print(f"âš ï¸� Found {dup_count} duplicate rows. Dropping them...")
        df.drop_duplicates(keep="first", inplace=True)
        print("âœ… Duplicate rows dropped.")
    
    print()
    print_md_title("Shape & Columns", "ğŸ§¾")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print()
    print_md_title("Dataset Info", "ğŸ“‹")
    print()
    df.info()

    print()
    print_md_title("Numerical Features Summary", "ğŸ”¢")
    display(df.describe().T)

    print()
    print_md_title("Categorical Features Summary", "ğŸ”¤")
    display(df.describe(include="object").T)

    print()
    print_md_title("Missing Values", "â�“")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    total_cells = df.size
    missing_percentage = (total_missing / total_cells) * 100

    if total_missing == 0:
        print("âœ… No missing values found.")
    else:
        print(f"âš ï¸� Missing values detected:")
        print(missing[missing > 0])
        print(f"\nTotal Missing: {total_missing} values ({missing_percentage:.2f}% of the dataset)")


data_overview(df)


data_overview(df_test)


# Numerical features
numeric_columns = df.select_dtypes(include=['number']).columns

# Categorical features
categoric_features = df.select_dtypes(include=['object', 'category']).columns.tolist()


# Calculating gender distribution
sex_counts = df['Sex'].value_counts()

# Color palette
colors = ['#FFA500', '#FFCC99']

# Plotting pie chart
plt.figure(figsize=(5, 5))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.2f%%',
        startangle=90, colors=colors, textprops={'fontsize': 12})
plt.title('Distribution of Sex', fontsize=14)
plt.axis('equal')
plt.show()


import math

# Determining grid dimensions
n_cols = 2
n_rows = math.ceil(len(numeric_columns) / n_cols)

# Creating figure ve axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

# Plotting for each numeric column
for i, col in enumerate(numeric_columns):
    sns.histplot(df[col], kde=True, color='orange', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

# Hiding any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Dropping id column and creating a new numeric_columns (num_cols) list
num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
n_cols = 2
n_rows = 4

# Color palette
palette = plt.get_cmap('Set2').colors

# Creating subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
axes = axes.flatten()

# Plotting for each numeric column
for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i], color=palette[i % len(palette)])
    axes[i].set_title(f'Boxplot of {col}', fontsize=12)

# Hiding any empty subplots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


def detect_outliers(df, col_name,tukey=1.5):
    ''' 
    this function detects outliers based on 1.5 time IQR and
    returns the number of lower and uper limit and number of outliers respectively
    '''
    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
                      
    upper_limit = third_quartile+(tukey*IQR)
    lower_limit = first_quartile-(tukey*IQR)
    outlier_count = 0
                      
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count +=1
    return lower_limit, upper_limit, outlier_count


threshold = 1.5
out_cols = []

for col in numeric_columns:
    print(
        f"{col}\nlower:{detect_outliers(df, col,threshold)[0]} \nupper:{detect_outliers(df, col,threshold)[1]}\
        \noutlier:{detect_outliers(df, col,threshold)[2]}\n*-*-*-*-*-*-*"
    )
    if detect_outliers(df, col,threshold)[2] > 0 :
        out_cols.append(col)
print(out_cols)  


# Capping outliers
for col in list(numeric_columns):
    lower, upper, _ = detect_outliers(df, col, threshold)
    df[col] = np.clip(df[col], lower, upper)


# After capping outliers

# Creating subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
axes = axes.flatten()

# Plotting for each numeric column
for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i], color=palette[i % len(palette)])
    axes[i].set_title(f'Boxplot of {col}', fontsize=12)

# Hiding any empty subplots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


n_cols = 2
n_rows = 3

# Creating subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
axes = axes.flatten()

# Plotting for each numeric column
for i, col in enumerate(num_cols):
    if col != 'Calories':
        sns.scatterplot(x=df[col], y=df["Calories"], alpha=0.5, ax=axes[i])
        axes[i].set_title(f'{col} vs Calories')

# Hiding any empty subplots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# BMI Column
df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
df_test['BMI'] = df_test['Weight'] / (df_test['Height'] / 100) ** 2


# Age_Range Column
bins = [0, 18, 30, 45, 100]
# labels = ['18-30', '31-45', '46-60', '60+']
labels = ['Young', 'Middle-aged', 'Older', 'Senior']

df['Age_Range'] = pd.cut(df['Age'], bins=bins, labels=labels)
df_test['Age_Range'] = pd.cut(df_test['Age'], bins=bins, labels=labels)


# Heart_Rate_to_Weight Column
df['Heart_Rate_to_Weight'] = df['Heart_Rate'] / df['Weight']
df_test['Heart_Rate_to_Weight'] = df_test['Heart_Rate'] / df_test['Weight']


# Duration_to_Weight Column
df['Duration_to_Weight'] = df['Duration'] / df['Weight']
df_test['Duration_to_Weight'] = df_test['Duration'] / df_test['Weight']


df.head()


df_test.head()


# saving clean data
df.to_csv("train_clean.csv", index=False)
df_test.to_csv("test_clean.csv", index=False)


df_train = pd.read_csv("/kaggle/working/train_clean.csv")
df = df_train.copy()
df_test = pd.read_csv("/kaggle/working/test_clean.csv")


# dropping unnecessary features from df and df_test
df.drop(columns="id", inplace=True)
df_test.drop(columns="id", inplace=True)


X = df.drop(columns="Calories")
y = df.Calories


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


categoric_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categoric_features


# Sex column
X_train = pd.get_dummies(X_train, columns=['Sex'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['Sex'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Sex'], drop_first=True)

# Making the True/False values â€‹â€‹of the Sex column 1 and 0 after encoding
X_train['Sex_male'] = X_train['Sex_male'].astype(int)
X_test['Sex_male'] = X_test['Sex_male'].astype(int)
df_test['Sex_male'] = df_test['Sex_male'].astype(int)


# Mapping the Age_Range column ordinally
age_map = {'Young': 0, 'Middle-aged': 1, 'Older': 2, 'Senior': 3}

X_train['Age_Range'] = X_train['Age_Range'].map(age_map)
X_test['Age_Range'] = X_test['Age_Range'].map(age_map)
df_test['Age_Range'] = df_test['Age_Range'].map(age_map)


X_train.head()


X_test.head()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Convert to pandas format
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
test = pd.DataFrame(df_test)

# KFold parameters
n_splits = 8
SEED = 101
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


# XGB Model 1 - Kaggle Score: 0.15563
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 5000,
    'learning_rate': 0.08,
    'max_depth': 15,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 1,
    'reg_lambda': 8,
    'random_state': SEED,
    'tree_method': 'auto'
}

xgb_scores = []
xgb_test_preds = []

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model1 = xgb.XGBRegressor(**xgb_params)
    model1.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=100,
        verbose=1000
    )
    
    val_pred = model1.predict(X_val, iteration_range=(0, model1.best_iteration + 1))
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    xgb_scores.append(rmse)
    
    test_pred = np.maximum(model1.predict(test, iteration_range=(0, model1.best_iteration + 1)), 0)
    xgb_test_preds.append(test_pred)
    
    print(f"Fold {i+1} RMSE: {rmse:.4f}")

print(f"\nXGBoost Mean RMSE: {np.mean(xgb_scores):.4f}")


# Since the predictions made on the test set are taken as many predictions with KFold,
# the final predictions should be taken as an average.
final_xgb_test_pred = np.mean(xgb_test_preds, axis=0)


# create submission_xgb file
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission["Calories"] = final_xgb_test_pred


submission.head(5)


# save submission_xgb file
submission.to_csv("submission_xgb1.csv", index=False)


# XGB Model 2 - Kaggle Score: 0.05945
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 3000,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 8,
    'random_state': SEED,
    'tree_method': 'auto'
}

xgb_scores = []
xgb_test_preds = []

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model2 = xgb.XGBRegressor(**xgb_params)
    model2.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=100,
        verbose=1000
    )
    
    val_pred = model2.predict(X_val, iteration_range=(0, model2.best_iteration + 1))
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    xgb_scores.append(rmse)
    
    test_pred = np.maximum(model2.predict(test, iteration_range=(0, model2.best_iteration + 1)), 0)
    xgb_test_preds.append(test_pred)
    
    print(f"Fold {i+1} RMSE: {rmse:.4f}")

print(f"\nXGBoost Mean RMSE: {np.mean(xgb_scores):.4f}")


final_xgb_test_pred = np.mean(xgb_test_preds, axis=0)

submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission["Calories"] = final_xgb_test_pred

submission.to_csv("submission_xgb2.csv", index=False)


# XGB Model 3 - Kaggle Score: 0.05937
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 4000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 2,
    'reg_lambda': 10,
    'random_state': SEED
}

xgb_scores = []
xgb_test_preds = []

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model3 = xgb.XGBRegressor(**xgb_params)
    model3.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=100,
        verbose=1000
    )
    
    val_pred = model3.predict(X_val, iteration_range=(0, model3.best_iteration + 1))
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    xgb_scores.append(rmse)
    
    test_pred = np.maximum(model3.predict(test, iteration_range=(0, model3.best_iteration + 1)), 0)
    xgb_test_preds.append(test_pred)
    
    print(f"Fold {i+1} RMSE: {rmse:.4f}")

print(f"\nXGBoost Mean RMSE: {np.mean(xgb_scores):.4f}")


final_xgb_test_pred = np.mean(xgb_test_preds, axis=0)

submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission["Calories"] = final_xgb_test_pred

submission.to_csv("submission_xgb3.csv", index=False)


# LightGBM Model 1 - Kaggle Score: 0.06000
import lightgbm as lgb

rmse_scores = []
lgb_test_preds = []

# K-Fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nğŸ”� Fold {fold+1}")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_lgbm = lgb.LGBMRegressor(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )

    model_lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    val_preds = model_lgbm.predict(X_val)
    val_rmse = mean_squared_error(y_val, val_preds, squared=False)
    rmse_scores.append(val_rmse)
    print(f"ğŸ“‰ Fold {fold+1} RMSE: {val_rmse:.4f}")

    # Her fold iÃ§in test setine tahmin
    test_preds = model_lgbm.predict(test)
    lgb_test_preds.append(test_preds)

# TÃ¼m fold'larÄ±n RMSE ortalamasÄ± ve sapmasÄ±
print(f"\nâœ… Ortalama RMSE: {np.mean(rmse_scores):.4f}")
print(f"ğŸ“Š Std. Sapma: {np.std(rmse_scores):.4f}")


final_lgbm_test_pred = np.mean(lgb_test_preds, axis=0)
final_lgbm_test_pred = np.abs(final_lgbm_test_pred)

submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission["Calories"] = final_lgbm_test_pred

submission.to_csv("submission_lgbm1.csv", index=False)


# ANN Model 1 - Kaggle Score: 0.05995

from tensorflow.keras import layers, models, callbacks, optimizers

# MinMax Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling
ann_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

ann_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Callbacks
early_stop = callbacks.EarlyStopping(patience=25, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# Model fitting
history = ann_model.fit(
    X_scaled, y_train,
    validation_split=0.1,
    epochs=300,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# Train RMSE
train_rmse = ann_model.evaluate(X_scaled, y_train, verbose=0)[1]

# Prediction with X_test
test_pred_ann = ann_model.predict(X_test_scaled).flatten()

# Scores
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, test_pred_ann, squared=False)
print(f"Test RMSE: {rmse:.4f}")


# Scaling for df_test
test_scaled = scaler.transform(df_test)

# Prediction with df_test
test_pred_ann = ann_model.predict(test_scaled).flatten()
test_pred_ann = np.clip(test_pred_ann, 0, None)


# Submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission["Calories"] = test_pred_ann
submission.to_csv("submission_ann.csv", index=False)

