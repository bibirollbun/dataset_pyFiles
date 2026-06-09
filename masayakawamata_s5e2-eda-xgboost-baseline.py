# Import Library
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')

display(train.head(3))
display(test.head(3))


print(f'Train has {train.shape[0]} rows, {train.shape[1]} columns')
print(f'Test  has {test.shape[0]} rows,  {test.shape[1]} columns')


train.describe()


test.describe()


analysis = train.copy()


analysis.isnull().sum()


categories = ["Brand", "Material", "Size", "Style", "Color"]

for category in categories:
    unique_values = analysis[category].unique()
    print(f'{category} categories: {unique_values}')


categories = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", 
              "Waterproof", "Style", "Color"]

for category in categories:
    analysis[category] = analysis[category].fillna("Missing")
    stats = analysis.groupby(category)["Price"].agg(["count", "mean", "median", "min", "max"]).sort_values(by="mean", ascending=False)
    missing_count = (analysis[category] == "Missing").sum()
    
    print(f"\n=== {category} vs Price ===")
    print(stats)
    print(f"Missing values: {missing_count}")

    plt.figure(figsize=(10, 5))
    stats["mean"].plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Average Price by {category}")
    plt.xlabel(category)
    plt.ylabel("Average Price")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


df = analysis[['Weight Capacity (kg)', 'Price']].dropna()

correlation = df.corr().iloc[0, 1]
print(f"Correlation between Weight Capacity (kg) and Price: {correlation:.4f}")

sample_size = 5000
df_sample = df.sample(n=sample_size, random_state=42)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_sample, x="Weight Capacity (kg)", y="Price", alpha=0.3)
sns.regplot(data=df_sample, x="Weight Capacity (kg)", y="Price", scatter=False, color="red")
plt.title(f"Scatter Plot: Weight Capacity (kg) vs Price (Corr={correlation:.4f})")
plt.xlabel("Weight Capacity (kg)")
plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


plt.figure(figsize=(8, 6))
plt.hexbin(df["Weight Capacity (kg)"], df["Price"], gridsize=50, cmap="Blues", mincnt=1)
plt.colorbar(label="Count in bin")
plt.title(f"2D Histogram: Weight Capacity (kg) vs Price (Corr={correlation:.4f})")
plt.xlabel("Weight Capacity (kg)")
plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


categories = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", 
              "Waterproof", "Style", "Color"]

fig, axes = plt.subplots(nrows=len(categories), ncols=2, figsize=(15, 5 * len(categories)))

for i, category in enumerate(categories):
    ax1 = axes[i, 0]  # KDEプロット用
    ax2 = axes[i, 1]  # ボックスプロット用
    
    # KDEプロット（カテゴリごとの価格分布）
    sns.kdeplot(data=analysis, x="Price", hue=category, fill=True, alpha=0.3, common_norm=False, ax=ax1)
    ax1.set_title(f"Price Distribution by {category}")
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Density")
    ax1.set_xscale("log")  # ログスケールで価格のばらつきを見やすく
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ボックスプロット（カテゴリごとの価格の分布）
    sns.boxplot(data=analysis, y=category, x="Price", palette="Set3", orient="h", ax=ax2)
    ax2.set_title(f"Price Boxplot by {category}")
    ax2.set_xlabel("Price")
    ax2.set_xscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)


plt.tight_layout()
plt.show()


categories = [
              "Brand", "Material", "Size", "Compartments", 
              "Laptop Compartment", "Waterproof", "Style", "Color"
             ]

for col in categories:
    train[col] = train[col].fillna('missing').astype('category')
    test[col] = test[col].fillna('missing').astype('category')


X = train.drop(columns=["Price"]).copy()
y = train["Price"].copy()


from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm  # Visualize Progress Bar

kf = KFold(n_splits=10, shuffle=True, random_state=42)

model_params = {
    "enable_categorical": True,
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

models = []
oof_predictions = np.zeros(len(train))
rmse_scores = []

print("Starting Cross-Validation...\n")

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=kf.get_n_splits(), desc="Cross-Validation Progress")):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    
    y_pred = model.predict(X_val)
    oof_predictions[val_idx] = y_pred
    models.append(model)  

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    
    print(f"Fold {fold+1}/{kf.get_n_splits()} - RMSE: {rmse:.4f}")

train["oof_predictions"] = oof_predictions

mean_rmse = np.mean(rmse_scores)
print(f"\nMean RMSE: {mean_rmse:.4f}")


results_df = pd.DataFrame({
    "Actual Price": train["Price"],
    "OOF Predicted Price": train["oof_predictions"]
})

print("\nFirst 10 OOF Predictions vs Actual Prices:")
display(results_df.head(10))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=results_df["Actual Price"], y=results_df["OOF Predicted Price"], alpha=0.5)
sns.regplot(x=results_df["Actual Price"], y=results_df["OOF Predicted Price"], scatter=False, color="red")

plt.title("Actual Price vs OOF Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("OOF Predicted Price")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


print("\nGenerating Test Predictions using 10 Fold Models Ensemble...")

test_preds = np.mean([model.predict(test) for model in models], axis=0)
test_results_df = pd.DataFrame({"Predicted Price": test_preds})

print("\nFirst 10 Predictions for Test Data:")
display(test_results_df.head(10))


sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sub['Price'] = test_preds
sub.to_csv('submission.csv', index=False)
sub.head(5)




