import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error





df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df.head(4)
train=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")



plt.figure(figsize=(12, 6))
ax = sns.countplot(y=df["Brand"], order=df["Brand"].value_counts().index, palette="coolwarm")

for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge", padding=4)

plt.xlabel("Count of Backpacks")
plt.ylabel("Brand")
plt.title("Number of Backpacks per Brand")
plt.show()


plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=df, x="Price", y="Brand", palette="muted", inner="quartile")

# Get unique brands in the correct order as they appear in the plot
brands = df["Brand"].unique()

for brand in brands:
    subset = df[df["Brand"] == brand]["Price"]
    
    Min = subset.min()  # Minimum price
    q1 = subset.quantile(0.25)  # First quartile (25th percentile)
    median = subset.median()  # Median (50th percentile)
    q3 = subset.quantile(0.75)  # Third quartile (75th percentile)
    Max = subset.max()  # Maximum price
    
    # Get the y position for the brand
    y_pos = list(df["Brand"].unique()).index(brand)
    
    # Annotate Min, Q1, Median, Q3, and Max values
    ax.text(Min, y_pos, f'Min: {Min:.2f}', ha='left', va='center', fontsize=10, color='blue')
    ax.text(q1, y_pos, f'Q1: {q1:.2f}', ha='right', va='center', fontsize=10, color='black')
    ax.text(median, y_pos, f'Median: {median:.2f}', ha='right', va='center', fontsize=10, color='red', fontweight='bold')
    ax.text(q3, y_pos, f'Q3: {q3:.2f}', ha='right', va='center', fontsize=10, color='black')
    ax.text(Max, y_pos, f'Max: {Max:.2f}', ha='right', va='center', fontsize=10, color='blue')

plt.xlabel("Price")
plt.ylabel("Brand")
plt.title("Price Distribution for Different Brands (With Quartiles & Min/Max)")
plt.show()




# Compute the crosstab and normalize by row to get percentages
material_pct = pd.crosstab(df["Brand"], df["Material"], normalize="index") * 100

# Plot the stacked bar chart with percentage values
plt.figure(figsize=(12, 6))
ax = material_pct.plot(kind="bar", stacked=True, colormap="viridis", figsize=(12, 6))

# Annotate the bars with percentage values
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=10, color="white", weight="bold")

plt.xlabel("Brand")
plt.ylabel("Percentage (%)")
plt.title("Material Distribution Across Brands (in %)")
plt.legend(title="Material")
plt.show()



plt.figure(figsize=(14, 6))

# Compute the average price for each Brand-Material combination
avg_price = df.groupby(["Brand", "Material"])["Price"].mean().reset_index()

# Create a grouped bar plot
sns.barplot(data=avg_price, x="Brand", y="Price", hue="Material", palette="tab10")

plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.title("Average Price of Each Material Across Brands")
plt.legend(title="Material", bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)  
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Weight Capacity (kg)", y="Brand", hue="Brand", palette="tab10")
plt.xlabel("Weight Capacity (kg)")
plt.ylabel("Brand")
plt.title("Weight Capacity Distribution Across Brands")
plt.legend(title="Brand", bbox_to_anchor=(1, 1))
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Size", y="Price", palette="muted", inner="quartile")
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Price Distribution for Different Backpack Sizes")
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Compartments", y="Price", hue="Brand", marker="o", palette="tab10")
plt.xlabel("Number of Compartments")
plt.ylabel("Average Price")
plt.title("Average Price by Number of Compartments Across Brands")
plt.legend(title="Brand", bbox_to_anchor=(1, 1))
plt.show()




from scipy.stats import ttest_ind, mannwhitneyu

# Split price data based on laptop compartment availability
price_with_laptop = df[df["Laptop Compartment"] == "Yes"]["Price"]
price_without_laptop = df[df["Laptop Compartment"] == "No"]["Price"]


# If non-normal, use Mann-Whitney U test
mw_pval = mannwhitneyu(price_with_laptop, price_without_laptop).pvalue
print(f"Mann-Whitney U p-value: {mw_pval:.5f}")



# Merge Training Data
full_train = pd.concat([train, train_extra], ignore_index=True)

# Remove Missing Data
full_train = full_train.dropna()
#test = test.dropna()

# Identify Categorical Columns
categorical_cols = full_train.select_dtypes(include=["object"]).columns

# Convert Categorical Columns to Numeric using Label Encoding
label_encoders = {}  # Store encoders for future use
for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit on full training data and transform
    full_train[col] = le.fit_transform(full_train[col])
    
    # Check if the column exists in test and transform
    if col in test.columns:
        test[col] = test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)  # Handle unseen values
    label_encoders[col] = le  # Save the encoder for potential reverse mapping





target = "Price"
features = [col for col in full_train.columns if col not in ["id", target]]  

X = full_train[features]
y = full_train[target]
X_test = test[features]  # Ensure X_test has the same structure as X_train

# Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=1, random_state=42)







xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)




valid_predictions = xgb_model.predict(X_valid)
mse = mean_squared_error(y_valid, valid_predictions)
rmse = np.sqrt(mse)

print(f" Mean Squared Error (MSE): {mse:.4f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")

# Predict on Test Data
test_predictions = xgb_model.predict(X_test)


submission = pd.DataFrame({"id": test["id"], "Predicted_Price": test_predictions})



submission.head(15)



submission.to_csv("/kaggle/working/submission.csv", index=False)




























