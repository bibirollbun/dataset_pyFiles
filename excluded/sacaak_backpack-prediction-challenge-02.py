import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df.head()


pd.set_option('display.max_columns', None) # It ensures that all columns are displayed in the output.
pd.set_option('display.width', 170) # It sets the maximum output width to 170 characters.
pd.set_option('display.float_format', lambda x: '%.3f' % x) # It displays decimal numbers with 3 decimal places. By default, pandas may use scientific notation (1.23e+03); this format is used to prevent it.
sns.set_style('darkgrid')


import warnings
warnings.filterwarnings("ignore")


df.info()


test_df.info()


df.shape


test_df.shape


df.drop(columns=["id"]).describe().T.style.background_gradient(axis=0, cmap='Purples')


test_df.drop(columns=["id"]).describe().T.style.background_gradient(axis=0, cmap='Purples')


print(df.isnull().mean().sort_values(ascending=False) * 100)


sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)


df.dropna(inplace=True)


df.isnull().sum()


print(test_df.isnull().mean().sort_values(ascending=False) * 100)


missing_values_cat = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values_cat)


most_common_waterproof = test_df['Waterproof'].mode()[0]
test_df['Waterproof'] = test_df['Waterproof'].fillna(most_common_waterproof)


from itertools import combinations
from scipy.stats import chi2_contingency

# All categoric variables
categorical_vars = ['Color', 'Brand', 'Material', 'Style', 'Size']

# List to store the results of the Chi-Square test
chi2_results = []

# Comparing all pairs of variables
for var1, var2 in combinations(categorical_vars, 2):
    contingency_table = test_df.pivot_table(index=var1, columns=var2, aggfunc='size', fill_value=0)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_results.append((var1, var2, p))

# Sort results by p-value
chi2_results.sort(key=lambda x: x[2])

# Print the most meaningful relationships on the screen
for var1, var2, p in chi2_results:
    print(f"{var1} ↔ {var2} | p-değeri: {p:.6f}")


# Find the most common Material for each Size
most_common_material_per_size = test_df.groupby('Size')['Material'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

# Fill in missing Material values according to Size
test_df['Material'] = test_df['Material'].fillna(test_df['Size'].map(most_common_material_per_size))


missing_values = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values)


most_common_color_per_material = test_df.groupby('Material')['Color'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Color'] = test_df['Color'].fillna(test_df['Material'].map(most_common_color_per_material))


missing_values = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values)


most_common_brand_per_material = test_df.groupby('Material')['Brand'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Brand'] = test_df['Brand'].fillna(test_df['Material'].map(most_common_brand_per_material))


missing_values = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values)


most_common_size_per_material = test_df.groupby('Material')['Size'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Size'] = test_df['Size'].fillna(test_df['Material'].map(most_common_size_per_material))


missing_values = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values)


most_common_style_per_color = test_df.groupby('Color')['Style'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Style'] = test_df['Style'].fillna(test_df['Color'].map(most_common_style_per_color))


missing_values = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values)


# 1. Fill Color by Brand
most_common_color_per_brand = test_df.groupby('Brand')['Color'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Color'] = test_df['Color'].fillna(test_df['Brand'].map(most_common_color_per_brand))

# 2. Fill Brand by Color
most_common_brand_per_color = test_df.groupby('Color')['Brand'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Brand'] = test_df['Brand'].fillna(test_df['Color'].map(most_common_brand_per_color))


missing_values = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values)


# 1. Fill Material by Color
most_common_material_per_color = test_df.groupby('Color')['Material'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Material'] = test_df['Material'].fillna(test_df['Color'].map(most_common_material_per_color))

# 2. Fill Size by Material
most_common_size_per_material = test_df.groupby('Material')['Size'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Size'] = test_df['Size'].fillna(test_df['Material'].map(most_common_size_per_material))


missing_values = test_df[['Color', 'Brand', 'Material', 'Style', 'Size']].isnull().sum()
print(missing_values)


test_df['Laptop Compartment'].isnull().sum()


most_common_laptop_compartment_per_color = test_df.groupby('Color')['Laptop Compartment'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

test_df['Laptop Compartment'] = test_df['Laptop Compartment'].fillna(test_df['Color'].map(most_common_laptop_compartment_per_color))

most_common_laptop_compartment_per_size = test_df.groupby('Size')['Laptop Compartment'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].fillna(test_df['Size'].map(most_common_laptop_compartment_per_size))

most_common_laptop_compartment_per_brand = test_df.groupby('Brand')['Laptop Compartment'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].fillna(test_df['Brand'].map(most_common_laptop_compartment_per_brand))

most_common_laptop_compartment_per_material = test_df.groupby('Material')['Laptop Compartment'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].fillna(test_df['Material'].map(most_common_laptop_compartment_per_material))

most_common_laptop_compartment_per_style = test_df.groupby('Style')['Laptop Compartment'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].fillna(test_df['Style'].map(most_common_laptop_compartment_per_style))


test_df['Laptop Compartment'].isnull().sum()


test_df['Weight Capacity (kg)'].isnull().sum()


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Cramér's V function
def cramers_v(x, y):
    contingency = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

categorical_cols = ['Color', 'Brand', 'Material', 'Style', 'Size']
for col in categorical_cols:
    v = cramers_v(test_df[col], test_df['Weight Capacity (kg)'])
    print(f"Cramér's V between {col} and Weight Capacity (kg): {v}")


most_common_weight_capacity_per_color = test_df.groupby('Color')['Weight Capacity (kg)'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

def fill_weight_capacity_based_on_color(row):
    if pd.isnull(row['Weight Capacity (kg)']):
        color = row['Color']
        # 'Color' değerine göre en yaygın 'Weight Capacity (kg)' değerini al
        return most_common_weight_capacity_per_color.get(color, None)
    return row['Weight Capacity (kg)']

test_df['Weight Capacity (kg)'] = test_df.apply(fill_weight_capacity_based_on_color, axis=1)


print(test_df.isnull().sum())


df.rename(columns={'Laptop Compartment':'Laptop_Compartment'}, inplace=True)
df.rename(columns={'Weight Capacity (kg)':'Weight_Capacity_(kg)'}, inplace=True)


df.columns


test_df.rename(columns={'Laptop Compartment':'Laptop_Compartment'}, inplace=True)
test_df.rename(columns={'Weight Capacity (kg)':'Weight_Capacity_(kg)'}, inplace=True)


test_df.columns


df.head()


df.Brand.value_counts()


df = pd.get_dummies(df, columns=["Brand"], drop_first=True, dtype=int)


df.head()


df.rename(columns={'Brand_Under Armour':'Brand_Under_Armour'}, inplace=True)


df.head()


test_df.head()


test_df = pd.get_dummies(test_df, columns=["Brand"], drop_first=True, dtype=int)


test_df.rename(columns={'Brand_Under Armour':'Brand_Under_Armour'}, inplace=True)


test_df.head()


df.Material.value_counts()


df = pd.get_dummies(df, columns=["Material"], drop_first=True, dtype=int)


df.head()


test_df = pd.get_dummies(test_df, columns=["Material"], drop_first=True, dtype=int)


test_df.head()


df['Size'].value_counts()


df = pd.get_dummies(df, columns=["Size"], drop_first=True, dtype=int)


df.head()


test_df = pd.get_dummies(test_df, columns=["Size"], drop_first=True, dtype=int)


test_df.head()


df['Laptop_Compartment'].value_counts()


df['Laptop_Compartment'] = df['Laptop_Compartment'].map({'Yes': 1, 'No': 0})


df.head()


test_df['Laptop_Compartment'].value_counts()


test_df['Laptop_Compartment'] = test_df['Laptop_Compartment'].map({'Yes': 1, 'No': 0})


test_df.head()


df.Waterproof.value_counts()


df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})


df.head()


test_df['Waterproof'] = test_df['Waterproof'].map({'Yes': 1, 'No': 0})


test_df.head()


df['Style'].value_counts()


df = pd.get_dummies(df, columns=["Style"], drop_first=True, dtype=int)


df.head()


test_df = pd.get_dummies(test_df, columns=["Style"], drop_first=True, dtype=int)


test_df.head()


from scipy.stats import chi2_contingency

crosstab = pd.crosstab(df['Color'], df['Price'])
chi2, p, dof, expected = chi2_contingency(crosstab)

print(f"Chi2: {chi2}, p-value: {p}")
if p < 0.05:
    print("There is a significant relationship between categorical variables.")
else:
    print("There is no significant relationship between categorical variables.")


sns.barplot(x='Color', y='Price', data=df)


df['Color'].value_counts()


df = pd.get_dummies(df, columns=["Color"], drop_first=True, dtype=int)


df.head()


test_df = pd.get_dummies(test_df, columns=["Color"], drop_first=True, dtype=int)


test_df.head()


from sklearn.ensemble import GradientBoostingRegressor


from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve


X = df.drop(columns=['Price', 'id'])
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=0)


gbm_model = GradientBoostingRegressor(random_state=17)


gbm_model.fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 5))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:5])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(gbm_model, X)


from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")


test_predictions = gbm_model.predict(test_df.drop(columns=['id']))

submission_df = pd.DataFrame({
    'id': test_df['id'],  
    'Price': test_predictions
})

submission_df.to_csv('submission.csv', index=False)

print("Submission file created and saved.")

