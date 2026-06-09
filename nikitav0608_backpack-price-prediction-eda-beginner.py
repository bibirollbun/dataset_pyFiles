import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Importing Libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt      
import seaborn as sns   
import math
from sklearn.model_selection import train_test_split     
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder    
from sklearn.linear_model import LinearRegression      
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer   
from sklearn.impute import SimpleImputer   
import scipy.stats as stats
from sklearn.pipeline import Pipeline,make_pipeline



train_data=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


train_data.shape


train_data.info()


train_data.head()


train_data.isnull().mean()*100


train_data.duplicated().sum()


train_data.describe()


train_data.describe(include ='object')


#Just to reduce the size of the file as Compartments Columns has integer values. 
train_data['Compartments'] = train_data['Compartments'].astype(int)


plt.figure(figsize=(6,3))      
plt.title("Distribution of Brands")      
sns.countplot(data = train_data,x='Brand',palette='magma')      
plt.show()


plt.figure(figsize=(6,3))      
plt.title("Distribution of Different Materials")      
sns.countplot(data = train_data ,x='Material',palette='twilight_shifted_r')      
plt.show()


plt.figure(figsize=(6,3))      
plt.title("Distribution of Different Styles of Backpacks")      
sns.countplot(data = train_data ,x='Style' ,palette='pastel')      
plt.show()


plt.figure(figsize=(6,3))      
plt.title("Laptop Compartment Backpacks")      
sns.countplot(data = train_data,x='Laptop Compartment',palette='twilight_shifted_r')      
plt.show()


plt.figure(figsize=(6,3))      
plt.title("Distribution of Size of Backpacks")      
sns.countplot(data = train_data ,x='Size',palette='twilight_shifted')      
plt.show()


plt.figure(figsize=(6,3))      
plt.title("Waterproof Backpacks")      
sns.countplot(data = train_data ,x='Waterproof',palette='pastel')      
plt.show()


plt.figure(figsize=(6,3))      
plt.title("Distribution of Different Compartment Backpacks")      
sns.countplot(data = train_data,x='Compartments',palette='pastel')      
plt.show()


numeric_col = train_data[['Weight Capacity (kg)','Price']]


print(numeric_col)


for col in numeric_col.columns:      
    print("Skewness in the feature",col,' = ',numeric_col[col].skew())


#Checking the distribution and how normal is the numeric features graphically      
for col in numeric_col.columns:      
    plt.figure(figsize=(14,4))      
    plt.subplot(121)      
    sns.histplot(numeric_col[col],bins=10,color='grey',kde = True)      
    plt.title(col)      
        
    plt.subplot(122)      
    stats.probplot(numeric_col[col],dist="norm" ,plot=plt)      
    plt.title(col)      

    plt.show()


for col in numeric_col.columns:      
    plt.figure(figsize=(6,3))         
    sns.boxplot(numeric_col[col],color = 'pink')      
    plt.title("Boxplot of "+col)   


plt.figure(figsize=(4,4))      
plt.title('Correlation Matrix of Weight Capacity and Price')      
sns.heatmap(numeric_col.corr(),fmt='.2f',linewidths=2,annot=True)      
plt.show()


pivot_df = train_data.pivot_table(index='Brand',columns='Compartments',aggfunc='size',fill_value=0)      
plt.figure(figsize=(8,5))      
sns.heatmap(pivot_df,cmap='Pastel1_r',annot=True,fmt="d",linewidths=1)      
plt.xlabel("Compartments")      
plt.ylabel("Brand")      
plt.title("Brand-wise Heatmap of Backpack Compartments")      
plt.show()


pivot_df = train_data.pivot_table(index='Brand',columns='Material',aggfunc='size',fill_value=0)      
plt.figure(figsize=(8,5))      
sns.heatmap(pivot_df,cmap='twilight_shifted',annot=True,fmt="d",linewidths=2)      
plt.xlabel("Compartments")      
plt.ylabel("Brand")      
plt.title("Brand-wise Heatmap of Color of Backpacks")      
plt.show()


pivot_df = train_data.pivot_table(index='Brand',columns='Size',aggfunc='size',fill_value=0)      
plt.figure(figsize=(8,5))      
sns.heatmap(pivot_df,cmap='BuPu',annot=True,fmt="d" ,linewidths=2)      
plt.xlabel("Compartments")      
plt.ylabel("Brand")      
plt.title("Brand-wise Heatmap of Size of Backpacks")      
plt.show()


plt.figure(figsize=(8,5))      
sns.boxplot(x='Brand',y='Weight Capacity (kg)',data=train_data,palette='viridis')      
plt.xlabel('Brand')      
plt.ylabel('Weight Capacity (kg)')      
plt.title('Brand-wise Weight Capacity Distribution')      
plt.show()


brand_max_weight = train_data.groupby('Brand')['Weight Capacity (kg)'].max()      
brand_max_price = train_data.groupby('Brand')['Price'].max()      
    
Brand_summary = pd.concat([brand_max_weight,brand_max_price],axis=1)      
 
Brand_summary.columns = ['Max Weight Capacity (kg)','Max Price']      

print(Brand_summary)


material_avg_price = train_data.groupby('Material')['Price'].mean()      
material_avg_weight_capacity = train_data.groupby('Material')['Weight Capacity (kg)'].mean()      

material_summary = pd.concat([material_avg_weight_capacity,material_avg_price],axis=1)      

material_summary.columns = ['Avg Weight Capacity (kg)','Avg Price']      

print(material_summary)


Compartment_avg_weight_capacity = train_data.groupby('Compartments')['Weight Capacity (kg)'].mean()      
Compartment_avg_price = train_data.groupby('Compartments')['Price'].mean()      

compartment_summary = pd.concat([Compartment_avg_weight_capacity,Compartment_avg_price],axis=1)      

compartment_summary.columns = ['Avg Weight Capacity (kg)','Avg Price']      

print(compartment_summary)



Laptop_Compartment_avg_weight_capacity = train_data.groupby('Laptop Compartment')['Weight Capacity (kg)'].mean()      
Laptop_Compartment_avg_price = train_data.groupby('Laptop Compartment')['Price'].mean()      

Laptop_compartment_summary = pd.concat([Laptop_Compartment_avg_weight_capacity,Laptop_Compartment_avg_price],axis=1)      

Laptop_compartment_summary.columns = ['Avg Weight Capacity (kg)','Avg Price']      

print(Laptop_compartment_summary)


Style_avg_weight_capacity = train_data.groupby('Style')['Weight Capacity (kg)'].mean()      
Style_avg_price = train_data.groupby('Style')['Price'].mean()      

Style_summary = pd.concat([Style_avg_weight_capacity,Style_avg_price],axis=1)      

Style_summary.columns = ['Avg Weight Capacity (kg)','Avg Price']      

print(Style_summary)


Color_avg_weight_capacity = train_data.groupby('Color')['Weight Capacity (kg)'].mean()      
Color_avg_price = train_data.groupby('Color')['Price'].mean()      

Color_summary = pd.concat([Color_avg_weight_capacity,Color_avg_price],axis=1)      

Color_summary.columns = ['Avg Weight Capacity (kg)','Avg Price']      

print(Color_summary)


Waterproof_avg_weight_capacity = train_data.groupby('Waterproof')['Weight Capacity (kg)'].mean()      
Waterproof_avg_price = train_data.groupby('Waterproof')['Price'].mean()      
 
Waterproof_summary = pd.concat([Waterproof_avg_weight_capacity,Waterproof_avg_price],axis=1)      

Waterproof_summary.columns = ['Avg Weight Capacity (kg)','Avg Price']      

print(Waterproof_summary)


brand_material_summary = train_data.groupby(['Brand', 'Material'])[['Weight Capacity (kg)', 'Price']].mean().reset_index()

# Fix column names in pivot
heatmap_data_weight = brand_material_summary.pivot(index="Brand", columns="Material", values="Weight Capacity (kg)")
heatmap_data_price = brand_material_summary.pivot(index="Brand", columns="Material", values="Price")

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap for Weight Capacity
sns.heatmap(heatmap_data_weight, cmap="Blues", annot=True, fmt=".1f", ax=axes[0], linewidths=2)
axes[0].set_title("Brand-Material Impact on Weight Capacity (kg)")

# Heatmap for Price
sns.heatmap(heatmap_data_price, cmap="Reds", annot=True, fmt=".0f", ax=axes[1], linewidths=2)
axes[1].set_title("Brand-Material Impact on Price")

plt.tight_layout()
plt.show()


brand_Size_summary = train_data.groupby(['Brand', 'Size'])[['Weight Capacity (kg)', 'Price']].mean().reset_index()

# Fix column names in pivot
heatmap_data_weight = brand_Size_summary.pivot(index="Brand", columns="Size", values="Weight Capacity (kg)")
heatmap_data_price = brand_Size_summary.pivot(index="Brand", columns="Size", values="Price")

# Create subplots for both heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap for Weight Capacity (Using Greens)
sns.heatmap(heatmap_data_weight, cmap="Greens", annot=True, fmt=".1f", ax=axes[0], linewidths=2)
axes[0].set_title("Brand-Size Impact on Weight Capacity (kg)")

# Heatmap for Price (Using Purples)
sns.heatmap(heatmap_data_price, cmap="Purples", annot=True, fmt=".0f", ax=axes[1], linewidths=2)
axes[1].set_title("Brand-Size Impact on Price")

plt.tight_layout()
plt.show()


brand_Laptop_compartment_summary = train_data.groupby(['Brand', 'Laptop Compartment'])[['Weight Capacity (kg)', 'Price']].mean().reset_index()

# Fix column names in pivot
heatmap_data_weight = brand_Laptop_compartment_summary.pivot(index="Brand", columns="Laptop Compartment", values="Weight Capacity (kg)")
heatmap_data_price = brand_Laptop_compartment_summary.pivot(index="Brand", columns="Laptop Compartment", values="Price")

# Create subplots for both heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap for Weight Capacity (Using BuPu colormap)
sns.heatmap(heatmap_data_weight, cmap="BuPu", annot=True, fmt=".1f", ax=axes[0], linewidths=2)
axes[0].set_title("Brand-Laptop Compartment Impact on Weight Capacity (kg)")

# Heatmap for Price (Using GnBu colormap)
sns.heatmap(heatmap_data_price, cmap="GnBu", annot=True, fmt=".0f", ax=axes[1], linewidths=2)
axes[1].set_title("Brand-Laptop Compartment Impact on Price")

plt.tight_layout()
plt.show()


brand_Style_summary = train_data.groupby(['Brand', 'Style'])[['Weight Capacity (kg)', 'Price']].mean().reset_index()

# Fix column names in pivot
heatmap_data_weight = brand_Style_summary.pivot(index="Brand", columns="Style", values="Weight Capacity (kg)")
heatmap_data_price = brand_Style_summary.pivot(index="Brand", columns="Style", values="Price")

# Create subplots for both heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap for Weight Capacity (Using PuBuGn colormap)
sns.heatmap(heatmap_data_weight, cmap="PuBuGn", annot=True, fmt=".1f", ax=axes[0], linewidths=2)
axes[0].set_title("Brand-Style Impact on Weight Capacity (kg)")

# Heatmap for Price (Using PuRd colormap)
sns.heatmap(heatmap_data_price, cmap="PuRd", annot=True, fmt=".0f", ax=axes[1], linewidths=2)
axes[1].set_title("Brand-Style Impact on Price")

plt.tight_layout()
plt.show()


#Numerical Imputation
numeric_processor = Pipeline(
    steps=[("imputation_median",SimpleImputer(missing_values=np.nan,strategy="median")),
           ("scaler",StandardScaler())]
)
numeric_processor


#categorical processing
Categorical_processor = ColumnTransformer(
    transformers=[
        ("Onehot", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), ["Brand", "Material", "Style", "Color"]),
        
        ("Ordinal", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder())
        ]), ["Size"]),
        
        ("Binary", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder())  # Used instead of LabelEncoder
        ]), ["Laptop Compartment", "Waterproof"])
    ]
)
Categorical_processor


Preprocessor=ColumnTransformer(
    [("Categorical",Categorical_processor,["Brand","Material","Size","Laptop Compartment","Waterproof","Style","Color"]),
     ("Numerical",numeric_processor,["Compartments","Weight Capacity (kg)"])]
)
Preprocessor


pipe = make_pipeline(Preprocessor,LinearRegression())
pipe


train_data.drop(columns='id',axis=1,inplace=True)
train_data.head()


target = train_data["Price"]
train_data.drop(columns='Price',axis=1,inplace=True)


x_train, x_test, y_train, y_test = train_test_split(
    train_data,target,test_size=0.33, random_state=42
)


pipe.fit(x_train,y_train)


y_pred = pipe.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = math.sqrt(mse)



print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"RÂ² Score: {r2}")



test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test.head()


test_data = test.drop(columns="id",axis=1)


test_pred = pipe.predict(test_data)


submission = pd.DataFrame({
    "id": test["id"],  
    "Price": test_pred
})

# Save predictions to CSV
submission.to_csv("submission.csv", index=False)

