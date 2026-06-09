import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Train dataset
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# Test dataset
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train.head()


print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")


df_train.info()


df_train.isnull().sum()


df_train.duplicated().sum()


numerical = ['Temparature',	'Humidity','Moisture','Nitrogen','Potassium','Phosphorous']

for col in numerical:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    

    sns.histplot(df_train[col], kde = True, bins = 20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    sns.boxplot(df_train[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

    print(f"Statistics for {col}")
    print(f"Skewness: {df_train[col].skew() :.2f}")
    


for col in ["Soil Type", "Crop Type"]:
    counts = df_train[col].value_counts()
    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1)
    sns.countplot(data = df_train, x = col, palette = "Set2")
    plt.title(f"Count of {col} values")
    plt.xticks(rotation = 90)
    plt.ylabel("Count")
    # plt.show()

    plt.subplot(1,2,2)
    plt.pie(counts,labels = counts.index,autopct = "%1.1f%%",startangle=90)
    plt.title(f"Percentage of {col}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Print unique and missing values
    print(f"Number of Unique {col}: {df_train[col].nunique()}")
    print(f"Missing Values in {col}: {df_train[col].isnull().sum()}")
    


plt.figure(figsize=(8, 6))
sns.scatterplot(x = df_train["Crop Type"],y = df_train["Fertilizer Name"] ,hue = df_train["Soil Type"])
plt.title("Crop and Fretilizer Name by Soil Type")
plt.xticks(rotation = 90)
plt.legend(loc = "upper right")
plt.show()


for col in numerical[:-1]:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x = df_train[col], y = df_train["Fertilizer Name"],color = "green")
    plt.title(f"{col} VS Fertillzer Name")
    plt.xlabel(col)
    plt.ylabel("Fertillizer Name")
    plt.show()

correlatton_mat = df_train[numerical].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlatton_mat, annot = True ,fmt=".2f")
plt.title('Correlation Matrix of Numerical Feature')
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(data = df_train , y = "Soil Type", order = df_train["Soil Type"].value_counts().index, hue = "Fertilizer Name",palette = "inferno" )
plt.title("Soil Type by Fertilizer  ")
plt.xlabel("Count")
plt.ylabel("Soil Type")
plt.legend(title = "Fertilizer Name", loc = "upper left", bbox_to_anchor = (1,1))
plt.show()

print("\n")
print("--------------------------------------------------------------------------------")

plt.figure(figsize = (12,6))
sns.countplot(data = df_train , y = df_train["Crop Type"], order = df_train["Crop Type"].value_counts().index, hue = "Fertilizer Name", palette = "inferno")
plt.title("Crop Type by Fertillizer ")
plt.xlabel("Count")
plt.ylabel("Crop Type")
plt.legend(title = "Fertilizer Name", loc = "upper left", bbox_to_anchor = (1,1))
plt.show()


for col in numerical[:-1]:
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=df_train, x = "Fertilizer Name", y = col, palette = "Set3")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


from xgboost import XGBClassifier
# Feature & Target separation
x = df_train.drop(columns=["Fertilizer Name"])
y = df_train["Fertilizer Name"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# Define numeric and categorical columns
numeric_features = ['id', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil Type', 'Crop Type']

# Preprocessor (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='multi:softprob',
        n_estimators=3200,
        learning_rate=0.045,
        max_depth=7,
        colsample_bytree=0.6,
        colsample_bylevel=0.8,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss'  # Important for newer versions of XGBoost
    ))
])

# Train the pipeline
pipeline.fit(x_train, y_train)


y_pred_probs = pipeline.predict_proba(x_test)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y_test]

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
map3_score = mapk(actual, top_3_preds)

print(f"âœ… MAP@3 Score: {map3_score:.5f}")


