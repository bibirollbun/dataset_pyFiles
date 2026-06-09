#import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('covtype.csv')


print("Shape:", df.shape)


df.head()


df.info()


df.describe()


df.isnull().sum()


plt.figure(figsize=(8,5))
sns.countplot(x="Cover_Type", data=df, color="green")
plt.title("Distribution of Forest Cover Types", fontsize=14)
plt.xlabel("Cover Type")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(x="Cover_Type", y="Elevation", data=df, palette="Greens", hue= 'Cover_Type', legend= False)

plt.title("Elevation Distribution by Forest Cover Type")
plt.xlabel("Cover Type")
plt.ylabel("Elevation (meters)")
plt.show()


numeric_cols = ['Elevation', 'Aspect', 'Slope',
                'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points']

plt.figure(figsize=(15, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[col], bins=30, kde=False, color="green")
    plt.title(col, fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="Greens", center=0)
plt.title("Correlation Heatmap (Numerical Features)", fontsize=14)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X = df.drop("Cover_Type", axis=1) #feature
y = df["Cover_Type"] #target


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Scale numeric columns
num_cols = ["Elevation", "Aspect", "Slope", 
            "Horizontal_Distance_To_Hydrology", 
            "Vertical_Distance_To_Hydrology", 
            "Horizontal_Distance_To_Roadways", 
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", 
            "Horizontal_Distance_To_Fire_Points"]

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.fit_transform(X_test[num_cols])


# make a sample 10%
sample_frac = 0.1

X_train_sample = X_train.sample(frac=sample_frac, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

X_test_sample = X_test.sample(frac=sample_frac, random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

print("Train sample shape:", X_train_sample.shape)
print("Test sample shape:", X_test_sample.shape)


from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1
)

y_train_sample = y_train_sample - 1
y_test_sample = y_test_sample - 1

xgb_model.fit(X_train_sample, y_train_sample)


import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=7,
    n_jobs=-1
)

lgb_model.fit(X_train_sample, y_train_sample)


from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    verbose=100
)

cat_model.fit(X_train_sample, y_train_sample)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {'XGBoost': xgb_model, 'LightGBM': lgb_model, 'CatBoost': cat_model}

for name, model in models.items():
    y_pred = model.predict(X_test_sample)
    print(f'---{name}---')
    print('Accuracy:', accuracy_score(y_test_sample, y_pred))
    print(classification_report(y_test_sample, y_pred))
    print(confusion_matrix(y_test_sample, y_pred))


y_train_adj = y_train - 1
y_test_adj = y_test - 1

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1
)


xgb_model.fit(X_train, y_train_adj)


y_pred = xgb_model.predict(X_test)


print("Accuracy:", accuracy_score(y_test_adj, y_pred))
print("\nClassification Report:\n", classification_report(y_test_adj, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_adj, y_pred))

