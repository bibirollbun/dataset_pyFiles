import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")


df.head()


df.shape


df.info()


df.isna().sum()


df.duplicated().sum()


df.describe()


categorical_features = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance', 'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification"]
for col in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=col, hue='Target', palette='viridis')
    plt.title(f'Distribution of {col} by Target')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()





plt.figure(figsize=(6, 4))
sns.countplot(x=df["Target"], palette="pastel")
plt.title("Distribution of Target Variable")
plt.xlabel("Target")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Target"], y=df["Admission grade"], palette="pastel")
plt.title("Distribution of Admission Grades by Target Category")
plt.xlabel("Target")
plt.ylabel("Admission Grade")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Target"], y=df["Age at enrollment"], palette="pastel")
plt.title("Distribution of Age at Enrollment by Target Category")
plt.xlabel("Target")
plt.ylabel("Age at Enrollment")
plt.xticks(rotation=45)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(x=df["Target"], y=df["Unemployment rate"], palette="pastel", ax=axes[0])
axes[0].set_title("Unemployment Rate by Target")
axes[0].set_xlabel("Target")
axes[0].set_ylabel("Unemployment Rate")

sns.boxplot(x=df["Target"], y=df["Inflation rate"], palette="pastel", ax=axes[1])
axes[1].set_title("Inflation Rate by Target")
axes[1].set_xlabel("Target")
axes[1].set_ylabel("Inflation Rate")

sns.boxplot(x=df["Target"], y=df["GDP"], palette="pastel", ax=axes[2])
axes[2].set_title("GDP by Target")
axes[2].set_xlabel("Target")
axes[2].set_ylabel("GDP")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x=df["Target"], hue=df["Gender"], palette="pastel")
plt.title("Gender Distribution by Target Category")
plt.xlabel("Target")
plt.ylabel("Count")
plt.legend(title="Gender", labels=["Female", "Male"])
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Gender"], y=df["Admission grade"], palette="pastel")
plt.title("Admission Grade Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Admission Grade")
plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(x=df["Gender"], y=df["Curricular units 1st sem (grade)"], palette="pastel", ax=axes[0])
axes[0].set_title("1st Semester Grades by Gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Grade")
axes[0].set_xticklabels(["Female", "Male"])

sns.boxplot(x=df["Gender"], y=df["Curricular units 2nd sem (grade)"], palette="pastel", ax=axes[1])
axes[1].set_title("2nd Semester Grades by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Grade")
axes[1].set_xticklabels(["Female", "Male"])

plt.tight_layout()
plt.show()


dropout_df = df[df["Target"] == "Dropout"]

dropout_counts = dropout_df["Gender"].value_counts(normalize=True) * 100

plt.figure(figsize=(6, 4))
sns.barplot(x=dropout_counts.index, y=dropout_counts.values, palette="pastel")
plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
plt.title("Dropout Percentage by Gender")
plt.xlabel("Gender")
plt.ylabel("Percentage of Dropout Students")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=dropout_df["Gender"], y=dropout_df["Admission grade"], palette="pastel")
plt.title("Admission Grade Distribution for Dropout Students by Gender")
plt.xlabel("Gender")
plt.ylabel("Admission Grade")
plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

male_dropout = dropout_df[dropout_df["Gender"] == 1] 
female_dropout = dropout_df[dropout_df["Gender"] == 0]  

sns.boxplot(x=dropout_df["Gender"], y=dropout_df["Unemployment rate"], palette="pastel", ax=axes[0])
axes[0].set_title("Unemployment Rate for Dropout Students by Gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Unemployment Rate")
axes[0].set_xticklabels(["Female", "Male"])

sns.boxplot(x=dropout_df["Gender"], y=dropout_df["Inflation rate"], palette="pastel", ax=axes[1])
axes[1].set_title("Inflation Rate for Dropout Students by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Inflation Rate")
axes[1].set_xticklabels(["Female", "Male"])

sns.boxplot(x=dropout_df["Gender"], y=dropout_df["GDP"], palette="pastel", ax=axes[2])
axes[2].set_title("GDP for Dropout Students by Gender")
axes[2].set_xlabel("Gender")
axes[2].set_ylabel("GDP")
axes[2].set_xticklabels(["Female", "Male"])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(x=df["Target"], y=df["Curricular units 1st sem (approved)"], hue=df["Gender"], palette="pastel", ax=axes[0])
axes[0].set_title("Approved Curricular Units (1st Sem) by Target & Gender")
axes[0].set_xlabel("Target")
axes[0].set_ylabel("Approved Units (1st Sem)")
axes[0].set_xticklabels(df["Target"].unique(), rotation=45)

sns.boxplot(x=df["Target"], y=df["Curricular units 2nd sem (approved)"], hue=df["Gender"], palette="pastel", ax=axes[1])
axes[1].set_title("Approved Curricular Units (2nd Sem) by Target & Gender")
axes[1].set_xlabel("Target")
axes[1].set_ylabel("Approved Units (2nd Sem)")
axes[1].set_xticklabels(df["Target"].unique(), rotation=45)

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(x=df["Target"], y=df["Mother's qualification"], hue=df["Gender"], palette="pastel", ax=axes[0])
axes[0].set_title("Mother's Qualification by Target & Gender")
axes[0].set_xlabel("Target")
axes[0].set_ylabel("Mother's Qualification Level")
axes[0].set_xticklabels(df["Target"].unique(), rotation=45)

sns.boxplot(x=df["Target"], y=df["Father's qualification"], hue=df["Gender"], palette="pastel", ax=axes[1])
axes[1].set_title("Father's Qualification by Target & Gender")
axes[1].set_xlabel("Target")
axes[1].set_ylabel("Father's Qualification Level")
axes[1].set_xticklabels(df["Target"].unique(), rotation=45)

plt.tight_layout()
plt.show()



from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


label_encoder = LabelEncoder()
df["Target"] = label_encoder.fit_transform(df["Target"])


X = df.drop(columns=["id", "Target"]) 
y = df["Target"]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)



scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)



X_val=scaler.transform(x_val)



model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Multi-class classification
])


model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32)



loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


plt.figure(figsize= (30, 8))
plt.style.use('fivethirtyeight')

plt.plot(history.history['loss'], 'r', label= 'Training loss')
plt.plot(history.history['val_loss'], 'g', label= 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout
plt.show()




