test_url = "/kaggle/input/playground-series-s4e1/test.csv"
train_url ="/kaggle/input/playground-series-s4e1/train.csv"
sample_submision = "/kaggle/input/playground-series-s4e1/sample_submission.csv"


import pandas as pd
import numpy as np
import seaborn as snb
import matplotlib.pyplot as plt
import plotly as px
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline



from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score , classification_report ,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , ReLU , Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping



data = pd.read_csv(train_url)



data.head()


data.info()


data.isnull().sum()


data.duplicated().sum()


desc = pd.DataFrame()
desc["Column"] = data.columns
desc["Count"] = data.count().values
desc["Unique"] = data.nunique().values
desc["Null"] = data.isnull().sum().values
desc["% Unique"] = round(data.nunique() / len(data)*100,2).values
desc["Min"] = data.min().values
desc["Max"] = data.max().values
#desc["Mean"] = data.mean().values
desc.sort_values(by= "Unique",ascending=True)


data.drop(["Surname","CustomerId","id"] , inplace = True , axis = 1)


snb.histplot(data["Age"], kde = True , color = "green")


snb.histplot(data["CreditScore"] , kde = True , color = "green")


data["F_CreditScore"] = pd.cut(data["CreditScore"],bins = [300,500,700,800,900] , labels = ["Low", "Moderate","High","Best"] )


#data["F_Age"] = pd.cut(data["Age"] , bins = [18,30,50,80] , labels = ["Young","Seniors" , "Old"])


#data["Balance"] = np.log1p(data["Balance"])


#data["EstimatedSalary"] = np.log1p(data["EstimatedSalary"])


# Initialize dictionary to store encoders for each column
encoders = {}

# Train Data Encoding
for col in data.select_dtypes(include=["object", "category"]):
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # ✅ Fit only once on train data
    encoders[col] = le


# Select X And Y Feature
x = data.drop("Exited" , axis = 1) # Feature
y = data.Exited # Target



%time
mi_score = mutual_info_classif(x,y)
mi_score = pd.DataFrame({"Features":x.columns , "Mi Score" : mi_score})
mi_score_sort = mi_score.sort_values(by = "Mi Score" , ascending = False)



%time
plt.figure(figsize = (10,12))
snb.barplot( x = "Mi Score", y = "Features" , data = mi_score_sort , palette="viridis")
plt.xlabel("MI Score")
plt.ylabel("Features")
plt.axvline(x=0.01, color='black', linestyle='dotted', linewidth=2)
plt.title("Feature Selection Based On MI Score")
plt.show()



%time
# Random Forest Model Train Karna
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x, y)

# Feature Importance Nikalna
rf_importance = rf.feature_importances_

# Random Forest Scores ko DataFrame me Convert Karna
rf_df = pd.DataFrame({"Feature": x.columns, "RF Score": rf_importance})

# RF Scores ko Descending Order me Sort Karna
rf_df = rf_df.sort_values(by="RF Score", ascending=False)
rf_df


plt.figure(figsize = (10,12))
snb.barplot( x = "RF Score", y = "Feature" , data = rf_df , palette="viridis")
plt.xlabel("Random Forest Feature Selection Score")
plt.ylabel("Random Forest Features")
plt.title("Feature Selection Based On Random Forest")
plt.axvline(x=0.05, color='red', linestyle='dotted', linewidth=2)
plt.show()
data.head()


data.head()



x = data.drop(["Exited","CreditScore"] , axis = 1 )
y = data.Exited
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Train Test Split
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size = 0.2 , random_state = 42)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()


stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)


data.head()



from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

# Build the model
model = Sequential([
    Input(shape=(x_train.shape[1],)),  # Explicit input definition
    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(32, activation='relu'),
    BatchNormalization(),

    Dense(8, activation='relu'),
    BatchNormalization(),

    Dense(1, activation='sigmoid')  # Binary Classification Output
])

# Compile the model
optimizer = Adam(learning_rate=0.005)  # Lower learning rate for better convergence
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_train, y_train),
                    epochs=50, batch_size=500, callbacks=[stop])  # Increased batch size for stability

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


predict = model.predict(x_test)


predict


y_test


from sklearn.metrics import accuracy_score

# Convert probabilities to binary outputs
predict_binary = (predict > 0.5).astype(int).flatten()  # Convert float predictions to 0 or 1

# Now compute accuracy
accuracy = accuracy_score(y_test, predict_binary)
print(f"Accuracy Score: {accuracy:.4f}")  # Print accuracy correctly


test = pd.read_csv(test_url)
test.head()



test["F_CreditScore"] = pd.cut(test["CreditScore"],bins = [300,500,700,800,900] , labels = ["Low", "Moderate","High","Best"] )
test.drop(["id","CustomerId","Surname","CreditScore"] , inplace = True , axis = 1)



# Test Data Encoding
for col in test.select_dtypes(include=["object", "category"]):
    le = encoders[col]  # Get trained encoder

    # Convert test data labels to match the encoder classes
    test[col] = test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)


test.head()


test["EstimatedSalary"] = np.log1p(test["EstimatedSalary"])
test["Balance"] = np.log1p(test["Balance"])
test = scaler.transform(test)


test_predict = model.predict(test)


id = pd.read_csv(test_url)
print(id.shape)
id = id["id"]
binary = (test_predict > 0.5).astype(int).flatten()
# Assuming `predictions` contains your model's output
submission = pd.DataFrame({
    "id": id,
    "Exited": binary
})

# Save as CSV (without index)
submission.to_csv("submission_2.csv", index=False)




