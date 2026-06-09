import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

sample_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_df.head().style.background_gradient(cmap='PuBu')


test_df.head().style.background_gradient(cmap='PuBu')


sample_df.head().style.background_gradient(cmap='PuBu')


from colorama import Fore, Style

# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "train_df shape: " + Style.RESET_ALL)
print(f"{train_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "train_df info: " + Style.RESET_ALL)
print(f"{train_df.info()}\n") 

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "train_df isnull sum: " + Style.RESET_ALL)
print(f"{train_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "train_df describe: " + Style.RESET_ALL)
print(f"{train_df.describe()}\n")



# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "test_df shape: " + Style.RESET_ALL)
print(f"{test_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "test_df info: " + Style.RESET_ALL)
print(f"{test_df.info()}\n") 

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "test_df isnull sum: " + Style.RESET_ALL)
print(f"{test_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "test_df describe: " + Style.RESET_ALL)
print(f"{test_df.describe()}\n")


test_df.describe().style.background_gradient(cmap='PuBu')


# Fill the missing value
test_df["winddirection"].fillna(test_df["winddirection"].median(), inplace=True)


numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_columns.remove('id')

train_df[numerical_columns].hist(figsize=(14, 10), bins=30, edgecolor="black", layout=(4, 3))
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()


# Select numerical columns, excluding 'id'
numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'id' in numerical_columns:
    numerical_columns.remove('id')

# Create subplots for better visualization
num_features = len(numerical_columns)
rows = (num_features // 3) + (num_features % 3 > 0)  # Adjust rows dynamically

fig, axes = plt.subplots(rows, 3, figsize=(14, 10))
axes = axes.flatten()  # Flatten to loop easily

for i, col in enumerate(numerical_columns):
    train_df.boxplot(column=[col], ax=axes[i], grid=False)
    axes[i].set_title(col)

# Remove empty subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()


train_df["rainfall"].value_counts().plot(kind="pie", autopct='%1.1f%%', shadow=True)
plt.title("Counts of Rainfall")
plt.ylabel('')  # Hide y-axis label
plt.show()


features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# Corrected pairplot
sns.pairplot(train_df, vars=features, hue="rainfall", diag_kind="hist")

plt.suptitle("Pairplot of Selected Features (Using Hist for Stability)", fontsize=16)
plt.show()



plt.figure(figsize= (12,8))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


train_df["Season"] = train_df["day"] % 365
def season(day):
    month = (day % 365 )// 30 + 1
    if month in [12, 1 ,2 ]:
        return 0 # winter
    elif month in [3, 4, 5]:
        return 1 # spring 
    elif month in [6, 7, 8]:
        return 2 # summer
    else:
        return 3 # autumn
train_df["Season"]= train_df["day"].apply(season)
test_df["Season"]= test_df["day"].apply(season)


# for the train data
train_df ["temp_range"] = train_df["maxtemp"] - train_df["mintemp"]

train_df["temp_dew_diff"] = train_df["temparature"] - train_df["dewpoint"]

train_df['humid_temp'] = train_df["humidity"] * train_df["temparature"]

train_df['cloud_sun_ratio'] = train_df["cloud"] /( train_df["sunshine"]+ 1)

# for the test data
test_df ["temp_range"] = test_df["maxtemp"] - test_df["mintemp"]

test_df["temp_dew_diff"] = test_df["temparature"] - test_df["dewpoint"]

test_df['humid_temp'] = test_df["humidity"] * test_df["temparature"]

test_df['cloud_sun_ratio'] = test_df["cloud"] /( test_df["sunshine"]+ 1)


plt.figure(figsize= (12,8))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


x = train_df.drop(columns = ["rainfall"])
y = train_df["rainfall"]


stander = StandardScaler()
x = pd.DataFrame(stander.fit_transform (x), columns = x.columns)


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# Running various models
models = []
models.append(("DecisionTreeClassifier",DecisionTreeClassifier()))
models.append(("RandomForestClassifier",RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_samples=1.0, max_features=1.0, max_depth=8, bootstrap=True)))
models.append(("KNeighborsClassifier",KNeighborsClassifier()))
models.append(("XGBClassifier",XGBClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = "accuracy"

for name, model in models:
    model.fit(x_train,y_train)


    y_pred = model.predict(x_test)
    prediction = [round(value) for value in y_pred]

    
    accuracy = roc_auc_score(y_test,prediction)
    print("Accuracy: %.2f%%" % (accuracy * 100.0),name)
    print("___________________________________________")


# Define classifiers
d_tree = DecisionTreeClassifier(splitter='best', max_depth=9, min_samples_split=10, max_features=6, random_state=5)

r_foresr = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, 
                              max_samples=1.0, max_features='sqrt', max_depth=8, bootstrap=True)

k_neig = KNeighborsClassifier(n_neighbors=5, algorithm='auto', leaf_size=30)

estimators = [('DTREE', d_tree), ('RFOREST', r_foresr), ('k_neig', k_neig)]

voting = VotingClassifier(estimators=estimators, voting='soft')

# Fit the model
voting.fit(x_train, y_train)

# Predict probabilities for the positive class (1)
y_pred_prob = voting.predict_proba(x_test)[:, 1]

# Compute ROC AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC Score (Soft Voting): {auc_score:.4f}")

y_pred = voting.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc_score:.4f}")


bag_dtree = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)

bag_dtree.fit(x_train,y_train)

y_pred = bag_dtree.predict(x_test)

auc_score = roc_auc_score(y_test,y_pred)
print(f"ROC AUC Score : {auc_score:.4f}")



bag_random = BaggingClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_samples=1.0, max_features=1.0, max_depth=8, bootstrap=True),
    n_estimators=50,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)

bag_random.fit(x_train,y_train)

y_pred = bag_random.predict(x_test)

auc_score = roc_auc_score(y_test,y_pred)
print(f"ROC AUC Score : {auc_score:.4f}")

