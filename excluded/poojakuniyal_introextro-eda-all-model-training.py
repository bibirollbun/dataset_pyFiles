import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


sns.set_style("darkgrid")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train_df.head()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test_df.head()


train_df.info()


print("shape of train dataset : ",train_df.shape)
print("shape of test dataset : ",test_df.shape)


train_df.isna().sum()


# Calculate percent missing
missing_percent = (train_df.isna().sum() / len(train_df)) * 100
missing_percent_sorted = missing_percent.sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
missing_percent_sorted.plot(kind='bar', color='skyblue')
plt.ylabel('Percentage of Missing Values')
plt.title('Missing Data Percentage by Column')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


test_df.isna().sum()


import missingno as msno
msno.bar(test_df)
plt.show()


test_id = test_df["id"]


train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)
train_df.columns


cat_features = train_df.select_dtypes(include="object").columns
cat_features = [i for i in cat_features if i not in "Personality"]
num_features = train_df.select_dtypes(exclude="object").columns 


def num_plots_dist(df, num_features):
    fig, axes = plt.subplots(len(num_features), 2, figsize=(15, len(num_features) * 5))

    if len(num_features) == 1:
        axes = np.array([axes])

    for i, col in enumerate(num_features):
        # KDE plot
        sns.kdeplot(data=df, x=col, ax=axes[i][0], fill=True, color='skyblue')
        axes[i][0].set_title(f"KDE Plot for {col}", fontsize=16, fontweight='bold')
        axes[i][0].set_xlabel(col, fontsize=14)
        axes[i][0].set_ylabel('Density', fontsize=14)
        axes[i][0].tick_params(axis='x', labelsize=12)
        axes[i][0].tick_params(axis='y', labelsize=12)

        # Violin plot
        sns.violinplot(data=df, x=col, ax=axes[i][1], color='darkblue')
        axes[i][1].set_title(f"Violin Plot for {col}", fontsize=16, fontweight='bold')
        axes[i][1].set_xlabel(col, fontsize=14)
        axes[i][1].set_ylabel('Value Distribution', fontsize=14)
        axes[i][1].tick_params(axis='x', labelsize=12)
        axes[i][1].tick_params(axis='y', labelsize=12)

    plt.tight_layout(pad=3)
    plt.show()


num_plots_dist(train_df, num_features)


for feature in cat_features:
    ct = pd.crosstab(train_df[feature], train_df["Personality"])
    plt.figure(figsize=(8,6))
    sns.heatmap(ct, annot=True, fmt="d",cmap="Set2")
    plt.title(f'{feature} vs Personality (Extr or Intro)', fontsize=16, fontweight='bold')
    plt.ylabel(feature)
    plt.xlabel("Personality")
    plt.tight_layout()
    plt.show()



def plot_bivariate_num(df, target, num_features):
    num_plots = len(num_features)
    num_rows = (num_plots + 1) // 2  # 2 plots per row

    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_features):
        sns.barplot(x=target, y=col, data=df, ax=axes[i], palette="flare")
        axes[i].set_title(f"{col} vs {target}", fontsize=16, fontweight='bold')
        axes[i].set_xlabel(target, fontsize=14)
        axes[i].set_ylabel(col, fontsize=14)
        axes[i].tick_params(axis='x', labelsize=12)
        axes[i].tick_params(axis='y', labelsize=12)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3)
    plt.show()


plot_bivariate_num(train_df, 'Personality', num_features)


sns.set_style('ticks')
sns.barplot(x='Time_spent_Alone', y='Friends_circle_size', data=train_df, color='blue');


sns.set_style('whitegrid')

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Social_event_attendance', 
    y='Post_frequency',
    hue='Personality',
    data=train_df,
    edgecolor='black'  # Add outlines for distinction
)

plt.title("Participation in Social Events vs Post Frequency by Personality Type", fontsize=16, fontweight='bold')
plt.xlabel("Social Event Attendance", fontsize=14)
plt.ylabel("Post Frequency", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Personality Type', fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.show()


# Calculate percentages
personality_counts = train_df["Personality"].value_counts(normalize=True) * 100

# Plot
plt.figure(figsize=(5,5))
plt.pie(
    personality_counts,
    labels=personality_counts.index,
    autopct='%.1f%%',
    startangle=140,
    colors=plt.cm.Set2.colors  # Soft, distinct colors
)
plt.title("Distribution of Personality Types", fontsize=16, fontweight='bold')
plt.show()


X= train_df.drop(columns=["Personality"])
y = train_df["Personality"]


X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,stratify=y, random_state=90)


X_train.shape, X_test.shape,y_train.shape,y_test.shape


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=90)

X_train[num_features] = imputer.fit_transform(X_train[num_features])
X_test[num_features] = imputer.transform(X_test[num_features])
test_df[num_features] = imputer.transform(test_df[num_features])


from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
X_test[cat_features] = cat_imputer.transform(X_test[cat_features])
test_df[cat_features] = cat_imputer.transform(test_df[cat_features])


X_train['Stage_fear'].unique(),X_train['Drained_after_socializing'].unique() 


test_df['Stage_fear'].unique(), test_df['Drained_after_socializing'].unique()


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
cat_mappings ={}

for col in cat_features:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    test_df[col] = le.transform(test_df[col])

    label_encoders[col] = le
    cat_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))


cat_mappings


target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)


corr = X_train.corr()
plt.Figure(figsize=(12,10))
sns.heatmap(corr)
plt.title("Correlation Matrix - Heatmap");


# !pip install statsmodels


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


x = add_constant(X_train)
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1]) ]



vif_data


from sklearn.svm import SVC


classifiers ={
    "Random forest": RandomForestClassifier(),
    "Lasso Logistic Regression": LogisticRegression(penalty='l1', solver='liblinear', C=1.0),
    "Ridge Logistic Regression": LogisticRegression(penalty='l2', solver='lbfgs', C=1.0),
    "Elasticnet Logisitc Regression":LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0),
    "Gradient Boosting" : GradientBoostingClassifier(random_state=42),
    "Support Vector Classifier" : SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naives bayes": GaussianNB(),
    "XGBoost" : XGBClassifier(random_state=42),
    "AdaBoostClassifier" : AdaBoostClassifier(random_state=42),
    "lightGBM" : LGBMClassifier(random_state=42)
}


metrics={
        "Model" :[],
        "Accuracy": []}


for model_name, classifier in classifiers.items():
    classifier.fit(X_train,y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics["Model"].append(model_name)
    metrics["Accuracy"].append(accuracy)


metrics_df= pd.DataFrame(metrics)
metrics_df


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint,  uniform


lgbm = LGBMClassifier()


param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 15),
    'num_leaves': randint(20, 150),
    'learning_rate': uniform(0.01, 0.2),
    'min_child_samples': randint(10, 100),
    'subsample': uniform(0.5, 0.5),         # Boosting bagging fraction
    'colsample_bytree': uniform(0.5, 0.5),  # Feature fraction
    'reg_alpha': uniform(0, 1),             # L1 regularization
    'reg_lambda': uniform(0, 1)             # L2 regularization
}



random_search = RandomizedSearchCV(estimator=lgbm,param_distributions=param_dist,
                                    n_iter=10, cv=3, verbose=2,
                                     random_state=90, scoring='accuracy') 


random_search.fit(X_train, y_train)


random_search.best_params_


best_lgbm_model = random_search.best_estimator_


y_pred = best_lgbm_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)


accuracy


final_test_pred = best_lgbm_model.predict(test_df)


final_test_pred


final_test_pred_inverse = target_encoder.inverse_transform(final_test_pred)



submission = pd.DataFrame({
    'id': test_id,
    'Personality': final_test_pred_inverse
})
submission.head(10)


submission.to_csv('submission.csv', index=False)




