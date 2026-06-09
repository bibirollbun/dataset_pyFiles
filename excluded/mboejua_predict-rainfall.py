#Modules needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import random
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import  StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor




#Import dataset 
train_x= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
test_x= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
test_y= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
y_train = train_x['rainfall']
train_x = train_x.drop(["rainfall"],axis="columns")
train_x.head()



train_x.describe()


#Scale train and test datasets
scaler = StandardScaler()
train_x_scaled = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns)
test_x_scaled = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns)

test_x_scaled.describe()


#Exploratory analysis on trainset
palettes = ["deep",  "pastel"]
palettes1 = ["red","blue"]
fig, axes = plt.subplots(len(train_x.columns),3,figsize=(40,10*len(train_x.columns)), constrained_layout = True)
for axs, xy in zip(axes,train_x.columns):
    sns.histplot(train_x[xy], ax=axs[0], color=palettes1[0], kde=True)
    axs[0].set_title(f"{xy} Histogram", fontsize=26)
    axs[0].set_xlabel(xy,fontsize=25)
    axs[0].set_ylabel("count" ,fontsize=25)
    axs[0].tick_params(axis='both', labelsize=20)
    
    #axs.set_ylabel("Rainfall")
    sns.boxplot(data= train_x, x=xy, ax=axs[1], palette=palettes[1], orient="h" )
    axs[1].set_title(f"{xy} Boxplot", fontsize=26)
    axs[1].set_xlabel(xy,fontsize=25)
    axs[1].tick_params(axis='both', labelsize=20)
    axs[1].annotate(f"\n Mean: {round(np.mean(train_x[xy]),2)} \n Median: {round(np.median(train_x[xy]),2)} \n Std: {round(np.std(train_x[xy]),2)}",
                    xy=(np.mean(train_x[xy]),(train_x[xy] == np.mean(train_x[xy])).sum()), fontsize=18
    )

    # Define percentiles
    #lower_bound, upper_bound = np.percentile(train_x[xy], [2.5, 97.5])
    #filtered_data = train_x[xy][(train_x[xy] >= lower_bound) & (train_x[xy] <= upper_bound)]

    # Plot histogram with filtered data
    sns.histplot(train_x_scaled[xy], ax=axs[2], color=palettes1[1], kde=True)
    axs[2].set_title(f"{xy} Histogram (StandardScaler)", fontsize=26)
    axs[2].set_xlabel(xy, fontsize=25)
    axs[2].set_ylabel("Count", fontsize=25)
    axs[2].tick_params(axis='both', labelsize=20)

    





sns.set_palette(random.choice(palettes))
sns.heatmap(train_x.corr(), cmap="coolwarm")



# Select only the temperature-related features as had high correlation values among each other
X = train_x[["mintemp", "temparature", "maxtemp"]]

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# Defining our datasets
X_train_scaled = train_x_scaled




# Test set count for winddirection and windspeed vary from other. Treating missing values
test_x.fillna(np.mean(test_x), inplace=True)
test_x.describe()


import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold , train_test_split
import matplotlib.pyplot as plt

# Define models
tree_models = ["gbtree", GradientBoostingClassifier, RandomForestClassifier]
fig, axes = plt.subplots(len(tree_models), 2, figsize=(16, 4 * len(tree_models)), constrained_layout=True)
xgboost_settings = np.arange(100, 702, 100)
forest_settings = np.arange(100, 1002, 50)
settings = [xgboost_settings, forest_settings, forest_settings]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
prob = {}
best_models = {}

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# **Step 1: Define 5 XGBoost parameter combinations using zip**
param_grid_xgb = list(zip(
    # n_estimators (more trees require lower learning rate)
    [50,75,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000],
    
    # learning_rate (decreases as n_estimators increases)
    [0.085,0.9, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.005],

    # max_depth (deeper trees capture more complexity but risk overfitting)
    [3,3, 4, 5, 6, 7, 8, 9, 10, 11],

    # subsample (reduces overfitting, should be between 0.6-1.0)
    [0.75,0.8, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],

    # colsample_bytree (number of features considered per tree)
    [0.675, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
))

for axs, rant, settings in zip(axes, tree_models, settings):

    if rant == "gbtree":
        test_accuracies = []
        xgb_models = []

        # **Step 2: Train 5 fixed XGBoost models with different parameters**
        for n_est, lr, max_d, sub_s, col_s in param_grid_xgb:
            clf = xgb.XGBClassifier(
                booster=rant,
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=max_d,
                subsample=sub_s,
                colsample_bytree=col_s,
                objective="binary:logistic",
                random_state=42,
                use_label_encoder=False
            )

            # Train & predict
            clf.fit(X_train_sub, y_train_sub) #y_train_sub, y_val
            y_pred = clf.predict(X_val)
            y_scores = clf.predict_proba(X_val)[:, 1]
            #y_pred = (y_scores >= 0.5).astype(int) 
            test_acc = accuracy_score(y_val, y_pred)

            test_accuracies.append((clf, test_acc, n_est, lr, max_d, sub_s, col_s))
            xgb_models.append(clf)

        # **Step 3: Select the best model based on test accuracy**
        best_model, best_test_acc, best_n_est, best_lr, best_max_d, best_sub_s, best_col_s = max(test_accuracies, key=lambda x: x[1])
        print(f"Best Test Accuracy: {best_test_acc:.4f}")
        print(f"Best XGB Parameters: n_estimators={best_n_est}, learning_rate={best_lr}, max_depth={best_max_d}, subsample={best_sub_s}, colsample_bytree={best_col_s}")

        best_models["gbtree"] = best_model
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict_proba(test_x_scaled)[:,1]
        

        # **Step 4: Compute ROC Curve for the best test-performing model**
        best_y_scores = best_model.predict_proba(train_x)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, best_y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        axs[1].plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
        axs[1].plot([0, 1], [0, 1], color="red", linestyle="--")
        axs[1].set_xlabel("False Positive Rate")
        axs[1].set_ylabel("True Positive Rate")
        axs[1].set_title(f"ROC Curve for Best XGBoost Model")
        axs[1].legend()




best_y_scores.shape


best_models1 = max(best_models, key=best_models.get)
print(str(best_models1).split("'")[0])
prob_r = pd.DataFrame({"id":test_x.index,
                       "Rainfall Prob":y_pred}
                      )

prob_r.to_csv('submission.csv', index=False)
prob_r.head()

