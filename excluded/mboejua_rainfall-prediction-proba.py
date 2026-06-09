#Modules needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import random
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split , StratifiedKFold, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor




#Import dataset 
train_x= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
test_x= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
test_y= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

train_x



train_x.describe()


#Exploratory analysis on trainset
palettes = ["deep",  "pastel"]
palettes1 = ["red","blue"]
fig, axes = plt.subplots(len(train_x.columns),3,figsize=(35,10*len(train_x.columns)), constrained_layout = True)
for axs, xy in zip(axes,train_x.columns):
    sns.histplot(train_x[xy], ax=axs[0], color=palettes1[0], kde=True)
    axs[0].set_title(f"{xy} Histogram", fontsize=26)
    axs[0].set_xlabel(xy,fontsize=25)
    axs[0].set_ylabel("count" ,fontsize=25)
    axs[0].tick_params(axis='both', labelsize=20)
    
    #axs.set_ylabel("Rainfall")
    
    sns.boxplot(data= train_x, x=xy, ax=axs[1], palette=palettes[1])
    axs[1].set_title(f"{xy} Boxplot", fontsize=26)
    axs[1].set_xlabel(xy,fontsize=25)
    axs[1].tick_params(axis='both', labelsize=20)
    axs[1].annotate(f"\n Mean: {round(np.mean(train_x[xy]),2)} \n Median: {round(np.median(train_x[xy]),2)} \n Std: {round(np.std(train_x[xy]),2)}",
                    xy=(np.mean(train_x[xy]),(train_x[xy] == np.mean(train_x[xy])).sum()), fontsize=18
    )

    # Define percentiles
    lower_bound, upper_bound = np.percentile(train_x[xy], [2.5, 97.5])
    filtered_data = train_x[xy][(train_x[xy] >= lower_bound) & (train_x[xy] <= upper_bound)]

    # Plot histogram with filtered data
    sns.histplot(filtered_data, ax=axs[2], color=palettes1[1], kde=True)
    axs[2].set_title(f"{xy} Histogram (2.5 - 97.5 percentile)", fontsize=26)
    axs[2].set_xlabel(xy, fontsize=25)
    axs[2].set_ylabel("Count", fontsize=25)
    axs[2].tick_params(axis='both', labelsize=20)





sns.set_palette(random.choice(palettes))
sns.heatmap(train_x.corr())



# Select only the temperature-related features as had high correlation values among each other
X = train_x[["mintemp", "temparature", "maxtemp"]]

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# Defining our datasets
X_train = train_x.drop(["rainfall"],axis="columns")
y_train = train_x['rainfall']



# Test set count for winddirection and windspeed vary from other. Treating missing values
test_x.fillna(np.mean(test_x), inplace=True)
test_x.describe()


tree_models = [ "gbtree", GradientBoostingClassifier, RandomForestClassifier ]#, "dart","gblinear"]
fig, axes = plt.subplots(len(tree_models),2,figsize=(16,4*len(tree_models)), constrained_layout=True)
xgboost_settings = np.arange(0, 202,25)
forest_settings =  np.arange(1, 202,25)
settings =[xgboost_settings,forest_settings,forest_settings]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)  
prob = {}
best_models ={}
for axs, rant,settings  in zip(axes, tree_models,settings):
    training_accuracy = []
    cross_val_acc = []
    settings= settings
    for forest in settings:
        
        if rant == GradientBoostingClassifier:
            clf = rant(n_estimators=forest, max_depth = 3, random_state=0, learning_rate = 0.1)
            clf.fit(X_train, y_train)
        # record cross val and whole training set accuracy
            training_accuracy.append(clf.score(X_train, y_train))
            y_scores = clf.predict_proba(X_train)[:, 1]  # Get probability for rainfall (1)
            cv_scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")
            cross_val_acc.append(cv_scores.mean())  
            
        elif rant == RandomForestClassifier:
            clf = rant(n_estimators=forest, max_depth = 3, random_state=0)
            clf.fit(X_train, y_train)
        
            training_accuracy.append(clf.score(X_train, y_train))
            y_scores = clf.predict_proba(X_train)[:, 1]
            cv_scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")
            cross_val_acc.append(cv_scores.mean())
            
            
        else:
            clf = xgb.XGBClassifier(
            booster=rant,   
            n_estimators=forest,   
            max_depth=4,        
            learning_rate=0.1,  
            reg_lambda=1,       
            reg_alpha=0,        
            objective="binary:logistic"
)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_scores = clf.predict_proba(X_train)[:, 1]  
            training_accuracy.append(accuracy_score(y_train, y_pred))
            cv_scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")
            cross_val_acc.append(cv_scores.mean())
    
    axs[0].plot(settings, training_accuracy, label="training accuracy", color =palettes1[1])
    axs[0].plot(settings, cross_val_acc, label="cross_val accuracy", color =palettes1[0])
    type = str(rant).split(".")[-1]
    axs[0].set_title(type)
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Alpha")
    axs[0].legend()
# Anno]tate best test accuracy point
    best_index = np.argmax(cross_val_acc)
    best_alpha = settings[best_index]
    best_cross_val_acc = cross_val_acc[best_index]

    axs[0].annotate(f"Best Cross_val Accur:{round((best_cross_val_acc),3)},aplha:{round(best_alpha,5)} ", 
             xy=(best_alpha, best_cross_val_acc), 
             xytext=(best_alpha+0.01 , best_cross_val_acc  ),  fontsize= 9.5,
             arrowprops=dict(facecolor='black', shrink=0.05))
    # Get predicted probabilities
    if rant == GradientBoostingClassifier:
        clf = rant(n_estimators=best_alpha, max_depth = 3, random_state=0, learning_rate = 0.1)
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_train)[:, 1]
        y_scores_test = clf.predict_proba(test_x)[:, 1] 
        prob[str(rant).split(".")[-1]] = y_scores_test
        best_models[str(rant).split(".")[-1]] = best_alpha
        
            
    elif rant == RandomForestClassifier:
        clf = rant(n_estimators=best_alpha, max_depth = 3, random_state=0)
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_train)[:, 1]
        y_scores_test = clf.predict_proba(test_x)[:, 1] 
        prob[str(rant).split(".")[-1]] = y_scores_test
        best_models[str(rant).split(".")[-1]] = best_alpha
    else:
        clf = xgb.XGBClassifier(
            booster=rant,   
            n_estimators=best_alpha,   
            max_depth=4,        
            learning_rate=0.1,  
            reg_lambda=1,       
            reg_alpha=0,         
            objective="binary:logistic"
)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        y_scores = clf.predict_proba(X_train)[:, 1] 
        y_scores_test = clf.predict_proba(test_x)[:, 1] 
        prob[rant] = y_scores_test
        best_models[rant] = best_alpha
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_train, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    axs[1].plot(fpr, tpr, color=palettes1[0], label=f"ROC curve (AUC = {roc_auc:.2f})")
    axs[1].plot([0, 1], [0, 1], color=palettes1[1], linestyle="--")  
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].set_title(f"ROC Curve for Accur: {round((best_cross_val_acc),3)}, aplha:{round(best_alpha,5)}")



    
    
    
    
    


best_models1 = max(best_models, key=best_models.get)
print(str(best_models1).split("'")[0])
prob_r = pd.DataFrame({"id":test_x.index,
                       "Rainfall Prob":prob[best_models1]}
                      )

prob_r.to_csv('submission.csv', index=False)
prob_r.head()

