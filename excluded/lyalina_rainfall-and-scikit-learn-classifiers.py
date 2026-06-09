import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold,cross_val_score,RepeatedKFold,GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC,NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier,GradientBoostingClassifier,AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss,brier_score_loss
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier

from scipy import signal
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings("ignore")        
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import seaborn as sns
import squarify

#
background_color='#fbfbfb'
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['figure.dpi'] = 100

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
clr = Style.RESET_ALL

train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

TARGET = 'rainfall'
RMV =['id','rainfall','maxtemp',	'temparature',	'mintemp',	'winddirection',	'windspeed'] #id

FEATURES = train_data.drop(RMV, axis=1).columns.tolist()

#sns.palplot(['#0d0d0d','#0e4f66','gray','#fbfbfb','#75141E','#0fbf8f','#2296b8','#9ac8d6'])



x=train_data.groupby(['rainfall'])['rainfall'].count()
y=len(train_data)
r=((x/y)).round(2)
ratio = pd.DataFrame(r).T


fig, ax = plt.subplots(1,1,figsize=(4.5, 1),dpi=150)
background_color = "#fbfbfb"
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color) 

ax.barh(ratio.index, ratio[1.0],  color='#0e4f66', alpha=0.9, ec=background_color, label='Rain')
ax.barh(ratio.index, ratio[0.0], left=ratio[1.0], color='gray', alpha=0.9,ec=background_color, label='Not rain')

ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.legend().set_visible(False)
for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)
    
for i in ratio.index:
    ax.annotate(f"{int(ratio[1.0][i]*100)}%", xy=(ratio[1.0][i]/2, i),va = 'center', ha='center',fontsize=20, fontweight='light', fontfamily='monoserif',color='white')
    ax.annotate("Rain", xy=(ratio[1.0][i]/2, -0.25),va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='sans-serif',color='white')
    
    
for i in ratio.index:
    ax.annotate(f"{int(ratio[0.0][i]*100)}%", xy=(ratio[1.0][i]+ratio[0.0][i]/2, i),va = 'center', ha='center',fontsize=20, fontweight='light', fontfamily='monoserif',color='white')
    ax.annotate("Not rain", xy=(ratio[1.0][i]+ratio[0.0][i]/2, -0.25),va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='sans-serif',color='white')


fig.text(0.125,1.1,'How often did it rained?', fontfamily='sans-serif',fontsize=12, fontweight='bold')


plt.show()



props=' z-index: 1; border: 1px solid #fbfbfb;'\
                         'background-color: #9ac8d6; color: #0e4f66;'\
                         'padding: 0.6em; border-radius: 0.1em;'

index_names = { 'selector': '.index_name', 'props': 'font-style: italic; color: darkgrey; font-weight:normal;'}
headers =     { 'selector': 'th:not(.index_name)',  'props': 'background-color: #fbfbfb; color: #0d0d0d;'}
caption =     { 'selector': 'caption','props': 'caption-side: top; color: 0d0d0d; font-size:2em; font-weight:normal; text-align: left; font-family: sans-serif'}

summary_styler = train_data.agg(["max", "min","mean", "std"]).style \
                   .format(precision=3) \
                   .relabel_index(["max", "min","mean", "std"]).set_properties(**{'color':'#0d0d0d', 'background-color': '#fbfbfb'})

display(train_data[0:5].style.format(precision=3).concat(summary_styler).map(lambda x: props).set_caption("Overview train dataset").set_table_styles([ caption, index_names, headers]))

summary_styler = test_data.agg(["max", "min","mean", "std"]).style \
                   .format(precision=3) \
                   .relabel_index(["max", "min","mean", "std"]).set_properties(**{'color':'#0d0d0d', 'background-color': '#fbfbfb'})
print("\n")
display(test_data[0:5].style.format(precision=3).concat(summary_styler).map(lambda x: props).set_caption("Overview test dataset").set_table_styles([ caption, index_names, headers]))


color_palette=['#2296b8',"#0e4f66"] #palette=color_palette,

ax = sns.pairplot(train_data[FEATURES+list(['rainfall'])], diag_kind="kde",  hue='rainfall', palette=color_palette, corner=True)




ax = train_data.drop('id', axis =1).plot.box(rot=90, figsize=(16, 6))
  
ax.text(0,-0.2,'Variable Box Plot', fontfamily='sans-serif',fontsize=14, fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.set_facecolor(background_color) 

#ax.set_yticks([])
ax.legend().set_visible(False)

for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

plt.show()



def map_day_to_month(day):    
    if day <= 31:
        return 1
    elif day <= 59:
        return 2
    elif day <= 90:
        return 3
    elif day <= 120:
        return 4
    elif day <= 151:
        return 5
    elif day <= 181:
        return 6
    elif day <= 212:
        return 7
    elif day <= 243:
        return 8
    elif day <= 273:
        return 9
    elif day <= 304:
        return 10
    elif day <= 334:
        return 11
    else:
        return 12

def FE(df):

    df['month'] = df['day'].apply(map_day_to_month)
    df['pressure_prev'] = df['pressure'].shift(1)
    df['winddirection_prev'] = df['winddirection'].shift(1)
    df['humidity_prev'] = df['humidity'].shift(1)
    df['dewpoint_prev'] = df['dewpoint'].shift(1)


    df['sunshine_mintemp'] =  df['sunshine'] * df['mintemp']
    df['humidity_cloud_ratio'] =   df['humidity'] + 1e-12 / (df['cloud'] + 1e-12)
   # df['cloud_humidity'] =   df['cloud'] * df['humidity']
    df['temparature_humidity_cloud'] = 0.9 * df['humidity'] + 0.1* df['temparature'] + df['cloud'] 
    #df['weather_index'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
        
    df['BIN_cloud']=0
    df.loc[df['cloud']<=20,'BIN_cloud']=0
    df.loc[(df['cloud']>20)&(df['cloud']<=40),'BIN_cloud']=1
    df.loc[(df['cloud']>40)&(df['cloud']<=60),'BIN_cloud']=2
    df.loc[(df['cloud']>60)&(df['cloud']<=80),'BIN_cloud']=3
    df.loc[df['cloud']>80,'BIN_cloud']=4
    
    df['BIN_humidity']=0
    df.loc[df['humidity']<=50,'BIN_humidity']=0
    df.loc[(df['humidity']>50)&(df['humidity']<=60),'BIN_humidity']=1
    df.loc[(df['humidity']>60)&(df['humidity']<=70),'BIN_humidity']=2
    df.loc[(df['humidity']>70)&(df['humidity']<=85),'BIN_humidity']=3
    df.loc[df['humidity']>85,'BIN_humidity']=4

  #  df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
   # df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
 #   
 #   features_avg = ['cloud', 'humidity']  
    # Relative features 
 #   for feature in features_avg:
 #       df[f'{feature}_relative'] = df[feature] / df.groupby('month')[feature].transform('max') 
        
    # Lag features 
   # for feature in features_avg:
   #     for lag in [1, 2]:
   #         df[f'{feature}_lag{lag}'] = df[feature].shift(lag).fillna(df[feature].median()) 

    # Rolling Median
    #for feature in features_avg:
   #     df[f'{feature}_rollings2'] = df[feature].rolling(2, min_periods=1).median()
      

    return df

FE(train_data)
FE(test_data)

dayhat = savgol_filter(train_data['day'].values,100, 1)
train_data['day2'] = dayhat.astype(np.int64)
test_data['day2'] = savgol_filter(test_data['day'].values,100, 1).astype(np.int64)

for col in train_data.columns:
   train_data[col] = train_data[col].fillna(np.mean(train_data[col]))
    
for col in test_data.columns:
   test_data[col] = test_data[col].fillna(np.mean(test_data[col]))

FEATURES = train_data.drop(RMV, axis=1).columns.tolist()


# Custom color map
colors = ['#aac8d6', "#66bbd0","#0e4f66"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

fig = plt.figure(figsize=(16,16), facecolor=background_color)
gs = fig.add_gridspec(2, 1)
gs.update(wspace=0.5, hspace=0.5)
ax0 = fig.add_subplot(gs[0,:])
ax1 = fig.add_subplot(gs[1,:])

corr_1 = np.absolute(train_data.query("rainfall == 1")[FEATURES].corr())
corr_0 = np.absolute(train_data.query("rainfall == 0")[FEATURES].corr())

#sns.set(font_scale=0.5)
# Heatmap
sns.heatmap(corr_1[corr_1>0.5], linewidth=0.5, annot=True, fmt='.1%', cbar=False, cmap=colormap, ax=ax0, annot_kws={"size": 5},)
sns.heatmap(corr_0[corr_1>0.5], linewidth=0.5, annot=True, fmt='.1%', cbar=False, cmap=colormap, ax=ax1 , annot_kws={"size": 5},)

for i in range(0, 2):
        locals()["ax"+str(i)].tick_params(axis='both', which='both', length=0, size=10)
        locals()["ax"+str(i)].set_facecolor(background_color)
     
# Text
ax0.text(0,-3,'Variable Correlation: Heatmap',fontsize=24, fontweight='bold')
ax0.text(0,-1.5,'when it was raining',fontsize=15, fontweight='bold')

ax1.text(0,-1.5,'when there was no rain',fontsize=15, fontweight='bold')


plt.show()


FEATURES = train_data.drop(RMV, axis=1).columns.tolist()
X = train_data[FEATURES].copy()
y = train_data[TARGET].copy()
test = test_data[FEATURES].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.3, random_state=42)

C = 10
kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3))

log_cols=["Classifier", "Accuracy", "LogLoss","BrierScore","AUC"]
log = pd.DataFrame(columns=log_cols)


y_test_predict = dict()
oof_test = dict()
oof_proba_test = dict()

if 1:
    min_features_to_select = 1 
    
    cv = StratifiedKFold(5)
    
    rfecv = RFECV(
        estimator=linear_model.LogisticRegression(),
        step=1,
        cv=cv,
        #scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(X_train, y_train)
    
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"{red}{rfecv.get_feature_names_out()}{clr}")
    
    X = train_data[rfecv.get_feature_names_out()].copy()
    y = train_data[TARGET].copy()
    test = test_data[rfecv.get_feature_names_out()].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.3, random_state=42)


from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score,confusion_matrix
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=80)

oversample = SMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.ravel())


rf_pipeline = Pipeline(steps = [('scale',RobustScaler()),('rf',MLPClassifier(alpha=10,activation='logistic',solver='lbfgs',  max_iter=1000, verbose =True, random_state=42,learning_rate_init=0.001, power_t=0.5,  tol=0.0001,early_stopping=True))])
logreg_pipeline = Pipeline(steps = [('scale',RobustScaler()),('LR',KNeighborsClassifier(n_neighbors = 50,weights =  'distance'))])
print('~ Fit SMOTE :')
print('Classifier1 valid dataset mean accuracy :',cross_val_score(rf_pipeline,X_train_resh,y_train_resh,cv=10,scoring='accuracy').mean())

rf_pipeline.fit(X_train_resh,y_train_resh)
logreg_pipeline.fit(X_train_resh,y_train_resh)

rf_pred_proba   = rf_pipeline.predict_proba(X_test)[:, 1] 
rf_pred   = rf_pipeline.predict(X_test)
rf_cm  = confusion_matrix(y_test,rf_pred )
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_br_score = brier_score_loss(y_test, rf_pred_proba)
rf_ls = log_loss(y_test, rf_pred_proba)

fpr, tpr, thresholds = roc_curve(y_test, rf_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"  for valid dataset accuracy: {blu}{rf_accuracy:0.1%}{clr}, log loss: {blu}{rf_ls:0.1%}{clr}, Brier: {blu}{rf_br_score:0.3}{clr}, AUC: {blu}{roc_auc:0.3}{clr}")


log_entry = pd.DataFrame([['Classifier1 SMOTE', rf_accuracy, rf_ls,rf_br_score,roc_auc]], columns=log_cols) 
log = pd.concat([log, log_entry], ignore_index=True)

print('Classifier2 valid dataset mean accuracy :',cross_val_score(logreg_pipeline,X_train_resh,y_train_resh,cv=10,scoring='accuracy').mean())

logreg_pred_proba   = logreg_pipeline.predict_proba(X_test)[:, 1] 
logreg_pred = logreg_pipeline.predict(X_test)
logreg_cm  = confusion_matrix(y_test,logreg_pred)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_br_score = brier_score_loss(y_test, logreg_pred_proba)
logreg_ls = log_loss(y_test, logreg_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, logreg_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"  for valid dataset accuracy: {blu}{logreg_accuracy:0.1%}{clr}, log loss: {blu}{logreg_ls:0.1%}{clr}, Brier: {blu}{logreg_br_score:0.3}{clr}, AUC: {blu}{roc_auc:0.3}{clr}")
print("\n")
log_entry = pd.DataFrame([["Classifier2 SMOTE", logreg_accuracy, logreg_ls,logreg_br_score,roc_auc]], columns=log_cols) 
log = pd.concat([log, log_entry], ignore_index=True)


#make predict

y_test_predict['Classifier1 SMOTE'] = rf_pipeline.fit(X_train_resh, y_train_resh).predict_proba(test)[:, 1]
y_test_predict['Classifier2 SMOTE'] = logreg_pipeline.fit(X_train_resh, y_train_resh).predict_proba(test)[:, 1]

# plot Confusion matrix 
fig = plt.figure(figsize=(15,18), dpi=150) 
fig.patch.set_facecolor(background_color) 
gs = fig.add_gridspec(5, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

sns.heatmap(rf_cm, linewidths=2.5,yticklabels=['Actual not rain','Actual rain'],xticklabels=['Predicted not rain','Predicted rain'], cmap=colormap, cbar=None,annot=True,fmt='d',ax=ax0,annot_kws={"fontsize":15})

ax0.set_facecolor(background_color) 
ax0.tick_params(axis=u'both', which=u'both',length=0)


for s in ["top","right","left"]:
    ax0.spines[s].set_visible(False)
    
ax0.text(0, -0.3, 'Confusion matrix SMOTE',fontsize=15, fontweight='bold', fontfamily='sans-serif')
    
plt.show()


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, rf_pred_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
#precision, recall, thresholds_pr = precision_recall_curve(y_test, rf_pred_proba)


#print("AUC:", roc_auc)

#plt.figure(figsize=(12, 5))
fig = plt.figure(figsize=(12, 5), dpi=100) 
fig.patch.set_facecolor(background_color) 

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()




# Fit estimators
ESTIMATORS = {
    "GaussianNB":        GaussianNB(var_smoothing=0.2),
    "Nearest Neighbors": KNeighborsClassifier(n_neighbors = 50,weights =  'distance'),
    "LinearDiscriminantAnalysis" :  LinearDiscriminantAnalysis(solver='svd',tol=0.000001),
    "LinearDiscriminantAnalysis1" : LinearDiscriminantAnalysis(solver='lsqr',tol=0.000001),
    "QuadraticDiscriminantAnalysis" : QuadraticDiscriminantAnalysis(reg_param=0.0001),
    "Random Forest":     RandomForestClassifier( max_depth=15, n_estimators=2000, max_features='log2', random_state=42    ),
    "Neural Net":        MLPClassifier(alpha=10,activation='logistic',solver='lbfgs',  max_iter=4000, verbose =False, random_state=42,learning_rate_init=0.01, power_t=0.5,  tol=0.0001,early_stopping=True),
    "L1 logistic":       linear_model.LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=2000),
    "L2 logistic (Multinomial)": linear_model.LogisticRegression( C=0.2, penalty="l2", solver="saga", max_iter=2000),
    "L2 logistic (OvR)": OneVsRestClassifier(linear_model.LogisticRegression(C=100, penalty="l2", solver="saga", max_iter=2000)),
    "Linear SVC":        SVC(kernel="linear",probability=True, C=0.025, random_state=42),
    "AdaBoost":          AdaBoostClassifier(random_state=42),
    "GPC":               GaussianProcessClassifier(kernel,random_state=42),
    "GradientBoostingClassifier":  GradientBoostingClassifier(),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.001, max_iter=2000, max_leaf_nodes=15, max_depth=17, min_samples_leaf=20, l2_regularization=0.0,
                                                                       early_stopping='auto', scoring='loss', tol=1e-04, verbose=0, random_state=42),

}


for name, clf in ESTIMATORS.items():
    
    print(f"~ Fit model {red}{name}{clr}")
    clf.fit(X_train, y_train)
    
    oof_test[name] = clf.predict(X_test)  
    oof_proba_test[name] = clf.predict_proba(X_test)[:, 1]  
    
    accuracy = accuracy_score(y_train, clf.predict(X_train))
    print(f"  for train dataset accuracy: {accuracy:0.1%}")   
    
    accuracy = accuracy_score(y_test, oof_test[name])
    
    ls = log_loss(y_test, oof_proba_test[name])
    print(f"  for valid dataset accuracy: {accuracy:0.1%}, log loss: {ls:0.1%}")

    print(f"Calibrate model {name}:")
    
    # With isotonic calibration
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method="isotonic")
    clf_isotonic.fit(X_train, y_train)
    prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
    
    # With sigmoid calibration
    clf_sigmoid = CalibratedClassifierCV(clf, cv=5, method="sigmoid")
    clf_sigmoid.fit(X_train, y_train)
    prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]
    
    print("  Brier score losses: (the smaller the better)")
    
    clf_score = brier_score_loss(y_test, oof_proba_test[name])
    print("      No calibration: %1.3f" % clf_score)
    
    clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic)
    print("      With isotonic calibration: %1.3f" % clf_isotonic_score)
    
    clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid)
    print("      With sigmoid calibration: %1.3f" % clf_sigmoid_score)

    scores = {
         'No calibration': clf_score,
         'Isotonic calibration': clf_isotonic_score,
         'Sigmoid calibration': clf_sigmoid_score
    }

    best_model = min(scores, key=scores.get)
    best_score = scores[best_model]
    
    #print(f"The best model is: {best_model} with a score of: {best_score:.3f}")
    
    
    if best_model == 'No calibration':
        y_test_predict[name] = clf.fit(X, y).predict_proba(test)[:, 1]
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        ls = log_loss(y_test, oof_proba_test[name])
        fpr, tpr, thresholds = roc_curve(y_test, oof_proba_test[name])
        roc_auc = auc(fpr, tpr)
        
    elif best_model == 'Isotonic calibration':
        y_test_predict[name] = clf_isotonic.fit(X, y).predict_proba(test)[:, 1]
        accuracy = accuracy_score(y_test, clf_isotonic.predict(X_test))
        ls = log_loss(y_test, prob_pos_isotonic)
        fpr, tpr, thresholds = roc_curve(y_test, prob_pos_isotonic)
        roc_auc = auc(fpr, tpr)
        
    else:  # best_model == 'Sigmoid calibration'
        y_test_predict[name] = clf_sigmoid.fit(X, y).predict_proba(test)[:, 1]
        accuracy = accuracy_score(y_test, clf_sigmoid.predict(X_test))
        ls = log_loss(y_test, prob_pos_sigmoid)
        fpr, tpr, thresholds = roc_curve(y_test, prob_pos_sigmoid)
        roc_auc = auc(fpr, tpr)

    print(f"  after calibration for valid dataset accuracy: {blu}{accuracy:0.1%}{clr}, log loss : {blu}{ls:0.1%}{clr}, AUC: {blu}{roc_auc:0.3}{clr}")
    
    log_entry = pd.DataFrame([[name, accuracy, ls,best_score,roc_auc]], columns=log_cols) 
    log = pd.concat([log, log_entry], ignore_index=True)

    order = np.lexsort((oof_proba_test[name],))
    
    fig, ax = plt.subplots(1,1,figsize=(6, 4),dpi=100)
    background_color = "#fbfbfb"
    fig.patch.set_facecolor(background_color)
    
    ax.grid(color=background_color, linestyle=':', axis='y', zorder=0,  dashes=(1,5))

    sns.lineplot(oof_proba_test[name][order], ax=ax,color="gray",  label="No calibration       (%1.3f)" % clf_score)
    sns.lineplot(prob_pos_isotonic[order], ax=ax, color="#0fbf8f",  label="Isotonic calibration (%1.3f)" % clf_isotonic_score)
    sns.lineplot(prob_pos_sigmoid[order], ax=ax, color="#2296b8", label="Sigmoid calibration (%1.3f)" % clf_sigmoid_score)


    ax.set_xlabel("Instances sorted according to predicted probability")    
    ax.set_ylabel("P(y=1)")
    plt.ylim([-0.05, 1.05])
    ax.legend(loc="lower right",prop={'size': 10})
    plt.title("Calibration plot")
    
    plt.show()



from imblearn.over_sampling import BorderlineSMOTE
oversample = BorderlineSMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X, y.ravel())


print('~ Fit BorderlineSMOTE :')
rf_pipeline.fit(X_train_resh,y_train_resh)
logreg_pipeline.fit(X_train_resh,y_train_resh)

print('Classifiers1 valid dataset mean accuracy :',cross_val_score(rf_pipeline,X_train_resh,y_train_resh,cv=10,scoring='accuracy').mean())

rf_pred_proba   = rf_pipeline.predict_proba(X_test)[:, 1] 
rf_pred   = rf_pipeline.predict(X_test)
rf_cm  = confusion_matrix(y_test,rf_pred )
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_br_score = brier_score_loss(y_test, rf_pred_proba)
rf_ls = log_loss(y_test, rf_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, rf_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"  for valid dataset accuracy: {blu}{logreg_accuracy:0.1%}{clr}, log loss: {blu}{logreg_ls:0.1%}{clr}, Brier: {blu}{logreg_br_score:0.3}{clr}, AUC: {blu}{roc_auc:0.3}{clr}")

log_entry = pd.DataFrame([['Classifier1 BSMOTE', rf_accuracy, rf_ls,rf_br_score,roc_auc]], columns=log_cols) 
log = pd.concat([log, log_entry], ignore_index=True)

print('Classifiers2 valid dataset mean accuracy :',cross_val_score(logreg_pipeline,X_train_resh,y_train_resh,cv=10,scoring='accuracy').mean())

logreg_pred_proba   = logreg_pipeline.predict_proba(X_test)[:, 1] 
logreg_pred = logreg_pipeline.predict(X_test)
logreg_cm  = confusion_matrix(y_test,logreg_pred)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_br_score = brier_score_loss(y_test, logreg_pred_proba)
logreg_ls = log_loss(y_test, logreg_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, logreg_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"  for valid dataset accuracy: {blu}{logreg_accuracy:0.1%}{clr}, log loss: {blu}{logreg_ls:0.1%}{clr}, Brier: {blu}{logreg_br_score:0.3}{clr}, AUC: {blu}{roc_auc:0.3}{clr}")
print("\n")

log_entry = pd.DataFrame([["Classifier2 BSMOTE", logreg_accuracy, logreg_ls,logreg_br_score,roc_auc]], columns=log_cols) 
log = pd.concat([log, log_entry], ignore_index=True)

#make predict
y_test_predict['Classifier1 BSMOTE'] = rf_pipeline.fit(X_train_resh, y_train_resh).predict_proba(test)[:, 1]
y_test_predict['Classifier2 BSMOTE'] = logreg_pipeline.fit(X_train_resh, y_train_resh).predict_proba(test)[:, 1]

# plot Confusion matrix 
fig = plt.figure(figsize=(15,18), dpi=150) 
fig.patch.set_facecolor(background_color) 
gs = fig.add_gridspec(5, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

sns.heatmap(rf_cm, linewidths=2.5,yticklabels=['Actual not rain','Actual rain'],xticklabels=['Predicted not rain','Predicted rain'], cmap=colormap, cbar=None,annot=True,fmt='d',ax=ax0,annot_kws={"fontsize":15})

ax0.set_facecolor(background_color) 
ax0.tick_params(axis=u'both', which=u'both',length=0)


for s in ["top","right","left"]:
    ax0.spines[s].set_visible(False)
    
ax0.text(0, -0.3, 'Confusion matrix Borderline SMOTE',fontsize=15, fontweight='bold', fontfamily='sans-serif')
    
plt.show()

# plot Confusion matrix 
fig = plt.figure(figsize=(15,18), dpi=150) 
fig.patch.set_facecolor(background_color) 
gs = fig.add_gridspec(5, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

sns.heatmap(logreg_cm, linewidths=2.5,yticklabels=['Actual not rain','Actual rain'],xticklabels=['Predicted not rain','Predicted rain'], cmap=colormap, cbar=None,annot=True,fmt='d',ax=ax0,annot_kws={"fontsize":15})

ax0.set_facecolor(background_color) 
ax0.tick_params(axis=u'both', which=u'both',length=0)


for s in ["top","right","left"]:
    ax0.spines[s].set_visible(False)
    
ax0.text(0, -0.3, 'Confusion matrix Borderline SMOTE Classifier2',fontsize=15, fontweight='bold', fontfamily='sans-serif')
    
plt.show()


# show result
fig, ax = plt.subplots(1,1,figsize=(log.shape[0]*0.5, log.shape[0]*0.5))
background_color = "#fbfbfb"
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color) 

ax.barh(log['Classifier'],log['Accuracy'], color="#0e4f66", zorder=2, height=0.5, label='Accuracy,%')
ax.axvline(np.mean(log['AUC']), ls=':', color='r', label='AUC (mean),%')
ax.barh(log['Classifier'],log['LogLoss'], color="#9ac8d6", zorder=3, height=0.2,label='LogLoss,%')
ax.barh(log['Classifier'],log['AUC'], color="#2296b8", zorder=1, height=0.8,label='AUC,%')
ax.axvline(np.mean(log['LogLoss']), ls=':', color='#6d6d6d', label='LogLoss (mean),%')

fig.legend(bbox_to_anchor=(0.8,0)).set_visible(True)



for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)
    
ax.tick_params(axis=u'both', which=u'both',length=0,labelsize=8)
ax.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5)) 
 

fig.text(0.125,1,'Compare score of fitting models', fontfamily='sans-serif',fontsize=15, fontweight='bold')


plt.show()


fig = plt.figure(figsize=(log.shape[0]*0.5, log.shape[0]*0.5), dpi=150)
gs = fig.add_gridspec(4, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

colors = ["#9ac8d6","lightgray"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
# Change background color
background_color = "#fbfbfb"
fig.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color)

# Overall
df = log.set_index('Classifier')
sns.heatmap(df.T, cmap=colormap,annot=True, fmt = "0.1%", linewidths=2.5,cbar=False,ax=ax0,annot_kws={"size": 6})

ax0.tick_params(axis=u'both', which=u'both',length=0)
ax0.text(0,-2,'So our results',fontfamily='sans-serif',fontsize=20,fontweight='bold')
ax0.text(0,0,'We trained several classifiers, used calibration and cross validation,\n and verified SMOTE and BorderSMOTE.',fontfamily='sans-serif',fontsize=14)

plt.show()


pred = pd.DataFrame(y_test_predict)

#choose model with accuracy above average and top3 for ranking
ls_clf = log.loc[log["AUC"] > np.mean(log["AUC"])]['Classifier'].to_list()
#ls_clf.append('Classifier1 SMOTE')
ls_clf_top = log.sort_values(by=['AUC'], ascending=False)['Classifier'].head(4).to_list()
#ls_clf_top.append('Classifier1 SMOTE')

VER = 3
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission['rainfall'] = np.mean(pred[ls_clf], axis =1)
submission.to_csv(f"submission.csv",index=False)

fig, ax = plt.subplots(1,1,figsize=(6, 4),dpi=100)

fig.patch.set_facecolor(background_color)



sns.histplot(x=submission['rainfall'], data=submission, ax=ax,color="#0e4f66", label="Predict proba", kde=True)
ax.legend().set_visible(False)

ax.set_xlabel("")    
ax.set_ylabel("")

ax.tick_params(axis=u'both', which=u'both',length=0)

plt.title("Rain probability distribution")
    
plt.show()


from scipy.stats import rankdata

VER = 'rank'
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

submission['rainfall'] = 3 * rankdata( pred[ls_clf_top[0]] ) + 2 * rankdata( pred[ls_clf_top[1]]) + 2 * rankdata( pred[ls_clf_top[2]]) + 5 * rankdata( pred[ls_clf_top[3]] )
submission['rainfall'] = rankdata( submission['rainfall']) / len(submission)

submission.to_csv(f"submission_v{VER}.csv",index=False)


fig, ax = plt.subplots(1,1,figsize=(6, 4),dpi=100)
background_color = "#fbfbfb"
fig.patch.set_facecolor(background_color)

sns.histplot(x=submission['rainfall'], data=submission, ax=ax,color="#0e4f66", label="Predict proba", kde=True)
ax.legend().set_visible(False)

ax.set_xlabel("")    
ax.set_ylabel("")
plt.title("Rain probability distribution TOP4 classifiers with ranking")
    
plt.show()



df = pd.concat([test_data, pred], axis=1)
df['sub'] = np.mean(pred[ls_clf], axis =1)
df = df.reset_index(drop=True)

color_palette=['#2296b8',"#0e4f66"] #palette=color_palette,

fig, ax = plt.subplots(1,1,figsize=(16, 4),dpi=100)
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color) 


sns.scatterplot(data=df, y=df[ls_clf_top[0]], x=df.index, color= "#0fbf8f",size=0.3) 
sns.scatterplot(data=df, y=df[ls_clf_top[1]], x=df.index,  color= "#2296b8" , alpha=0.5,size=0.3)
sns.scatterplot(data=df, y=df[ls_clf_top[2]], x=df.index,  color= "#2296b8" , alpha=0.5,size=0.3)
sns.scatterplot(data=df, y=df[ls_clf_top[3]], x=df.index, hue=test_data["cloud"], palette=color_palette, alpha=0.7, size=df[ls_clf_top[3]])

for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)
    
ax.legend().set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Predicted probability of rain (y)  by days of observation (x)")    
ax.set_ylabel("")

fig.text(0.125,1,'Looks like it\'ll be rainy...', fontfamily='sans-serif',fontsize=20, fontweight='bold')


plt.show()

