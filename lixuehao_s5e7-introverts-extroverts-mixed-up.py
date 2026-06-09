# The usual things
import numpy as np
import matplotlib.pyplot as plt
# Set matplotlib defaults
plt.rcdefaults()
import pandas as pd

# For confusion_dots_roc()
from sklearn import metrics

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metric
from sklearn.metrics import roc_auc_score   # roc_auc_score(y_true, y_score)

# For the imputer
from sklearn.experimental import enable_iterative_imputer # Required for IterativeImputer
from sklearn.impute import IterativeImputer

import warnings
warnings.simplefilter('ignore')


def confusion_dots_roc(y01, yprob, title, roc_label, plots_prefix="", 
                           plotsize=(10, 4), y_thresh = 0.5, plot_alpha = 0.35):
    # Make a "confusion dots" plot and show an ROC curve.
    #   y01 is the binary truth.
    #   yprob is the predicted proba from the model.
    # Copied from "Learning Optuna for CV" notebook, made some little changes.
  
    ysframe = pd.DataFrame([y01, yprob], index=['y', 'y_prob']).transpose()
    # Add a blurred y column
    ysframe['y (blurred)'] = ysframe['y'] + 0.1 * np.random.randn(len(ysframe))

    # Plot the real y (blurred) vs the predicted probability
    # Note the flipped ylim values.
    ysframe.plot.scatter('y_prob', 'y (blurred)', figsize=plotsize,
                         s=2, xlim=(0.0, 1.0), ylim=(1.8, -0.8), alpha=plot_alpha)
    # show the "correct" locations on the plot
    plt.plot([0.0, y_thresh], [0.0, 0.0], '-',
        color='green', linewidth=3)
    plt.plot([y_thresh, y_thresh], [0.0, 1.0], '-',
        color='gray', linewidth=2)
    plt.plot([y_thresh, 1.0], [1.0, 1.0], '-',
        color='green', linewidth=3)
    plt.title("Confusion-dots Plot: " + title, fontsize=16)
    # some labels
    ythr2 = y_thresh/2.0
    plt.text(ythr2 - 0.03, 1.52, "FN", fontsize=16, color='red')
    plt.text(ythr2 + 0.5 - 0.03, 1.52, "TP", fontsize=16, color='green')
    plt.text(ythr2 - 0.03, -0.50, "TN", fontsize=16, color='green')
    plt.text(ythr2 + 0.5 - 0.03, -0.50, "FP", fontsize=16, color='red')

    if len(plots_prefix) > 0: plt.savefig(plots_prefix+"_dots.png")
    plt.show()

    # Make the ROC plot -- How to adjust its plot size?!?
    fpr, tpr, thresholds = metrics.roc_curve(y01, yprob)
    roc_auc = metrics.auc(fpr, tpr)
    thisfig = plt.figure(figsize=(3,3))
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name=roc_label)
    display.plot()
    plt.plot([0.,1.],[0.,1.],c='gray',alpha=0.3)
    if len(plots_prefix) > 0: plt.savefig(plots_prefix+"_roc.png")
    plt.show()


# Get the train data
if True:
    train = pd.read_csv("../input/playground-series-s5e7/train.csv")
else:  # use the "original" data as train
    path = '/kaggle/input/extrovert-vs-introvert-behavior-data/'
    original_data = "personality_dataset.csv"
    train = pd.read_csv(path+original_data).reset_index()

# and test
test = pd.read_csv("../input/playground-series-s5e7/test.csv")


# Make any column adjustments here:

# Add an NaN count column based on the 7 initial feature columns
train["numnan"] = train.isnull().sum(axis=1)
test["numnan"] = test.isnull().sum(axis=1)

# Convert Stage... and Drained... into numeric values with N/A in middle
for this_df in [train, test]:
    for this_col in ["Stage_fear","Drained_after_socializing"]:
        newvals = np.zeros(len(this_df)) + 0.5
        newvals[this_df[this_col] == "No"] = 0.0
        newvals[this_df[this_col] == "Yes"] = 1.0
        this_df[this_col] = newvals

# The feature column names (same for train and test):
feat_cols = list(train.columns[[1,2,3,4,5,6,7,9]])

# Five columns to impute

# Make train_imp, test_imp to use with LR to determine the "obvious" personality
train_imp = train.copy()
test_imp = test.copy()
# Constant imputing: set NaN to an overlap value for each feature using plots in:
#   https://www.kaggle.com/code/elainedazzio/20250701-pg7-eda
if False: #for this_df in [train_imp, test_imp]:
    this_df[feat_cols[0]] = this_df[feat_cols[0]].fillna(3.5)
    this_df[feat_cols[2]] = this_df[feat_cols[2]].fillna(3.3)
    this_df[feat_cols[3]] = this_df[feat_cols[3]].fillna(2.9)
    this_df[feat_cols[5]] = this_df[feat_cols[5]].fillna(4.9)
    this_df[feat_cols[6]] = this_df[feat_cols[6]].fillna(2.9)   
# Iterative Imputer, imputing integer values like the data
impute_cols = list(train_imp.columns[[1,3,4,6,7]])
if True:
    imputer_train = IterativeImputer(max_iter=10, random_state=42)
    print(f"Impute missing values in train, test for columns: \n{impute_cols}")
    train_imp[impute_cols] = (imputer_train.fit_transform(train_imp[impute_cols])).astype(int)
    # use train imputer on test
    test_imp[impute_cols] = (imputer_train.transform(test_imp[impute_cols])).astype(int)


# For the RF models, replace NaN with -2 to use them as a feature category
for this_df in [train, test]:
    this_df[feat_cols[0]] = this_df[feat_cols[0]].fillna(-2)
    this_df[feat_cols[2]] = this_df[feat_cols[2]].fillna(-2)
    this_df[feat_cols[3]] = this_df[feat_cols[3]].fillna(-2)
    this_df[feat_cols[5]] = this_df[feat_cols[5]].fillna(-2)
    this_df[feat_cols[6]] = this_df[feat_cols[6]].fillna(-2)
    
# Keep an integer version of train, use blurred version, below, for training
train_int = train.copy()
# Test integer version
test_int = test.copy() 

# Blur the train values to add some regularization and RF variety
# Show the histograms of the imputed, blurred values.
for this_col in impute_cols:
    np.random.seed(5)  # Use the same blurring each time
    train[this_col] = train[this_col] + 0.3*np.random.randn(len(train))
    test[this_col] = test[this_col] + 0.3*np.random.randn(len(test))
    #
    plt.figure(figsize=(8,2.5))
    plt.hist(train[this_col],bins=100)
    plt.title("Histogram of "+this_col)
    plt.show()


print(train.info())
##print(train.describe())
# Look for exact duplicates of feature sets - None
##print(train.duplicated(subset=feat_cols).sum())
train.tail(5)


print(test.info())
##print(test.describe())
test.tail(5)


# Show numbers of unique values for the features counts, large for blurred values
print(train.nunique())

# See specific unique values:
##for this_col in train.columns[1:]:
##    print(train[this_col].value_counts())


# LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
#                    intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
#                    max_iter=100, multi_class='deprecated', verbose=0, warm_start=False,
#                    n_jobs=None, l1_ratio=None)

# First pass using all training data
X = train_imp[feat_cols]
y = np.array(1.0*(train_imp["Personality"] == "Extrovert"))
lrmod = LogisticRegression(random_state=17, C=10.0).fit(X,y)

##print(lrmod.classes_)
y_hat = lrmod.predict_proba(train_imp[feat_cols])[:,1]
print(lrmod.n_iter_[0])
print(lrmod.coef_[0,:])
print(lrmod.score(X,y))
print("ROC AUC = {:.4f}".format(roc_auc_score(y, y_hat)))

confusion_dots_roc(y, y_hat, "LR-all", "LR-all", plots_prefix="", plotsize=(10, 4))


# Oriented like the confusion dots plot:
# True, False are the real label values ("y axis"),
# 1, 0 are the thresholded (x-axis) predictions
pred_thresh = 0.50
pd.crosstab(y != 0, 1.0*np.array(y_hat > pred_thresh))


# Looking at the confusion dots plot, remove the "clearly misclassified" ones
exclude_these = ((y == 1) & (y_hat < 0.25)) | ((y == 0) & (y_hat > 0.85))


# Repeat LR with exclusions 
Xe = train_imp.loc[~exclude_these,feat_cols]
ye = np.array(1.0*(train_imp.loc[~exclude_these,"Personality"] == "Extrovert"))
lrmode = LogisticRegression(random_state=17, C=10.0).fit(Xe,ye)
ye_hat = lrmode.predict_proba(train_imp.loc[~exclude_these,feat_cols])[:,1]
print(lrmode.n_iter_[0])
print(lrmode.coef_[0,:])
print(lrmode.score(Xe,ye))
print("ROC AUC = {:.4f}".format(roc_auc_score(ye, ye_hat)))


# Adjust the threshold for minimum errors
# Oriented like the confusion dots plot:
# True, False are the real label values ("y axis"),
# 1, 0 are the thresholded (x-axis) predictions
pred_thresh = 0.50
pd.crosstab(ye != 0, 1.0*np.array(ye_hat > pred_thresh))


confusion_dots_roc(ye, ye_hat, "LR-ex", "LR-ex", plots_prefix="", plotsize=(10, 4),
                  y_thresh=pred_thresh)


# Make the predictions on all data using the with-exclusions model
y_hat = lrmode.predict_proba(train_imp[feat_cols])[:,1]
print("ROC AUC = {:.4f}".format(roc_auc_score(y, y_hat)))


# Oriented like the confusion dots plot:
# True, False are the real label values ("y axis"),
# 1, 0 are the thresholded (x-axis) predictions

# Final selection for predictions
pred_thresh = 0.50
pd.crosstab(y != 0, 1.0*np.array(y_hat > pred_thresh))


confusion_dots_roc(y, y_hat, "LR-ex on All", "LR-ex on All", plots_prefix="", plotsize=(10, 4),
                  y_thresh=pred_thresh)


#  RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None,
#                     min_samples_split=2, min_samples_leaf=1, max_features='auto',
#                     max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1,
#                     random_state=None, verbose=0, min_density=None, compute_importances=None)


# RF model to predict Looks-Like Intros that should be labeled as Extros

# Select just the LL Intros
llintros = y_hat < pred_thresh
print("pred_thresh = ",pred_thresh,"(separates intros and extros)")

# Predictions with just LL Intros 
Xi = train.loc[llintros,feat_cols]
yi = np.array(1.0*(train.loc[llintros,"Personality"] == "Extrovert"))

# Random Forest
rfmodi = RandomForestClassifier(n_estimators=250, random_state=17,
                            max_depth=4, max_features=4,   # ~ sqrt
                                min_samples_leaf=5,
                               ).fit(Xi, yi, sample_weight=0.10+0.90*(yi))

# Make predictions using the unblurred, integer values to reduce memorization
yi_hat = rfmodi.predict_proba(train_int.loc[llintros,feat_cols])[:,1]
print(rfmodi.classes_)
print(feat_cols, "\n", rfmodi.feature_importances_)
print("ROC AUC = {:.4f}".format(roc_auc_score(yi, yi_hat)))


intro_thresh = 0.60

confusion_dots_roc(yi, yi_hat, "RF-Intros", "RF-Intros", plots_prefix="", plotsize=(10, 4),
                  y_thresh=intro_thresh)

pd.crosstab(yi != 0, 1.0*np.array(yi_hat > intro_thresh))


# RF model to predict Looks-Like Extros that should be labeled Intros

# Select just the LL Extros
llextros = y_hat > pred_thresh
print("pred_thresh = ",pred_thresh,"(separates intros and extros)")

# Predictions with just LL Extros 
Xx = train.loc[llextros,feat_cols]
yx = np.array(1.0*(train.loc[llextros,"Personality"] == "Extrovert"))

# LR model
##lrmodx = LogisticRegression(random_state=17, C=10.0, class_weight={0:1.0,1:0.04}).fit(Xx,yx)
##yx_hat = lrmodx.predict_proba(train.loc[llextros,feat_cols])[:,1]

# Random Forest
rfmodx = RandomForestClassifier(n_estimators=250, random_state=17,
                            max_depth=4, max_features=4,   # ~ sqrt
                                  min_samples_leaf=5,
                               ).fit(Xx, yx, sample_weight=0.04+0.96*(1-yx))

# Make predictions using the unblurred, integer values to reduce memorization
yx_hat = rfmodx.predict_proba(train_int.loc[llextros,feat_cols])[:,1]
print(rfmodx.classes_)
print(feat_cols, "\n", rfmodx.feature_importances_)
print("ROC AUC = {:.4f}".format(roc_auc_score(yx, yx_hat)))


extro_thresh = 0.20

confusion_dots_roc(yx, yx_hat, "RF-Extros", "RF-Extros", plots_prefix="", plotsize=(10, 4),
                  y_thresh=extro_thresh)

pd.crosstab(yx != 0, 1.0*np.array(yx_hat > extro_thresh))


# Create the train predictions the way test will be created
# Get nominal predictions from LR model
y_train = lrmode.predict_proba(train_imp[feat_cols])[:,1]
y01_train = 1.0*np.array(y_train > pred_thresh)

# intros --> extros
y_train_intro = rfmodi.predict_proba(train_int[feat_cols])[:,1]
# Find           predicted intro   with a   high intros-extro value
change_these = (y_train < pred_thresh) & (y_train_intro > intro_thresh)
y01_train[change_these] = 1.0   # Make the intro into an extro
print(sum(1*change_these),"intros changed to extros")

# extros --> intros
y_train_extro = rfmodx.predict_proba(train_int[feat_cols])[:,1]
# Find           predicted extro   with a   low extros-extro value
change_these = (y_train > pred_thresh) & (y_train_extro < extro_thresh)
y01_train[change_these] = 0.0   # Make the extro into an intro
print(sum(1*change_these),"extros changed to intros")

print("ROC AUC = {:.4f}".format(roc_auc_score(y, y01_train)))
pd.crosstab(y != 0, y01_train)


# Can adjust this to be different from intro_thresh
# Since Test is 1/3 of Train, expect only 1/3 as many reliably detected
# So 12, 4 changed in train --expect--> 4, 1 in test; adjust thresholds to get those or less.
test_intro_thresh = 0.543+0.020
test_extro_thresh = 0.270-0.050


# Make and output the Test predictions
y_test = lrmode.predict_proba(test_imp[feat_cols])[:,1]
y01_test = 1.0*np.array(y_test > pred_thresh)

# intros --> extros
y_test_intro = rfmodi.predict_proba(test_int[feat_cols])[:,1]
# Find           predicted intros    with   high intros value
change_these = (y_test < pred_thresh) & (y_test_intro > test_intro_thresh)
y01_test[change_these] = 1.0   # Make the intro into an extro
print(sum(1*change_these),"intros changed to extros")
print(test.loc[change_these,"id"].values)

print("")

# extros --> intros
y_test_extro = rfmodx.predict_proba(test_int[feat_cols])[:,1]
# Find           predicted extro   with a   high extros value
change_these = (y_test > pred_thresh) & (y_test_extro < test_extro_thresh)
y01_test[change_these] = 0.0   # Make the extro into an intro
print(sum(1*change_these),"extros changed to intros")
print(test.loc[change_these,"id"].values)



# The 10 data points @Tilii7 thinks will decide the private leader:
#   19612 19668 20017 20541 21932 22547 22559 23336 23418 23844
# https://www.kaggle.com/competitions/playground-series-s5e7/discussion/590008 

# fyi, all ones RF wanted to change, merging from v23 on:

# [19386 20001 20017...... 20519 20934........ 21138 21294. 21499.... 21537 21843.
#    22172 23142. 23328.. 23892. 24351]

# [19876.... 19292 20380 21800 21932 22286 22782 23336. 23350...... 23418.....] 



# _ _ _
# Samples and their (known?) values, taken from discussion post:
#   https://www.kaggle.com/competitions/playground-series-s5e7/discussion/588664#3244247
match_results = (
"18556    Introvert \
18572    Extrovert \
18864    Introvert \
18970    Extrovert \
19135    Extrovert \
19167    Extrovert \
19257    Introvert \
19301    Extrovert \
19327    Extrovert \
19388    Extrovert \
19531    Extrovert \
19597    Extrovert \
19997    Introvert \
20112    Introvert \
20240    Introvert \
20531    Introvert \
20659    Extrovert \
21266    Extrovert \
21315    Extrovert \
21648    Introvert \
21920    Extrovert \
22182    Introvert \
22356    Introvert \
22424    Extrovert \
22481    Extrovert \
22600    Extrovert \
22606    Introvert \
22711    Introvert \
23215    Extrovert \
23454    Extrovert \
23655    Extrovert \
23750    Extrovert \
23817    Extrovert \
24008    Introvert \
24493    Introvert")

match_results = match_results.replace("    "," ")
match_results = match_results.split(" ")
print("Out of", int(len(match_results)/2), "matching values, these changed:")
# Can insert if False: here to not make the (disqualifying?!) match changes.
#    https://www.kaggle.com/competitions/playground-series-s5e7/discussion/590849
##for ipair in range(int(len(match_results)/2)):
if False:
    pair_id = int(match_results[2*ipair])
    pair_pers = match_results[2*ipair+1]
    y01_orig = y01_test[pair_id-18524]
    # Change the y01 values
    y01_test[pair_id-18524] = 1*(pair_pers == "Extrovert")
    # print the ones that are changed
    if (((pair_pers == "Extrovert") and y01_orig < 0.5) or
                ((pair_pers == "Introvert") and y01_orig > 0.5)):
        print(" ", pair_id, pair_pers, "set y01 =",
                  y01_test[pair_id-18524], "(was "+str(y01_orig)+")")

# 8 are different from my nominal LR predictions


test["Personality"] = "Introvert"
test.loc[y01_test > 0.5, "Personality"] = "Extrovert"
test


test[['id','Personality']].to_csv("submission.csv", header=True, index=False)


##!head submission.csv


# Versions summary
#                 Model based on 2nd Logistic Regression excluding the clear "errors"
#             --> Filling NaNs in the 5 integer features with "overlap"/ambiguous values.
# v 6  0.974898   y_test > 0.09  C=1.0
# v 5  0.974898   y_test > 0.1
# v 4  0.974898   y_test > 0.20   31 errors
# v 2  0.974.89   y_test > 0.28   32 errors
# v 3  0.973279   y_test > 0.40   33 errors
# v 7  0.974.89   y_test > 0.28
#             --> Added the nan count, numnan, to the features before imputing.
# v 8  0.974.89   Also tried predicting "flipped" Intros and Extros - LR AUCs 0.65, 0.63 .
# v 9  0.974.89   y_test > 0.28 Using the Original Data in place of train.
#             --> Use RF for the intros and extros separately.
#                 Can predict 6 of the training intros should be extros.
# v10             Use the model to make intro-extro changes to the test predictions.
#                 Saving v10 messed up, lost some work. v11-13 quick saves to be sure.
#                 RF model (est=250, depth=3) used to find a few intros that should be extros.
# v14  0.974.89   Start with LR predictions, then change none with intros thresh > 0.63 .
# v15  0.974898   Use intro_thresh = 0.60 1 changed
# v16  0.974898   Use test intro_thresh = 0.57 2 changed
# v17  0.974.89   32 errors  Use test intro_thresh = 0.55 8 changed
#             --> Included the 35 "Match" changes given in discussion post.
# v18  0.974898   31 errors
# v19  0.970040   37 errors  Used opposite of "Match" changes as a check
# v20  0.974898   31 errors  LR and Matches, no Intros corrections, test_intro_thresh = 0.77
# v21  0.9757+8   30 errors  Intro corrections: test_intro_thresh = 0.57 2 changed
# v22  0.974.89   Removed Stage,Drain from the NaN count, adjusted pred_thresh. Add RF for Extros
# v23  0.9757+8 * 30 errors  Changed NaN count back. Intros = 0.57 2 changed; Extros 0 changed
# v24  0.974898   31 errors  Intros = 0.57 2 changed; Extros 0.27 2 changed 
# v25  0.974898   31 errors  Intros = 0.56 5 changed; Extros 0 changed
#             --> Using Iterative Imputer on the 5 integer features.
# v26  0.974898   Using blurred feats. ROCs: 0.7350, 0.7182 Threshs: 0.55 3 changed, 0.30 1 changed
# v27  0.974898   RFs: feat.s=4.  ROCs: 0.7398, 0.7142  Threshs: 0.545 3 changed, 0.29 2 changed
# v28  0.974898 * RFs: feat=4,depth=4. ROCs: 0.7806,0.7434 Threshs: 0.54 3 change, 0.29 2 change
# v29  0.973279   33 errors  v28 with Threshs: 0.530 6 change, 0.325 3 change
# v30  0.974898   31 errors  v28 with Threshs: 0.54 3 change, 0.325 3 change   So:
#                 The additional Extro that changed from v28 to v30 is not in the 1235 LB.
#                 2 of the 3 Intros that changed between v29 and v30 are in the 1235, 1 is not.
#                 Bottom line: the RFs are not finding misclassified test samples.
# v32  0.973279   33 errors Used min_sample_leaf=5 for RFs, gives slight changes in AUC.
#                            Threshs: 0.540 6 change, 0.325 4 change
# v33  0.974898   31 errors Filled the 5 imputed feat.s' NaNs = -1; random blur to 0.1; keep min-leaf=5
# v34  0.974898   31 errors Use train-imputed for LR, train w/NaNs=-2, blur=0.3 for RF
#                            Threshs: 0.543 3 change, 0.270 3 change
# v35  0.974.89   32 errors Same as v34 but don't change the 8 "known, matched" ones from disc. post
# v36  0.973279   33 errors Same as v34 but Threshs: *0.540 5 change*, 0.270 3 change
# v37  0.974.89   32 errors Same as v34 but predictions made with train_int, test_int. Matches are changed.
#                            Threshs: 0.563 3 change, 0.220 2 change
# v38                       Same as v37, but no "known, matched" ones changed.




