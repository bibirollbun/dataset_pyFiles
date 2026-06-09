import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('/kaggle/input/digit-recognizer-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer-challenge/test.csv')


# Separate features and labels
y_train = train_data['label']
x_train = train_data.drop('label', axis=1)  # Still a DataFrame here


fig, axis = plt.subplots(1, 10, figsize=(10, 5))
for i, ax in enumerate(axis):
    img = train_data.iloc[i, 1:].values.reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {train_data.iloc[i, 0]}")
    ax.axis("off")
plt.show()


x = train_data.values[:7000, 1:]  # Use .values to get the numpy array, skip label column
y = train_data.values[:7000, 0]

some_digit = x[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="gray")
plt.axis("off")
plt.show()

y[0]
y=y.astype(np.uint8)


x_train, x_test, y_train, y_test = x[:6000],x[6000:],y[:6000],y[6000:] 


y_train_1=(y_train==1)
y_test_1=(y_test==1)
print(y_test_1)


from sklearn.linear_model import SGDClassifier


sgd_clf = SGDClassifier(random_state =42)
sgd_clf.fit(x_train,y_train_1)


sgd_clf.predict([some_digit])


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)



for train_index, test_index in skfolds.split(x_train, y_train_1):
    clone_clf = clone(sgd_clf)
    
    x_train_folds = x_train[train_index]
    y_train_folds = y_train_1[train_index]
    x_test_fold = x_train[test_index]
    y_test_fold = y_train_1[test_index]

    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred=clone_clf.predict(x_test_fold)
    n_correct=sum(y_pred == y_test_fold)
    
    print(n_correct/len(y_pred))


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,x_train,y_train_1,cv=3, scoring="accuracy")


from sklearn.base import BaseEstimator

class Never1Classifier(BaseEstimator):
    def fit(self,x,y=None):
        pass
    def predict(self,x):
        return np.zeros((len(x),1),dtype=bool)


never_1_clf=Never1Classifier()
cross_val_score(never_1_clf,x_train,y_train_1,cv=3,scoring="accuracy")


#cross_val_predict() performs K-fold cross-validation, 
#but instead of returning the evaluation scores,
# it returns the predictions made on each test fold
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_1, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_1, y_train_pred)


y_train_perfect_predictions = y_train_1
confusion_matrix(y_train_1, y_train_perfect_predictions)


from sklearn.metrics import precision_score, recall_score

precision_score(y_train_1, y_train_pred)


recall_score(y_train_1, y_train_pred)


from sklearn.metrics import f1_score
f1_score(y_train_1, y_train_pred)


y_scores = sgd_clf.decision_function([some_digit])
y_scores



threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)


threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


 y_scores = cross_val_predict(sgd_clf, x_train, y_train_1, cv=3, method="decision_function")


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_1, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    # Grid and legend
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.title("Precision and Recall vs Threshold")
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()



threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
precision_score(y_train_1, y_train_pred_90)


recall_score(y_train_1, y_train_pred_90)


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_1, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
 
plot_roc_curve(fpr, tpr)
plt.show()


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_1, cv=3,method="predict_proba")

 


y_scores_forest = y_probas_forest[:, 1]   # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_1,y_scores_forest)


plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


roc_auc_score=(y_train_1, y_scores_forest)


sgd_clf.fit(x_train, y_train)  # y_train, not y_train_5
sgd_clf.predict([some_digit])


some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores


np.argmax(some_digit_scores)


sgd_clf.classes_


sgd_clf.classes_[1]





from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(x_train, y_train)
ovo_clf.predict([some_digit])


len(ovo_clf.estimators_)


forest_clf.fit(x_train, y_train)
forest_clf.predict([some_digit])


 forest_clf.predict_proba([some_digit])


cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
cross_val_score(sgd_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")


y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


 plt.matshow(conf_mx, cmap=plt.cm.gray)
 plt.show()


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
 #Now let’s fill the diagonal with zeros to keep only the errors, and let’s plot the result:
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# Define this before calling plot_digits
def plot_digits(instances, images_per_row=10, **options):
    size = 28  # for MNIST: 28x28 images
    images_per_row = min(len(instances), images_per_row)
    n_rows = (len(instances) - 1) // images_per_row + 1
    n_empty = n_rows * images_per_row - len(instances)
    padded = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)
    image_grid = padded.reshape((n_rows, images_per_row, size, size))
    big_image = image_grid.swapaxes(1, 2).reshape((n_rows * size, images_per_row * size))
    plt.imshow(big_image, cmap='binary', **options)
    plt.axis("off")


cl_a, cl_b = 1, 5
x_aa = x_train[(y_train == cl_a) & (y_train_pred == cl_a)]
x_ab = x_train[(y_train == cl_a) & (y_train_pred == cl_b)]
x_ba = x_train[(y_train == cl_b) & (y_train_pred == cl_a)]
x_bb = x_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(x_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(x_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(x_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(x_bb[:25], images_per_row=5)
plt.show()







