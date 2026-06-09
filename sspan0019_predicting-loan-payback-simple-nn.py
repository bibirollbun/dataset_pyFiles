import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')


train.info()


test.info()


TARGET = 'loan_paid_back'
NUM_FEATURES = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
CAT_FEATURES = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']


train.duplicated().sum()


test.duplicated().sum()


train.isna().sum()


test.isna().sum()


import matplotlib.pyplot as plt

for feature in CAT_FEATURES:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    train[feature].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], title=feature)
    train.groupby(feature)[TARGET].mean().plot.bar(ax=ax[1], title='Average')
    plt.show()


import scipy.stats as stats
import seaborn as sns

for col in NUM_FEATURES:
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.hist([train[train[TARGET] == 0][col], train[train[TARGET] == 1][col]], 
             bins=20, stacked=True, label=['Not Paid Back', 'Paid Back'], 
             color=['red', 'green'])
    plt.axvline(train[train[TARGET] == 0][col].mean(), color='darkred', linestyle='--', linewidth=2, label='Avg Not Paid Back')
    plt.axvline(train[train[TARGET] == 1][col].mean(), color='darkgreen', linestyle='--', linewidth=2, label='Avg Paid Back')
    plt.title(f"Histogram of {col}")
    plt.legend()

    plt.subplot(1, 3, 2)
    stats.probplot(train[col], dist="norm", plot=plt)
    plt.title(f"QQ plot of {col}")

    plt.subplot(1, 3, 3)
    sns.boxenplot(x=train[col])
    plt.title(f"Boxen plot of {col}")

    plt.tight_layout()
    plt.show()


train['grade'] = ''
test['grade'] = ''

for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
    train.loc[train['grade_subgrade'].str.startswith(grade), 'grade'] = grade
    test.loc[test['grade_subgrade'].str.startswith(grade), 'grade'] = grade

train = train.drop(columns=['grade_subgrade'])
test = test.drop(columns=['grade_subgrade'])

CAT_FEATURES.append('grade')
CAT_FEATURES.remove('grade_subgrade')


train = pd.get_dummies(train, columns=CAT_FEATURES, drop_first=True)
test = pd.get_dummies(test, columns=CAT_FEATURES, drop_first=True)


train = train.astype({col: int for col in train.select_dtypes(include='bool').columns})
test = test.astype({col: int for col in test.select_dtypes(include='bool').columns})


import numpy as np

for feature in NUM_FEATURES:
    train[feature] = np.log1p(train[feature])
    test[feature] = np.log1p(test[feature])


train.head()


test.head()


from sklearn.model_selection import train_test_split

X = train.drop(columns=[TARGET])
y = train[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"\nValidation target distribution:\n{y_val.value_counts(normalize=True)}")


import tensorflow as tf
from tensorflow import keras

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        
        cross_entropy_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
        weight_neg = (1 - alpha) * (1 - y_true) * tf.math.pow(y_pred, gamma)
        
        loss = weight * cross_entropy + weight_neg * cross_entropy_neg
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val), 
    epochs=100, 
    batch_size=256,
    callbacks=[reduce_lr, early_stopping])


y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)


from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Paid Back', 'Paid Back'],
            yticklabels=['Not Paid Back', 'Paid Back'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


y_test_pred_prob = model.predict(test)

submission = pd.DataFrame({
    'id': test.index,
    'loan_paid_back': y_test_pred_prob.flatten()
})

submission.to_csv('submission.csv', index=False)

