# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head(-10)


test.head(-10)


def preprocess_data(train, test):
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    cat_cols.remove('Fertilizer Name')
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        le_dict[col] = le
    
    le_target = LabelEncoder()
    train['Fertilizer Name'] = le_target.fit_transform(train['Fertilizer Name'])
    class_names = le_target.classes_
    
    # Split features and target
    X = train.drop(['id', 'Fertilizer Name'], axis=1)
    y = train['Fertilizer Name']
    test_ids = test['id']
    test = test.drop('id', axis=1)
    
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=['number']).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])
    
    return X, y, test, test_ids, class_names, le_target

X, y, test_data, test_ids, class_names, le_target = preprocess_data(train, test)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(
    objective='multi:softprob',
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='gpu_hist' 
)


xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=10
)



plt.figure(figsize=(12, 8))
plot_importance(xgb_model, max_num_features=20)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png')
plt.show()


plt.figure(figsize=(20, 10))
plot_tree(xgb_model, num_trees=0, rankdir='LR')
plt.title('XGBoost Decision Tree Visualization')
plt.tight_layout()
plt.savefig('xgboost_decision_tree.png')
plt.show()


def create_nn_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


y_train_nn = y_train.copy()
y_val_nn = y_val.copy()

nn_model = create_nn_model(X_train.shape[1], len(class_names))


callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5)
]


history = nn_model.fit(
    X_train, y_train_nn,
    validation_data=(X_val, y_val_nn),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)



plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('nn_training_history.png')
plt.show()


def ensemble_predictions(models, X, top_k=3):
    # Get probabilities from each model
    xgb_probs = models[0].predict_proba(X)
    nn_probs = models[1].predict(X)

    combined_probs = 0.6 * xgb_probs + 0.4 * nn_probs
    
    # Get top k predictions
    top_k_preds = np.argsort(combined_probs, axis=1)[:, -top_k:][:, ::-1]
    return top_k_preds

val_preds = ensemble_predictions([xgb_model, nn_model], X_val)
val_true_labels = le_target.inverse_transform(y_val)
val_pred_labels = [le_target.inverse_transform(pred) for pred in val_preds]

# MAP@3
def mapk(actual, predicted, k=3):
    return np.mean([1 if a in p[:k] else 0 for a, p in zip(actual, predicted)])

val_score = mapk(val_true_labels, val_pred_labels)
print(f"Validation MAP@3: {val_score:.4f}")

# Predict on test set
test_preds = ensemble_predictions([xgb_model, nn_model], test_data)
test_pred_labels = [' '.join(le_target.inverse_transform(pred)) for pred in test_preds]# Create submission


submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': test_pred_labels
})

submission.to_csv('submission_ensemble.csv', index=False)

