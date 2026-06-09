import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_path = '/kaggle/input/playground-series-s5e3/train.csv'
test_path = '/kaggle/input/playground-series-s5e3/test.csv'
sample_path = '/kaggle/input/playground-series-s5e3/sample_submission.csv'


X_train = pd.read_csv(train_path)
X_test = pd.read_csv(test_path)
Sample_df = pd.read_csv(sample_path)


X_train.head()


X_test.head()


X_train.info()


X_test.info()


X_test['winddirection'].value_counts()


X_test['winddirection'] = X_test['winddirection'].fillna(70.0)



X_test.info()


X_train.describe()


correlation_matrix = X_train.corr()
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('correlation matrix')


X_train['rainfall'].value_counts()


y_train = X_train['rainfall']


del X_train['rainfall']


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)















# Score 0.73987
'''
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train_scaled, y_train)
'''


#pred = lgb_model.predict(X_test_scaled)



# Score 0.75060
'''
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(iterations=100, random_state=42)
cat_model.fit(X_train_scaled, y_train)
pred = cat_model.predict(X_test_scaled)
'''





# Score 0.78278

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, Lion, Nadam, RMSprop, SGD,AdamW

from tensorflow.keras.layers import ELU, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight

model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dense(64),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dropout(0.2),
    
    Dense(32),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dense(32),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dropout(0.2),
    Dense(16),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dense(16),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


class_weight = {0:3.0, 1:1.0}

# Modell trainieren
history = model.fit(X_train_scaled, y_train, 
                    class_weight=class_weight,
                    epochs=30, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1)



pred = model.predict(X_test_scaled)
pred_binary = (pred > 0.5).astype(int)
pred_flat = pred_binary.flatten()

output = pd.DataFrame({'id': X_test['id'],
                      'rainfall':pred_flat})


output.to_csv('submission.csv', index=False)
output.head()

'''


# Score 0.77835

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, Lion, Nadam, RMSprop, SGD,AdamW

from tensorflow.keras.layers import ELU, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight

model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dense(64),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dropout(0.2),
    
    Dense(32),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dense(32),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dropout(0.2),
    Dense(16),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dense(16),
    BatchNormalization(),
    ELU(alpha=1.0),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])


class_weight = {0:3.0, 1:1.0}


# EarlyStopping-Callback definieren
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Modell trainieren
history = model.fit(X_train_scaled, y_train, 
                    class_weight=class_weight,
                    epochs=30, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stopping])






pred = model.predict(X_test_scaled)
pred_binary = (pred > 0.5).astype(int)
pred_flat = pred_binary.flatten()

output = pd.DataFrame({'id': X_test['id'],
                      'rainfall':pred_flat})


output.to_csv('submission.csv', index=False)
output.head()
'''





from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix




'''
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}     
]

svc = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1,
                           verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
'''


# Score 0.75945
'''
param_grid = [    
    {'kernel': ['rbf'], 'C': [10,50, 100,150], 'gamma': [1.0, 0.1, 0.01, 0.05, 0.001,]}     
]

svc = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1,
                           verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
'''


'''
best_model = grid_search.best_estimator_
pred = best_model.predict(X_test_scaled)

output = pd.DataFrame({'id': X_test['id'],
                      'rainfall':pred})


output.to_csv('submission.csv', index=False)
output.head()
'''


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Annahme: X_train, y_train und X_test sind bereits definiert

# Modell erstellen
model = LinearSVC()

# Modell trainieren
model.fit(X_train_scaled, y_train)

# Vorhersagen auf dem Testdatensatz treffen
y_pred = model.predict(X_test_scaled)

# Modellleistung bewerten
#accuracy = accuracy_score(y_test, y_pred)  # Annahme: y_test ist definiert
#print("Genauigkeit:", accuracy)

output = pd.DataFrame({'id': X_test['id'],
                      'rainfall':y_pred})


output.to_csv('submission.csv', index=False)
output.head()


Sample_df


#output = pd.DataFrame({'id': X_test['id'],
                      'rainfall':pred})


#output.to_csv('submission.csv', index=False)
#output.head()




