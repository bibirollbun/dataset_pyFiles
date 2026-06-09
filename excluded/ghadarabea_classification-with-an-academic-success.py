import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from tensorflow.keras.metrics import Precision, Recall, F1Score


df = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.info()


df.describe()


df.drop(columns=['id'], inplace=True)


df.isna().sum()


df.duplicated().sum()


encoder = LabelEncoder()

df['Target'] = encoder.fit_transform(df['Target'])


df['Target'].value_counts()


df.corr()


plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Example")
plt.show()


######################## positive #########################
# -------------------------------------------------------------->>>>> Curricular units 2nd sem (approved): الوحدات الدراسية للفصل الثاني (معتمدة بعد التقييم)
# -------------------------------------------------------------->>>>> Curricular units 2nd sem (grade): درجة الوحدات الدراسية للفصل الثاني
# -------------------------------------------------------------->>>>> Curricular units 1st sem (approved): الوحدات الدراسية للفصل الأول (معتمدة بعد التقييم)
# -------------------------------------------------------------->>>>> Curricular units 1st sem (grade): درجة الوحدات الدراسية للفصل الأول

# -------------------------------------------------->>>>> Tuition fees up to date: الرسوم الدراسية مدفوعة حتى الآن
# -------------------------------------------------->>>>> Scholarship holder: حاصل على منحة دراسية

# ------------------------------------>>>>> Curricular units 2nd sem (enrolled): الوحدات الدراسية للفصل الثاني (مسجلة)
# ------------------------------------>>>>> Curricular units 1st sem (enrolled): الوحدات الدراسية للفصل الأول (مسجلة)

# ------------------------>>>>> Curricular units 2nd sem (evaluations): الوحدات الدراسية للفصل الثاني (خضعت للتقييم)
# ------------------------>>>>> Curricular units 1st sem (evaluations): الوحدات الدراسية للفصل الأول (خضعت للتقييم)

# -------->> Admission grade: درجة القبول
# -------->> Displaced: نازح
# -------->> Course: البرنامج الدراسي / التخصص
# -------->> Previous qualification (grade): درجة المؤهل السابق

# ---->> Application order: ترتيب التقديم
# ---->> Daytime/evening attendance: الحضور (صباحي/مسائي)

# --> GDP: الناتج المحلي الإجمالي

######################### negative ###############################
# ------------------------>>>>> Application mode: وضع التقديم
# ------------------------>>>>> Age at enrollment: العمر عند التسجيل
# ------------------------>>>>> Debtor: مدين




########### low_effect #####################
# Curricular units 1st sem (credited): الوحدات الدراسية للفصل الأول (معتمدة)
# Curricular units 1st sem (without evaluations): الوحدات الدراسية للفصل الأول (لم تخضع للتقييم)

# Curricular units 2nd sem (credited): الوحدات الدراسية للفصل الثاني (معتمدة)
# Curricular units 2nd sem (without evaluations): الوحدات الدراسية للفصل الثاني (لم تخضع للتقييم)


################ Drop ###############
# International: دولي
# Unemployment rate: معدل البطالة
# Inflation rate: معدل التضخم
# Nationality: الجنسية
# Mother's qualification: مؤهل الأم
# Father's qualification: مؤهل الأب
# Mother's occupation: مهنة الأم
# Father's occupation: مهنة الأب
# Educational special needs: احتياجات تعليمية خاصة
# Gender: الجنس
# Marital status: الحالة الاجتماعية
# Previous qualification: المؤهل السابق


df = df.drop(columns=["Nacionality", "Father's qualification", "Mother's occupation", "Father's occupation"
                     , "Educational special needs","International", "Unemployment rate", "Inflation rate"
                     , "Gender", "Curricular units 1st sem (credited)", "Curricular units 2nd sem (credited)"])


plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Example")
plt.show()


X = df.drop(columns=['Target'])
y = df['Target']


encoder = LabelEncoder()

y = encoder.fit_transform(y)


encoder = OneHotEncoder(sparse_output=False)

y = encoder.fit_transform(y.reshape(-1, 1))


X


y


for i in X.columns:
    print(f'{i} => {df[i].nunique()}')
    print(30 * '-')


scaler = StandardScaler()

X = scaler.fit_transform(X)


X


X_train, X_dummy, y_train, y_dummy = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_dummy, y_dummy, test_size=0.5, random_state=42, stratify=y_dummy)


X_train.shape


model = Sequential([
    Dense(1024, activation='relu', input_dim=(X_train.shape[1])),
    # BatchNormalization(),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', 'precision'])
model.summary()


modelcheckpoints = ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False)
earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
logger = CSVLogger('model.csv')
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)


hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=32, callbacks=[modelcheckpoints, reducelr, earlystopping, logger])


model.save('model2.h5')


from tensorflow.keras.models import load_model


model2 = load_model('/kaggle/working/model.keras')


model2.evaluate(X_valid, y_valid)


model.evaluate(X_valid, y_valid)


model3 = load_model('/kaggle/working/model2.h5')


model3.evaluate(X_valid, y_valid)


val_acc = hist.history['val_accuracy']
val_acc


epochs = [i+1 for i in range(len(val_acc))]
epochs



tr_loss = hist.history['loss']
val_loss = hist.history['val_loss']
tr_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = [i+1 for i in range(len(tr_loss))]

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, color='green', label='Train Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, color='green', label='Train Accuracy')
plt.plot(epochs, val_acc, color='red', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


X = df.drop(columns=['Target'])
y = df['Target']


X


encoder = LabelEncoder()

y = encoder.fit_transform(y)


y


scaler = StandardScaler()

X = scaler.fit_transform(X)


X


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


rf = RandomForestClassifier(n_estimators=100, max_depth=9)

rf.fit(X_train, y_train)


print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))


xgb = XGBClassifier(n_estimators=25)

xgb.fit(X_train, y_train)


print(xgb.score(X_train, y_train))
print(xgb.score(X_test, y_test))


lgbm = LGBMClassifier(n_estimators=200)

lgbm.fit(X_train, y_train)


print(lgbm.score(X_train, y_train))
print(lgbm.score(X_test, y_test))


test_df = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
test_df


test_df = test_df.drop(columns=["Nacionality", "Father's qualification", "Mother's occupation", "Father's occupation"
                     , "Educational special needs","International", "Unemployment rate", "Inflation rate"
                     , "Gender", "Curricular units 1st sem (credited)", "Curricular units 2nd sem (credited)"])


prep_data = scaler.transform(test_df.iloc[:, 1:])


pred = lgbm.predict(prep_data)


pred


pred = encoder.inverse_transform(pred)


pred


sabmission = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')
sabmission.head()


sabmission['Target'] = pred


sabmission.to_csv('submission.csv', index=False)

