import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error #, mean_absolute_error, r2_score

TRAIN_PATH = "/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = "/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"


df = pd.read_csv(TRAIN_PATH)
df_unk = pd.read_csv(TEST_PATH)


df = df.dropna(subset=['CORRUCYSTIC_DENSITY'])


num_cols = df.select_dtypes(include=[np.number]).columns.drop('CORRUCYSTIC_DENSITY').tolist()


imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[num_cols])
y = df['CORRUCYSTIC_DENSITY'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)


print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"R²: {r2:.4f}")



X_unk = imputer.transform(df_unk[num_cols])
y_unk_pred = model.predict(X_unk)



submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': df_unk['LOCAL_IDENTIFIER'].astype(int),
    'CORRUCYSTIC_DENSITY': y_unk_pred.astype(float)
})

submission.to_csv('submission.csv', index=False)
#print("Submission shape:", submission.shape)
print(submission.head())


