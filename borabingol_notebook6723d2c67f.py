import pandas as pd

# 1. CSV dosyasını okuma
input_path = "/kaggle/input/unpretrained-plz-dont-give-any-error-messages/submission.csv"  # Girdi dosyanızın tam yolunu buraya yazın
df = pd.read_csv(input_path)

# 2. Belirli sütunları float32 olarak ayarlama
float_columns = ["normal_mild", "moderate", "severe"]  # Dönüştürülecek sütunlar
df[float_columns] = df[float_columns].astype("float32")

# 3. Verileri submission.csv olarak kaydetme
df.to_csv("submission.csv", index=False)

print("Veriler 'submission.csv' adıyla başarıyla kaydedildi. Belirli sütunlar float32'ye dönüştürüldü.")


