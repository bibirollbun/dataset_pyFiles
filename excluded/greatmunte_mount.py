# ============================================================
#                 MACHINE LEARNING – REGRESSION PROJECT
#  التنبؤ بإيرادات الأفلام (TMDB Box Office Prediction Dataset)
#
#  الهدف: تطبيق دورة حياة مشروع تعلم الآلة (ML Pipeline)
#  (Frame → Collect → Prepare → Split → Model → Train → Evaluate)
# ============================================================


# ============================================================
# 1) IMPORT LIBRARIES (استدعاء المكتبات)
# ============================================================
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 2) LOAD DATA (تحميل البيانات)
# نوع المشكلة: انحدار (Regression)
# ============================================================
train_data = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")
test_data  = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/test.csv")


# ============================================================
# 3) PREPARE DATA (تنظيف وتجهيز البيانات)
# ============================================================

# دالة هندسة الميزات: استخلاص "Director" و "Genre"
def get_director(crew_str):
    try:
        crew_list = ast.literal_eval(crew_str)
        for m in crew_list:
            if m["job"] == "Director":
                return m["name"]
    except:
        return "Unknown"
    return "Unknown"

def get_first_genre(genres_str):
    try:
        g_list = ast.literal_eval(genres_str)
        if len(g_list) > 0:
            return g_list[0]["name"]
    except:
        return "Unknown"
    return "Unknown"

train_data["director"] = train_data["crew"].apply(get_director)
test_data["director"]  = test_data["crew"].apply(get_director)
train_data["genre_main"] = train_data["genres"].apply(get_first_genre)
test_data["genre_main"]  = test_data["genres"].apply(get_first_genre)

# تنظيف البيانات: معالجة القيم المفقودة (budget) باستخدام الوسيط (Median)
train_data["budget"] = pd.to_numeric(train_data["budget"], errors="coerce")
train_data["budget"] = train_data["budget"].replace(0, np.nan)
train_data["budget"] = train_data["budget"].fillna(train_data["budget"].median())

test_data["budget"] = pd.to_numeric(test_data["budget"], errors="coerce")
test_data["budget"] = test_data["budget"].replace(0, np.nan)
test_data["budget"] = test_data["budget"].fillna(train_data["budget"].median())


# ============================================================
# 4) SELECT FEATURES & TARGET
# ============================================================
features = ["budget", "popularity", "director", "genre_main"]
target = "revenue"
X = train_data[features]
y = train_data[target]
X_test_final = test_data[features]


# ============================================================
# 5) ENCODING & SCALING (المعالجة المسبقة)
# ============================================================
# الترميز (One-Hot Encoding)
combined = pd.concat([X, X_test_final], axis=0)
combined_encoded = pd.get_dummies(combined, columns=["director", "genre_main"], drop_first=True)
X_encoded = combined_encoded.iloc[:len(X), :]
X_test_encoded = combined_encoded.iloc[len(X):, :]

# توحيد القيم (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


# ============================================================
# 6) SPLIT DATA (تقسيم البيانات)
# التقسيم الثلاثي: 60% تدريب - 20% تحقق - 20% اختبار
# ============================================================
# فصل 20% كاختبار داخلي
X_temp, X_test_internal, y_temp, y_test_internal = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# فصل Train و Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)


# ============================================================
# 7 & 8) BUILD AND TRAIN MODELS (بناء وتدريب النماذج)
# ============================================================
lr = LinearRegression()
dt = DecisionTreeRegressor(max_depth=8, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)


# ============================================================
# 9) VALIDATION EVALUATION (تقييم الأداء)
# تم تنسيق الأرقام لعرضها بالمليون (M$) لتسهيل الفهم
# ============================================================
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # تنسيق المخرجات ليتم عرضها بوحدة M$
    mae_m = f"{(mae / 1000000):.2f} M$"
    rmse_m = f"{(rmse / 1000000):.2f} M$"

    print(f"\n--- {name} | Validation Evaluation ---")
    print(f"MAE : {mae_m}")
    print(f"RMSE: {rmse_m}") # هذا هو المقياس الأهم والأبسط للعرض

# Linear Regression Validation Results
evaluate("Linear Regression", y_val, lr.predict(X_val))

# Decision Tree Validation Results
evaluate("Decision Tree", y_val, dt.predict(X_val))


# ============================================================
# 10) FINAL PREDICTIONS (عرض النتائج)
# ============================================================
# استخدام النموذج الأفضل (Decision Tree)
test_predictions_dt = dt.predict(X_test_scaled)

print("\n=== Sample Predictions (3 Movies from MIDDLE) — Decision Tree Model ===\n")

# دالة لتنسيق الأرقام إلى مليون دولار (M$) لتبسيط العرض
def format_to_millions(num_series):
    return (num_series / 1000000).round(2).astype(str) + " M$"

# نختار 3 أفلام تبدأ من منتصف مجموعة البيانات (الفهرس 2000)
start_index = 2000
end_index = start_index + 3

# اختيار العينة وتضمين التنبؤات المقابلة
sample_movies = test_data.iloc[start_index:end_index].copy()
sample_movies = sample_movies[["id", "original_title", "budget", "popularity"]]
sample_movies["predicted_revenue"] = test_predictions_dt[start_index:end_index]

# تطبيق التنسيق على الميزانية والإيرادات المتوقعة
sample_movies["budget"] = format_to_millions(sample_movies["budget"])
sample_movies["predicted_revenue"] = format_to_millions(sample_movies["predicted_revenue"])

print(sample_movies)


