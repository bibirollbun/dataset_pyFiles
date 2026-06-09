import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb



train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


encode = LabelEncoder()



cat_cols = train_df.select_dtypes(include=['object']).columns

for col in cat_cols:
    train_df[col] = encode.fit_transform(train_df[col])
    test_df[col] = encode.transform(test_df[col])


train_df.head(10)


x = train_df.drop(['diagnosed_diabetes', 'id'], axis=1)
y = train_df['diagnosed_diabetes']



x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=2,test_size=0.2)


from sklearn.ensemble import GradientBoostingClassifier

# Improved XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    random_state=2
)
model.fit(x_train,y_train)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=2,
    n_jobs=-1
)
rf_model.fit(x_train,y_train)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    random_state=2
)
gb_model.fit(x_train,y_train)



prediction = model.predict(x_test)
train_df_accuracy_score = accuracy_score(y_test,prediction)


print(train_df_accuracy_score)


# Compare all models on test set
xgb_pred = model.predict(x_test)
rf_pred = rf_model.predict(x_test)
gb_pred = gb_model.predict(x_test)

xgb_acc = accuracy_score(y_test, xgb_pred)
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)

print("=== Model Comparison ===")
print(f"Improved XGBoost: {xgb_acc:.4f}")
print(f"Random Forest: {rf_acc:.4f}")
print(f"Gradient Boosting: {gb_acc:.4f}")

# Select best model
models_scores = [('XGBoost', model, xgb_acc), ('RandomForest', rf_model, rf_acc), ('GradientBoosting', gb_model, gb_acc)]
best_name, best_model_final, best_acc = max(models_scores, key=lambda x: x[2])
print(f"\nBest Model: {best_name} with accuracy {best_acc:.4f}")

# Make predictions with best model
X_test_submission = test_df.drop(['id'], axis=1)
test_predictions = best_model_final.predict(X_test_submission)

submission_df['diagnosed_diabetes'] = test_predictions
submission_df.to_csv('submission_improved.csv', index=False)

print(f"Submission file saved as 'submission_improved.csv' using {best_name}")


