import pandas as pd
import numpy as np
import keras
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from keras_nlp.models import DistilBertClassifier, DistilBertPreprocessor



MODEL_PATH = "/kaggle/input/distil_bert/keras/distil_bert_base_en/3"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
SEED = 42
NUM_CLASSES = 2


df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")



def prepare_input(row):
    return (
        f"Comment: {row['body']} "
        f"[Rule: {row['rule']}] "
        f"Positive Examples: {row['positive_example_1']} | {row['positive_example_2']} "
        f"Negative Examples: {row['negative_example_1']} | {row['negative_example_2']}"
    )

df['input_text'] = df.apply(prepare_input, axis=1)
test_df['input_text'] = test_df.apply(
    lambda row: f"Comment: {row['body']} [Rule: {row['rule']}]", axis=1
)


X_train, X_val, y_train, y_val = train_test_split(
    df['input_text'], df['rule_violation'], stratify=df['rule_violation'], test_size=0.2, random_state=SEED
)



preprocessor = DistilBertPreprocessor.from_preset(
    MODEL_PATH,
    sequence_length=MAX_LEN,
)

classifier = DistilBertClassifier.from_preset(
    MODEL_PATH,
    num_classes=NUM_CLASSES,
    preprocessor=preprocessor,
)

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    metrics=["accuracy"],
    jit_compile=True
)


classifier.fit(
    x=X_train.tolist(), y=y_train.tolist(),
    validation_data=(X_val.tolist(), y_val.tolist()),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)



y_pred_val = classifier.predict(X_val.tolist())
y_probs = tf.nn.softmax(y_pred_val).numpy()[:, 1]
y_preds = (y_probs >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_val, y_preds))

auc = roc_auc_score(y_val, y_probs)
print(f"ROC AUC: {auc:.4f}")


fpr, tpr, _ = roc_curve(y_val, y_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()


test_preds_logits = classifier.predict(test_df['input_text'].tolist())
test_probs = tf.nn.softmax(test_preds_logits).numpy()[:, 1]
submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")
submission['rule_violation'] = test_probs
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
submission.head()

