!pip install torch


import torch
import tensorflow as tf
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
device_name = tf.config.list_physical_devices('GPU')


!pip install keras-core --upgrade
!pip install -q keras-nlp
!pip install seaborn


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras_core as keras
import keras_nlp
import seaborn as sns
import matplotlib.pyplot as plt


print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("KerasNLP version:", keras_nlp.__version__)


DATA_DIR = '/kaggle/input/llm-detect-ai-generated-text/'

for dirname, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_essays.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_essays.csv'))

# Quick overview
print(train_df.head())
print(train_df.info())
print(train_df['generated'].value_counts())



df_train_prompts = pd.read_csv(DATA_DIR + "train_prompts.csv")
print(df_train_prompts.info())
df_train_prompts.head()


df_train_essays = pd.read_csv(DATA_DIR + "train_essays.csv")
print(df_train_essays.info())
df_train_essays.head()


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays,
                   x="prompt_id")

abs_values = df_train_essays['prompt_id'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of prompt ID")


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays,
                   x="generated")

abs_values = df_train_essays['generated'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of Generated Text")


df_test_essays = pd.read_csv(DATA_DIR + "test_essays.csv")
print(df_test_essays.info())
df_test_essays.head()


df_test_essays["text"].apply(lambda x : len(x))


df_train_essays_ext = pd.read_csv('/kaggle/input/daigt-proper-train-dataset/train_drcat_04.csv')

df_train_essays_ext.rename(columns = {"label":"generated"}, inplace=True)

df_train_essays_ext.info()


df_train_essays_ext.head()


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays_ext,
                   x="generated")

abs_values = df_train_essays_ext['generated'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of Generated Text")


df_train_essays



tf.test.gpu_device_name()
import tensorflow as tf

# List available GPUs
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))



df_train_essays_final = pd.concat([df_train_essays_ext[["text", "generated"]], df_train_essays[["text", "generated"]]])

df_train_essays_final.info()


df_train_essays["text_length"] = df_train_essays["text"].apply(lambda x : len(x.split()))


fig = plt.figure(figsize=(40,50))
plot = sns.displot(data=df_train_essays,
                 x="text_length", bins=30, kde=True)
plot.fig.suptitle("Distribution of the length per essay - Train dataset")



df_train_essays["text_length"].mean() + df_train_essays["text_length"].std()


# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_essays_final["text"],
                                                    df_train_essays_final["generated"],
                                                    test_size=0.33,
                                                    random_state=42)


import keras_nlp
import keras
import matplotlib.pyplot as plt
from keras_core.optimizers import Adam

# Constants
SEQ_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 20

# Load preprocessor
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=SEQ_LENGTH,
)

# Load classifier
classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
    activation=None,
    preprocessor=preprocessor,
)

# Unfreeze the DistilBERT backbone
classifier.backbone.trainable = True

# Compile with lower learning rate for fine-tuning
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(1e-5),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)


# Fit the model
history = classifier.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)



def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        np.argmax(y_pred, axis=1),
        display_labels=["Not Generated","Generated"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
    f1_score = tp / (tp+((fn+fp)/2))

    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))



y_pred_test = classifier.predict(X_test)


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
displayConfusionMatrix(y_test, y_pred_test,  "Test")


print(history.history.keys())


import matplotlib.pyplot as plt



import matplotlib.pyplot as plt

def plot_metrics(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axs[0].plot(history.history['mean_metric_wrapper'], label='Train Accuracy')
    axs[0].plot(history.history['val_mean_metric_wrapper'], label='Val Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    # Loss
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Val Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Call the function to show plot
plot_metrics(history)



import tensorflow as tf


import pandas as pd


classifier.summary()



print(classifier.input)



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get the predicted probabilities for the positive class (second class)
y_scores = classifier.predict(X_test)  # Get probabilities from the model

# If the model outputs probabilities, `y_scores` will be a 2D array (for multi-class)
# For binary classification, take the probability of the positive class
y_scores = y_scores[:, 1]  # Assuming the second column is the probability for class 1

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get precision and recall values
precision, recall, _ = precision_recall_curve(y_test, y_scores)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.fill_between(recall, precision, alpha=0.2, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()



# Histogram of predicted probabilities (if binary classification)
plt.figure(figsize=(8, 6))
plt.hist(y_scores, bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()



# Compare true vs predicted labels
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=2, alpha=0.5, label='True Labels')
plt.hist(y_pred_test, bins=2, alpha=0.5, label='Predicted Labels')
plt.legend(loc='upper right')
plt.title('True vs Predicted Labels')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()



# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import time


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Step 3: Initialize models
model1 = LogisticRegression(max_iter=1000)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model3 = SVC(probability=True, random_state=42)



start = time.time()
model1.fit(X_train, y_train)
training_time_model1 = time.time() - start


start = time.time()
model2.fit(X_train, y_train)
training_time_model2 = time.time() - start

start = time.time()
model3.fit(X_train, y_train)
training_time_model3 = time.time() - start

# Step 5: Get predictions and probabilities for ROC curve
y_pred_model1 = model1.predict(X_test)
y_pred_model2 = model2.predict(X_test)
y_pred_model3 = model3.predict(X_test)


y_scores_model1 = model1.predict_proba(X_test)[:, 1]
y_scores_model2 = model2.predict_proba(X_test)[:, 1]
y_scores_model3 = model3.predict_proba(X_test)[:, 1]

# Step 6: Calculate performance metrics
def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }



metrics_model1 = get_metrics(y_test, y_pred_model1)
metrics_model2 = get_metrics(y_test, y_pred_model2)
metrics_model3 = get_metrics(y_test, y_pred_model3)

# Step 7: Plot the ROC curve
fpr1, tpr1, _ = roc_curve(y_test, y_scores_model1)
fpr2, tpr2, _ = roc_curve(y_test, y_scores_model2)
fpr3, tpr3, _ = roc_curve(y_test, y_scores_model3)

roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)



# Step 8: Plot the Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred_model1)
cm2 = confusion_matrix(y_test, y_pred_model2)
cm3 = confusion_matrix(y_test, y_pred_model3)



# Step 9: Model Comparison (Metrics Table)
model_comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
    'Accuracy': [metrics_model1['accuracy'], metrics_model2['accuracy'], metrics_model3['accuracy']],
    'Precision': [metrics_model1['precision'], metrics_model2['precision'], metrics_model3['precision']],
    'Recall': [metrics_model1['recall'], metrics_model2['recall'], metrics_model3['recall']],
    'F1 Score': [metrics_model1['f1'], metrics_model2['f1'], metrics_model3['f1']],
    'Training Time (s)': [training_time_model1, training_time_model2, training_time_model3]
})

print(model_comparison_df)



# Step 10: Plot the ROC curve for all models
plt.figure(figsize=(10, 6))
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc2:.2f})')
plt.plot(fpr3, tpr3, color='green', lw=2, label=f'SVM (AUC = {roc_auc3:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Comparison')
plt.legend(loc='lower right')
plt.show()



# Step 11: Plot the confusion matrices for all models
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
ax[0].set_title("Logistic Regression Confusion Matrix")

sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax[1], cbar=False)
ax[1].set_title("Random Forest Confusion Matrix")

sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues', ax=ax[2], cbar=False)
ax[2].set_title("SVM Confusion Matrix")

plt.tight_layout()
plt.show()


# Step 12: Plot the Model Comparison Bar Plot
import matplotlib.pyplot as plt
import numpy as np

model_names = ['Logistic Regression', 'Random Forest', 'SVM']
accuracies = [metrics_model1['accuracy'], metrics_model2['accuracy'], metrics_model3['accuracy']]
precisions = [metrics_model1['precision'], metrics_model2['precision'], metrics_model3['precision']]
recalls = [metrics_model1['recall'], metrics_model2['recall'], metrics_model3['recall']]
f1_scores = [metrics_model1['f1'], metrics_model2['f1'], metrics_model3['f1']]

x = np.arange(len(model_names))
width = 0.2

# Define pastel or muted pinkish, greyish, and bluish colors
accuracy_color = '#D8A7A1'  # Muted pinkish grey
precision_color = '#A1B6D8'  # Soft blue-grey
recall_color = '#D8C1A1'     # Light beige-grey
f1_color = '#B5A1D8'         # Soft purple-grey

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars with the soft muted colors
ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color=accuracy_color)
ax.bar(x - 0.5*width, precisions, width, label='Precision', color=precision_color)
ax.bar(x + 0.5*width, recalls, width, label='Recall', color=recall_color)
ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color=f1_color)

# Set labels, title, and x-ticks
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison Metrics')
ax.set_xticks(x)
ax.set_xticklabels(model_names)

# Add the legend
ax.legend()

# Adjust layout for a clean view
plt.tight_layout()

# Show the plot
plt.show()





