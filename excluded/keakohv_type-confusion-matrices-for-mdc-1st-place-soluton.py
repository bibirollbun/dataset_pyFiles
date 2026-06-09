import pandas as pd

df_doi_oof = pd.read_csv('/kaggle/input/mdc-1st-place-solution-catboost-and-qwen/catboost_doi_type_classifier/oof_predictions.csv')


df_doi_oof[['true_label','pred_label']]


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels = sorted(df_doi_oof['true_label'].unique())

cm = confusion_matrix(df_doi_oof['true_label'], df_doi_oof['pred_label'], labels=labels)

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for DOI using 6-fold CV OOF predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


labels = sorted(df_doi_oof['true_label'].unique())

# Compute confusion matrix with normalization
cm = confusion_matrix(
    df_doi_oof['true_label'],
    df_doi_oof['pred_label'],
    labels=labels,
    normalize='true'   # Normalize by true labels (rows)
)

# Convert fractions to percentages
cm_percent = cm * 100

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
disp.plot(cmap='Blues', values_format=".1f")  # Show 1 decimal precision for %
plt.title("Confusion Matrix for DOI using 6-fold CV OOF predictions (Percentages)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


df_accession_preds = pd.read_csv('/kaggle/input/mdc-1st-place-solution-catboost-and-qwen/submission_acc.csv')


df_accession_preds


df_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')


df_labels = df_labels[df_labels.type != "Missing"].copy()
df_labels = df_labels[~df_labels.dataset_id.str.contains("doi.org")]


df_labels


df_accession_preds


df_accession_preds_limited = df_accession_preds[df_accession_preds.dataset_id.isin(df_labels.dataset_id.unique())]


df_accession_preds_limited = df_accession_preds_limited.rename(columns={'type':'type_pred'})


df_acc_ids = df_accession_preds_limited.merge(df_labels, on=['article_id','dataset_id'])


df_acc_ids


labels = sorted(df_acc_ids['type'].unique())

cm = confusion_matrix(df_acc_ids['type'], df_acc_ids['type_pred'], labels=labels)

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for ACCESSION IDS using\n predicted ACCESSION IDS that appear in training set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


labels = sorted(df_acc_ids['type'].unique())

cm = confusion_matrix(df_acc_ids['type'], df_acc_ids['type_pred'], labels=labels, normalize='true')
cm_percent = cm * 100

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
disp.plot(cmap='Blues', values_format=".1f")
plt.title(f"Confusion Matrix for ACCESSION IDS using\n predicted ids that appear in training set (Percentages)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

