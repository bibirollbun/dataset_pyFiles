from pathlib import Path
import pandas as pd
from fastai.vision.all import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Rutas
train_dir = Path('/kaggle/input/spr-x-ray-gender/kaggle/kaggle/train/')  # 000000.png, 000001.png, ...
labels_csv = Path('/kaggle/input/spr-x-ray-gender/train_gender.csv')     # columnas: imageId, gender

# Cargar CSV y normalizar columnas
df = pd.read_csv(labels_csv)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={'imageId': 'Image', 'gender': 'Gender'})

# Detectar automáticamente el ancho de los nombres
width = max(len(p.stem) for p in train_dir.glob('*.png'))

# Normalizar los IDs: string, sin extensión, zero-padding al ancho detectado
df['Image'] = (df['Image'].astype(str)
                         .str.strip()
                         .str.replace(r'\.(png|jpg|jpeg)$', '', regex=True)
                         .str.zfill(width))

# Diccionario id -> gender (0/1)
df['Gender'] = df['Gender'].astype(int)
id2gender = df.set_index('Image')['Gender'].to_dict()

# Mapeo opcional de 0/1 a clases de texto
int2class = {0: 'mujer', 1: 'hombre'}

def label_func(o: Path):
    g = id2gender[o.stem]           # lanzará KeyError si falta en el CSV
    return int2class[int(g)]

# DataBlock
sz = 224
db = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=list(int2class.values()))),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_func,
    item_tfms=Resize(sz*2, method='pad'),
    batch_tfms=[*aug_transforms(size=sz), Normalize.from_stats(*imagenet_stats)]
)

dls = db.dataloaders(train_dir, bs = 64)
dls.show_batch(max_n=9)



# DEBUG: taking a look to the data before we load the trainer
from pathlib import Path

# Items de cada split
train_items = list(dls.train_ds.items)
valid_items = list(dls.valid_ds.items)

# Construye DataFrames con imageId y etiqueta (int y texto)
train_df_dbg = pd.DataFrame({
    'split': 'train',
    'relpath': [str(Path(o).relative_to(train_dir)) for o in train_items],
    'imageId': [Path(o).stem for o in train_items],
})
train_df_dbg['label_int'] = train_df_dbg['imageId'].map(id2gender)
train_df_dbg['label'] = train_df_dbg['label_int'].map(int2class)

valid_df_dbg = pd.DataFrame({
    'split': 'valid',
    'relpath': [str(Path(o).relative_to(train_dir)) for o in valid_items],
    'imageId': [Path(o).stem for o in valid_items],
})
valid_df_dbg['label_int'] = valid_df_dbg['imageId'].map(id2gender)
valid_df_dbg['label'] = valid_df_dbg['label_int'].map(int2class)

dbg = pd.concat([train_df_dbg, valid_df_dbg], ignore_index=True)

# Muestra una muestra de la tabla y recuentos
print("\n Preview de la tabla que entra al loader ")
print(dbg.head(20))  # cambia 20 por lo que quieras

print("\n Recuento por split y clase ")
print(dbg.groupby(['split','label']).size())

# Comprueba si hay imágenes sin etiqueta en el CSV
missing = dbg[dbg['label_int'].isna()]
if not missing.empty:
    print("\n Imágenes sin etiqueta en el CSV (primeras 10):")
    print(missing.head(10))



model_dir = Path('/') / 'kaggle' / 'working' / 'models'
model_dir


learn = vision_learner(dls,
                      resnet34,
                      metrics=error_rate,
                      loss_func=LabelSmoothingCrossEntropy(),
                      cbs=[BnFreeze,
                          SaveModelCallback(monitor='error_rate'),
                          ShowGraphCallback,
                          ],
                      model_dir=model_dir,
                      ).to_fp16()


# Scores check

preds, targs = learn.get_preds()

if preds.shape[1] == 2:
    pred_labels = preds.argmax(dim=1)
    probas = preds[:, 1]
else:
    pred_labels = preds.argmax(dim=1)
    probas = None

acc = accuracy_score(targs, pred_labels)
prec = precision_score(targs, pred_labels, average='weighted')
rec = recall_score(targs, pred_labels, average='weighted')
f1 = f1_score(targs, pred_labels, average='weighted')

auc = roc_auc_score(targs, probas) if probas is not None else None

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
if auc is not None:
    print(f"AUC-ROC:   {auc:.4f}")


@delegates(learn.fit_one_cycle)
def train(learn, name, lr, n_epochs=5, **kwargs):
    learn.fit_one_cycle(n_epochs, lr, **kwargs)
    learn.save(name)


lr = defaults.lr


train(learn, 'stage_1', lr, n_epochs=1)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# Scores check

preds, targs = learn.get_preds()

if preds.shape[1] == 2:
    pred_labels = preds.argmax(dim=1)
    probas = preds[:, 1]
else:
    pred_labels = preds.argmax(dim=1)
    probas = None

acc = accuracy_score(targs, pred_labels)
prec = precision_score(targs, pred_labels, average='weighted')
rec = recall_score(targs, pred_labels, average='weighted')
f1 = f1_score(targs, pred_labels, average='weighted')

auc = roc_auc_score(targs, probas) if probas is not None else None

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
if auc is not None:
    print(f"AUC-ROC:   {auc:.4f}")


learn.lr_find()


train(learn, 'stage_2', lr=0.0008022644514217973, n_epochs=10)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# Scores check

preds, targs = learn.get_preds()

if preds.shape[1] == 2:
    pred_labels = preds.argmax(dim=1)
    probas = preds[:, 1]
else:
    pred_labels = preds.argmax(dim=1)
    probas = None

acc = accuracy_score(targs, pred_labels)
prec = precision_score(targs, pred_labels, average='weighted')
rec = recall_score(targs, pred_labels, average='weighted')
f1 = f1_score(targs, pred_labels, average='weighted')

auc = roc_auc_score(targs, probas) if probas is not None else None

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
if auc is not None:
    print(f"AUC-ROC:   {auc:.4f}")


learn.export('/kaggle/working/models/complete_model.pkl')


learn.predict('/kaggle/input/spr-x-ray-gender/kaggle/kaggle/test/000000.png')


learn.predict('/kaggle/input/spr-x-ray-gender/kaggle/kaggle/test/000001.png')


# Archivos de test en orden determinista
test_dir = Path('/kaggle/input/spr-x-ray-gender/kaggle/kaggle/test')
test_files = get_image_files(test_dir).sorted()  # '000000.png', '000001.png', ...

# DataLoader de test con la MISMA preproc que train/valid
test_dl = dls.test_dl(test_files, bs=dls.bs)

# Probabilidades por clase
probs, _ = learn.get_preds(dl=test_dl)  # shape: [N, 2] en tu caso

# Índice de la clase positiva (como i[1] en tu ejemplo TF)
pos_idx = dls.vocab.o2i['hombre'] if hasattr(dls.vocab, 'o2i') else list(dls.vocab).index('hombre')

p_pos = probs[:, pos_idx].detach().cpu().numpy()  # prob. de 'hombre'

# imageId sin ceros a la izquierda (para coincidir con sample_submission)
image_ids = [int(p.stem) for p in test_files]  # '000123' -> 123

# DataFrame y guardado
submission = pd.DataFrame({'imageId': image_ids, 'gender': p_pos})
submission = submission.sort_values('imageId').reset_index(drop=True)
submission.to_csv('/kaggle/working/submission.csv', index=False)

print(submission.head())
print('Guardado en /kaggle/working/submission.csv  -> filas:', len(submission))

