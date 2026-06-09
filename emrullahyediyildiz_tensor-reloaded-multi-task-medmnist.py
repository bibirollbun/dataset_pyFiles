# LÃ¤dt mehrere .npz-Dateien (train/val/test) in ein Dictionary
# ============================================================
from pathlib import Path
import numpy as np

# 1) Welche Teil-DatensÃ¤tze sollen geladen werden?
TASKS = [
    "bloodmnist","breastmnist","dermamnist","octmnist",
    "organamnist","organcmnist","organsmnist","pathmnist",
    "pneumoniamnist","retinamnist","tissuemnist",
]

# 2) Ãœbliche Suchpfade (Kaggle & lokal)
ROOTS = [Path("/kaggle/input"), Path("/kaggle/working"), Path("data"), Path(".")]

def find_npz(name: str):
    """Sucht {name}.npz in gÃ¤ngigen Ordnern und gibt ersten Treffer zurÃ¼ck."""
    p = Path("data")/f"{name}.npz"
    if p.exists(): return p
    for r in ROOTS:
        hits = list(r.rglob(f"{name}.npz"))
        if hits:
            return sorted(hits, key=lambda x: len(str(x)))[0]
    return None

def load_npz(path: Path):
    """Liest train/val/test aus .npz und liefert einheitliche Keys."""
    z = np.load(path)
    g = lambda k: z[k] if k in z.files else None
    xtr, ytr = g("train_images"), g("train_labels")
    xva, yva = g("val_images"),   g("val_labels")
    xte, yte = g("test_images"),  g("test_labels")
    squeeze = lambda a: (a.squeeze() if (a is not None and a.ndim > 1) else a)
    ytr, yva, yte = map(squeeze, (ytr, yva, yte))
    n_classes = int(np.unique(ytr).size) if ytr is not None else None
    shapes = {k:(v.shape if v is not None else None)
              for k,v in {"x_train":xtr,"x_val":xva,"x_test":xte}.items()}
    return dict(x_train=xtr,y_train=ytr,x_val=xva,y_val=yva,x_test=xte,y_test=yte,
                n_classes=n_classes, shapes=shapes)

# 3) Alle gewÃ¼nschten Tasks laden
data_dict, missing = {}, []
for t in TASKS:
    p = find_npz(t)
    if p: data_dict[t] = load_npz(p)
    else: missing.append(t)

# 4) Kurzreport
print("Geladen:")
for t, d in data_dict.items():
    s = d["shapes"]
    print(f"{t:12s} | train={s['x_train']} val={s['x_val']} test={s['x_test']} | classes={d['n_classes']}")
if missing:
    print("\nFehlt (Datei nicht gefunden):", ", ".join(missing))


X = data_dict["pathmnist"]["x_train"]
y = data_dict["pathmnist"]["y_train"]
print("shape:", X.shape, "dtype:", X.dtype, "min/max:", X.min(), X.max())



import matplotlib.pyplot as plt
plt.imshow(X[0])      # PathMNIST ist RGB
plt.axis("off"); plt.show()



# ðŸ‘€ Schnell-Check: Bilder + Klassenverteilung (kurz & deutsch)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt

def inspect(task: str, n: int = 6, shuffle: bool = True):
    X, y = data_dict[task]["x_train"], data_dict[task]["y_train"]
    idx = np.arange(len(X))
    if shuffle: np.random.shuffle(idx)
    idx = idx[:min(n, len(X))]

    # --- Beispielbilder ---
    cols, rows = min(n, 6), 1
    plt.figure(figsize=(cols*2.1, 2.2))
    for i, k in enumerate(idx, 1):
        ax = plt.subplot(rows, cols, i)
        img = X[k]
        if img.ndim == 3 and img.shape[-1] == 3: ax.imshow(img)              # RGB
        else:                                   ax.imshow(img.squeeze(), cmap="gray")  # Grau
        ax.set_title(f"y={int(y[k])}", fontsize=9); ax.axis("off")
    plt.suptitle(f"{task} â€“ Beispielbilder", y=1.05)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

    # --- Klassenverteilung ---
    vals, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(4.2, 3.2))
    plt.bar(vals.astype(int), counts)
    plt.xlabel("Klasse"); plt.ylabel("Anzahl"); plt.title(f"Klassenverteilung â€“ {task}")
    plt.tight_layout(); plt.show()





 inspect("breastmnist", n=6)


inspect("pathmnist", n=6)


# === Keras Mini-CNN fÃ¼r einen gewÃ¤hlten Task ===
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

def train_task(task: str, epochs: int = 8, batch: int = 256):
    d = data_dict[task]
    Xtr, ytr = d["x_train"], d["y_train"].astype("int32")
    Xva, yva = d["x_val"], d["y_val"]
    n_classes = d["n_classes"]

    # Datentyp/Skalierung
    Xtr = Xtr.astype("float32")/255.0
    if Xva is None:  # falls kein val-Split vorhanden â†’ Tail als Val nehmen
        Xtr, Xva = Xtr[:-2000], Xtr[-2000:]
        ytr, yva = ytr[:-2000], ytr[-2000:]
    else:
        Xva = Xva.astype("float32")/255.0

    # Eingabe-KanÃ¤le bestimmen (1 = grau, 3 = RGB)
    ch = 1 if Xtr.shape[-1] == 1 else 3

    # Modell: klein, robust
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, ch)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    )

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs, batch_size=batch, verbose=1,
        callbacks=[cb]
    )

    model.save(f"{task}_minicnn.keras")
    print(f"ðŸ’¾ gespeichert: {task}_minicnn.keras")
    return model, hist

def evaluate_task(model, task: str):
    d = data_dict[task]
    Xte, yte = d["x_test"], d["y_test"].astype("int32")
    Xte = Xte.astype("float32")/255.0

    # Vorhersage + Report
    pred = model.predict(Xte, batch_size=512).argmax(1)
    print(classification_report(yte, pred, digits=3))

    # Confusion Matrix (klein & simpel)
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(4,3))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"CM â€“ {task}"); plt.xlabel("pred"); plt.ylabel("true")
    plt.colorbar(); plt.tight_layout(); plt.show()



inspect("pathmnist", n=6)


model, hist = train_task("pathmnist", epochs=8, batch=256)


evaluate_task(model, "pathmnist")


# Listet alle .npz, die Kaggle aktuell findet
from pathlib import Path
roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path("data"), Path(".")]
npz_files = sorted({str(p) for r in roots for p in r.rglob("*.npz")})
print("Gefundene .npz:", *npz_files, sep="\n- ")



# Sucht alle .npz, lÃ¤dt sie in data_dict und zeigt eine Kurz-Ãœbersicht
from pathlib import Path
import numpy as np

# Alle .npz finden (Kaggle + lokal)
ROOTS = [Path("/kaggle/input"), Path("/kaggle/working"), Path("data"), Path(".")]
NPZ   = sorted({p for r in ROOTS for p in r.rglob("*.npz")})
TASKS = sorted({p.stem.lower() for p in NPZ})
PATHS = {p.stem.lower(): p for p in NPZ}
print("Gefunden:", TASKS)

def _load_npz(path: Path):
    """Liest train/val/test; Labels quetschen; Shapes merken."""
    z = np.load(path)
    g = lambda k: z[k] if k in z.files else None
    xtr, ytr = g("train_images"), g("train_labels")
    xva, yva = g("val_images"),   g("val_labels")
    xte, yte = g("test_images"),  g("test_labels")
    # Labels (N,1) -> (N,)
    for name in ("ytr","yva","yte"):
        arr = locals()[name]
        if arr is not None and arr.ndim > 1:
            locals()[name] = arr.squeeze()
    n_classes = int(np.unique(ytr).size) if ytr is not None else None
    shapes = {"x_train": None if xtr is None else xtr.shape,
              "x_val":   None if xva is None else xva.shape,
              "x_test":  None if xte is None else xte.shape}
    return dict(x_train=xtr,y_train=ytr,x_val=xva,y_val=yva,x_test=xte,y_test=yte,
                n_classes=n_classes, shapes=shapes)

data_dict = {t: _load_npz(PATHS[t]) for t in TASKS}
for t, d in data_dict.items():
    s = d["shapes"]; print(f"{t:12s} | train={s['x_train']} val={s['x_val']} test={s['x_test']} | classes={d['n_classes']}")



# Kurzer visueller Check pro Task
import numpy as np, matplotlib.pyplot as plt

def inspect(task: str, n: int = 6):
    X, y = data_dict[task]["x_train"], data_dict[task]["y_train"]
    idx = np.random.choice(len(X), size=min(n, len(X)), replace=False)

    # Beispielbilder
    cols = min(n, 6)
    plt.figure(figsize=(cols*2.1, 2.2))
    for i, k in enumerate(idx, 1):
        ax = plt.subplot(1, cols, i)
        img = X[k]
        if img.ndim == 3 and img.shape[-1] == 3: ax.imshow(img)              # RGB
        else:                                   ax.imshow(img.squeeze(), cmap="gray")  # Grau
        ax.set_title(f"y={int(y[k])}", fontsize=9); ax.axis("off")
    plt.suptitle(f"{task} â€“ Beispielbilder", y=1.06)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

    # Klassenverteilung
    vals, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(4.2, 3.2))
    plt.bar(vals.astype(int), counts)
    plt.xlabel("Klasse"); plt.ylabel("Anzahl"); plt.title(f"Klassenverteilung â€“ {task}")
    plt.tight_layout(); plt.show()


inspect("pathmnist", n=6)


# Kompakter Trainer (passt sich an Grau/RGB & Klassenzahl an)
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np, matplotlib.pyplot as plt, gc, random

def _ensure_channels(x):
    # (N,28,28) -> (N,28,28,1); (N,28,28,1/3) bleibt
    return x[..., None] if x.ndim == 3 else x

def train_task(task: str, epochs: int = 8, batch: int = 256, seed: int = 42):
    # Seeds
    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)

    d = data_dict[task]
    Xtr, ytr = _ensure_channels(d["x_train"]).astype("float32")/255.0, d["y_train"].astype("int32")
    Xva, yva = d["x_val"], d["y_val"]
    if Xva is None:
        # Fallback: letzten 2000 fÃ¼r Val (oder 10%)
        split = max(2000, int(0.1*len(Xtr)))
        Xtr, Xva = Xtr[:-split], Xtr[-split:]
        ytr, yva = ytr[:-split], ytr[-split:]
    else:
        Xva = _ensure_channels(Xva).astype("float32")/255.0
        yva = yva.astype("int32")

    ch = Xtr.shape[-1]
    n_classes = int(np.unique(ytr).size)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, ch)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=epochs, batch_size=batch, verbose=1, callbacks=[cb])

    model.save(f"{task}_minicnn.keras")
    print(f"ðŸ’¾ gespeichert: {task}_minicnn.keras")
    return model, hist

def evaluate_task(model, task: str):
    d = data_dict[task]
    Xte = _ensure_channels(d["x_test"]).astype("float32")/255.0
    yte = d["y_test"].astype("int32")

    # Accuracy
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"Test-Accuracy â€“ {task}: {acc:.3f}")

    # Report + kleine CM
    pred = model.predict(Xte, batch_size=512, verbose=0).argmax(1)
    print(classification_report(yte, pred, digits=3))

    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(4,3))
    plt.imshow(cm, cmap="Blues"); plt.title(f"CM â€“ {task}")
    plt.xlabel("pred"); plt.ylabel("true"); plt.colorbar(); plt.tight_layout(); plt.show()



# ---- Konfiguration ----
EPOCHS = 6
BATCH  = 256

# a) EIN Task (empfohlen zum Start)
TASK = "pathmnist"
# inspect(TASK, n=6)
model, hist = train_task(TASK, epochs=EPOCHS, batch=BATCH)
evaluate_task(model, TASK)

# b) ALLE verfÃ¼gbaren Tasks (nacheinander, kurz)
results = {}
for t in TASKS:
    print("\n","="*20, t, "="*20)
    m, _ = train_task(t, epochs=EPOCHS, batch=BATCH)
    d = data_dict[t]
    Xte = (d["x_test"][..., None] if d["x_test"].ndim==3 else d["x_test"]).astype("float32")/255.0
    yte = d["y_test"].astype("int32")
    _, acc = m.evaluate(Xte, yte, verbose=0)
    results[t] = acc
    del m; tf.keras.backend.clear_session(); gc.collect()

print("\nErgebnisse:", {k: round(v,3) for k,v in results.items()})




def _ensure_channels(x):
    return x[..., None] if x.ndim == 3 else x


def train_task(task: str, epochs: int = 8, batch: int = 256,
               aug: bool | None = None,         # alte Schreibweise
               use_aug: bool | None = None,      # neue Schreibweise
               use_class_weights: bool = False):
    # Flag zusammenfÃ¼hren (Standard: True)
    if use_aug is None and aug is None:
        aug_flag = True
    else:
        aug_flag = use_aug if use_aug is not None else bool(aug)

    d = data_dict[task]
    Xtr = _ensure_channels(d["x_train"]).astype("float32")/255.0
    ytr = d["y_train"].astype("int32")
    Xva, yva = d["x_val"], d["y_val"]
    if Xva is None:
        split = max(2000, int(0.1*len(Xtr)))
        Xtr, Xva = Xtr[:-split], Xtr[-split:]
        ytr, yva = ytr[:-split], ytr[-split:]
    else:
        Xva = _ensure_channels(Xva).astype("float32")/255.0
        yva = yva.astype("int32")

    ch = Xtr.shape[-1]
    n_classes = int(np.unique(ytr).size)

    aug_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ]) if aug_flag else tf.keras.Sequential([])

    inputs = tf.keras.layers.Input(shape=(28, 28, ch))
    x = aug_layer(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]

    class_weights = None
    if use_class_weights:
        cw = compute_class_weight(class_weight="balanced", classes=np.unique(ytr), y=ytr)
        class_weights = {i: float(w) for i, w in enumerate(cw)}

    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=epochs, batch_size=batch, verbose=1,
                     callbacks=cb, class_weight=class_weights)

    model.save(f"{task}_minicnn.keras")
    print(f"ðŸ’¾ gespeichert: {task}_minicnn.keras")
    return model, hist

def evaluate_task(model, task: str):
    d = data_dict[task]
    Xte = _ensure_channels(d["x_test"]).astype("float32")/255.0
    yte = d["y_test"].astype("int32")
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"Test-Accuracy â€“ {task}: {acc:.3f}")
    pred = model.predict(Xte, batch_size=512, verbose=0).argmax(1)
    print(classification_report(yte, pred, digits=3))
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(4,3)); plt.imshow(cm, cmap="Blues")
    plt.title(f"CM â€“ {task}"); plt.xlabel("pred"); plt.ylabel("true")
    plt.colorbar(); plt.tight_layout(); plt.show()



TASK = "pathmnist"
model, _ = train_task(TASK, epochs=6, batch=256, use_aug=True, use_class_weights=False)
evaluate_task(model, TASK)



import pandas as pd
from tensorflow import keras

tasks = sorted(data_dict.keys())  # oder Liste manuell setzen
rows = []
gid = 0

for t in tasks:
    m = keras.models.load_model(f"{t}_minicnn.keras")
    X = (_ensure_channels(data_dict[t]["x_test"]).astype("float32")/255.0)
    yhat = m.predict(X, batch_size=512, verbose=0).argmax(1)
    for i, lbl in enumerate(yhat):
        rows.append((gid, i, t, int(lbl)))
        gid += 1

sub = pd.DataFrame(rows, columns=["id","id_image_in_task","task_name","label"])
sub.to_csv("submission.csv", index=False)
print("âœ… submission.csv geschrieben:", sub.shape)
sub.head()


