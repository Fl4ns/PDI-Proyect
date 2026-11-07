#!/usr/bin/env python3
"""
05_entrenar.py — Entrena DenseNet121 multi-etiqueta en dos etapas.

Uso:
    # 1) Prueba rápida con el subconjunto (debe aprender / sobreajustar)
    python 05_entrenar.py --images-dir /ruta/images --debug \
                          --epochs-head 3 --epochs-ft 0

    # 2) Entrenamiento real
    python 05_entrenar.py --images-dir /ruta/images \
                          --epochs-head 3 --epochs-ft 10

Requiere (generados por los scripts previos):
    data/splits/train.csv
    data/splits/val.csv
    data/splits/class_weights.json
    data/splits/debug_train.csv     (si --debug)
    data/splits/debug_val.csv       (si --debug)

Produce:
    models/modelo_base.keras        Mejor modelo por val_auc
    models/log_etapa1.csv           Métricas de la etapa 1
    models/log_etapa2.csv           Métricas de la etapa 2

Dependencias: tensorflow pandas numpy
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Silenciar logs informativos de TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Activar mixed precision para ROCm
keras.mixed_precision.set_global_policy("mixed_float16")

# ============================================================
# CONFIGURACIÓN
# ============================================================
LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]
NUM_CLASSES = len(LABELS)
IMG_SIZE = 224
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
HEAD_LAYERS = ("gap", "drop", "probs")
LAST_CONV_LAYER = "relu"


# ============================================================
# PREPROCESAMIENTO (idéntico al de inferencia)
# ============================================================
def to_224_rgb(img):
    """(H,W), (H,W,1) o (H,W,3) uint8 -> (224,224,3) uint8."""
    img = tf.convert_to_tensor(img)
    if img.shape.rank == 2:
        img = img[..., tf.newaxis]
    img = tf.cast(img, tf.float32)
    if img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.image.grayscale_to_rgb(img)
    return tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)


def normalize(img_u8):
    """uint8 (...,224,224,3) -> float32 preprocess_input de DenseNet."""
    return tf.keras.applications.densenet.preprocess_input(tf.cast(img_u8, tf.float32))


# ============================================================
# MODELO
# ============================================================
def build_model(freeze_base=True, weights="imagenet", dropout=0.3):
    base = keras.applications.DenseNet121(
        include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(NUM_CLASSES, activation="sigmoid", name="probs")(x)
    model = keras.Model(base.input, out, name="densenet121_cxr14")
    set_backbone_trainable(model, trainable=not freeze_base)
    return model


def set_backbone_trainable(model, trainable):
    for l in model.layers:
        if l.name not in HEAD_LAYERS:
            l.trainable = trainable
    # BatchNorm SIEMPRE congeladas: estabiliza el ajuste fino con lotes chicos
    for l in model.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def unfreeze_last(model, n_layers=60):
    backbone = [l for l in model.layers if l.name not in HEAD_LAYERS]
    for l in backbone[-n_layers:]:
        if not isinstance(l, layers.BatchNormalization):
            l.trainable = True


def make_weighted_bce(pos_weight):
    """BCE multi-etiqueta con peso por clase en el término positivo."""
    pw = tf.constant(np.asarray(pos_weight, dtype="float32"))

    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        return tf.reduce_mean(
            -(pw * y_true * tf.math.log(y_pred)
              + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        )
    return loss


def compile_model(model, lr=1e-3, pos_weight=None):
    loss = make_weighted_bce(pos_weight) if pos_weight is not None else "binary_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss,
        metrics=[keras.metrics.AUC(multi_label=True, num_labels=NUM_CLASSES, name="auc")],
    )
    return model


# ============================================================
# PIPELINE tf.data
# ============================================================
def build_path_index(images_dir):
    """nombre -> ruta, buscando en subcarpetas (Kaggle: images_001/images/...)."""
    index = {p.name: str(p) for p in Path(images_dir).rglob("*.png")}
    if not index:
        raise FileNotFoundError(f"No se encontraron PNG en {images_dir}")
    return index


def make_augmenter():
    """Rotación ±10°, zoom ±10%, brillo y contraste leves.
    SIN volteo horizontal: cambia el lado del corazón."""
    return keras.Sequential([
        layers.RandomRotation(10 / 360, fill_mode="constant", fill_value=0.0),
        layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
        layers.RandomBrightness(0.1, value_range=(0, 255)),
        layers.RandomContrast(0.1),
    ], name="augment")


def _load(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=1)
    return to_224_rgb(img), label


def make_dataset(csv_path, images_dir, batch_size=32, training=False,
                 shuffle_buffer=4096, limit=None, cache=False, seed=SEED):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    index = build_path_index(images_dir)
    missing = [f for f in df["Image Index"] if f not in index]
    if missing:
        raise FileNotFoundError(
            f"Faltan {len(missing)} imágenes en {images_dir}. "
            f"Ejemplos: {missing[:3]}"
        )

    paths = [index[f] for f in df["Image Index"]]
    labels = df[LABELS].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(min(len(df), shuffle_buffer), seed=seed,
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if training:
        aug = make_augmenter()

        def _aug(x, y):
            x = tf.clip_by_value(aug(tf.cast(x, tf.float32), training=True), 0.0, 255.0)
            return x, y

        ds = ds.map(_aug, num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ============================================================
# CALLBACKS
# ============================================================
def make_callbacks(models_dir, tag, patience=3, best_so_far=None):
    """best_so_far evita que la etapa 2 sobrescriba un modelo mejor de la etapa 1."""
    models_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            models_dir / "modelo_base.keras",
            monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
            initial_value_threshold=best_so_far,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(models_dir / f"log_{tag}.csv")),
    ]


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, type=Path,
                    help="Carpeta raíz con los PNG (busca en subcarpetas)")
    ap.add_argument("--splits-dir", default=Path("data/splits"), type=Path)
    ap.add_argument("--models-dir", default=Path("models"), type=Path)
    ap.add_argument("--debug", action="store_true",
                    help="Usa debug_train.csv / debug_val.csv en lugar de train/val")
    ap.add_argument("--limit", type=int, default=None,
                    help="Usar solo las primeras N filas de cada CSV")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=3,
                    help="Épocas con el backbone congelado")
    ap.add_argument("--epochs-ft", type=int, default=10,
                    help="Épocas de ajuste fino (últimas capas)")
    ap.add_argument("--unfreeze", type=int, default=60,
                    help="Cuántas capas del backbone descongelar en la etapa 2")
    ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                    help="Pesos iniciales del backbone")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Desactiva la pérdida ponderada (útil para depurar)")
    ap.add_argument("--resume", default=None, type=Path,
                    help="Continuar desde un .keras existente")
    a = ap.parse_args()

    # --- Verificaciones ---
    print("=" * 60)
    print("05_entrenar.py")
    print("=" * 60)
    tf.get_logger().setLevel("ERROR")
    print(f"TensorFlow : {tf.__version__}")
    print(f"Imágenes   : {a.images_dir}")
    print(f"Splits     : {a.splits_dir}")
    print(f"Modelos    : {a.models_dir}")
    print(f"Modo       : {'DEBUG (subconjunto)' if a.debug else 'COMPLETO'}")
    print(f"Weights    : {a.weights}")
    print("=" * 60)

    tr_csv = a.splits_dir / ("debug_train.csv" if a.debug else "train.csv")
    va_csv = a.splits_dir / ("debug_val.csv" if a.debug else "val.csv")
    for p in (tr_csv, va_csv):
        if not p.exists():
            raise SystemExit(f"Falta {p}. Ejecute primero 02_split_dataset.py")

    # --- Pesos por clase ---
    pos_weight = None
    weights_json = a.splits_dir / "class_weights.json"
    if not a.no_class_weights:
        if not weights_json.exists():
            raise SystemExit(
                f"Falta {weights_json}. Ejecute primero 03_calcular_pesos.py "
                f"o pase --no-class-weights"
            )
        w = json.loads(weights_json.read_text(encoding="utf-8"))
        pos_weight = w["pos_weight_sqrt"]
        print(f"Pesos cargados (raíz): {pos_weight}")

    # --- Datasets ---
    print(f"\nCargando datos...")
    train_ds = make_dataset(tr_csv, a.images_dir, a.batch_size,
                              training=True, limit=a.limit, cache=a.debug)
    val_ds = make_dataset(va_csv, a.images_dir, a.batch_size,
                            training=False, limit=a.limit, cache=a.debug)
    n_train = sum(1 for _ in pd.read_csv(tr_csv, chunksize=100_000)) and len(pd.read_csv(tr_csv))
    n_val = len(pd.read_csv(va_csv))
    if a.limit:
        n_train = min(n_train, a.limit)
        n_val = min(n_val, a.limit)
    print(f"  train: {n_train} imágenes, {int(np.ceil(n_train / a.batch_size))} lotes/época")
    print(f"  val  : {n_val} imágenes")

    # --- Modelo ---
    if a.resume:
        print(f"\nReanudando desde {a.resume}")
        model = keras.models.load_model(a.resume, compile=False)
    else:
        print(f"\nConstruyendo DenseNet121 (weights={a.weights})...")
        model = build_model(freeze_base=True,
                              weights=None if a.weights == "none" else a.weights)

    print(f"  Parámetros totales   : {model.count_params():,}")
    print(f"  Parámetros entrenables: {sum(np.prod(v.shape) for v in model.trainable_weights):,}")

    # --- Etapa 1: backbone congelado ---
    best_auc = None
    if a.epochs_head > 0:
        print("\n" + "=" * 60)
        print(f"ETAPA 1 — Backbone congelado ({a.epochs_head} épocas, lr=1e-3)")
        print("=" * 60)
        compile_model(model, lr=1e-3, pos_weight=pos_weight)
        h1 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=a.epochs_head,
            callbacks=make_callbacks(a.models_dir, "etapa1"),
        )
        best_auc = max(h1.history["val_auc"])
        print(f"\nEtapa 1 terminada. Mejor val_auc: {best_auc:.4f}")

    # --- Etapa 2: ajuste fino ---
    if a.epochs_ft > 0:
        print("\n" + "=" * 60)
        print(f"ETAPA 2 — Ajuste fino ({a.epochs_ft} épocas, lr=1e-5, unfreeze={a.unfreeze})")
        print("=" * 60)
        unfreeze_last(model, a.unfreeze)
        compile_model(model, lr=1e-5, pos_weight=pos_weight)
        print(f"  Parámetros entrenables ahora: "
              f"{sum(np.prod(v.shape) for v in model.trainable_weights):,}")
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=a.epochs_ft,
            callbacks=make_callbacks(a.models_dir, "etapa2",
                                       best_so_far=best_auc),
        )

    print("\n" + "=" * 60)
    print(f"Listo. Mejor modelo en {a.models_dir / 'modelo_base.keras'}")
    print(f"Logs: {a.models_dir}/log_etapa1.csv, log_etapa2.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
