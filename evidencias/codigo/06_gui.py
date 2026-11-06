#!/usr/bin/env python3
"""
06_gui.py — Interfaz gráfica de ENTREGA.

Detección y localización de patologías torácicas en radiografías de tórax.
Formato visual según el Prototipo 1 (sección 9.2.1 del informe).

Ejecución:
    # Modo real (con modelo entrenado)
    python 06_gui.py

    # Modo stub (sin modelo, para dev)
    RX_USE_STUBS=1 python 06_gui.py

Requisitos:
    - models/modelo_base.keras
    - tensorflow, pydicom, Pillow, opencv-python, gradio, numpy, pandas

Salidas por exportación (en ./resultados/YYYYMMDD_HHMMSS/):
    resultado_*.png     Imagen anotada
    resultado_*.csv     Tabla de las 14 patologías
    resultado_*.json    Metadatos completos

Herramienta de apoyo. NO reemplaza el criterio clínico.
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path

# --- Silenciar logs antes de importar TensorFlow ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURACIÓN
# ============================================================
VERSION_APP = "0.1"
LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]
NUM_CLASSES = len(LABELS)
IMG_SIZE = 224
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "modelo_base.keras"
EXPORT_DIR = Path(__file__).resolve().parent / "resultados"
LAST_CONV_LAYER = "relu"
UMBRAL_DEFECTO = 0.30
USE_STUBS = os.environ.get("RX_USE_STUBS", "0") == "1"
AVISO = "Herramienta de apoyo; no reemplaza el diagnóstico médico"

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 200, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (180, 245, 60), (250, 130, 180),
    (0, 128, 128), (180, 150, 255), (170, 110, 40), (255, 250, 200),
]


# ============================================================
# LÍMITE DE VRAM (evita que la GPU sature y el SO se congele)
# ============================================================
def _configurar_gpu():
    """Limita el crecimiento de VRAM para no asfixiar al driver gráfico."""
    if USE_STUBS:
        return
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"[GPU] No se pudo configurar {gpu}: {e}")
            print(f"[GPU] {len(gpus)} GPU(s) configurada(s) con memory_growth")
    except Exception as e:
        print(f"[GPU] Error configurando: {e}")


# ============================================================
# TEMA
# ============================================================
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#eef2f8",
    body_background_fill_dark="#eef2f8",
    body_text_color="#22303f",
    body_text_color_dark="#22303f",
    body_text_color_subdued="#6b7686",
    body_text_color_subdued_dark="#6b7686",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f8fafd",
    background_fill_secondary_dark="#f8fafd",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_label_background_fill="#e8f0fa",
    block_label_background_fill_dark="#e8f0fa",
    block_label_text_color="#1f4e79",
    block_label_text_color_dark="#1f4e79",
    block_title_text_color="#1f4e79",
    block_title_text_color_dark="#1f4e79",
    border_color_primary="#d6dde8",
    border_color_primary_dark="#d6dde8",
    color_accent="#1f4e79",
    color_accent_soft="#e8f0fa",
    color_accent_soft_dark="#e8f0fa",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#d6dde8",
    input_border_color_dark="#d6dde8",
)


# ============================================================
# PREFLIGHT
# ============================================================
def preflight():
    errores = []
    try:
        import tensorflow as tf
    except ImportError:
        errores.append("Falta TensorFlow. Instalar: pip install tensorflow")
        tf = None
    if not USE_STUBS and not MODEL_PATH.exists():
        errores.append(
            f"No existe el modelo: {MODEL_PATH}\n"
            f"Entrenar primero con: python 05_entrenar.py --images-dir <RUTA>"
        )
    if errores:
        print("=" * 60)
        print("ERROR AL INICIAR LA GUI")
        print("=" * 60)
        for e in errores:
            print(" •", e)
        print("=" * 60)
        sys.exit(1)
    print("=" * 60)
    print(f"GUI de entrega — versión {VERSION_APP}")
    print(f"Python     : {platform.python_version()} ({platform.system()})")
    if tf is not None:
        print(f"TensorFlow : {tf.__version__}")
    if USE_STUBS:
        print("ADVERTENCIA: modo STUB activo.")
    else:
        st = MODEL_PATH.stat()
        fecha = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"Modelo     : {MODEL_PATH.name}  ({st.st_size / 1024 / 1024:.1f} MB, {fecha})")
    print(f"Exports    : {EXPORT_DIR}")
    print("=" * 60)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_model_meta = {"loaded": False}


def model_meta():
    if _model_meta["loaded"]:
        return _model_meta
    if not MODEL_PATH.exists():
        return {"path": str(MODEL_PATH), "exists": False}
    st = MODEL_PATH.stat()
    _model_meta.update({
        "path": str(MODEL_PATH),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": _sha256(MODEL_PATH),
        "loaded": True,
    })
    return _model_meta


# ============================================================
# CARGA DE IMAGEN
# ============================================================
def _first(v):
    try:
        return float(v[0])
    except TypeError:
        return float(v)


def load_dicom(path):
    try:
        ds = pydicom.dcmread(str(path))
    except InvalidDicomError as e:
        raise ValueError("El archivo no es un DICOM válido") from e
    if "PixelData" not in ds:
        raise ValueError("El DICOM no contiene imagen")
    photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photo not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(f"Se esperaba escala de grises y llegó {photo}")
    try:
        arr = ds.pixel_array
    except Exception as e:
        raise ValueError(f"No se pudo decodificar: {e}") from e
    if arr.ndim != 2:
        raise ValueError(f"Se esperaba 2D y llegó {arr.shape}")
    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    bits = int(getattr(ds, "BitsStored", 8))
    if wc is not None and ww is not None:
        c, wd = _first(wc), _first(ww)
        lo, hi = c - wd / 2.0, c + wd / 2.0
    elif bits <= 8 and slope == 1 and intercept == 0:
        lo, hi = 0.0, 255.0
    else:
        lo, hi = np.percentile(arr, (0.5, 99.5))
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
    if photo == "MONOCHROME1":
        out = 255.0 - out
    return np.round(out).astype(np.uint8)


def load_png(path):
    return np.array(Image.open(path).convert("L"))


def cargar_imagen(file):
    path = Path(getattr(file, "name", file))
    suf = path.suffix.lower()
    try:
        if suf == ".dcm":
            return load_dicom(path), "dicom", path
        if suf in (".png", ".jpg", ".jpeg"):
            return load_png(path), "png", path
        raise ValueError(f"Extensión no soportada: {suf}")
    except Exception as e:
        raise ValueError(f"{path.name}: {e}") from e


# ============================================================
# STUBS (dev)
# ============================================================
def _preprocess_stub(img_u8):
    ys = np.linspace(0, img_u8.shape[0] - 1, IMG_SIZE).astype(int)
    xs = np.linspace(0, img_u8.shape[1] - 1, IMG_SIZE).astype(int)
    small = img_u8[np.ix_(ys, xs)].astype("float32") / 255.0
    return np.repeat(small[..., None], 3, axis=2)


def _predict_stub(x):
    seed = int(hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).beta(0.6, 2.5, size=NUM_CLASSES).astype("float32")


def _heatmap_stub(x, class_idx):
    rng = np.random.default_rng(class_idx + 1)
    cy, cx = rng.integers(50, IMG_SIZE - 50, size=2)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    m = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 25.0 ** 2)))
    return (m / m.max()).astype("float32")


# ============================================================
# PRE / INFERENCIA / GRAD-CAM
# ============================================================
def _preprocess_real(img_u8):
    import tensorflow as tf
    img = tf.convert_to_tensor(img_u8)
    if img.shape.rank == 2:
        img = img[..., tf.newaxis]
    img = tf.cast(img, tf.float32)
    if img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.image.grayscale_to_rgb(img)
    img_u8 = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)
    x = tf.keras.applications.densenet.preprocess_input(tf.cast(img_u8, tf.float32))
    return x.numpy().astype(np.float32)


def preprocess(img_u8):
    return _preprocess_stub(img_u8) if USE_STUBS else _preprocess_real(img_u8)


_modelo_cache = {"m": None}


def _get_modelo():
    if _modelo_cache["m"] is None:
        from tensorflow import keras
        _modelo_cache["m"] = keras.models.load_model(MODEL_PATH, compile=False)
    return _modelo_cache["m"]


def predict(x):
    if USE_STUBS:
        return _predict_stub(x)
    return _get_modelo().predict(x[None], verbose=0)[0]


def heatmap(x, class_idx):
    if USE_STUBS:
        return _heatmap_stub(x, class_idx)
    import tensorflow as tf
    from tensorflow import keras
    model = _get_modelo()
    xt = tf.convert_to_tensor(np.asarray(x, dtype=np.float32)[None])
    gm = keras.Model(model.input, [model.get_layer(LAST_CONV_LAYER).output, model.output])
    with tf.GradientTape() as tape:
        tape.watch(xt)
        conv, preds = gm(xt, training=False)
        score = preds[:, int(class_idx)]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE), method="bilinear")[0, :, :, 0].numpy()
    m = cam.max()
    return (cam / m).astype(np.float32) if m > 0 else np.zeros_like(cam, dtype=np.float32)


def to_boxes(hm, thr=0.5, orig_shape=(1024, 1024)):
    hm = np.asarray(hm, dtype=np.float32)
    mask = (hm >= thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [(stats[i, cv2.CC_STAT_AREA], stats[i]) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 0.005 * hm.size]
    comps.sort(key=lambda c: -c[0])
    H, W = orig_shape
    sy, sx = H / hm.shape[0], W / hm.shape[1]
    boxes = []
    for _, s in comps[:2]:
        x0 = int(round(s[cv2.CC_STAT_LEFT] * sx))
        y0 = int(round(s[cv2.CC_STAT_TOP] * sy))
        x1 = int(round((s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]) * sx))
        y1 = int(round((s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]) * sy))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return boxes


# ============================================================
# DIBUJO
# ============================================================
def _to_rgb(img_u8):
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB) if img_u8.ndim == 2 else img_u8


def overlay_heatmap(img_u8, hm, alpha=0.45):
    img = _to_rgb(np.asarray(img_u8, dtype=np.uint8))
    h = cv2.resize(np.asarray(hm, dtype=np.float32), (img.shape[1], img.shape[0]))
    color = cv2.cvtColor(
        cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    w = (alpha * np.clip(h, 0, 1))[..., None]
    return (img * (1 - w) + color * w).astype(np.uint8)


def draw_boxes(img_rgb, boxes, labels, probs):
    im = Image.fromarray(img_rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    for (x, y, w, h), lab, p in zip(boxes, labels, probs):
        color = PALETTE[LABELS.index(lab) % len(PALETTE)] if lab in LABELS else (255, 0, 0)
        d.rectangle([x, y, x + w, y + h], outline=color, width=max(2, im.width // 300))
        txt = f"{lab} · {p * 100:.0f}%"
        try:
            tw = d.textlength(txt)
        except Exception:
            tw = len(txt) * 6
        tx, ty = x + 4, max(0, y - 16)
        d.rectangle([tx, ty, tx + tw + 6, ty + 14], fill=color)
        d.text((tx + 3, ty + 1), txt, fill=(255, 255, 255))
    return np.asarray(im)


# ============================================================
# HELPERS DE ESTADO
# ============================================================
def _status(msg):
    return f'<div id="statusbar-inner"><p>{msg}<span class="warn">⚠ {AVISO}</span></p></div>'


def _render(state, mostradas, ver_heatmap, ver_cajas):
    if state is None:
        return None
    img = state["img"]
    mapa = None
    if ver_heatmap:
        mapas = [state["heatmaps"][lab] for lab in mostradas if lab in state["heatmaps"]]
        if mapas:
            mapa = np.maximum.reduce(mapas)
    base = overlay_heatmap(img, mapa) if mapa is not None else _to_rgb(img)
    if not ver_cajas:
        return base
    boxes, labels, probs = [], [], []
    for lab in mostradas:
        for b in state["boxes"].get(lab, []):
            boxes.append(b)
            labels.append(lab)
            probs.append(state["probs"][LABELS.index(lab)])
    return draw_boxes(base, boxes, labels, probs) if boxes else base


def _results_html(probs, umbral):
    items = []
    for i in np.argsort(-probs):
        lab = LABELS[int(i)]
        p = float(probs[i])
        pct = p * 100
        below = p < umbral
        cls = "result-item below" if below else "result-item"
        checked = "" if below else "checked"
        items.append(f"""
        <label class="{cls}">
            <input type="checkbox" class="sel" data-label="{lab}" {checked}>
            <div class="result-content">
                <div class="result-row">
                    <span class="result-name">{lab}</span>
                    <span class="result-pct">{pct:.1f}%</span>
                </div>
                <div class="result-bar">
                    <div class="result-fill" style="width:{pct:.2f}%"></div>
                </div>
            </div>
        </label>""")
    return (
        '<div class="results-header">Resultados del análisis</div>'
        '<div class="results-list">' + "".join(items) + "</div>"
    )


def _results_empty():
    return (
        '<div class="results-header">Resultados del análisis</div>'
        '<div class="results-empty">'
        '<p style="font-size: 1.1rem; color: #4a5568; margin: 20px 0 8px;">Esperando análisis…</p>'
        '<p class="hint">Cargue una radiografía y pulse Analizar.</p>'
        '</div>'
    )


# ============================================================
# ACCIONES
# ============================================================
def on_upload(file):
    if file is None:
        return None, _results_empty(), None, _status("Listo. Cargue una radiografía DICOM.")
    try:
        img, modo, path = cargar_imagen(file)
    except Exception as e:
        raise gr.Error(f"El archivo no es válido: {e}")
    return (
        _to_rgb(img),
        _results_empty(),
        None,
        _status(f"Imagen cargada: {path.name} ({modo.upper()}). Pulse Analizar."),
    )


def on_analizar(file, umbral):
    if file is None:
        raise gr.Error("Primero cargue un archivo DICOM.")
    try:
        img, modo, path = cargar_imagen(file)
        x = preprocess(img)
        probs = np.asarray(predict(x), dtype=np.float32)
        if probs.shape != (NUM_CLASSES,):
            raise RuntimeError(f"Forma de salida inesperada: {probs.shape}")
        heatmaps, boxes = {}, {}
        for i, lab in enumerate(LABELS):
            hm = heatmap(x, i)
            heatmaps[lab] = hm
            boxes[lab] = to_boxes(hm, 0.5, img.shape)
    except Exception as e:
        print(f"[GUI] error en análisis: {e}\n{traceback.format_exc()}")
        raise gr.Error(f"Error al analizar: {e}")

    detectadas = [LABELS[int(i)] for i in np.argsort(-probs) if probs[i] >= umbral]
    state = {
        "img": img,
        "modo": modo,
        "path": str(path),
        "shape": img.shape,
        "probs": probs,
        "boxes": boxes,
        "heatmaps": heatmaps,
        "umbral": float(umbral),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "sha256_img": hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest(),
        "model_meta": model_meta(),
    }
    render = _render(state, set(detectadas), ver_heatmap=False, ver_cajas=True)
    html = _results_html(probs, umbral)
    sel_json = json.dumps(detectadas)
    msg = (
        f"{len(detectadas)} patología(s) sobre el umbral {umbral:.2f}: {', '.join(detectadas)}."
        if detectadas
        else f"Sin hallazgos sobre el umbral {umbral:.2f}."
    )
    print(f"[ANALYZE] detectadas={len(detectadas)} html_len={len(html)}")
    gr.Info(f"✅ Análisis completo: {len(detectadas)} detección(es)", duration=3)
    return render, html, state, sel_json, _status(msg)


def on_render(sel_json, state, ver_heatmap, ver_cajas):
    if state is None:
        return gr.update()
    try:
        sel = set(json.loads(sel_json))
    except Exception:
        sel = set()
    return _render(state, sel, bool(ver_heatmap), bool(ver_cajas))


def on_exportar(state, sel_json, ver_heatmap, ver_cajas):
    """Exporta PNG + CSV + JSON a ./resultados/YYYYMMDD_HHMMSS/."""
    if state is None:
        raise gr.Error("Primero pulse Analizar para generar los resultados.")

    try:
        mostradas = set(json.loads(sel_json)) if sel_json else set()
    except Exception:
        mostradas = set()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EXPORT_DIR / ts
    out.mkdir(parents=True, exist_ok=True)

    try:
        # 1) PNG anotado
        render = _render(state, mostradas, bool(ver_heatmap), bool(ver_cajas))
        if render is None:
            raise RuntimeError("_render devolvió None")
        png = out / f"resultado_{ts}.png"
        Image.fromarray(render).save(png)

        # 2) CSV con las 14 filas
        filas = []
        for lab in LABELS:
            i = LABELS.index(lab)
            p = float(state["probs"][i])
            bxs = state["boxes"].get(lab, [])
            filas.append({
                "Patologia": lab,
                "Probabilidad": round(p, 6),
                "Sobre_umbral": bool(p >= state["umbral"]),
                "Mostrada": bool(lab in mostradas),
                "N_cajas": int(len(bxs)),
                "Cajas_xywh": ";".join(
                    f"({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])})"
                    for b in bxs
                ),
            })
        csv = out / f"resultado_{ts}.csv"
        pd.DataFrame(filas).to_csv(csv, index=False)

        # 3) JSON con metadata
        meta = {
            "app": {"version": VERSION_APP, "timestamp": ts},
            "modelo": state.get("model_meta", {}),
            "imagen": {
                "path_original": str(state.get("path", "")),
                "modo": str(state.get("modo", "")),
                "shape": [int(x) for x in state.get("shape", [])],
                "sha256": str(state.get("sha256_img", "")),
            },
            "analisis": {
                "umbral": float(state["umbral"]),
                "ver_heatmap": bool(ver_heatmap),
                "ver_cajas": bool(ver_cajas),
                "labels_mostradas": sorted(str(x) for x in mostradas),
            },
            "predicciones": [
                {
                    "label": lab,
                    "prob": round(float(state["probs"][LABELS.index(lab)]), 6),
                    "boxes": [
                        [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                        for b in state["boxes"].get(lab, [])
                    ],
                }
                for lab in LABELS
            ],
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pydicom": pydicom.__version__,
                "opencv": cv2.__version__,
                "gradio": gr.__version__,
            },
        }
        try:
            import tensorflow as tf
            meta["software"]["tensorflow"] = tf.__version__
        except Exception:
            pass

        json_path = out / f"resultado_{ts}.json"
        json_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[EXPORT OK] 3 archivos en {out}")
        gr.Info(f"✅ Exportado en resultados/{ts}/", duration=4)
        return [
            str(png.resolve()),
            str(csv.resolve()),
            str(json_path.resolve()),
        ]

    except Exception as e:
        print(f"[EXPORT ERROR] {e}\n{traceback.format_exc()}")
        raise gr.Error(f"Error al exportar: {e}")


# ============================================================
# CSS
# ============================================================
CSS = """
/* ============================================================
   1. RESET Y FORZAR MODO CLARO
   ============================================================ */
html, body {
    color-scheme: light !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
    background: #eef2f8 !important;
}
html.dark, body.dark, .dark {
    color-scheme: light !important;
}

/* ============================================================
   2. CONTENEDOR PRINCIPAL A PANTALLA COMPLETA
   ============================================================ */
gradio-app, .gradio-container {
    height: 100vh !important;
    max-height: 100vh !important;
    max-width: 100vw !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0 !important;
    border-radius: 0 !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    background: #eef2f8 !important;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container .contain {
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
    max-width: none !important;
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}

/* ============================================================
   3. BARRA DE TÍTULO
   ============================================================ */
#titlebar {
    background: linear-gradient(180deg, #1f4e79, #2e6fa3) !important;
    color: #fff !important;
    padding: 10px 18px !important;
    font-size: 0.92rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
}
#titlebar span { color: #fff !important; }

/* ============================================================
   4. FILA PRINCIPAL — altura calculada explícita
   Reserva: titlebar (~45) + toolbar (~55) + statusbar (~35) + padding (~65)
   ============================================================ */
#main-row {
    height: calc(100vh - 200px) !important;
    min-height: 320px !important;
    max-height: calc(100vh - 200px) !important;
    flex: 0 0 auto !important;
    overflow: hidden !important;
    padding: 10px !important;
    gap: 10px !important;
    background: #eef2f8 !important;
    align-items: stretch !important;
}

/* ============================================================
   5. VISOR — altura 100% del padre, imagen se adapta dentro
   ============================================================ */
#viewer-wrap {
    position: relative !important;     /* ← clave: contenedor de referencia */
    background: #0d0d0d !important;
    border-radius: 6px !important;
    padding: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 300px !important;
    width: 100% !important;
    overflow: hidden !important;
}

/* El wrapper directo del componente Image ocupa todo el área (inset absoluto) */
#viewer-wrap > div {
    position: absolute !important;
    top: 8px !important;
    left: 8px !important;
    right: 8px !important;
    bottom: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    background: transparent !important;
    border: none !important;
}

/* Niveles internos: solo heredan tamaño, NO se les fuerza flex vertical */
#viewer-wrap > div > div,
#viewer-wrap > div > div > div,
#viewer-wrap .image-container,
#viewer-wrap .image-frame,
#viewer-wrap .gradio-image {
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* La imagen se adapta, nunca se deforma */
#viewer-wrap img {
    max-height: 100% !important;
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 4px !important;
    display: block !important;
}

/* Ocultar la toolbar interna del gr.Image (botones de descargar/fullscreen) */
#viewer-wrap .icon-button-wrapper,
#viewer-wrap .toolbar,
#viewer-wrap .icon-buttons,
#viewer-wrap .download-button,
#viewer-wrap .fullscreen-button,
#viewer-wrap button[aria-label*="descarg"],
#viewer-wrap button[aria-label*="Download"],
#viewer-wrap button[aria-label*="fullscreen"],
#viewer-wrap button[aria-label*="Full"] {
    display: none !important;
}

/* ============================================================
   6. PANEL DE RESULTADOS — grid 1fr auto
   ============================================================ */
#results-wrap {
    background: #ffffff !important;
    border: 1px solid #d6dde8 !important;
    border-radius: 6px !important;
    padding: 0 !important;
    height: 100% !important;
    min-height: 300px !important;
    overflow: hidden !important;
    display: grid !important;
    grid-template-rows: 1fr auto !important;
    gap: 0 !important;
}

#results-panel {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    min-height: 150px !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #ffffff !important;
    border: none !important;
}
#results-panel > div,
#results-panel .prose {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
    width: 100% !important;
}

.results-header {
    padding: 10px 14px;
    border-bottom: 1px solid #d6dde8;
    font-weight: 600;
    font-size: 0.9rem;
    color: #1f4e79;
    background: #e8f0fa;
    position: sticky;
    top: 0;
    z-index: 1;
}
.results-list { display: flex; flex-direction: column; }
.results-empty { padding: 30px 16px; color: #6b7686; font-size: 0.9rem; text-align: center; }
.results-empty .hint { font-size: 0.8rem; color: #90a4ae; margin-top: 8px; }

.result-item {
    display: grid;
    grid-template-columns: 20px 1fr;
    gap: 10px;
    padding: 8px 14px;
    cursor: pointer;
    border-bottom: 1px solid #f0f4f9;
    align-items: center;
    transition: background 0.15s;
}
.result-item:hover { background: #f8fafd; }
.result-item input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: #1f4e79;
    cursor: pointer;
    margin: 0;
}
.result-content { min-width: 0; }
.result-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 4px;
    gap: 8px;
}
.result-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: #22303f;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.result-pct {
    font-size: 0.8rem;
    color: #6b7686;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
}
.result-bar {
    height: 7px;
    background: #e6ebf2;
    border-radius: 4px;
    overflow: hidden;
}
.result-fill {
    height: 100%;
    background: linear-gradient(90deg, #4fc3f7 0%, #2e6fa3 60%, #1f4e79 100%);
    border-radius: 4px;
    transition: width 0.45s cubic-bezier(0.4, 0, 0.2, 1);
}
.result-item.below .result-fill {
    background: linear-gradient(90deg, #cfd8dc, #90a4ae);
}
.result-item.below .result-name { color: #90a4ae; font-weight: 500; }
.result-item.below .result-pct  { color: #b0bec5; }

/* ============================================================
   7. ACCORDION DE EXPORT (dentro del panel derecho)
   ============================================================ */
#export-wrap {
    background: #f8fafd !important;
    border-top: 1px solid #d6dde8 !important;
    padding: 4px 10px !important;
    max-height: 220px !important;
    overflow-y: auto !important;
    border-radius: 0 0 6px 6px !important;
}
#export-wrap .label-wrap {
    padding: 6px 0 !important;
    font-size: 0.85rem !important;
}
#export-wrap .file-preview { min-height: 40px !important; }

/* ============================================================
   8. TOOLBAR — siempre visible abajo
   ============================================================ */
#toolbar {
    background: #f8fafd !important;
    border-top: 1px solid #d6dde8 !important;
    padding: 10px 14px !important;
    gap: 10px !important;
    align-items: center !important;
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
    z-index: 10 !important;
}
#toolbar button {
    background: #1f4e79 !important;
    color: #fff !important;
    border: 1px solid #1f4e79 !important;
    padding: 7px 14px !important;
    border-radius: 4px !important;
    font-size: 0.83rem !important;
    min-width: unset !important;
}
#toolbar button:hover { background: #2e6fa3 !important; }
#toolbar .upload-btn button {
    background: #fff !important;
    color: #1f4e79 !important;
}
#toolbar label {
    font-size: 0.83rem !important;
    color: #4a5568 !important;
}

/* ============================================================
   9. STATUS BAR — siempre visible abajo
   ============================================================ */
#statusbar {
    background: #e8f0fa !important;
    padding: 8px 14px !important;
    font-size: 0.8rem !important;
    border-top: 1px solid #d6dde8 !important;
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
    z-index: 10 !important;
}
#statusbar p {
    margin: 0 !important;
    color: #1f4e79 !important;
}
#statusbar .warn {
    color: #b58a00 !important;
    margin-left: 12px;
    font-weight: 500;
}

/* ============================================================
   10. TEXTBOX PUENTE OCULTO
   ============================================================ */
.hidden-tb {
    position: absolute !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
"""

HEAD = """
<script>
(function() {
    function setNativeValue(el, value) {
        const proto = Object.getPrototypeOf(el);
        const desc = Object.getOwnPropertyDescriptor(proto, 'value');
        if (desc && desc.set) desc.set.call(el, value);
        else el.value = value;
    }
    function syncSelection() {
        const checked = Array.from(
            document.querySelectorAll('#results-panel input.sel:checked')
        ).map(cb => cb.dataset.label);

        const container = document.getElementById('sel-json');
        if (!container) return;
        const input = container.querySelector('textarea, input[type="text"]');
        if (!input) return;

        const val = JSON.stringify(checked);
        if (input.value === val) return;

        setNativeValue(input, val);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }
    document.addEventListener('change', function(e) {
        if (e.target && e.target.matches && e.target.matches('#results-panel input.sel')) {
            syncSelection();
        }
    });
})();
</script>
"""


# ============================================================
# LAYOUT
# ============================================================
def construir():
    with gr.Blocks(title="Detección de Patologías Torácicas") as demo:
        gr.HTML(f"""
        <div id="titlebar">
          <span>🩺 Detección de Patologías Torácicas — v{VERSION_APP}</span>
          <span style="opacity:0.75;font-size:0.82rem;"></span>
        </div>
        """)

        state = gr.State(None)
        sel_json = gr.Textbox(
            value="[]",
            elem_id="sel-json",
            elem_classes=["hidden-tb"],
            show_label=False,
            container=False,
        )

        with gr.Row(elem_id="main-row", equal_height=True):
            with gr.Column(scale=5, elem_id="viewer-wrap", min_width=400):
                visor = gr.Image(
                    type="numpy",
                    show_label=False,
                    interactive=False,
                    container=False,
                )
            with gr.Column(scale=3, elem_id="results-wrap", min_width=340):
                results_html = gr.HTML(_results_empty(), elem_id="results-panel")
                with gr.Accordion(
                    "📥 Archivos exportados",
                    open=True,
                    elem_id="export-wrap",
                ):
                    salida = gr.File(
                        label="",
                        file_count="multiple",
                        show_label=False,
                        height=140,
                    )

        with gr.Row(elem_id="toolbar"):
            upload = gr.UploadButton(
                "📂 Cargar DICOM",
                file_types=[".dcm", ".png"],
                file_count="single",
                scale=0,
                min_width=150,
                elem_classes=["upload-btn"],
            )
            analizar = gr.Button("▶ Analizar", scale=0, min_width=120)
            exportar = gr.Button("💾 Exportar", scale=0, min_width=120)
            ver_heatmap = gr.Checkbox(
                label="🗺 Mapa de calor", value=False,
                scale=0, min_width=160,
            )
            ver_cajas = gr.Checkbox(
                label="📦 Cajas", value=True,
                scale=0, min_width=120,
            )
            umbral = gr.Slider(
                0.05, 0.95, value=UMBRAL_DEFECTO, step=0.05,
                label="Umbral", scale=1,
            )

        status = gr.HTML(
            _status("Listo. Cargue una radiografía DICOM."),
            elem_id="statusbar",
        )

        # --- Eventos ---
        upload.upload(
            on_upload,
            [upload],
            [visor, results_html, state, status],
        )
        analizar.click(
            on_analizar,
            [upload, umbral],
            [visor, results_html, state, sel_json, status],
        )
        umbral.release(
            on_analizar,
            [upload, umbral],
            [visor, results_html, state, sel_json, status],
        )
        sel_json.change(
            on_render,
            [sel_json, state, ver_heatmap, ver_cajas],
            [visor],
            queue=False,
        )
        ver_heatmap.change(
            on_render,
            [sel_json, state, ver_heatmap, ver_cajas],
            [visor],
            queue=False,
        )
        ver_cajas.change(
            on_render,
            [sel_json, state, ver_heatmap, ver_cajas],
            [visor],
            queue=False,
        )
        exportar.click(
            on_exportar,
            [state, sel_json, ver_heatmap, ver_cajas],
            [salida],
        )

    return demo


def _find_free_port(start=7860, end=7900):
    import socket
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No hay puertos libres entre {start} y {end}")


if __name__ == "__main__":
    preflight()
    _configurar_gpu()
    demo = construir()

    # Cola con concurrencia limitada (evita saturación de la GPU)
    demo.queue(default_concurrency_limit=4)

    env_port = os.environ.get("GRADIO_SERVER_PORT")
    port = int(env_port) if env_port else _find_free_port(7860, 7900)
    print(f"Abriendo GUI en http://127.0.0.1:{port}")

    demo.launch(
        css=CSS,
        head=HEAD,
        theme=THEME,
        server_name="127.0.0.1",
        server_port=port,
    )
