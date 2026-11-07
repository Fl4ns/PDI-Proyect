#!/usr/bin/env python3
"""
11_reporte_tests.py — Ejecuta los tests y genera el reporte para la bitácora.

Uso:
    python 11_reporte_tests.py
    python 11_reporte_tests.py --include-model-tests    # incluye los marcados como requires_model

Produce:
    reportes/tests_sin_modelo.md     Reporte en Markdown (por defecto)
    reportes/tests_completo.md       Reporte completo (si --include-model-tests)

Los tests que requieren modelo entrenado se marcan con @pytest.mark.requires_model
y se excluyen por defecto. Los que sí corren (DCM, PRE, splits, modelo vacío,
LOC sin inferencia) se ejecutan normalmente.

No necesita modelo entrenado ni GPU.

Dependencias: pytest
"""
import argparse
import datetime
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reportes"


# ============================================================
# UTILIDADES
# ============================================================
def _check_pytest():
    """Verifica que pytest esté instalado antes de nada."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("ERROR: pytest no está instalado.")
        print("       pip install pytest")
        sys.exit(1)


def _check_tests_dir():
    """Verifica que exista tests/ con archivos de test."""
    tests = ROOT / "tests"
    if not tests.exists():
        print(f"ERROR: no existe {tests}")
        print("       Copia los archivos de tests/ del proyecto.")
        sys.exit(1)
    py_files = list(tests.glob("test_*.py"))
    if not py_files:
        print(f"ERROR: no hay archivos test_*.py en {tests}")
        sys.exit(1)
    print(f"Tests encontrados: {len(py_files)} archivos")
    for f in py_files:
        print(f"  - {f.name}")


def _run_pytest(xml_path, include_model_tests=False):
    """Ejecuta pytest y devuelve el código de salida."""
    cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={xml_path}"]
    if not include_model_tests:
        cmd += ["-m", "not requires_model"]

    print(f"\nEjecutando: {' '.join(cmd)}")
    print("-" * 60)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    print("-" * 60)
    return result.returncode


def _parse_junit(xml_path):
    """Lee el XML de JUnit y devuelve lista de dicts con cada testcase."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    casos = []
    for tc in root.iter("testcase"):
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        time = tc.get("time", "0")

        estado = "OK"
        mensaje = ""
        if tc.find("failure") is not None:
            estado = "FALLÓ"
            mensaje = tc.find("failure").get("message", "")[:200]
        elif tc.find("error") is not None:
            estado = "ERROR"
            mensaje = tc.find("error").get("message", "")[:200]
        elif tc.find("skipped") is not None:
            estado = "OMITIDA"
            mensaje = tc.find("skipped").get("message", "")[:200]

        # Módulo: última parte del classname (ej: "tests.test_dcm")
        modulo = classname.split(".")[-1] if classname else "—"

        # Buscar un ID (pu1, pm1, etc.) en el nombre del test
        m = re.search(r"(pu\d|pm\d|ps\d|int\d|t\d+)", name.lower())
        id_txt = m.group(1).upper() if m else "—"

        casos.append({
            "modulo": modulo,
            "nombre": name,
            "id": id_txt,
            "estado": estado,
            "time": float(time),
            "mensaje": mensaje,
        })

    return casos


def _resumen_por_modulo(casos):
    """Agrupa los resultados por módulo de test."""
    modulos = {}
    for c in casos:
        m = c["modulo"]
        if m not in modulos:
            modulos[m] = {"OK": 0, "FALLÓ": 0, "ERROR": 0, "OMITIDA": 0, "total": 0, "time": 0.0}
        modulos[m][c["estado"]] += 1
        modulos[m]["total"] += 1
        modulos[m]["time"] += c["time"]
    return modulos


def _cobertura_por_objetivo(casos):
    """Mapea cada test a un objetivo del proyecto (heurística por nombre de archivo)."""
    mapping = {
        "test_dcm": ("O4", "Lector DICOM"),
        "test_pre": ("O4", "Preprocesamiento"),
        "test_pipeline": ("O4", "Pipeline tf.data"),
        "test_splits": ("O1", "División por paciente"),
        "test_split": ("O1", "División por paciente"),
        "test_model": ("O2", "Arquitectura del modelo"),
        "test_predict": ("O2", "Inferencia"),
        "test_loc": ("O3", "Grad-CAM y cajas"),
        "test_consistency": ("O4", "Consistencia DICOM/PNG"),
        "test_curvas": ("O2", "Evaluación y curvas"),
        "test_eval": ("O3", "Evaluación de localización"),
        "test_draw": ("O4", "Dibujo y export"),
        "test_app": ("O4", "GUI (stubs)"),
        "test_splits_pesos": ("O1", "Pesos y subset"),
    }
    filas = {}
    for c in casos:
        mod = c["modulo"]
        clave = next((k for k in mapping if k in mod), None)
        if clave:
            objetivo, descripcion = mapping[clave]
        else:
            objetivo, descripcion = "—", mod
        if (objetivo, descripcion) not in filas:
            filas[(objetivo, descripcion)] = {"OK": 0, "FALLÓ": 0, "ERROR": 0, "OMITIDA": 0, "total": 0}
        filas[(objetivo, descripcion)][c["estado"]] += 1
        filas[(objetivo, descripcion)]["total"] += 1
    return filas


def _build_markdown(casos, include_model_tests):
    """Construye el contenido del reporte Markdown."""
    total = len(casos)
    n_ok = sum(1 for c in casos if c["estado"] == "OK")
    n_fail = sum(1 for c in casos if c["estado"] in ("FALLÓ", "ERROR"))
    n_skip = sum(1 for c in casos if c["estado"] == "OMITIDA")
    tiempo = sum(c["time"] for c in casos)

    ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alcance = "completo" if include_model_tests else "sin dependencia del modelo"

    lines = []
    lines.append(f"# Reporte de pruebas — {alcance}")
    lines.append("")
    lines.append(f"**Fecha de ejecución**: {ahora}")
    lines.append(f"**Duración total**: {tiempo:.2f} s")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append(f"| Métrica | Valor |")
    lines.append(f"|---|---:|")
    lines.append(f"| Total | {total} |")
    lines.append(f"| Aprobadas | {n_ok} |")
    lines.append(f"| Fallidas | {n_fail} |")
    lines.append(f"| Omitidas | {n_skip} |")
    lines.append(f"| Duración | {tiempo:.2f} s |")
    lines.append("")

    # Tabla por módulo
    lines.append("## Por módulo")
    lines.append("")
    lines.append("| Módulo | Total | OK | Falló | Omitida | Tiempo |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    modulos = _resumen_por_modulo(casos)
    for mod in sorted(modulos.keys()):
        m = modulos[mod]
        lines.append(f"| {mod} | {m['total']} | {m['OK']} | {m['FALLÓ'] + m['ERROR']} | {m['OMITIDA']} | {m['time']:.3f} s |")
    lines.append("")

    # Cobertura por objetivo
    lines.append("## Cobertura por objetivo")
    lines.append("")
    lines.append("| Objetivo | Módulo | Total | OK | Falló | Omitida |")
    lines.append("|---|---|---:|---:|---:|---:|")
    filas = _cobertura_por_objetivo(casos)
    for (obj, desc), f in sorted(filas.items()):
        lines.append(f"| {obj} | {desc} | {f['total']} | {f['OK']} | {f['FALLÓ'] + f['ERROR']} | {f['OMITIDA']} |")
    lines.append("")

    # Detalle por test
    lines.append("## Detalle")
    lines.append("")
    lines.append("| ID | Módulo | Prueba | Estado | Tiempo |")
    lines.append("|---|---|---|---|---:|")
    for c in casos:
        emoji = {"OK": "✅", "FALLÓ": "❌", "ERROR": "⚠️", "OMITIDA": "⏭️"}.get(c["estado"], "")
        lines.append(f"| {c['id']} | {c['modulo']} | {c['nombre']} | {emoji} {c['estado']} | {c['time']:.3f} s |")
    lines.append("")

    # Fallas
    fallas = [c for c in casos if c["estado"] in ("FALLÓ", "ERROR")]
    if fallas:
        lines.append("## Fallas a corregir")
        lines.append("")
        for c in fallas:
            lines.append(f"### `{c['nombre']}` ({c['modulo']})")
            lines.append("")
            lines.append(f"```")
            lines.append(c["mensaje"] or "(sin mensaje)")
            lines.append(f"```")
            lines.append("")

    # Omitidas
    omitidas = [c for c in casos if c["estado"] == "OMITIDA"]
    if omitidas:
        lines.append("## Omitidas")
        lines.append("")
        lines.append("Estas pruebas requieren el modelo entrenado o TensorFlow, y se "
                     "ejecutarán después del entrenamiento completo.")
        lines.append("")
        for c in omitidas:
            motivo = c["mensaje"].split("\n")[0] if c["mensaje"] else "sin motivo especificado"
            lines.append(f"- `{c['nombre']}` ({c['modulo']}): {motivo}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-model-tests", action="store_true",
                    help="Incluye los tests marcados con @pytest.mark.requires_model")
    ap.add_argument("--out", type=Path, default=None,
                    help="Ruta del .md de salida (por defecto reportes/tests_*.md)")
    a = ap.parse_args()

    # Pre-flight
    print("=" * 60)
    print("11_reporte_tests.py")
    print("=" * 60)
    _check_pytest()
    _check_tests_dir()

    # Ejecutar pytest
    OUT_DIR.mkdir(exist_ok=True)
    xml = ROOT / "resultados_pytest.xml"
    _run_pytest(xml, a.include_model_tests)

    if not xml.exists():
        print("\nERROR: pytest no generó el XML de resultados.")
        sys.exit(1)

    # Parsear
    casos = _parse_junit(xml)

    # Construir Markdown
    md = _build_markdown(casos, a.include_model_tests)

    # Guardar
    if a.out is None:
        suffix = "completo" if a.include_model_tests else "sin_modelo"
        a.out = OUT_DIR / f"tests_{suffix}.md"
    a.out.write_text(md, encoding="utf-8")

    # Limpiar XML
    xml.unlink(missing_ok=True)

    # Resumen consola
    total = len(casos)
    ok = sum(1 for c in casos if c["estado"] == "OK")
    fail = sum(1 for c in casos if c["estado"] in ("FALLÓ", "ERROR"))
    skip = sum(1 for c in casos if c["estado"] == "OMITIDA")

    print()
    print("=" * 60)
    print(f"Total: {total} | OK: {ok} | Fallaron: {fail} | Omitidas: {skip}")
    print(f"Reporte: {a.out}")
    print("=" * 60)

    # Código de salida
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()