# Reporte de pruebas — sin dependencia del modelo

**Fecha de ejecución**: 2026-11-06 13:47:50
**Duración total**: 24.12 s

## Resumen

| Métrica | Valor |
|---|---:|
| Total | 26 |
| Aprobadas | 26 |
| Fallidas | 0 |
| Omitidas | 0 |
| Duración | 24.12 s |

## Por módulo

| Módulo | Total | OK | Falló | Omitida | Tiempo |
|---|---:|---:|---:|---:|---:|
| test_dcm | 6 | 6 | 0 | 0 | 1.225 s |
| test_loc | 3 | 3 | 0 | 0 | 0.002 s |
| test_model | 3 | 3 | 0 | 0 | 21.601 s |
| test_pre | 3 | 3 | 0 | 0 | 0.597 s |
| test_split | 4 | 4 | 0 | 0 | 0.394 s |
| test_subset | 2 | 2 | 0 | 0 | 0.197 s |
| test_verify | 2 | 2 | 0 | 0 | 0.100 s |
| test_weights | 3 | 3 | 0 | 0 | 0.004 s |

## Cobertura por objetivo

| Objetivo | Módulo | Total | OK | Falló | Omitida |
|---|---|---:|---:|---:|---:|
| O1 | División por paciente | 4 | 4 | 0 | 0 |
| O2 | Arquitectura del modelo | 3 | 3 | 0 | 0 |
| O3 | Grad-CAM y cajas | 3 | 3 | 0 | 0 |
| O4 | Lector DICOM | 6 | 6 | 0 | 0 |
| O4 | Preprocesamiento | 3 | 3 | 0 | 0 |
| — | test_subset | 2 | 2 | 0 | 0 |
| — | test_verify | 2 | 2 | 0 | 0 |
| — | test_weights | 3 | 3 | 0 | 0 |

## Detalle

| ID | Módulo | Prueba | Estado | Tiempo |
|---|---|---|---|---:|
| PU1 | test_dcm | test_pu1_lee_monochrome2 | ✅ OK | 1.219 s |
| PU1 | test_dcm | test_pu1_monochrome1_invierte | ✅ OK | 0.002 s |
| PU1 | test_dcm | test_pu1_12bits_con_ventana | ✅ OK | 0.002 s |
| PU1 | test_dcm | test_pu1_16bits_sin_ventana_usa_percentiles | ✅ OK | 0.002 s |
| PU2 | test_dcm | test_pu2_rechaza_no_dicom | ✅ OK | 0.000 s |
| PU2 | test_dcm | test_pu2_archivo_inexistente | ✅ OK | 0.000 s |
| PU5 | test_loc | test_pu5_to_boxes_dentro_de_limites | ✅ OK | 0.002 s |
| PU5 | test_loc | test_pu5_to_boxes_vacio | ✅ OK | 0.000 s |
| PU5 | test_loc | test_pu5_to_boxes_escala | ✅ OK | 0.000 s |
| PU4 | test_model | test_pu4_salida_14_en_0_1 | ✅ OK | 20.373 s |
| PU4 | test_model | test_pu4_capa_gradcam | ✅ OK | 0.574 s |
| PU4 | test_model | test_pu4_compila_con_pesos | ✅ OK | 0.654 s |
| PU3 | test_pre | test_pu3_to_224_rgb_forma | ✅ OK | 0.511 s |
| PU3 | test_pre | test_pu3_to_224_rgb_acepta_1_y_3_canales | ✅ OK | 0.073 s |
| PU3 | test_pre | test_pu3_normalize_tipo_y_rango | ✅ OK | 0.013 s |
| PM1 | test_split | test_pm1_split_sin_pacientes_repetidos | ✅ OK | 0.194 s |
| PM1 | test_split | test_pm1_check_no_overlap_detecta_repetidos | ✅ OK | 0.000 s |
| — | test_split | test_columnas_multihot_en_orden | ✅ OK | 0.100 s |
| — | test_split | test_split_proporcion_val | ✅ OK | 0.100 s |
| PM1 | test_subset | test_pm1_subset_contiene_todas_las_clases | ✅ OK | 0.099 s |
| PM1 | test_subset | test_pm1_subset_respeta_min_pos | ✅ OK | 0.098 s |
| PU1 | test_verify | test_pu1_conteos_ok | ✅ OK | 0.098 s |
| PU1 | test_verify | test_pu1_add_multihot_con_multiples_etiquetas | ✅ OK | 0.002 s |
| PM1 | test_weights | test_pm1_weights_devuelve_14 | ✅ OK | 0.002 s |
| PM1 | test_weights | test_pm1_weights_hernia_es_el_mayor | ✅ OK | 0.001 s |
| PM1 | test_weights | test_pm1_weights_todos_positivos | ✅ OK | 0.001 s |
