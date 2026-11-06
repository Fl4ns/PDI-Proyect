# Análisis del dataset ChestX-ray14

**Total de imágenes analizadas**: 112,120

**Pacientes únicos**: 30,805


## 1. Distribución de clases

| Patología | Positivos | Frecuencia |
|---|---:|---:|
| Atelectasis | 11,559 | 10.31% |
| Consolidation | 4,667 | 4.16% |
| Infiltration | 19,894 | 17.74% |
| Pneumothorax | 5,302 | 4.73% |
| Edema | 2,303 | 2.05% |
| Emphysema | 2,516 | 2.24% |
| Fibrosis | 1,686 | 1.50% |
| Effusion | 13,317 | 11.88% |
| Pneumonia | 1,431 | 1.28% |
| Pleural_Thickening | 3,385 | 3.02% |
| Cardiomegaly | 2,776 | 2.48% |
| Nodule | 6,331 | 5.65% |
| Mass | 5,782 | 5.16% |
| Hernia | 227 | 0.20% |

**Sin hallazgos (todas 0)**: 60,361 (53.84%)


## 2. Nº de hallazgos por imagen

| Nº patologías | Imágenes | % |
|---|---:|---:|
| 0 | 60,361 | 53.84% |
| 1 | 30,963 | 27.62% |
| 2 | 14,306 | 12.76% |
| 3 | 4,856 | 4.33% |
| 4 | 1,247 | 1.11% |
| 5 | 301 | 0.27% |
| 6 | 67 | 0.06% |
| 7 | 16 | 0.01% |
| 8 | 1 | 0.00% |
| 9 | 2 | 0.00% |

## 3. Pares con mayor co-ocurrencia (lift)

Lift > 1 → aparecen juntas más de lo esperado por azar.  

Lift < 1 → se repelen (una excluye a la otra).


| Par | Co-ocurrencias | Lift |
|---|---:|---:|
| Edema + Pneumonia | 340 | 11.567 |
| Pneumothorax + Emphysema | 747 | 6.278 |
| Fibrosis + Pleural_Thickening | 176 | 3.458 |
| Effusion + Cardiomegaly | 1,063 | 3.224 |
| Nodule + Mass | 906 | 2.775 |
| Pleural_Thickening + Mass | 452 | 2.589 |
| Atelectasis + Consolidation | 1,223 | 2.542 |
| Consolidation + Mass | 610 | 2.535 |
| Infiltration + Edema | 981 | 2.401 |
| Atelectasis + Effusion | 3,275 | 2.385 |

### Pares que menos co-ocurren

| Par | Co-ocurrencias | Lift |
|---|---:|---:|
| Fibrosis + Pneumonia | 11 | 0.511 |
| Consolidation + Hernia | 4 | 0.423 |
| Pneumothorax + Cardiomegaly | 49 | 0.373 |
| Pneumothorax + Edema | 33 | 0.303 |
| Edema + Fibrosis | 9 | 0.26 |

## 4. Interpretación clínica (generada desde los datos)

Los pares con mayor lift reflejan asociaciones fisiopatológicas documentadas en la literatura. Se listan los 5 principales con su interpretación clínica y los 3 pares con menor lift.


### Top 5 pares por lift

| Par | Casos | Lift | Interpretación |
|---|---:|---:|---|
| Edema + Pneumonia | 340 | 11.567 | Edema perilesional y complicación aspirativa de la neumonía. |
| Pneumothorax + Emphysema | 747 | 6.278 | Rotura de bullas enfisematosas → neumotórax espontáneo. |
| Fibrosis + Pleural_Thickening | 176 | 3.458 | Fibrosis subpleural con engrosamiento pleural periférico. |
| Effusion + Cardiomegaly | 1,063 | 3.224 | Insuficiencia cardíaca congestiva: corazón grande + derrame pleural. |
| Nodule + Mass | 906 | 2.775 | Espectro continuo: un nódulo es una masa < 3 cm. |

### Top 5 pares por casos absolutos (mayor confiabilidad estadística)

Ordenar por casos da una idea más robusta que ordenar solo por lift, porque los lifts extremos con pocos casos son frágiles.


| Par | Casos | Lift | Interpretación |
|---|---:|---:|---|
| Infiltration + Effusion | 4,000 | 1.693 | Ambos generan opacidades difusas difíciles de separar visualmente. |
| Atelectasis + Effusion | 3,275 | 2.385 | Atelectasia compresiva: el líquido colapsa el pulmón adyacente. |
| Atelectasis + Infiltration | 3,264 | 1.591 | Ambos producen opacidades; muchas veces es difícil distinguirlos en la placa. |
| Infiltration + Nodule | 1,546 | 1.376 | Los nódulos suelen acompañarse de infiltrado perilesional por inflamación o hemorragia local. |
| Consolidation + Effusion | 1,287 | 2.322 | Comparten signo radiológico: opacidad por ocupación del espacio. |

### Pares con menor lift (exclusión parcial)

| Par | Casos | Lift | Interpretación |
|---|---:|---:|---|
| Pneumothorax + Cardiomegaly | 49 | 0.373 | La cardiomegalia es central; el neumotórax suele originarse en la periferia. |
| Pneumothorax + Edema | 33 | 0.303 | Patrones incompatibles: edema bilateral y difuso vs. neumotórax unilateral. |
| Edema + Fibrosis | 9 | 0.26 | Procesos opuestos: edema agudo y alveolar vs. fibrosis crónica e intersticial. |

**Observación sobre clases raras**: Hernia (227 casos) y Pneumonia (1.431 casos) tienen lifts extremos porque cualquier co-ocurrencia es estadísticamente significativa con tan pocos casos. Los pares que las involucran deben interpretarse con cautela.


## 5. Figuras generadas

- `co_ocurrencia.png`: matriz 14×14 (lift)
- `distribucion.png`: nº de hallazgos por imagen
- `co_ocurrencia.csv`: matriz numérica de conteos
- `pares_co_ocurrencia.csv`: ranking completo de los 91 pares
