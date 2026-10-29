# ¿Se puede predecir R1 del Melate? - Equipo R1

Examen final de **Aprendizaje Profundo** (Especialidad en Ciencia de Datos, Universidad
Panamericana), trabajo del **Equipo R1**. El reto: dar un solo número para la columna R1 del
Melate, calificado contra tres sorteos de octubre de 2026 con `10 - 0.3 * |sorteo - predicción|`.

**Resultado:** R1 es el menor de seis números entre 1 y 56, así que su distribución se conoce
exactamente (mediana 6). Se entrenaron redes MLP, LSTM, GRU, CNN y Transformer contra esa línea
base estadística; el mejor modelo de Deep Learning no le ganó a la constante en la prueba final,
y la predicción enviada fue **6**.

*Final exam for a Deep Learning course: can deep networks predict the lowest number of Mexico's
Melate lottery? Five architectures were tuned against the exact statistical baseline; the best of
them did not beat it on the held-out test, which is the finding.*

| Empieza aquí | |
|---|---|
| `LEEME.md` | Qué hay en la carpeta y cómo correrlo |
| `RESUMEN.md` | Resultados con tablas y gráficas |
| `Presentacion_R1_Melate.html` | La presentación (abrir en el navegador) |
| `melate_r1.ipynb` | El notebook con el análisis completo |
