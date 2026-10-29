# Predicción de R1 del Melate - Equipo R1

Examen final de Aprendizaje Profundo. Un solo número para la columna R1, calificado contra los sorteos
del 23, 25 y 28 de octubre con `10 - 0.3 * |sorteo - predicción|`.

## Qué hay en la carpeta

| Archivo | Qué es |
|---|---|
| `Presentacion_R1_Melate.html` | La presentación. Doble clic para abrirla en el navegador. Flechas o barra espaciadora para avanzar; botón de pantalla completa abajo a la derecha. Funciona sin internet (sin internet solo cambian las fuentes). |
| `RESUMEN.md` | Resumen de resultados con todas las tablas y gráficas. |
| `melate_r1.ipynb` | El notebook con todo el análisis y sus salidas. |
| `melate_r1.py` | El mismo notebook como script (es la fuente: `construir_notebook.py` genera el `.ipynb` a partir de él). |
| `melate_nucleo.py` | Código compartido: datos, modelos (MLP, LSTM, GRU, CNN, Transformer), entrenamiento, métodos estadísticos. |
| `experimentos.py` | La búsqueda grande de configuraciones de redes, la prueba final y la predicción. |
| `resumen.py` | Escribe `RESUMEN.md` a partir de `resultados/resultados.json`. |
| `presentacion/` | Plantilla de la presentación y `ensamblar.py`, que le inserta los resultados. |
| `resultados/` | `busqueda.jsonl` (una línea por configuración), `finalistas.json`, `resultados.json`, `figuras/`. |
| `Melate.csv` | El histórico original. Nunca se modifica. |
| `Melate_actualizado.csv` | La descarga oficial más reciente. Solo se usa si conserva idénticos todos los sorteos de `Melate.csv`. |

Ningún número de la presentación ni del resumen está escrito a mano: todos salen de `resultados/resultados.json`.

## Cómo correrlo

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python experimentos.py                  # búsqueda + prueba final + predicción (1-2 horas en CPU)
python construir_notebook.py            # genera y ejecuta melate_r1.ipynb, escribe resultados.json y RESUMEN.md
python presentacion/ensamblar.py        # escribe Presentacion_R1_Melate.html
```

`experimentos.py` guarda cada configuración al terminarla. Si se interrumpe, se vuelve a correr igual y
continúa donde se quedó. `python experimentos.py --rapido --configs 15` hace una prueba de todo el flujo en
unos minutos (en `resultados/rapido/`).

## Si Windows bloquea Python ("An Application Control policy has blocked this file")

Es Smart App Control de Windows, que bloquea DLL sin reputación. El 22 de octubre bloqueó las versiones
nuevas de pandas, torch, Pillow y scikit-learn. No lo desactives (en Windows 11 no se puede volver a
encender sin reinstalar): instala las versiones fijas de `requirements.txt`, que sí cargan. En esta
computadora están en `C:\Users\Hector\venvs\melate-dl2`.

## El 22 de octubre: actualizar con el sorteo del 21

La búsqueda y la prueba final se quedan con los datos con que se corrieron (`finalistas.json` dice hasta qué
concurso). Solo la predicción se recalcula con el histórico nuevo:

```
python experimentos.py --actualizar-prediccion   # descarga, reentrena a los finalistas, predice el siguiente sorteo
python construir_notebook.py
python presentacion/ensamblar.py
```

El primer paso tarda 20-40 minutos (8 finalistas x 3 semillas). Para que no lo corte el límite de una terminal,
lánzalo como proceso aparte, por ejemplo en PowerShell:
`Start-Process python -ArgumentList "-u experimentos.py --actualizar-prediccion" -RedirectStandardOutput resultados\actualizar.log -NoNewWindow`.

La regla de decisión no cambia con la actualización: si el campeón le gana a la mediana se decide en la prueba
final, que no se vuelve a correr. Lo único que puede moverse es el número que predice el campeón.

**El 24, antes de presentar:** si se repiten esos tres comandos después del sorteo del 23, la diapositiva de la
predicción muestra sola el resultado real de cada sorteo ya publicado (23, 25, 28) y su calificación.

Cada fila de `busqueda.jsonl` y `finalistas.json` guarda el último concurso que vio. Si cambian los datos,
`experimentos.py` no mezcla resultados de dos versiones del histórico.

## El método, en corto

1. **Datos.** R1..R6 se publican ordenados, así que R1 es el menor de los seis números. El universo cambió de
   1..39 (1984) a 1..56 (diciembre de 2007); solo se usa la época actual.
2. **Estadística.** R1 tiene una distribución exacta, `P(R1 = k) = C(56-k, 5) / C(56, 6)`, y los datos se
   ajustan a ella. Como el examen castiga el error absoluto, la mejor constante es la mediana (6). Cuatro pruebas
   (frecuencias, Ljung-Box, rachas, independencia) buscan memoria en el sorteo.
3. **Deep Learning.** 240 configuraciones (5 familias, salida L1 / MSE / clasificación, ventanas de 5 a 50
   sorteos, 3 formas de entrada, hiperparámetros) compiten hacia adelante en 500 sorteos de validación.
4. **Prueba final.** El campeón se elige solo con validación y se evalúa una vez en los últimos 500 sorteos.
   Se envía su número solo si le gana a la mediana con un intervalo de confianza de 95 % por encima de cero; si
   no, se envía la mediana.
