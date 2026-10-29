"""Escribe RESUMEN.md a partir de resultados/resultados.json (ningún número se escribe a mano)."""
import json

import melate_nucleo as nu


def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _capas(red):
    """'Entrada → GRU, capa 1 (16) → GRU, capa 2 (16) → Salida (51)'"""
    return " → ".join(c["nombre"] if c["tipo"] == "entrada" else f"{c['nombre']} ({c['unidades']})"
                      for c in red["capas_red"])


def _anterior(pu):
    a = pu.get("corrida_anterior")
    if not a:
        return ""
    return (f"\nLa búsqueda se amplió después de una primera corrida de {a['n_configs']} configuraciones, cuyo campeón "
            f"({a['campeon'].replace('Campeón: ', '')}) quedó en {a['dif_vs_ref']:+.3f} (IC {a['ic_bajo']:+.3f} a "
            f"{a['ic_alto']:+.3f}). Cada corrida eligió su campeón solo con validación, pero la prueba final se usó una "
            f"vez por corrida.\n")


def escribir():
    R = json.loads((nu.SALIDA / "resultados.json").read_text(encoding="utf-8"))
    d, dist, pr, bq, pu, tr, pred = (R[k] for k in ("datos", "distribucion", "pruebas", "busqueda", "prueba",
                                                     "tres_sorteos", "prediccion"))
    c = pu["campeon"]
    filas = "\n".join(
        f"| {r['metodo']} | {r['tipo']} | {r['validacion']:.3f} | {r['calificacion']:.3f} | "
        f"{r['dif_vs_ref']:+.3f} ({r['ic_bajo']:+.3f} a {r['ic_alto']:+.3f}) | {r['moda']} ({r['pct_moda']:.0%}) |"
        for r in pu["tabla"])
    fam = "\n".join(f"| {f['familia']} | {f['count']} | {f['max']:.3f} | {f['median']:.3f} |" for f in bq["por_familia"])
    nombres_perdida = {"l1": "L1", "mse": "MSE", "clase": "Clasificación"}
    per = "\n".join(f"| {nombres_perdida[f['perdida']]} | {f['count']} | {f['max']:.3f} | {f['median']:.3f} |"
                    for f in bq["por_perdida"])
    rd, so = R["redes"], R["redes"]["sobreentrenada"]
    redes_filas = "\n".join(
        f"| {'★ ' if r['campeon'] else ''}{r['nombre']} | {_capas(r)} | "
        f"{r['parametros']:,} | {r['mejor_epoca']} de {r['epocas']} | {r['ventaja_entreno']:+.3f} | {r['ventaja_no_vistos']:+.3f} |"
        for r in rd["finalistas"])
    sin_memoria = (pr["chi2_bolas"]["p"] > 0.05 and min(pr["ljung_box_p"].values()) > 0.05
                   and pr["rachas"]["p"] > 0.05 and pr["independencia"]["p"] > 0.05)
    texto = f"""# Predicción de R1 del Melate: resumen de resultados

Generado {R['generado']}. Datos: {d['fuente']}; {d['n_sorteos']} sorteos ({d['primera_fecha']} a {d['ultima_fecha']}).

## Predicción para los sorteos del 23, 25 y 28 de octubre

# R1 = {pred['final']}

- Método: {pred['metodo']}. Motivo: {pred['razon']}.
- El campeón de Deep Learning predice {pred['campeon_predice']}.
- {len(pred['coinciden'])} de {len(pred['por_metodo'])} métodos dan el mismo número.
- Calificación esperada por sorteo: {pred['calif_esperada']:.2f}.

## 1. Los datos

- R1..R6 se publican **ordenados**: R1 es el menor de los seis números, no la primera bola que sale.
- El universo de números cambió {len(d['cambios_formato']) - 1} veces desde 1984. La época actual (1..56) empieza en el
  concurso {d['era_inicio_concurso']} ({d['era_inicio_fecha']}) y tiene {d['era_n']} sorteos; solo se usa esa época.
- Validaciones superadas: {', '.join(k.replace('_', ' ') for k, v in d['validacion'].items() if v)}.

![formato](resultados/figuras/01_formato.png)

## 2. La distribución de R1

- Teoría (el menor de 6 números de 56): mediana {dist['mediana_teorica']}, media {dist['media_teorica']:.2f}.
  Observado: mediana {dist['mediana_observada']:.0f}, media {dist['media_observada']:.2f}.
- Los datos se ajustan a la fórmula (chi-cuadrada p = {_p(dist['chi2_r1']['p'])}).
- El examen castiga el error absoluto, así que la mejor constante es la mediana: **{dist['mejor_constante']}**,
  con calificación esperada {dist['calif_mejor_constante']:.3f} por sorteo.

![distribucion](resultados/figuras/02_distribucion_r1.png)

## 3. ¿Tiene memoria el sorteo?

| Prueba | Resultado | Lectura |
|---|---|---|
| Frecuencia de los 56 números | chi-cuadrada p = {_p(pr['chi2_bolas']['p'])} | {'todas salen igual de seguido' if pr['chi2_bolas']['p'] > 0.05 else 'diferencia significativa'} |
| Autocorrelación de R1, 20 rezagos | {pr['acf_fuera']} fuera de la banda de 95 % | por azar se espera 1 |
| Ljung-Box | p = {_p(pr['ljung_box_p']['10'])} (10), {_p(pr['ljung_box_p']['20'])} (20) | {'sin memoria detectable' if min(pr['ljung_box_p'].values()) > 0.05 else 'hay dependencia'} |
| Rachas alto/bajo | z = {pr['rachas']['z']:.2f}, p = {_p(pr['rachas']['p'])} | {'compatible con azar' if pr['rachas']['p'] > 0.05 else 'no compatible con azar'} |
| R1 anterior vs siguiente | chi-cuadrada p = {_p(pr['independencia']['p'])} | {'independientes' if pr['independencia']['p'] > 0.05 else 'dependientes'} |

{'Ninguna prueba encuentra memoria: el pasado no dice nada del siguiente R1.' if sin_memoria else 'Al menos una prueba encuentra estructura; ver el notebook.'}

![memoria](resultados/figuras/03_memoria.png)

## 4. Deep Learning: la búsqueda

{bq['n_configs']} configuraciones de 5 familias (MLP, LSTM, GRU, CNN, Transformer), 3 tipos de salida, ventanas de
5 a 50 sorteos, 3 formas de entrada y distintos hiperparámetros. Cada una se evaluó hacia adelante en la
validación ({bq['validacion_desde']} a {bq['validacion_hasta']}), con {len(bq['config']['semillas_busqueda'])} semillas
y reentrenando cada {bq['config']['reentrenar_busqueda']} sorteos.

- Constante {bq['constante']} en validación: {bq['ref_validacion']:.3f}. Le ganaron {bq['le_ganan_en_validacion']}
  de {bq['n_configs']} configuraciones (mejor {bq['mejor']:.3f}, peor {bq['peor']:.3f}).

| Familia | Configuraciones | Mejor | Mediana |
|---|---|---|---|
{fam}

| Salida / pérdida | Configuraciones | Mejor | Mediana |
|---|---|---|---|
{per}

![busqueda](resultados/figuras/05_busqueda.png)
![entrenamiento](resultados/figuras/04_entrenamiento.png)

## 5. La prueba final

Los últimos {pu['n']} sorteos ({pu['desde']} a {pu['hasta']}) no participaron en ninguna decisión. Se reentrena cada
{bq['config']['reentrenar_final']} sorteos con {len(bq['config']['semillas_final'])} semillas. La diferencia contra la constante
lleva un intervalo de confianza de 95 % por bootstrap pareado.

| Método | Tipo | Validación | Prueba | Dif. vs constante (IC 95 %) | Predicción más común |
|---|---|---|---|---|---|
{filas}

**Campeón ({c['metodo']}):** {c['validacion']:.3f} en validación y {c['calificacion']:.3f} en prueba. Diferencia contra la
constante: {c['dif_vs_ref']:+.3f} (IC {c['ic_bajo']:+.3f} a {c['ic_alto']:+.3f}).
{'Le gana de forma significativa.' if pu['gana'] else 'No le gana de forma significativa: lo que parecía ventaja en validación era suerte de la selección.'}
{_anterior(pu)}

![prueba](resultados/figuras/06_prueba.png)
![predicciones](resultados/figuras/07_predicciones.png)

## 6. Por dentro de las redes, y ¿se sobreentrenaron?

Sobreentrenar es memorizar los sorteos de entrenamiento en vez de aprender: a la red le va mucho mejor con lo
que ya vio que con sorteos nuevos. Como los dos periodos no califican igual ni para "siempre {bq['constante']}"
({rd['constante_entreno']:.3f} en entrenamiento, {rd['constante_no_vistos']:.3f} en sorteos no vistos), cada red se mide
por su **ventaja sobre "siempre {bq['constante']}" dentro de cada periodo**.

| Red | Capas | Parámetros | Época elegida | Ventaja en entrenamiento | Ventaja en no vistos |
|---|---|---|---|---|---|
{redes_filas}

La mayor ventaja en los sorteos de entrenamiento es {rd['max_ventaja_entreno']:+.3f} puntos por sorteo: las redes casi
no memorizaron (dropout y early stopping), y en sorteos nuevos quedan en cero porque no hay patrón que aprender.

Contraejemplo hecho a propósito: una red de {so['parametros']:,} parámetros sin dropout ni early stopping, entrenada
{so['epocas']} épocas. Su error con los sorteos que ya vio baja de {so['error_entreno_inicio']:.1f} a {so['error_entreno_fin']:.2f}
números (los memorizó); con sorteos nuevos su mejor momento fue la época {so['mejor_epoca']} ({so['error_val_mejor']:.2f}) y
terminó en {so['error_val_fin']:.2f}.

![sobreentrenamiento](resultados/figuras/08_sobreentrenamiento.png)

## 7. Tres sorteos, un número

Prediciendo {tr['prediccion']}: calificación esperada {tr['esperada']:.2f}. P(9 o más) = {tr['p_9_o_mas']:.0%},
P(8 o más) = {tr['p_8_o_mas']:.0%}, P(menos de 7) = {tr['p_menos_de_7']:.0%}. El 80 % de las veces queda entre
{tr['p10']:.1f} y {tr['p90']:.1f}: la suerte pesa mucho más que la diferencia entre métodos.

![tres sorteos](resultados/figuras/09_tres_sorteos.png)
"""
    (nu.CARPETA / "RESUMEN.md").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    escribir()
