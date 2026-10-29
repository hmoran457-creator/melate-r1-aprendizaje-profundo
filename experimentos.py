"""Búsqueda de configuraciones de Deep Learning con un protocolo que no se engaña a sí mismo.

  Etapa 1 (búsqueda): muchas configuraciones compiten en los sorteos de VALIDACIÓN.
  Etapa 2 (campeón): la mejor configuración de validación (o el ensamble de las 5 mejores, lo que gane
                     en validación) se evalúa UNA vez en los sorteos de PRUEBA, que nadie había visto.
                     Para comparar, también se evalúan ahí la mejor de cada familia y los métodos estadísticos.
  Etapa 3 (predicción): los finalistas se entrenan con toda la época y predicen el siguiente sorteo.

Uso:  python experimentos.py [--configs 120] [--procesos 3] [--sin-descarga]
Cada configuración terminada se agrega a resultados/busqueda.jsonl, así que si se interrumpe se puede
volver a correr y continúa donde se quedó.
"""
import argparse
import itertools
import json
import multiprocessing as mp
import random
import time

import numpy as np

import melate_nucleo as nu

PARAMETROS = {"semillas_busqueda": [0, 1], "reentrenar_busqueda": 250,
              "semillas_final": [0, 1, 2], "reentrenar_final": 100}
# --rapido: prueba de humo de todo el flujo en pocos minutos, en una carpeta aparte
PARAMETROS_RAPIDO = {"semillas_busqueda": [0], "reentrenar_busqueda": 500,
                     "semillas_final": [0], "reentrenar_final": 500}

ESPACIO = {
    "familia": list(nu.FAMILIAS), "perdida": ["l1", "clase"], "ventana": [5, 10, 20, 50],
    "entrada": ["escalada", "multihot", "ambas"], "oculto": [16, 32, 64, 128], "capas": [1, 2],
    "dropout": [0.0, 0.2, 0.4], "lr": [1e-3, 3e-3], "weight_decay": [0.0, 1e-4, 1e-3],
}
BASE = {"entrada": "escalada", "oculto": 64, "capas": 2, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}


def configuraciones(n):
    """Las configuraciones 'de libro' (cada familia x pérdida, incluida MSE) más una muestra aleatoria."""
    fijas = [dict(BASE, familia=f, perdida=p, ventana=10, oculto=64 if f == "MLP" else 32,
                  capas=2 if f in ("MLP", "CNN") else 1)
             for f, p in itertools.product(nu.FAMILIAS, nu.PERDIDAS)]
    rng = random.Random(2026)
    vistas = {json.dumps(c, sort_keys=True) for c in fijas}
    azar = []
    while len(fijas) + len(azar) < n:
        c = {k: rng.choice(v) for k, v in ESPACIO.items()}
        clave = json.dumps(c, sort_keys=True)
        if clave not in vistas:
            vistas.add(clave)
            azar.append(c)
    return [dict(c, id=i) for i, c in enumerate(fijas + azar)]


_ERA = None


def _iniciar_trabajador(hilos):
    global _ERA
    import torch
    torch.set_num_threads(hilos)
    _, _ERA, _ = nu.cargar_era(descargar=False)


def _evaluar_validacion(tarea):
    cfg, p = tarea
    t0 = time.time()
    validacion, _ = nu.particion(_ERA)
    pred = nu.walk_forward(cfg, _ERA, validacion, p["reentrenar_busqueda"], p["semillas_busqueda"])
    y = _ERA.R1.to_numpy()[validacion]
    return dict(cfg, nombre=nu.nombre_cfg(cfg), calificacion=float(nu.calificacion(y, pred).mean()),
                error_abs=float(np.abs(y - pred).mean()), pred=pred.tolist(), segundos=round(time.time() - t0, 1),
                ultimo_concurso=_version())


def _version():
    """Último concurso de los datos que leyó este proceso: cada resultado lo guarda para no mezclar versiones."""
    return int(_ERA.CONCURSO.iloc[-1])


def _evaluar_prueba(tarea):
    cfg, p = tarea
    _, prueba = nu.particion(_ERA)
    return cfg["id"], _version(), nu.walk_forward(cfg, _ERA, prueba, p["reentrenar_final"], p["semillas_final"]).tolist()


def _siguiente(tarea):
    cfg, p = tarea
    valor, crudo = nu.predecir_siguiente(cfg, _ERA, p["semillas_final"])
    return cfg["id"], _version(), valor, np.asarray(crudo).tolist()


def _comprobar_version(resultados, version):
    otras = {v for _, v, *_ in resultados if v != version}
    assert not otras, f"un proceso leyó datos hasta el concurso {otras}, el principal hasta {version}"


def leer_busqueda(ruta):
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def gbm(era, objetivos, reentrenar_cada):
    """Comparador de ML clásico: gradient boosting con pérdida absoluta sobre los últimos 10 sorteos."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = nu.matriz_entrada(era, "escalada")
    y = era.R1.to_numpy()
    preds = []
    for b in range(0, len(objetivos), reentrenar_cada):
        bloque = objetivos[b:b + reentrenar_cada]
        entreno = np.arange(nu.VENTANA_MAX, bloque[0])
        modelo = HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.05,
                                               early_stopping=True, validation_fraction=0.15, random_state=0)
        modelo.fit(nu.ventanas(m, entreno, 10).reshape(len(entreno), -1), y[entreno])
        p = modelo.predict(nu.ventanas(m, bloque, 10).reshape(len(bloque), -1))
        preds.append(np.clip(np.rint(p), 1, nu.R1_MAX).astype(int))
    return np.concatenate(preds)


def actualizar_prediccion(ruta, era, fuente, P, a):
    """Vuelve a entrenar a los finalistas con el histórico más reciente y predice el siguiente sorteo.
    La búsqueda y la prueba final no se tocan: siguen siendo las de los datos con que se corrieron."""
    fin = json.loads(ruta.read_text(encoding="utf-8"))
    version = int(era.CONCURSO.iloc[-1])
    print(f"Datos: {fuente}; búsqueda y prueba hechas hasta el concurso {fin['ultimo_concurso']}, "
          f"predicción con datos hasta el {version}", flush=True)
    cfgs = [({k: c[k] for k in ["id", *ESPACIO]}, P) for c in fin["finalistas"].values()]
    with mp.Pool(a.procesos, initializer=_iniciar_trabajador, initargs=(a.hilos,)) as pool:
        res = pool.map(_siguiente, cfgs)
    _comprobar_version(res, version)
    fin["siguiente"] = {str(i): {"prediccion": v, "salida": c} for i, _, v, c in res}
    fin["siguiente_ensamble"] = int(np.rint(np.median([fin["siguiente"][str(i)]["prediccion"] for i in fin["top5"]])))
    fin["siguiente_datos_hasta"] = f"{era.FECHA.iloc[-1]:%Y-%m-%d}"
    fin["siguiente_ultimo_concurso"] = version
    ruta.write_text(json.dumps(fin, ensure_ascii=False), encoding="utf-8")
    print({k: v["prediccion"] for k, v in fin["siguiente"].items()}, "ensamble:", fin["siguiente_ensamble"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=120)
    ap.add_argument("--procesos", type=int, default=3)
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--sin-descarga", action="store_true")
    ap.add_argument("--solo-busqueda", action="store_true")
    ap.add_argument("--rapido", action="store_true", help="prueba de humo en resultados/rapido/")
    ap.add_argument("--actualizar-prediccion", action="store_true",
                    help="descarga el histórico y solo vuelve a predecir el siguiente sorteo con los finalistas")
    a = ap.parse_args()
    P = PARAMETROS_RAPIDO if a.rapido else PARAMETROS
    carpeta = nu.SALIDA / "rapido" if a.rapido else nu.SALIDA
    carpeta.mkdir(parents=True, exist_ok=True)
    BUSQUEDA, FINALISTAS = carpeta / "busqueda.jsonl", carpeta / "finalistas.json"

    datos, era, fuente = nu.cargar_era(descargar=not a.sin_descarga)
    version = int(era.CONCURSO.iloc[-1])
    if a.actualizar_prediccion:
        return actualizar_prediccion(FINALISTAS, era, fuente, P, a)
    validacion, prueba = nu.particion(era)
    y = era.R1.to_numpy()
    print(f"Datos: {fuente}; época {len(era)} sorteos hasta el concurso {version}; "
          f"validación {len(validacion)}, prueba {len(prueba)}", flush=True)

    # ---- Etapa 1: búsqueda en validación (reanudable; solo cuenta lo calculado con estos mismos datos) ----
    hechas = {r["id"] for r in leer_busqueda(BUSQUEDA) if r.get("ultimo_concurso") == version}
    pendientes = [(c, P) for c in configuraciones(a.configs) if c["id"] not in hechas]
    print(f"Etapa 1: {len(hechas)} configuraciones ya evaluadas, {len(pendientes)} pendientes", flush=True)
    t0 = time.time()
    with mp.Pool(a.procesos, initializer=_iniciar_trabajador, initargs=(a.hilos,)) as pool:
        with BUSQUEDA.open("a", encoding="utf-8") as f:
            for i, r in enumerate(pool.imap_unordered(_evaluar_validacion, pendientes), 1):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{i}/{len(pendientes)}] {r['calificacion']:.3f}  {r['nombre']}  ({r['segundos']} s; "
                      f"total {time.time() - t0:.0f} s)", flush=True)
    if a.solo_busqueda:
        return

    # ---- Etapa 2: elegir el campeón SOLO con validación, y probarlo una vez ----
    busqueda = sorted((r for r in leer_busqueda(BUSQUEDA) if r.get("ultimo_concurso") == version),
                      key=lambda r: -r["calificacion"])
    estad_val, constante = nu.metodos_estadisticos(era, validacion)
    top5 = busqueda[:5]
    ensamble_val = np.rint(np.median([r["pred"] for r in top5], axis=0)).astype(int)
    cal_ens = float(nu.calificacion(y[validacion], ensamble_val).mean())
    campeon = "ensamble" if cal_ens > busqueda[0]["calificacion"] else busqueda[0]["id"]
    mejor_familia = {}
    for r in busqueda:
        mejor_familia.setdefault(r["familia"], r)
    mse = max((r for r in busqueda if r["perdida"] == "mse"), key=lambda r: r["calificacion"])
    finalistas = {r["id"]: r for r in [*top5, *mejor_familia.values(), mse]}
    print(f"Etapa 2: campeón en validación = {campeon}; {len(finalistas)} finalistas a prueba", flush=True)
    cfgs = [({k: r[k] for k in ["id", *ESPACIO]}, P) for r in finalistas.values()]
    with mp.Pool(a.procesos, initializer=_iniciar_trabajador, initargs=(a.hilos,)) as pool:
        res_prueba, res_sig = pool.map(_evaluar_prueba, cfgs), pool.map(_siguiente, cfgs)
    _comprobar_version(res_prueba + res_sig, version)
    pred_prueba = {i: p for i, _, p in res_prueba}
    siguiente = {i: (v, c) for i, _, v, c in res_sig}
    ensamble_prueba = np.rint(np.median([pred_prueba[r["id"]] for r in top5], axis=0)).astype(int)
    ensamble_sig = int(np.rint(np.median([siguiente[r["id"]][0] for r in top5])))
    estad_prueba, _ = nu.metodos_estadisticos(era, prueba)
    print("  gradient boosting...", flush=True)
    gbm_val, gbm_prueba = gbm(era, validacion, P["reentrenar_busqueda"]), gbm(era, prueba, P["reentrenar_final"])

    salida = {
        "fuente": fuente, "datos_hasta": f"{era.FECHA.iloc[-1]:%Y-%m-%d}", "ultimo_concurso": int(era.CONCURSO.iloc[-1]),
        "n_configs": len(busqueda), "constante": constante,
        "validacion": {"desde": f"{era.FECHA.iloc[validacion[0]]:%Y-%m-%d}", "hasta": f"{era.FECHA.iloc[validacion[-1]]:%Y-%m-%d}",
                       "y": y[validacion].tolist(), "estadisticos": {k: v.tolist() for k, v in estad_val.items()},
                       "gbm": gbm_val.tolist(), "ensamble": ensamble_val.tolist(), "ensamble_calificacion": cal_ens},
        "prueba": {"desde": f"{era.FECHA.iloc[prueba[0]]:%Y-%m-%d}", "hasta": f"{era.FECHA.iloc[prueba[-1]]:%Y-%m-%d}",
                   "y": y[prueba].tolist(), "estadisticos": {k: v.tolist() for k, v in estad_prueba.items()},
                   "gbm": gbm_prueba.tolist(), "ensamble": ensamble_prueba.tolist(),
                   "redes": {str(k): v for k, v in pred_prueba.items()}},
        "campeon": campeon, "top5": [r["id"] for r in top5],
        "mejor_por_familia": {f: r["id"] for f, r in mejor_familia.items()}, "mejor_mse": mse["id"],
        "finalistas": {str(i): {k: r[k] for k in ["id", "nombre", *ESPACIO, "calificacion"]} for i, r in finalistas.items()},
        "siguiente": {str(i): {"prediccion": v, "salida": c} for i, (v, c) in siguiente.items()},
        "siguiente_ensamble": ensamble_sig,
        "siguiente_datos_hasta": f"{era.FECHA.iloc[-1]:%Y-%m-%d}", "siguiente_ultimo_concurso": version,
        "config": dict(P, espacio=ESPACIO, rapido=a.rapido),
        "segundos_total": round(time.time() - t0),
    }
    FINALISTAS.write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(f"Listo en {time.time() - t0:.0f} s -> {FINALISTAS}", flush=True)


if __name__ == "__main__":
    main()
