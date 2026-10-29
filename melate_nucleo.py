"""Código compartido: datos, calificación, modelos de Deep Learning, métodos estadísticos y la prueba
hacia adelante. Lo usan `experimentos.py` (la búsqueda grande) y `melate_r1.py` (el notebook)."""
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CARPETA = Path(__file__).resolve().parent
CSV_ORIGINAL = CARPETA / "Melate.csv"
CSV_ACTUALIZADO = CARPETA / "Melate_actualizado.csv"
URL_HISTORICO = "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUA"
SALIDA = CARPETA / "resultados"

N_BOLAS = 56                     # universo actual: 1..56
K_BOLAS = 6                      # números naturales por sorteo (R1..R6); R7 es el adicional
R1_MAX = N_BOLAS - K_BOLAS + 1   # el menor de 6 bolas nunca pasa de 51
COLS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]

# Partición fija de la época actual (en sorteos objetivo, contando desde el final):
TAM_PRUEBA = 500       # prueba final: no se mira hasta elegir el campeón
TAM_VALIDACION = 500   # aquí compite la búsqueda de configuraciones
VENTANA_MAX = 50       # todas las configuraciones predicen los mismos sorteos objetivo


def calificacion(real, prediccion):
    """La función del examen: 10 - 0.3 * |real - predicción|."""
    return 10 - 0.3 * np.abs(np.asarray(real) - np.asarray(prediccion))


# ----------------------------------------------------------------------------------------------- datos
def leer(ruta):
    df = pd.read_csv(ruta)
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%d/%m/%Y")
    return df.sort_values("CONCURSO").reset_index(drop=True)


def actualizar_datos(descargar=True):
    """Descarga el histórico oficial. Melate.csv nunca se modifica; la descarga se guarda aparte y solo
    se usa si conserva idénticos todos los sorteos del original."""
    original = leer(CSV_ORIGINAL)
    if not descargar:
        if CSV_ACTUALIZADO.exists():
            previo = leer(CSV_ACTUALIZADO)
            if _coincide(original, previo):
                return previo, f"descarga previa ({len(previo) - len(original)} sorteos nuevos)"
        return original, "original (sin descarga)"
    temporal = CSV_ACTUALIZADO.with_suffix(".tmp")
    try:
        peticion = urllib.request.Request(URL_HISTORICO, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(peticion, timeout=60) as r:
            temporal.write_bytes(r.read())
        nuevo = leer(temporal)
    except Exception as e:  # sin red o formato cambiado: seguimos sin la descarga
        print(f"No se pudo descargar ({type(e).__name__})")
        temporal.unlink(missing_ok=True)
        return actualizar_datos(descargar=False)
    if not _coincide(original, nuevo):
        temporal.unlink()
        print("La descarga no coincide con Melate.csv en sorteos ya publicados; se descarta")
        return original, "original (descarga inconsistente)"
    temporal.replace(CSV_ACTUALIZADO)
    return nuevo, f"descarga oficial ({len(nuevo) - len(original)} sorteos nuevos)"


def _coincide(original, nuevo):
    comunes = original.merge(nuevo, on="CONCURSO", suffixes=("", "_n"))
    return len(comunes) == len(original) and all(
        (comunes[c] == comunes[c + "_n"]).all() for c in COLS + ["FECHA"])


def validar(datos):
    v = {
        "producto_unico": bool(datos["NPRODUCTO"].nunique() == 1),
        "concursos_consecutivos": bool((datos["CONCURSO"].diff().dropna() == 1).all()),
        "concurso_inicia_en_1": bool(datos["CONCURSO"].iloc[0] == 1),
        "fechas_no_decrecientes": bool(datos["FECHA"].is_monotonic_increasing),  # dos pares de 1997 comparten fecha
        "r1_a_r6_ordenados": bool((np.diff(datos[COLS[:6]].to_numpy(), axis=1) > 0).all()),
        "r7_distinto_de_r1_r6": bool((datos[COLS[:6]].to_numpy() != datos[["R7"]].to_numpy()).all()),
    }
    assert all(v.values()), v
    return v


def inicio_era(datos):
    """Primer concurso con un número mayor que 51: desde ahí el universo es 1..56."""
    return int(datos.loc[datos[COLS].max(axis=1) > 51, "CONCURSO"].iloc[0])


def cargar_era(descargar=False):
    datos, fuente = actualizar_datos(descargar)
    validar(datos)
    era = datos[datos.CONCURSO >= inicio_era(datos)].reset_index(drop=True)
    return datos, era, fuente


def particion(era):
    """Índices (en la serie de la época) de los sorteos objetivo de validación y de prueba."""
    n = len(era)
    prueba = np.arange(n - TAM_PRUEBA, n)
    validacion = np.arange(n - TAM_PRUEBA - TAM_VALIDACION, n - TAM_PRUEBA)
    assert validacion[0] >= VENTANA_MAX + 300, "muy pocos sorteos para entrenar"
    return validacion, prueba


# --------------------------------------------------------------------------------------- entradas
def matriz_entrada(era, entrada):
    """Una fila por sorteo: 'escalada' = R1..R6 en 0-1 (6), 'multihot' = qué bolas salieron (56),
    'ambas' = las dos (62)."""
    nums = era[COLS[:6]].to_numpy()
    escalada = ((nums - 1) / (N_BOLAS - 1)).astype(np.float32)
    multihot = np.zeros((len(era), N_BOLAS), dtype=np.float32)
    np.put_along_axis(multihot, nums - 1, 1.0, axis=1)
    return {"escalada": escalada, "multihot": multihot,
            "ambas": np.concatenate([escalada, multihot], axis=1)}[entrada]


def ventanas(m, objetivos, k):
    """X[i] = filas objetivos[i]-k .. objetivos[i]-1: solo sorteos ANTERIORES al que se predice."""
    return np.stack([m[t - k:t] for t in objetivos])


# ---------------------------------------------------------------------------------------- modelos
class MLP(nn.Module):
    def __init__(self, k, d, salidas, oculto=64, dropout=0.2, capas=2):
        super().__init__()
        partes, ancho = [nn.Flatten()], k * d
        for _ in range(capas):
            partes += [nn.Linear(ancho, oculto), nn.ReLU(), nn.Dropout(dropout)]
            ancho = oculto
        self.red = nn.Sequential(*partes, nn.Linear(ancho, salidas))

    def forward(self, x):
        return self.red(x)


class Recurrente(nn.Module):
    def __init__(self, k, d, salidas, oculto=32, dropout=0.2, capas=1, celda="lstm"):
        super().__init__()
        tipo = nn.LSTM if celda == "lstm" else nn.GRU
        self.rnn = tipo(d, oculto, num_layers=capas, batch_first=True, dropout=dropout if capas > 1 else 0)
        self.cabeza = nn.Sequential(nn.Dropout(dropout), nn.Linear(oculto, salidas))

    def forward(self, x):
        salida, _ = self.rnn(x)
        return self.cabeza(salida[:, -1])


class CNN(nn.Module):
    """Convoluciones 1D sobre el eje del tiempo (estilo TCN pequeño)."""
    def __init__(self, k, d, salidas, oculto=32, dropout=0.2, capas=2):
        super().__init__()
        partes, ancho = [], d
        for i in range(capas):
            partes += [nn.Conv1d(ancho, oculto, kernel_size=3, padding=2 ** i, dilation=2 ** i),
                       nn.ReLU(), nn.Dropout(dropout)]
            ancho = oculto
        self.conv = nn.Sequential(*partes)
        self.cabeza = nn.Linear(oculto, salidas)

    def forward(self, x):
        return self.cabeza(self.conv(x.transpose(1, 2)).mean(-1))


class Transformer(nn.Module):
    def __init__(self, k, d, salidas, oculto=32, dropout=0.2, capas=1):
        super().__init__()
        self.entrada = nn.Linear(d, oculto)
        self.posicion = nn.Parameter(torch.zeros(1, k, oculto))
        capa = nn.TransformerEncoderLayer(oculto, nhead=4, dim_feedforward=2 * oculto, dropout=dropout,
                                          batch_first=True)
        self.encoder = nn.TransformerEncoder(capa, num_layers=capas, enable_nested_tensor=False)
        self.cabeza = nn.Linear(oculto, salidas)

    def forward(self, x):
        return self.cabeza(self.encoder(self.entrada(x) + self.posicion)[:, -1])


FAMILIAS = {
    "MLP": MLP,
    "LSTM": lambda k, d, s, **h: Recurrente(k, d, s, celda="lstm", **h),
    "GRU": lambda k, d, s, **h: Recurrente(k, d, s, celda="gru", **h),
    "CNN": CNN,
    "Transformer": Transformer,
}
PERDIDAS = ("l1", "mse", "clase")


def mediana_de_pmf(p):
    return np.argmax(np.cumsum(p, axis=-1) >= 0.5, axis=-1) + 1


def entrenar(cfg, X, y, semilla, epocas=300, paciencia=20, lote=128, historial=False):
    """Entrena una red con validación temporal (último 15 % del entrenamiento) y early stopping.
    Devuelve una función que da, por sorteo, el valor continuo (regresión) o la distribución 1..51."""
    torch.manual_seed(semilla)
    clase = cfg["perdida"] == "clase"
    red = FAMILIAS[cfg["familia"]](X.shape[1], X.shape[2], R1_MAX if clase else 1,
                                   oculto=cfg["oculto"], dropout=cfg["dropout"], capas=cfg["capas"])
    opt = torch.optim.Adam(red.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if clase:
        objetivo = torch.tensor(y - 1, dtype=torch.long)
        perdida = nn.CrossEntropyLoss()
    else:
        objetivo = torch.tensor(y / N_BOLAS, dtype=torch.float32)[:, None]
        perdida = nn.L1Loss() if cfg["perdida"] == "l1" else nn.MSELoss()
    Xt = torch.tensor(X)
    corte = int(len(X) * 0.85)
    mejor, mejor_estado, sin_mejora, hist = math.inf, None, 0, []
    gen = torch.Generator().manual_seed(semilla)
    for _ in range(epocas):
        red.train()
        orden = torch.randperm(corte, generator=gen)
        suma = 0.0
        for i in range(0, corte, lote):
            b = orden[i:i + lote]
            opt.zero_grad()
            valor = perdida(red(Xt[b]), objetivo[b])
            valor.backward()
            opt.step()
            suma += valor.item() * len(b)
        red.eval()
        with torch.no_grad():
            val = perdida(red(Xt[corte:]), objetivo[corte:]).item()
        hist.append((suma / corte, val))
        if val < mejor - 1e-6:
            mejor, sin_mejora = val, 0
            mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if mejor_estado is not None:   # None solo si la pérdida de validación fue NaN desde la primera época
        red.load_state_dict(mejor_estado)
    red.eval()

    def predecir(Xn):
        with torch.no_grad():
            s = red(torch.tensor(Xn))
        return torch.softmax(s, -1).numpy() if clase else s[:, 0].numpy() * N_BOLAS

    return (predecir, hist) if historial else predecir


def a_entero(cfg, salidas_semillas):
    """Promedia semillas y da la predicción entera que se enviaría."""
    prom = np.mean(salidas_semillas, axis=0)
    if cfg["perdida"] == "clase":
        return mediana_de_pmf(prom)
    return np.clip(np.rint(prom), 1, R1_MAX).astype(int)


def walk_forward(cfg, era, objetivos, reentrenar_cada, semillas, devolver_crudo=False):
    """Para cada bloque de sorteos objetivo: entrena con TODOS los sorteos anteriores al bloque
    (desde VENTANA_MAX, para que toda configuración vea los mismos objetivos) y predice el bloque."""
    m = matriz_entrada(era, cfg["entrada"])
    y_todo = era.R1.to_numpy()
    k = cfg["ventana"]
    preds, crudos = [], []
    for b in range(0, len(objetivos), reentrenar_cada):
        bloque = objetivos[b:b + reentrenar_cada]
        entreno = np.arange(VENTANA_MAX, bloque[0])
        X, y = ventanas(m, entreno, k), y_todo[entreno]
        Xb = ventanas(m, bloque, k)
        salidas = [entrenar(cfg, X, y, s)(Xb) for s in semillas]
        preds.append(a_entero(cfg, salidas))
        crudos.append(np.mean(salidas, axis=0))
    preds = np.concatenate(preds)
    return (preds, np.concatenate(crudos)) if devolver_crudo else preds


def predecir_siguiente(cfg, era, semillas):
    """Entrena con toda la época y predice el sorteo que sigue al último publicado."""
    m = matriz_entrada(era, cfg["entrada"])
    k = cfg["ventana"]
    entreno = np.arange(VENTANA_MAX, len(era))
    X, y = ventanas(m, entreno, k), era.R1.to_numpy()[entreno]
    salidas = [entrenar(cfg, X, y, s)(m[None, -k:]) for s in semillas]
    return int(a_entero(cfg, salidas)[0]), np.mean(salidas, axis=0)[0]


def nombre_cfg(cfg):
    return (f"{cfg['familia']}-{cfg['perdida'].upper()} k={cfg['ventana']} {cfg['entrada']} "
            f"h={cfg['oculto']}x{cfg['capas']} do={cfg['dropout']} lr={cfg['lr']} wd={cfg['weight_decay']}")


# ------------------------------------------------------------------------- métodos estadísticos
def pmf_teorica():
    v = np.arange(1, R1_MAX + 1)
    return np.array([math.comb(N_BOLAS - k, K_BOLAS - 1) for k in v]) / math.comb(N_BOLAS, K_BOLAS)


def _mediana_simulada_ponderada(pesos, rng, n=4000):
    """Mediana de R1 si las bolas salieran con probabilidad proporcional a `pesos` (sin reemplazo)."""
    llaves = rng.random((n, N_BOLAS)) ** (1.0 / pesos)      # muestreo ponderado sin reemplazo
    r1 = np.sort(np.argsort(-llaves, axis=1)[:, :K_BOLAS], axis=1)[:, 0] + 1
    return int(np.median(r1))


def metodos_estadisticos(era, objetivos):
    """Predicciones enteras de cada método estadístico para cada sorteo objetivo (solo usa el pasado)."""
    r1 = era.R1.to_numpy()
    nums = era[COLS[:6]].to_numpy()
    pmf = pmf_teorica()
    constante = int(np.argmax(np.cumsum(pmf) >= 0.5) + 1)
    rng = np.random.default_rng(0)
    p = {}
    p[f"Constante {constante} (teoría)"] = np.full(len(objetivos), constante)
    p["Mediana histórica"] = np.array([int(np.median(r1[:t])) for t in objetivos])
    p["Media histórica"] = np.array([int(np.rint(r1[:t].mean())) for t in objetivos])
    p["Último R1"] = r1[objetivos - 1]
    p["Mediana últimos 10"] = np.array([int(np.median(r1[t - 10:t])) for t in objetivos])
    p["Mediana últimos 50"] = np.array([int(np.median(r1[t - 50:t])) for t in objetivos])
    # Cadena de Markov: mediana de R1 condicionada al grupo del R1 anterior (estimada con el pasado)
    grupos = np.digitize(r1, [4, 7, 11])
    markov = []
    for t in objetivos:
        g = grupos[t - 1]
        siguientes = r1[1:t][grupos[:t - 1] == g]
        markov.append(int(np.median(siguientes)) if len(siguientes) else constante)
    p["Cadena de Markov"] = np.array(markov)
    # Números "calientes": frecuencia de cada bola en los últimos 100 sorteos con suavizado de Dirichlet
    calientes = []
    for t in objetivos:
        f = np.bincount(nums[t - 100:t].ravel() - 1, minlength=N_BOLAS) + 10.0
        calientes.append(_mediana_simulada_ponderada(f / f.sum(), rng))
    p["Números calientes (Dirichlet)"] = np.array(calientes)
    return p, constante
