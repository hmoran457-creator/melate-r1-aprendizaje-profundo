"""Inserta resultados/resultados.json en plantilla.html y escribe la presentación.

Salidas:
  ../Presentacion_R1_Melate.html  archivo para abrir con doble clic (con doctype y charset; funciona sin internet,
                                  solo cambian las fuentes)
  artifact.html                   la misma página sin el esqueleto, para publicarla como Artifact
Uso:  python presentacion/ensamblar.py [--rapido]
"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent


def main():
    origen = RAIZ / "resultados" / ("rapido" if "--rapido" in sys.argv else "") / "resultados.json"
    datos = json.loads(origen.read_text(encoding="utf-8"))
    carga = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    # --corta: la versión de 11 diapositivas, con sus propios archivos; sin la opción todo queda igual que antes
    sufijo = "_corta" if "--corta" in sys.argv else ""
    plantilla = (AQUI / f"plantilla{sufijo}.html").read_text(encoding="utf-8")
    assert plantilla.count("/*__DATOS__*/") == 1
    fragmento = plantilla.replace("/*__DATOS__*/", carga)
    (AQUI / f"artifact{sufijo}.html").write_text(fragmento, encoding="utf-8")

    cabeza, cuerpo = fragmento.split("</style>", 1)
    pagina = ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
              f"{cabeza}</style>\n</head>\n<body>\n{cuerpo}\n</body>\n</html>\n")
    assert pagina.find('charset="utf-8"') < 1024
    (RAIZ / f"Presentacion_R1_Melate{sufijo}.html").write_text(pagina, encoding="utf-8")
    print(f"Datos de {origen.relative_to(RAIZ)} ({len(carga):,} bytes) -> Presentacion_R1_Melate{sufijo}.html y presentacion/artifact{sufijo}.html")


if __name__ == "__main__":
    main()
