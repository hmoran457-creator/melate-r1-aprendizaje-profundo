"""Convierte melate_r1.py (celdas marcadas con `# %%`) en melate_r1.ipynb y lo ejecuta para guardar las salidas.

Uso:  python construir_notebook.py           (convierte y ejecuta)
      python construir_notebook.py --sin-ejecutar
"""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

CARPETA = Path(__file__).resolve().parent
FUENTE, DESTINO = CARPETA / "melate_r1.py", CARPETA / "melate_r1.ipynb"


def celdas(texto):
    for bloque in re.split(r"^# %%", texto, flags=re.M)[1:]:
        encabezado, _, cuerpo = bloque.partition("\n")
        if "[markdown]" in encabezado:
            lineas = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in cuerpo.strip().splitlines()]
            yield nbformat.v4.new_markdown_cell("\n".join(lineas))
        else:
            yield nbformat.v4.new_code_cell(cuerpo.strip())


def main():
    nb = nbformat.v4.new_notebook()
    nb.cells = list(celdas(FUENTE.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if "--sin-ejecutar" not in sys.argv:
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(CARPETA)}}).execute()
    nbformat.write(nb, DESTINO)
    print(f"Escrito {DESTINO.name} ({len(nb.cells)} celdas)")


if __name__ == "__main__":
    main()
