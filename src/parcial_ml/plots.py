"""
plots.py
--------
Funciones de graficacion reutilizables. Cada figura pedida se genera
llamando a la misma funcion con distintos argumentos (no se repite codigo
cambiando solo el grado).
"""

from pathlib import Path

import matplotlib.pyplot as plt


def graficar_riesgos(tabla, grado_train=None, grado_val=None, ruta=None):
    """
    Ejercicio 1(b): riesgo de entrenamiento y de validacion vs. grado, en una
    misma figura. Si se dan grado_train / grado_val, marca los minimos.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tabla["grado"], tabla["riesgo_train"], "o-", label="Riesgo de entrenamiento (train_A)")
    ax.plot(tabla["grado"], tabla["riesgo_val"], "s-", label="Riesgo de validación")

    if grado_train is not None:
        r = tabla.loc[tabla["grado"] == grado_train, "riesgo_train"].iloc[0]
        ax.plot(grado_train, r, "o", color="C0", markersize=13, mfc="none", mew=2)
    if grado_val is not None:
        r = tabla.loc[tabla["grado"] == grado_val, "riesgo_val"].iloc[0]
        ax.plot(grado_val, r, "s", color="C1", markersize=13, mfc="none", mew=2)

    ax.set_xlabel("Grado del polinomio d")
    ax.set_ylabel("Riesgo empírico cuadrático")
    ax.set_title("Complejidad y generalización")
    ax.set_xticks(list(tabla["grado"]))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _guardar(fig, ruta)
    return fig


def graficar_hipotesis(grado, malla, predicciones, nombres, entrenamiento=None, ruta=None):
    """
    Ejercicio 2(a): las tres hipotesis aprendidas h_{A,d}, h_{B,d}, h_{C,d}
    (una por conjunto de entrenamiento) sobre la malla comun, para un grado.

    predicciones : arreglo (3, m), filas en el orden de `nombres`.
    entrenamiento : dict {nombre: (x, y)} opcional, para mostrar los datos.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, nombre in enumerate(nombres):
        color = f"C{i}"
        ax.plot(malla, predicciones[i], color=color, label=f"ĥ_{nombre[-1]},{grado}")
        if entrenamiento is not None:
            x, y = entrenamiento[nombre]
            ax.scatter(x, y, s=14, color=color, alpha=0.35)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Hipótesis aprendidas con cada muestra — grado d = {grado}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _guardar(fig, ruta)
    return fig


def _guardar(fig, ruta):
    if ruta is not None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
