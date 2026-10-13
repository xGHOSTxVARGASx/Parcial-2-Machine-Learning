"""
analysis.py
-----------
Funciones de analisis del parcial. Aqui vive toda la logica que los tres
ejercicios comparten, de modo que el cuaderno solo importe y presente:

  * Ejercicio 1: recorrer_grados, grado_optimo
  * Ejercicio 2: ajustar_por_conjunto, crear_malla, sensibilidad_por_grado
  * Ejercicio 3: riesgo_validacion_promedio, tabla_seleccion
"""

import numpy as np
import pandas as pd

from .models import ajustar_polinomio, predecir
from .risk import riesgo_modelo

GRADOS_EJ1 = tuple(range(1, 16))      # d = 1, 2, ..., 15
GRADOS_COMPARACION = (2, 6, 15)       # d en {2, 6, 15}
M_MALLA = 300                         # puntos de la malla comun
MALLA_INF, MALLA_SUP = -2.5, 2.5      # extremos de la malla


# ---------------------------------------------------------------------------
# Ejercicio 1
# ---------------------------------------------------------------------------
def recorrer_grados(x_train, y_train, x_val, y_val, grados=GRADOS_EJ1):
    """
    Ajusta un polinomio de cada grado SOLO con (x_train, y_train) y calcula su
    riesgo en entrenamiento y en validacion.

    Retorna un DataFrame con columnas: grado, riesgo_train, riesgo_val.
    """
    filas = []
    for d in grados:
        modelo = ajustar_polinomio(x_train, y_train, d)
        filas.append(
            {
                "grado": d,
                "riesgo_train": riesgo_modelo(modelo, x_train, y_train),
                "riesgo_val": riesgo_modelo(modelo, x_val, y_val),
            }
        )
    return pd.DataFrame(filas)


def grado_optimo(tabla, columna):
    """
    Grado que minimiza `columna` ('riesgo_train' o 'riesgo_val') en la tabla
    de recorrer_grados. Retorna (grado, riesgo_minimo).
    """
    fila = tabla.loc[tabla[columna].idxmin()]
    return int(fila["grado"]), float(fila[columna])


# ---------------------------------------------------------------------------
# Ejercicio 2
# ---------------------------------------------------------------------------
def ajustar_por_conjunto(entrenamiento, grados=GRADOS_COMPARACION):
    """
    Ajusta un modelo para cada (conjunto de entrenamiento, grado).

    Parametros
    ----------
    entrenamiento : dict {nombre: (x, y)}, p. ej. {'train_A': (x, y), ...}

    Retorna un diccionario {(nombre, grado): modelo}, es decir los modelos
    h_{S,d} del enunciado.
    """
    return {
        (nombre, d): ajustar_polinomio(x, y, d)
        for nombre, (x, y) in entrenamiento.items()
        for d in grados
    }


def crear_malla(m=M_MALLA, inferior=MALLA_INF, superior=MALLA_SUP):
    """Malla de m puntos igualmente espaciados en [inferior, superior]."""
    return np.linspace(inferior, superior, m)


def sensibilidad_por_grado(modelos, nombres, grados, malla):
    """
    Evalua, para cada grado, las hipotesis h_{S,d} sobre la malla comun y
    calcula la hipotesis media y la varianza V_d:

        h_bar_d(x_j) = (1/3) * sum_S h_{S,d}(x_j)
        V_d          = (1/m) * sum_j (1/3) * sum_S ( h_{S,d}(x_j) - h_bar_d(x_j) )^2

    Retorna un diccionario {grado: {'predicciones': (3, m), 'h_media': (m,),
    'V': float}}, con las filas de 'predicciones' en el orden de `nombres`.
    """
    resultado = {}
    for d in grados:
        preds = np.vstack([predecir(modelos[(n, d)], malla) for n in nombres])
        h_media = preds.mean(axis=0)
        V = float(np.mean(np.mean((preds - h_media) ** 2, axis=0)))
        resultado[d] = {"predicciones": preds, "h_media": h_media, "V": V}
    return resultado


# ---------------------------------------------------------------------------
# Ejercicio 3
# ---------------------------------------------------------------------------
def riesgo_validacion_promedio(modelos, nombres, grado, x_val, y_val):
    """
    R_val(d) = (1/3) * sum_S R_validation(h_{S,d}).

    Retorna (promedio, lista_de_riesgos_por_conjunto).
    """
    riesgos = [riesgo_modelo(modelos[(n, grado)], x_val, y_val) for n in nombres]
    return float(np.mean(riesgos)), riesgos


def tabla_seleccion(modelos, nombres, grados, x_val, y_val, sensibilidad):
    """
    Tabla con columnas: grado, R_val (riesgo promedio de validacion) y V_d.
    """
    filas = []
    for d in grados:
        r_val, _ = riesgo_validacion_promedio(modelos, nombres, d, x_val, y_val)
        filas.append({"grado": d, "R_val": r_val, "V_d": sensibilidad[d]["V"]})
    return pd.DataFrame(filas)
