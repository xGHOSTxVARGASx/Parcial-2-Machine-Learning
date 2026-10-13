"""
data.py
-------
Carga de los conjuntos de datos del parcial (train_A, train_B, train_C y
validation). Todos tienen las columnas `x` y `y`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# data.py esta en segundo_parcial/src/parcial_ml/data.py
# parents[0] = parcial_ml, parents[1] = src, parents[2] = segundo_parcial (raiz)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

NOMBRES_ENTRENAMIENTO = ("train_A", "train_B", "train_C")
NOMBRE_VALIDACION = "validation"


def cargar_conjunto(nombre, data_dir=DATA_DIR):
    """
    Carga data/<nombre>.csv y retorna (x, y) como arreglos de numpy.

    Parametros
    ----------
    nombre : str
        'train_A', 'train_B', 'train_C' o 'validation' (sin extension).
    """
    ruta = Path(data_dir) / f"{nombre}.csv"
    df = pd.read_csv(ruta)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return x, y


def cargar_entrenamiento(data_dir=DATA_DIR):
    """
    Carga los tres conjuntos de entrenamiento.

    Retorna un diccionario {'train_A': (x, y), 'train_B': (x, y), 'train_C': (x, y)}.
    """
    return {n: cargar_conjunto(n, data_dir) for n in NOMBRES_ENTRENAMIENTO}


def cargar_validacion(data_dir=DATA_DIR):
    """
    Carga validation.csv. Este conjunto NO debe usarse para ajustar modelos,
    solo para evaluarlos.
    """
    return cargar_conjunto(NOMBRE_VALIDACION, data_dir)
