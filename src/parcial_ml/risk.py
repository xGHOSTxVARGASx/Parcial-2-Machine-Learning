"""
risk.py
-------
Riesgo empirico con perdida cuadratica, implementado como funcion propia
(el enunciado no permite delegarlo a scikit-learn):

    R_S(h) = (1/n) * sum_i ( y_i - h(x_i) )^2
"""

import numpy as np

from .models import predecir


def riesgo_cuadratico(y_real, y_pred):
    """Riesgo empirico cuadratico a partir de valores reales y predichos."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_real - y_pred) ** 2))


def riesgo_modelo(modelo, x, y):
    """Riesgo empirico de un modelo ajustado sobre el conjunto (x, y)."""
    return riesgo_cuadratico(y, predecir(modelo, x))
