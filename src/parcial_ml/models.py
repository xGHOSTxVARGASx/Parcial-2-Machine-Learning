"""
models.py
---------
Ajuste y prediccion de regresiones polinomiales de grado d, usando
PolynomialFeatures + LinearRegression de scikit-learn.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def ajustar_polinomio(x, y, grado):
    """
    Ajusta una regresion polinomial de grado `grado` (con intercepto) por
    minimos cuadrados.

    El intercepto lo aporta LinearRegression, por eso PolynomialFeatures se
    usa con include_bias=False (columnas x, x^2, ..., x^grado).

    Retorna el modelo ya ajustado (un Pipeline de scikit-learn).
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    modelo = make_pipeline(
        PolynomialFeatures(degree=grado, include_bias=False),
        LinearRegression(),
    )
    modelo.fit(x, y)
    return modelo


def predecir(modelo, x):
    """Evalua el modelo ajustado en los puntos x (arreglo 1D)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return modelo.predict(x)
