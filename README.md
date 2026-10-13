# Segundo Examen Parcial — Introducción al Machine Learning

**Universidad El Bosque — Programa de Matemáticas y Estadística — 2026-2**

**Integrante:** Daniel Vargas

Experimento computacional sobre regresiones polinomiales (grados 1 a 15):
complejidad y generalización, sensibilidad a la muestra (sesgo–varianza) y
selección del modelo, usando la pérdida cuadrática y el riesgo empírico.

## Estructura del repositorio

```
segundo_parcial/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- data/                    train_A, train_B, train_C y validation (.csv)
|-- src/parcial_ml/          paquete local con el codigo reutilizable
|   |-- data.py              carga de los conjuntos de datos
|   |-- models.py            ajuste y prediccion de polinomios
|   |-- risk.py              riesgo empirico cuadratico (funcion propia)
|   |-- analysis.py          recorrido de grados, h media, V_d y seleccion
|   '-- plots.py             funciones de graficacion
|-- notebooks/
|   '-- parcial.ipynb        ejercicios 1, 2 y 3 ejecutados, con respuestas
'-- resultados/              figuras generadas por el cuaderno
```

El cuaderno no redefine la carga de datos, el ajuste, el riesgo, la
sensibilidad ni la selección: importa todo desde `parcial_ml`. El archivo
`validation.csv` se usa únicamente para evaluar los modelos, nunca para
ajustarlos.

## Instalación

Con Python 3.9+ instalado, desde la raíz de `segundo_parcial/`:

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
```

El último comando instala el paquete local `parcial_ml` (definido en
`pyproject.toml`, código en `src/parcial_ml/`) en modo editable, de modo que
el cuaderno pueda hacer `from parcial_ml.data import ...`.

## Reproducir los resultados

Ejecutar el cuaderno completo desde un kernel nuevo (regenera también las
figuras en `resultados/`):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/parcial.ipynb
```

o de forma interactiva con `jupyter notebook notebooks/parcial.ipynb` y
"Kernel → Restart & Run All".

Figuras generadas en `resultados/`:

- `ej1_riesgo_vs_grado.png`: riesgo de entrenamiento y validación vs. grado.
- `ej2_hipotesis_d2.png`, `ej2_hipotesis_d6.png`, `ej2_hipotesis_d15.png`:
  las tres hipótesis aprendidas (una por muestra) para cada grado.
