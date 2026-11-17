# Python-Data-Analyst

# 📊 Análisis de Estrategias de Precios y Demanda Comercial con Python y Machine Learning

![Python](https://shields.io)
![Pandas](https://shields.io)
![NumPy](https://shields.io)
![Scikit-Learn](https://shields.io)
![Matplotlib](https://shields.io)

Proyecto final desarrollado para el curso **Python for Data Analytics** en el **Instituto de Educación Superior Privado IDAT** (Lima, Perú - 2026). El repositorio contiene el flujo completo de ciencia de datos: limpieza, estadística descriptiva e inferencial, visualización comercial y modelado predictivo orientado a optimizar estrategias de precios en el sector retail (*Retail Supermarket S.A.*).

---

## 🎯 Contexto y Objetivos del Negocio

### Problemática
La empresa tomaba decisiones sobre promociones y descuentos basándose únicamente en la experiencia y observación del mercado. Esto generaba riesgo de descuentos excesivos (pérdida de rentabilidad) o promociones insuficientes, sumado a la incertidumbre sobre la respuesta de la demanda en sus **76 sucursales**.

### Objetivos
* **General:** Identificar la relación cuantitativa entre los precios de venta y la demanda para respaldar la toma de decisiones comerciales.
* **Específicos:**
  * Analizar el impacto de las variaciones de precio en las unidades vendidas (`Units Sold`).
  * Identificar las tiendas con mejor desempeño comercial.
  * Evaluar el efecto de la brecha de descuento (`Price_Gap`) sobre la rotación de productos.
  * Construir un modelo predictivo para simular escenarios de precios.

---

## 🗂️ Estructura del Dataset y Procesamiento

El análisis parte de una base de datos histórica (`dataset.csv`) con **150,150 registros** y 5 variables principales (`ID`, `Store ID`, `Base Price`, `Total Price`, `Units Sold`).

1. **Limpieza de Datos:** Mediante programación orientada a objetos (clase `AnalizadorRetail`), se verificó la ausencia de duplicados y se depuró 1 registro nulo en `Total Price`, obteniendo **150,149 registros válidos**.
2. **Ingeniería de Características:**
   * `Price_Gap`: Diferencia entre el precio regular y el precio final (`Base Price - Total Price`).
   * `Tiene_Descuento`: Variable categórica que segmenta las transacciones promocionales frente a las de precio regular.
3. **Detección de Valores Atípicos:** Aplicando el método del rango intercuartílico (IQR), se fijó un límite superior de **125.00 unidades**, identificando **10,706 outliers** asociados a picos extraordinarios de demanda.

---

## 📈 Hallazgos Estadísticos y Visuales

### Estadística Descriptiva e Inferencial
* **Comportamiento General:** Media de **51.67 unidades** vendidas por operación (IC 95%: `51.37 - 51.98`), mediana de **35.00 unidades** y desviación estándar de **60.21 unidades**.
* **Prueba de Hipótesis (t-Student):** Al comparar las ventas **Con Descuento** (promedio de **84.31 unidades**) frente a **Precio Regular** (promedio de **40.86 unidades**), se obtuvo un estadístico `t = 85.7142` y un valor `p < 0.0001`. **Se rechaza la hipótesis nula**, comprobando estadísticamente que aplicar descuentos duplica el volumen de ventas.

### Insights del Tablero Comercial (`tablero_resultados.png`)
* **Elasticidad al Precio:** Los mayores picos de demanda (superiores a 1,000 unidades) ocurren cuando el precio final se sitúa por debajo de **S/ 200.00**.
* **Rango Operativo:** La mayor concentración de transacciones se ubica entre **S/ 90.00 y S/ 235.00**.
* **Participación por Estrategia:** El **40.6%** de las unidades totales se comercializan bajo descuento, mientras que el **59.4%** se venden a precio regular.
* **Top 5 Sucursales Líderes (de 76 analizadas):**
  1. **Tienda 8023:** 296,478 unidades
  2. **Tienda 9845:** 261,410 unidades
  3. **Tienda 9823:** 233,846 unidades
  4. **Tienda 9112:** 218,127 unidades
  5. **Tienda 9613:** 205,710 unidades

---

## 🤖 Modelo Predictivo de Demanda (Machine Learning)

Se entrenó un modelo de **Regresión Lineal Múltiple** (`scikit-learn`) dividiendo el dataset en **80% entrenamiento** (120,119 registros) y **20% prueba** (30,030 registros).

* **Métricas de Evaluación:** `R² = 0.1488` | `MAE = 32.49 unidades`.
* **Coeficientes clave:** `Total Price (-0.3000)` confirma la relación inversa con el precio final, mientras que `Price_Gap (+0.4442)` demuestra el impulso positivo de ampliar la brecha de descuento.

### Simulación de Escenarios Comerciales

| Escenario Comercial | Base Price (S/) | Total Price (S/) | Descuento (`Price_Gap`) | Demanda Estimada |
| :--- | :---: | :---: | :---: | :---: |
| **Escenario 1:** Promoción agresiva | 120.00 | 95.00 | S/ 25.00 | **76 unidades** |
| **Escenario 2:** Precio regular medio | 150.00 | 150.00 | S/ 0.00 | **53 unidades** |
| **Escenario 3:** Producto alto con oferta | 250.00 | 220.00 | S/ 30.00 | **60 unidades** |

---

## 💡 Recomendaciones Comerciales

1. **Estrategia de Precios:** Fijar precios promocionales en el rango competitivo de **S/ 95.00 a S/ 150.00** para maximizar la rotación sin sacrificar margen en productos de ticket alto.
2. **Gestión de Inventario:** Priorizar el abastecimiento de stock promocional en el **Top 5 de tiendas líderes** (`8023`, `9845`, `9823`, `9112` y `9613`).

---

## 📁 Archivos del Repositorio

* `proyecto_final.py`: Script principal con la clase `AnalizadorRetail`, pruebas estadísticas y modelo de Machine Learning.
* `dataset.csv`: Base de datos histórica original.
* `ventas_por_tienda.csv`: Ranking exportado con el total de unidades vendidas por sucursal.
* `tablero_resultados.png`: Dashboard visual generado con Matplotlib.

---

## 👤 Autor y Créditos

* **Autor:** Robert Nilton Rojas Santos
* **Docente:** Ing. Henry Pablo Vega Ayala
* **Institución:** Instituto de Educación Superior Privado IDAT — Lima, Perú (2026)
