import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score



class AnalizadorRetail:
    def __init__(self, ruta_archivo):
        self.ruta_archivo = ruta_archivo
        self.df = None
        self.df_limpio = None

    def cargar_y_explorar(self):
        print("=" * 65)
        print("1. EXPLORACIÓN INICIAL DEL DATASET (5 PASOS)")
        print("=" * 65)
        self.df = pd.read_csv(self.ruta_archivo)
        print(f"Dimensiones (filas, columnas): {self.df.shape}")
        print("\nPrimeros 5 registros:")
        print(self.df.head())
        print("\nInformación de tipos de datos y nulos:")
        print(self.df.info())
        print("\nResumen estadístico inicial:")
        print(self.df.describe().round(2))

    def limpiar_y_transformar(self):
        print("\n" + "=" * 65)
        print("2. TRATAMIENTO, LIMPIEZA Y NUEVAS VARIABLES")
        print("=" * 65)

        df_temp = self.df.drop_duplicates().dropna().copy()

        df_temp["Price_Gap"] = df_temp["Base Price"] - df_temp["Total Price"]

        df_temp["Tiene_Descuento"] = np.where(df_temp["Price_Gap"] > 0, "Con Descuento", "Precio Regular")

        q1 = df_temp["Units Sold"].quantile(0.25)
        q3 = df_temp["Units Sold"].quantile(0.75)
        iqr = q3 - q1
        tope_superior = q3 + 1.5 * iqr
        outliers = df_temp[df_temp["Units Sold"] > tope_superior]

        print(f"Registros válidos tras limpieza: {len(df_temp)}")
        print(f"Tope superior (IQR) para Units Sold: {tope_superior:.2f} unidades")
        print(f"Valores atípicos (outliers) detectados en ventas: {len(outliers)} registros")

        self.df_limpio = df_temp
        return self.df_limpio



def ejecutar_analisis_estadistico(df):
    print("\n" + "=" * 65)
    print("3. ANÁLISIS ESTADÍSTICO E INFERENCIA (SCIPY & NUMPY)")
    print("=" * 65)

    ventas = df["Units Sold"].values
    print(f"Media de unidades vendidas (NumPy): {np.mean(ventas):.2f}")
    print(f"Mediana de unidades vendidas: {df['Units Sold'].median():.2f}")
    print(f"Desviación estándar (NumPy): {np.std(ventas):.2f}")

    ic_95 = st.t.interval(
        0.95,
        len(df["Units Sold"]) - 1,
        loc=df["Units Sold"].mean(),
        scale=st.sem(df["Units Sold"])
    )
    print(f"Intervalo de Confianza (95%) de ventas promedio: ({ic_95[0]:.2f}, {ic_95[1]:.2f})")

    con_desc = df[df["Tiene_Descuento"] == "Con Descuento"]["Units Sold"]
    sin_desc = df[df["Tiene_Descuento"] == "Precio Regular"]["Units Sold"]

    t_stat, p_val = st.ttest_ind(con_desc, sin_desc, equal_var=False)
    print(f"\nPrueba t-Student (Con Descuento vs Precio Regular):")
    print(f"  Promedio Con Descuento: {con_desc.mean():.2f} unidades")
    print(f"  Promedio Precio Regular: {sin_desc.mean():.2f} unidades")
    print(f"  Estadístico t = {t_stat:.4f} | Valor p = {p_val:.4e}")
    if p_val < 0.05:
        print("  Conclusión: Se rechaza H0 (p < 0.05). El descuento impacta significativamente en las ventas.")
    else:
        print("  Conclusión: No se rechaza H0 (p >= 0.05).")

    ventas_tienda = (
        df.groupby("Store ID")["Units Sold"]
        .sum()
        .reset_index()
        .rename(columns={"Units Sold": "Total_Units_Sold"})
        .sort_values(by="Total_Units_Sold", ascending=False)
    )
    ventas_tienda.to_csv("ventas_por_tienda.csv", index=False)
    print("\nTop 5 Tiendas con mayor volumen de ventas (exportado a ventas_por_tienda.csv):")
    print(ventas_tienda.head(5).to_string(index=False))
    return ventas_tienda



def generar_tablero_graficos(df, ventas_tienda):
    print("\n" + "=" * 65)
    print("4. GENERANDO TABLERO DE GRÁFICOS (MATPLOTLIB 2x2)")
    print("=" * 65)

    plt.figure(figsize=(14, 10))
    plt.suptitle("Tablero de Análisis Comercial - Retail Supermarket S.A.", fontsize=14, fontweight="bold")

    plt.subplot(2, 2, 1)
    plt.scatter(df["Total Price"], df["Units Sold"], alpha=0.4, color="steelblue", edgecolors="none")
    plt.title("1. Relación Precio Final vs. Unidades Vendidas")
    plt.xlabel("Total Price (S/)")
    plt.ylabel("Units Sold")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 2, 2)
    top_10 = ventas_tienda.head(10)
    plt.bar(top_10["Store ID"].astype(str), top_10["Total_Units_Sold"], color="teal")
    plt.title("2. Top 10 Tiendas por Volumen de Ventas")
    plt.xlabel("Store ID")
    plt.ylabel("Total Unidades Vendidas")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(2, 2, 3)
    plt.hist(df["Total Price"], bins=15, color="coral", edgecolor="black")
    plt.title("3. Distribución de Precios Finales (Total Price)")
    plt.xlabel("Rango de Precios (S/)")
    plt.ylabel("Frecuencia de Registros")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(2, 2, 4)
    resumen_desc = df.groupby("Tiene_Descuento")["Units Sold"].sum()
    plt.pie(
        resumen_desc.values,
        labels=resumen_desc.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#FFC107"]
    )
    plt.title("4. Participación de Unidades Vendidas por Estrategia de Precio")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("tablero_resultados.png", dpi=150)
    print("Gráfico guardado exitosamente como 'tablero_resultados.png'")
    plt.show()



def entrenar_modelo_ml(df):
    print("\n" + "=" * 65)
    print("5. MODELO PREDICTIVO DE DEMANDA (REGRESIÓN LINEAL)")
    print("=" * 65)

    X = df[["Total Price", "Base Price", "Price_Gap"]]
    y = df["Units Sold"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Intercepto del modelo: {modelo.intercept_:.4f}")
    for col, coef in zip(X.columns, modelo.coef_):
        print(f"Coeficiente ({col}): {coef:.4f}")
    print(f"Coeficiente de determinación R² (Test): {r2:.4f}")
    print(f"Error Absoluto Medio (MAE): {mae:.2f} unidades")

    escenarios = pd.DataFrame({
        "Total Price": [95.0, 150.0, 220.0],
        "Base Price": [120.0, 150.0, 250.0],
        "Price_Gap": [25.0, 0.0, 30.0]
    })
    predicciones = modelo.predict(escenarios)
    escenarios["Demanda_Estimada_Units"] = np.round(predicciones, 0)
    print("\nSimulación de Demanda según Escenarios de Precio:")
    print(escenarios.to_string(index=False))



if __name__ == "__main__":
    archivo_datos = "dataset.csv"
    analizador = AnalizadorRetail(archivo_datos)
    analizador.cargar_y_explorar()
    df_procesado = analizador.limpiar_y_transformar()
    tabla_tiendas = ejecutar_analisis_estadistico(df_procesado)
    generar_tablero_graficos(df_procesado, tabla_tiendas)
    entrenar_modelo_ml(df_procesado)


    #Instalación de Librerías en VSC: pip install numpy pandas matplotlib scipy scikit-learn
    #Detener proceso: Ctrl + C
    #Ejecucuión en Terminal: python proyecto_final.py
