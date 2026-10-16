import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import json

def preprocesar_matriz_kmeans(df_maestra):
    """Recibe la matriz de 42 variables explicativas y ejecuta la depuración,
    winsorización, corrección de asimetría, estandarización y filtrado
    iterativo de multicolinealidad por VIF/GVIF"""
    df = df_maestra.copy()

    cols_meta = ["Cliente", "RFM Combinado", "Negocio Principal"]
    cols_meta_presentes = [c for c in cols_meta if c in df.columns]
    df_meta = df[cols_meta_presentes].copy()
    df_num = df.drop(columns=cols_meta_presentes)

    cols_redundancia_exacta = []
    if "RFM Ponderado" in df_num.columns:
        cols_redundancia_exacta.append("RFM Ponderado")
    if "Amplitud de Portafolio" in df_num.columns:
        cols_redundancia_exacta.append("Amplitud de Portafolio")
    if "% Gasto SERVICIOS" in df_num.columns:
        cols_redundancia_exacta.append("% Gasto SERVICIOS")

    if cols_redundancia_exacta:
        df_num = df_num.drop(columns=cols_redundancia_exacta)

    # IMPUTACION ESTRUCTURAL DE NULOS
    if "Intervalo_Promedio_Dias" in df_num.columns:
        p99_5_intervalo = df_num["Intervalo_Promedio_Dias"].quantile(0.995)
        df_num["Intervalo_Promedio_Dias"] = df_num["Intervalo_Promedio_Dias"].fillna(p99_5_intervalo)
        print(f"Imputados NaNs en 'Intervalo_Promedio_Dias' con p99.5: {p99_5_intervalo:.2f} dias")

    # Clasificacion de variables numericas por naturaleza matematica
    cols_binarias = [c for c in df_num.columns if set(df_num[c].dropna().unique()).issubset({0, 1})]
    cols_ordinales = [c for c in df_num.columns if c.startswith("Puntaje")]
    cols_acotadas = [
        c for c in df_num.columns 
        if c.startswith("% Gasto") or c in [
            "Indice IHH", "Indice Concentracion PDV", "Indice Concentracion Asesor"
        ]
    ]
    cols_continuas = [c for c in df_num.columns if c not in cols_binarias + cols_ordinales + cols_acotadas]

    # WINSORIZACION
    print("\nAplicando Winsorización al p99.5 sobre variables continuas:")
    for col in cols_continuas:
        p99_5 = df_num[col].quantile(0.995)
        df_num[col] = np.where(df_num[col] > p99_5, p99_5, df_num[col])

    # FILTRO DE VARIANZA CASI NULA (VARIANCE THRESHOLD)
    vt_bin = VarianceThreshold(threshold=0.01 * 0.99)
    vt_bin.fit(df_num[cols_binarias])
    bin_sobrevivientes = list(df_num[cols_binarias].columns[vt_bin.get_support()])

    cols_resto = cols_continuas + cols_acotadas + cols_ordinales
    vt_rest = VarianceThreshold(threshold=0.0)
    vt_rest.fit(df_num[cols_resto])
    resto_sobrevivientes = list(df_num[cols_resto].columns[vt_rest.get_support()])

    cols_descarte_var = set(df_num.columns) - set(bin_sobrevivientes + resto_sobrevivientes)
    if cols_descarte_var:
        print(f"Eliminadas: {len(cols_descarte_var)} variables por varianza casi nula: {list(cols_descarte_var)}")
    
    df_num = df_num[bin_sobrevivientes + resto_sobrevivientes]
    cols_continuas = [c for c in cols_continuas if c in df_num.columns]
    cols_acotadas = [c for c in cols_acotadas if c in df_num.columns]

    # CORRECCION DE ASIMETRIA (LOG1P / YEO-JOHNSON) Y ESTANDARIZACION
    print("\nEvaluando asimetria (|skew| > 1.0) y aplicando escalamiento Z-Score:")
    scaler = StandardScaler()

    for col in cols_continuas:
        val_skew = skew(df_num[col])
        if abs(val_skew) > 1.0:
            if (df_num[col] < 0).any():
                pt = PowerTransformer(method="yeo-johnson")
                df_num[col] = pt.fit_transform(df_num[[col]])
            else:
                df_num[col] = np.log1p(df_num[col])

    cols_a_escalar = cols_continuas + cols_acotadas
    df_num[cols_a_escalar] = scaler.fit_transform(df_num[cols_a_escalar])

    # MATRIZ DE CORRELACION HIBRIDA
    def calcular_matriz_hibrida(df_c):
        n = df_c.shape[1]
        m_corr = pd.DataFrame(index=df_c.columns, columns=df_c.columns, dtype=float)
        for i in range(n):
            for j in range(i, n):
                c1, c2 = df_c.columns[i], df_c.columns[j]
                if c1 in cols_binarias and c2 in cols_binarias:
                    c_val = df_c[c1].corr(df_c[c2], method="pearson")
                else:
                    c_val, _ = spearmanr(df_c[c1], df_c[c2])
                m_corr.loc[c1, c2] = c_val
                m_corr.loc[c2, c1] = c_val
        return m_corr

    # PRE-FILTRO DE CORRELACION BIVARIADA EXTREMA (|r| > 0.90)
    matriz_corr_inicial = calcular_matriz_hibrida(df_num)
    cols_a_eliminar_corr = set()
    for i in range(len(df_num.columns)):
        for j in range(i + 1, len(df_num.columns)):
            col1 = df_num.columns[i]
            col2 = df_num.columns[j]
            if abs(matriz_corr_inicial.loc[col1, col2]) > 0.90:
                col_a_borrar = col2 if "Puntaje" not in col2 else col1
                cols_a_eliminar_corr.add(col_a_borrar)

    if cols_a_eliminar_corr:
        print(f"Pre-filtro (|r| > 0.90): Eliminadas {len(cols_a_eliminar_corr)} variables: {list(cols_a_eliminar_corr)}")
        df_num = df_num.drop(columns=list(cols_a_eliminar_corr))

    # FILTRADO ITERATIVO BUCLE VIF / GVIF (UMBRAL > 5.0)
    print("\nEjecutando bucle iterativo de depuración por VIF/GVIF:")
    eliminadas_vif = []

    while True:
        X = sm.add_constant(df_num)
        vifs = []
        for i in range(1, X.shape[1]):
            try:
                vif_val = variance_inflation_factor(X.values, i)
            except Exception:
                vif_val = 999.0
            vifs.append(vif_val)

        vif_series = pd.Series(vifs, index=df_num.columns).sort_values(ascending=False)
        max_vif = vif_series.max()
        var_max = vif_series.idxmax()

        if max_vif > 5.0:
            matriz_hibrida = calcular_matriz_hibrida(df_num)
            corrs_var = matriz_hibrida[var_max].drop(var_max)
            pareja_alta = corrs_var[corrs_var.abs() > 0.7]

            if not pareja_alta.empty:
                var_pareja = pareja_alta.abs().idxmax()
                print(f"[VIF={max_vif:.2f}] Colinealidad bivariada entre '{var_max}' y '{var_pareja}' (r={pareja_alta[var_pareja]:.2f}) -> Eliminando '{var_max}'")
            else:
                print(f"[VIF={max_vif:.2f}] Multicolinealidad multivariada en '{var_max}' -> Eliminando variable")

            eliminadas_vif.append(var_max)
            df_num = df_num.drop(columns=[var_max])
        else:
            break

    print(f"\nProcessed: {len(eliminadas_vif)} variables eliminadas por VIF/GVIF.")
    print(f"Matriz resultante lista para K-Means: {df_num.shape[1]} variables numéricas optimas.")

    df_final_preprocesado = pd.concat([df_meta.reset_index(drop=True), df_num.reset_index(drop=True)], axis=1)
    return df_final_preprocesado, df_num

# ==========================================
# EJECUCION PRINCIPAL
# ==========================================
if __name__ == "__main__":
    from pathlib import Path

    CARPETA_SCRIPT = Path(__file__).resolve().parent
    RUTA_PARQUET = (
        CARPETA_SCRIPT / "../Datos/feature_eng/features_init/matriz_features_clientes.parquet"
    ).resolve()

    carpeta_salida = (CARPETA_SCRIPT / "../Datos/feature_eng/features_kmeans").resolve()

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    output_parquet = (carpeta_salida / "matriz_preprocesada_kmeans.parquet")
    output_csv = (carpeta_salida / "matriz_preprocesada_kmeans.csv")
    output_meta = (carpeta_salida / "matriz_preprocesada_kmeans_meta.json")

    print(f"Cargando matriz maestra desde '{RUTA_PARQUET}'...")

    if not RUTA_PARQUET.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {RUTA_PARQUET}"
        )

    df_maestra = pd.read_parquet(RUTA_PARQUET)
    print(f"Matriz de entrada: {df_maestra.shape[0]:,} filas, {df_maestra.shape[1]} columnas.")

    df_completo_preprocesado, X_preprocesado_kmeans = (preprocesar_matriz_kmeans(df_maestra))
    df_completo_preprocesado.to_parquet(output_parquet, index=False)
    df_completo_preprocesado.to_csv(output_csv, sep=";", index=False, encoding="utf-8-sig")
    
    meta_actual = {
        "filas": len(df_completo_preprocesado),
        "columnas": list(df_completo_preprocesado.columns)
    }
    output_meta.write_text(json.dumps(meta_actual, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadatos guardados: {output_meta}")
    print("Exportacion completada correctamente.")

    print("\nExportacion completada")
    print(f"Parquet: {output_parquet}")
    print(f"CSV: {output_csv}")
    print(f"Dimensiones de la matriz completa: {df_completo_preprocesado.shape}")
    print(f"Dimensiones de la matriz para K-Means: {X_preprocesado_kmeans.shape}")
