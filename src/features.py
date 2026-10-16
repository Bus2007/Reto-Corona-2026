import json
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_parquet("Datos/Pre-processed/base_unificada_transaccional_clientes.parquet")

#RFM
def recencia(df):
    df_temp = df.copy()
    df_temp["Fecha Pedido"] = pd.to_datetime(df_temp["Fecha Pedido"])
    
    ultima_fecha_df = df_temp["Fecha Pedido"].max()

    df_recencia = (df_temp.groupby("Cliente")["Fecha Pedido"].max().reset_index())
    df_recencia["Recencia"] = (ultima_fecha_df - df_recencia["Fecha Pedido"]).dt.days
    
    return df_recencia[["Cliente", "Recencia"]]

def frecuencia(df):
    df_frec = (df.groupby("Cliente").agg(
            Frecuencia_Pedidos=(
                "ID Pedido",
                "nunique",
            ),  # Numero de compras 
            Total_Lineas_Compradas=(
                "Producto",
                "count",
            ),  # Total de items comprados
            Total_Unidades_Fisicas=(
                "Cantidad",
                "sum",
            ),  # Suma de volumenes de producto
        )
        .reset_index()
    )

    # Promedio de productos distintos por compra realizada
    df_frec["Items Promedio Por Pedido"] = (df_frec["Total_Lineas_Compradas"] / df_frec["Frecuencia_Pedidos"])
    return df_frec

def valor_monetario(df):
    df_mon = (df.groupby("Cliente").agg(Valor_Monetario_Total=("Valor", "sum"),
            Devoluciones=("Valor",lambda x: x[x < 0].sum())).reset_index())
    df_mon["Devoluciones"] = df_mon["Devoluciones"].abs()
    return df_mon

def matriz_RFM(df):
    df_R = recencia(df)
    df_F = frecuencia(df)
    df_M = valor_monetario(df)
    
    df_RFM = df_R.merge(df_F, how="inner", on="Cliente").merge(df_M, how="inner", on="Cliente")
    df_RFM["Recibo Promedio Por Pedido"] = (df_RFM["Valor_Monetario_Total"] / df_RFM["Frecuencia_Pedidos"])
    
    return df_RFM

def discretizar_RFM(df_RFM):
    df_discretizado = df_RFM.copy()

    df_discretizado["Puntaje R"] = pd.qcut(df_discretizado["Recencia"], q=5, labels=[5,4,3,2,1]).astype(int)
    
    condiciones = [
        df_discretizado["Frecuencia_Pedidos"] == 1,
        df_discretizado["Frecuencia_Pedidos"] == 2,
        df_discretizado["Frecuencia_Pedidos"] == 3,
        df_discretizado["Frecuencia_Pedidos"].isin([4, 5])
    ]
    resultados = [1, 2, 3, 4]
    df_discretizado["Puntaje F"] = np.select(condiciones, resultados, default=5)

    df_discretizado["Puntaje M"] = pd.qcut(df_discretizado["Valor_Monetario_Total"].rank(method="first"),
                                           q=5, labels=[1,2,3,4,5]).astype(int)
    
    df_discretizado["RFM Combinado"] = (df_discretizado["Puntaje R"].astype(str) + 
                                        df_discretizado["Puntaje F"].astype(str) + 
                                        df_discretizado["Puntaje M"].astype(str))

    df_discretizado["RFM Ponderado"] = ((df_discretizado["Puntaje R"]*0.4)+(df_discretizado["Puntaje F"]*0.2)
                                        +(df_discretizado["Puntaje M"]*0.4))
    #print(df_discretizado.head(5))
    return df_discretizado


# Afinifad de producto
def share_of_wallet(df):
    negocios = df["Negocio"].unique()
    gasto_negocio = (df.groupby(["Cliente", "Negocio"])["Valor"].sum().unstack(fill_value=0))
    for negocio in negocios:
        if negocio not in gasto_negocio.columns:
            gasto_negocio[negocio] = 0.0
            
    gasto_negocio=gasto_negocio[negocios]
    gasto_total = gasto_negocio.sum(axis=1)
    sow = gasto_negocio.div(gasto_total, axis=0).fillna(0)
    sow.columns = [f"% Gasto {col}" for col in sow.columns]
    sow["Negocio Principal"] = gasto_negocio.idxmax(axis=1)
    
    #print(sow.sample(5))
    return sow.reset_index()

def indice_IHH(df_sow):
    columnas_pct = [col for col in df_sow.columns if col.startswith("% Gasto ")]
    df_ihh = pd.DataFrame()
    df_ihh["Cliente"] = df_sow["Cliente"]
    df_ihh["Indice IHH"] = (df_sow[columnas_pct]**2).sum(axis=1)
    
    #print(df_ihh.sample(5))
    return df_ihh

def intensidad_intercategoria(df):
    negocios = df["Negocio"].unique()
    cant_negocios = (df.groupby("Cliente")["Negocio"].nunique().reset_index()).rename(columns={"Negocio": "Cantidad Negocios Comprados"})
    cant_negocios["Amplitud de Portafolio"] = cant_negocios["Cantidad Negocios Comprados"]/len(negocios) 
    #CUANTAS CATEGORIAS COMPRA EL CLIENTE
    
    temp_co_ocurrencia = (df.groupby(["Cliente", "ID Pedido"])["Negocio"].nunique().reset_index(name="Cant_Negocios_Pedido"))
    co_ocurrencia = (temp_co_ocurrencia.groupby("Cliente")["Cant_Negocios_Pedido"].mean().reset_index()
                     .rename(columns={"Cant_Negocios_Pedido": "Promedio Negocios Por Pedido"}))
    # QUE TAN MEZCLADA ESTA LA FACTURA
    
    presencia = (df.groupby(["Cliente", "Negocio"])["Valor"].sum().unstack(fill_value=0))
    for negocio in negocios:
        if negocio not in presencia.columns:
            presencia[negocio]=0.0

    binaria = (presencia[negocios]>0).astype(int)
    df_co_compras = pd.DataFrame(index=binaria.index)
    
    for cat1, cat2 in combinations(negocios, 2):
        n1 = (cat1.replace(" ", "_"))
        n2 = (cat2.replace(" ", "_"))
        nombre_col = f"Co_Compra_{n1}_x_{n2}"
        df_co_compras[nombre_col] = ((binaria[cat1] == 1) & (binaria[cat2] == 1)).astype(int)

    df_co_compras = df_co_compras.reset_index()
    # PARES DE CATEGORIAS

    df_intensidad = cant_negocios.merge(co_ocurrencia, on="Cliente", how="inner").merge(df_co_compras, on="Cliente", how="inner")
    return df_intensidad

# Habitos temporales
def habitos_temporales(df):
    df_temp = df.copy()
    df_temp["Fecha Pedido"] = pd.to_datetime(df_temp["Fecha Pedido"])
    fecha_corte = df_temp["Fecha Pedido"].max()

    df_tiempo = (df_temp.groupby("Cliente").agg(Primera_Fecha=("Fecha Pedido", "min"), 
                                                Ultima_Fecha=("Fecha Pedido", "max"), 
                                                Total_Pedidos=("ID Pedido", "nunique")).reset_index())

    df_tiempo["Dias antiguedad"] = (fecha_corte - df_tiempo["Primera_Fecha"]).dt.days
    df_tiempo["Rango Dias Activo"] = (df_tiempo["Ultima_Fecha"] - df_tiempo["Primera_Fecha"]).dt.days

    # Intervalo promedio inter-compra en días
    df_tiempo["Intervalo_Promedio_Dias"] = np.where(df_tiempo["Total_Pedidos"] > 1, df_tiempo["Rango Dias Activo"] / (df_tiempo["Total_Pedidos"] - 1), np.nan)

    return df_tiempo[["Cliente", "Dias antiguedad", "Rango Dias Activo", "Intervalo_Promedio_Dias"]]

# Distribucion geografica y canales de venta
def canales_y_geografia(df):
    df_geog = (df.groupby("Cliente").agg(Municipios_Distintos=("Municipio", "nunique"), #Movilidad por municipio
                                         Puntos_Venta_Distintos=("Punto de Venta", "nunique"), #Movilidad por punto de venta
                                         Asesores_Distintos=("Asesor", "nunique"), #Cuantos asesores lo han atendido
                                         Total_Transacciones=("ID Pedido", "count"),).reset_index()) 
                                            #cuantas compras ha hecho el cliente

    if "Asesor" in df.columns:
        asesor_top = (df.groupby(["Cliente", "Asesor"])["ID Pedido"].count().reset_index(name="Trans_Asesor"))
        max_asesor = (asesor_top.groupby("Cliente")["Trans_Asesor"].max().reset_index(name="Max_Trans_Asesor"))
        df_geog = df_geog.merge(max_asesor, on="Cliente", how="left")
        df_geog["Indice Concentracion Asesor"] = (df_geog["Max_Trans_Asesor"] / df_geog["Total_Transacciones"])
    else:
        df_geog["Indice Concentracion Asesor"] = 1.0

    pdv_top = (df.groupby(["Cliente", "Punto de Venta"])["ID Pedido"].count().reset_index(name="Trans_PDV"))
    max_pdv = (pdv_top.groupby("Cliente")["Trans_PDV"].max().reset_index(name="Max_Trans_PDV"))
    df_geog = df_geog.merge(max_pdv, on="Cliente", how="left")
    df_geog["Indice Concentracion PDV"] = (df_geog["Max_Trans_PDV"] / df_geog["Total_Transacciones"])

    cols_retorno = [
        "Cliente",
        "Municipios_Distintos",
        "Puntos_Venta_Distintos",
        "Asesores_Distintos",
        "Indice Concentracion Asesor",
        "Indice Concentracion PDV",
    ]
    return df_geog[cols_retorno]

def construir_matriz_features(df):
    """
    Unifica todas las dimensiones extraídas en un unico DataFrame por Cliente.
    """
    df_rfm = discretizar_RFM(matriz_RFM(df))
    df_sow = share_of_wallet(df)
    df_ihh = indice_IHH(df_sow)
    df_intensidad = intensidad_intercategoria(df)
    df_tiempo = habitos_temporales(df)
    df_geog = canales_y_geografia(df)

    df_features = (
        df_rfm.merge(df_sow, on="Cliente", how="left")
        .merge(df_ihh, on="Cliente", how="left")
        .merge(df_intensidad, on="Cliente", how="left")
        .merge(df_tiempo, on="Cliente", how="left")
        .merge(df_geog, on="Cliente", how="left")
    )
    return df_features


#=====================================
# EXPORTACION DE LA MATRIZ DE FEATURES
#=====================================

import json
from pathlib import Path

# Ruta de salida relativa a la ubicacion de este archivo .py
CARPETA_SCRIPT = Path(__file__).resolve().parent

RUTA_DATOS = (
    CARPETA_SCRIPT / "../Datos/feature_eng/features_init"
).resolve()

RUTA_DATOS.mkdir(parents=True, exist_ok=True)

NOMBRE_BASE = "matriz_features_clientes"

ruta_csv = RUTA_DATOS / f"{NOMBRE_BASE}.csv"
ruta_parquet = RUTA_DATOS / f"{NOMBRE_BASE}.parquet"
ruta_meta = RUTA_DATOS / f".{NOMBRE_BASE}.meta.json"


def exportar_base_unificada(df):
    meta_actual = {
        "filas": len(df),
        "columnas": list(df.columns)
    }

    print("\n=====================================")
    print("EXPORTACION DE LA MATRIZ DE FEATURES")
    print("=====================================")
    print(f"Carpeta de salida: {RUTA_DATOS}")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    try:
        df.to_csv(ruta_csv, sep=";", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        print(f"CSV guardado correctamente: {ruta_csv}")

        df.to_parquet(ruta_parquet, index=False)
        print(f"Parquet guardado correctamente: {ruta_parquet}")

        ruta_meta.write_text(json.dumps(meta_actual, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Metadatos guardados: {ruta_meta}")
        print("Exportacion completada correctamente.")

    except Exception as e:
        print(f"\nERROR DURANTE LA EXPORTACION: {e}")
        raise

print("Iniciando construccion de la matriz de features...")

df_features = construir_matriz_features(df)
print(f"Matriz construida: {df_features.shape}")

exportar_base_unificada(df_features)
