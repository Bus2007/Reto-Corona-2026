import pandas as pd
import numpy as np

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
    df_frec["Items_Promedio_Por_Pedido"] = (df_frec["Total_Lineas_Compradas"] / df_frec["Frecuencia_Pedidos"])
    return df_frec

def valor_monetario(df):
    df_mon = (df.groupby("Cliente").agg(Valor_Monetario_Total=("Valor", "sum"),
            Devoluciones=("Valor",lambda x: x[x < 0].sum()
                          )
            ).reset_index()
              )
    df_mon["Devoluciones"] = df_mon["Devoluciones"].abs()
    return df_mon

def matriz_RFM(df):
    df_R = recencia(df)
    df_F = frecuencia(df)
    df_M = valor_monetario(df)
    
    df_RFM = df_R.merge(df_F, how="inner", on="Cliente").merge(df_M, how="inner", on="Cliente")
    df_RFM["Recibo_Promedio_Por_Pedido"] = (df_RFM["Valor_Monetario_Total"] / df_RFM["Frecuencia_Pedidos"])
    
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
    print(df_discretizado.head(5))
    return df_discretizado

df_rfm_base = matriz_RFM(df)

# 2. Aplicar discretización ordinal (Scores 1-5)
df_rfm_discretizado = discretizar_RFM(df_rfm_base)

print(f"Matriz RFM procesada para {len(df_rfm_discretizado):,} clientes.")
print("\nVista previa de la base RFM Discretizada:")
print(
    df_rfm_discretizado[
        [
            "Cliente",
            "Recencia",
            "Frecuencia_Pedidos",
            "Valor_Monetario_Total",
            "Puntaje R",
            "Puntaje F",
            "Puntaje M",
            "RFM Combinado",
        ]
    ].head()
)