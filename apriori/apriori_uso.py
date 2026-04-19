import pandas as pd
import apriori.apriori as apriori

df = pd.read_csv('data_secretariado.csv')

# limpieza
columnas_a_eliminar = ['ID_VICTIMA', 'FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO', 'CVE_ENT', 'CVE_MUN']
df = df.drop(columns=columnas_a_eliminar)

df = df.replace('CONFIDENCIAL', pd.NA)
df = df.replace('NO ESPECIFICADO', pd.NA)
df = df.dropna() 

# aplicación apriori
transacciones = apriori.df_a_transacciones(df)
itemset_frecuentes = apriori.algoritmo_apriori(transacciones, soporte_min=0.1)
reglas = apriori.generar_reglas(itemset_frecuentes, confianza_min=0.5)

print(reglas.to_string(index=False))