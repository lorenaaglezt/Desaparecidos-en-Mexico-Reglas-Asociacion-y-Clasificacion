import pandas as pd
from itertools import combinations


def df_a_transacciones(df):
    """
    Convierte un DataFrame de Pandas en una lista de transacciones (lista de listas)
    formateada para el algoritmo apriori.
    
    Args:
        df (pd.DataFrame): El DataFrame con los datos originales.
    
    Returns:
        list: Una lista de listas lista que representa las transacciones del dataframe
    """
    columnas = df.columns.tolist()
    transacciones = []
    for index, fila in df.iterrows():
        transaccion = [f"{col}={fila[col]}" for col in columnas]
        transacciones.append(transaccion)
            
    return transacciones


def obtener_itemsets_frecuentes_1(transacciones, soporte_min_conteo):
    """
    Escanea toda la base de datos para contar cuántas veces aparece cada elemento individualmente. 
    Luego, descarta los que no alcanzan el umbral mínimo.

    Args:
        transacciones (list): Lista de listas. Cada lista representa un registro con sus características
        soporte_min_conteo (int): Número de veces que debe aparecer un elemento para considerarlo frecuenta

    Returns:
        dict: Diccionario donde las claves son los elementos frecuentes encapsulados en un frozenset 
              y los valores son su conteo exacto
    """
    conteo = {}
    for transaccion in transacciones:
        for item in transaccion:
            # Usamos frozenset para poder usar el set como llave en el diccionario
            llave = frozenset([item])
            conteo[llave] = conteo.get(llave, 0) + 1
            
    # Filtramos por el soporte mínimo
    return {itemset: count for itemset, count in conteo.items() if count >= soporte_min_conteo}


def generar_candidatos(itemsets_frecuentes, k):
    """
    Toma los conjuntos frecuentes del paso anterior (tamaño k-1) y los combina entre sí para crear 
    nuevos grupos candidatos más grandes (de tamaño k).

    Args:
        itemsets_frecuentes (dict): Diccionario de elementos frecuentes
        k (int): Tamaño de los nuevos conjuntos

    Returns:
        set: Conjunto que contiene múltiples frozenset de tamaño k, es importante aclarar que aquí son
             candidatos, aún no se confirma si son frecuentes
    """
    candidatos = set()
    lista_itemsets = list(itemsets_frecuentes.keys())
    
    for i in range(len(lista_itemsets)):
        for j in range(i + 1, len(lista_itemsets)):
            # Unimos dos itemsets
            union = lista_itemsets[i] | lista_itemsets[j]
            # Si el tamaño de la unión es exactamente k, es un candidato válido
            if len(union) == k:
                candidatos.add(union)
    return candidatos


def filtrar_candidatos(transacciones, candidatos, soporte_min_conteo):
    """
    Escanea la base de datos completa para ver cuántas veces aparecen juntos realmente los elementos de cada 
    conjunto candidato.

    Args:
        transacciones (list): Base de datos
        candidatos (set): Conjunto de candidatos a revisar
        soporte_min_conteo (int): Número de veces que debe aparecer un elemento para considerarlo frecuenta

    Returns:
        dict: Diccionario con los candidatos que su conteo fue mayor o igual a soporte_min_conteo
    """
    conteo = {candidato: 0 for candidato in candidatos}
    
    for transaccion in transacciones:
        transaccion_set = set(transaccion)
        for candidato in candidatos:
            if candidato.issubset(transaccion_set):
                conteo[candidato] += 1
                
    return {itemset: count for itemset, count in conteo.items() if count >= soporte_min_conteo}


def algoritmo_apriori(transacciones, soporte_min=0.1):
    """
    Encuentra grupos de elementos que suelen aparecer juntos en grandes conjuntos de datos.

    Args:
        transacciones (list): Base de datos con las transacciones
        min_support (float, optional): Porcentaje de soporte mínimo deseado. Defaults to 0.1.

    Returns:
        dict: Diccionario que contiene todos los conjuntos frecuentes encontrados de cualquier tamaño 
              y su respectivo porcentaje de aparición (de 0 a 1).
    """
    total_transacciones = len(transacciones)
    min_support_count = soporte_min * total_transacciones
    
    itemsets_todos = {}
    
    # Obtener frecuentes de tamaño 1
    itemsets_k = obtener_itemsets_frecuentes_1(transacciones, min_support_count)
    itemsets_todos.update(itemsets_k)
    
    # Bucle para k = 2, 3, 4...
    k = 2
    while itemsets_k:
        candidatos = generar_candidatos(itemsets_k, k)
        itemsets_k = filtrar_candidatos(transacciones, candidatos, min_support_count)
        itemsets_todos.update(itemsets_k)
        k += 1
        
    # Devolvemos los itemsets con su soporte en porcentaje (0 a 1)
    return {itemset: count / total_transacciones for itemset, count in itemsets_todos.items()}


def generar_reglas(itemsets_frecuentes, confianza_min):
    """Toma los conjuntos de elementos frecuentes y aplica probabilidad condicional para encontrar reglas lógicas.

    Args:
        itemsets_frecuentes (dict): Diccionario generado por el algoritmo apriori
        confianza_min (float): El umbral mínimo de certeza que debe tener la regla

    Returns:
        pandas.DataFrame: Tabla estructurada y ordenada con las reglas de asociación
    """
    reglas = []
    
    # Para cada itemset frecuente encontrado en apriori
    for itemset, support in itemsets_frecuentes.items():
        if len(itemset) > 1:
            # Iteramos para crear todas las combinaciones posibles de antecedente -> consecuente
            for i in range(1, len(itemset)):
                for antecedente in combinations(itemset, i):
                    antecedente = frozenset(antecedente)
                    consecuente = itemset - antecedente
                    
                    soporte_antecedente = itemsets_frecuentes[antecedente]
                    confianza = support / soporte_antecedente
                    
                    if confianza >= confianza_min:
                        soporte_consecuente = itemsets_frecuentes[consecuente]
                        lift = confianza / soporte_consecuente
                        
                        reglas.append({
                            'Antecedente': set(antecedente),
                            'Consecuente': set(consecuente),
                            'Soporte': support,
                            'Confianza': confianza,
                            'Lift': lift
                        })
                        
    return pd.DataFrame(reglas).sort_values(by='Lift', ascending=False)