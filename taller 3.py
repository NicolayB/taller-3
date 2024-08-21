import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("real_estate_valuation_data_set.csv")
print(df.head())

# Crear la matriz de variables y de precios
X = df[df.columns[1:-1]]
Y = df[df.columns[-1]]

# Estadísticas descriptivas de los precios de venta
print(Y.describe())