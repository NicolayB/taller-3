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

# Gráficos de las variables
print(sns.pairplot(df.iloc[:,1:]))

correl = df.iloc[:,1:].corr()
print(sns.heatmap(correl, cmap="Blues", annot=True))

# Correlación de las variables y los preciosa
correlacion = pd.DataFrame(X.corrwith(Y))
print(sns.heatmap(correlacion, cmap="Blues", annot=True))

# Diagramas de regresión para cada variable
colum = X.columns
print(sns.pairplot(df, x_vars=colum[0:2], y_vars="Y house price of unit area", height=7, kind="reg"))
print(sns.pairplot(df, x_vars=colum[2:4], y_vars="Y house price of unit area", height=7, kind="reg"))
print(sns.pairplot(df, x_vars=colum[4:], y_vars="Y house price of unit area", height=7, kind="reg"))