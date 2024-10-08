import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv("real_estate_valuation_data_set.csv")
print(df.head())

# Crear la matriz de variables y de precios
X = df[df.columns[1:-1]]
Y = df[df.columns[-1]]

# Estadísticas descriptivas de las variables
print(X.describe())
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

# Modelo de regresión
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

linreg = LinearRegression()
linreg.fit(x_train,y_train)
intercepto = linreg.intercept_
print("Intercepto:", intercepto)
coeficientes = linreg.coef_
print("Coeficientes:", coeficientes)

# Predicción
y_pred = linreg.predict(x_test)

# Error absoluto medio
MAE = metrics.mean_absolute_error(y_test,y_pred)
# Error cuadrático medio
MSE = metrics.mean_squared_error(y_test,y_pred)
# Raiz MSE
RMSE = np.sqrt(MSE)

print("MAE:", MAE)
print("MSE:", MSE)
print("SMSE:", RMSE)

# Eliminación de la variable X6
X_2 = X.drop("X6 longitude", axis=1)
X_2

# Nuevo test
x_train, x_test, y_train, y_test = train_test_split(X_2, Y, random_state=0)
linreg.fit(x_train,y_train)
intercepto = linreg.intercept_
print("Intercepto:", intercepto)
coeficientes = linreg.coef_
print("Coeficientes:", coeficientes)
y_pred = linreg.predict(x_test)
MAE = metrics.mean_absolute_error(y_test,y_pred)
MSE = metrics.mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)

print("MAE:", MAE)
print("MSE:", MSE)
print("SMSE:", RMSE)

# Validación cruzada
# MAE
sc = cross_val_score(linreg, X, Y, cv=5, scoring="neg_mean_absolute_error")
mae_sc = -sc
print(mae_sc)
print(mae_sc.mean())
# MSE
scores = cross_val_score(linreg, X, Y, cv=5, scoring="neg_mean_squared_error")
mse_scores = - scores
print(mse_scores)
print(mse_scores.mean())
# Raiz de los MSE
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)
print(rmse_scores.mean())

# Statsmodel
# Con todas las variables iniciales
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
# Agregar la columna de 1
x_train = sm.add_constant(x_train)
# Modelo de regresión
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# Sacando X6
x_train, x_test, y_train, y_test = train_test_split(X_2, Y, random_state=0)
# Agregar la columna de 1
x_train = sm.add_constant(x_train)
# Modelo de regresión
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# Influencia de las observaciones
fig = sm.graphics.influence_plot(model)
# Puntos de influencia
# Distancia de cook
model_cook = model.get_influence().cooks_distance[0]

n = x_train.shape[0]

# Umbral
critical_d = 4/n
print("Umbral con distancia de Cook:", critical_d)

# Posibles outliers con influencia
outliers = model_cook > critical_d
print(x_train.index[outliers])
print(model_cook[outliers])

# Datos de entrenamiento sin el outlier más alto
x_train = x_train.drop(index=[270])
y_train = y_train.drop(index=[270])

# Modelo de regresión una vez eliminado el valor atípico más alto
model = sm.OLS(y_train,x_train).fit()
print(model.summary())

# Puntos de influencia
# Distancia de cook
model_cook = model.get_influence().cooks_distance[0]

n = x_train.shape[0]

# Umbral
critical_d = 4/n
print("Umbral con distancia de Cook:", critical_d)

# Posibles outliers con influencia
outliers = model_cook > critical_d
print(x_train.index[outliers])

# Eliminación de todos los datos atípicos
indices = x_train.index[outliers]
x_train = x_train.drop(index=indices)
y_train = y_train.drop(index=indices)

# Modelo de regresión sin todos los valores atípicos
model = sm.OLS(y_train,x_train).fit()
print(model.summary())