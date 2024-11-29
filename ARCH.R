#Modelo ARCH (modelar series con varianza heteroscedástica)
install.packages("tseries")
library(tseries)
# Crear datos simulados
set.seed(42)
returns <- rnorm(1000)
# Ajustar el modelo ARCH
arch_model <- garch(returns, order = c(0, 1))
summary(arch_model)
# Graficar el ajuste del modelo ARCH
plot(arch_model)
#Con retornos:
# Instalar y cargar las librerías necesarias
install.packages("quantmod")
install.packages("tseries")
library(quantmod)
library(tseries)
# Descargar datos históricos de una acción (por ejemplo, Apple Inc.)
getSymbols("AAPL", from = "2020-01-01", to = Sys.Date())
# Obtener precios de cierre
precios_cierre <- Cl(AAPL)
# Calcular los retornos logarítmicos
retornos <- diff(log(precios_cierre))
# Eliminar los NA resultantes de la diferencia
retornos <- na.omit(retornos)
# Ajustar el modelo ARCH
arch_model <- garch(retornos, order = c(0, 1))
# Mostrar los resultados del modelo
summary(arch_model)

#Modelo ARCH con pronósticos:
# Instalar y cargar las librerías necesarias
install.packages("quantmod")
install.packages("rugarch")
library(quantmod)
library(rugarch)
# Descargar datos históricos de una acción (por ejemplo, Apple Inc.)
getSymbols("AAPL", from = "2020-01-01", to = Sys.Date())
# Obtener precios de cierre
precios_cierre <- Cl(AAPL)
# Calcular los retornos logarítmicos
retornos <- diff(log(precios_cierre))
# Eliminar los NA resultantes de la diferencia
retornos <- na.omit(retornos)
# Especificar el modelo ARCH
spec <- ugarchspec(
variance.model = list(model = "sGARCH", garchOrder = c(1, 0)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm")
# Ajustar el modelo ARCH
fit <- ugarchfit(spec = spec, data = retornos)
# Hacer un pronóstico de 10 días
forecast <- ugarchforecast(fit, n.ahead = 10)
# Mostrar los resultados del pronóstico
print(forecast)


#Modelo GARCH (modelar series con varianza heteroscedástica)
install.packages("rugarch")
library(rugarch)
# Crear datos simulados
set.seed(42)
returns <- rnorm(1000)
# Especificar el modelo GARCH(1,1)
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0)))
# Ajustar el modelo GARCH
garch_model <- ugarchfit(spec = spec, data = returns)
summary(garch_model)
# Graficar el ajuste del modelo GARCH
plot(garch_model)

#Modelo E-GARCH (modelar asimetrías en la volatilidad)
install.packages("rugarch")
library(rugarch)
# Crear datos simulados
set.seed(42)
returns <- rnorm(1000)
# Especificar el modelo E-GARCH(1,1)
spec <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0)))

# Ajustar el modelo E-GARCH
egarch_model <- ugarchfit(spec = spec, data = returns)
summary(egarch_model)
# Graficar el ajuste del modelo E-GARCH
plot(egarch_model)

#Modelo Threshold-ARCH (Varianza Condicional puede responder a cambios en el signo de los residuos)
install.packages("fGarch")
library(fGarch)
# Crear datos simulados
set.seed(42)
returns <- rnorm(1000)
# Ajustar el modelo T-ARCH(1)
tarch_model <- garchFit(~ garch(1, 0) + thres(1), data = returns)
summary(tarch_model)
# Graficar el ajuste del modelo T-ARCH
plot(tarch_model)
