Vectores Autorregresivos
library(vars)
datos<-read.csv(file.choose())
datos
# Crear datos simulados
set.seed(42)
data <- data.frame(Variable1 = rnorm(100),Variable2 = rnorm(100))
# Ajustar el modelo VAR
var_model <- VAR(data, p = 2)  # p = número de rezagos
summary(var_model)
# Realizar predicciones
forecast <- predict(var_model, n.ahead = 10)
print(forecast)
# Crear fechas trimestrales
fechas <- seq(as.Date("1979-01-01"), by = "quarter", length.out = 40)
datos <- data.frame(Fecha = fechas, M1 = rnorm(40), R = rnorm(40))
datos
var_model <- VAR(datos[, -1], p =2)
summary(var_model)
# Realizar predicciones
forecast <- predict(var_model, n.ahead = 10)
print(forecast)
# Generar la respuesta impulso
irf_result <- irf(var_model, impulse = "M1", response = "R", n.ahead = 10, boot = TRUE)
plot(irf_result)
