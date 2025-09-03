T=read.csv(file.choose())
ss=ts(T)
ss
plot(ss, main="USD/MXN",ylab="days",col="blue")
library(tseries)
arch_model <- garch(ss, order = c(0, 1))
summary(arch_model)
# Graficar el ajuste del modelo ARCH
plot(arch_model)
# Instalar y cargar las librerías necesarias
library(quantmod)
library(rugarch)
forecast <- ugarchforecast(ss, n.ahead = 10)
library(rugarch)

# Especificación de un ARCH(1)
spec_arch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,0)), # ARCH(1) = GARCH(1,0)
  mean.model     = list(armaOrder = c(0,0)), 
  distribution.model = "norm")

# Ajuste del modelo (ejemplo con serie ss)
fit_arch <- ugarchfit(spec = spec_arch, data = ss)

# Pronóstico a 10 pasos adelante
forecast_arch <- ugarchforecast(fit_arch, n.ahead = 10)

# Mostrar resultados
show(forecast_arch)

# Si quieres extraer la varianza condicional pronosticada:
sigma_forecast <- sigma(forecast_arch)

# Y las medias esperadas:
mean_forecast <- fitted(forecast_arch)

library(rugarch)
library(ggplot2)

# --- 1. Especificación ARCH(1) ---
spec_arch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,0)), # ARCH(1)
  mean.model     = list(armaOrder = c(0,0)), 
  distribution.model = "norm"
)

# --- 2. Ajuste del modelo (ejemplo con tu serie ss) ---
fit_arch <- ugarchfit(spec = spec_arch, data = ss)

# --- 3. Pronóstico (10 pasos adelante) ---
forecast_arch <- ugarchforecast(fit_arch, n.ahead = 10)

# Extraer pronósticos
mean_forecast  <- as.numeric(fitted(forecast_arch))      # medias
sigma_forecast <- as.numeric(sigma(forecast_arch))       # desviaciones estándar

# --- 4. Construcción de un data.frame para graficar ---
n <- length(ss)
time_index <- 1:n
forecast_index <- (n+1):(n+10)

df_hist <- data.frame(
  time = time_index,
  value = as.numeric(ss)
)

df_forecast <- data.frame(
  time = forecast_index,
  mean = mean_forecast,
  lower = mean_forecast - 1.96 * sigma_forecast,
  upper = mean_forecast + 1.96 * sigma_forecast
)

# --- 5. Gráfico con ggplot2 ---
ggplot(df_hist, aes(x = time, y = value)) +
  geom_line(color = "black") +
  geom_line(data = df_forecast, aes(x = time, y = mean), color = "blue") +
  geom_ribbon(data = df_forecast, aes(x = time, ymin = lower, ymax = upper),
              fill = "blue", alpha = 0.2, inherit.aes = FALSE) +
  labs(title = "Pronóstico ARCH(1)",
       x = "Tiempo", y = "Valor") +
  theme_minimal()
