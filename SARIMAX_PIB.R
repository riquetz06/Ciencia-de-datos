#SARIMAX PIB
datos_pib <- read.csv(file.choose())
datos_pib
tly <- ts(datos_pib[ , 1], start = c(2000, 1), end = c(2024, 2), freq = 4)
plot(tly)
descomp<-decompose(tly)
plot(descomp)
#H0: La serie no es estacionaria. p.value≥0.05
#H1: La serie es estacionaria. p.value≤0.05
# Prueba de Dickey-Fuller (ADF)
adf.test(tly)
adf <- ur.df(tly,type = "none", lags = 0)
# Prueba de Phillips-Perron (PP)
pp.test(tly)
pp <- ur.pp(tly)
summary(pp)
Prueba KPSS
library(urca)
kpss <- ur.kpss(tly)
summary(kpss)
estacionaria <- cbind(tly)
muestra_end <- window(estacionaria,tart = c(2000, 1), end = c(2024,2),freq = 4)
# Correlograma
ts_cor(tly)
sarima <- Arima(muestra_end, order = c(1, 0, 0), seasonal = list(order = c(0, 0, 1), period = 4))
summary(sarima)
coeftest(sarima)
#H0: Los residuos no se comportan como ruido blanco. p.value≥0.05
#H1: Los residuos se comportan como ruido blanco. p.value≤0.05
residuals <- resid(sarima)
adf.test(residuals)
Prueba Ljung-Box
#H0: No hay autocorrelacion, los residuos se distribuyen de forma independiente. p.value≥0.05
#H1: Si hay autocorrelacion, los residuos no se distribuyen de forma independiente p.value≤0.05
checkresiduals(sarima)
predsarima <- forecast(sarima, h = 8)
summary(predsarima)
