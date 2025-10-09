# ============================================
# GLM - Diagnósticos completos en R
# ============================================

# Instalar y cargar librerías necesarias
paquetes <- c("lmtest", "car", "nortest", "ggplot2", "MASS")
instalar <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(instalar)) install.packages(instalar)
lapply(paquetes, library, character.only = TRUE)

# --------------------------------------------
# 1. Cargar datos
# --------------------------------------------
cat("Selecciona tu archivo CSV...\n")
datos <- read.csv(file.choose(), header = TRUE)
str(datos)
head(datos)

# Asegúrate de tener columnas 'x' y 'y'
# 'x' puede ser numérica o categórica; 'y' depende del modelo

# --------------------------------------------
# 2. Modelo Gaussiano (Regresión Lineal)
# --------------------------------------------
modelo_gauss <- glm(y ~ x, data = datos, family = gaussian())
summary(modelo_gauss)

ggplot(datos, aes(x=x, y=y)) +
  geom_point(color="blue") +
  geom_smooth(method="glm", formula=y~x, color="red", se=FALSE) +
  ggtitle("Regresión Lineal (Gaussian)")

res <- resid(modelo_gauss)
fit <- fitted(modelo_gauss)

# Homocedasticidad
plot(fit, res, main="Homocedasticidad (Lineal)", xlab="Ajustados", ylab="Residuos")
abline(h=0, col="red")
print(bptest(modelo_gauss))

# Autocorrelación
print(durbinWatsonTest(modelo_gauss))

# Normalidad
qqPlot(modelo_gauss)
hist(res, main="Histograma de residuos")
print(jarque.bera.test(res))
print(lillie.test(res))

# --------------------------------------------
# 3. Modelo Logístico (Binomial)
# --------------------------------------------
modelo_logit <- glm(y ~ x, data = datos, family = binomial())
summary(modelo_logit)

datos$pred_logit <- fitted(modelo_logit)
ggplot(datos, aes(x=x, y=y)) +
  geom_point(color="blue") +
  geom_line(aes(y=pred_logit), color="red") +
  ggtitle("Regresión Logística")

res_logit <- residuals(modelo_logit, type="pearson")
plot(fitted(modelo_logit), res_logit, main="Residuos Logísticos", xlab="Ajustados", ylab="Residuos")
abline(h=0, col="red")

# --------------------------------------------
# 4. Modelo Poisson
# --------------------------------------------
modelo_pois <- glm(y ~ x, data = datos, family = poisson())
summary(modelo_pois)

datos$pred_pois <- fitted(modelo_pois)
ggplot(datos, aes(x=x, y=y)) +
  geom_point(color="blue") +
  geom_line(aes(y=pred_pois), color="red") +
  ggtitle("Regresión Poisson")

res_pois <- residuals(modelo_pois, type="pearson")
plot(fitted(modelo_pois), res_pois, main="Residuos Poisson", xlab="Ajustados", ylab="Residuos")
abline(h=0, col="red")

# --------------------------------------------
# 5. Modelo Gamma
# --------------------------------------------
modelo_gamma <- glm(y ~ x, data = datos, family = Gamma(link="log"))
summary(modelo_gamma)

datos$pred_gamma <- fitted(modelo_gamma)
ggplot(datos, aes(x=x, y=y)) +
  geom_point(color="blue") +
  geom_line(aes(y=pred_gamma), color="red") +
  ggtitle("Regresión Gamma")

res_gamma <- residuals(modelo_gamma, type="pearson")
plot(fitted(modelo_gamma), res_gamma, main="Residuos Gamma", xlab="Ajustados", ylab="Residuos")
abline(h=0, col="red")

# --------------------------------------------
# 6. Modelo Binomial Negativa
# --------------------------------------------
modelo_nb <- glm.nb(y ~ x, data = datos)
summary(modelo_nb)

datos$pred_nb <- fitted(modelo_nb)
ggplot(datos, aes(x=x, y=y)) +
  geom_point(color="blue") +
  geom_line(aes(y=pred_nb), color="red") +
  ggtitle("Regresión Binomial Negativa")

res_nb <- residuals(modelo_nb, type="pearson")
plot(fitted(modelo_nb), res_nb, main="Residuos Binomial Negativa", xlab="Ajustados", ylab="Residuos")
abline(h=0, col="red")


