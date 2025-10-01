# =======================================
# 1. Cargar librerías
# =======================================
library(vars)

# =======================================
# 2. Leer tus datos
# =======================================
# Supongamos que tu CSV tiene columna "Fecha"
datos <- read.csv("mis_datos.csv")

# Convertir Fecha a tipo Date
datos$Fecha <- as.Date(datos$Fecha)

# Establecer las fechas como índice (opcional)
rownames(datos) <- datos$Fecha
datos <- datos[, -1]  # quitamos columna de fechas

# =======================================
# 3. Si son precios, calculamos rendimientos logarítmicos
# =======================================
rendimientos <- na.omit(diff(log(as.matrix(datos))))

# =======================================
# 4. Selección de rezagos
# =======================================
lag_select <- VARselect(rendimientos, lag.max = 10, type = "const")
print(lag_select$selection)

# =======================================
# 5. Ajustar modelo VAR
# =======================================
var_model <- VAR(rendimientos, p = lag_select$selection["AIC(n)"], type = "const")
summary(var_model)

# =======================================
# 6. Pronóstico
# =======================================
pred <- predict(var_model, n.ahead = 10)
plot(pred)

# =======================================
# 7. Respuesta impulso (opcional)
# =======================================
irf_result <- irf(var_model, impulse = colnames(rendimientos)[1],
                  response = colnames(rendimientos)[2],
                  n.ahead = 10, boot = TRUE)
plot(irf_result)
