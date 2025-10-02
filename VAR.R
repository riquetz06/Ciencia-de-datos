# ===========================
# 1. Cargar librerías
# ===========================
install.packages(c("vars", "readr", "dplyr"))
library(vars)
library(readr)
library(dplyr)

# ===========================
# 2. Cargar CSV
# ===========================
# Cambia la ruta si no está en tu working directory
ruta <- "VAR_Financieras.csv"

# Detectar separador y decimales
# Si tus números tienen coma de miles y punto decimal:
df <- read_delim(ruta, delim = ",", locale = locale(decimal_mark = "."))

# Revisar primeras filas
head(df)
str(df)

# ===========================
# 3. Convertir columnas a numéricas
# ===========================
# Eliminar comas de miles
cols <- colnames(df)[-1]  # asumimos que la primera columna es Fecha
for (col in cols) {
  df[[col]] <- as.numeric(gsub(",", "", df[[col]]))
}

# Eliminar columnas vacías
df <- df %>% select(where(~!all(is.na(.))))

# ===========================
# 4. Convertir precios a rendimientos
# ===========================
# Detectar precios (todos positivos y cambios grandes)
max_diff <- max(diff(as.matrix(df[cols])), na.rm = TRUE)

if(all(df[cols] > 0) & max_diff > 1) {
  cat("Se detectaron PRECIOS. Calculando rendimientos...\n")
  df_ret <- df
  for(col in cols) {
    df_ret[[col]] <- c(NA, diff(log(df[[col]])))  # rendimientos logarítmicos
  }
  df_ret <- na.omit(df_ret)
} else {
  cat("Se detectaron RENDIMIENTOS. Usando datos directamente...\n")
  df_ret <- df %>% select(all_of(cols))
}

# Revisar dimensiones
cat("Filas y columnas después de limpieza:", dim(df_ret), "\n")

# ===========================
# 5. Ajustar VAR
# ===========================
# Convertir a ts (opcional)
df_ts <- ts(df_ret, frequency = 1)

# Ajustar VAR con 2 rezagos
var_model <- VAR(df_ts, p = 2, type = "const")
summary(var_model)

# ===========================
# 6. Funciones Impulso-Respuesta (IRF)
# ===========================
irf_model <- irf(var_model, impulse = cols, response = cols, n.ahead = 10, boot = TRUE)
print(irf_model)

# Graficar IRF
plot(irf_model)
