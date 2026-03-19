install.packages("AER")  # Para pruebas estadísticas
library(plm)
library(AER)

#Simular datos de panel
set.seed(123)
N <- 100  # Número de países
T <- 10   # Años
id <- rep(1:N, each=T)
time <- rep(1:T, N)

# Variables
y <- rnorm(N*T, mean=50, sd=10)  # PIB per cápita
x <- rnorm(N*T, mean=5, sd=2)    # Inversión en educación
z <- rnorm(N*T, mean=10, sd=3)   # Otras variables de control

# Crear dataframe
df <- data.frame(id, time, y, x, z)
df <- pdata.frame(df, index=c("id", "time"))  # Convertir a datos de panel

#Aplicar estimador Arellano-Bond
library(plm)

# Modelo GMM en diferencias
modelo_ab <- pgmm(y ~ lag(y, 1) + x | lag(y, 2:99),  
                  data=df, effect="twoways", model="twosteps", transformation="d")

summary(modelo_ab)  # Ver resultados

#Comentarios
Coeficiente de lag(y,1): Muestra la persistencia del PIB.

Coeficiente de x: Si es significativo y positivo, indica que la inversión en educación impulsa el crecimiento.

Prueba de Hansen: Evalúa la validez de los instrumentos (p-valor alto = instrumentos válidos).

Prueba AR(2): Debe ser no significativa (p>0.05) para asegurar que no hay correlación serial en diferencias de segundo orden.

#Visualizar Data frame
head(df)        # Muestra las primeras 6 filas
tail(df)        # Últimas 6 filas
View(df)        # Abre una ventana interactiva con toda la tabla (solo en RStudio)
summary(df)     # Estadísticas descriptivas

#Exportar a Excel
install.packages("writexl")
library(writexl)
write_xlsx(as.data.frame(df), path = "datos_panel.xlsx") #Exportar dataframe
getwd()         # Muestra el directorio actual
setwd("C:/Ruta/Donde/Guardar")  # Si quiero cambiar el directorio
