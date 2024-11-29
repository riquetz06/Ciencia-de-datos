ejemplo<-scan()
max(ejemplo)
pie(ejemplo)
hist(ejemplo)
normal<-rnorm(250)
hist(normal)
hist(normal, breaks=50, freq=F)
hist(normal, breaks=50, freq=F, main= "HISTOGRAMA DIST NORMAL",
+ xlab="numeros del eje x", ylab="números del eje y",
+ xlim=c(-3,3), ylim=c(0,0.6)
+ , col="51")
curve(dnorm, add=T)

#Exportar de R a Excel 
# Instalar y cargar la biblioteca necesaria
install.packages("writexl")
library(writexl)
# Exportar el data frame a un archivo Excel
write_xlsx(auto_data, "ruta/donde/deseas/guardar/auto_data.xlsx")
