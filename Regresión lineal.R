datos <- read.csv(file.choose())
datos
library(ggplot2)
ggplot(datos,aes(x=x,y=y))+geom_point()
modelo<- lm(y~x1+x2, data=datos)
ggplot(datos,aes(x=x,y=y))+geom_point()+geom_smooth(method=lm,se=FALSE)
summary(modelo)
