datos <- read.csv(file.choose())
datos
library(ggplot2)
ggplot(datos,aes(x=x,y=y))+geom_point()
modelo<- lm(y~x1+x2, data=datos)
ggplot(datos,aes(x=x,y=y))+geom_point()+geom_smooth(method=lm,se=FALSE)
summary(modelo)

# Modelo CAPM-Ejemplo
datos <- read.csv(file.choose())
datos
library(ggplot2)
ggplot(datos,aes(x=Prima,y=Ln_AMXL))+geom_point()
modelo<- lm(Ln_AMXL~Prima, data=datos)
ggplot(datos,aes(x=Prima,y=Ln_AMXL))+geom_point()+geom_smooth(method=lm,se=FALSE)
summary(modelo)

# Homoscedasticidad y Autocorrelacion
library(stargazer)
stargazer(modelo,title="modelo estimado", type="text")
library(lmtest)
prueba_white<-bptest(modelo,~I(x1^2)+I(x2^2)+x1*x2, data=datos)
print(prueba_white)
# produce residual vs. fitted plot
res <- resid(modelo)
res
plot(fitted(modelo), res)
# add a horizontal line at 0 
abline(0,0)
dwtest(modelo,alternative="two.sided",iterations=1000)

# Pruebas de normalidad
library(fitdistrplus)
ajuste_normal<-fitdist(data=modelo$residuals,distr="norm")
plot(ajuste_normal)
library(tseries)
jb_test<- jarque.bera.test(res)
print(jb_test)
library(nortest)
lillie.test(modelo$residuals)

