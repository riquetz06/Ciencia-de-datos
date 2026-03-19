library(plm)

data <- data.frame(
empresa = c(1,1,1,1,2,2,2,2,3,3,3,3),
anio = c(2018,2019,2020,2021,2018,2019,2020,2021,2018,2019,2020,2021),
inversion = c(120,135,140,160,90,95,100,110,200,210,220,240),
flujo_caja = c(80,85,90,100,60,65,70,75,120,130,135,150),
ventas = c(400,420,450,470,300,320,340,360,600,630,660,700)
)

pdata <- pdata.frame(data, index=c("empresa","anio"))

modelo <- pgmm(
inversion ~ lag(inversion,1) + flujo_caja + ventas |
lag(inversion,2:3),
data = pdata,
effect = "individual",
model = "twosteps",
transformation = "d"
)

summary(modelo)
