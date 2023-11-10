rm(list=ls())
library(gstat)
library(sp)
library(gdata)
library(ggplot2)
library(pastecs)
library(gridExtra) 


# Lectura de datos
ran_points <- read.csv("../taller_02/muestreo_aleatorio.csv")
opoints <- ran_points <- subset(ran_points, evi > 0)
# ran_points <- ran_points[sample(1:nrow(ran_points), 1000), ]

# Lista de variables
names(ran_points)
str(ran_points)


# Crear objeto espacial expl?cito

class(ran_points)
coordinates(ran_points) <- c("x", "y")
str(ran_points)


# Graficar mapa con localizaci?n x e y
plot(ran_points, asp = 1, pch = 1)


# Análisis exploratorio
(base_stat <- stat.desc(ran_points)) |> round(3)


# Número de pares
n <- length(ran_points$x)
npair <- n*(n-1)/2
npair


# Construcción de Varigrama --------------

# Variograma Isotrópico  =================

# Por defecto #############################
(v <- variogram(evi~ 1, data = ran_points, cloud=F))
plot(v)

(diag_ <- sqrt(sum(diff(t(ran_points@bbox))^2)) )
(v_i <- variogram(evi~ 1, data = ran_points, cutoff = diag_/3, width = diag_/3/15))
plot(v_i)

# Buscando otros puntos de corte #############################
# cutoff: Distancia de separación espacial hasta la cual los pares de puntos se incluyen en las estimaciones de semivarianza; 
#         De forma predeterminada, la longitud de la diagonal del cuadro que abarca los datos se divide por tres.
# width: intervalos de distancia en los que se agrupan los pares de puntos de datos para las estimaciones de semivarianza.
mvar <- lapply(1:4, function(i) {
  cf <- i*1000*2
  v_ <- variogram(evi ~ 1, data=ran_points, cutoff = cf, width=cf/15)
  plot(v_, main=cf,  plot.numbers = F)
})
do.call(grid.arrange, mvar)

v_map <- variogram(evi ~ 1, data=ran_points, cutoff = 8000, width=600, map=T) #, alpha = c(45))
plot(v_map)


# Variograma Anisotrópico  =================
v_a <- variogram(evi ~ 1, data=ran_points, cutoff = 8000, width=600, map=F, alpha=c(0, 45, 90, 135))
plot(v_a)


# Modelos de variograma #################
show.vgms()

# Ajuste Manual #########################
v_a_mod1 <- vgm(nugget=0.005, psill=0.005, range = 4000, model = "Sph")
plot(v_a, pl = F, model = v_a_mod1)

v_a_mod2 <- vgm(nugget=0.005, psill=0.009, range = 4000, model = "Exp")
plot(v_a, pl = F, model = v_a_mod2)

v_i_mod <- vgm(nugget=0.005, psill=0.009, range = 4000, model = "Exp")
plot(v_i, pl = F, model = v_i_mod)

plot(v_i, pl = T, model = v_i_mod)

# Ajuste Automático ####################
v_i_auto <- fit.variogram(v_i, v_i_mod)
plot(v_i, pl = T, model = v_i_auto)


# Ajuste -------------------------------
obs <- v_i$gamma
est_i <- variogramLine(v_i_mod, tail(attributes(v_i)$boundaries, n=1), length(obs))$gamma
est_i_auto <- variogramLine(v_i_auto, tail(attributes(v_i)$boundaries, n=1), length(obs))$gamma

# Metricas
rmse <- function(actual, predicted) {
  # RMSE: calcula error cuadratico medio entre observados y estimados
  se <- (actual - predicted)^2
  mse <- mean(se)
  return(sqrt(mse))
}

data.frame(cor=c(cor(obs, est_i), cor(obs, est_i_auto)),
           rmse=c(rmse(obs, est_i), rmse(obs, est_i_auto)),
           row.names=c("manual", "auto"))


# Actividades ---------------------------
#
# 1. Tome una muestra aleatoria de puntos y vea cómo varían los resultados ¿cómo cambian las conclusiones? 
#    Pruebe con un n de 500, 1000, 2000, 4000 y 8000
# 2. Probar diferentes valores de EVI/NDVI ¿cómo cambian las conclusiones a diferentes rangos seleccionados?
# 3. Si selecciona por tipo de cobertura de suelo ¿cómo varía el resultado?
# 4. Pruebe diferentes modelos de variograma y verifique cuál ajusta de mejor manera.
# 5. Basado en los experimentos anteriores ¿existe una dirección clara de cambio? 
#    ¿Cambia o depende mucho de la variable? ¿qué modelo es mejor?


#---------------------------------------
#----------- Parte 2 -------------------
#---------------------------------------

# Cálculo de variograma cruzado usando objetos gstat --------------

# Creación de objeto gstat para modelo de variograma
(vc_i <- variogram(evi ~ elevation, data = ran_points, cutoff = diag_/3, width = diag_/3/15))
plot(vc_i)

gstat


g <- gstat(NULL,"evi", evi~elevation * slope * aspect_sin * aspect_cos, ran_points)
g <- gstat(g, "ndvi", ndvi~elevation, ran_points)

(v <- variogram(g, cutoff = diag_/3, width = diag_/3/15)) # creating variogram models
plot(v)

gm = gstat(g, model = vgm(nugget=0.005, psill=0.009, range = 4000,  model = "Gau"), 
           fill.all = TRUE) #model variograms
g_fit = fit.lmc(v, gm) # fit models
plot(v, g_fit) # plot variograms and models

g_fit

v_map <- variogram(g, cutoff = diag_/3, width = diag_/3/15, map=T) # creating variogram models
plot(v_map, threshold = 5, col.regions = terrain.colors)


# Cálculo de coeficiente de codispersion ======
library(SpatialPack)

x <- opoints$evi
y <- opoints$elevation

coords <- opoints[, c("x", "y")]

# calcular el coeficiente de codispersión
(z <- codisp(x, y, coords))
plot(z)
