library(pgmm)

# Wine clustering example with three random starts and the CUU model.

data("wine")
x<-wine[,-1]
x<-scale(x)

wine_clust<-pgmmEM(x, rG=1:4, rq=1:4, zstart=1, loop=3, modelSubset=c("CUU"))

table(wine[,1], wine_clust$map)
adjustedRandIndex(wine$Type, wine_clust$map)
