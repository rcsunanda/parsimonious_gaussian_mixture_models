library(pgmm)

# Wine clustering example with three random starts and the CUU model.

# Load data
data("coffee")
class <- coffee$class
table(class)
x<-coffee[,-c(1,2)] # Remove class column (Variety) and Country column
#x<-scale(x)
head(x)

# Fit PGMM models
# models_to_run <- c("UUU", "UUC", "UCU", "UCC", "CUU", "CUC", "CCU", "CCC")
models_to_run <- c("UCU", "UCC", "CUU", "CUC", "CCU", "CCC")
coffee_clust<-pgmmEM(x, rG=1:5, rq=1:5, zstart=1, loop=3, modelSubset=models_to_run)

# Take the max BIC value among the different PGMMs for each (G,q) pair
bic_array = simplify2array(coffee_clust$bic) # Convert to 3D array dim = [5, 5, 6] (rows=G, cols=q, depth=PGGM models)
max_bic = apply(bic_array, 1:2, max) # Max along the 3rd dim (PGMM models)

# Draw heatmap of max BIC
library("lattice")
heatmap(t(max_bic), Rowv=NA, Colv=NA, scale="none", xlab="G", ylab="q", main="BIC values heatmap") # do transpose (x_axis=G, y_axis=q)

# Print clustering table (map)
table(coffee[,1], coffee_clust$map)

# Compute Rand indexes
#library(mclust)
#adjustedRandIndex(coffee$Variety, wine_clust$map)
library(EMCluster)
RRand(coffee$Variety, coffee_clust$map)

# Fit Standard GMM (mclust ) model
mod1 <- Mclust(x, G=1:5)
summary(mod1, parameters = FALSE)

table(coffee[,1], mod1$classification)

RRand(coffee$Variety, mod1$classification)
#adjustedRandIndex(coffee$Variety, mod1$classification)
