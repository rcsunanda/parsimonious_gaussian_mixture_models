library(mclust)

data(coffee)
class <- coffee$class
table(class)

X <- coffee[,-1]
head(X)

# clPairs(X, class)
# BIC <- mclustBIC(X)
# plot(BIC)
# summary(BIC)

mod1 <- Mclust(X)
summary(mod1, parameters = TRUE)

adjustedRandIndex(coffee$Variety, mod1$classification)
