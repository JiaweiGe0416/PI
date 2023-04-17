### Reading FRED-MD data 
require(ggplot2)
require(corrr)
require(ggcorrplot)
library(FactoMineR)
require(factoextra)

data = read.csv('/Users/jiawei/Dropbox (Princeton)/research/prediction band/mycode/paper code/FRED-MD.csv')

## remove first and last row 
data_1 = data[-c(1, nrow(data)),]
col_na = NULL
for(j in 1:ncol(data_1))
  col_na[j] = sum(is.na(data_1[, j]))

## Remove the column with >= 15 NAs 

data_2 = data_1[ ,-which(col_na > 14)]

## Remove all the rows with at least one NA 
row_na = NULL
for(j in 1:nrow(data_2))
  row_na[j] = sum(is.na(data_2[j,]))

data_3 = data_2[-which(row_na > 0),]



#### Data normalization #### 
X_val_Yihong = data_3[, !names(data_3) %in% c("UNRATE", "HOUST", "FEDFUNDS")]
X_val = X_val_Yihong[,-1] ## Remove the dates 
X_val = matrix(unlist(X_val), nrow(X_val), ncol(X_val)) # convert to a matrix 
X_val_scaled = scale(X_val)
write.csv(X_val_scaled, file = "Scaled_values_of_X.csv")

Y_val_Yihong = data_3[, names(data_3) %in% c("UNRATE", "HOUST", "FEDFUNDS")]
Y_val = matrix(unlist(Y_val_Yihong), nrow(Y_val_Yihong), ncol(Y_val_Yihong))
Y_val_scaled = scale(Y_val)
write.csv(Y_val_scaled, file = "Scaled_values_of_Y.csv")

corr_matrix <- cor(X_val_scaled)
#ggcorrplot(corr_matrix)

X.pca = princomp(corr_matrix)
summary(X.pca)
fviz_eig(X.pca, addlabels = TRUE) ## for scree plot
## Here it seems top three varibles are sufficient. 

X_val_reduced = X_val_scaled %*% X.pca$loadings[,1]
write.csv(X_val_reduced, file = "X_pca_top1.csv")

