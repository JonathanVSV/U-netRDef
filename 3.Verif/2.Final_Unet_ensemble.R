library(raster)

imagery <- "MSSAR"

max_class <- 3
min_class <- 1

# Read images
Class_a <- raster(paste0("FullImageClassification_a_",imagery,"_revnov.tif"))
Class_b <- raster(paste0("FullImageClassification_b_",imagery,"_revnov.tif"))
Prob_a <- raster(paste0("FullImageProbabilities_a_",imagery,"_revnov.tif"))
Prob_b <- raster(paste0("FullImageProbabilities_b_",imagery,"_revnov.tif"))

plot(Prob_a)

Probs_merge <- stack(Prob_a, Prob_b)

# Get max value in x, y, and attributes (1:3)
Prob_max <- calc(Probs_merge, function(x) max(x, na.rm = T))

Prob_a_pass <- Prob_a == Prob_max
Prob_b_pass <- Prob_b == Prob_max

# Mask low probability values in grid A or B
Class_a <- mask(Class_a, Prob_a_pass)
Class_b <- mask(Class_b, Prob_b_pass)

Prob_a <- mask(Prob_a, Prob_a_pass)
Prob_b <- mask(Prob_b, Prob_b_pass)

# Stack grid A and B predictions
Classes <- stack(Class_a, Class_b)
Probs <- stack(Prob_a, Prob_b)

# Get max values from stack, should correspond to non-NA values
Classes <- calc(Classes, function(x) max(x, na.rm = T))
Probs <- calc(Probs, function(x) max(x, na.rm = T))

# Remove shitty pixels
Classes[Classes > max_class | Classes < min_class] <- NA

writeRaster(Classes, paste0("Final_Class_ensemble_",imagery,"_revnov.tif"), format = "GTiff", overwrite = T)
writeRaster(Probs, paste0("Final_Probs_ensemble_",imagery,"_revnov.tif"), format = "GTiff", overwrite = T)