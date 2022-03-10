library(raster)

max_class <- 3
min_class <- 1

# Read images
Class_a <- raster("FullImageClassification_a_MSSAR.tif")
Class_b <- raster("FullImageClassification_b_MSSAR.tif")
Prob_a <- raster("FullImageProbabilities_a_MSSAR.tif")
Prob_b <- raster("FullImageProbabilities_b_MSSAR.tif")

plot(Prob_a)

Probs_merge <- stack(Prob_a, Prob_b)
plot(Probs_merge)

# Get max value in x, y, and attributes (1:3)
Prob_max <- calc(Probs_merge, function(x) max(x, na.rm = T))


# plot(Prob_max)

# writeRaster(Prob_max, "Max_prob.tif", format = "GTiff")

# Read Raster
# Prob_max <- raster("Max_prob.tif")

Prob_a_pass <- Prob_a == Prob_max
# plot(Prob_a_pass)
Prob_b_pass <- Prob_b == Prob_max
# plot(Prob_b_pass)

Class_a <- mask(Class_a, Prob_a_pass)
Class_b <- mask(Class_b, Prob_b_pass)

Prob_a <- mask(Prob_a, Prob_a_pass)
Prob_b <- mask(Prob_b, Prob_b_pass)

Classes <- stack(Class_a, Class_b)
Probs <- stack(Prob_a, Prob_b)

Classes <- calc(Classes, function(x) max(x, na.rm = T))
Probs <- calc(Probs, function(x) max(x, na.rm = T))

# Remove shitty pixels
Classes[Classes>max_class | Classes < min_class] <- NA
# plot(Classes)
# plot(Probs)

writeRaster(Classes, "Final_Class_ensemble.tif", format = "GTiff", overwrite = T)
writeRaster(Probs, "Final_Probs_ensemble.tif", format = "GTiff", overwrite = T)