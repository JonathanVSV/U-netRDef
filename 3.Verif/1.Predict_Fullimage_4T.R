library(keras)
library(dplyr)
library(reticulate)
library(raster)
library(sf)

source("multilabel_dice_coefficient.R")

##---------------------Constant def---------------------------------------
# Select model that had the best performance
model_location <- paste0("DeforhypTune_16batches_MSSAR_4T_revnov2022-11-24/2022-11-25T08-47-01Z/U128model_f1lossMSSARfilters64Epochs9layers4dropout0.1_lr1e-04_adam_2022-11-21.h5")
# Image
imagery <- "MSSAR"
# Label data
n_classes <- 3 
time_obs <- 4
channels <- ifelse(imagery == "MSSAR", 6, 
                   ifelse(imagery == "MS", 4, 2))

# Training hyperparameters
batch_size <- 16
learn_rate <- 1e-4
epochs <- 9
dropout <- 0.1
filters_firstlayer <- 64
num_layers <- 4

activation_func_out <- "softmax" 

# Image dimensions
img_width <- 128
img_height <- 128

# Image dimensions for export (probably redundant)
img_width_pred <- 128
img_height_pred <- 128

# --------------------------Read first set of tiles-------------------------------------------------
# Numpy zip file option
np <- import("numpy")

# Load npz
npz2 <- np$load(paste0("Lacandona_Defor_fullImg_a_",imagery,"_4T_revnov.npz"))

test_x_data <- npz2$f[["x_test"]]

###---------------------------Model Definition---------------------------------------
f1score <- custom_metric("f1score", function(y_true, y_pred, smooth = 1) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})

# sum of binary_crossentropy and soft-Dice
cce_dice_loss <- custom_metric("cce_dice_loss", function(y_true, y_pred) {
  result <- loss_categorical_crossentropy(y_true, y_pred) +
    (1 - multilabel_dice_coefficient(y_true, y_pred))
  return(result)
})

model <- load_model_hdf5(model_location,
                         custom_objects = c("f1score" = f1score,
                                            "multilabel_dice_coefficient" = multilabel_dice_coefficient,
                                            "cce_dice_loss" = cce_dice_loss))

# PRedicted images list
rast_pred_list <- vector(length = dim(test_x_data)[1], mode = "list")
# Probabilities raster list
rast_prob_list <- vector(length = dim(test_x_data)[1], mode = "list")

# Get predicted class and probability
for(j in 1:dim(test_x_data)[1]){
  image_real<-array_reshape(test_x_data[j,,,,1:channels],c(1,time_obs,img_width_pred,img_height_pred,channels))
  
  result_pred <- predict(
    object = model,
    batch_size = batch_size,
    x = image_real,
    steps = 10,
    verbose = 1)
  
  #See predicted image
  image_pred <- array_reshape(result_pred[1,,,,1:n_classes],c(1,1,img_width_pred,img_height_pred,n_classes))
  
  # rast_list[[j]] <- image_pred
  
  # Get max probability value by cell in all the classes, i.e., dim 3
  maxval_bycell <- apply(image_pred[1,,,,] , c(1,2) , 
                         function(x) 
                           ifelse(all(is.na(x)), NA, max(x, na.rm = TRUE))) 
  
  resul2 <- lapply(1:n_classes, function(i){
    # Set class as the pixel that match the max prob (remember this comes from softmax classificator 0-1)
    temp <- image_pred[1,,,,i] == maxval_bycell
    # temp <- image_pred[1,,,i] >= 0.5
    # temp2 <- image_pred[1,,,i] < 0.5 & image_pred[1,,,i] >= 0.1
    temp[temp == TRUE] <- i
    # temp[temp2 == TRUE] <- n_classes + 1
    temp[temp == FALSE] <- 0
    as.matrix(temp)
    
  })
  
  # Pensar en si podrían haber empates
  resul2 <- Reduce('+', resul2)
  
  # To see spatial distribution of errors
  # rast_list[[j]] <- image_real
  rast_pred_list[[j]] <- resul2
  rast_prob_list[[j]] <- maxval_bycell

}

# ------------------Fill layer into original positions-----------------------------------
# Locations of image files
optic_names <- "S1y2_9immedian_2A_6B_2019-02-01_2019-04-30_median_10mBandsMaxCCL100.tif"

# Read images as stacks
optic <- stack(optic_names)

dim(optic)

# Create splits vector, to rebuild original image
splits <- c(floor(dim(optic)[1] / 128),floor(dim(optic)[2] / 128))

x_splits <- seq(1, (floor(dim(optic)[1] / 128) * 128), 128)
y_splits <- seq(1, (floor(dim(optic)[2] / 128) * 128), 128)

# Create empty arrays to fill it with the previous info
pred_im <- array(0,
                   dim = c(splits[1] * 128,
                           splits[2] * 128))
prob_im <- array(0,
                 dim = c(splits[1] * 128,
                         splits[2] * 128))

# Fill the array
aux <- 1
for(i in 1:(length(x_splits)-1)){
  for(j in 1:(length(y_splits)-1)){
    pred_im[x_splits[i]:(x_splits[i]+127),y_splits[j]:(y_splits[j]+127)] <- rast_pred_list[[aux]]
    prob_im[x_splits[i]:(x_splits[i]+127),y_splits[j]:(y_splits[j]+127)] <- rast_prob_list[[aux]]
    aux <- aux+1
  }    
}

pred_im <- raster(as.matrix(pred_im))
prob_im <- raster(as.matrix(prob_im))

# ----------------------Set spatial info into the "spatialess raster"-----------------------------------
# Get exactly the subset of the original image as the ones included in the tile construction
# Remember that we used a floor to just get the number of rows and columns that were exactly divided by 128
optic <- crop(optic, extent(optic, 1, x_splits[length(x_splits)]+127, 1, y_splits[length(y_splits)]+127))

extent(pred_im) <- extent(optic)
crs(pred_im) <- crs(optic)

extent(prob_im) <- extent(optic)
crs(prob_im) <- crs(optic)

res(pred_im) 
res(optic)

# Write rasters
writeRaster(pred_im,
            paste0("FullImageClassification_a_",imagery,"_revnov.tif"),
            format = "GTiff",
            overwrite = T)

writeRaster(prob_im,
            paste0("FullImageProbabilities_a_",imagery,"_revnov.tif"),
            format = "GTiff",
            overwrite = T)

## ----------------Second grid--------------------------------------------------------------------
# --------------------------Read first set of tiles-------------------------------------------------
# Do the same but with grid B
# Load npz
npz2 <- np$load(paste0("Lacandona_Defor_fullImg_b_",imagery,"_4T_revnov.npz"))

test_x_data <- npz2$f[["x_test"]]

# PRedicted images
rast_pred_list <- vector(length = dim(test_x_data)[1], mode = "list")
# Probabilities raster
rast_prob_list <- vector(length = dim(test_x_data)[1], mode = "list")

# Accuracy
for(j in 1:dim(test_x_data)[1]){
  image_real<-array_reshape(test_x_data[j,,,,1:channels],c(1,time_obs,img_width_pred,img_height_pred,channels))
  
  result_pred <- predict(
    object = model,
    batch_size = batch_size,
    x = image_real,
    steps = 10,
    verbose = 1)
  
  #See predicted image
  image_pred <- array_reshape(result_pred[1,,,,1:n_classes],c(1,1,img_width_pred,img_height_pred,n_classes))
  
  # rast_list[[j]] <- image_pred
  
  # Get max probability value by cell in all the classes, i.e., dim 3
  maxval_bycell <- apply(image_pred[1,,,,] , c(1,2) , 
                         function(x) 
                           ifelse(all(is.na(x)), NA, max(x, na.rm = TRUE))) 
  
  resul2 <- lapply(1:n_classes, function(i){
    # Set class as the pixel that match the max prob (remember this comes from softmax classificator 0-1)
    temp <- image_pred[1,,,,i] == maxval_bycell
    # temp <- image_pred[1,,,i] >= 0.5
    # temp2 <- image_pred[1,,,i] < 0.5 & image_pred[1,,,i] >= 0.1
    temp[temp == TRUE] <- i
    # temp[temp2 == TRUE] <- n_classes + 1
    temp[temp == FALSE] <- 0
    as.matrix(temp)
    
  })
  
  # Pensar en si podrían haber empates
  resul2 <- Reduce('+', resul2)
  
  # To see spatial distribution of errors
  # rast_list[[j]] <- image_real
  rast_pred_list[[j]] <- resul2
  rast_prob_list[[j]] <- maxval_bycell
  
}

rm(test_x_data)

# ------------------Fill layer into original positions-----------------------------------
# Locations of image files
optic_names <- "S1y2_9immedian_2A_6B_2019-02-01_2019-04-30_median_10mBandsMaxCCL100.tif"


# Read images as stacks
optic <- stack(optic_names)

dim(optic)

# Get original splits as we are filling the raster from starting position 1, but starting to fill values in 65. In the end the splits rows and columns are referenced to the original window of analysis.
splits_orig <- c(floor(dim(optic)[1] / 128),floor(dim(optic)[2] / 128))

x_splits_orig <- seq(1, (floor(dim(optic)[1] / 128) * 128), 128)
y_splits_orig <- seq(1, (floor(dim(optic)[2] / 128) * 128), 128)

# Less splits as its starts on 65
splits <- c(floor((dim(optic)[1]-65) / 128),floor((dim(optic)[2]-65) / 128))

x_splits <- seq(65, (floor((dim(optic)[1]-65) / 128) * 128), 128)
y_splits <- seq(65, (floor((dim(optic)[2]-65) / 128) * 128), 128)

# Create empty arrays to fill it with the previous info
pred_im <- array(0,
                 dim = c(splits[1] * 128,
                         splits[2] * 128))
prob_im <- array(0,
                 dim = c(splits[1] * 128,
                         splits[2] * 128))

# Fill the array
aux <- 1
for(i in 1:(length(x_splits)-1)){
  for(j in 1:(length(y_splits)-1)){
    pred_im[x_splits[i]:(x_splits[i]+127),y_splits[j]:(y_splits[j]+127)] <- rast_pred_list[[aux]]
    prob_im[x_splits[i]:(x_splits[i]+127),y_splits[j]:(y_splits[j]+127)] <- rast_prob_list[[aux]]
    aux <- aux+1
  }    
}


pred_im <- raster(as.matrix(pred_im))
prob_im <- raster(as.matrix(prob_im))

# Remove unused objects
rm(rast_pred_list, rast_prob_list)

# -----------------------Set spatial info into the "spatialess raster"---------------------------------
# Get exactly the subset of the original image as the ones included in the shard construction
# Remember that we used a floor to just get the number of rows and columns that were exactly divided by 128
# Remove last entry that no longer fits the original data because of the 64 offset
# Form: raster, x min, x max, y min, y max.

optic <- crop(optic, extent(optic, 1, x_splits_orig[length(x_splits_orig)]+127, 1, y_splits_orig[length(y_splits_orig)]+127))

# Create shp for extent
extent(pred_im) <- extent(optic)
crs(pred_im) <- crs(optic)

extent(prob_im) <- extent(optic)
crs(prob_im) <- crs(optic)

res(pred_im) 
res(optic)

writeRaster(pred_im,
            paste0("FullImageClassification_b_",imagery,"_revnov.tif"),
            format = "GTiff",
            overwrite = T)

writeRaster(prob_im,
            paste0("FullImageProbabilities_b_",imagery,"_revnov.tif"),
            format = "GTiff",
            overwrite = T)
