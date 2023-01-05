library(sf)
library(raster)
library(rray)
library(dplyr)
library(fasterize)
library(reticulate)

# Variables definition
# Very similar to the variables in 1.DirectArray_TrainTestData_4cluster_4T
input_type <- "MSSAR"
n_classes <- 3

channels <- ifelse(input_type == "MSSAR", 6, 
                   ifelse(input_type == "MS", 4, 2))
timeObs <- 4

# Size of windows in pixels (training data)
img_width <- 256
img_height <- 256
# Size of windows in pixels (augumented training data)
img_width_exp <- 128
img_height_exp <- 128

# Helper variable
nless1 <- img_width_exp-1 

# Number of 128pix-squares that are going to be obtained per 256 pix-squares
num_squares <- 3^2
# probs_4crops <- 1 / (sqrt(num_squares)-1) 
# Number of mirrored images per 128pix-image
# num_mirrors <- 2
multip_im <- 1

#----------------------First graticule-------------------------------------------------

# Load images stack MS + SAR
im2019_1 <- stack("S1y2_9immedian_2A_6B_2019-02-01_2019-04-30_median_10mBandsMaxCCL100.tif")
im2019_2 <- stack("S1y2_9immedian_2A_6B_2019-05-01_2019-09-30_median_10mBandsMaxCCL100.tif")
im2019_3 <- stack("S1y2_9immedian_2A_6B_2019-10-01_2020-01-31_median_10mBandsMaxCCL100.tif")
im2020 <- stack("S1y2_9immedian_2A_6B_2020-02-01_2020-04-30_median_10mBandsMaxCCL100.tif")

dim1 <- dim(im2019_1)[1]
dim2 <- dim(im2019_1)[2]

# Get the number of splits (128 x 128 pix) to cover the complete image
splits <- c(floor(dim1 / img_height_exp),floor(dim2 / img_width_exp))

# Create vector with sequence of numbers
x_splits <- seq(1, (floor(dim1 / img_height_exp) * img_height_exp), img_height_exp)
y_splits <- seq(1, (floor(dim2 / img_width_exp) * img_width_exp), img_width_exp)

# Transform images into arrays
im2019_1 <- as.array(im2019_1)
im2019_2 <- as.array(im2019_2)
im2019_3 <- as.array(im2019_3)
im2020 <- as.array(im2020)

# Create empty arrays to fill it with the previous info
pred_x_data<-array(0,
                   dim = c(length(x_splits) * length(y_splits),
                           timeObs,
                           img_width_exp,
                           img_width_exp,
                           channels))

# Fill the arrays
aux <- 1
for(i in 1:(length(x_splits)-1)){
  for(j in 1:(length(y_splits)-1)){
    pred_x_data[aux,1,1:img_height_exp,1:img_width_exp,1:channels] <- im2019_1[x_splits[i]:(x_splits[i]+nless1),
                                                     y_splits[j]:(y_splits[j]+nless1),
                                                     1:channels]
    pred_x_data[aux,2,1:img_height_exp,1:img_width_exp,1:channels] <- im2019_2[x_splits[i]:(x_splits[i]+nless1),
                                                          y_splits[j]:(y_splits[j]+nless1),
                                                          1:channels]
    pred_x_data[aux,3,1:img_height_exp,1:img_width_exp,1:channels] <- im2019_3[x_splits[i]:(x_splits[i]+nless1),
                                                          y_splits[j]:(y_splits[j]+nless1),
                                                          1:channels]
    pred_x_data[aux,4,1:img_height_exp,1:img_width_exp,1:channels] <- im2020[x_splits[i]:(x_splits[i]+nless1),
                                                          y_splits[j]:(y_splits[j]+nless1),
                                                          1:channels]
    aux <- aux+1
  }    
}

# once we've got the arrays we need to normalize the data according to mean and sd values
# generated in 1.DirectArray_TrainTestData_4cluster_4T
bands_mean <- read.csv(paste0("Mean_x",input_type,"_4T_revnov.csv"))[,1]
bands_sd <- read.csv(paste0("sd_x",input_type,"_4T_revnov.csv"))[,1]

# Normalize data (only features data)
for(i in 1:dim(pred_x_data)[1]){
  for(j in 1:dim(pred_x_data)[5]){
    pred_x_data[i,,,,j] <- (pred_x_data[i,,,,j] - bands_mean[j]) / bands_sd[j]
  }
}


# Save the complete image divided in tiles of 128-pixel squares of grid A
np <- import("numpy")

np$savez(paste0("Lacandona_Defor_fullImg_a_",input_type,"_4T_revnov.npz"), 
         x_test = pred_x_data)

# -----------------------------Second graticule-----------------------------------------
# Make second grid with a vertical and horizontal offset of hald the size of the tiles
# i.e., (128 / 2) + 1 = 65.
splits <- c(floor((dim1-65) / img_height_exp),floor((dim2-65) / img_width_exp))

x_splits <- seq(65, (floor((dim1-65) / img_height_exp) * img_height_exp), img_height_exp)
y_splits <- seq(65, (floor((dim2-65) / img_width_exp) * img_width_exp), img_width_exp)

# Create empty arrays to fill it with the previous info
pred_x_data<-array(0,
                   dim = c(length(x_splits) * length(y_splits),
                           timeObs,
                           img_width_exp,
                           img_width_exp,
                           channels))

# Fill the arrays
aux <- 1
for(i in 1:(length(x_splits)-1)){
  for(j in 1:(length(y_splits)-1)){
    pred_x_data[aux,1,1:img_height_exp,1:img_width_exp,1:channels] <- im2019_1[x_splits[i]:(x_splits[i]+nless1),
                                                                               y_splits[j]:(y_splits[j]+nless1),
                                                                               1:channels]
    pred_x_data[aux,2,1:img_height_exp,1:img_width_exp,1:channels] <- im2019_2[x_splits[i]:(x_splits[i]+nless1),
                                                                               y_splits[j]:(y_splits[j]+nless1),
                                                                               1:channels]
    pred_x_data[aux,3,1:img_height_exp,1:img_width_exp,1:channels] <- im2019_3[x_splits[i]:(x_splits[i]+nless1),
                                                                               y_splits[j]:(y_splits[j]+nless1),
                                                                               1:channels]
    pred_x_data[aux,4,1:img_height_exp,1:img_width_exp,1:channels] <- im2020[x_splits[i]:(x_splits[i]+nless1),
                                                        y_splits[j]:(y_splits[j]+nless1),
                                                        1:channels]
    aux <- aux+1
  }    
}

# Normalize data (only features data); already read mean and sd files
for(i in 1:dim(pred_x_data)[1]){
  for(j in 1:dim(pred_x_data)[5]){
    pred_x_data[i,,,,j] <- (pred_x_data[i,,,,j] - bands_mean[j]) / bands_sd[j]
  }
}

# Save the complete image divided in tiles of 128-pixel squares of grid B
np$savez(paste0("Lacandona_Defor_fullImg_b_",input_type,"_4T_revnov.npz"), 
         x_test = pred_x_data)
