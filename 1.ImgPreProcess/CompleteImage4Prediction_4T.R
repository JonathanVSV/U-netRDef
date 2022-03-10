library(sf)
library(raster)
library(rray)
library(dplyr)
library(fasterize)
# library(imager)
library(reticulate)

# Check mask creation, 'cause class number is hard set
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

nless1 <- img_width_exp-1 

# Number of 128pix-squares that are going to be obtained per 256 pix-squares
# In this case we are going to use 25 (5 x 5 128 x 128 pix) using a 32 pix offset
# This is cropping the image as 1-129, 1-129; 33-161, 1-129; 65-193, 1-129; 97-225, 1-129; 129-256, 1-129 
num_squares <- 3^2
probs_4crops <- 1 / (sqrt(num_squares)-1) 
# Number of mirrored images per 128pix-image
# num_mirrors <- 2
multip_im <- 1

#----------------------First graticule-------------------------------------------------

# Load images stack MS + SAR
im2019_1 <- stack("im2019MSSAR_stack_NAfill.tif")
im2019_2 <- stack("im2019MSSAR_rain1_stack_NAfill.tif")
im2019_3 <- stack("im2019MSSAR_rain2_stack_NAfill.tif")
im2020 <- stack("im2020MSSAR_stack_NAfill.tif")

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

print("pred_x_data")
print(pred_x_data[1:10,1,1:10,1:10,1])

# plot(as.raster(pred_x_data[2700,1:128,1:128,1:3]/2000))

# once we've got the arrays we need to normalize the data according to mean and sd values
bands_mean <- read.csv(paste0("Mean_x",input_type,"_4T.csv"))[,1]
bands_sd <- read.csv(paste0("sd_x",input_type,"_4T.csv"))[,1]

print(paste0("bands_mean: ", bands_mean))
print(paste0("bands_sd: ", bands_sd))

# Normalize data (only features data)
for(i in 1:dim(pred_x_data)[1]){
  for(j in 1:dim(pred_x_data)[5]){
    pred_x_data[i,,,,j] <- (pred_x_data[i,,,,j] - bands_mean[j]) / bands_sd[j]
  }
}

print("pred_x_data")
print(pred_x_data[1:10,1,1:10,1:10,1])

# plot(as.raster((pred_x_data[2700,1:128,1:128,1:3]+2)/4))

# Other option: saving files as numpy zip files
# This option was prefered as a single file can contain both training and test data
np <- import("numpy")

print(paste0("dims pred_x_data: ", dim(pred_x_data)))

np$savez(paste0("Lacandona_Defor_fullImg_a_",input_type,"_4T.npz"), 
         x_test = pred_x_data)

# -----------------------------Second graticule-----------------------------------------
# No need to reload and crop optic and radar. Skip that parte (commented section)

# one less than the original split because this starts with an offset of 64, son 1 tile less
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

# Other option: saving files as numpy zip files
# This option was prefered as a single file can contain both training and test data
# np <- import("numpy")

np$savez(paste0("Lacandona_Defor_fullImg_b_",input_type,"_4T.npz"), 
         x_test = pred_x_data)
