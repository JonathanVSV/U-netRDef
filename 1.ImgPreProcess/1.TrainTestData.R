library(sf)
library(raster)
library(rray)
library(dplyr)
library(fasterize)
library(reticulate)

# Variable definition
input_type <- "MSSAR"
n_classes <- 3 # no deforestation and two deforestation classes

# bands
channels <- ifelse(input_type == "MSSAR", 6, 
                   ifelse(input_type == "MS", 4, 2))

# Depth in time dimension
timeObs <- 4 #(2019 and 2020)

# Size of windows in pixels (training data)
img_width <- 256
img_height <- 256
# Size of windows in pixels (augumented training data)
img_width_exp <- 128
img_height_exp <- 128

# Number of 128pix-squares that are going to be obtained per 256 pix-squares
num_squares <- 3^2
# Probability values to make crops of image
probs_4crops <- 1 / (sqrt(num_squares)-1) 
# If mirrored images is used this can be a larger number
multip_im <- 1

#----------------------------------Load and pre process stuff-------------------------------
# Load images stack MS + SAR
im2019_1 <- stack("S1y2_9immedian_2A_6B_2019-02-01_2019-04-30_median_10mBandsMaxCCL100.tif")
im2019_2 <- stack("S1y2_9immedian_2A_6B_2019-05-01_2019-09-30_median_10mBandsMaxCCL100.tif")
im2019_3 <- stack("S1y2_9immedian_2A_6B_2019-10-01_2020-01-31_median_10mBandsMaxCCL100.tif")
im2020 <- stack("S1y2_9immedian_2A_6B_2020-02-01_2020-04-30_median_10mBandsMaxCCL100.tif")

# Depending on the input image type subset certain bands
if(input_type != "MSSAR"){
  if(input_type == "MS"){
    im2019_1 <- im2019_1[[1:4]]
    im2019_2 <- im2019_2[[1:4]]
    im2019_3 <- im2019_3[[1:4]]
    im2020 <- im2020[[1:4]]
  }else{
    im2019_1 <- im2019_1[[5:6]]
    im2019_2 <- im2019_2[[5:6]]
    im2019_3 <- im2019_3[[5:6]]
    im2020 <- im2020[[5:6]]
  }
}

# Manual delineated polygons
defor_patch <- st_read("defor_patch_valid_2022_revnov.shp")
# Rasterize the data using im2019_1 as template
defor_im <- fasterize(defor_patch,
                      im2019_1[[1]],
                      background = 0,
                      field = "id2022")

# Squared grids used to determine train test verif datasets
# Recommended to have an id field so it is easier to later
# identify areas used for train, test, verif
areas128 <- st_read("Grids_defor.shp")

# Cut all images according to the grids for train test verif
# Make sure that all areas consist of 128 x 128 areas (final crop)
# If you have more images in the temporal dimension, the images can be placed
# inside a list and perform this task as an lapply nested inside other lapply
im2019_1_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2019_1,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

# Crop 128 px areas for input images
im2019_2_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2019_2,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

# Crop 128 px areas for input images
im2019_3_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2019_3,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

im2020_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2020,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

# Crop the raster of manually delineated deforestations
defor_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(defor_im,poly)
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

# Create an empty list to save cropped class info
defor_tiles_list <- vector(mode = "list", 
                           length = length(defor_tiles))

# Change masks to binary representations
# List with two dimensions, first index is number of image
# Second index is type of cover mask
for (j in 1:length(defor_tiles_list)){
  defor_tiles_list[[j]] <- stack(lapply(0:(n_classes-1), function(i){
    defor_tiles[[j]] == i
  }))
}

# ---------------Convert info to arrays----------------------------------------------------
# Change image (features data) to type array
cropped_im_matrt1 <- lapply(im2019_1_tiles, as.array)
cropped_im_matrt2 <- lapply(im2019_2_tiles, as.array)
cropped_im_matrt3 <- lapply(im2019_3_tiles, as.array)
cropped_im_matrt4 <- lapply(im2020_tiles, as.array)

# Change class info (labels data) to type array
cropped_mask_matr <- lapply(defor_tiles_list, as.array)

# Remove unused objects
rm(im2019_1_tiles,im2019_2_tiles,im2019_3_tiles,im2020_tiles,defor_tiles_list,defor_tiles)

# Create empty arrays to fill it with the previous info
# Dims: batch, time, height, width, channels
train_x_data<-array(0,
                    dim = c(length(cropped_im_matrt1),
                            timeObs,
                            img_height,
                            img_width,
                            channels))
train_y_data<-array(0,
                    dim = c(length(cropped_mask_matr),
                            1,
                            img_height,
                            img_width,
                            (n_classes)))

# Fill the arrays
for(i in 1:length(cropped_im_matrt1)) {
  train_x_data[i,1,1:img_width,1:img_height,1:channels] <- cropped_im_matrt1[[i]]
  train_x_data[i,2,1:img_width,1:img_height,1:channels] <- cropped_im_matrt2[[i]]
  train_x_data[i,3,1:img_width,1:img_height,1:channels] <- cropped_im_matrt3[[i]]
  train_x_data[i,4,1:img_width,1:img_height,1:channels] <- cropped_im_matrt4[[i]]
}

for(i in 1:length(cropped_mask_matr)) {
  train_y_data[i,1,1:img_width,1:img_height,1:(n_classes)] <- cropped_mask_matr[[i]]
  # print(paste0("first array fill i run: ",i))
}

rm(cropped_im_matrt1,cropped_im_matrt2,cropped_mask_matr)

# ---------------Create 128x128 px arrays----------------------------------------------------
# Create 128 x 128 arrays
train_x_data_128<-array(0,
                        dim = c(dim(train_x_data)[1] * num_squares,
                                timeObs,
                                img_height_exp,
                                img_width_exp,
                                channels))
train_y_data_128<-array(0,
                        dim = c(dim(train_y_data)[1] * num_squares,
                                1,
                                img_height_exp,
                                img_width_exp,
                                (n_classes)))

# Vectors with starting positions to make 128 x 128 pix squares
temp <- unique(quantile(seq(1,img_width_exp+1), probs = seq(0, 1, probs_4crops)))
# The valuse that are going to be used to crop
crops <- c(temp)
crops

# Value to sum to each of the previous values
sum_pix <- img_height_exp - 1
aux <- 1

# Fill the empty arrays with the info, 256 x 256 images get cropped in the 128 x 128 px
for(i in 1:dim(train_x_data)[1]) {
  for(j in 1:length(crops)){
    for(k in 1:length(crops)){
      train_x_data_128[aux,,,,] <- train_x_data[i,
                                                1:timeObs,
                                                crops[j]:(crops[j]+sum_pix),
                                                crops[k]:(crops[k]+sum_pix),
                                                1:channels]
      aux <- aux + 1
    }
  }
}

aux <- 1
for(i in 1:dim(train_y_data)[1]) {
  for(j in 1:length(crops)){
    for(k in 1:length(crops)){
      train_y_data_128[aux,,,,] <- train_y_data[i,
                                                ,
                                                crops[j]:(crops[j]+sum_pix),
                                                crops[k]:(crops[k]+sum_pix),
                                                1:(n_classes)]
      aux <- aux + 1
    }
  }
}

rm(train_x_data,train_y_data)

# Optional part if augumentation includes mirrored images
# # ---------------Mirrored Images----------------------------------------------------
# # Get number of total images (original + 2 mirrors)
# multip_im <- num_mirrors + 1
# 
# # Create 128 x 128 empty arrays, self + 2 mirrors
# train_x_data_final<-array(0,
#                           dim = c(dim(train_x_data_128)[1] * multip_im,
#                                   timeObs,
#                                   img_height_exp,
#                                   img_width_exp,
#                                   channels))
# train_y_data_final<-array(0,
#                           dim = c(dim(train_y_data_128)[1] * multip_im,
#                                   1,
#                                   img_height_exp,
#                                   img_width_exp,
#                                   (n_classes)))
# 
# 
# aux <- 1
# 
# # Create mirrored images and fill the empty arrays
# for(i in 1:dim(train_x_data_128)[1]) {
#   train_x_data_final[aux,,,,] <- train_x_data_128[i,,,,]
#   aux = aux + 1
#   train_x_data_final[aux,,,,] <- rray_flip(train_x_data_128[i,,,,],axis = 2)
#   aux = aux + 1
#   train_x_data_final[aux,,,,] <- rray_flip(train_x_data_128[i,,,,],axis = 3)
#   aux = aux + 1
# }
# 
# aux <- 1
# 
# for(i in 1:dim(train_y_data_128)[1]) {
#   train_y_data_final[aux,,,,] <- train_y_data_128[i,,,,]
#   aux = aux + 1
#   train_y_data_final[aux,,,,] <- rray_flip(train_y_data_128[i,,,,],axis = 2)
#   aux = aux + 1
#   train_y_data_final[aux,,,,] <- rray_flip(train_y_data_128[i,,,,],axis = 3)
#   aux = aux + 1
# }
# 
# rm(train_x_data_128, train_y_data_128)


# --------------------------Avoid mirror images-------------------------------------
# Just pass the data from _data_128 to _data_final
train_x_data_final <- train_x_data_128
train_y_data_final <- train_y_data_128

rm(train_x_data_128,train_y_data_128)

# ---------------Standardize Images----------------------------------------------------
# Before exporting all the data, let's calculate per band mean and SD to normalize the data
# Leave it without na.rm to check that no NA are left in the training data
# if there are NA U-net will break
mean_x <- apply(train_x_data_final, 5, function(x) mean(x))
sd_x <- apply(train_x_data_final, 5, function(x) sd(x))

write.csv(mean_x, paste0("Mean_x",input_type,"_4T_revnov.csv"), row.names = F)
write.csv(sd_x, paste0("sd_x",input_type,"_4T_revnov.csv"), row.names = F)

# Normalize data (only features data)
for(i in 1:dim(train_x_data_final)[1]){
  for(j in 1:dim(train_x_data_final)[5]){
    train_x_data_final[i,,,,j] <- (train_x_data_final[i,,,,j] - mean_x[j]) / sd_x[j]
  }
}

#---------------Image checks----------------------------------------------------
#Check min and max values
apply(train_x_data_final, 5, function(x) min(x))
apply(train_x_data_final, 5, function(x) max(x))

# ---------------Test and Train sets----------------------------------------------------
# Create train and test sets
# Number of images (128 x 128 px) that come from the same 256 x 256 image
imgs_share <- (multip_im * num_squares)

# Images that are going to be saved as test data; and thus, removed from the training data
# With this procedure we completely omit all the augumented images from a single
# 256 x 256 px square.
testers <- seq(1, dim(train_x_data_final)[1], imgs_share)

# Make sample reproducible
set.seed(8)
# Set training data as 60 % of data ( 1 - 0.6 = 0.4)
testers_rows<- sample(testers, floor(length(testers)*0.4))
testers_rows<-unlist(lapply(testers_rows, function(x) seq(x, x+(imgs_share-1),1)))

# Define test and training data
test_x_data_final <- train_x_data_final[testers_rows,,,,]
# Force drop = F, avoid eliminating single temporal dimension
test_y_data_final <- train_y_data_final[testers_rows,1,,,,drop = F]

# Verif dataset
# Make sample reproducible
verifers <- seq(1, dim(test_x_data_final)[1], imgs_share)
set.seed(10)
# Set verification data as half (0.5) of the test data, i.e., 20 % of total
verifers_rows<- sample(verifers, floor(length(verifers)*0.5))
verifers_rows<-unlist(lapply(verifers_rows, function(x) seq(x, x+(imgs_share-1),1)))

verif_x_data_final <- test_x_data_final[verifers_rows,,,,]
verif_y_data_final <- test_y_data_final[verifers_rows,1,,,,drop = F]

# Remove verifers from test dataset
test_x_data_final <- test_x_data_final[-verifers_rows,,,,] 
test_y_data_final <- test_y_data_final[-verifers_rows,1,,,,drop = F]

# Train dataset
# Set training data as the data not contained in test and verif data
train_x_data_final <- train_x_data_final[-testers_rows,,,,]
train_y_data_final <- train_y_data_final[-testers_rows,,,,,drop = F]

# ---------------Export to npz----------------------------------------------------
# Other option: saving files as numpy zip files
# This option was prefered as a single file can contain both training and test data
np <- import("numpy")

# Save train test verif datasets
np$savez(paste0("Lacandona_Defor_FebApr_20192020_",input_type,"_Ene2022_4T_revnov",".npz"), 
         x_train = train_x_data_final, 
         y_train = train_y_data_final, 
         x_test = test_x_data_final, 
         y_test = test_y_data_final,
         x_verif = verif_x_data_final,
         y_verif = verif_y_data_final)

