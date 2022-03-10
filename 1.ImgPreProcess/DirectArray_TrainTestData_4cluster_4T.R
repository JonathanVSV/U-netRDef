library(sf)
library(raster)
library(rray)
library(dplyr)
library(fasterize)
# library(imager)
library(reticulate)

# Check mask creation, 'cause class number is hard set
input_type <- "MS"
n_classes <- 2 # Without background = no change

channels <- ifelse(input_type == "MSSAR", 6, 
                   ifelse(input_type == "MS", 4, 2))

timeObs <- 4 #(2019 and 2020)

# Size of windows in pixels (training data)
img_width <- 256
img_height <- 256
# Size of windows in pixels (augumented training data)
img_width_exp <- 128
img_height_exp <- 128

# Number of 128pix-squares that are going to be obtained per 256 pix-squares
# In this case we are going to use 25 (5 x 5 128 x 128 pix) using a 32 pix offset
# This is cropping the image as 1-129, 1-129; 33-161, 1-129; 65-193, 1-129; 97-225, 1-129; 129-256, 1-129 
num_squares <- 3^2
probs_4crops <- 1 / (sqrt(num_squares)-1) 
# Number of mirrored images per 128pix-image
# num_mirrors <- 2
multip_im <- 1

#----------------------------------Load and pre process stuff-------------------------------
# Load images stack MS + SAR
im2019_1 <- stack("im2019MSSAR_stack_NAfill.tif")
im2019_2 <- stack("im2019MSSAR_rain1_stack_NAfill.tif")
im2019_3 <- stack("im2019MSSAR_rain2_stack_NAfill.tif")
im2020 <- stack("im2020MSSAR_stack_NAfill.tif")

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

# Traning data
# defor_patch <- st_read("D:/Drive/Jonathan_trabaggio/Doctorado/GeoInfo/Defor_points/Defor4Unet/defor_patch_valid.shp")
defor_patch <- st_read("defor_patch_valid_2022.shp")

# Make 128 x 128 areas for training
# Get centroids and project info to utm 15 n
# centroids <- defor_patch %>%
#   st_centroid() %>%
#   # Change to utm 15 N
#   st_transform(32615)
# 
# areas128 <- centroids %>%
#   # Get squares 256 x 256 px squares (10 m res)
#   st_buffer(dist = 1280,
#             endCapStyle = "SQUARE") %>%
#   # Select only field that will be y in models
#   dplyr::select(finid) %>%
#   # Eliminate defor in other years
#   filter(finid < 3) %>%
#   st_transform(4326) %>%
# # PRUEBA
#   slice(1:10)

# areas128 <- st_read("D:/Drive/Jonathan_trabaggio/Doctorado/GeoInfo/Defor_points/Defor4Unet/areas128.shp")
areas128 <- st_read("Grids_defor.shp")

# defor_im <- raster("Raster/deforImage_crop.tif")
defor_im <- raster("deforImage_crop.tif")

# plot(defor_im)
# Crop 128 px areas for input images
im2019_1_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2019_1,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

print(paste0("length im2019_1: ",length(im2019_1_tiles)))

# Crop 128 px areas for input images
im2019_2_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2019_2,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

print(paste0("length im2019_2: ",length(im2019_2_tiles)))

# Crop 128 px areas for input images
im2019_3_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2019_3,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

print(paste0("length im2019_3: ",length(im2019_3_tiles)))

im2020_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(im2020,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

print(paste0("length im 2020: ", length(im2020_tiles)))


# Crop 128 px areas for training
defor_tiles <- lapply(1:nrow(areas128), function(i){
  poly <- areas128 %>% slice(i)
  # Subset by poly
  temp <- crop(defor_im,poly)
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
})

print(paste0("length defor_tiles: ", length(defor_tiles)))

# Create an empty list to save cropped class info
defor_tiles_list <- vector(mode = "list", 
                           length = length(defor_tiles))

# Change masks to binary representations
# List with two dimensions, first index is number of image
# Second index is type of cover mask


# Change masks to binary representations
# List with two dimensions, first index is number of image
# Second index is type of cover mask
for (j in 1:length(defor_tiles_list)){
  defor_tiles_list[[j]] <- stack(lapply(0:n_classes, function(i){
    defor_tiles[[j]] == i
  }))
}

print(paste0("length defor tiles list: ", length(defor_tiles_list)))
# Plot
# plot(im2019_tiles[[7]])
# plot(im2020_tiles[[7]])
# plot(defor_tiles_list[[17]])

# ---------------Convert info to arrays----------------------------------------------------
# Change image (features data) to type array
cropped_im_matrt1 <- lapply(im2019_1_tiles, as.array)
cropped_im_matrt2 <- lapply(im2019_2_tiles, as.array)
cropped_im_matrt3 <- lapply(im2019_3_tiles, as.array)
cropped_im_matrt4 <- lapply(im2020_tiles, as.array)

# Change class info (labels data) to type array
cropped_mask_matr <- lapply(defor_tiles_list, as.array)

print(paste0("cropped mask_matr: ", length(cropped_mask_matr)))

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
                            (n_classes+1)))

# Fill the arrays
for(i in 1:length(cropped_im_matrt1)) {
  train_x_data[i,1,1:img_width,1:img_height,1:channels] <- cropped_im_matrt1[[i]]
  train_x_data[i,2,1:img_width,1:img_height,1:channels] <- cropped_im_matrt2[[i]]
  train_x_data[i,3,1:img_width,1:img_height,1:channels] <- cropped_im_matrt3[[i]]
  train_x_data[i,4,1:img_width,1:img_height,1:channels] <- cropped_im_matrt4[[i]]
}

for(i in 1:length(cropped_mask_matr)) {
  train_y_data[i,1,1:img_width,1:img_height,1:(n_classes+1)] <- cropped_mask_matr[[i]]
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
                                (n_classes+1)))

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
                                                1:(n_classes+1)]
      aux <- aux + 1
    }
  }
}

rm(train_x_data,train_y_data)

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
#                                   (n_classes+1)))
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

train_x_data_final <- train_x_data_128
train_y_data_final <- train_y_data_128

rm(train_x_data_128,train_y_data_128)

# ---------------Normalize Images----------------------------------------------------
# Before exporting all the data, let's calculate per band mean and SD to normalize the data
# Leave it without na.rm to check that no NA are left in the training data
# if there are NA U-net will break
mean_x <- apply(train_x_data_final, 5, function(x) mean(x))
sd_x <- apply(train_x_data_final, 5, function(x) sd(x))

write.csv(mean_x, paste0("Mean_x",input_type,"_4T.csv"), row.names = F)
write.csv(sd_x, paste0("sd_x",input_type,"_4T.csv"), row.names = F)

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
testers_rows<- sample(testers, floor(length(testers)*0.3))
testers_rows<-unlist(lapply(testers_rows, function(x) seq(x, x+(imgs_share-1),1)))

# Define test and training data
test_x_data_final <- train_x_data_final[testers_rows,,,,]
# Force drop = F, avoid eliminating single temporal dimension
test_y_data_final <- train_y_data_final[testers_rows,1,,,,drop = F]

train_x_data_final <- train_x_data_final[-testers_rows,,,,]
train_y_data_final <- train_y_data_final[-testers_rows,,,,,drop = F]

# ---------------Export to npz----------------------------------------------------
# Other option: saving files as numpy zip files
# This option was prefered as a single file can contain both training and test data
np <- import("numpy")

np$savez(paste0("Lacandona_Defor_FebApr_20192020_",input_type,"_Ene2022_4T",".npz"), 
         x_train = train_x_data_final, 
         y_train = train_y_data_final, 
         x_test = test_x_data_final, 
         y_test = test_y_data_final)
