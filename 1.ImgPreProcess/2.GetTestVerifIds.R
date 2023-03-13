library(sf)
library(raster)
# library(rray)
library(dplyr)
library(fasterize)
# library(imager)
library(reticulate)
library(dplyr)

# Check mask creation, 'cause class number is hard set
input_type <- "MSSAR"
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

# Grids used to separate train, test, verif sets
# Must have a field named true_id to identify each 
# area, used in the following steps
areas128 <- st_read("Grids_defor.shp")

# ---------------Test and Train sets----------------------------------------------------
train_x_data_final <- rep(areas128$true_id, each = imgs_share)
# Create train and test sets
# Number of images (128 x 128 px) that come from the same 256 x 256 image
imgs_share <- (multip_im * num_squares)

# Images that are going to be saved as test data; and thus, removed from the training data
# With this procedure we completely omit all the augumented images from a single
testers <- seq(1, length(train_x_data_final), imgs_share)

# Make sample reproducible
# Make sure it is the same seed as the one used in 
# 1.DirectArray_TrainTestData
set.seed(8)
testers_rows<- sample(testers, floor(length(testers)*0.4))
testers_rows<-unlist(lapply(testers_rows, function(x) seq(x, x+(imgs_share-1),1)))

# Define test and training data
test_x_data_final <- train_x_data_final[testers_rows]
# Force drop = F, avoid eliminating single temporal dimension
# test_y_data_final <- train_y_data_final[testers_rows]

# Verif dataset
# Make sample reproducible
verifers <- seq(1, length(test_x_data_final), imgs_share)
# Make sure to use same seed as the one used in 
# 1.DirectArray_TrainTestData
set.seed(10)
verifers_rows<- sample(verifers, floor(length(verifers)*0.5))
verifers_rows<-unlist(lapply(verifers_rows, function(x) seq(x, x+(imgs_share-1),1)))

verif_x_data_final <- test_x_data_final[verifers_rows]
# verif_y_data_final <- test_y_data_final[verifers_rows,1,,,,drop = F]

# Remove verifers from test dataset
test_x_data_final <- test_x_data_final[-verifers_rows] 
train_x_data_final <- train_x_data_final[-testers_rows]

unique(verif_x_data_final)
unique(test_x_data_final)

verif <- data.frame(true_id = unique(verif_x_data_final),
                    type = "verif")
test <- data.frame(true_id = unique(test_x_data_final),
                   type = "test")
train <- data.frame(true_id = unique(train_x_data_final),
                   type = "train")
df <- verif |>
  bind_rows(test) |>
  bind_rows(train)

grids_type <- areas128 |>
  left_join(df, by = "true_id")

st_write(grids_type, "Grids_defor_dataset.shp",
         append = FALSE)
# test_y_data_final <- test_y_data_final[-verifers_rows,1,,,,drop = F]

