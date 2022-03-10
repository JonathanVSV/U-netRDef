# library(unet)
# libraries we're going to need later
library(keras)
# library(tfdatasets)
# library(rsample)
library(dplyr)
# library(tidyr)
# library(tibble)
library(reticulate)
library(tfruns)
library(raster)
# library(yardstick)
# library(rhdf5)

source("multilabel_dice_coefficient.R")

# Best Optic Radar
model_location <- paste0("DeforhypTune_16batches_MSSAR_4T2022-02-04/2022-02-07T06-35-51Z/U128model_f1lossMSSARfilters64Epochs9layers5dropout0.1_lr1e-04_adam_2022-02-04.h5")

# El que le sigue de 4 hidden layers
# DeforhypTune_16batches_MSSAR_4T2022-02-04/2022-02-05T00-06-57Z

# Imagen
imagery <- "MSSAR"

# Label data
n_classes <- 3 # 0 no defor, 1 defor de old forest, 2 defor de plantations o secondary forest
time_obs <- 4
channels <- ifelse(imagery == "MSSAR", 6, 
                   ifelse(imagery == "MS", 4, 2))

print("Checker")
print(paste0("input: ", imagery, ", channels: ", channels))

# Training data
batch_size <- 16
learn_rate <- 1e-4
epochs <- 9
dropout <- 0.1
filters_firstlayer <- 64
num_layers <- 5

# Los demas params
activation_func_out <- "softmax"
# 
# #Dimensiones de las imagenes en cols y rows, sacarlas de QGIS
img_width <- 128
img_height <- 128
# 
# #Este es el tamaño que sale después de hacer las convoluciones sin utlilizar padding; checar summary(model) pa ver las dimensiones finales
img_width_pred <- 128
img_height_pred <- 128

# Numpy zip file option
np <- import("numpy")

# Load npz
npz2 <- np$load(paste0("Lacandona_Defor_FebApr_20192020_",imagery,"_Ene2022_4T",".npz"))


# See files
npz2$files

test_x_data <- npz2$f[["x_test"]]
test_y_data <- npz2$f[["y_test"]]

rm(npz2)

###---------------------------Model Definition---------------------------------------
# Aquí está el error, por alguna razón no corre el f1score bien. Checarlo a fondo

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


##-------------------Result Evaluation-----------------------------
# Para cargar el modelo usando la metric de f1score
model <- load_model_hdf5(model_location,
                         custom_objects = c("f1score" = f1score,
                                            "multilabel_dice_coefficient" = multilabel_dice_coefficient,
                                            "cce_dice_loss" = cce_dice_loss))

# Viene la siguiente parte

# result <- evaluate(
#    object = model,
#    generator = train_x_data,
#    steps = 5)

#De todas maneras el error no es tan bajo
# result$loss
# result$acc

# !!!!Evaluate gives an error!!!!
# score <- model %>% evaluate(
#   test_x_data, test_y_data,
#   verbose = 1
# )
# 
# score

#library(cowplot)
accuracy <- vector(length = dim(test_x_data)[1])
conf_matrix <- vector(length = dim(test_x_data)[1], mode = "list")

#Classes 
classes <- c("NoCambio","DeforBosque","DeforSecPlant")
classes_num <- c(seq(1, n_classes, 1))


#rast_list <- vector(length = dim(test_x_data)[1], mode = "list")
# PRedicted images
rast_pred_list <- vector(length = dim(test_x_data)[1], mode = "list")
# Ground truth images
rast_gt_list <- vector(length = dim(test_x_data)[1], mode = "list")
# Probabilities raster
prob_list <- vector(length = dim(test_x_data)[1], mode = "list")

# Aguas con esta parte porque si sale un error en los plots, se quedan vacías algunas listas y entonces se promedia con 0 la precisión.

# Accuracy
for(j in 1:dim(test_x_data)[1]){
  image_real<-array_reshape(test_x_data[j,,,,1:channels],c(1,time_obs,img_width_pred,img_height_pred,channels))
  
  result_pred <- predict(
    object = model,
    batch_size = batch_size,
    x = image_real,
    steps = 10,
    verbose = 1)
  
  image_class <- array_reshape(test_y_data[j,,,,1:n_classes],c(1,1,img_width_pred,img_height_pred,n_classes))
  
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
  
  resul <- lapply(1:n_classes, function(i){
    temp <- image_class[1,,,,i]==1
    temp[temp == TRUE] <- i
    temp[temp == FALSE] <- 0
    as.matrix(temp)
    
  })
  
  resul_class <- Reduce('+', resul)
  
  # To see spatial distributino of errors
  # rast_list[[j]] <- image_real
  rast_pred_list[[j]] <- resul2
  prob_list[[j]] <- maxval_bycell
  rast_gt_list[[j]] <- resul_class
  
  # Confusion matrix, completing missing levels so that the diagonal corresponds to 
  # correctly predicted pixels
  # Resul2 is predicted, resul_class is ground truth
  confusion_matrix<- table(factor(resul2,classes_num), factor(resul_class, classes_num))
  conf_matrix[[j]] <- confusion_matrix
  #Calculate overall precision
  accuracy[j] <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
}

# Total mean accuracy
# Watch out for zeros in this calculation
mean(accuracy)


# 128 x 128 version tiene accuracy como de 0.70
# 256 x 256 version tiene accuracy como de 0.76
# Get sum of all confusion matrices
total_conf_mat <- Reduce('+', conf_matrix)
colnames(total_conf_mat) <- classes
rownames(total_conf_mat) <- classes
total_conf_mat

# Get product and user accuracy
users_accuracy <- sapply(1:nrow(total_conf_mat), function(i){
  total_conf_mat[i,i] / sum( total_conf_mat[i,])
})
names(users_accuracy) <- colnames(total_conf_mat)
users_accuracy

producers_accuracy <- sapply(1:ncol(total_conf_mat), function(i){
  total_conf_mat[i,i] / sum( total_conf_mat[,i])
})
names(producers_accuracy) <- row.names(total_conf_mat)
producers_accuracy

overall_acc <- sum(diag(as.matrix(total_conf_mat))) / sum(as.matrix(total_conf_mat))

ncol(total_conf_mat)
total_conf_mat <- cbind(total_conf_mat,users_accuracy)
total_conf_mat <- rbind(total_conf_mat,c(producers_accuracy,overall_acc))

# write.csv(total_conf_mat, paste0("ConfusionMatrix_TestData_","U128model",imagery,"batch_",batch_size,"_Epochs",epochs,"_lr",learn_rate,"_adam_4T",Sys.Date(),".csv"))
# write.csv(mean(accuracy), paste0("MeanAccuracy_TestData_","U128model",imagery,"batch_",batch_size,"_Epochs",epochs,"_lr",learn_rate,"_adam_4T",Sys.Date(),".csv"))


# For Cochrans test, Get pred vs GT
predvsobs <- data.frame(pred = unlist(as.vector(rast_pred_list)),
                        obs = unlist(as.vector(rast_gt_list)))

write.csv(predvsobs,
          paste0("PredvsObs_", imagery, "_4T.csv"))

# Get pdf plots
# pdf(paste0("pred_vs_GT_UnetDefor",imagery,"_4T_CCEF1score.pdf"),
#     width = 8,
#     height = 6)
# par(mfrow=c(3,3))
#  for(j in 1:length(rast_pred_list)){
#    # plot(as.raster(rast_list[[j]][,1,,,1:3]))
#    # plot(as.raster(rast_list[[j]][,4,,,1:3]))
#    plot(raster(rast_pred_list[[j]]),
# 	main = "predicted class",
# 	breaks = c(0:3),
# 	col = c("forestgreen","firebrick2","darkorange"))
#    plot(raster(rast_gt_list[[j]]),
# 	main = "manual classification",
# 	breaks = c(0:3),
# 	col = c("forestgreen","firebrick2","darkorange"))
#    plot(raster(prob_list[[j]]),
#         main = "probability")
#  }
# dev.off()
# #--------------------------f1score calc----------------------------------------------------------------
# 
# # Prepare data
# # total_conf_matrix <- read.csv(paste0("ConfusionMatrix_TestData_","U128model",imagery,"Epochs",epochs,"_lr",learn_rate,"_adam:","_2021-10-12",".csv"))
# 
# # Random forests
# # total_conf_matrix <- read.csv(paste0("RF_conf_matrix_testset_10_2021-03-19.csv"))
# 
# # Remove prod user accuracy and class names
# # For OpticRadar and Optic
# total_conf_matrix <- total_conf_mat [-4,-c(4)]
# # For Radar
# # total_conf_mat <- total_conf_matrix [-11,-c(1,12)]
# 
# # ## ----------------------------Yardstick evaluation-------------------------------------------------------
# # Remake simple counts from confusion matrix
# # OpticRadar and Optic
# colnames(total_conf_matrix) <- 1:n_classes
# # Radar
# # colnames(total_conf_mat) <- 1:10
# 
# temp <- total_conf_matrix %>%
#   as_tibble() %>%
#   rownames_to_column() %>%
#   pivot_longer(cols = -rowname,
#                names_to = "Class_pred",
#                values_to = "count") %>%
#   rename("Class_true" = "rowname")
# 
# temp_gt_pred <- lapply(1:nrow(temp), function(i){
#   data.frame(gt = rep(temp$Class_true[i], temp$count[i]),
#              pred = rep(temp$Class_pred[i], temp$count[i]))
# })
# temp_gt_pred <- bind_rows(temp_gt_pred)
# 
# estimates_keras_tbl <- tibble(
#   truth      = factor(temp_gt_pred[,1], levels = 1:n_classes),
#   estimate   = factor(temp_gt_pred[,2], levels = 1:n_classes),
#   # class_prob = as.numeric(unlist(temp_probs))
# )
# 
# # Confusion Table
# estimates_keras_tbl %>%
#   conf_mat(truth, estimate)
# 
# # Accuracy
# accuracy <- estimates_keras_tbl %>%
#   metrics(truth, estimate)
# 
# ## F1 score, Esto da F1 score de 0.62 (macro)
# # micro = 0.76; macro_weighted = 0.77
# f1score <- estimates_keras_tbl %>%
#   f_meas(truth,
#          estimate,
#          estimator = "macro",
#          beta = 1)
# 
# accuracy %>%
#   add_row(f1score) %>%
#   write.csv(paste0("AccF1score_","U128model",imagery,"Epochs",epochs,"_lr",learn_rate,"_adam_",Sys.Date(),".csv"))
