# Ahorita esta con la version bis, namas pa checar si da igual o casi igua, si no ya pa no volver a correr todos los modelos

# library(unet)
# libraries we're going to need later
library(keras)
library(tensorflow)
# library(tfdatasets)
# library(rsample)
# library(tidyverse)
library(reticulate)
library(tfruns)

source("Unet3d_bis.R")
source("multilabel_dice_coefficient.R")
# source("multilabel_dice_loss.R")

np <- import("numpy")

##---------------------Constant def---------------------------------------
# imagery
input_type <- "MSSAR"

#Training data
# batch_size <- 64
channels <- ifelse(input_type == "MSSAR", 6, 
                   ifelse(input_type == "MS", 4, 2))
time_obs <- 4
activation_func_out <- "softmax"

# Hyperparameter flags 

FLAGS <- flags(
  flag_numeric("batch_size", 0),
  flag_numeric("learn_rate", 0),
  flag_numeric("epochs", 0),
  flag_numeric("dropout", 0),
  flag_numeric("filters_firstlayer",0),
  flag_numeric("num_layers", 0)
)

#Label data
n_classes <- 3 # Estoy metiendo 0:2, que es 1 a 3

#Dimensiones de las imagenes en cols y rows, sacarlas de QGIS
img_width <- 128
img_height <- 128

#Este es el tamaño que sale despuas de hacer las convoluciones sin utlilizar padding; checar summary(model) pa ver las dimensiones finales
img_width_pred <- 128
img_height_pred <- 128

#---------------------------Check image read-------------------------

# Numpy zip file option
# Load npz
npz2 <- np$load(paste0("Lacandona_Defor_FebApr_20192020_",input_type,"_Ene2022_4T",".npz"))
# See files
#npz2$files

train_x_data <- npz2$f[["x_train"]]
train_y_data <- npz2$f[["y_train"]]

# Aqui hay que hacer algo con los datos para que estan en un rango entre 0 y 1, quizas como un hist stretch
test_x_data <- npz2$f[["x_test"]]
test_y_data <- npz2$f[["y_test"]]

#----------------------------Sample weights----------------------------------------
# Remember doing this will drop two extra dimensions (temporal and spectral)
# samplew <- 1+(train_y_data[,,,,1, drop = F]*-1)
# To ignore "no change" class leave it as 0

# No change will be 0.0086, while change = 1
# Weight determined from y_test data
# samplew[samplew==0] <- 0.0086
# samplew[samplew==1] <- 5

#samplew2 <- array(0, dim = c(dim(samplew)[1],
#                             dim(samplew)[2]+dim(samplew)[2],
#                             dim(samplew)[3],
#                             dim(samplew)[4],
#                             dim(samplew)[5]))
#samplew2[,1,,,] <- samplew
#samplew2[,2,,,] <- samplew

#rm(samplew)

###---------------------------Model Definition---------------------------------------
# Recordar que el dropout es para evitar overfit, quizas ahorita no usarlo
model <- unet3d_bis(input_shape = c(time_obs,img_width, img_height, channels),
                num_classes = n_classes,
                dropout = FLAGS$dropout,
                #Aqui con 64 filtros como que ya no cabe en la memoria y corre super lento
                filters = FLAGS$filters_firstlayer,
                # Con menos layers como que no agarra nada
                # 4 layers hace la max convolucion quede una imagen 4 x 4 pix; Hacer mas layers quizas no tiene mucho sentido. Asi es el modelo original de U-net con aplicaciones remote sensing
                num_layers = FLAGS$num_layers,
                output_activation = activation_func_out)

summary(model)

# Aqui esta el error, por alguna razon no corre el f1score bien. Checarlo a fondo

# f1score <- custom_metric("f1score", function(y_true, y_pred, smooth = 1) {
#   y_true_f <- k_flatten(y_true)
#   y_pred_f <- k_flatten(y_pred)
#   intersection <- k_sum(y_true_f * y_pred_f)
#   (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
# })

# dice_coef <- function(y_true, y_pred, smooth = 1.0) {
#   y_true_f <- k_flatten(y_true)
#   y_pred_f <- k_flatten(y_pred)
#   intersection <- k_sum(y_true_f * y_pred_f)
#   result <- (2 * intersection + smooth) /
#     (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
#   return(result)
# }

# sum of binary_crossentropy and soft-Dice
# Based on imageseg package and Isaienkov et al., 2020
cce_dice_loss <- custom_metric("cce_dice_loss", function(y_true, y_pred) {
  result <- 0.2 * loss_categorical_crossentropy(y_true, y_pred) +
    (0.8 * (1 - multilabel_dice_coefficient(y_true, y_pred)))
  return(result)
})

# f1score <- custom_metric("f1score", function(y_true, y_pred, smooth = 1) {
#   y_pred <- k_round(y_pred)
#   
#   precision <- k_sum(y_pred*y_true)/(k_sum(y_pred)+k_epsilon())
#   recall    <- k_sum(y_pred*y_true)/(k_sum(y_true)+k_epsilon())
#   
#   f1 <- (2*precision*recall)/(precision+recall+k_epsilon())
#   f1 <- tf$where
#   k_mean(f1)
# })

model %>% compile(
  optimizer = optimizer_adam(learning_rate = FLAGS$learn_rate),
  # Se supone que para problemas de multiclass la funcion loss sugerida es esta, en lugar de la binary que es para clasificacion binaria; ver sparse categorical (para cuando los labels no estan codificados como one-hot) o solo categorical (cuando estan categorizados como one-hot)
  loss = cce_dice_loss, #"categorical_crossentropy", 
  # loss_weights = list(samplew),
  # sample_weight_mode = "temporal",
  # weighted_metrics = list(f1score, "categorical_accuracy"),
  metrics = list(multilabel_dice_coefficient,
                 "categorical_accuracy")
)

#early_stopping <- callback_early_stopping(monitor = 'cce_dice_val_loss', 
#                                          min_delta = 0.01,
#                                          mode = "min",
#                                          patience = 10,
#                                          restore_best_weights = T)

##-------------------------------------Run Model-------------------------------

#Por alguna razon a veces el modelo se estanca, sera por la seed??
history <- model %>% fit(
  # Model and inputs
  # X as independent vars
  x = train_x_data,
  y = train_y_data,
  # Remember we can use validation split also, but right now, prefer not, to force to leave out certain images from the trainaing (as we are augumenting images)
  validation_data = list(test_x_data, test_y_data),
  # Epochs
  epochs = FLAGS$epochs,
  # steps_per_epoch = steps_per_epoch,
  # Batch_size
  batch_size = FLAGS$batch_size,
  shuffle = T,
  # sample_weight = list(samplew),
  # 3d conv do not accept class weights
  # class_weight = list("0"=1,
  #                    "1"=30,
  #                    "2"=30),
  # Verbose or not
  verbose = 1,
  # callbacks = list(early_stopping)
)

# Before predicting class, save model close and re open R to free disk space
save_model_hdf5(model, paste0("U128model_f1loss",input_type,
                              "filters", FLAGS$filters_firstlayer,
                              "Epochs",FLAGS$epochs,
                              "layers", FLAGS$num_layers,
                              "dropout", FLAGS$dropout,
                              "_lr",FLAGS$learn_rate,
                              "_adam",
                              "_2022-02-11",".h5"))

# plot(history)

# score <- model %>% evaluate(
#  test_x_data, test_y_data,
#  verbose = 0
#)

#write.csv(score,paste0("U128model",input_type,
#                              "filters", FLAGS$filters_firstlayer,
#                              "Epochs",FLAGS$epochs,
#                              "layers", FLAGS$num_layers,
#                              "dropout", FLAGS$dropout,
#                              "_lr",FLAGS$learn_rate,
#                              "_adam",
#                              "_2021-12-01",".csv"))

#cat('Test loss:', score$loss, '\n')
#cat('Test accuracy:', score$acc, '\n')
#cat('Test f1score:', score$f1score, '\n')
#cat('Train loss:', score$loss, '\n')
#cat('Train accuracy:', score$acc, '\n')
#cat('Train f1score:', score$f1score, '\n')

rm(list=ls())
