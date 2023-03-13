library(keras)
library(tensorflow)
library(reticulate)
library(tfruns)

# Load Unet3d configuration
source("Unet3d_bis.R")
# Load multilabel dice coefficient
# Used as synonym for avgF1-score
source("multilabel_dice_coefficient.R")

np <- import("numpy")

##---------------------Constant def---------------------------------------
# imagery
input_type <- "MSSAR"

# Number of bands
channels <- ifelse(input_type == "MSSAR", 6, 
                   ifelse(input_type == "MS", 4, 2))
# Number of observations in the time dimension
time_obs <- 4
# Activation function for output
activation_func_out <- "softmax"

# Hyperparameter flags; set as 0, so all values come from the tfruns
FLAGS <- flags(
  flag_numeric("batch_size", 0),
  flag_numeric("learn_rate", 0),
  flag_numeric("epochs", 0),
  flag_numeric("dropout", 0),
  flag_numeric("filters_firstlayer",0),
  flag_numeric("num_layers", 0)
)

#Label data
n_classes <- 3

# height and width of tiles
img_width <- 128
img_height <- 128

# height and width of output tiles
img_width_pred <- 128
img_height_pred <- 128

#---------------------------Check image read-------------------------

# Numpy zip file option
# Load npz
npz2 <- np$load(paste0("Lacandona_Defor_FebApr_20192020_",input_type,"_Ene2022_4T_revnov",".npz"))

train_x_data <- npz2$f[["x_train"]]
train_y_data <- npz2$f[["y_train"]]

# Data over which the algorithm will use as verification
# It's the same to use test as verif or viceversa. The only
# thing is that one is used in the training and the other just for testing
test_x_data <- npz2$f[["x_test"]]
test_y_data <- npz2$f[["y_test"]]

###---------------------------Model Definition---------------------------------------
# Set model
model <- unet3d_bis(input_shape = c(time_obs,img_width, img_height, channels),
                num_classes = n_classes,
                dropout = FLAGS$dropout,
                # Take values from the FLAGS
                filters = FLAGS$filters_firstlayer,
                num_layers = FLAGS$num_layers,
                output_activation = activation_func_out)

summary(model)

# sum of binary_crossentropy and soft-Dice
# Based on imageseg package and Isaienkov et al., 2020
cce_dice_loss <- custom_metric("cce_dice_loss", function(y_true, y_pred) {
  result <- 0.2 * loss_categorical_crossentropy(y_true, y_pred) +
    (0.8 * (1 - multilabel_dice_coefficient(y_true, y_pred)))
  return(result)
})

model %>% compile(
  optimizer = optimizer_adam(learning_rate = FLAGS$learn_rate),
  loss = cce_dice_loss, #"categorical_crossentropy", 
  metrics = list(multilabel_dice_coefficient,
                 "categorical_accuracy")
)

# If early stopping is implemented
#early_stopping <- callback_early_stopping(monitor = 'cce_dice_val_loss', 
#                                          min_delta = 0.01,
#                                          mode = "min",
#                                          patience = 10,
#                                          restore_best_weights = T)

##-------------------------------------Run Model-------------------------------

history <- model %>% fit(
  # Model and inputs
  # X as independent vars
  x = train_x_data,
  y = train_y_data,
  # Remember we can use validation split also, but right now, prefer not, to force to leave out certain images from the training (as we are augumenting images)
  validation_data = list(test_x_data, test_y_data),
  # Epochs
  epochs = FLAGS$epochs,
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
                              "_2022-11-26",".h5"))

rm(list=ls())
