library(keras)
library(tfruns)

imagery <- "MSSAR"
date <- Sys.Date()

# Find the best evaluation accuracy
# Set the hyperparameters that are going to be tested to search for the optimum configuration
runs <- tuning_run(paste0("Unet_singleTrain_4cluster.R"),
                   runs_dir = paste0("DeforhypTune_16batches_",imagery,"_4T_revnov",date),
                   sample = 1,
                   flags = list(batch_size = c(16),
                                learn_rate = c(1e-4),
                                epochs = c(9),
                                dropout = c(0.1, 0.3, 0.5), 
                                filters_firstlayer = c(64),
                                num_layers = c(4,5)),
                   echo = F,
                   confirm = F) 



# Save all the runs info in a csv
write.csv(runs, 
          paste0("Runs_DeforhypTune_16batch_4-5layer_",imagery,"_4T_",date,".csv"),
          row.names = F)
