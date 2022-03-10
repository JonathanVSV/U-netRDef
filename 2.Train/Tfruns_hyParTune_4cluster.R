library(keras)
library(tfruns)

# memory.limit(size=56000)
# memory.limit()

# 64 batch size no aguanta la memoria con el modelo más complejo
imagery <- "MS"
date <- Sys.Date()

# find the best evaluation accuracy
runs <- tuning_run(paste0("Unet_singleTrain_4cluster.R"),
                   runs_dir = paste0("DeforhypTune_16batches_",imagery,"_4T",date),
                   sample = 1,
                   flags = list(batch_size = c(16),
                                learn_rate = c(1e-4),
                                epochs = c(9),
                                dropout = c(0.1, 0.3, 0.5), 
                                filters_firstlayer = c(64),
                                num_layers = c(4,5)),
                   echo = F,
                   confirm = F) 



#runs

write.csv(runs, 
          paste0("Runs_DeforhypTune_16batch_5layer_",imagery,"_4T_",date,".csv"),
          row.names = F)
