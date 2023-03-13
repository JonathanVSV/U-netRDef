
library(dplyr)
library(tidyr)
library(reticulate)
library(tfruns)
library(raster)
library(yardstick)
library(sf)
library(fasterize)

imagery <- "MSSAR"
img_width <- 128
img_height <- 128

# Import manually delineated deforestation polygons
defor_gt <- st_read("defor_patch_valid_2022_revnov.shp")
# Grids for train test verif
all_grid  <- st_read("Grids_defor_datasets.shp") 
# Predicted final image
im_pred <- stack(paste0("Final_Class_ensemble_",imagery,"_revnov.tif"))

# Get MS image
multi <- stack("S1y2_9immedian_2A_6B_2019-02-01_2019-04-30_median_10mBandsMaxCCL100.tif")

# Fix train grid to match exactly the same as the one used in Unet 256 x 256 pixels
# Crop 128 px areas for input images
all_tiles <- lapply(1:nrow(all_grid), function(i){
  poly <- all_grid %>% slice(i)
  # Subset by poly
  temp <- crop(multi,poly)
  # Force to crop image to 256 tiles
  temp <- crop(temp,extent(temp, 2, (2+img_width-1), 2, (2+img_height-1)))
  # Return new extent as an sf object
  st_as_sf(st_as_sfc(st_bbox(temp), crs = 4326))
})

# Bind all vectors in a single file
all_grid <- dplyr::bind_rows(all_tiles) |>
  mutate(true_id = all_grid$true_id,
         type = all_grid$type) |>
  dplyr::select(true_id, everything()) |>
  st_set_crs(4326)

verif_grid <- all_grid |>
  filter(type == "verif")
test_grid <- all_grid |>
  filter(type == "test")

# Intersect ground truth deforestation and verif and test grids
all_verif <- st_intersection(defor_gt,
                             verif_grid) |>
  st_cast("MULTIPOLYGON")
all_test <- st_intersection(defor_gt,
                            test_grid) |>
  st_cast("MULTIPOLYGON")

# Convert to raster
all_verif_rast <- fasterize(sf = all_verif,
                            raster = multi[[1]],
                            background = 0,
                            field = "id2022")

# Remove all background = 0 that is not inside train
all_verif_rast <- raster::mask(all_verif_rast, verif_grid)

all_test_rast <- fasterize(sf = all_test,
                           raster = multi[[1]],
                           background = 0,
                           field = "id2022")

# Remove all background = 0 that is not inside train
all_test_rast <- raster::mask(all_test_rast, test_grid)

# Convert each pixel to point
pts_verif <- rasterToPoints(all_verif_rast,
                            spatial = F,
                            progress = "text")

pts_test <- rasterToPoints(all_test_rast,
                           spatial = F,
                           progress = "text")

# Convert points to sf
pts_verif <- st_as_sf(as.data.frame(pts_verif),
                      coords = c("x", "y"),
                      crs = 4326)

pts_test <- st_as_sf(as.data.frame(pts_test),
                     coords = c("x", "y"),
                     crs = 4326)

# Extract values of pixels
point_pred_verif <- raster::extract(im_pred,
                                  pts_verif,
                                  df = T)

point_pred_test <- raster::extract(im_pred,
                                 pts_test,
                                 df = T)

point_obs_verif <- raster::extract(all_verif_rast,
                                  pts_verif,
                                  df = T)

point_obs_test <- raster::extract(all_test_rast,
                                 pts_test,
                                 df = T)

# Transform numeric classes into factors and match 
# pred (1 based) and obs (0 based)
defor_test_results <- tibble(obs = point_obs_test$layer,
                        pred = point_pred_test$Final_Class_ensemble_MSSAR_revnov) |>
  # Make comparable scales in obs and pred values. Pred values are 1 based, while obs 0.
  mutate(across(pred, function(x) case_when(x == 1 ~ "NoDeforestation",
                                            x == 2 ~ "Old-growthForestLoss",
                                            x == 3 ~ "SecondPlantLoss"))) |>
  mutate(across(obs, function(x) case_when(x == 0 ~ "NoDeforestation",
                                           x == 1 ~ "Old-growthForestLoss",
                                           x == 2 ~ "SecondPlantLoss"))) |>
  mutate(across(c(obs, pred), as.factor))

defor_verif_results <- tibble(obs = point_obs_verif$layer,
                              pred = point_pred_verif$Final_Class_ensemble_MSSAR_revnov) |>
  # Make comparable scales in obs and pred values. Pred values are 1 based, while obs 0.
  mutate(across(pred, function(x) case_when(x == 1 ~ "NoDeforestation",
                                            x == 2 ~ "Old-growthForestLoss",
                                            x == 3 ~ "SecondPlantLoss"))) |>
  mutate(across(obs, function(x) case_when(x == 0 ~ "NoDeforestation",
                                           x == 1 ~ "Old-growthForestLoss",
                                           x == 2 ~ "SecondPlantLoss"))) |>
  mutate(across(c(obs, pred), as.factor))

# Calculate confusion matrix and evaluation metrics
conf_matrix_verif <- defor_verif_results  |>
  conf_mat(truth = obs,
           estimate = pred) 

conf_matrix_verif <- as.data.frame(conf_matrix_verif$table) |>
  pivot_wider(id_cols = Prediction,
              names_from = Truth,
              values_from = Freq)

conf_matrix_test<- defor_test_results  |>
  conf_mat(truth = obs,
           estimate = pred) 

conf_matrix_test <- as.data.frame(conf_matrix_test$table) |>
  pivot_wider(id_cols = Prediction,
              names_from = Truth,
              values_from = Freq)

defor_metrics <- metric_set(accuracy,
                            # roc_auc,
                            # precision,
                            # recall,
                            f_meas)

verif_estimates <- defor_verif_results |>
  defor_metrics(truth = obs,
                estimate = pred)

test_estimates <- defor_test_results |>
  defor_metrics(truth = obs,
                estimate = pred)

# Write to csv
write.csv(defor_verif_results,
          paste0("PredvsObs_verif_", imagery, "_4T_revnov.csv"))
write.csv(verif_estimates,
          paste0("Evaluation_verif_", imagery, "_4T_revnov.csv"))
write.csv(conf_matrix_verif, paste0("ConfusionMatrix_verifData_","U128model",imagery,"_revnov.csv"))

write.csv(defor_test_results,
          paste0("PredvsObs_test_", imagery, "_4T_revnov.csv"))
write.csv(test_estimates,
          paste0("Evaluation_test_", imagery, "_4T_revnov.csv"))
write.csv(conf_matrix_test, paste0("ConfusionMatrix_testData_","U128model",imagery,"_revnov.csv"))

