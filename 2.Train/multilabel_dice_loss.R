multilabel_dice_loss <- custom_metric("multilabel_dice_loss", function( y_true, y_pred )
{
  dimensionality <- 3L
  smoothingFactor <- 0.1
  
  y_dims <- unlist( k_int_shape( y_pred ) )
  
  numberOfLabels <- as.integer( y_dims[length( y_dims )] )
  
  y_true_permuted <- k_permute_dimensions(
    y_true, pattern = c( 5L, 1L, 2L, 3L, 4L) )
  y_pred_permuted <- k_permute_dimensions(
    y_pred, pattern = c( 5L, 1L, 2L, 3L, 4L) )
  
  y_true_label <- k_gather( y_true_permuted, indices = c( 1 ) )
  y_pred_label <- k_gather( y_pred_permuted, indices = c( 1 ) )
  y_true_label_f <- k_flatten( y_true_label )
  y_pred_label_f <- k_flatten( y_pred_label )
  intersection <- y_true_label_f * y_pred_label_f
  union <- y_true_label_f + y_pred_label_f #- intersection
  
  numerator <-  k_sum( intersection )
  denominator <-  k_sum( union )
  
  # unionOverlap <- numerator / denominator
  
  f1_score1 <- ( 2 * numerator + smoothingFactor ) /
    ( denominator + smoothingFactor )
  
  y_true_label <- k_gather( y_true_permuted, indices = c( 2 ) )
  y_pred_label <- k_gather( y_pred_permuted, indices = c( 2 ) )
  y_true_label_f <- k_flatten( y_true_label )
  y_pred_label_f <- k_flatten( y_pred_label )
  intersection <- y_true_label_f * y_pred_label_f
  union <- y_true_label_f + y_pred_label_f #- intersection
  
  numerator <-  k_sum( intersection )
  denominator <-  k_sum( union )
  
  # unionOverlap <- numerator / denominator
  
  f1_score2 <- ( 2 * numerator + smoothingFactor ) /
    ( denominator + smoothingFactor )
  
  y_true_label <- k_gather( y_true_permuted, indices = c( 3 ) )
  y_pred_label <- k_gather( y_pred_permuted, indices = c( 3 ) )
  y_true_label_f <- k_flatten( y_true_label )
  y_pred_label_f <- k_flatten( y_pred_label )
  intersection <- y_true_label_f * y_pred_label_f
  union <- y_true_label_f + y_pred_label_f #- intersection
  
  numerator <-  k_sum( intersection )
  denominator <-  k_sum( union )
  
  # unionOverlap <- numerator / denominator
  
  f1_score3 <- ( 2 * numerator + smoothingFactor ) /
    ( denominator + smoothingFactor )
  
  f1_score <- (f1_score1 + f1_score2 + f1_score3) / 3
  1-f1_score
})