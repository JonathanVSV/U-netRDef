conv3d_block <- function(inputs, use_batch_norm = TRUE, dropout = 0.3,
                         # Check kernel size dimensions, usually, first, depth, height, width
                         filters = 16, kernel_size = c(3, 3, 3), activation = "relu",
                         kernel_initializer = "he_normal", padding = "same") {
  
  x <- keras::layer_conv_3d(
    inputs,
    filters = filters,
    kernel_size = kernel_size,
    activation = activation,
    kernel_initializer = kernel_initializer,
    padding = padding
  )
  
  if (use_batch_norm) {
    x <- keras::layer_batch_normalization(x)
  }
  
  if (dropout > 0) {
    x <- keras::layer_dropout(x, rate = dropout)
  }
  
  x <- keras::layer_conv_3d(
    x,
    filters = filters,
    kernel_size = kernel_size,
    activation = activation,
    kernel_initializer = kernel_initializer,
    padding = padding
  )
  
  if (use_batch_norm) {
    x <- keras::layer_batch_normalization(x)
  }
  
  x
}

#' U-Net: Convolutional Networks for Biomedical Image Segmentation
#'
#' @param input_shape Dimensionality of the input (integer) not including the
#'   samples axis. Must be length 3 numeric vector.
#' @param num_classes Number of classes.
#' @param dropout Dropout rate applied between downsampling and upsampling phases.
#' @param filters Number of filters of the first convolution.
#' @param num_layers Number of downsizing blocks in the encoder.
#' @param  output_activation Activation in the output layer.
#'
#' @export
unet3d_bis <- function(input_shape, num_classes = 1, dropout = 0.5, filters = 64,
                   num_layers = 4, output_activation = "sigmoid") {
  
  
  input <- keras::layer_input(shape = input_shape)
  
  x <- input
  down_layers <- list()
  
  for (i in seq_len(num_layers)) {
    
    x <- conv3d_block(
      inputs = x,
      filters = filters,
      use_batch_norm = FALSE,
      dropout = 0,
      padding = "same"
    )
    
    down_layers[[i]] <- x
    
    if(i <= 2){
      x <- keras::layer_max_pooling_3d(x, pool_size = c(2,2,2), strides = c(2,2,2))
    }else{
      x <- keras::layer_max_pooling_3d(x, pool_size = c(1,2,2), strides = c(1,2,2))
    }
    filters <- filters * 2
    
  }
  
  if (dropout > 0) {
    x <- keras::layer_dropout(x, rate = dropout)
  }
  
  x <- conv3d_block(
    inputs = x,
    filters = filters,
    use_batch_norm = FALSE,
    dropout = 0.0,
    padding = 'same'
  )
  
  aux <- 1
  for (conv in rev(down_layers)) {
    
    filters <- filters / 2L
    if(aux <= (num_layers - 2)){
      x <- keras::layer_conv_3d_transpose(
        x,
        filters = filters,
        kernel_size = c(1,2,2),
        padding = "same",
        strides = c(1,2,2)
      )
    }else{
      x <- keras::layer_conv_3d_transpose(
        x,
        filters = filters,
        kernel_size = c(2,2,2),
        padding = "same",
        strides = c(2,2,2)
      )
    }
    
    x <- keras::layer_concatenate(list(conv, x))
    x <- conv3d_block(
      inputs = x,
      filters = filters,
      use_batch_norm = FALSE,
      dropout = 0.0,
      padding = 'same'
    )
    
    aux <- aux+1
  }
  
  output <- keras::layer_conv_3d(
    x,
    filters = num_classes,
    kernel_size = c(4,1,1),
    activation = output_activation
  )
  
  model <- keras::keras_model(input, output)
  
  model
}

