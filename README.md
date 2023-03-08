# U-net3DR

U-net 3D implemented in R for deforestation detection.

The repository contains several scripts used to fit a U-net 3D model using keras (tensorflow backend) in R. This is an extension of the other repository: U-netR, where a U-net 2D was used to perform LULC a classification.

The structure of the repository is the following:

1.ImgPreProcess makes some pre preprocessing and augmentation of the data. 1. TrainTestData transforms each pair of x (optic and radar images) and y (manually labeled images (256 x 256 pixels)) into arrays that contain the training and test set of augumented images. In this procedure for each 256 x 256 px image, 27 128 x 128 px images are generated. From these latter, 9 images correspond to cropped images (using an offset of 32 pixels), 9 to vertically mirrored images and 9 to horizontally mirrored images. Finally the images are stored in arrays and saved as an "npz" file. Additionally, this script divides the data into train, verif, test sets. 2. GetTestVerifIds is a script that generated a file which contains which areas were selected for the train, verif and test sets. This file will be later be used in the final verification step. 3. CompleteImage4Prediction makes the necessary preprocessing of the whole images so they can be feeded into the fitted U-Net 3D and make a prediction over the complete study area.

2.Train constructs the U-net's desired architecture, define the hyperparameters that are going to be explored, fit the model and save it. 1.aUnet_singleTrain contains the script to build a U-Net and train it; while 1.bTfruns_hyParTune can be used to explore different hyperparameters values, using a grid approach. This last script calls the first one. Unet3d_bis and multilabel_dice_coefficients contain scripts that are called inside 1.aUnet_singleTrain to build the U-net 3D (Unet3d_bis) and use some custom metrics (multilabel_dice_coefficients)

The 3.Verif folder contains three scripts. The first one (Predict_Fullimage), makes the class predictions and get the probabilities of corresponding to each class for the complete study area (based on the files exported by 3. CompleteImage4Prediction). This script makes the predictions using two different grids, where the center of one grid overlaps with the edges of the other. Then these predictions and probabilities are used to set the class as the one with the highest probability in either of the two grids used to make the predictions (Final_Unet_ensemble). Finally, the 3.Verification_ConfMat script is used to evaluate the observed values vs the predictions made by the U-net 3D on the verif and test sets, without any augmentation.

![U-net 3D classification](/5.Results/preview.png?raw=true "Deforestation detection using U-Net 3D")