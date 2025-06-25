# U-net3DR

U-net 3D implemented in R for deforestation detection.

The repository contains several scripts used to fit a U-net 3D model using keras (tensorflow backend) in R. The complete method and results can be consulted in: Solórzano, J. V., Mas, J. F., Gallardo-Cruz, J. A., Gao, Y., & Fernández-Montes de Oca, A. (2023). Deforestation detection using a spatio-temporal deep learning approach with synthetic aperture radar and multispectral images. ISPRS Journal of Photogrammetry and Remote Sensing 199, 87–101. https://doi.org/10.1016/j.isprsjprs.2023.03.017.

The structure of the repository is the following:

1.ImgPreProcess makes some pre preprocessing and augmentation of the data. 1. TrainTestData transforms each pair of x (optic and radar images) and y (manually labeled polygons) into arrays that contain the training and test set of augumented images. In this procedure for each 256 x 256 px image, 27 128 x 128 px images are generated. From these latter, 9 images correspond to cropped images (using an offset of 32 pixels), 9 to vertically mirrored images and 9 to horizontally mirrored images. Finally the images are stored in arrays and saved as an "npz" file. Additionally, this script divides the data into train, verif, test sets. 2. GetTestVerifIds is a script that generates a file which contains data about which areas were selected for the train, verif and test sets. This file will be later be used in the final verification step. 3. CompleteImage4Prediction makes the necessary preprocessing of the whole images so they can be feeded into the fitted U-Net 3D and make a prediction over the complete study area.

2.Train constructs the U-net's desired architecture, define the hyperparameters that are going to be explored (grid approach), fit the model and save it. 1.aUnet_singleTrain contains the script to build a U-Net and train it; while 1.bTfruns_hyParTune can be used to explore different hyperparameters values, using a grid approach. This last script calls the first one. Unet3d_bis and multilabel_dice_coefficients contain scripts that are called inside 1.aUnet_singleTrain to build the U-net 3D (Unet3d_bis) and use some custom metrics (multilabel_dice_coefficients).

The 3.Verif folder contains three scripts. The first one (Predict_Fullimage), makes the class predictions and get the probabilities of corresponding to each class for the complete study area (based on the files exported by 3. CompleteImage4Prediction). This script makes the predictions using two different grids, where the center of one grid overlaps with the edges of the other. Then these predictions and probabilities are used to set the class as the one with the highest probability in either of the two grids used to make the predictions (Final_Unet_ensemble). Finally, the 3.Verification_ConfMat script is used to evaluate the observed values vs the predictions made by the U-net 3D on the verif and test sets, without any augmentation.

The 4. Data folder contains the deforestation polygons, the grids used to divide the image into tiles to make the train, verif and test sets, as well as the model used to make the final classification.

The 5. Results folder contains the GeoTiff of the final classification, as well as its corresponding probabilty GeoTiff.

Visualization of the U-Net model, made with [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

![U-net 3D deforestation detection](/5.Results/unet3d_4T.jpg?raw=true "U-net 3D diagram")

Preview of the final deforestation classification obtained with U-net 3D MS+SAR.

![U-net 3D classification](/5.Results/preview.png?raw=true "Deforestation detection using U-Net 3D")