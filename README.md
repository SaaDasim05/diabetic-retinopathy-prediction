# diabetic-retinopathy-prediction
Predicts the presence and severity of Diabetic Retinopathy from retinal images using a two-stage deep learning pipeline (binary and multi-class classification).

How It Works

1. Training Phase

The data set includes retinal fundus images classified into five different classes:



Healthy (No Diabetic Retinopathy)



Mild DR



Moderate DR



Severe DR



Proliferative DR



The model training is done in two phases:



Stage 1: A binary classifier model is trained with EfficientNetB3 (or any other similar convolutional neural network) to classify the images between DR and Non-DR.



Stage 2: A multi-class classification model is trained to classify the stage of Diabetic Retinopathy (Mild to Proliferative) for images classified as DR-positive.



2. Prediction Phase

During inference:



The trained models (.h5 format) are loaded.



The new input image is fed into the binary classification model in order to find out if the Diabetic Retinopathy is present.



In case DR is identified, the image is passed on to the stage classification model, which estimates the severity level of the condition.



The output consists of the estimated DR stage (if present) as well as medical advice or recommendations pertaining to that stage.
