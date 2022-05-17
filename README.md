# Prediction-of-heart-attack-risk

The python scripts are tested and run on Spyder (Python 3.8).

This file aimed to predict the risk of herat attack using classification model.
Thefile for training purposes is named as heart.attack.py
The file for deployment purposes is named as heart_attack_deploy.py

Data folder stored the raw data for this analysis.
Original sources of the data can be found in the link below:
https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

Best model identified for this analysis is the random forest, with accuracy of 75%.
Classification report and the confusion matrix of the relevant model can be found in the folder named figures.
A screenshot of the interface of application built by streamlit also can be found in this folder.

In order to run the deployment files, you will need to run with two pickle files included
Best model is saved as best_model_forest.pkl and the Min Max Scaler involved also uploaded as mms.pkl

