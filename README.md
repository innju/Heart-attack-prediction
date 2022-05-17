# Heart attack prediction
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)

<br>This analysis aimed to develop an app to predict the chance of a person having heart attack.
<br>The python scripts are tested and run on Spyder (Python 3.8).

## DATA SOURCE:
<br>Original sources of the data can be found in the link below:
<br>https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
<br>Thanks to the data contributor: Mr.Rashik Rahman

## FOLDERS UPLOADED:
<br>Main folder named heart_attack can be found at the repository page.
<br>Click to see another two folders inside it, which are data and figures.
<br>Data folder stored the raw data for this analysis.
<br>Figures folder stored the figure of classification report,confusion matrix of the best model and the screenshot of the interface of the application built.
<br>heart.attack.py refer to the file for training and heart_attack_deploy.py refer to the file for deployment.
<br>Deployment file is required to run with the two pickle files saved. 
<br>Best model is saved as best_model_forest.pkl and the Min Max Scaler involved also uploaded as mms.pkl

## ABOUT THE MODEL:
<br>Extra tree classifier served the purpose of feature selection.
<br>It select the six features out of thirteen features given, which are 'cp','thalachh','exng','oldpeak','caa', and 'thall', 
<br>with the value of feature importances calculated.
<br>Then, machine pipelines is used to compare between decision tree and random forest model.
<br>Best model identified for this analysis is the random forest model, with accuracy of 75%.
<br>If you have anaconda installed to your devices, you can view the application by follow the steps below:
<br>Open Anaconda prompt > conda activate (name of ur environment) > cd (main folder path) > streamlit run (file name for deployment)
<br>Outcome of the app is either low risk or risky.

Thanks for reading.
