# Heart attack prediction
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)

<br>This analysis aimed to develop an app to predict the chance of a person having heart attack.
<br>The python scripts are tested and run on Spyder (Python 3.8).

### DATA SOURCE:
Original sources of the data can be found in the link below:
<br>[Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
<br>Thanks to the data contributor: Mr.Rashik Rahman

### FOLDERS UPLOADED:
Main folder named heart_attack can be found at the repository page.
<br>The content of folder is listed as below:
<br>**heart attack folder**: Main folder consist of data folder, figure folder, python and pickle files
<br>**Data folder**: raw data
<br>**Figures folder**: classification report,confusion matrix of the best model and the interface of the application
<br>**heart.attack.py**: training file
<br>**heart_attack_deploy.py**: deployment file.
<br>**best_model_forest.pkl**: best model saved to the file
<br>**mms.pkl**: MinMaxScaler saved to the file

<br>Deployment file is required to run with the two pickle files saved. 

### ABOUT THE MODEL:
**Extra tree classifier** served the purpose of **feature selection**.
<br>It select the six features out of thirteen features given.
<br>Features selected are **'cp','thalachh','exng','oldpeak','caa', and 'thall'**. 
<br>It select features based on feature importances calculated.
<br>Then, machine pipelines is used to compare between decision tree and random forest model.
<br>Best model identified for this analysis is the **random forest model**, with **accuracy of 75%**.

![Image](https://github.com/innju/Heart-attack-prediction/blob/main/heart_attack/figures/classification_report_random_forest.png)

The precision calculated for class 0 and class 1 is 0.80 and 0.75 respectively. The class 0 refer to category with low risk to get heart attack while class 1 refer to category with high risk to get heart attack. High precision indicates high number of data predicted positive and it's true. Recall also should be high to indicates the number of data predicted correctly out from all the positive classes.The recall calculated for class 0 and class 1 is 0.69 and 0.84 respectively. 

![Image](https://github.com/innju/Heart-attack-prediction/blob/main/heart_attack/figures/confusion_matrix_random_forest.png)


### VIEW THE APP:
If you have anaconda installed on your device, you can view the application by follow the steps below:
<br>Open Anaconda prompt > conda activate (name of ur environment) > cd (main folder path) > streamlit run (file name for deployment)
<br>Outcome of the app is either low risk or risky.
<br>The interface of the app is shown as below:

![Image](https://github.com/innju/Heart-attack-prediction/blob/main/heart_attack/figures/streamlit_app.png)



<br>Thanks for reading.
