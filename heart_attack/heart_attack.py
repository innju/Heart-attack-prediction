# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:48:32 2022

This script is used to develop a machine learning model and app that can be
used to predict the chance of a person having heart attack.

@author: User
"""

#%% Packages
import os
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pickle
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

#%% Static code
LOAD_DATA_PATH = os.path.join(os.getcwd() ,'data','heart.csv') 
#load data from folder named "data"
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')
# save scaler path
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'best_model_SVM.pkl')
# save model path

#%% EDA
# 1) Load data
df= pd.read_csv(LOAD_DATA_PATH)

# 2) Data inspection/visualization
df.head()
df.info()
# all in integer form except oldpeak in float64 form
df.describe().T 
# no. of count is the same for all columns
# no missing data
df.boxplot() # check for outlier
df.isnull().sum() 
# check null value
# no null value
df.duplicated().sum() # 1 duplicate
df.drop_duplicates(inplace=True)

# 3) Data cleaning
# Not needed

# 4) Feature selection
# use ExtraTreesClassifier
X = df.drop(labels=['output'],axis=1) # features
y = df['output'] # target

# feature selection using ExtraTreesClassifier
print(X.shape) # X data is data without target variable
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
print(clf.feature_importances_) # value of feature importance calculated
fs_model = SelectFromModel(clf, prefit=True) # prefit=True means pass the prefit model directly
X_new = fs_model.transform(X) # select only high value of feature importance
print(X_new.shape) # 6 features selected based on feature importance
feature_idx = fs_model.get_support()
feature_name = X.columns[feature_idx]
print(feature_name)
# 6 features selected are 'cp', 'thalachh', 'exng', 'oldpeak', 'caa', 'thall'


# 5) Data preprocessing
# 6 features selected are cp,caa,thall,exng,oldpeak and thalachh
X_input = df[['cp', 'thalachh', 'exng', 'oldpeak', 'caa', 'thall']]
y = df['output']

mms = MinMaxScaler()
X_input= mms.fit_transform(X_input)
# save scaler pickle file
mms_scaler='mms.pkl'
pickle.dump(mms,open(mms_scaler,'wb'))

X_train, X_test, y_train,y_test = train_test_split(X_input,y, test_size=0.3)


#%% Machine Learning pipeline
# ML pipeline with different classifier
steps_NB = [('NB',GaussianNB())]
steps_SVM = [('SVM',svm.SVC())]
steps_tree = [('Tree',DecisionTreeClassifier())]
steps_forest = [('Forest',RandomForestClassifier())]

# create pipeline
pipeline_NB = Pipeline(steps_NB) #To load the steps into the pipeline
pipeline_SVM = Pipeline(steps_SVM) 
pipeline_tree = Pipeline(steps_tree)
pipeline_forest = Pipeline(steps_forest)
#create a list to store all the created pipelines
pipelines= [pipeline_NB, pipeline_SVM,pipeline_tree,pipeline_forest]

#fitting of data
for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
pipe_dict = {0:'NB', 1:'SVM', 2:'Tree', 3:'Forest'}

#%% Performance evaluation
# find out the best model
# view the classification report and confusion matrix of all model
for i,model in enumerate(pipelines):
    y_pred = model.predict(X_test)
    target_names = ['low risk', 'high risk']
    print(pipe_dict[i])
    print(classification_report(y_test, y_pred,target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
# class 0 represent low risk to get heart attack
# SVM model achieves highest accuracy, 0.87%
# it is selected as the best model

#show visualization of best model(forest) selected
plot_confusion_matrix(pipeline_SVM,X_test,y_test)
plt.show()
# based on confusion matrix, the value of true positive and true negative is high
# means model tend to predict correctly most of the time
#%% Save optimal model
pickle.dump(pipeline_SVM,open(MODEL_SAVE_PATH,'wb'))










