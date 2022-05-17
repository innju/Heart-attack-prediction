# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:26:09 2022

This script is about the deployment of the best model saved.

@author: User
"""

# packages
import os
import pickle
import streamlit as st
import numpy as np

#%% static code
# load path consisting of data
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'mms.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'best_model_forest.pkl')

#%% 1) Load model and scaler
scaler = pickle.load(open(SCALER_SAVE_PATH,'rb'))
model = pickle.load(open(MODEL_SAVE_PATH,'rb'))

#%% Deployment using streamlit
# create the form
with st.form('Predicition for risk of heart attack'):
    st.write("Patient's info")
    #features selected based on feature selection
    cp = int(st.number_input('cp, chest pain type where 1: Typical angina,2: Atypical angina,3: Non-anginal pain, 4: Asymptomatic'))
    thalachh = st.number_input(' thalachh, Maximum hear rate achieved')
    exng = int(st.number_input('exng, Exercise induced angina where 1:Yes, 2: No'))
    oldpeak = st.number_input('oldpeak, ST depression induced by exercise relative to rest')
    caa = int(st.number_input('caa, Number of major vessels'))
    thall = int(st.number_input('thall,Thalessemin'))
    
    submitted= st.form_submit_button('Submit')
    
    # to observe if the information appear if i click submit
    if submitted == True:
        patient_info = np.array([cp,thalachh,exng,oldpeak,caa,thall])
        patient_info_scaled= scaler.transform(np.expand_dims(patient_info,axis=0))
        outcome = model.predict(patient_info_scaled)
        heart_attack_chance ={0:'low risk',1:'risky'}
        st.write(heart_attack_chance[np.argmax(outcome)])
        
        if np.argmax(outcome) == 1:
            st.warning('You have high risk for heart attack!')
        else:
            st.success('You have low risk for heart attack!')
            


