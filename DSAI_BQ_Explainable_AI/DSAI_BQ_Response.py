
"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
© Copyright 2022, California, Department of Motor Vehicle, all rights reserved.
The source code and all its associated artifacts belong to the California Department of Motor Vehicle (CA, DMV), and no one has any ownership
and control over this source code and its belongings. Any attempt to copy the source code or repurpose the source code and lead to criminal
prosecution. Don't hesitate to contact DMV for further information on this copyright statement.

Release Notes and Development Platform:
The source code was developed on the Google Cloud platform using Google Cloud Functions serverless computing architecture. The Cloud
Functions gen 2 version automatically deploys the cloud function on Google Cloud Run as a service under the same name as the Cloud
Functions. The initial version of this code was created to quickly demonstrate the role of MLOps in the ELP process and to create an MVP. Later,
this code will be optimized, and Python OOP concepts will be introduced to increase the code reusability and efficiency.
____________________________________________________________________________________________________________
Development Platform                | Developer       | Reviewer   | Release  | Version  | Date
____________________________________|_________________|____________|__________|__________|__________________
Google Cloud Serverless Computing   | DMV Consultant  | Ajay Gupta | Initial  | 1.0      | 09/18/2022

-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

import requests
import os
import json
import streamlit as vAR_st
from DSAI_GPT.DSAI_chatgpt import Get_Chat_DMV_Input



def GetConfigDetailsResponse():
    

    vAR_input = Get_Chat_DMV_Input()
    if len(vAR_input)>8 or len(vAR_input)==0:
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.write('')
            vAR_st.info("**Hint for user input:** Input length must be between 1 to 8 characters")
    elif vAR_input:
        vAR_response = GetConfigDetails(vAR_input)
        
        col1,col2,col3 = vAR_st.columns([5,10,5])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('ELP Configuration - Explainable AI')
            vAR_st.write('')
            vAR_st.write('')
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.json(vAR_response)
        



def GetConfigDetails(vAR_config):

    vAR_url = os.environ["HTTPS_URL"]

    vAR_headers = {"Content-Type": "application/json; charset=utf-8","x-api-key":os.environ["X_API_KEY"]}

    vAR_data = {
        "licencePlateConfig": vAR_config,
        "function": 6,
    }

    vAR_response = requests.post(vAR_url, headers=vAR_headers, json=vAR_data)

    return vAR_response.json()
