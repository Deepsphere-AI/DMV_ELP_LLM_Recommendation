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


import streamlit as vAR_st
vAR_st.set_page_config(page_title="DMV Recommendation", layout="wide")

from DSAI_Utility.DSAI_Utility import All_Initialization,CSS_Property
from DSAI_GPT.DSAI_gpt3 import DMVRecommendationGPT,VehicleCodeDivisionGPT,VehicleLawDescGPT
from DSAI_GPT.DSAI_chatgpt import DMVRecommendationChatGPT,VehicleCodeDivisionChatGPT,VehicleLawDescChatGPT
from DSAI_PaLM.DSAI_PaLM import DMVRecommendationPaLM

import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'C:\Users\ds_007\Downloads\personalized-learning-340207-699519426800.json'

if __name__=='__main__':
    vAR_hide_footer = """<style>
            footer {visibility: hidden;}
            </style>
            """
    vAR_st.markdown(vAR_hide_footer, unsafe_allow_html=True)
    # try:
    # Applying CSS properties for web page
    CSS_Property("DSAI_Utility/DSAI_style.css")
    # Initializing Basic Componentes of Web Page
    All_Initialization()


    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Select the Model & API')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Select the Functionality')
    with col4:
        vAR_st.write('')
        vAR_option = vAR_st.selectbox('',('Select a Model','GPT-3', 'ChatGPT',"Google's PaLM"))
        vAR_st.write('')
        vAR_option2 = vAR_st.selectbox('',('Select anyone','ELP Recommendation', 'Vehicle Code Divisions','Vehicle Law Descripiton'))

    if vAR_option=='ChatGPT' and vAR_option2=='ELP Recommendation':
        # Calling ChatGPT
        DMVRecommendationChatGPT()
    elif vAR_option=='GPT-3' and vAR_option2=='ELP Recommendation':
        # Calling GPT3
        DMVRecommendationGPT()
    elif vAR_option=='ChatGPT' and vAR_option2=='Vehicle Code Divisions':
        VehicleCodeDivisionChatGPT()
    elif vAR_option=='GPT-3' and vAR_option2=='Vehicle Code Divisions':
        VehicleCodeDivisionGPT()
    elif vAR_option=='ChatGPT' and vAR_option2=='Vehicle Law Descripiton':
        VehicleLawDescChatGPT()
    elif vAR_option=='GPT-3' and vAR_option2=='Vehicle Law Descripiton':
        VehicleLawDescGPT()
    elif vAR_option=="Google's PaLM" and vAR_option2=='ELP Recommendation':
        DMVRecommendationPaLM()
    else:
        pass


    # except BaseException as exception:
        # print('Error in main function - ', exception)
        # exception = 'Something went wrong - '+str(exception)
        # vAR_st.error(exception)
