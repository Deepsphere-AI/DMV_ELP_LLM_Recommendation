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
# from DSAI_GPT.DSAI_gpt3 import DMVRecommendationGPT,VehicleCodeDivisionGPT,VehicleLawDescGPT
from DSAI_PaLM.DSAI_PaLM_Chat import DMVRecommendationPaLMChat
from DSAI_PaLM.DSAI_PaLM_Text import DMVRecommendationPaLMText
from DSAI_LLM_Comparison.DSAI_LLM_Comparison import DMVRecommendationModelComparison
from DSAI_Azure_GPT.DSAI_Azure_GPT35 import DMVRecommendationAzureGPT,VehicleCodeDivisionAzureGPT,VehicleLawDescAzureGPT
from DSAI_BQ_Explainable_AI.DSAI_BQ_Response import GetConfigDetailsResponse
from DSAI_Driver_Risk_Score_Prediction.DSAI_Risk_Prediction import DriverRiskPrediction

import os


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
        vAR_option = vAR_st.selectbox('',('Select a Model',"Google's PaLM(text-bison)","Google's PaLM(chat-bison)","Azure OpenAI ChatGPT(GPT-3.5)","LLM Model Comparison"))
        vAR_st.write('')
        vAR_option2 = vAR_st.selectbox('',('Select anyone','ELP Recommendation', 'Prompt - Vehicle Code','Prompt - Vehicle Code Legislative Text','ELP Explainable AI','Driver Risk Score Prediction'))


    





    # if vAR_option=='GPT-4(ChatGPT)' and vAR_option2=='ELP Recommendation':
        
    #     DMVRecommendationChatGPT()
    # elif vAR_option=='GPT-3' and vAR_option2=='ELP Recommendation':
    #     # Calling GPT3
    #     DMVRecommendationGPT()
    # elif vAR_option=='GPT-4(ChatGPT)' and vAR_option2=='Prompt - Vehicle Code':
    #     VehicleCodeDivisionChatGPT()
    # elif vAR_option=='GPT-3' and vAR_option2=='Prompt - Vehicle Code':
    #     VehicleCodeDivisionGPT()
    # elif vAR_option=='GPT-4(ChatGPT)' and vAR_option2=='Prompt - Vehicle Code Legislative Text':
    #     VehicleLawDescChatGPT()
    # elif vAR_option=='GPT-3' and vAR_option2=='Prompt - Vehicle Code Legislative Text':
    #     VehicleLawDescGPT()

    if vAR_option=='Azure OpenAI ChatGPT(GPT-3.5)' and vAR_option2=='Prompt - Vehicle Code':
        VehicleCodeDivisionAzureGPT()

    elif vAR_option=='Azure OpenAI ChatGPT(GPT-3.5)' and vAR_option2=='Prompt - Vehicle Code Legislative Text':
        VehicleLawDescAzureGPT()



    elif vAR_option=="Google's PaLM(chat-bison)" and vAR_option2=='ELP Recommendation':
        DMVRecommendationPaLMChat()
    elif vAR_option=="Google's PaLM(text-bison)" and vAR_option2=='ELP Recommendation':
        DMVRecommendationPaLMText()

    elif vAR_option=="Azure OpenAI ChatGPT(GPT-3.5)" and vAR_option2=='ELP Recommendation':        
        DMVRecommendationAzureGPT()
    elif vAR_option=="LLM Model Comparison" and vAR_option2=='ELP Recommendation':
        DMVRecommendationModelComparison()
    
    elif vAR_option2=="ELP Explainable AI":
        GetConfigDetailsResponse()
        
    elif vAR_option2=="Driver Risk Score Prediction":
        DriverRiskPrediction()





    else:
        pass


    # except BaseException as exception:
    #     print('Error in main function - ', exception)
    #     exception = 'Something went wrong - '+str(exception)
    #     vAR_st.error(exception)
