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

from DSAI_GPT.DSAI_gpt3 import Get_Chat_DMV_Input
from DSAI_LLM_Comparison.DSAI_Model_Recommendations import Chat_Conversation_GPT3,Chat_Conversation,DMVRecommendationPaLMChatModel,DMVRecommendationPaLMTextModel,Azure_OpenAI_GPT35


import os
import pandas as pd
pd.set_option('display.max_colwidth', 500)
import streamlit as vAR_st
import json




def DMVRecommendationModelComparison():
    
    vAR_input = Get_Chat_DMV_Input()
    if len(vAR_input)>8 or len(vAR_input)==0:
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.write('')
            vAR_st.info("**Hint for user input:** Input length must be between 1 to 8 characters")
    elif vAR_input:
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            # vAR_response_gpt3 = Chat_Conversation_GPT3(vAR_input)
            # vAR_response_gpt3_dict = json.loads(vAR_response_gpt3)
            # vAR_response_gpt3_dict["Model"] = "GPT3"
            # vAR_response_gpt3_df = pd.DataFrame(vAR_response_gpt3_dict,index=[0])
            # print('GPT3 COMPLETED')
            # vAR_st.info('GPT3 COMPLETED')
            
            
            # vAR_response_gpt4 = Chat_Conversation(vAR_input)
            # vAR_response_gpt4_dict = json.loads(vAR_response_gpt4)
            # vAR_response_gpt4_dict["Model"] = "GPT-4(ChatGPT)"
            # vAR_response_gpt4_df = pd.DataFrame(vAR_response_gpt4_dict,index=[0])
            # print('GPT-4(ChatGPT) COMPLETED')
            # vAR_st.info('GPT-4(ChatGPT) COMPLETED')
            
            
            vAR_response_palm_chat = ""
            vAR_response_palm_chat_df = pd.DataFrame()
            try:
                
                vAR_response_palm_chat = DMVRecommendationPaLMChatModel(os.environ["PROJECT_ID"],vAR_input,"us-central1")
                vAR_response_palm_chat_dict = json.loads(vAR_response_palm_chat)
                vAR_response_palm_chat_dict["Model"] = "PaLM Chat"
                vAR_response_palm_chat_df = pd.DataFrame(vAR_response_palm_chat_dict,index=[0])
                print('PaLM Chat COMPLETED')
                vAR_st.info('PaLM Chat COMPLETED')
            except BaseException as e:
                print("LLM Comparison exception block")
                vAR_st.error("PaLM Chat Model Fails to Give Response in Expected Format. Model Response: "+vAR_response_palm_chat)
                
                
            
            
            
            vAR_response_palm_text = DMVRecommendationPaLMTextModel(os.environ["PROJECT_ID"],vAR_input,"us-central1")
            vAR_response_palm_text_dict = json.loads(vAR_response_palm_text)
            vAR_response_palm_text_dict["Model"] = "PaLM Text"
            vAR_response_palm_text_df = pd.DataFrame(vAR_response_palm_text_dict,index=[0])
            print('PaLM Text COMPLETED')
            vAR_st.info('PaLM Text COMPLETED')

            vAR_response_azure_gpt_35 = Azure_OpenAI_GPT35(vAR_input)
            vAR_response_azure_gpt_35_dict = json.loads(vAR_response_azure_gpt_35)
            vAR_response_azure_gpt_35_dict["Model"] = "Azure OpenAI ChatGPT(GPT-3.5)"
            vAR_response_azure_gpt_35_df = pd.DataFrame(vAR_response_azure_gpt_35_dict,index=[0])
            print('Azure OpenAI ChatGPT(GPT-3.5) COMPLETED')
            vAR_st.info('Azure OpenAI ChatGPT(GPT-3.5) COMPLETED')
            
            # vAR_final_response_df = pd.concat([vAR_response_gpt3_df, vAR_response_gpt4_df], ignore_index=True)
            # vAR_final_response_df = pd.concat([vAR_final_response_df, vAR_response_palm_chat_df], ignore_index=True)
            # vAR_final_response_df = pd.concat([vAR_final_response_df, vAR_response_palm_text_df], ignore_index=True)

            vAR_final_response_df = pd.concat([vAR_response_palm_text_df, vAR_response_palm_chat_df], ignore_index=True)
            vAR_final_response_df = pd.concat([vAR_final_response_df, vAR_response_azure_gpt_35_df], ignore_index=True)

            
            # Since append deprecated latest version of pandas
            
            # vAR_final_response_df = vAR_response_gpt3_df.append(vAR_response_gpt4_df,ignore_index=True)
            # vAR_final_response_df = vAR_final_response_df.append(vAR_response_palm_chat_df,ignore_index=True)
            # vAR_final_response_df = vAR_final_response_df.append(vAR_response_palm_text_df,ignore_index=True)

            vAR_res_csv = Convert_Df(vAR_final_response_df)
            vAR_st.write('')
            vAR_st.write('')
            
            
            vAR_st.download_button(
        label="Download LLM Comparison as CSV",
        data=vAR_res_csv,
        file_name='LLMComparison_'+vAR_input+'.csv',
        mime='text/csv',
    )

    
    
@vAR_st.cache
def Convert_Df(df):
    return df.to_csv().encode('utf-8')
