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

from DSAI_GPT.DSAI_chatgpt import Get_Chat_DMV_Input
import streamlit as vAR_st
import json
import pandas as pd

import os
import openai

openai.api_type = os.getenv("API_TYPE")
openai.api_base = os.getenv("API_BASE")
openai.api_version = os.getenv("API_VERSION")
openai.api_key = os.getenv("AZURE_API_KEY")

def Azure_OpenAI_GPT35(vAR_input):
    response = openai.ChatCompletion.create(
    engine="GPT-35-DEV-Deployment",
    messages = [{"role":"system","content":"Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:\n1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.\n2.Deny the configuration if any one of the above categories probability value is greater than 0.2. Otherwise, accept the configuration.\n3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed."},{"role":"user","content":"Given configuration is : 'omfg'"},{"role":"assistant","content":"Category | Probability | Reason\n--- | --- | ---\nProfanity | 0.9 | 'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.\nObscene | 0.8 | 'omfg' is considered to be an obscene expression.\nInsult | 0.7 | 'omfg' can be used as an insult, depending on the context.\nHate | 0.5 | 'omfg' is not typically used to express hate, but it could be used in a hateful manner.\nToxic | 0.6 | 'omfg' can be used in a toxic manner, depending on the context.\nThreat | 0.3 | 'omfg' is not typically used to express a threat.\n\n{\"CONCLUSION\": \n\"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than 0.2.\",\n\"RECOMMENDED CONFIGURATION\": \"LUVU2\",\n\"REASON\": \"The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters.\"}"},{"role":"user","content":"Given configuration is : 'prakash'"},{"role":"assistant","content":"Category | Probability | Reason\n--- | --- | ---\nProfanity | 0.0 | 'prakash' is not a profane word.\nObscene | 0.0 | 'prakash' is not an obscene word.\nInsult | 0.0 | 'prakash' is not an insult.\nHate | 0.0 | 'prakash' is not a hateful word.\nToxic | 0.0 | 'prakash' is not a toxic word.\nThreat | 0.0 | 'prakash' is not a threat.\n\n{\"CONCLUSION\": \n\"The configuration 'prakash' is ACCEPTED as the probability value of all categories are less than 0.2.\",\n\"RECOMMENDED CONFIGURATION\": \"N/A\",\n\"REASON\": \"N/A\"}"},{"role":"user","content":"Given configuration is :'"+vAR_input+"'"},{"role":"assistant","content":"}"}],
    temperature=0,
    max_tokens=800,
    top_p=1,
    presence_penalty=0.9,
    stop=None)
    print('Azure response - ',response)
    return response["choices"][0]["message"]["content"]





# Calling defined functions for profanity validation
def DMVRecommendationAzureGPT():
    

    vAR_input = Get_Chat_DMV_Input()
    if len(vAR_input)>8 or len(vAR_input)==0:
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.write('')
            vAR_st.info("**Hint for user input:** Input length must be between 1 to 8 characters")
    elif vAR_input:
        vAR_response = Azure_OpenAI_GPT35(vAR_input)
        vAR_dict_start = vAR_response.index("{")
        vAR_dict = vAR_response[vAR_dict_start:]
        vAR_res_json = json.loads(vAR_dict)
        vAR_res_df = pd.DataFrame(vAR_res_json,index=[0])
        col1,col2,col3 = vAR_st.columns([1,15,1])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Azure OpenAI GPT-3.5(ChatGPT) Model Response and Recommendation')
            vAR_st.write('')
            vAR_st.write('')
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.write('')
            vAR_response_truncated = vAR_response[:vAR_dict_start]

            vAR_st.write(vAR_response_truncated)
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.table(vAR_res_df)
