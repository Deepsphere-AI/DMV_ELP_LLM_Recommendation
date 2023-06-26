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


import vertexai
from vertexai.preview.language_models import ChatModel,InputOutputTextPair
import os


import streamlit as vAR_st
import json
import pandas as pd

def DMVRecommendationPaLMChatModel(
    project_id: str,
    vAR_input: str,
    location: str = "us-central1",
    ) :
    """Predict using a Large Language Model."""

    vertexai.init(project=os.environ["PROJECT_ID"], location="us-central1")
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat(
        context="""Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:
    1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.
    2.Deny the configuration if any one of the above categories probability value is greater than 0.2. Otherwise, accept the configuration.
    3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed.""",
        examples=[
            InputOutputTextPair(
                input_text="""Given configuration is : 'omfg'""",
                output_text="""
Category | Probability | Reason
--- | --- | ---
Profanity | 0.9 | 'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.
Obscene | 0.8 | 'omfg' is considered to be an obscene expression.
Insult | 0.7 | 'omfg' can be used as an insult, depending on the context.
Hate | 0.5 | 'omfg' is not typically used to express hate, but it could be used in a hateful manner.
Toxic | 0.6 | 'omfg' can be used in a toxic manner, depending on the context.
Threat | 0.3 | 'omfg' is not typically used to express a threat.

{"CONCLUSION": 
"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than 0.2.",
"RECOMMENDED CONFIGURATION": "LUVU2",
"REASON": "The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."}"""
            ),
            InputOutputTextPair(
                input_text="""Given configuration is : 'motor'""",
                output_text="""
Category | Probability | Reason
--- | --- | ---
Profanity | 0.0 | 'motor' is not a profane word.
Obscene | 0.0 | 'motor' is not an obscene word.
Insult | 0.0 | 'motor' is not an insult.
Hate | 0.0 | 'motor' is not a hateful word.
Toxic | 0.0 | 'motor' is not a toxic word.
Threat | 0.0 | 'motor' is not a threat.

{"CONCLUSION": 
"The configuration 'motor' is ACCEPTED as the probability value of all categories are less than 0.2.",
"RECOMMENDED CONFIGURATION": "N/A",
"REASON": "N/A"}"""
            )
        ]
    )
    response = chat.send_message("Given configuration is :'"+vAR_input+"'", **parameters)
    print(f"Response from Model: {response.text}")
    return response.text




















# Calling PaLM API method
def DMVRecommendationPaLMChat():
    

    vAR_input = Get_PaLM_DMV_Input()
    if len(vAR_input)>8 or len(vAR_input)==0:
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.write('')
            vAR_st.info("**Hint for user input:** Input length must be between 1 to 8 characters")
    elif vAR_input:
        col1,col2,col3 = vAR_st.columns([1,15,1])
        vAR_response = ""
        try:
            vAR_response = DMVRecommendationPaLMChatModel(os.environ["PROJECT_ID"], vAR_input,"us-central1")
            vAR_dict_start = vAR_response.index("{")
            vAR_dict = vAR_response[vAR_dict_start:]
            vAR_res_json = json.loads(vAR_dict)
            vAR_res_df = pd.DataFrame(vAR_res_json,index=[0])
            
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.subheader("Google's PaLM Model Response and Recommendation")
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
        except BaseException as e:
            col1,col2,col3 = vAR_st.columns([2,15,2])
            with col2:
                vAR_st.write('')
                print("Exception Occurs(Model fails to give the response in the expected format)")
                vAR_st.error("Exception Occurs(Model fails to give the response in the expected format) - "+str(e))
                vAR_st.error("Response From Model : "+vAR_response)



        


# Read User Input For ELP Configuration
def Get_PaLM_DMV_Input():
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("ELP Configuration")
        
        vAR_st.write('')
        vAR_st.write('')
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_input = vAR_st.text_input('',placeholder='Enter ELP Configuration')
        return vAR_input





