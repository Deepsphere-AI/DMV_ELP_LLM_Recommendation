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














# Below Code Block Development in-progress - this is for California State Vehicle Law code and description similarity prediction



# Vehicle Code Divisions

def VehicleCodeDivisionAzureGPT():
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_div_tuple = ('Select Anyone','DIVISION 1. WORDS AND PHRASES DEFINED [100 - 681]', 'DIVISION 2. ADMINISTRATION [1500 - 3093]','DIVISION 3. REGISTRATION OF VEHICLES AND CERTIFICATES OF TITLE [4000 - 9808]',
                                       'DIVISION 3.5. REGISTRATION AND TRANSFER OF VESSELS [9840 - 9928]','DIVISION 3.6. VEHICLE SALES [9950 - 9993]','DIVISION 4. SPECIAL ANTITHEFT LAWS [10500 - 10904]',
                                       'DIVISION 5. OCCUPATIONAL LICENSING AND BUSINESS REGULATIONS [11100 - 12217]',"DIVISION 6. DRIVERS' LICENSES [12500 - 15326]",
                                       'DIVISION 6.5. MOTOR VEHICLE TRANSACTIONS WITH MINORS [15500 - 15501]','DIVISION 6.7. UNATTENDED CHILD IN MOTOR VEHICLE SAFETY ACT [15600 - 15632]',
                                       'DIVISION 7. FINANCIAL RESPONSIBILITY LAWS [16000 - 16560]','DIVISION 9. CIVIL LIABILITY [17000 - 17714]','DIVISION 10. ACCIDENTS AND ACCIDENT REPORTS [20000 - 20018]',
                                       'DIVISION 11. RULES OF THE ROAD [21000 - 23336]','DIVISION 11.5. SENTENCING FOR DRIVING WHILE UNDER THE INFLUENCE [23500 - 23675]','DIVISION 12. EQUIPMENT OF VEHICLES [24000 - 28160]',
                                       'DIVISION 13. TOWING AND LOADING EQUIPMENT [29000 - 31560]','DIVISION 14. TRANSPORTATION OF EXPLOSIVES [31600 - 31620]','DIVISION 14.1. TRANSPORTATION OF HAZARDOUS MATERIAL [32000 - 32053]',
                                       'DIVISION 14.3. TRANSPORTATION OF INHALATION HAZARDS [32100 - 32109]','DIVISION 14.5. TRANSPORTATION OF RADIOACTIVE MATERIALS [33000 - 33002]','DIVISION 14.7. FLAMMABLE AND COMBUSTIBLE LIQUIDS [34000 - 34100]',
                                       'DIVISION 14.8. SAFETY REGULATIONS [34500 - 34520.5]','DIVISION 14.85. MOTOR CARRIERS OF PROPERTY PERMIT ACT [34600 - 34672]','DIVISION 14.86. Private Carriers of Passengers Registration Act [34680 - 34693]',
                                       'DIVISION 14.9. MOTOR VEHICLE DAMAGE CONTROL [34700 - 34725]','DIVISION 15. SIZE, WEIGHT, AND LOAD [35000 - 35796]','DIVISION 16. IMPLEMENTS OF HUSBANDRY [36000 - 36800]',
                                       'DIVISION 16.5. OFF-HIGHWAY VEHICLES [38000 - 38604]','DIVISION 16.6. Autonomous Vehicles [38750 - 38755]','DIVISION 16.7. REGISTRATION AND LICENSING OF BICYCLES [39000 - 39011]',
                                       'DIVISION 17. OFFENSES AND PROSECUTION [40000.1 - 41610]','DIVISION 18. PENALTIES AND DISPOSITION OF FEES, FINES, AND FORFEITURES [42000 - 42277]')
    vAR_code = ''
    vAR_div = ''
    vAR_temp_div = ('DIVISION 1. WORDS AND PHRASES DEFINED [100 - 681]','DIVISION 2. ADMINISTRATION [1500 - 3093]')
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Vehicle Code Division")
    with col4:
        vAR_st.write('')
        vAR_div = vAR_st.selectbox('',vAR_div_tuple)
        vAR_st.write('')

    if vAR_div!='Select Anyone':
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        with col2:
                if vAR_div in vAR_temp_div:
                    vAR_st.write('')
                    vAR_st.subheader("Select Vehicle Code")
                    
        with col4:
                if vAR_div=='DIVISION 1. WORDS AND PHRASES DEFINED [100 - 681]':
                    vAR_code = vAR_st.selectbox('',('Select Anyone','100','102','105','108','109','110'))
                
                elif vAR_div=='DIVISION 2. ADMINISTRATION [1500 - 3093]':
                    vAR_code = vAR_st.selectbox('',('Select Anyone','1500','1501','1502','1503','1504','1505'))

                elif vAR_div  in vAR_div_tuple:
                    vAR_st.info('Development is in-progress...')



                print('code - ',vAR_code)
                print('div - ',vAR_div)
                if vAR_code!='Select Anyone' and vAR_code!='':
                    vAR_response = AzureGPTResponse(vAR_div,vAR_code)
                    vAR_st.write(vAR_response)
                    vAR_st.write('')
                    vAR_st.write('')



def AzureGPTResponse(vAR_div,vAR_code):
    prompt = "Can you give me california vehicle law for "+vAR_div+" and section : "+vAR_code
    print('VEH prompt - ',prompt)
    response = openai.ChatCompletion.create(
    engine="GPT-35-DEV-Deployment",
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    max_tokens=2000,
    top_p=1,
    # frequency_penalty=0,
    presence_penalty=0.9,

)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']







# Vehicle Law Description Match

def VehicleLawDescAzureGPT():
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Legislative Text/Description")
    with col4:
        vAR_st.write('')
        vAR_text = vAR_st.text_area('Enter the Vehicle Code Legislative Text to Match the Vehicle Codes','')
        vAR_st.write('')

    if vAR_text:
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    
                    
        with col4:
            vAR_response = AzureGPTResponseVEHDesc(vAR_text)
            vAR_st.write(vAR_response)
            vAR_st.write('')
            vAR_st.write('')




def AzureGPTResponseVEHDesc(vAR_text):
    prompt = "Can you give relevant Vehicle Code and description from california VEH CODE and legislative law for given text: '"+vAR_text+"'"
    print('VEH prompt - ',prompt)
    response = openai.ChatCompletion.create(
    engine="GPT-35-DEV-Deployment",
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    max_tokens=2000,
    top_p=1,
    # frequency_penalty=0,
    presence_penalty=0.9,

)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']
