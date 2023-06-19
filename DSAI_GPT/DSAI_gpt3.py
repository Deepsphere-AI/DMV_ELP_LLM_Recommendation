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



import os
import openai
import streamlit as vAR_st
import json
import pandas as pd

openai.api_key = os.environ["API_KEY"]


# Calling defined functions for profanity validation
def DMVRecommendationGPT():
    

    vAR_input = Get_Chat_DMV_Input()
    if len(vAR_input)>8 or len(vAR_input)==0:
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.write('')
            vAR_st.info("**Hint for user input:** Input length must be between 1 to 8 characters")
    elif vAR_input:
        vAR_response = Chat_Conversation(vAR_input)
        vAR_dict_start = vAR_response.index("{")
        vAR_dict = vAR_response[vAR_dict_start:]
        vAR_res_json = json.loads(vAR_dict)
        vAR_res_df = pd.DataFrame(vAR_res_json,index=[0])
        col1,col2,col3 = vAR_st.columns([1,15,1])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('ChatGPT Model Response and Recommendation')
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


        


# OpenAI API Call
def Chat_Conversation(vAR_input):

    prompt = """Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:
1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.
2.Denied the configuration if any one of the above categories probability value is greater than or equal to 0.2 Otherwise, accept the configuration.
3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed.

Given configuration is : 'omfg'

Category | Probability | Reason
--- | --- | ---
Profanity | 0.9 | 'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.
Obscene | 0.8 | 'omfg' is considered to be an obscene expression.
Insult | 0.7 | 'omfg' can be used as an insult, depending on the context.
Hate | 0.5 | 'omfg' is not typically used to express hate, but it could be used in a hateful manner.
Toxic | 0.6 | 'omfg' can be used in a toxic manner, depending on the context.
Threat | 0.3 | 'omfg' is not typically used to express a threat.

{"CONCLUSION": 
"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than or equal to 0.2.",
"RECOMMENDED CONFIGURATION": "LUVU2",
"REASON": "The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."}


Given configuration is : 'racism'

Category | Probability | Reason
--- | --- | ---
Profanity | 0 | The word 'racism' does not contain any profane language.
Obscene | 0 | The word 'racism' does not contain any obscene language.
Insult | 0.5 | The word 'racism' can be used as an insult, depending on the context.
Hate | 0.8 | The word 'racism' is often used to express hatred towards a certain group of people.
Toxic | 0.7 | The word 'racism' can be used to express toxic views and opinions.
Threat | 0 | The word 'racism' does not contain any threatening language.

{"CONCLUSION":
"The configuration 'RACISM' is DENIED as the probability value for Obscene, Insult, and Hate categories are greater than or equal to 0.2.",
"RECOMMENDED CONFIGURATION": "EQUALITY", 
"REASON": "This configuration does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters. It is recommended because it conveys a positive message of equality and respect."}

Given configuration is : '"""+vAR_input.lower()+"'"
    # prompt = "Please provide the probability value and reason for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table for the given word.'"+vAR_input.lower()+"'"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.9,
    stop=[" Human:", " AI:"]
    )
    print('Chat prompt - ',prompt)
    return response["choices"][0]["text"]



# Read User Input For ELP Configuration
def Get_Chat_DMV_Input():
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






# Below Code Block Development in-progress - this is for California State Vehicle Law code and description similarity prediction

# Vehicle Code Divisions

def VehicleCodeDivisionGPT():
    
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
                    vAR_response = GPTResponse(vAR_div,vAR_code)
                    vAR_st.write(vAR_response)
                    vAR_st.write('')
                    vAR_st.write('')





def GPTResponse(vAR_div,vAR_code):

    prompt = "Can you give me california vehicle law for "+vAR_div+" and section : "+vAR_code
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.9
    )
    print('VEH GPT prompt - ',prompt)
    return response["choices"][0]["text"]







# Vehicle Law Description Match

def VehicleLawDescGPT():
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Legislative Text/Description")
    with col4:
        vAR_st.write('')
        vAR_text = vAR_st.text_area('Enter Vehicle Law Description to Match the Vehicle Codes','')
        vAR_st.write('')

    if vAR_text:
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    
                    
        with col4:
            vAR_response = GPTResponseVEHDesc(vAR_text)
            vAR_st.write(vAR_response)
            vAR_st.write('')
            vAR_st.write('')


def GPTResponseVEHDesc(vAR_text):

    prompt = "Can you give relevant Vehicle Code and description from california VEH CODE and legislative law for given text: '"+vAR_text+"'"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.9
    )
    print('VEH GPT prompt - ',prompt)
    return response["choices"][0]["text"]