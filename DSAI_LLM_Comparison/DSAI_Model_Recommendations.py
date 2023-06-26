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
import pandas as pd
import streamlit as vAR_st
import openai
import vertexai
from vertexai.preview.language_models import ChatModel,InputOutputTextPair
from vertexai.language_models import TextGenerationModel


# OpenAI API Call
def Chat_Conversation_GPT3(vAR_input):

    prompt = """Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:
1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.
2.Denied the configuration if any one of the above categories probability value is greater than or equal to 0.2 Otherwise, accept the configuration.
3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed.

Given configuration is : 'omfg'

{"Profanity":0.9,"Profanity Reason":"'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.",
            "Obscene":0.8,"Obscene Reason":"'omfg' is considered to be an obscene expression.",
            "Insult":0.7,"Insult Reason":"'omfg' can be used as an insult, depending on the context.",
            "Hate":0.5,"Hate Reason":"'omfg' is not typically used to express hate, but it could be used in a hateful manner.",
            "Toxic":0.6,"Toxic Reason":"'omfg' can be used in a toxic manner, depending on the context.",
            "Threat":0.3,"Threat Reason":"'omfg' is not typically used to express a threat.",
"Conclusion":"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than or equal to 0.2.","Recommended Configuration":"LUVU2",
"Recommendation Reason":"The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."}



Given configuration is : 'motor'

{"Profanity":0.0,"Profanity Reason":"'motor' is not a profane word.",
            "Obscene":0.0,"Obscene Reason":"'motor' is not an obscene word.",
            "Insult":0.0,"Insult Reason":"'motor' is not an insult word.",
            "Hate":0.0,"Hate Reason":"'motor' is not a hateful word.",
            "Toxic":0.0,"Toxic Reason":"'motor' is not a toxic word.",
            "Threat":0.0,"Threat Reason":"'motor' is not athreat.",
"Conclusion":"The configuration 'motor' is ACCEPTED as the probability value of all categories are less than 0.2.","Recommended Configuration":"N/A",
"Recommendation Reason":"N/A"}

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






# OpenAI API Call
def Chat_Conversation(vAR_input):

    response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    model="gpt-4",
    messages=[
        {"role": "user", "content": """Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:
1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.
2.Deny the configuration if any one of the above categories probability value is greater than 0.2. Otherwise, accept the configuration.
3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed.

Given configuration is : 'omfg'

{"Profanity":0.9,"Profanity Reason":"'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.",
            "Obscene":0.8,"Obscene Reason":"'omfg' is considered to be an obscene expression.",
            "Insult":0.7,"Insult Reason":"'omfg' can be used as an insult, depending on the context.",
            "Hate":0.5,"Hate Reason":"'omfg' is not typically used to express hate, but it could be used in a hateful manner.",
            "Toxic":0.6,"Toxic Reason":"'omfg' can be used in a toxic manner, depending on the context.",
            "Threat":0.3,"Threat Reason":"'omfg' is not typically used to express a threat.",
"Conclusion":"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than or equal to 0.2.","Recommended Configuration":"LUVU2",
"Recommendation Reason":"The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."}



Given configuration is : 'motor'

{"Profanity":0.0,"Profanity Reason":"'motor' is not a profane word.",
            "Obscene":0.0,"Obscene Reason":"'motor' is not an obscene word.",
            "Insult":0.0,"Insult Reason":"'motor' is not an insult word.",
            "Hate":0.0,"Hate Reason":"'motor' is not a hateful word.",
            "Toxic":0.0,"Toxic Reason":"'motor' is not a toxic word.",
            "Threat":0.0,"Threat Reason":"'motor' is not athreat.",
"Conclusion":"The configuration 'motor' is ACCEPTED as the probability value of all categories are less than 0.2.","Recommended Configuration":"N/A",
"Recommendation Reason":"N/A"}

Given configuration is :'"""+vAR_input.lower()+"'"},
    ],
    temperature=0,
    max_tokens=2000,
    top_p=1,
    # frequency_penalty=0,
    presence_penalty=0.9,

)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']






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
{"Profanity":0.9,"Profanity Reason":"'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.",
            "Obscene":0.8,"Obscene Reason":"'omfg' is considered to be an obscene expression.",
            "Insult":0.7,"Insult Reason":"'omfg' can be used as an insult, depending on the context.",
            "Hate":0.5,"Hate Reason":"'omfg' is not typically used to express hate, but it could be used in a hateful manner.",
            "Toxic":0.6,"Toxic Reason":"'omfg' can be used in a toxic manner, depending on the context.",
            "Threat":0.3,"Threat Reason":"'omfg' is not typically used to express a threat.",
"Conclusion":"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than or equal to 0.2.","Recommended Configuration":"LUVU2",
"Recommendation Reason":"The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."}

"""
            ),
            InputOutputTextPair(
                input_text="""Given configuration is : 'motor'""",
                output_text="""
{"Profanity":0.0,"Profanity Reason":"'motor' is not a profane word.",
            "Obscene":0.0,"Obscene Reason":"'motor' is not an obscene word.",
            "Insult":0.0,"Insult Reason":"'motor' is not an insult word.",
            "Hate":0.0,"Hate Reason":"'motor' is not a hateful word.",
            "Toxic":0.0,"Toxic Reason":"'motor' is not a toxic word.",
            "Threat":0.0,"Threat Reason":"'motor' is not athreat.",
"Conclusion":"The configuration 'motor' is ACCEPTED as the probability value of all categories are less than 0.2.","Recommended Configuration":"N/A",
"Recommendation Reason":"N/A"}

"""
            )
        ]
    )
    response = chat.send_message("Given configuration is :'"+vAR_input+"'", **parameters)
    print(f"Response from Model: {response.text}")
    return response.text










def DMVRecommendationPaLMTextModel(
    project_id: str,
    vAR_input: str,
    location: str = "us-central1",
    ) :
    """Predict using a Large Language Model."""


    vertexai.init(project=os.environ["PROJECT_ID"], location="us-central1")
    parameters = {
        "temperature": 0,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(
        """Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:
    1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.
    2.Deny the configuration if any one of the above categories probability value is greater than 0.2. Otherwise, accept the configuration.
    3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed.

input: Given configuration is : 'omfg'
output: 
{"Profanity":0.9,"Profanity Reason":"'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.",
            "Obscene":0.8,"Obscene Reason":"'omfg' is considered to be an obscene expression.",
            "Insult":0.7,"Insult Reason":"'omfg' can be used as an insult, depending on the context.",
            "Hate":0.5,"Hate Reason":"'omfg' is not typically used to express hate, but it could be used in a hateful manner.",
            "Toxic":0.6,"Toxic Reason":"'omfg' can be used in a toxic manner, depending on the context.",
            "Threat":0.3,"Threat Reason":"'omfg' is not typically used to express a threat.",
"Conclusion":"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than or equal to 0.2.","Recommended Configuration":"LUVU2",
"Recommendation Reason":"The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."}



input: Given configuration is : 'motor'
output: 

{"Profanity":0.0,"Profanity Reason":"'motor' is not a profane word.",
            "Obscene":0.0,"Obscene Reason":"'motor' is not an obscene word.",
            "Insult":0.0,"Insult Reason":"'motor' is not an insult word.",
            "Hate":0.0,"Hate Reason":"'motor' is not a hateful word.",
            "Toxic":0.0,"Toxic Reason":"'motor' is not a toxic word.",
            "Threat":0.0,"Threat Reason":"'motor' is not athreat.",
"Conclusion":"The configuration 'motor' is ACCEPTED as the probability value of all categories are less than 0.2.","Recommended Configuration":"N/A",
"Recommendation Reason":"N/A"}


input: Given configuration is :'"""+vAR_input+"'"+
"""output:
""",
        **parameters
    )
    print(f"Response from Model: {response.text}")
    return response.text





def Azure_OpenAI_GPT35(vAR_input):
    openai.api_type = os.getenv("API_TYPE")
    openai.api_base = os.getenv("API_BASE")
    openai.api_version = os.getenv("API_VERSION")
    openai.api_key = os.getenv("AZURE_API_KEY")
    response = openai.ChatCompletion.create(
    engine="GPT-35-DEV-Deployment",
    messages = [{"role":"system","content":"""Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:\n1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.\n2.Deny the configuration if any one of the above categories probability value is greater than 0.2. Otherwise, accept the configuration.\n3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed."""},{"role":"user","content":"Given configuration is : 'omfg'"},{"role":"assistant","content":"""{"Profanity":0.9,"Profanity Reason":"'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.",
            "Obscene":0.8,"Obscene Reason":"'omfg' is considered to be an obscene expression.",
            "Insult":0.7,"Insult Reason":"'omfg' can be used as an insult, depending on the context.",
            "Hate":0.5,"Hate Reason":"'omfg' is not typically used to express hate, but it could be used in a hateful manner.",
            "Toxic":0.6,"Toxic Reason":"'omfg' can be used in a toxic manner, depending on the context.",
            "Threat":0.3,"Threat Reason":"'omfg' is not typically used to express a threat.",
"Conclusion":"The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than or equal to 0.2.","Recommended Configuration":"LUVU2",
"Recommendation Reason":"The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."}
"""},{"role":"user","content":"Given configuration is : 'motor'"},{"role":"assistant","content":"""
{"Profanity":0.0,"Profanity Reason":"'motor' is not a profane word.",
            "Obscene":0.0,"Obscene Reason":"'motor' is not an obscene word.",
            "Insult":0.0,"Insult Reason":"'motor' is not an insult word.",
            "Hate":0.0,"Hate Reason":"'motor' is not a hateful word.",
            "Toxic":0.0,"Toxic Reason":"'motor' is not a toxic word.",
            "Threat":0.0,"Threat Reason":"'motor' is not athreat.",
"Conclusion":"The configuration 'motor' is ACCEPTED as the probability value of all categories are less than 0.2.","Recommended Configuration":"N/A",
"Recommendation Reason":"N/A"}

"""},{"role":"user","content":"Given configuration is :'"+vAR_input+"'"}],
    temperature=0,
    max_tokens=800,
    top_p=1,
    presence_penalty=0.9,
    stop=None)
    print('Azure response - ',response)
    return response["choices"][0]["message"]["content"]