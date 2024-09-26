from langchain_openai import AzureChatOpenAI, AzureOpenAI
import os
from langchain_core.messages import HumanMessage
import base64
from pathlib import Path
from config import api_key, azure_endpoint, api_version


model = AzureChatOpenAI(
    deployment_name="gpt4o-mini",#"embedding"
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    max_tokens = 4000,
    temperature = 0.1)
 
model_explanation = AzureChatOpenAI(
    deployment_name="gpt4o-mini",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    max_tokens = 4000,
    temperature = 0.1)
"""
Input: path from image want to summery

Process: 
1. turn image to base64
2. message send to model
3. summery information

Output: text including information of image
"""
def summary_img(path):
    image_path = Path(path)

    #1. Read the image file and encode it as base64
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")
    #2. message send input image and summarize information
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe information from picture"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    #3. output of image
    ai_msg = model.invoke([message]) # output
    return ai_msg.content + "in" + path


"""
Input: path from text want to summery

Process: 
1. read text from txt file

Output: content
"""
def summary_text(path):
    #1. read text from txt file
    with open(path, 'r', encoding='utf-8') as file:
        user_input = file.read().strip()
    return user_input + "in" + path # output






