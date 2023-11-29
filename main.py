import copy
import asyncio
import requests
import json

from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel

class Input(BaseModel):
    info: str
    aqi: int

class Disease(BaseModel):
    description: str

print("Loading model...")
llm = Llama(model_path="./models/mistral-7b-v0.1.Q4_0.gguf")

print("Model loaded!")

app = FastAPI()

@app.get("/")
async def hello():
    return {"hello": "wooooooorld"}

@app.post("/daily/advice")
async def model(input: Input):
    stream = llm("""<|im_start|>system
You are a helpful chatbot. You are going to answer questions about health and give appropriate advice (for that day) based on the health information provided. The response should be in json format. Each advice is an element which consists of 2 parts: the name of advice, and its detailed description. You must response at least 3 advices.
<|im_end|>
<|im_start|>user
{}. What are some advice for exposing to a place with AQI of {}? The advices should be some tips for them to stay healthy while exposing.<|im_end|>
<|im_start|>assistant""".format(input.info, input.aqi), max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    raw = result['choices'][0]['text']
    raw = raw.replace("\n", "")
    raw = raw.replace("\\", "")
    return json.loads(raw)

@app.post("/predict/disease")
async def predict(input: Disease):
    template = """<|im_start|>system
You are a helpful chatbot. You will be provided a paragraph describing health issues of a person. From that paragraph, extract the list of diseases that person has. 
The response should be in json format. The disease(s) should be in a list called "diseases". 
<|im_end|>
<|im_start|>user
Here is the paragraph to extract information from: {}<|im_end|>
<|im_start|>assistant""".format(input.description)
    stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    raw = result['choices'][0]['text']
    raw = raw.replace("\n", "")
    raw = raw.replace("\\", "")
    return json.loads(raw)