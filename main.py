# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=trailing-newlines
# pylint: disable=trailing-whitespace
# pylint: disable=missing-final-newline
import copy
import json
from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel

class DayAdvice(BaseModel):
    no2: str
    o3: str
    so2: str
    pm2_5: str
    pm10: str
    aqi: str

class PersonalAdvice(BaseModel):
    description: str
    no2: str
    o3: str
    so2: str
    pm2_5: str
    pm10: str
    aqi: str

class Disease(BaseModel):
    description: str

print("Loading model...")
llm = Llama(model_path="./models/mistral-7b-openorca.Q4_K_M.gguf", n_gpu_layers=50, n_ctx=4096)

print("Model loaded!")

def clean(text):
    text = text.replace("\n", "")
    text = text.replace("\\", "")
    return text

app = FastAPI()

@app.get("/")
async def hello():
    return {"hello": "world"}

@app.post("/day/advice")
async def day_advice(input: DayAdvice):
    stream = llm("""<|im_start|>system
You are a helpful chatbot and a doctor with 20 years of experience and a scientist about air quality. You will be provided information about the air quality of a place, including 
the AQI, NO2, O3, SO2, PM2.5, PM10. From that information, you will give an advice for a person to stay healthy in that day in that place. The advice should be 2 to 3 sentences.
The response should be in json format. The advice is in a string called "advice".
<|im_end|>
<|im_start|>user
The air quality metrics are as the following: AQI={}, NO2={}, O3={}, SO2={}, PM2.5={}, PM10={}<|im_end|>
<|im_start|>assistant""".format(input.aqi, input.no2, input.o3, input.so2, input.pm2_5, input.pm10), max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    raw = clean(result['choices'][0]['text'])
    return json.loads(raw)

@app.post("/predict/disease")
async def predict_disease(input: Disease):
    template = """<|im_start|>system
You are a helpful chatbot and a doctor with 20 years of experience. You will be provided a paragraph describing health issues of a person. From that paragraph, extract the list of diseases that person has. 
The response should be in json format. The disease(s) should be in a list called "diseases". 
<|im_end|>
<|im_start|>user
Here is the paragraph to extract information from: {}<|im_end|>
<|im_start|>assistant""".format(input.description)
    stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    raw = clean(result['choices'][0]['text'])
    return json.loads(raw)

@app.post("/personal/advice")
async def personal_advice(input: PersonalAdvice):
    stream = llm("""<|im_start|>system
You are a helpful chatbot and a doctor with 20 years of experience and a scientist about air quality. You will be provided information about the air quality of a place, including 
the AQI, NO2, O3, SO2, PM2.5, PM10; and a description about health issues of a person. From that information, you will give a list of 3 advices for that person to stay healthy in that day in that place. The advice should be short in 1 sentence. 
The response should be in json format. The advices should be in a list called "advices".
<|im_end|>
<|im_start|>user
The air quality metrics are as the following: AQI={}, NO2={}, O3={}, SO2={}, PM2.5={}, PM10={}. The health issues are: {}<|im_end|>
<|im_start|>assistant""".format(input.aqi, input.no2, input.o3, input.so2, input.pm2_5, input.pm10, input.description), max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    raw = clean(result['choices'][0]['text'])
    return json.loads(raw)
