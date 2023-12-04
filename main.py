# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=trailing-newlines
# pylint: disable=trailing-whitespace
# pylint: disable=missing-final-newline
# pylint: disable=line-too-long
# pylint: disable=consider-using-f-string
import copy
import json
from functools import wraps
from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel

MAX_RETRY = 5

class WorkoutGenerator(BaseModel):
    partOfBody: str
    level: str
    goal: str
    typeOfWorkout: str
    issues_description: str


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

class Prompt(BaseModel):
    system_promt: str
    user_promt: str

print("Loading model...")
llm = Llama(model_path="./models/mistral-7b-openorca.Q4_K_M.gguf", n_gpu_layers=50, n_ctx=4096)

print("Model loaded!")

def logging(func):
    @wraps(func)
    async def decorator(*args, **kwargs):
        response = await func(*args, **kwargs)
        print("GOT RESPONSE: ", response)
        return response
    return decorator

def clean(text):
    text = text.replace("\n", "")
    text = text.replace("\\", "")
    return text

app = FastAPI()

@app.get("/")
async def hello():
    return {"hello": "world"}

@app.post("/day/advice")
@logging
async def day_advice(req: DayAdvice):
    template = """<|im_start|>system
CONTEXT: You are a virtual assistant designed to provide personalized advice to individuals based on current Air Quality Index (AQI) metrics. Users will input the AQI values for specific pollutants, and your role is to analyze these values and offer actionable guidance. INPUT: Users will provide AQI values for key pollutants such as NO2, O3, SO2, PM2.5, and PM10, along with the overall AQI. DESIRED OUTPUT: Your responses should briefly assess the overall air quality based on the provided AQI values. Offer practical advice tailored to the current air quality situation. This may include recommendations for outdoor activities, indoor precautions, or health considerations. The advice should be 2 to 3 sentences. Note that you should not use the pronoun \"I\" or \"We\" in your advice. The response should be in json format. The advice is in a string called "advice". 
<|im_end|>
<|im_start|>user
The air quality metrics are as the following: AQI={}, NO2={}, O3={}, SO2={}, PM2.5={}, PM10={}<|im_end|>
<|im_start|>assistant""".format(req.aqi, req.no2, req.o3, req.so2, req.pm2_5, req.pm10)
    stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    retry = 0
    response = None
    while retry < MAX_RETRY:
        try:
            response = json.loads(clean(result['choices'][0]['text']))
            if len(response) > 1 or "advice" not in response:
                raise ValueError("Wrong format: more than 1 advice object in list")
            break
        except ValueError as e:
            print("Error: ", e)
            retry += 1
            stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
            result = copy.deepcopy(stream)
            continue

    return response

@app.post("/predict/disease")
@logging
async def predict_disease(req: Disease):
    template = """<|im_start|>system
You are a helpful chatbot and a doctor with 20 years of experience. You will be provided a paragraph describing health issues of a person. From that paragraph, extract the list of diseases that person has. 
The response should be in json format. The disease(s) should be in a list called "diseases". 
<|im_end|>
<|im_start|>user
Here is the paragraph to extract information from: {}<|im_end|>
<|im_start|>assistant""".format(req.description)
    stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    retry = 0
    response = None
    while retry < MAX_RETRY:
        try:
            response = json.loads(clean(result['choices'][0]['text']))
            if len(response) > 1 or "diseases" not in response:
                raise ValueError("Wrong format: more than 1 diseases object in list")
            break
        except ValueError as e:
            print("Error: ", e)
            retry += 1
            stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
            result = copy.deepcopy(stream)
            continue

    return response

@app.post("/personal/advice")
@logging
async def personal_advice(req: PersonalAdvice):
    template = """<|im_start|>system
CONTEXT: You are a virtual assistant designed to offer tailored advice based on Air Quality Index (AQI) metrics for individuals with specific health conditions. Users will input their current health issues along with AQI values for key pollutants. Your role is to analyze these factors and provide personalized guidance. INPUT: Users will provide a brief description of the user's current health issues or pre-existing conditions; and AQI values for key pollutants (NO2, O3, SO2, PM2.5, PM10) and the overall AQI. DESIRED OUTPUT: Your response should include the assess the impact of the current air quality on the specified health condition; offer specific advice based on the user's health issues. This may include recommendations for activities, precautions, or environmental adjustments; and provide relevant facts or insights about how different pollutants can affect individuals with the specified health condition. Note that you should not use the pronoun \"I\" or \"We\" in your advice. The response should be in json format. The advices are in a list called \"advices\",where each advice is a string and be an element of that list. The json response should only have 1 element is the advices list and nothing else.
<|im_end|>
<|im_start|>user
The air quality metrics are as the following: AQI={}, NO2={}, O3={}, SO2={}, PM2.5={}, PM10={}. The health issues are: {}<|im_end|>
<|im_start|>assistant""".format(req.aqi, req.no2, req.o3, req.so2, req.pm2_5, req.pm10, req.description)
    stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    retry = 0
    response = None
    while retry < MAX_RETRY:
        try:
            response = json.loads(clean(result['choices'][0]['text']))
            if len(response) > 1 or "advices" not in response:
                raise ValueError("Wrong format: should be a list or not contain advices key")
            break
        except ValueError as e:
            print("Error: ", e)
            retry += 1
            stream = llm(template, max_tokens=500,  stop=["<|im_end|>"], stream=False)
            result = copy.deepcopy(stream)
            continue

    return response

@app.post("/predict/general")
async def predict_general(req: Prompt):
    stream = llm("""<|im_start|>system
{}  
<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant""".format(req.system_promt, req.user_promt), max_tokens=500,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    return result['choices'][0]['text']

@app.post("/generate/workout")
async def generate_workout(req: WorkoutGenerator):
    template = """<|im_start|>system
Develop a comprehensive and tailored workout plan catering to individuals with specific health issues. Prioritize safety by implementing clear instructions and guidelines. The input will adhere to the following json template:
{
  "partOfBody": "[part of the body to work on]",
  "level": "[difficulty of the workout: beginner, medium, advanced]",
  "goal": "[lose weight, gain muscle, improve stamina]",
  "typeOfWorkout": "[cardio, strength training, yoga, stretching]",
  "issues_description": "[text describing user's health issues]"
}
To ensure accuracy and safety, the output format must encompass the following json details response. Workout plans should be tailored to the user's health issues and fitness level. The workout plan should be in a list called "workoutPlan", where each element is a json object with the following format:
{
"workoutPlan": [{
  "id": "[unique identifier]",
  "name": "[exercise name]",
  "force": "[push or pull]",
  "level": "[beginner, medium, advanced]",
  "mechanic": "[compound or isolation]",
  "equipment": "[required equipment]",
  "primaryMuscles": ["[primary muscle worked]"],
  "secondaryMuscles": ["[secondary muscles worked]"],
  "instructions": [
    "[clear and detailed step-by-step instructions for the exercise]"
  ],
  "category": "[cardio, strength, yoga, stretching]"
}]
}
In crafting the workout plan, ensure that the instructions are unambiguous and provide clarity on proper form, breathing techniques, and any modifications necessary for individuals with health issues. Consider variations for different fitness levels within the chosen difficulty level.
<|im_end|>
<|im_start|>user
{
    "partOfBody": {},
    "level": {},
    "goal": {},
    "typeOfWorkout": {},
    "issues_description": {}
}
<|im_end|>
<|im_start|>assistant""".format(req.partOfBody, req.level, req.goal, req.typeOfWorkout, req.issues_description)
    stream = llm(template, max_tokens=1024,  stop=["<|im_end|>"], stream=False)
    result = copy.deepcopy(stream)
    retry = 0
    response = None
    while retry < MAX_RETRY:
        try:
            response = json.loads(clean(result['choices'][0]['text']))
            if len(response) > 1 or "workoutPlan" not in response:
                raise ValueError("Wrong format: more than 1 workoutPlan object in list")
            break
        except ValueError as e:
            print("Error: ", e)
            retry += 1
            stream = llm(template, max_tokens=1024,  stop=["<|im_end|>"], stream=False)
            result = copy.deepcopy(stream)
            continue

    return response