import json

from fastapi import FastAPI, UploadFile
from fastapi.params import File
import os
import consumer


app = FastAPI()
user = 'default'


@app.get("/swe/numberprediction/rest/data/v1.0/json/en/trained_model_name")
async def predictImage(filter2: bool = False):
    if filter2:
        user_history = user + '_history.txt'
        string = "["
        with open(user_history, 'r') as history:
            string += history.read()
        string = string[:-2]
        string += "]"
        return json.loads(string)
    else: # we have knn - so not implemented
        user_history = user + '_history.txt'
        string = "["
        with open(user_history, 'r') as history:
            string += history.read()
        string = string[:-2]
        string += "]"
        return json.loads(string)

@app.post("/swe/numberprediction/rest/data/v1.0/json/en/trained_model_name")
async def predict_picture(image: UploadFile = File(...)):
    location = "./test.png"
    # received picture from user 'default'
    user_history = user + '_history.txt'
    with open(location, "wb+") as file_object:
        file_object.write(image.file.read())
    return_value = consumer.predictImage(location)
    with open(user_history, 'a+') as history:
        history.write(return_value)
    return json.loads(return_value)



@app.delete("/swe/numberprediction/rest/data/v1.0/json/en/trained_model_name")
async def predictImage():
    user_history = user + '_history.txt'
    os.remove(user_history)
    return {"message": "history for user '" + user + "' deleted"}
