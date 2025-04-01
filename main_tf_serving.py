from http.client import responses

import numpy as np
from fastapi import FastAPI,File,UploadFile
import uvicorn# Runs the FastAPI app as a server.
from io import BytesIO#BytesIO Converts uploaded file data into a format that PIL (Python Imaging Library) can read.
from PIL import Image#Opens image files and processes them.
import tensorflow as tf
from tensorflow import keras
import requests

app= FastAPI()#instance

endpoint = "http://localhost:8501/v1/models/potataoes_model:predict"#dynamically uses latest version ;; local host 8502 for tf serving
Class_names = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")#endpoint
async def ping():#to call this function to know whether server is alive or not
    return "Hello, I am alive"#A simple GET request to check if the server is running
def read_file_as_image(data) -> np.ndarray:  #takes data as input and return np.ndarray as output
    try:
     image = np.array(Image.open(BytesIO(data)))
     return image
    except Exception as e:
      raise ValueError(f"Error reading image file: {e}")
@app.post("/predict")
async def predict(   #it will deal with the file of ur potato plant leaf send by ur mobile application

    file: UploadFile = File(...) #uploadfile is the datatype and ryt side is the default value

):
    image = read_file_as_image(await file.read()) #now read the file and convert into numpy array which our model can understand
# async def and await in FastAPI to handle file uploads without blocking the server. This allows multiple users to upload files simultaneously without slowing things down. await file.read() ensures
    # the file is read asynchronously, so the server can keep handling other requests instead of waiting.
    img_batch=np.expand_dims(image, 0)
    json_data = {
        "instances" : img_batch.tolist()
    }
    responses = requests.post(endpoint,json=json_data)
    prediction = np.array(responses.json()["predictions"][0])
    predicted_class = Class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)