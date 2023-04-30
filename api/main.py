from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()


MODEL = tf.keras.models.load_model("D:/my docs/Projects/GreenScan/models/2")
CLASS_NAMES = [
    'Apple Scab Leaf',
    'Apple leaf',
    'Apple rust leaf',
    'Bell_pepper leaf',
    'Bell_pepper leaf spot',
    'Blueberry leaf',
    'Cherry leaf',
    'Corn Gray leaf spot',
    'Corn leaf blight',
    'Corn rust leaf',
    'Peach leaf',
    'Potato leaf early blight',
    'Potato leaf late blight',
    'Raspberry leaf',
    'Soyabean leaf',
    'Squash Powdery mildew leaf',
    'Strawberry leaf',
    'Tomato Early blight leaf',
    'Tomato Septoria leaf spot',
    'Tomato leaf',
    'Tomato leaf bacterial spot',
    'Tomato leaf late blight',
    'Tomato leaf mosaic virus',
    'Tomato leaf yellow virus',
    'Tomato mold leaf',
    'Tomato two spotted spider mites leaf',
    'grape leaf',
    'grape leaf black rot'
    ]

@app.get("/ping")
async def ping():
    return "Hello, I'm alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    ):
    
    img = read_file_as_image(await file.read())
    image_batch = np.expand_dims(img, 0)
    prediction = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }
    

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)