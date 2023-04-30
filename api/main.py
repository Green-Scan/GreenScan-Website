from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# MODEL = tf.keras.models.load_model("D:/my docs/Projects/GreenScan/models/2")

# data_info = pd.read_csv("disease_info.csv")

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
    # prediction = MODEL.predict(image_batch)
    # index = np.argmax(prediction[0])
    # predicted_class = data_info[index, 1]
    # class_description = data_info[index, 2]
    # poss_steps = data_info[index, 3]
    # image_url = 
    # confidence = np.max(prediction[0])
    # return {
    #     'class' : data_info[index, 1],
    #     'confidence' : float(confidence),
    #     'description' : data_info[index, 2],
    #     'poss-steps' : data_info[index, 3],
    #     'image_url' : data_info[index, 4],
    #     'supplement_name' : data_info[index, 5],
    #     'supplement_img' : data_info[index, 6],
    #     'buy_link' : data_info[index, 7]
    # }

# THE BELOW CODE IS JUST FOR TESTING. UNCOMMENT THE ABOVE CODE FOR ACTUAL PREDICTION AND RESULTS.

    return {
        'class' : "Apple : Scab",
        'confidence' : float(0.99879),
        'description' : "Apple scab is the most common disease of apple and crabapple trees in Minnesota. Scab is caused by a fungus that infects both leaves and fruit. Scabby fruit are often unfit for eating. Infected leaves have olive green to brown spots. Leaves with many leaf spots turn yellow and fall off early. Leaf loss weakens the tree when it occurs many years in a row. Planting disease resistant varieties is the best way to manage scab.",
        'poss_steps' : "Choose resistant varieties when possible.\n Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring. \nWater in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur. \nSpread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores. \nFor best control, spray liquid copper soap early, two weeks before symptoms normally appear. \nAlternatively, begin applications when disease first appears, and repeat at 7 to 10 day intervals up to blossom drop.",
        'image_url' : 'https://extension.umn.edu/sites/extension.umn.edu/files/apple-scab-1.jpg',
        'supplement_name' : 'Katyayani Prozol Propiconazole 25% EC Systematic Fungicide',
        'supplement_img' : 'https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcRfq9MLrPL9tFkuFbGb98fMGDdl67v4I2iDLYCVprdsdGaXURCl9UNEr8v_65X1hKrYF5NjSvB01HOGexg-3CJxjkVSu9zPNJ2AunP09vPa0gjEILskTILx&usqp=CAE',
        'buy_link' : 'https://agribegri.com/products/buy-propiconazole--25-ec-systematic-fungicide-online-.php'
    }
    

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)