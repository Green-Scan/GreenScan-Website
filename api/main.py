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

MODEL = tf.keras.models.load_model("/Users/aviral/miniforge3/Green/model/1")

data_info = pd.read_csv("/Users/aviral/miniforge3/Green/GreenScan-Website/api/disease_info.csv",encoding='cp1252')

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
    image_batch = np.expand_dims(img, axis=0)
    prediction = MODEL.predict(image_batch)
    index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    return {
        'class' : str(data_info['disease_name'].values[index]),
        'confidence' : float(confidence),
        'description' : str(data_info['description'].values[index]),
        'poss_steps' : str(data_info['possible_steps'].values[index]),
        'image_url' : str(data_info['image_url'].values[index]),
        'supplement_name' : str(data_info['supplement_name'].values[index]),
        'supplement_img' : str(data_info['supplement_image'].values[index]),
        'buy_link' : str(data_info['buy_link'].values[index])
}

# return {
#         'class' : str(data_info.loc[[index], 'disease_name']),
#         'confidence' : float(confidence),
#         'description' : str(data_info.loc[[index], 'description']),
#         'poss-steps' : str(data_info.loc[[index], 'possible_steps']),
#         'image_url' : str(data_info.loc[[index], 'image_url']),
#         'supplement_name' : str(data_info.loc[[index], 'supplement_name']),
#         'supplement_img' : str(data_info.loc[[index], 'supplement_image']),
#         'buy_link' : str(data_info.iloc[[index],'buy_link'])
#     }

# THE BELOW CODE IS JUST FOR TESTING. UNCOMMENT THE ABOVE CODE FOR ACTUAL PREDICTION AND RESULTS.

    # return {
    #     'class' : "Apple : Scab",
    #     'confidence' : float(0.99879),
    #     'description' : "Apple scab is the most common disease of apple and crabapple trees in Minnesota. Scab is caused by a fungus that infects both leaves and fruit. Scabby fruit are often unfit for eating. Infected leaves have olive green to brown spots. Leaves with many leaf spots turn yellow and fall off early. Leaf loss weakens the tree when it occurs many years in a row. Planting disease resistant varieties is the best way to manage scab.",
    #     'poss_steps' : "Choose resistant varieties when possible.\n Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring. \nWater in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur. \nSpread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores. \nFor best control, spray liquid copper soap early, two weeks before symptoms normally appear. \nAlternatively, begin applications when disease first appears, and repeat at 7 to 10 day intervals up to blossom drop.",
    #     'image_url' : 'https://extension.umn.edu/sites/extension.umn.edu/files/apple-scab-1.jpg',
    #     'supplement_name' : 'Katyayani Prozol Propiconazole 25% EC Systematic Fungicide',
    #     'supplement_img' : 'https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcRfq9MLrPL9tFkuFbGb98fMGDdl67v4I2iDLYCVprdsdGaXURCl9UNEr8v_65X1hKrYF5NjSvB01HOGexg-3CJxjkVSu9zPNJ2AunP09vPa0gjEILskTILx&usqp=CAE',
    #     'buy_link' : 'https://agribegri.com/products/buy-propiconazole--25-ec-systematic-fungicide-online-.php'
    # }
    

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8080)