from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = FastAPI()

# Carregar o modelo
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'Models', 'models h5', 'history_model_20240716_050332.h5') # alterar pro arquivo do ultimo modelo treinado
model = load_model(model_path)

# Classes de carros
class_names = ['Audi', 'HyundaiCreta', 'MahindraScorpio', 'RollsRoyce', 'Swift', 'TataSafari', 'ToyotaInnova']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # carrega e preprocessar a imagem
    image = load_img(file.file, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    # faz a previsao 
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return {"classe: ": predicted_class, " confianca: ": float(confidence)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
