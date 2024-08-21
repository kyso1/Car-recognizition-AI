import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os

# Carregar o modelo salvo
model_path = os.path.join('Models', 'models h5', 'history_model_20240821_165700.h5')  # Substitua pelo nome do seu modelo salvo
model = load_model(model_path)

# Classes do modelo
class_names = ['Audi', 'HyundaiCreta', 'MahindraScorpio', 'RollsRoyce', 'Swift', 'TataSafari', 'ToyotaInnova']

# Função para fazer a previsão
class Main():   
    def classify_image(img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        return predicted_class

    # Função para carregar a imagem e fazer a previsão
    def load_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.resize((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            panel.configure(image=img_tk)
            panel.image = img_tk

            # Fazer a previsão
            result = classify_image(file_path)
            messagebox.showinfo("Resultado", f"Este carro é um {result}.")

    # Configurar a interface gráfica
    root = tk.Tk()
    root.title("Classificador de Carros")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    panel = tk.Label(frame)
    panel.pack()

    btn = tk.Button(frame, text="Carregar Imagem", command=load_image)
    btn.pack(pady=10)

    root.mainloop()
