import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os
import sys
import json
from datetime import datetime 
import GUI

# Definir a codificação padrão para utf-8 pra evitar b.o na hora de salvar 
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Carrega e pre-processa as imagens
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Definindo o caminho base para a pasta DataSet
base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataSet')

# Caminhos para os subdiretórios de treinamento e validação
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Criando o ImageDataGenerator
datagen = ImageDataGenerator(validation_split=0.2)

# Configurando o gerador de dados para treinamento
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Configurando o gerador de dados para validação
validation_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

# Constroi o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(512, (3,3), activation='relu'),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(7, activation='softmax')  # Seta a quantidade de classes 
])

# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Definindo o caminho base relativo ao arquivo atual
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Verificando se as pastas Models/json e Models/models h5 existem, senão cria
json_dir = os.path.join(base_path, 'Models', 'json')
models_h5_dir = os.path.join(base_path, 'Models', 'models_h5')

if not os.path.exists(json_dir):
    os.makedirs(json_dir)
if not os.path.exists(models_h5_dir):
    os.makedirs(models_h5_dir)

# pega data e hora pra poder adicionar no nome dos arquivos para salvar
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Caminho para salvar o modelo
model_save_path = os.path.join(models_h5_dir, f'history_model_{current_time}.h5')

# Caminho para salvar o histórico de treinamento
history_save_path = os.path.join(json_dir, f'historico_de_treinamento_{current_time}.json')

# Salvando o modelo
model.save(model_save_path)
print(f'Modelo salvo em: {model_save_path}')

# Salvando o histórico de treinamento
with open(history_save_path, 'w') as histfile:
    json.dump(history.history, histfile)
print(f'Histórico de treinamento salvo em: {history_save_path}')

GUI.main()

print('Aperte qualquer tecla para fechar.')
input()