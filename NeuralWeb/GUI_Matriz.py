import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Carregar o modelo salvo
model_path = os.path.join('Models', 'models h5', 'history_model_20240821_165700.h5')  # Substitua pelo nome do seu modelo salvo
model = tf.keras.models.load_model(model_path)

# Classes do modelo
class_names = ['Audi', 'HyundaiCreta', 'MahindraScorpio', 'RollsRoyce', 'Swift', 'TataSafari', 'ToyotaInnova']

# Pre-processamento das imagens do conjunto de validação
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = datagen.flow_from_directory(
    'C:/Users/gian1/OneDrive/Documentos/Facul/TrabalhosIA/Car-recognizition-AI/DataSet/test',  # Alterar para o diretório correto
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Obter as previsões
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Obter as classes verdadeiras
y_true = validation_generator.classes

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)

# Plotando a Matriz de Confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# Relatório de Classificação
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
