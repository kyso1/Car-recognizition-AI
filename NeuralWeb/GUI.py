import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Carros")
        self.root.geometry(self.get_window_size())  # Ajusta o tamanho da janela com base na tela
        self.current_frame = None
        self.model = None  # Inicializa o atributo do modelo como None

        # Caminhos
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.base_dir, 'Models', 'models_h5')
        self.dataset_dir = os.path.join(self.base_dir, 'DataSet', 'test')
        
        self.root.iconbitmap(os.path.join(self.base_dir, 'NeuralWeb', 'f1-logo.ico'))

        # Inicializar tela principal
        self.create_main_frame()

    def get_window_size(self):
        # Obtém o tamanho da tela e define a janela para 80% do tamanho da tela
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        return f"{window_width}x{window_height}"

    def create_main_frame(self):
        self.clear_frame()

        # Criar um frame principal com layout de grid
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)

        # Adicionar uma imagem de logotipo
        self.logo_img = Image.open(os.path.join(self.base_dir, 'NeuralWeb', 'heroIMG.png'))
        self.logo_img = self.logo_img.resize((500, 500))  # Aumenta o tamanho do logo
        self.logo_tk = ImageTk.PhotoImage(self.logo_img)
        self.logo_label = tk.Label(self.main_frame, image=self.logo_tk)
        self.logo_label.pack(pady=20)  # Centraliza horizontalmente

        # Frame para os botões e dropdown
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(pady=20)  # Centraliza verticalmente

        # Dropdown para selecionar o modelo
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self.control_frame, textvariable=self.model_var, state="readonly", width=40)
        self.model_dropdown.pack(pady=10)
        self.populate_model_dropdown()

        # Ação para carregar o modelo automaticamente ao selecionar no dropdown
        self.model_dropdown.bind("<<ComboboxSelected>>", self.load_selected_model)

        # Botões de funcionalidades
        self.btn_confusion_matrix = self.create_styled_button(self.control_frame, "Matriz de Confusão", self.show_confusion_matrix)
        self.btn_confusion_matrix.pack(pady=10)

        self.btn_roc_curve = self.create_styled_button(self.control_frame, "Curva ROC", self.show_roc_curve)
        self.btn_roc_curve.pack(pady=10)

        self.btn_image_classification = self.create_styled_button(self.control_frame, "Classificar Imagem", self.show_image_classification)
        self.btn_image_classification.pack(pady=10)

    def create_styled_button(self, parent, text, command):
        button = tk.Button(parent, text=text, command=command, width=30, height=2, font=('Arial', 14),
                           bg='#007bff', fg='white', relief='flat', bd=0)
        button.bind("<Enter>", lambda e: button.config(bg='#0056b3'))  # Hover effect
        button.bind("<Leave>", lambda e: button.config(bg='#007bff'))  # Reset to original color
        button.bind("<Button-1>", lambda e: button.config(bg='#45d41e'))  # Click effect
        return button

    def populate_model_dropdown(self):
        models = [f for f in os.listdir(self.model_dir) if f.endswith('.h5')]
        if models:
            self.model_dropdown['values'] = models
            self.model_dropdown.current(0)
            # Carregar o primeiro modelo por padrão se ainda não estiver carregado
            if not self.model:
                self.load_selected_model()
        else:
            messagebox.showwarning("Aviso", "Nenhum modelo encontrado na pasta de modelos.")

    def load_selected_model(self, event=None):
        model_name = self.model_var.get()
        if model_name:
            model_path = os.path.join(self.model_dir, model_name)
            
            # Descarregar o modelo atual se houver um carregado
            if self.model:
                del self.model
                self.model = None
            
            # Carregar o novo modelo
            self.model = load_model(model_path)
            messagebox.showinfo("Modelo Carregado", f"Modelo {model_name} carregado com sucesso!")
        else:
            messagebox.showwarning("Erro", "Nenhum modelo selecionado.")

    def show_confusion_matrix(self):
        self.clear_frame()
        self.confusion_frame = tk.Frame(self.root)
        self.confusion_frame.pack(fill='both', expand=True)

        tk.Label(self.confusion_frame, text="Matriz de Confusão", font=('Arial', 18)).pack(pady=10)
        tk.Button(self.confusion_frame, text="Voltar", command=self.create_main_frame, font=('Arial', 14)).pack(pady=10)

        if not self.model:
            messagebox.showwarning("Erro", "Nenhum modelo carregado.")
            return

        self.plot_confusion_matrix(self.confusion_frame)

    def plot_confusion_matrix(self, parent):
        validation_generator = self.get_validation_generator()
        Y_pred = self.model.predict(validation_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = validation_generator.classes

        cm = confusion_matrix(y_true, y_pred)
        class_names = list(validation_generator.class_indices.keys())

        fig, ax = plt.subplots(figsize=(12, 10))  # Ajuste o tamanho do gráfico
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)

        # Ajustar rotação das etiquetas dos eixos
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')
        ax.set_title('Matriz de Confusão')

        # Configura a rotação dos rótulos do eixo y para horizontal
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Configura a rotação dos rótulos do eixo x (se precisar de ajuste)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Ajuste conforme necessário

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)

    def show_roc_curve(self):
        self.clear_frame()
        self.roc_frame = tk.Frame(self.root)
        self.roc_frame.pack(fill='both', expand=True)

        tk.Label(self.roc_frame, text="Curva ROC", font=('Arial', 18)).pack(pady=10)
        tk.Button(self.roc_frame, text="Voltar", command=self.create_main_frame, font=('Arial', 14)).pack(pady=10)

        if not self.model:
            messagebox.showwarning("Erro", "Nenhum modelo carregado.")
            return

        self.plot_roc_curve(self.roc_frame)

    def plot_roc_curve(self, parent):
        validation_generator = self.get_validation_generator()
        Y_pred = self.model.predict(validation_generator)
        y_true = validation_generator.classes
        class_names = list(validation_generator.class_indices.keys())
        n_classes = len(class_names)

        # Binarizar as classes verdadeiras para o cálculo do ROC
        y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], Y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots(figsize=(12, 10))  # Ajuste o tamanho do gráfico
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'Classe {class_names[i]} (área = {roc_auc[i]:.2f})')

        ax.plot([0, 1], [0, 1], 'k--')  # Linha diagonal para referência
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title('Curvas ROC Multiclasse')
        ax.legend(loc="lower right")

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_image_classification(self):
        self.clear_frame()
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill='both', expand=True)

        tk.Label(self.image_frame, text="Classificação de Imagem", font=('Arial', 18)).pack(pady=10)
        tk.Button(self.image_frame, text="Voltar", command=self.create_main_frame, font=('Arial', 14)).pack(pady=10)

        self.btn_load_image = tk.Button(self.image_frame, text="Carregar Imagem", command=self.load_image, font=('Arial', 14))
        self.btn_load_image.pack(pady=10)

        self.panel = tk.Label(self.image_frame)
        self.panel.pack()

        self.result_label = tk.Label(self.image_frame, text="", font=('Arial', 16))
        self.result_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.image_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.info_label = tk.Label(self.image_frame, text="", font=('Arial', 12))
        self.info_label.pack(pady=10)

        self.details_frame = tk.Frame(self.image_frame)
        self.details_frame.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.resize((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            self.panel.configure(image=img_tk)
            self.panel.image = img_tk

            self.progress_bar.start()
            self.info_label.config(text="Classificando...")

            result, confidence, details = self.classify_image(file_path)
            self.progress_bar.stop()
            self.info_label.config(text="Classificação concluída.")

            if result:
                self.result_label.config(text=f"Este carro é um {result}. Confiança: {confidence:.2f}%", bg='#d4edda')
                self.show_classification_details(details)
            else:
                self.result_label.config(text="Não foi possível classificar a imagem.", bg='#f8d7da')

    def classify_image(self, img_path):
        if not self.model:
            messagebox.showwarning("Erro", "Nenhum modelo carregado.")
            return None, 0.0, {}

        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = self.model.predict(img_array)
        class_names = list(self.get_validation_generator().class_indices.keys())
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = np.max(prediction) * 100
        details = dict(zip(class_names, prediction[0]))
        return predicted_class, confidence, details

    def show_classification_details(self, details):            
        # Exibir gráfico de barras para probabilidades
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(details.keys())
        probabilities = list(details.values())
        ax.barh(classes, probabilities, color='skyblue')
        ax.set_xlabel('Probabilidade')
        ax.set_title('Probabilidades por Classe')

        canvas = FigureCanvasTkAgg(fig, master=self.details_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_model_architecture(self):
        self.clear_frame()
        self.arch_frame = tk.Frame(self.root)
        self.arch_frame.pack(fill='both', expand=True)

        tk.Label(self.arch_frame, text="Arquitetura da Rede Neural", font=('Arial', 18)).pack(pady=10)
        tk.Button(self.arch_frame, text="Voltar", command=self.create_main_frame, font=('Arial', 14)).pack(pady=10)

        if not self.model:
            messagebox.showwarning("Erro", "Nenhum modelo carregado.")
            return

        # Salvar a arquitetura como uma imagem
        model_plot_path = os.path.join(self.base_dir, 'NeuralWeb', 'model_architecture.png')
        plot_model(self.model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)

        # Exibir a imagem da arquitetura
        arch_img = Image.open(model_plot_path)
        arch_img = arch_img.resize((800, 600))  # Ajustar o tamanho da imagem conforme necessário
        arch_tk = ImageTk.PhotoImage(arch_img)
        arch_label = tk.Label(self.arch_frame, image=arch_tk)
        arch_label.image = arch_tk  # Manter uma referência para evitar a coleta de lixo
        arch_label.pack()

    def show_layer_visualization(self):
        self.clear_frame()
        self.layer_frame = tk.Frame(self.root)
        self.layer_frame.pack(fill='both', expand=True)

        tk.Label(self.layer_frame, text="Visualização das Camadas", font=('Arial', 18)).pack(pady=10)
        tk.Button(self.layer_frame, text="Voltar", command=self.create_main_frame, font=('Arial', 14)).pack(pady=10)

        if not self.model:
            messagebox.showwarning("Erro", "Nenhum modelo carregado.")
            return

        layer_names = [layer.name for layer in self.model.layers]
        self.layer_var = tk.StringVar()
        self.layer_dropdown = ttk.Combobox(self.layer_frame, textvariable=self.layer_var, values=layer_names, state="readonly", width=40)
        self.layer_dropdown.pack(pady=10)
        self.layer_dropdown.bind("<<ComboboxSelected>>", self.load_layer_visualization)

        self.layer_panel = tk.Label(self.layer_frame)
        self.layer_panel.pack()

    def load_layer_visualization(self, event=None):
        layer_name = self.layer_var.get()
        if layer_name:
            layer = self.model.get_layer(name=layer_name)
            if 'conv' in layer.name:
                # Exemplo de visualização de filtros
                self.visualize_conv_layer(layer)
            else:
                messagebox.showinfo("Info", "Visualização para esta camada não suportada.")
        else:
            messagebox.showwarning("Erro", "Nenhuma camada selecionada.")

    def visualize_conv_layer(self, layer):
        filters, biases = layer.get_weights()
        n_filters = filters.shape[-1]
        fig, axes = plt.subplots(1, n_filters, figsize=(20, 20))
        for i in range(n_filters):
            ax = axes[i]
            ax.imshow(filters[:, :, :, i], cmap='viridis')
            ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.layer_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def get_validation_generator(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
        validation_generator = datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        return validation_generator

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
