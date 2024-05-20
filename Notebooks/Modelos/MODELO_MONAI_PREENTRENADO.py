#!/usr/bin/env python
# coding: utf-8

# # Clasificación de Imágenes con Modelo de MONAI (DenseNet201 Preentrenado)
# 
# Este cuaderno está dedicado a la investigación y entrenamiento de un modelo básico de MONAI, usando DenseNet201 con pesos preentrenados.
# 
# Tanto el modelo como las transformaciones han cambiado respecto al resto de modelos.
# La única similitud se mantiene en la carga de datos y la representación de imágenes, que se basan en código de python normal.

# ## Imports

# In[7]:


import os
import tifffile
import tempfile
import re
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from tqdm import tqdm
from torch.hub import load_state_dict_from_url
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from typing import Sequence
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader, PILReader
from monai.metrics import ROCAUCMetric
from monai.utils.module import look_up_option
from monai.networks.nets import DenseNet
from monai.optimizers import Novograd
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    ScaleIntensity, 
    EnsureChannelFirst, 
    ScaleIntensity, 
    RandRotate, 
    RandFlip, 
    RandZoom, 
    RandGaussianNoise, 
    RandShiftIntensity, 
    NormalizeIntensity, 
    ToTensor
)
from monai.utils import set_determinism

# Configure MONAI
print_config()


# ## Celulas de Entrenamiento y Test

# In[8]:


root_dir = "/home/jose/TFG/Data/Celulas"

# Directorio que contiene tus datos locales
data_dir_train = os.path.join(root_dir, "entrenamiento")
data_dir_test = os.path.join(root_dir, "test")
# Verifica si el directorio de datos existe
if not os.path.exists(data_dir_train):
    raise FileNotFoundError(f"El directorio de datos de entrenamiento {data_dir_train} no existe.")
if not os.path.exists(data_dir_test):
    raise FileNotFoundError(f"El directorio de datos de prueba {data_dir_test} no existe.")
print(data_dir_train)
print(data_dir_test)


# ## Semilla determinista

# In[9]:


set_determinism(seed=8)


# ## Cargar los Datafolders
# 

# In[11]:


# Obtener los nombres de las clases de entrenamiento
class_names_train = sorted(x for x in os.listdir(data_dir_train) if os.path.isdir(os.path.join(data_dir_train, x)) and x != '.ipynb_checkpoints')
num_class_train = len(class_names_train)

# Obtener la lista de rutas a las imágenes TIFF y las etiquetas de clase correspondientes
image_files_list_train = []
image_class_train = []

for i, class_name_train in enumerate(class_names_train):
    class_dir_train = os.path.join(data_dir_train, class_name_train)
    image_files_train = [os.path.join(class_dir_train, x) for x in os.listdir(class_dir_train) if x.endswith('.tiff') and not x.startswith('.')]  # Filtrar archivos TIFF y ocultos
    image_files_list_train.extend(image_files_train)
    image_class_train.extend([class_name_train] * len(image_files_train))  # Guardar el nombre de la clase en lugar del índice

# Crear un DataFrame con las rutas de las imágenes y las etiquetas de clase
df_train = pd.DataFrame({
    'image_path': image_files_list_train,
    'class_name': image_class_train
})

num_total_train = len(image_class_train)
image_width, image_height = Image.open(image_files_list_train[0]).size

# Mostrar información sobre los datos
print(f"Total de imágenes: {num_total_train}")
print(f"Dimensiones de las imágenes: {image_width} x {image_height}")
print(f"Nombres de las etiquetas: {class_names_train}")
print(f"Número de imágenes por etiqueta: {df_train['class_name'].value_counts().tolist()}")

# Gráfico de barras para mostrar el número de imágenes por etiqueta de clase
plt.figure(figsize=(10, 6))
df_train['class_name'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Número de Imágenes por Clase')
plt.xlabel('Clase')
plt.ylabel('Número de Imágenes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Mostrar una muestra de una imagen por clase
plt.figure(figsize=(10, 6))
num_classes_display = min(9, len(class_names_train))
for i, class_name in enumerate(class_names_train[:num_classes_display]):
    class_sample = df_train[df_train['class_name'] == class_name].sample(n=1).iloc[0]
    image = Image.open(class_sample['image_path'])
    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')
plt.suptitle('Ejemplo de Imágenes por Clase', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ## Celulas de prueba

# In[12]:


# Obtener nombres de las clases
class_names_test = sorted([x for x in os.listdir(data_dir_test) if os.path.isdir(os.path.join(data_dir_test, x)) and x != '.ipynb_checkpoints'])
num_class_test = len(class_names_test)

# Función para obtener rutas de imágenes y sus etiquetas
def get_image_files_and_labels(class_name):
    class_dir = os.path.join(data_dir_test, class_name)
    image_files = [os.path.join(class_dir, x) for x in os.listdir(class_dir) if x.endswith('.tiff') and not x.startswith('.')]
    return image_files, [class_name] * len(image_files)

# Usar ThreadPoolExecutor para paralelizar la lectura de archivos
image_files_list_test = []
image_class_test = []

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(get_image_files_and_labels, class_name) for class_name in class_names_test]
    for future in futures:
        image_files, labels = future.result()
        image_files_list_test.extend(image_files)
        image_class_test.extend(labels)

num_total_test = len(image_class_test)

# Leer una imagen para obtener las dimensiones
sample_image = PIL.Image.open(image_files_list_test[0])
image_width, image_height = sample_image.size

# Crear un DataFrame para manejar los datos
df_test = pd.DataFrame({
    'image_path': image_files_list_test,
    'class_name': image_class_test
})

# Mostrar información sobre los datos de prueba
print(f"Total de imágenes: {num_total_test}")
print(f"Dimensiones de las imágenes: {image_width} x {image_height}")
print(f"Nombres de las etiquetas: {class_names_test}")
print(f"Número de imágenes por etiqueta: {df_test['class_name'].value_counts().tolist()}")

# Gráfico de barras para mostrar el número de imágenes por etiqueta de clase en el conjunto de prueba
plt.figure(figsize=(10, 6))
df_test['class_name'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Número de Imágenes por Clase en el Conjunto de Prueba')
plt.xlabel('Clase')
plt.ylabel('Número de Imágenes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Mostrar una muestra de una imagen por clase en el conjunto de prueba
plt.figure(figsize=(10, 6))
num_classes_display = min(9, len(class_names_test))
for i, class_name in enumerate(class_names_test[:num_classes_display]):
    class_sample = df_test[df_test['class_name'] == class_name].sample(n=1).iloc[0]
    image = PIL.Image.open(class_sample['image_path'])
    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')
plt.suptitle('Ejemplo de Imágenes por Clase en el Conjunto de Prueba', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ## IMÁGENES DE ENTRENAMIENTO

# In[13]:


# Crear subgráficos
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# Seleccionar aleatoriamente 9 imágenes sin reemplazo
random_indices = np.random.choice(num_total_train, size=9, replace=False)

# Mostrar las imágenes y etiquetas
for i, idx in enumerate(random_indices):
    ax = axs[i // 3, i % 3]
    image_path = image_files_list_train[idx]
    image = PIL.Image.open(image_path)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    class_name = image_class_train[idx]  # Obtener el nombre de la clase directamente
    ax.set_title(f"Clase: {class_name}\nÍndice: {idx}")
    ax.axis("off")

# Ajustar diseño y mostrar
plt.tight_layout()
plt.show()

# Crear un DataFrame para la distribución de clases
class_distribution = df_train['class_name'].value_counts().reset_index()
class_distribution.columns = ['class_name', 'count']

# Mostrar gráfico de barras de la distribución de clases
plt.figure(figsize=(10, 6))
plt.bar(class_distribution['class_name'], class_distribution['count'], color='skyblue')
plt.title('Distribución de Clases en el Conjunto de Entrenamiento')
plt.xlabel('Clase')
plt.ylabel('Número de Imágenes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ## IMAGENES DE PRUEBA

# In[14]:


# Crear subgráficos
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# Seleccionar aleatoriamente 9 imágenes sin reemplazo
random_indices = np.random.choice(num_total_test, size=9, replace=False)

# Mostrar las imágenes y etiquetas
for i, idx in enumerate(random_indices):
    ax = axs[i // 3, i % 3]
    image_path = image_files_list_test[idx]
    image = PIL.Image.open(image_path)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    class_name = image_class_test[idx]  # Obtener el nombre de la clase directamente
    ax.set_title(f"Clase: {class_name}\nÍndice: {idx}")
    ax.axis("off")

# Ajustar diseño y mostrar
plt.tight_layout()
plt.show()

# Crear un DataFrame para la distribución de clases
class_distribution_test = pd.DataFrame({
    'class_name': class_names_test,
    'count': [image_class_test.count(class_name) for class_name in class_names_test]
})

# Mostrar gráfico de barras de la distribución de clases
plt.figure(figsize=(10, 6))
plt.bar(class_distribution_test['class_name'], class_distribution_test['count'], color='skyblue')
plt.title('Distribución de Clases en el Conjunto de Pruebas')
plt.xlabel('Clase')
plt.ylabel('Número de Imágenes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ## Preparar conjunto de datos de entrenamiento, validación y test
# 
# Se elige de manera aleatoria un 10% de los datos de entrenamiento para la validación.

# In[15]:


val_frac = 0.1
length_train = len(image_files_list_train)
length_test = len(image_files_list_test)
indices_train = np.arange(length_train)
indices_test = np.arange(length_test)
np.random.shuffle(indices_train)
np.random.shuffle(indices_test)
test_split = int(length_test)
val_split = int(val_frac * length_train) + test_split
test_indices = indices_test[:test_split]
val_indices = indices_train[test_split:val_split]
train_indices = indices_train[val_split:]

train_x = [image_files_list_train[i] for i in train_indices]
train_y = [image_class_train[i] for i in train_indices]
val_x = [image_files_list_train[i] for i in val_indices]
val_y = [image_class_train[i] for i in val_indices]
test_x = [image_files_list_test[i] for i in test_indices]
test_y = [image_class_test[i] for i in test_indices]

print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")


# ## Definición de las Transformaciones y Balanceo de Clases

# In[16]:


train_transforms = Compose([
    LoadImage(image_only=True, reader=tifffile.imread),
    EnsureChannelFirst(),
    ScaleIntensity(),
    RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
])

val_transforms = Compose([
    LoadImage(image_only=True, reader=tifffile.imread), 
    EnsureChannelFirst(), 
    ScaleIntensity(), 
    RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
])
# Calcular la distribución de clases en el conjunto de datos original de entrenamiento
class_counter_train = Counter(train_y)
min_class_count = min(class_counter_train.values())
print(min_class_count)

# Calcular el número máximo de imágenes aleatorias a generar para cualquier clase
max_random_images = min_class_count - 1
print(max_random_images)

# Calcular el número total de imágenes en el conjunto de datos original de entrenamiento
total_images_train = len(train_y)
print(total_images_train)

# Calcular el número de imágenes adicionales necesarias para igualar las clases
extra_images_needed = max_random_images * len(class_counter_train) - total_images_train
print(extra_images_needed)
# Ajustar el número de imágenes aleatorias por clase para igualar las clases
class_random_images = {cls: max_random_images for cls in class_counter_train.keys()}
print(class_random_images)
for cls, count in class_counter_train.items():
    if extra_images_needed > 0:
        additional_images = min(extra_images_needed, max_random_images)
        class_random_images[cls] += additional_images
        extra_images_needed -= additional_images

# Clase de Dataset con transformaciones aleatorias
class RandomTransformDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms, num_random_images):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms
        self.num_random_images = num_random_images
        self.random_transforms = Compose([
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ])

    def __len__(self):
        return len(self.image_files) * (self.num_random_images + 1)

    def __getitem__(self, index):
        original_index = index // (self.num_random_images + 1)
        filename = self.image_files[original_index]
        original_image = self.transforms(filename)  # Utiliza la ruta de archivo directamente
        if index % (self.num_random_images + 1) == 0:
            # Devolver la imagen original
            return original_image, self.labels[original_index]
        else:
            # Aplicar transformaciones aleatorias pre-generadas
            random_image = self.random_transforms(original_image)
            return random_image, self.labels[original_index]



# ## Cargar datos de Entrenamiento, Validación y Prueba

# In[17]:


# Crear el conjunto de datos de entrenamiento con las transformaciones aleatorias proporcionales
train_random_x = []
train_random_y = []
for x, y in zip(train_x, train_y):
    train_random_x.extend([x] * (class_random_images[y] + 1))
    train_random_y.extend([y] * (class_random_images[y] + 1))

# Create the dataset using the RandomTransformDataset class
train_ds = RandomTransformDataset(train_x, train_y, train_transforms, max_random_images)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

# Crear un nuevo conjunto de datos de validación con transformaciones aleatorias
val_ds = RandomTransformDataset(val_x, val_y, val_transforms, max_random_images)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)


# Use the same test dataset and loader as before
test_ds = RandomTransformDataset(test_x, test_y, val_transforms, max_random_images)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=4)

# Tamaño total del conjunto de datos de entrenamiento después de aplicar las transformaciones aleatorias
total_train_dataset_size = len(train_x) * (max_random_images + 1)
iterations_per_epoch = total_train_dataset_size // 32
print(f"Tamaño total del conjunto de datos de entrenamiento después de aplicar las transformaciones aleatorias: {total_train_dataset_size}")
print(f"Número de iteraciones por época: {iterations_per_epoch}")


# ## Entrenamiento del Modelo

# ### Importación de funciones auxiliares de métricas

# In[18]:


# Función para cargar los pesos preentrenados de un modelo
def _load_state_dict(model: nn.Module, arch: str, progress: bool):
    """
    This function is used to load pretrained models.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.

    """
    model_urls = {
        "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    }
    model_url = look_up_option(arch, model_urls, None)
    if model_url is None:
        raise ValueError(
            "only 'densenet201' is supported to load pretrained weights."
        )

    pattern = re.compile(
        r"^(.*denselayer\d+)(\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + ".layers" + res.group(2) + res.group(3)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

# Función para calcular métricas de evaluación
def compute_metric(predictions, targets):
    pred_classes = predictions.argmax(dim=1)
    correct = (pred_classes == targets).sum().item()
    total = len(targets)
    accuracy = correct / total
    
    tp = ((pred_classes == 1) & (targets == 1)).sum().item()
    tn = ((pred_classes == 0) & (targets == 0)).sum().item()
    fp = ((pred_classes == 1) & (targets == 0)).sum().item()
    fn = ((pred_classes == 0) & (targets == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity
    }

def compute_composite_metric(metrics):
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    specificity = metrics["specificity"]
    composite_metric = (0.4 * accuracy) + (0.2 * precision) + (0.2 * recall) + (0.2 * specificity)
    return composite_metric


# ### Definición del Modelo

# In[19]:


class DenseNet201(DenseNet):
    """DenseNet201 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 48, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            if spatial_dims > 2:
                raise NotImplementedError(
                    "Too many dimensions. PyTorch only provides pretrained weights for 2D images."
                )
            _load_state_dict(self, "densenet201", progress)


# ### Configuración del Entrenamiento

# In[20]:


# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciación del modelo
num_classes = 4
model = DenseNet201(
    spatial_dims=2,
    in_channels=3,
    out_channels=num_classes,
    init_features=64,
    growth_rate=32,
    block_config=(6, 12, 48, 32),
    pretrained=True
).to(device)

# Definición del optimizador, función de pérdida y programador de tasa de aprendizaje
optimizer = Novograd(model.parameters(), lr=1e-4)
loss_function = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) 

# Parámetros de entrenamiento
val_interval = 1
max_epochs = 10
patience = 3
best_metric = 0
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
early_stopping_counter = 0


# ### Ciclo de Entrenamiento

# In[22]:


# Determinar el directorio de las métricas
metric_dir = "/home/jose/TFG/Metricas"

# Crear un diccionario para mapear las etiquetas a enteros
label_to_int = {label: idx for idx, label in enumerate(class_names_train)}

# Listas para almacenar las métricas a lo largo de las épocas
epoch_loss_values = []
metric_values = []
accuracy_values = []
precision_values = []
recall_values = []
specificity_values = []

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{max_epochs}', unit='batch')
    for batch_data in train_loader:
        inputs = batch_data[0].to(device)
        labels = torch.tensor([label_to_int[label] for label in batch_data[1]]).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        pbar.update(1)
    pbar.close()
    epoch_loss /= len(train_loader)
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), torch.tensor([label_to_int[label] for label in val_data[1]]).to(device)
                outputs = model(val_images)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            metrics = compute_metric(y_pred, y)
            composite_metric = compute_composite_metric(metrics)

            # Guardar métricas individuales
            accuracy_values.append(metrics["accuracy"])
            precision_values.append(metrics["precision"])
            recall_values.append(metrics["recall"])
            specificity_values.append(metrics["specificity"])
            metric_values.append(composite_metric)

            if composite_metric >= best_metric:
                early_stopping_counter = 0
                best_metric = composite_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(metric_dir, "mejor_modelo_monai_preentrenado.pth"))
                print("Guardado nuevo mejor modelo de métrica compuesta")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Detención temprana en la época {epoch + 1} ya que la métrica de validación no ha mejorado durante {patience} épocas.")
                    break
            print(
                f"Época actual: {epoch + 1} Métricas: {metrics}"
                f" Mejor métrica compuesta: {best_metric:.4f}"
                f" en la época: {best_metric_epoch}"
            )
    scheduler.step()

print(f"Entrenamiento completado, mejor métrica compuesta: {best_metric:.4f} en la época: {best_metric_epoch}")


# ### Visualización de Resultados

# In[23]:


# Visualización de resultados
plt.figure(figsize=(18, 10))

# Gráfico de la pérdida promedio por época
plt.subplot(2, 3, 1)
plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, marker='o', linestyle='-', color='b', label='Pérdida')
plt.title("Pérdida Promedio por Época")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.grid(True)
plt.legend()
print(epoch_loss_values)
# Gráfico de la métrica compuesta por época
plt.subplot(2, 3, 2)
plt.plot(range(1, len(metric_values) + 1), metric_values, marker='o', linestyle='-', color='g', label='Métrica Compuesta')
plt.title("Métrica Compuesta por Época")
plt.xlabel("Época")
plt.ylabel("Métrica Compuesta")
plt.grid(True)
plt.legend()
print(metric_values)
# Gráfico de la precisión por época
plt.subplot(2, 3, 3)
plt.plot(range(1, len(precision_values) + 1), precision_values, marker='o', linestyle='-', color='r', label='Precisión')
plt.title("Precisión por Época")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.grid(True)
plt.legend()
print(precision_values)
     
# Gráfico del recall por época
plt.subplot(2, 3, 4)
plt.plot(range(1, len(recall_values) + 1), recall_values, marker='o', linestyle='-', color='m', label='Recall')
plt.title("Recall por Época")
plt.xlabel("Época")
plt.ylabel("Recall")
plt.grid(True)
plt.legend()
print(recall_values)

# Gráfico de la especificidad por época
plt.subplot(2, 3, 5)
plt.plot(range(1, len(specificity_values) + 1), specificity_values, marker='o', linestyle='-', color='c', label='Especificidad')
plt.title("Especificidad por Época")
plt.xlabel("Época")
plt.ylabel("Especificidad")
plt.grid(True)
plt.legend()
print(specificity_values)
plt.tight_layout(pad=3.0)
plt.show()


# ## Evaluar el modelo con los datos de prueba

# In[25]:


# Cargar el mejor modelo
model.load_state_dict(torch.load(os.path.join(metric_dir, "mejor_modelo_monai_preentrenado.pth")))
model.eval()
y_true = []
y_pred = []

# Mostrar el número total de muestras en el conjunto de datos de prueba
print(f"Total number of samples in the test dataset: {len(test_loader.dataset)}")

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = torch.tensor([label_to_int[label] for label in test_labels]).to(device)
        
        # Realizar predicciones
        outputs = model(test_images)
        pred = outputs.argmax(dim=1)
        
        # Almacenar etiquetas verdaderas y predichas
        y_true.extend(test_labels.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# Mostrar el número total de muestras evaluadas
print(f"Total number of samples evaluated: {len(y_true)}")

# Calcular y mostrar métricas de rendimiento
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names_train)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Mostrar el informe de clasificación
print(classification_report(y_true, y_pred, target_names=class_names_train, digits=4, zero_division=1))

