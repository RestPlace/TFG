# TFG-
Estudio e implementación de frameworks que soporten el uso de librerías de NVIDIA de una aplicación orientada a trabajar con imágenes sanitarias


# Clasificación de Imágenes con Modelo de MONAI (DenseNet201 Preentrenado)

Este proyecto está dedicado a la investigación y entrenamiento de un modelo básico de MONAI, usando DenseNet201 con pesos preentrenados.

## Tabla de Contenidos
- [Introducción](#introducción)
- [Instalación](#instalación)
- [Uso](#uso)
- [Características](#características)
- [Dependencias](#dependencias)
- [Configuración](#configuración)
- [Documentación](#documentación)
- [Ejemplos](#ejemplos)
- [Solución de Problemas](#solución-de-problemas)
- [Contribuidores](#contribuidores)
- [Licencia](#licencia)

## Introducción
Este cuaderno está dedicado a la investigación y entrenamiento de un modelo básico de MONAI, usando DenseNet201 con pesos preentrenados. Tanto el modelo como las transformaciones han cambiado respecto al resto de modelos. La única similitud se mantiene en la carga de datos y la representación de imágenes, que se basan en código de Python normal.

## Instalación
Para instalar todas las dependencias necesarias, siga los pasos a continuación.

### Requisitos Previos
- CUDA
- Python 3.8+

### Pasos de Instalación
1. **Clonar el repositorio**:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2. **Crear y activar un entorno virtual**:
    ```bash
    python -m venv env
    source env/bin/activate  # En Windows, use `env\Scriptsctivate`
    ```

3. **Instalar las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

### Dependencias Principales
A continuación se listan las principales bibliotecas y sus versiones necesarias para este proyecto:

- `monai==1.3.1rc2`
- `numpy==1.26.4`
- `torch==2.2.2+cu121`
- `torchvision==0.17.2+cu121`
- `scikit-image==0.22.0`
- `scipy==1.11.4`
- `pandas==2.1.4`
- `tqdm==4.65.0`
- `pillow==10.2.0`
- `tensorboard==2.16.2`
- `pynrrd==1.0.0`
- `clearml==1.15.1`
- `gdown==4.7.3`
- `lmdb==1.4.1`
- `psutil==5.9.0`
- `einops==0.7.0`
- `mlflow==2.11.3`

Para instalar las dependencias opcionales, visite: [Documentación de MONAI](https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies).

## Uso
Para entrenar y evaluar el modelo, puede utilizar las celdas proporcionadas en el notebook `MODELO_MONAI_PREENTRENADO.ipynb`. Asegúrese de configurar las rutas correctas a los datos de entrenamiento y prueba.

### Ejecución del Notebook
1. Abra el notebook:
    ```bash
    jupyter notebook MODELO_MONAI_PREENTRENADO.ipynb
    ```

2. Ejecute las celdas en orden para realizar el entrenamiento y la evaluación del modelo.

## Características
- Modelo DenseNet201 preentrenado.
- Transformaciones personalizadas para el preprocesamiento de datos.
- Integración con CUDA para aceleración con GPU.

## Dependencias
A continuación se listan todas las dependencias utilizadas en este proyecto:
```python
# Librerías de Python
import os
import tifffile
import tempfile
import re
import matplotlib.pyplot as plt
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from tqdm import tqdm
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from typing import Sequence
# Librerías de MONAI
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
    RandRotate,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    RandShiftIntensity,
    NormalizeIntensity,
    ToTensor
)
from monai.utils import set_determinism
```

## Configuración
Para configurar las rutas de los datos de entrenamiento y prueba, modifique las siguientes variables en el notebook:
```python
root_dir = "/ruta/a/tus/datos"
data_dir_train = os.path.join(root_dir, "entrenamiento")
data_dir_test = os.path.join(root_dir, "test")
```

## Documentación
La documentación detallada del proyecto se encuentra en el notebook `MODELO_MONAI_PREENTRENADO.ipynb`.

## Ejemplos
A continuación se muestra un ejemplo de cómo se configura y ejecuta el entrenamiento del modelo:
```python
# Configuración de los datos
root_dir = "/home/jose/TFG/Data/Celulas"
data_dir_train = os.path.join(root_dir, "entrenamiento")
data_dir_test = os.path.join(root_dir, "test")

# Verifica si el directorio de datos existe
if not os.path.exists(data_dir_train):
    raise FileNotFoundError(f"El directorio de datos de entrenamiento {data_dir_train} no existe.")
if not os.path.exists(data_dir_test):
    raise FileNotFoundError(f"El directorio de datos de prueba {data_dir_test} no existe.")
print(data_dir_train)
print(data_dir_test)
```

## Solución de Problemas
Si encuentra algún problema, verifique que todas las dependencias estén correctamente instaladas y que las rutas a los datos sean correctas. Puede consultar la documentación de MONAI para más detalles sobre la instalación y configuración.

## Contribuidores
- [Tu Nombre](https://github.com/tu_usuario)

