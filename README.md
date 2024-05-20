
# Proyecto de Modelos de Deep Learning

## Introducción
Este proyecto incluye varios cuadernos y scripts de modelos de Deep Learning utilizando diferentes frameworks: MONAI, MONAI preentrenado, PyTorch y TensorFlow. Cada uno de estos cuadernos está diseñado para tareas específicas de procesamiento y análisis de imágenes médicas. Este README proporciona una guía completa para su instalación, configuración y uso.

## Tabla de Contenidos
- [Instalación](#instalación)
- [Dependencias](#dependencias)
- [Configuración](#configuración)
- [Uso](#uso)
  - [Modelo MONAI](#modelo-monai)
  - [Modelo MONAI Preentrenado](#modelo-monai-preentrenado)
  - [Modelo PyTorch](#modelo-pytorch)
  - [Modelo TensorFlow](#modelo-tensorflow)
- [Recomendaciones](#recomendaciones)
- [Instalación de Dependencias Opcionales](#instalación-de-dependencias-opcionales)
  - [Instalación de Dependencias Opcionales de MONAI](#instalación-de-dependencias-opcionales-de-monai)
- [Scripts](#scripts)
- [Ejemplos](#ejemplos-de-ejecución)
- [Licencia](#licencia)

## Instalación
Para instalar las dependencias y configurar el entorno para ejecutar los scripts, sigue los siguientes pasos:

1. Clona este repositorio:
    ```bash
    git clone https://github.com/RestPlace/TFG
    cd TFG
    ```

2. Crea un entorno virtual:
    ```bash
    python -m venv env
    source env/bin/activate  # En Windows usa `env\Scripts\activate`
    ```

3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Dependencias
Las principales dependencias para este proyecto incluyen:
- Python 3.8 o superior
- MONAI
- PyTorch
- TensorFlow
- CUDA (recomendado para aceleración GPU)

Asegúrate de que tu sistema tenga los controladores de CUDA adecuados instalados. Puedes verificar la compatibilidad y descargar CUDA desde el [sitio oficial de NVIDIA](https://developer.nvidia.com/cuda-downloads).

### Instalación de Python
Para instalar Python, visita el [sitio oficial de Python](https://www.python.org/downloads/) y descarga la versión 3.8 o superior para tu sistema operativo. Sigue las instrucciones de instalación proporcionadas.

## Configuración
Para configurar las rutas de los datos de entrenamiento y prueba, modifique las siguientes variables en el cuaderno correspondiente:

```bash
root_dir = "/ruta/a/tus/datos"
metric_dir = "ruta/donde/estan/tus/metricas"
```

## Uso

### Modelo MONAI
Tanto el cuaderno `MODELO_MONAI_BASICO.ipynb` como `MODELO_MONAI_PREENTRENADO.ipynb`   implementan un modelo básico y uno preentrenado respectivamente, pero comparten las mismas dependencias. Asegúrate de tener las siguientes dependencias instaladas:
```bash
pip install monai
pip install torch
pip install torchvision
pip install torchaudio
pip install numpy
```

### Modelo PyTorch
El cuaderno `MODELO_PYTORCH_FINAL.ipynb` está implementado en PyTorch puro. Asegúrate de tener PyTorch y las bibliotecas necesarias instaladas:
```bash
pip install torch
pip install torchvision
pip install torchaudio
pip install numpy
```

### Modelo TensorFlow
El cuaderno `MODELO_TENSORFLOW_FINAL.ipynb` utiliza TensorFlow para la implementación del modelo. Instala TensorFlow y las bibliotecas necesarias:
```bash
pip install tensorflow
pip install numpy
```

## Recomendaciones
Es altamente recomendable instalar CUDA para acelerar el procesamiento en GPU, especialmente para modelos grandes y complejos. La instalación de CUDA puede mejorar significativamente el rendimiento de entrenamiento e inferencia.

## Instalación de Dependencias Opcionales

### Instalación de Dependencias Opcionales de MONAI
Para instalar todas las dependencias opcionales de MONAI, ejecuta el siguiente comando:
```bash
pip install monai[all]
```
Esto instalará todas las dependencias adicionales que pueden ser útiles para varios tipos de tareas y configuraciones en MONAI.

## Scripts
Este proyecto tiene las versiones convertidas a Python para poder ejecutarlas sin contar con las gráficas ni las representaciones de imágenes.

## Ejemplos de ejecución

```bash
python MODELO_MONAI_BASICO.py
python MODELO_MONAI_PREENTRENADO.py
python MODELO_PYTORCH_FINAL.py
python MODELO_TENSORFLOW_FINAL.py
```

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.
