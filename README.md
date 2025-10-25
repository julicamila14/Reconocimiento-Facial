Reconocimiento Facial con OpenCV y Gradio

Este proyecto es un prototipo de control de asistencia para una PYME alimenticia.
Utiliza OpenCV para la detección y validación de rostros mediante el algoritmo LBPH (Local Binary Patterns Histograms).

🚀 ¿Cómo funciona?

1 La cámara captura el rostro del empleado.

2 Con el modelo Haarcascade se detecta el área de la cara.

3 Si existe un modelo entrenado con LBPH, se compara y se valida la identidad del empleado.

4 Una vez validado, el sistema muestra en la interfaz el nombre del empleado y en consola se imprime el resultado.

5 La sesión de captura se detiene automáticamente después de la validación.

Estructura del Proyecto

├── src/
│   ├── app_gradio.py               # Interfaz principal en Gradio
│   └── enroll_lbph.py              # funciones de entrenamiento
├── data/
│   ├── empleados/
│   │   └── ID_name_lastname       # Carpeta de fotos del empleado
│   │       ├── img1.png            # Foto del empleado
│   │       └── img2.png            # Foto del empleado
│   └── modelos/
│       ├── lbph.yml                # Modelo entrenado
│       └── label_map.npy           # Diccionario de etiquetas
└── README.md

⚙️ Requisitos:

    - Python 3.9 o superior
    - pip install opencv-contrib-python numpy

🧪 Entrenamiento del modelo (LBPH)

    1 Agrega las fotos por empleado (frontal, bien iluminadas) en:

        data/empleados/
        ├─ ID_NAME_LASTNAME/
        │ ├─ img1.jpg

    2 Entrená el modelo:

        python -m src.utils.enroll_lbph
        
        se generaran los archivos data/modelos/lbph.yml y data/modelos/label_map.npy


▶️ Cómo levantar la aplicación

    1 Desde la raíz de Deteccion de rostro, ejecutá:

        $  uvicorn src.main:app --reload
        Por defecto, la aplicación se abre en local: http://127.0.0.1:8000/app

    2 Levantar en Ngrok

        $ ngrok http 8000
        se proporciona un URL de ngrok ej: https://c034727d6c74.ngrok-free.app/app
    

Como inicar el reconocimiento facial en el navegador:

Pasos reconocimiento: 

    1 Brindar permisos de camara

    2 Comenzar reconocimiento -> click en "Iniciar reconocimiento facial"

Verificacion de registros:

    Para ver los registro del dia -> click en boton: "Ver registros del dia"

--------------------------------------------------------------------------------------