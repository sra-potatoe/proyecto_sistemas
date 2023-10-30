import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
import tkinter as tk
import io

# Credenciales y configuración de Firebase
cred = credentials.Certificate("C:/Users/Erick/Documents/proyecto_sistemas/uniface-9cd0b-firebase-adminsdk-n0htw-d99e0a7d62.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://uniface-9cd0b-default-rtdb.firebaseio.com',
    'storageBucket': 'uniface-9cd0b.appspot.com'
})

# Referencia a la base de datos
ref = db.reference('/usuarios')

# Inicialización del cliente de almacenamiento
bucket = storage.bucket()

# Clasificador de detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicialización de la cámara
cap = cv2.VideoCapture(0)

# Función para guardar el rostro y datos en la base de datos
def guardar_rostro():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el fotograma. Saliendo...")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Captura el rostro
        (x, y, w, h) = faces[0]
        detected_face = frame[y:y + h, x:x + w]

        # Almacena la imagen del rostro en Firebase Storage
        image_bytes = cv2.imencode('.jpg', detected_face)[1].tobytes()
        buffer = io.BytesIO(image_bytes)
        blob = bucket.blob(f'rostros/rostro_{len(ref.get()) + 1}.jpg')
        blob.upload_from_file(buffer, content_type='image/jpeg')

        # Crear una ventana Tkinter para el formulario de datos
        ventana_datos = tk.Tk()
        ventana_datos.title("Ingresar datos del usuario")

        # Campos de entrada para los datos
        tk.Label(ventana_datos, text="Ingresa los datos a registrar:").pack()
        tk.Label(ventana_datos, text="Nombre:").pack()
        entrada_nombre = tk.Entry(ventana_datos)
        entrada_nombre.pack()
        tk.Label(ventana_datos, text="Apellido:").pack()
        entrada_apellido = tk.Entry(ventana_datos)
        entrada_apellido.pack()
        tk.Label(ventana_datos, text="Carrera:").pack()
        entrada_carrera = tk.Entry(ventana_datos)
        entrada_carrera.pack()
        tk.Label(ventana_datos, text="Cargo:").pack()
        entrada_cargo = tk.Entry(ventana_datos)
        entrada_cargo.pack()
        tk.Label(ventana_datos, text="Semestre:").pack()
        entrada_semestre = tk.Entry(ventana_datos)
        entrada_semestre.pack()
        tk.Label(ventana_datos, text="Área:").pack()
        entrada_area = tk.Entry(ventana_datos)
        entrada_area.pack()

        def guardar_datos():
            nombre = entrada_nombre.get()
            apellido = entrada_apellido.get()
            carrera = entrada_carrera.get()
            cargo = entrada_cargo.get()
            semestre = entrada_semestre.get()
            area = entrada_area.get()

            datos = {
                'nombre': nombre,
                'apellido': apellido,
                'carrera': carrera,
                'cargo': cargo,
                'semestre': semestre,
                'area': area,
                'imagen_rostro': blob.public_url
            }

            ref.push(datos)
            ventana_datos.destroy()

        tk.Button(ventana_datos, text="Guardar Datos", command=guardar_datos).pack()

    cv2.imshow('Reconocimiento Facial', frame)

# Crear ventana para la interfaz de usuario
ventana = tk.Tk()
ventana.title("Detección de Rostros")

# Ciclo principal para el reconocimiento facial
while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo capturar el fotograma. Saliendo...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
