import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from firebase_admin import firestore

import numpy as np
import tkinter as tk
import io

# Credenciales y configuración de Firebase
cred = credentials.Certificate("C:/Users/Erick/Documents/proyecto_sistemas/uniface-9cd0b-firebase-adminsdk-n0htw-d99e0a7d62.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://uniface-9cd0b-default-rtdb.firebaseio.com',
    'storageBucket': 'uniface-9cd0b.appspot.com'
})
db_firestore= firestore.client()
# Referencia a la base de datos
ref = db.reference('/usuarios')

# Inicialización del cliente de almacenamiento
bucket = storage.bucket()

# Clasificador de detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara. Verifica que esté conectada y funcionando.")
else:
    print("La cámara se ha abierto correctamente.")

#db
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

        #Firebase Storage
        image_bytes = cv2.imencode('.jpg', detected_face)[1].tobytes()
        buffer = io.BytesIO(image_bytes)
        blob = bucket.blob(f'rostros/rostro_{len(ref.get()) + 1}.jpg')
        blob.upload_from_file(buffer, content_type='image/jpeg')


        ventana_datos = tk.Tk()
        ventana_datos.title("Ingresar datos del usuario")


        tk.Label(ventana_datos, text="Ingresa los datos a registrar:").pack()
        tk.Label(ventana_datos, text="Apellido:").pack()
        entrada_apellido = tk.Entry(ventana_datos)
        entrada_apellido.pack()
        tk.Label(ventana_datos, text="Área:").pack()
        entrada_area = tk.Entry(ventana_datos)
        entrada_area.pack()
        tk.Label(ventana_datos, text="Cargo:").pack()
        entrada_cargo = tk.Entry(ventana_datos)
        entrada_cargo.pack()
        tk.Label(ventana_datos, text="Carrera:").pack()
        entrada_carrera = tk.Entry(ventana_datos)
        entrada_carrera.pack()
        tk.Label(ventana_datos, text="Nombre:").pack()
        entrada_nombre = tk.Entry(ventana_datos)
        entrada_nombre.pack()
        tk.Label(ventana_datos, text="Semestre:").pack()
        entrada_semestre = tk.Entry(ventana_datos)
        entrada_semestre.pack()

        def guardar_datos():
            apellido = entrada_apellido.get()
            area = entrada_area.get()
            cargo = entrada_cargo.get()
            carrera = entrada_carrera.get()
            nombre = entrada_nombre.get()
            semestre = entrada_semestre.get()

            datos = {
                'apellido': apellido,
                'area': area,
                'cargo': cargo,
                'carrera': carrera,
                'nombre': nombre,
                'semestre': semestre,
                'imagen_rostro': blob.public_url
            }

            # Guardar datos en Realtime Database
            ref.push(datos)

            # Guardar datos en Firestore
            db_firestore.collection('usuarios').add(datos)

            ventana_datos.destroy()

        tk.Button(ventana_datos, text="Guardar Datos", command=guardar_datos).pack()

    cv2.imshow('Reconocimiento Facial', frame)

 ##mmodificar
    cv2.imshow('Reconocimiento Facial', frame)

ventana = tk.Tk()
ventana.title("Detección de Rostros")

tk.Button(ventana, text="Iniciar Detección y Guardar", command=guardar_rostro).pack()

ventana.mainloop()

cap.release()
cv2.destroyAllWindows()
