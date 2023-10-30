import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
import tkinter as tk
import io
import face_recognition

# Credenciales y configuración de Firebase
cred = credentials.Certificate(
    "C:/Users/Erick/Documents/proyecto_sistemas/uniface-9cd0b-firebase-adminsdk-n0htw-d99e0a7d62.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://uniface-9cd0b-default-rtdb.firebaseio.com',
    'storageBucket': 'uniface-9cd0b.appspot.com'
})

# Referencia a la base de datos
ref = db.reference('/usuarios')

# Inicialización del cliente de almacenamiento
bucket = storage.bucket()

# Cargar las imágenes de rostros almacenadas en la base de datos
usuarios = ref.get()
rostros_base_datos = []
for key, usuario in usuarios.items():
    if 'imagen_rostro' in usuario:
        image_url = usuario['imagen_rostro']
        blob = storage.bucket().blob(image_url)
        image_data = blob.download_as_bytes()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rostros_base_datos.append(img)

# Inicialización de la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara. Verifica que esté conectada y funcionando.")
else:
    print("La cámara se ha abierto correctamente.")


# Función para realizar el reconocimiento facial
def reconocer_rostros():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el fotograma. Saliendo...")
        return

    # Convierte la imagen capturada en RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta los rostros en la imagen capturada
    rostros_detectados = face_recognition.face_locations(frame_rgb)

    for (top, right, bottom, left) in rostros_detectados:
        # Dibuja un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Codifica el rostro detectado
        rostro_encoded = face_recognition.face_encodings(frame_rgb, [(top, right, bottom, left)])[0]

        # Compara el rostro con los rostros en la base de datos
        for i, rostro_bd in enumerate(rostros_base_datos):
            rostro_bd_encoded = face_recognition.face_encodings(rostro_bd)[0]
            comparacion = face_recognition.compare_faces([rostro_bd_encoded], rostro_encoded)

            if comparacion[0]:
                # Si el rostro coincide con uno en la base de datos, muestra el nombre o ID en la imagen
                nombre_usuario = list(usuarios.keys())[i]
                cv2.putText(frame, nombre_usuario, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Reconocimiento Facial', frame)


# Bucle principal para el reconocimiento facial
while True:
    reconocer_rostros()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
