import cv2
import firebase_admin
from firebase_admin import credentials
from google.cloud import storage
import face_recognition
import requests
import numpy as np
import os

imagesPath= "C:/Users/Erick/Documents/FaceRecognition/DataBase/Faces"

# Credenciales y configuración de Firebase
cred = credentials.Certificate("C:/Users/Erick/Documents/FaceRecognition/uniface-9cd0b-firebase-adminsdk-n0htw-d99e0a7d62.json")
firebase_admin.initialize_app(cred)

# Inicializa el cliente de Firebase Storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('uniface-9cd0b.appspot.com')  # Reemplaza con tu nombre de bucket

# Detector facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

database_url = "https://uniface-9cd0b-default-rtdb.firebaseio.com"
response = requests.get(database_url)

if response.status_code == 200:
    data = response.json()  # Supongo que la respuesta es un JSON con los datos de los rostros y sus identificadores

    # Descarga las imágenes desde Firebase Storage
    blobs = bucket.list_blobs(prefix='rostros/')

    for blob in blobs:

        # Obtiene el nombre del archivo
        image_name = blob.name.split('/')[-1]

        # Descarga la imagen
        image_data = blob.download_as_bytes()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)

        # Carga la imagen y realiza el reconocimiento facial
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (150, 150))

            # Realiza el reconocimiento facial
            face_encodings = face_recognition.face_encodings(face)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0].tolist()

                # Puedes almacenar los datos en Firebase Firestore o en otro lugar según tus necesidades
                print("Face Recognized:", image_name)

        # Muestra la imagen con las caras detectadas
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not os.path.exists("faces"):
            os.makedirs("faces")
            print("Nueva carpeta: faces")
        # Detector facial
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        count = 0
        for imageName in os.listdir(imagesPath):
            print(imageName)
            image = cv2.imread(imagesPath + "/" + imageName)
            faces = faceClassif.detectMultiScale(image, 1.1, 5)
            for (x, y, w, h) in faces:
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face = image[y:y + h, x:x + w]
                face = cv2.resize(face, (150, 150))
                cv2.imwrite("faces/" + str(count) + ".jpg", face)
                count += 1
                # cv2.imshow("face", face)
                # cv2.waitKey(0)
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imshow("Image", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
