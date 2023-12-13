import cv2
import firebase_admin
from firebase_admin import credentials, db, storage
import face_recognition
import numpy as np
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Credenciales y configuración de Firebase
cred = credentials.Certificate(
    "C:/Users/Erick/Documents/FaceRecognition/uniface-9cd0b-firebase-adminsdk-n0htw-d99e0a7d62.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://uniface-9cd0b-default-rtdb.firebaseio.com',
    'storageBucket': 'uniface-9cd0b.appspot.com'
})

# Referencia a la base de datos
db_ref = db.reference('rostros')

# Directorio con las imágenes locales
data_path = "C:/Users/Erick/Documents/FaceRecognition/DataBase/Faces"
face_images = [f for f in os.listdir(data_path) if f.endswith('.jpg')]

# Cargar las imágenes y codificaciones de rostros locales
known_face_encodings = []
known_face_names = []

for face_image in face_images:
    image_path = os.path.join(data_path, face_image)
    face = face_recognition.load_image_file(image_path)

    # Asegurarse de que hay caras en la imagen antes de intentar codificar
    face_locations = face_recognition.face_locations(face)
    if face_locations:
        encoding = face_recognition.face_encodings(face, [face_locations[0]])[0]
        name = os.path.splitext(face_image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con mediapipe
        results = face_mesh.process(image=frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    mp_face_mesh.FACE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 25))
                )
                face_locations = face_recognition.face_locations(frame_rgb)

                if face_locations:
                    # Codificar el rostro detectado
                    actual_face_encoding = face_recognition.face_encodings(frame_rgb, [face_locations[0]])[0]

                    # Comparar con las codificaciones en Firebase
                    database_faces = db_ref.get()
                    found = False
                    name = "Usuario desconocido"

                    matches = face_recognition.compare_faces(known_face_encodings, actual_face_encoding, tolerance=0.6)
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    color = (125, 220, 0) if name != "Desconocido" else (50, 50, 255)

                    cv2.rectangle(frame, (face_locations[0][3], face_locations[0][0]),
                                  (face_locations[0][1], face_locations[0][2]), color, 2)
                    cv2.putText(frame, name, (face_locations[0][3], face_locations[0][2] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

                    if database_faces:
                        for key, value in database_faces.items():
                            stored_face_encoding = value.get('encoding')
                            stored_face_encoding = [float(enc) for enc in stored_face_encoding]
                            result = face_recognition.compare_faces([stored_face_encoding], actual_face_encoding)
                            if True in result:
                                found = True
                                name = value.get('name')
                                break

                    color = (125, 220, 0) if found else (50, 50, 255)

                    cv2.rectangle(frame, (face_locations[0][3], face_locations[0][0]),
                                  (face_locations[0][1], face_locations[0][2]), color, 2)
                    cv2.putText(frame, name, (face_locations[0][3], face_locations[0][2] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar el frame
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()