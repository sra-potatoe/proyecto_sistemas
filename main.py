import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara. Verifica que esté conectada y funcionando.")
else:
    print("La cámara se ha abierto correctamente.")

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

cap.release()
cv2.destroyAllWindows()
