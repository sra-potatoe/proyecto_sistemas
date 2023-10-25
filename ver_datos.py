import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import tkinter as tk
from PIL import Image, ImageTk
import urllib.request
import io

# Inicializa Firebase con el archivo JSON de credenciales
cred = credentials.Certificate(
    "C:/Users/Erick/Documents/proyecto_sistemas/uniface-9cd0b-firebase-adminsdk-n0htw-d99e0a7d62.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'uniface-9cd0b.appspot.com'  # Reemplaza con el nombre de tu proyecto en Firebase
})

# Inicializa el cliente de almacenamiento de Firebase
bucket = storage.bucket()


# Función para mostrar las imágenes de los rostros en una ventana
def show_faces_in_window():
    data = bucket.list_blobs(prefix="rostros/")

    # Crea una ventana principal
    root = tk.Tk()
    root.title("Rostros Almacenados")

    for blob in data:
        try:
            # Descargar la imagen desde Firebase Storage
            image_data = blob.download_as_string()

            # Verificar que sea una imagen válida
            image = Image.open(io.BytesIO(image_data))
            if image.format in ('JPEG', 'PNG', 'BMP'):
                photo = ImageTk.PhotoImage(image)

                # Crear una etiqueta con la imagen
                label = tk.Label(root, image=photo)
                label.pack()
            else:
                print(f"El archivo no es una imagen válida: {blob.name}")

        except Exception as e:
            print(f"Error al mostrar la imagen: {str(e)}")

    root.mainloop()


# Llama a la función para mostrar los rostros en una ventana
show_faces_in_window()
