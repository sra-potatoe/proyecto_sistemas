import cv2
import inference
import supervision as sv

# Definir la clave de API de Roboflow
api_key = "5v9Y1tbXIgSkEgpu5nmu"  # Reemplaza con tu clave de API de Roboflow

# Inicializar el annotator
annotator = sv.BoxAnnotator()

# Función para manejar predicciones
def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    detections = detections[detections.confidence > 0.9]
    print(detections)
    cv2.imshow(
        "Prediction",
        annotator.annotate(
            scene=image,
            detections=detections,
            labels=labels
        )
    )
    cv2.waitKey(1)

# Configuración del streaming e inferencia con la clave de API
inference.Stream(
    source="webcam",  # or rtsp stream or camera id
    model="unifaceidcards/1",  # from Universe
    output_channel_order="BGR",
    use_main_thread=True,  # for opencv display
    api_key=api_key,  # Pasar la clave de API directamente
    on_prediction=on_prediction,
)
