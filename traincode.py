# Installs the ultralytics library, which includes the YOLO(You Only Look Once) object detection framework
!pip install ultralytics

# Imports the YOLO class from the ultralytics library, allowing you to use YOLO models for object detection, segmentation, or classification.
from ultralytics import YOLO

# Mounts Google Drive to the Colab environment, enabling access to files stored in Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# Lists the files and directories inside the folder to verify dataset files.
!ls /content/drive/MyDrive/datasetold

# Creates a configuration file (dataoldtrzy.yaml) that defines the dataset structure (train, validation, test paths) and class names for training a YOLO model.
dataoldtrzy_yaml = """
path: /content/drive/MyDrive/datasetold
train: images/train
val: images/valid
test: images/test
names:
  0: 'singletree'
"""
with open('dataoldtrzy.yaml', 'w') as f:
    f.write(dataoldtrzy_yaml)
print("Plik dataoldtrzy.yaml utworzony!")

#Training the model
from ultralytics import YOLO
# Load pre-trained YOLOv11 model
model = YOLO("yolo11s.pt")
# Start training with customized parameters
results = model.train(
    data="dataoldtrzy.yaml",         # Ścieżka do pliku YAML
    epochs=300,                # Liczba epok
    batch=16,                 # Zwiększony batch size (przy odpowiednim GPU)
    imgsz=640,                # Rozdzielczość obrazów
    project="runs/trainpearsolddwa",     # Gdzie zapisać wyniki treningu
    name="pearsoldtrzy",     # Nazwa treningu
    lr0=0.001,                # Początkowa wartość learning rate
    patience=0,               # Wczesne zatrzymanie, jeśli nie ma poprawy
    augment=True,             # Zaawansowana augmentacja danych
    workers=4,                 # Liczba wątków do przetwarzania danych
    device='cuda'
)

# Testing the trained model on the test set
from ultralytics import YOLO
# Load trained model from correct path
model = YOLO("runs/trainpearsolddwa/pearsoldtrzy/weights/best.pt")
# Test the model on the test set (make sure you have a test section defined in dataoldtrzy.yaml)
results = model.val(data="dataoldtrzy.yaml", split="test", device='cuda')
# View results summary (precision, recall, mAP@50 etc.)
print(results)

# Testing the trained model on a test set with label visualization on images
from ultralytics import YOLO
import cv2
import glob
from google.colab.patches import cv2_imshow
# Load trained model (modified path to current model)
model = YOLO("runs/trainpearsolddwa/pearsoldtrzy/weights/best.pt")
# Function to predict and display an image with labels
def detect_and_show(image_path):
    # Make a prediction with a set confidence threshold (conf)
    results = model.predict(source=image_path, conf=0.5)
    # Load original image
    image = cv2.imread(image_path)
    # Iterate through all detected objects and draw rectangles and labels
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0]
        cls = result.cls[0]
        # Drawing a rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Creating text with label and confidence
        text = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # View image in Colab (or save if you prefer)
    cv2_imshow(image)
# Path to folder from test set
test_folder = "/content/drive/MyDrive/datasetold/images/test"
# We download all images (for JPG and jpg extensions, you can add others if needed)
image_paths = glob.glob(test_folder + "/*.jpg")
image_paths += glob.glob(test_folder + "/*.JPG")
# We iterate over all images and perform detection
for image_path in image_paths:
    print(f"Przetwarzanie obrazu: {image_path}")
    detect_and_show(image_path)