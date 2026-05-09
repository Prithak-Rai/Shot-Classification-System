# Import All the Required Libraries
from ultralytics import YOLO

#Load the Yolov12 Model
model = YOLO("yolo12n.pt")

# Predictions
#results = model.predict(source="input/a.mp4", save=True)
results = model.track(source="input/a.mp4", save=True, persist=True)