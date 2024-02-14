import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd

# get YOLOv8 model
yolo_weights_path = 'yolov8n.pt'

# apply
model = YOLO(yolo_weights_path)

# Load image
image_path = "/home/melon/Documents/Thermal_Camera_Detection/photo/bus.jpg"  # ??? ?? ??? ??????
frame = cv2.imread(image_path)

# YOLOv8 detection
results = model.predict(frame)
# print("type", type(results))
# print(results)
# print("------")

if not isinstance(results, list):
    results = [results]

# print(results[0].boxes)
    
print(len(results))
num = 1
for result in results:
    print("num: ", num)
    print(result.boxes)
    num = num + 1

'''# Results ??? ? ??? ?? ?? ?? ??
for result in results:
    # result.boxes? ?? ??? ???? Boxes ?????.
    boxes = result.boxes.xyxy[0].cpu().numpy()

    # print(len(boxes))
    # print(boxes)

    # ???? ?? ???
    for box in boxes:
        conf = box[4]
        if conf > 0.6:
            label = int(box[5])
            xmin, ymin, xmax, ymax = map(int, box[:4])

            # ?? ???
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {label}, Conf: {conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display image with detections
cv2.imshow('Object Detection', frame)

# Save the image with detections
output_image_path = 'output_image.jpg'  # ??? ??? ?? ??? ??????
cv2.imwrite(output_image_path, frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''