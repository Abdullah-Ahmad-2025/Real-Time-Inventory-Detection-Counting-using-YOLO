import cv2
import numpy as np
import time
from roboflow import Roboflow

# ROBOFLOW MODEL
rf = Roboflow(api_key="MY_API_KEY")  # ADD API KEY TO EXECUTE
project = rf.workspace("abdullah-sethi-yavji").project("item-counter-shopping-vfinal")
model = project.version(1).model

# INVENTORY DATA
counts = {
    "red_tag": 0,
    "blue-tag": 0,
    "green-tag": 0
}

prices = {
    "red_tag": 100,
    "blue-tag": 150,
    "green-tag": 200
}

total_price = 0

# To avoid duplicate counting
previous_presence = {
    "red_tag": False,
    "blue-tag": False,
    "green-tag": False
}

# VIDEO CAPTURE
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# SCAN ZONE
SCAN_Y1 = 300
SCAN_Y2 = 360

# FPS CONTROL
prev_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Running YOLO inference
    result = model.predict(frame, confidence=70, overlap=30).json()

    current_presence = {
        "red_tag": False,
        "blue-tag": False,
        "green-tag": False
    }

    #  PROCESS DETECTIONS
    for pred in result["predictions"]:

        label = pred["class"].lower()

        if label not in current_presence:
            continue

        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])

        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label.upper(), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Check if object center is inside scan zone
        if SCAN_Y1 < y < SCAN_Y2:
            current_presence[label] = True

    # COUNTING LOGIC
    for color in counts:

        if current_presence[color] and not previous_presence[color]:

            counts[color] += 1
            total_price += prices[color]

    previous_presence = current_presence.copy()

    #  DRAW SCAN ZONE
    cv2.line(frame, (0, SCAN_Y1), (640, SCAN_Y1), (0, 255, 0), 2)
    cv2.line(frame, (0, SCAN_Y2), (640, SCAN_Y2), (0, 255, 0), 2)
    cv2.putText(frame, "SCAN ZONE", (10, SCAN_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #  DISPLAY COUNTS
    cv2.putText(frame, f"RED: {counts['red_tag']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"BLUE: {counts['blue-tag']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"GREEN: {counts['green-tag']}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"TOTAL: Rs {total_price}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # FPS DISPLAY

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps}", (520, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO Inventory Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
print("FINAL COUNTS:")
print("RED : ",counts["red_tag"])
print("BLUE: ",counts["blue-tag"])
print("GREEN: ",counts["green-tag"])
print("")
print("TOTAL PRICE: Rs", total_price)

cap.release()
cv2.destroyAllWindows()
