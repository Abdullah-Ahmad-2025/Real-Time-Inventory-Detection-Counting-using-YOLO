🛒 Real-Time Inventory Detection & Counting using YOLO

A real-time AI-powered inventory detection and counting system built using YOLO Object Detection, designed to simulate a retail checkout or smart inventory scenario.

This project detects color-tagged items, counts them accurately as they pass through a defined scan zone, and avoids duplicate counts using temporal logic.

🚀 Project Overview

The goal of this project is to build an intelligent alternative to barcode scanning using computer vision.

Instead of barcodes:

Each product has a distinct visual tag

A YOLO model detects and classifies items

Items are counted only once when passing through a scan region

The system is robust to lighting changes, tag shape variations, and camera movement.

✨ Key Features

🎯 Custom YOLO object detection model

📦 Detects Red, Blue, and Green tagged products

🧠 Smart counting using scan-zone + frame-based presence logic

🔁 Prevents duplicate counting

📊 Displays real-time bounding boxes and labels

⚡ High accuracy with smooth inference (cloud-based)

🧠 Tech Stack

YOLO (Object Detection)

Roboflow (Dataset labeling, augmentation & training)

Python

OpenCV

NumPy

📊 Model Performance

Trained on ~279 custom-labeled images

Metric	Value
Precision	~98.2%
Recall	100%
mAP@50	~99%
🖼️ Dataset & Training

Images captured in real-world lighting conditions

Each object manually labeled with bounding boxes

Dataset split into train / validation sets

Training and evaluation performed via Roboflow

⚠️ Note: Dataset download is restricted on the free Roboflow plan.
The demo uses Roboflow cloud inference.

🎥 Demo

📌 Demo Video:
(Attach your Roboflow demo video / screen recording here)

The demo shows:

Real-time detection

Bounding boxes with class labels

Accurate counting inside the scan zone

🧮 Counting Logic (Conceptual)

Define a horizontal scan zone

Detect objects using YOLO

Track object center points

Increment count only when:

Object enters the scan zone

It was not present in the previous frame

This prevents duplicate counting even when objects remain visible across multiple frames.

⚠️ Deployment Note (Important)

Cloud inference (Roboflow) runs smoothly due to GPU acceleration

Local real-time inference may be slower on CPU-only systems

Future versions will use:

Local YOLO weights

ONNX / TensorRT optimization

Object tracking (SORT / DeepSORT)

📈 Future Improvements

✅ Local YOLO inference (Ultralytics)

🧠 Object tracking instead of frame-based logic

📉 FPS benchmarking and optimization

📦 Expand to more product classes

🏷️ Shape-based tag validation (circle / triangle / parallelogram)

🧪 Learning Outcomes

This project helped me understand:

Difference between classical CV vs deep learning

Importance of dataset quality

Object detection evaluation metrics

Real-world deployment challenges (FPS, latency, hardware limits)

👨‍💻 Author

Abdullah Ahmad Sethi
BS AI Student
Passionate about Computer Vision, Machine Learning & Applied AI

🔗 LinkedIn: www.linkedin.com/in/abdullahh-ahmadd

⭐ Acknowledgements

Roboflow for dataset management and training

YOLO community for open-source research

OpenCV for visualization and processing

⭐ If you found this project interesting, consider starring the repository!
