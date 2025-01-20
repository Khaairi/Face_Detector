# Face Detection Model Comparison
This repository focuses on the comparison of various face detection models, including Haarcascade, MTCNN, YuNet, and RetinaFace, using the [WIDER Face Dataset](https://yjxiong.me/event_recog/WIDER/) for evaluation. The goal is to analyze the performance of these models based on metrics such as mAP, speed, and robustness across different scenarios and image complexities.
## Features
### **Model Implementation**
- **Haarcascade**: A classical approach using cascade classifiers.
- **MTCNN (Multi-Task Cascaded Convolutional Networks)**: A deep learning-based method for face detection and alignment.
- **YuNet**: A lightweight face detection model designed for efficient real-time performance.
- **RetinaFace**: A state-of-the-art deep learning model with superior accuracy and alignment capabilities.
### **Dataset**
- **WIDER Face Dataset**: A widely used benchmark for face detection containing diverse and challenging images. For this evaluation, only WIDER evaluation dataset been used that consists of 3222 image data separated into 61 categories
### **Evaluation Metrics**
- PASCAL VOC mAP.
- COCO mAP.
- Confusion Metric (True Positive, True Negative, etc).
- Inference time (model speed).
## Results
| **Model**    | **Inference time per image (second)** | **True Positive (TP)** | **False Positive (FP)** | **False Negative (FN)** | **PASCAL VOC mAP** | **COCO mAP [.50:.05:.95]** |
|--------------|---------------------------------------|------------------------|-------------------------|-------------------------|--------------------|----------------------------|
| MTCNN        | 0.2785                                | 15971                  | 1785                    | 23733                   | 0.443              | 0.226                      |
| Haar Cascade | 0.112                                 | 5114                   | 2578                    | 34590                   | 0.123              | 0.031                      |
| YuNet        | 0.084                                 | 13458                  | 2691                    | 26246                   | 0.313              | 0.148                      |
| Retina Face  | 1.176                                 | 22202                  | 533                     | 17502                   | 0.543              | 0.342                      |
