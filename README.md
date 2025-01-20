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
### **Inference Results**
| **Model**    | **Image 1** | **Image 2** | **Image 3** | 
|--------------|---------------------------------------|------------------------|-------------------------|
| Ground Truth |  ![WhatsApp Image 2025-01-20 at 7 30 58 PM](https://github.com/user-attachments/assets/a216c2c6-fd06-4bc5-b6e9-09945005a9da)|![WhatsApp Image 2025-01-20 at 7 28 06 PM](https://github.com/user-attachments/assets/d1479afb-e2e0-473e-b51c-53ff1fbe9c7f)|![WhatsApp Image 2025-01-20 at 7 28 38 PM](https://github.com/user-attachments/assets/45cf7b1c-70e6-4953-bca3-78a2ebd31714)|
| MTCNN        |![WhatsApp Image 2025-01-20 at 7 27 28 PM](https://github.com/user-attachments/assets/3e0d6e85-a33c-4112-b000-c4233be575a4)|![WhatsApp Image 2025-01-20 at 7 27 49 PM](https://github.com/user-attachments/assets/2b59bcf5-ee21-4493-af41-39ef11b790a4)|![WhatsApp Image 2025-01-20 at 7 29 51 PM](https://github.com/user-attachments/assets/d276239c-9cd5-41e2-9db8-e962caa33acc)|
| Haar Cascade |![WhatsApp Image 2025-01-20 at 7 31 23 PM](https://github.com/user-attachments/assets/8290e93e-ec1b-45f9-86ec-513797cc5278)|![WhatsApp Image 2025-01-20 at 7 31 42 PM](https://github.com/user-attachments/assets/3491c1a8-ff3d-4a5d-99e5-3f55e5032689)|![WhatsApp Image 2025-01-20 at 7 32 12 PM](https://github.com/user-attachments/assets/129d5a45-faab-440c-8afd-b04d870d41af)| 
| YuNet        |![WhatsApp Image 2025-01-20 at 7 33 03 PM](https://github.com/user-attachments/assets/9d605492-96bf-45d3-bea5-a75067d89dad)|![WhatsApp Image 2025-01-20 at 7 33 23 PM](https://github.com/user-attachments/assets/85640493-882e-4869-9662-dc2a09a53d95)|![WhatsApp Image 2025-01-20 at 7 33 56 PM](https://github.com/user-attachments/assets/13f428e9-cd8e-4deb-8f4a-476177a032dd)| 
| Retina Face  |![WhatsApp Image 2025-01-20 at 7 35 57 PM](https://github.com/user-attachments/assets/ea93dbca-29d0-4b48-bdfe-c3b398570c24)|![WhatsApp Image 2025-01-20 at 7 36 13 PM](https://github.com/user-attachments/assets/fc7f0eb1-804c-4830-ac9c-e197f731b5df)|![WhatsApp Image 2025-01-20 at 7 36 47 PM](https://github.com/user-attachments/assets/5ad88ea4-ccc7-43a1-bd1d-36f8069ea376)| 
