import json
import os
from PIL import Image
import numpy as np
import argparse
from facenet_pytorch import MTCNN
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from retinaface import RetinaFace

def load_wider_face_data(dataset_path, annotation_file, sample_size=1000, seed=42):
    print('Start loading dataset...')

    # Cek apakah file anotasi ada
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found at: {annotation_file}")
        return [], []

    # Baca file anotasi JSON
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    # Buat map untuk mencocokkan id gambar dengan file_name dan anotasinya
    image_info_map = {img["id"]: img["file_name"] for img in annotations["images"]}
    annotation_map = {}
    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]  # bbox dalam format [x, y, width, height]

        if image_id not in annotation_map:
            annotation_map[image_id] = []
        annotation_map[image_id].append(bbox)

    # Inisialisasi list untuk ground_truth dan images
    images = []
    ground_truths = []

    # Ambil ID gambar dan lakukan shuffle
    image_ids = list(image_info_map.keys())
    np.random.seed(seed)
    np.random.shuffle(image_ids)

    # Batasi jumlah gambar ke sample_size
    selected_image_ids = image_ids[:sample_size]

    # Iterasi melalui setiap gambar berdasarkan `selected_image_ids`
    for image_id in selected_image_ids:
        file_name = image_info_map.get(image_id)
        if file_name is None:
            print(f"No file name found for image_id {image_id}")
            continue

        image_path = os.path.join(dataset_path, file_name)

        # Load gambar jika path-nya ada
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(np.array(image))  # Konversi ke format array

                # Dapatkan ground truth bounding boxes dari anotasi berdasarkan image_id
                gt_boxes = annotation_map.get(image_id, [])
                ground_truths.append(gt_boxes)
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")
        else:
            print(f"Image not found at path: {image_path}")

    return images, ground_truths

def display_image_with_annotations(image, bboxes):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # Tambahkan bounding boxes ke dalam gambar
    ax = plt.gca()
    for bbox in bboxes:
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis("off")
    plt.show()

def predict_faces(images, model, num_samples=10):
    predictions = []
    for i in range(num_samples):
        # Konversi gambar ke format PIL Image jika belum dalam bentuk ini
        if(args.model == 'mtcnn'):
            image = Image.fromarray(images[i]) if isinstance(images[i], np.ndarray) else images[i]

            # Dapatkan prediksi bounding boxes dari MTCNN
            boxes, _ = model.detect(image)

            # Jika tidak ada deteksi, buat kotak kosong
            if boxes is None:
                boxes = []
        elif(args.model == 'haarcascade'):
            # Convert image to grayscale
            image = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)

            # Detect faces using Haar Cascade
            faces = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Convert to format [x1, y1, x2, y2] for consistency
            boxes = [[x, y, x + w, y + h] for (x, y, w, h) in faces]

        elif(args.model == 'yunet'):
            image = images[i]

            # Convert to grayscale (YuNet expects colored image)
            image_height, image_width = image.shape[:2]
            
            # Resize model input to image dimensions
            model.setInputSize((image_width, image_height))

            # Detect faces with YuNet
            _, detected_faces = model.detect(image)

            # Process bounding boxes if detection is successful
            if detected_faces is not None:
                boxes = [
                    [x, y, x + w, y + h] for (x, y, w, h) in detected_faces[:, :4]
                ]
            else:
                boxes = []
        elif(args.model == 'retinaface'):
            # Convert the image from RGB to BGR (RetinaFace often expects BGR images)
            bgr_image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
            detections = RetinaFace.detect_faces(bgr_image)
            boxes = []
            if detections is not None:
                for det_id, det in detections.items():
                    # Extract bounding box and score
                    boxes.append(det["facial_area"])

        predictions.append(boxes)
    return predictions

def display_predictions_and_ground_truth(images, predictions, ground_truths, num_samples=10):
    for i in range(num_samples):
        image = images[i]
        pred_boxes = predictions[i]
        gt_boxes = ground_truths[i]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Gambar prediksi (kiri)
        axes[0].imshow(image)
        if pred_boxes is not None:
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                axes[0].add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
                )
        axes[0].set_title(f"Predictions for Image {i+1}")
        axes[0].axis("off")

        # Gambar ground truth (kanan)
        axes[1].imshow(image)
        if gt_boxes:
            for gt_box in gt_boxes:
                x, y, width, height = gt_box
                axes[1].add_patch(
                    plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='green', facecolor='none')
                )
        axes[1].set_title(f"Ground Truth for Image {i+1}")
        axes[1].axis("off")

        plt.show()

def load_model_mtcnn(device):
    print('Start MTCNN Model Loading...')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    print('Model Loaded')
    return mtcnn

def load_model_haar():
    print('Start Haarcascade Model Loading...')
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise IOError("Failed to load Haar Cascade model.")
    print('Model Loaded')
    return face_cascade

# Load YuNet model from OpenCV
def load_model_yunet():
    print("Loading YuNet Model...")
    model_path = "face_detection_yunet_2023mar.onnx"
    face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold=0.5)
    print("Model Loaded")
    return face_detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate model')
    parser.add_argument('dataset', metavar='dataset', type=str, help='enter your dataset')
    parser.add_argument('model', metavar='model', type=str, help='enter your model')
    args = parser.parse_args()

    dataset_path = f"../Face Detection/Datasets/{args.dataset}/Images"
    annotation_file = f"../Face Detection/Datasets/{args.dataset}/annotations.json"
    images, ground_truths = load_wider_face_data(dataset_path, annotation_file)
    print(f"Total images loaded: {len(images)}")
    print(f"Total ground truths loaded: {len(ground_truths)}")

    # for i in range(3):
    #     print(f"\nImage {i + 1} shape: {images[i].shape}")
    #     print(f"Ground Truths for Image {i + 1}: {ground_truths[i]}")  # Bounding boxes
    #     display_image_with_annotations(images[i], ground_truths[i])


    if(args.model == 'mtcnn'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_mtcnn(device=device)
    elif(args.model == 'haarcascade'):
        model = load_model_haar()
    elif(args.model == 'yunet'):
        model = load_model_yunet()
    elif(args.model == 'retinaface'):
        model = RetinaFace

    num_samples = 50
    predictions = predict_faces(images, model, num_samples=num_samples)

    display_predictions_and_ground_truth(images, predictions, ground_truths, num_samples=num_samples)