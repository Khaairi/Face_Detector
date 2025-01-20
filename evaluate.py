import json
import os
from PIL import Image
import time
import numpy as np
import argparse
from facenet_pytorch import MTCNN
import torch
import cv2
from mean_average_precision import MetricBuilder
from retinaface import RetinaFace

# Function to load WIDER Face data
def load_wider_face_data(dataset_path, annotation_file, sample_size=1000, seed=42):
    print('Start loading dataset...')
    
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found at: {annotation_file}")
        return [], []

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    image_info_map = {img["id"]: img["file_name"] for img in annotations["images"]}
    annotation_map = {}
    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]  # bbox dalam format [x, y, width, height]
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        converted_bbox = [x, y, x2, y2]

        if image_id not in annotation_map:
            annotation_map[image_id] = []
        annotation_map[image_id].append(converted_bbox)

    images = []
    ground_truths = []

    image_ids = list(image_info_map.keys())
    np.random.seed(seed)
    np.random.shuffle(image_ids)

    # selected_image_ids = image_ids[:sample_size]
    selected_image_ids = image_ids

    for image_id in selected_image_ids:
        file_name = image_info_map.get(image_id)
        if file_name is None:
            print(f"No file name found for image_id {image_id}")
            continue

        image_path = os.path.join(dataset_path, file_name)

        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(np.array(image))

                gt_boxes = annotation_map.get(image_id, [])
                ground_truths.append(gt_boxes)
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")
        else:
            print(f"Image not found at path: {image_path}")

    return images, ground_truths

# Load models
def load_model_mtcnn(device):
    print('Start Model Loading...')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    print('Model Loaded')
    return mtcnn

def load_model_haar():
    print('Loading Haar Cascade Model...')
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise IOError("Failed to load Haar Cascade model.")
    print('Model Loaded')
    return face_cascade

def load_model_yunet():
    print("Loading YuNet Model...")
    model_path = "face_detection_yunet_2023mar.onnx"
    face_detector = cv2.FaceDetectorYN.create(model_path, "", (300, 300), score_threshold=0.5)
    print("Model Loaded")
    return face_detector

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('dataset', metavar='dataset', type=str, help='Enter your dataset')
    parser.add_argument('model', metavar='model', type=str, help='Enter your model')
    args = parser.parse_args()

    dataset_path = f"../Face Detection/Datasets/{args.dataset}/Images"
    annotation_file = f"../Face Detection/Datasets/{args.dataset}/annotations.json"
    images, ground_truths = load_wider_face_data(dataset_path, annotation_file)
    print(f"Total images loaded: {len(images)}")
    print(f"Total ground truths loaded: {len(ground_truths)}")

    inference_times = []
    detection = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if(args.model == 'mtcnn'):
        model = load_model_mtcnn(device=device)
    elif(args.model == 'haarcascade'):
        model = load_model_haar()
    elif(args.model == 'yunet'):
        model = load_model_yunet()

    # Initialize map metric function
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for idx, (image, gt_boxes) in enumerate(zip(images, ground_truths), start=1):
        print(f"Processing image {idx}/{len(images)}...")
        start_time = time.time()
        max_score = 1000

        boxes_with_scores = []

        if(args.model == 'mtcnn'):
            boxes, probs = model.detect(image)
            if boxes is not None:
                boxes_with_scores = [(box.tolist(), score) for box, score in zip(boxes, probs)]
            else:
                boxes_with_scores = []
        elif(args.model == 'haarcascade'):
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detected_faces = model.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            boxes_with_scores = [([x, y, x + w, y + h], 1.0) for (x, y, w, h) in detected_faces]
        elif(args.model == 'yunet'):
            image_height, image_width = image.shape[:2]
            model.setInputSize((image_width, image_height))
            _, detected_faces = model.detect(image)
            if detected_faces is not None:
                boxes_with_scores = [([x, y, x + w, y + h], score/max_score) for (x, y, w, h, score) in detected_faces[:, :5]]
            else:
                boxes_with_scores = []
        elif(args.model == 'retinaface'):
            # Convert the image from RGB to BGR (RetinaFace often expects BGR images)
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            detections = RetinaFace.detect_faces(bgr_image)
            if detections is not None:
                for det_id, det in detections.items():
                    # Extract bounding box and score
                    facial_area = det["facial_area"]
                    score = det["score"]
                    x1, y1, x2, y2 = facial_area  # Unpack the bounding box coordinates
                    boxes_with_scores.append(([x1, y1, x2, y2], score))

        # Prepare the predictions for mAP calculation
        preds = []
        for box, score in boxes_with_scores:
            preds.append([box[0], box[1], box[2], box[3], 0, score])

        # Add ground truth boxes to mAP calculation (class 0, no difficult, no crowd)
        gt = []
        for box in gt_boxes:
            gt.append([box[0], box[1], box[2], box[3], 0, 0, 0])

        # Add to the metric function
        metric_fn.add(np.array(preds), np.array(gt))

        detection.append(boxes_with_scores)
        end_time = time.time()
        inference_times.append(end_time - start_time)

    # mAP = metric_fn.value(iou_thresholds=0.5)["mAP"]
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Total inference time: {sum(inference_times):.4f} seconds")
    print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
    # print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

    # compute PASCAL VOC metric at the all points
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

    # compute metric COCO metric
    print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
