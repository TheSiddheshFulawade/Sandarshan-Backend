import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Specific classes you want to detect
DESIRED_CLASSES = [43, 63, 67, 76]  # knife, laptop, cell phone, scissors
CLASS_NAMES = {
    43: 'knife',
    63: 'laptop', 
    67: 'cell phone', 
    76: 'scissors'
}

def detect_specific_objects(model, image_path):
    """
    Detect only specified classes in the given image
    
    Args:
        model (YOLO): Loaded YOLO model
        image_path (str): Path to the input image
    
    Returns:
        Filtered detection results
    """
    # Run inference
    results = model(image_path)
    
    # Filter results to include only desired classes
    filtered_results = []
    for result in results:
        # Convert result to numpy for easier manipulation
        boxes = result.boxes.data.cpu().numpy()
        
        # Filter boxes to keep only desired classes
        mask = np.isin(boxes[:, 5].astype(int), DESIRED_CLASSES)
        filtered_boxes = boxes[mask]
        
        # Prepare filtered results
        for box in filtered_boxes:
            x1, y1, x2, y2 = box[:4]
            class_id = int(box[5])
            confidence = box[4]
            
            filtered_results.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
    
    return filtered_results

def visualize_detections(image_path, detections):
    """
    Visualize detected objects on the image
    
    Args:
        image_path (str): Path to the input image
        detections (list): List of detection results
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Draw bounding boxes and labels
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save or display the image
    cv2.imwrite('detected_objects.jpg', image)
    print("Detections saved to detected_objects.jpg")

def main():
    # Load the YOLOv11 model
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO("D:\\ReBIT Hackathon\\Yolov11\\yolo11x.pt").to(device)
    
    # Path to your input image
    image_path = r"D:\ReBIT Hackathon\Yolov11\sample.png"
    
    # Detect specific objects
    detections = detect_specific_objects(model, image_path)
    
    # Print detections
    for detection in detections:
        print(f"Detected {detection['class_name']} with confidence {detection['confidence']:.2f}")
    
    # Visualize detections
    visualize_detections(image_path, detections)

if __name__ == "__main__":
    main()