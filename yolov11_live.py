import cv2
import torch
from ultralytics import YOLO

class WebcamObjectDetector:
    def __init__(self, model_path, desired_classes):
        """
        Initialize the webcam object detector
        
        Args:
            model_path (str): Path to the YOLO model
            desired_classes (list): List of class IDs to detect
        """
        # Specific classes you want to detect
        self.DESIRED_CLASSES = desired_classes
        self.CLASS_NAMES = {
            43: 'knife',
            63: 'laptop', 
            67: 'cell phone', 
            76: 'scissors'
        }
        
        # Load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        
        # ip cam
        # camera_url = "http://100.71.225.2:8080/video"
        # camera_url = "192.168.105.168:554"

        # self.cap = cv2.VideoCapture(camera_url)

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        
        # Check if webcam opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam")
        
    def detect_objects(self, frame):
        """
        Detect specified objects in a single frame
        
        Args:
            frame (numpy.ndarray): Input frame from webcam
        
        Returns:
            list: Filtered detection results
        """
        # Run inference
        results = self.model(frame)
        
        filtered_results = []
        for result in results:
            # Convert result to numpy for easier manipulation
            boxes = result.boxes.data.cpu().numpy()
            
            # Filter boxes to keep only desired classes
            mask = torch.isin(
                torch.tensor(boxes[:, 5].astype(int)), 
                torch.tensor(self.DESIRED_CLASSES)
            ).numpy()
            filtered_boxes = boxes[mask]
            
            # Prepare filtered results
            for box in filtered_boxes:
                x1, y1, x2, y2 = box[:4]
                class_id = int(box[5])
                confidence = box[4]
                
                filtered_results.append({
                    'class_id': class_id,
                    'class_name': self.CLASS_NAMES[class_id],
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return filtered_results
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detection results
        
        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """
        Run real-time object detection on webcam feed
        """
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror-like effect
                frame = cv2.flip(frame, 1)
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Draw detections on frame
                frame_with_detections = self.draw_detections(frame, detections)
                
                # Display frame
                cv2.imshow('YOLOv11 Real-Time Object Detection', frame_with_detections)
                
                # Print detections to console
                if detections:
                    print("Detected objects:")
                    for det in detections:
                        print(f"{det['class_name']} (confidence: {det['confidence']:.2f})")
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    # Path to your YOLOv11 model
    MODEL_PATH = r"D:\ReBIT Hackathon\Yolov11\yolo11x.pt"
    
    # Classes to detect (43: knife, 63: laptop, 67: cell phone, 76: scissors)
    DESIRED_CLASSES = [43, 63, 67, 76]
    
    # Create and run detector
    detector = WebcamObjectDetector(MODEL_PATH, DESIRED_CLASSES)
    detector.run()

if __name__ == "__main__":
    main()