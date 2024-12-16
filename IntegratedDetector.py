import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import winsound
from ultralytics import YOLO

class IntegratedDetector:
    def __init__(self, reference_dir, model_path, desired_classes):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Face Recognition Setup
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=self.device)
        self.face_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Object Detection Setup
        self.object_model = YOLO(model_path).to(self.device)
        
        # Face Recognition Parameters
        self.embeddings = {}
        self.threshold = 0.6
        
        # Object Detection Parameters
        self.DESIRED_CLASSES = desired_classes
        self.CLASS_NAMES = {
            0: 'person',
            43: 'knife',
            63: 'laptop', 
            67: 'cell phone', 
            76: 'scissors'
        }
        
        # Load Reference Images
        self.load_reference_images(reference_dir)
        
    def load_reference_images(self, reference_dir):
        print("Loading reference images...")
        for person_name in os.listdir(reference_dir):
            person_path = os.path.join(reference_dir, person_name)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        image = Image.open(img_path).convert('RGB')
                        with torch.no_grad():
                            face = self.mtcnn(image)
                            if face is not None:
                                face = face.unsqueeze(0).to(self.device)
                                embedding = self.face_model(face).cpu().numpy()
                                self.embeddings[person_name] = embedding
                                print(f"Loaded embedding for {person_name}")
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        print("Reference images loaded successfully.")
    
    def detect_faces(self, frame):
        # Convert frame to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, _ = self.mtcnn.detect(rgb_frame)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Draw rectangle around detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Crop the detected face
                face_crop = rgb_frame[y1:y2, x1:x2]
                face_image = Image.fromarray(face_crop)
                
                try:
                    # Detect and preprocess the face
                    with torch.no_grad():
                        face_tensor = self.mtcnn(face_image)
                        if face_tensor is not None:
                            face_tensor = face_tensor.unsqueeze(0).to(self.device)
                            face_embedding = self.face_model(face_tensor).cpu().numpy()
                            
                            # Compare with all stored embeddings
                            best_match = None
                            highest_similarity = -1
                            
                            for person_name, ref_embedding in self.embeddings.items():
                                similarity = np.dot(ref_embedding, face_embedding.T) / (
                                    np.linalg.norm(ref_embedding) * np.linalg.norm(face_embedding)
                                )
                                similarity = similarity[0][0]
                                
                                if similarity > highest_similarity:
                                    highest_similarity = similarity
                                    best_match = person_name
                            
                            # Display result
                            if highest_similarity > self.threshold:
                                cv2.putText(frame, best_match, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "Unknown", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                winsound.Beep(frequency=1000, duration=500)
                        else:
                            print("Face not properly detected for verification.")
                except Exception as e:
                    print(f"Error processing face: {e}")
        
        return frame
    
    def detect_objects(self, frame):
        # Run YOLO inference
        results = self.object_model(frame)
        
        # Person and desired object tracking
        person_count = 0
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
                
                # Count persons
                if class_id == 0:  # Person class
                    person_count += 1
                    # Add person count text above bounding box
                    cv2.putText(frame, f'Person {person_count}', 
                                (int(x1), int(y1)-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                              (0, 255, 0), 2)
                
                # Draw label for other objects
                if class_id != 0:
                    label = f"{self.CLASS_NAMES[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                filtered_results.append({
                    'class_id': class_id,
                    'class_name': self.CLASS_NAMES[class_id],
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Display total person count
        cv2.putText(frame, f'Total Persons: {person_count}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, filtered_results
    
    def run(self):
        camera_url = "http://100.71.225.2:8080/video"   

        cap = cv2.VideoCapture(camera_url)

        # Open webcam
        # cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open webcam")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Detect and recognize faces
                frame = self.detect_faces(frame)
                
                # Detect objects and count persons
                frame, detections = self.detect_objects(frame)
                
                # Display frame
                cv2.imshow('Integrated Detection', frame)
                
                # Print detections to console
                if detections:
                    print("Detected objects:")
                    for det in detections:
                        print(f"{det['class_name']} (confidence: {det['confidence']:.2f})")
                
                # Break loop if 'x' is pressed
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break
        
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()

def main():
    # Paths and configurations
    REFERENCE_DIR = "employee_database"
    MODEL_PATH = r"D:\ReBIT Hackathon\Yolov11\yolo11x.pt"
    
    # Classes to detect (including person)
    DESIRED_CLASSES = [0, 43, 63, 67, 76]
    
    # Create and run integrated detector
    detector = IntegratedDetector(REFERENCE_DIR, MODEL_PATH, DESIRED_CLASSES)
    detector.run()

if __name__ == "__main__":
    main()