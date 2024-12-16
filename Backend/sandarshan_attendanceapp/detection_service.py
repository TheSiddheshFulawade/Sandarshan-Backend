import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import django
import sys

# Add the project root to Python path
project_root = r'D:\ReBIT Hackathon\Yolov11\Backend'
sys.path.append(project_root)

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sandarshan.settings')
django.setup()

from sandarshan_userapp.models import EmployeeUser, GuestUser
from ultralytics import YOLO

class IntegratedDetectionService:
    def __init__(self, model_path=r"D:\ReBIT Hackathon\Yolov11\yolo11x.pt"):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Face Recognition Setup
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=self.device)
        self.face_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Object Detection Setup
        self.DESIRED_CLASSES = [0, 43, 63, 67, 76]  # Added 'person'
        self.CLASS_NAMES = {
            0: 'person',
            43: 'knife',
            63: 'laptop', 
            67: 'cell phone', 
            76: 'scissors'
        }
        self.object_model = YOLO(model_path).to(self.device)
        
        # Face Recognition Embeddings
        self.embeddings = {}
        self.threshold = 0.6
        self.load_reference_images()
    
    def load_reference_images(self, reference_dir=None):
        print("Loading reference images...")
        
        # Combine Employee and Guest Users
        all_users = list(EmployeeUser.objects.all()) + list(GuestUser.objects.all())
        
        for user in all_users:
            # Assume each user has a profile picture in a specific directory
            user_image_dir = os.path.join(project_root, 'employee_database', user.name)
            
            # Create directory if it doesn't exist
            os.makedirs(user_image_dir, exist_ok=True)
            
            # Look for image files
            image_files = [f for f in os.listdir(user_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            
            for img_file in image_files:
                img_path = os.path.join(user_image_dir, img_file)
                try:
                    image = Image.open(img_path).convert('RGB')
                    with torch.no_grad():
                        face = self.mtcnn(image)
                        if face is not None:
                            face = face.unsqueeze(0).to(self.device)
                            embedding = self.face_model(face).cpu().numpy()
                            self.embeddings[user.name] = embedding
                            print(f"Loaded embedding for {user.name}")
                            break  # Use first valid image
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if not self.embeddings:
            print("WARNING: No reference images found. Face recognition may not work.")
    
    def recognize_face(self, face_image):
        try:
            with torch.no_grad():
                face_tensor = self.mtcnn(face_image)
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    face_embedding = self.face_model(face_tensor).cpu().numpy()

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

                    return (best_match, highest_similarity) if highest_similarity > self.threshold else (None, highest_similarity)
                return (None, 0)
        except Exception as e:
            print(f"Face recognition error: {e}")
            return (None, 0)
    
    def detect_objects(self, frame):
        try:
            results = self.object_model(frame)
            
            filtered_results = []
            for result in results:
                boxes = result.boxes.data.cpu().numpy()
                
                mask = torch.isin(
                    torch.tensor(boxes[:, 5].astype(int)), 
                    torch.tensor(self.DESIRED_CLASSES)
                ).numpy()
                filtered_boxes = boxes[mask]
                
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
        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def process_frame(self, frame):
        """Integrated processing of face recognition and object detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face Detection
        face_results = []
        boxes, _ = self.mtcnn.detect(rgb_frame)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_crop = rgb_frame[y1:y2, x1:x2]
                face_image = Image.fromarray(face_crop)
                
                recognized_name, confidence = self.recognize_face(face_image)
                
                face_results.append({
                    'bbox': (x1, y1, x2, y2),  # Full coordinates
                    'name': recognized_name,
                    'confidence': confidence
                })
        
        # Object Detection
        object_results = self.detect_objects(frame)
        
        return {
            'faces': face_results,
            'objects': object_results
        }

# Initialize the service
detection_service = IntegratedDetectionService()