import cv2
import torch
import multiprocessing as mp
import time
from ultralytics import YOLO

class TailgatingDetector:
    def __init__(self, model_path, authorized_count_queue):
        """
        Initialize tailgating detector
        
        Args:
            model_path (str): Path to YOLOv11 model
            authorized_count_queue (mp.Queue): Queue to receive authorized count
        """
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        
        # Authorization and detection tracking
        self.authorized_count_queue = authorized_count_queue
        self.current_authorized_count = 0
        
        # Detection parameters
        self.detection_interval = 3  # seconds between detection checks
        self.detection_threshold = 1  # max additional persons allowed
    
    def detect_tailgating(self):
        """
        Continuously monitor for tailgating
        """
        while True:
            # Check for new authorized count
            try:
                # Non-blocking check for new authorized count
                while not self.authorized_count_queue.empty():
                    authorized_count = self.authorized_count_queue.get_nowait()
                    
                    # Check for exit signal
                    if authorized_count is None:
                        print("Exiting tailgating detection...")
                        return
                    
                    # Update authorized count
                    self.current_authorized_count += authorized_count
                    print(f"🔑 Updated Authorized Count: {self.current_authorized_count}")
            except Exception:
                pass
            
            # Read frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect persons
            results = self.model(frame)
            
            # Count persons (class 0 is person in COCO dataset)
            person_ids = [0]
            persons = [box for result in results for box in result.boxes.data 
                       if int(box[5]) in person_ids]
            
            current_person_count = len(persons)
            
            # Tailgating check
            if self.current_authorized_count > 0:
                # If detected persons exceed authorized count plus threshold
                if current_person_count > self.current_authorized_count + self.detection_threshold:
                    print("⚠️ TAILGATING DETECTED!")
                    print(f"Authorized Count: {self.current_authorized_count}")
                    print(f"Detected Persons: {current_person_count}")
                    
                    # Optional: Add alarm, logging, etc.
                else:
                    print(f"Normal Entry: {current_person_count} person(s)")
            
            # Draw bounding boxes for visualization
            for person in persons:
                x1, y1, x2, y2 = map(int, person[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Tailgating Detection', frame)
            
            # Break loop on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to prevent high CPU usage
            time.sleep(0.1)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def start_tailgating_detector(model_path, authorized_count_queue):
    """
    Function to start tailgating detector in a separate process
    
    Args:
        model_path (str): Path to YOLOv11 model
        authorized_count_queue (mp.Queue): Queue to receive authorized count
    """
    detector = TailgatingDetector(model_path, authorized_count_queue)
    detector.detect_tailgating()

def main():
    # Model path
    MODEL_PATH = r"D:\ReBIT Hackathon\Yolov11\yolo11m.pt"
    
    # Create a queue for sharing authorized count
    authorized_count_queue = mp.Queue()
    
    # Create and start tailgating detector process
    detector_process = mp.Process(
        target=start_tailgating_detector, 
        args=(MODEL_PATH, authorized_count_queue)
    )
    detector_process.start()
    
    # Separate script to handle PIN entry will manage the authorized_count_queue
    # This main method is just to keep the process running
    detector_process.join()

if __name__ == "__main__":
    main()