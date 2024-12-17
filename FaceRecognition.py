import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import pyttsx3
import winsound
import time
import vlc

# Initialize MTCNN and FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
audio_file = "Alert.mp3"
alert = vlc.MediaPlayer("Alert.mp3")
engine = pyttsx3.init()

voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

# Load reference images and compute embeddings
reference_dir = "employee_database"
embeddings = {}
threshold = 0.6  # Similarity threshold for matching

print("Loading reference images...")
for person_name in os.listdir(reference_dir):
    person_path = os.path.join(reference_dir, person_name)
    if os.path.isdir(person_path):  # Ensure it's a directory
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                with torch.no_grad():
                    face = mtcnn(image)
                    if face is not None:
                        face = face.unsqueeze(0).to(device)
                        embedding = model(face).cpu().numpy()
                        embeddings[person_name] = embedding
                        print(f"Loaded embedding for {person_name}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Reference images loaded successfully.")

# Initialize webcam
cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('rtsp://admin:PEUHQA@192.168.105.36:554/ucas/11', cv2.CAP_FFMPEG)
# cap.set(cv2.CAP_PROP_FPS, 30)  # Disable buffering  

# fmpeg = FFmpeg()
# cap = fmpeg.input(url='rtsp://admin:PEUHQA@192.168.105.36:554/h264Preview_01_main')
# cap = cap.filter('buffer', size=0)

threshold = 0.6
blink_counter = 0
blink_threshold = 3
blink_detected = False
start_time = None
timeout_duration = 10

# unknown_start_time = None
# unknown_face_detected = False
# alert_duration = 5  # seconds threshold

frames = 10
unknown_frame_count = 0
max_unknown_frames = int(1*frames)  #Seconds * Frames 
alert_raised = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Convert frame to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Draw rectangle around detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop the detected face
            face_crop = rgb_frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                try:
                    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                    # Detect eyes within the face region
                    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=7, minSize=(20, 20))

                    # Eye blink detection logic
                    # eyes_closed = len(eyes) == 0  # If no eyes are detected, it assumes eyes are closed

                    # if start_time is None:
                    #     start_time = time.time()

                    # if eyes_closed:
                    #     blink_counter += 1
                    # else:
                    #     if blink_counter >= blink_threshold:
                    #         blink_detected = True
                    #     blink_counter = 0

                    # elapsed_time = time.time()-start_time

                    # if elapsed_time > timeout_duration and not blink_detected:
                    #     cv2.putText(frame, "Spoofing Detected!", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    #     print("No blink detected within timeout.")
                    #     winsound.Beep(frequency=1000, duration=500)

                    # if not blink_detected:
                    #     cv2.putText(frame, "Blink to verify...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    #     continue  # Wait for blink before proceeding
                    # else:
                    #     cv2.putText(frame, "Blink verified", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                except cv2.error as e:
                    print(f"Error in processing cropped face: {e}")
            else:
                continue

            face_image = Image.fromarray(face_crop)

            try:
                # Detect and preprocess the face
                with torch.no_grad():
                    face_tensor = mtcnn(face_image)
                    if face_tensor is not None:
                        face_tensor = face_tensor.unsqueeze(0).to(device)
                        face_embedding = model(face_tensor).cpu().numpy()

                        # Compare with all stored embeddings
                        best_match = None
                        highest_similarity = -1

                        for person_name, ref_embedding in embeddings.items():
                            similarity = np.dot(ref_embedding, face_embedding.T) / (
                                np.linalg.norm(ref_embedding) * np.linalg.norm(face_embedding)
                            )
                            similarity = similarity[0][0]  # Extract scalar from the matrix

                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match = person_name

                        # Display result
                        if highest_similarity > threshold:
                            cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            # engine.say(f"{best_match} is Approved")
                            engine.runAndWait()
                            unknown_face_detected = False
                            unknown_start_time = None
                        else:
                            unknown_frame_count += 1

                            # Track the duration of unknown face detection
                            if unknown_frame_count > max_unknown_frames and not alert_raised:
                                print("Unknown face detected for more than 3 seconds!")
                                alert.play()
                                alert_raised = True
                            else:
                                cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        if highest_similarity > threshold:
                            unknown_frame_count = 0
                            alert_raised = False

                    else:
                        print("Face not properly detected for verification.")
            except Exception as e:
                print(f"Error processing face: {e}")

    else:
        print("No faces detected in the frame.")

    # Show the video feed
    cv2.imshow('Sandarashan', frame)

    # Exit the loop when 'x' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()