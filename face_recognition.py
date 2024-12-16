import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import pyttsx3
import winsound

# Initialize MTCNN and FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# engine = pyttsx3.init()

# voices = engine.getProperty('voices')
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
cap = cv2.VideoCapture(0)

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

            # Crop the detected face
            face_crop = rgb_frame[y1:y2, x1:x2]
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
                        else:
                            cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            winsound.Beep(frequency=1000, duration=500)
                            # engine.say(f"Alert! an unknown person has entered the premises")
                            # engine.runAndWait()
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
