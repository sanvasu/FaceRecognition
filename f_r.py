import cv2
import face_recognition
import os

# Load known face encodings and their names
known_face_encodings = []
known_face_names = []

# Load known faces from a specified directory
def load_known_faces(known_faces_dir):
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(known_faces_dir, filename)
            try:
                image = face_recognition.load_image_file(image_path)

                # Check if the image is in the expected format
                if image.ndim != 3 or image.shape[2] != 3:
                    print(f"Warning: The image {filename} is not an RGB image.")
                    continue

                # Get the face encoding (128-dimension feature vector)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    face_encoding = face_encodings[0]
                    print(f"Loaded encoding for {os.path.splitext(filename)[0]}: {face_encoding.shape}")
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(os.path.splitext(filename)[0])   # Use the filename (without extension) as the name
                else:
                    print(f"Warning: No faces found in the image {filename}.")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

# Directory containing known face images
known_faces_dir = 'known_faces'  # Change this to your directory containing known faces
load_known_faces(known_faces_dir)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Failed to grab frame")
        break

    # Check if the frame is valid
    if frame is None or frame.size == 0:
        print("Warning: Captured an empty frame.")
        continue  # Skip this iteration

    # Convert frame to RGB (for face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check the shape of the image
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        print("Warning: The frame is not in RGB format or does not have 3 channels.")
        continue  # Skip this iteration

    # Detect all face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Detect face encodings (feature vectors) for each face in the current frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name if no match is found

        # If a match was found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw the name of the person below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
