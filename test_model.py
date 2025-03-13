import cv2
import face_recognition
import os
import pickle

# Load known faces and names from pickle file
def load_known_faces():
    if os.path.exists('known_faces.pkl'):
        with open('known_faces.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return [], []

# Preprocess the image (resize and normalize lighting)
def preprocess_image(image):
    """Resize and normalize the image for consistent face recognition."""
    image = resize_image(image)
    return adjust_lighting(image)

def resize_image(image, max_width=800):
    """Resize image to a maximum width while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if width > max_width:
        scaling_factor = max_width / width
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height))
    return image

def adjust_lighting(image):
    """Normalize lighting in the image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    updated_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

# Capture a single photo from the webcam
def capture_single_person():
    print("Capturing photo of a single person...")
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret:
        print("Failed to grab frame from webcam.")
        return None

    # Preprocess the frame
    frame = preprocess_image(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_encodings[0]  # Assume single face for individual capture
    else:
        print("No faces detected. Please try again.")
        return None

# Process a group photo provided by the user
def process_group_photo():
    image_path = input("Enter the path to the group photo: ")
    if not os.path.exists(image_path):
        print(f"Error: The file at {image_path} does not exist.")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load the image from {image_path}.")
        return None

    # Preprocess the image
    image = preprocess_image(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    if face_locations:
        return face_recognition.face_encodings(rgb_image, face_locations), face_locations, image
    else:
        print("No faces detected in the group photo.")
        return None, None, None

# Recognize faces by comparing with known encodings
def recognize_faces(face_encodings, known_face_encodings, known_face_names, tolerance=0.6):
    recognized_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        recognized_names.append(name)
    return recognized_names

def main():
    print("Welcome to the Face Recognition Program!")
    known_face_encodings, known_face_names = load_known_faces()

    choice = input("Would you like to capture a single person (1) or provide a group photo (2)? (Enter 1 or 2): ")

    if choice == '1':
        face_encoding = capture_single_person()
        if face_encoding is not None:
            recognized_name = recognize_faces([face_encoding], known_face_encodings, known_face_names)
            if recognized_name != ["Unknown"]:
                print(f"Recognized: {recognized_name[0]}")
            else:
                print("Face not recognized.")
        else:
            print("Failed to capture a valid face.")

    elif choice == '2':
        face_encodings, face_locations, group_image = process_group_photo()
        if face_encodings and face_locations:
            recognized_names = recognize_faces(face_encodings, known_face_encodings, known_face_names)
            print("Recognized faces:", recognized_names)

            # Draw rectangles and names on the image
            for (top, right, bottom, left), name in zip(face_locations, recognized_names):
                cv2.rectangle(group_image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(group_image, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            # Show the annotated group image
            cv2.imshow("Group Photo", group_image)
            print("Press any key to close the image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to process the group photo.")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
