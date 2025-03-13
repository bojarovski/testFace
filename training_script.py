import face_recognition
import os
import pickle
import cv2

# Folder containing training images (one image per person or group)
training_images_folder = 'training_images'

# Create empty lists for face encodings and names
known_face_encodings = []
known_face_names = []

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

# Loop through images and preprocess before encoding
for image_name in os.listdir(training_images_folder):
    image_path = os.path.join(training_images_folder, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_name}. Skipping.")
        continue

    image = preprocess_image(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_image)

    if not face_encodings:
        print(f"No faces detected in {image_name}. Skipping.")
        continue

    name = os.path.splitext(image_name)[0]
    known_face_encodings.append(face_encodings[0])
    known_face_names.append(name)

# Save encodings and names
with open('known_faces.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Training completed. Known faces saved.")
