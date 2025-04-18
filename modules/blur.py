import cv2
import os

def blur_faces(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be read")

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Blur each face found
    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
        image[y:y+h, x:x+w] = blurred_roi

    # Save and return the path to the blurred image
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_blurred{ext}"
    cv2.imwrite(output_path, image)

    return output_path
