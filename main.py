import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('emotion_detection_model_3_100epochs.h5', compile=False)

# Class labels for emotions
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face
        roi_gray = gray_frame[y:y+h, x:x+w]

        # Resize the face to 48x48 pixels (required by the model)
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Normalize pixel values (0-255 -> 0-1)
        roi_gray = roi_gray / 255.0

        # Add a batch dimension and a channel dimension
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion
        predictions = model.predict(roi_gray)
        emotion_label = class_labels[np.argmax(predictions)]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the predicted emotion
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame with the emotion label
    cv2.imshow('Real-Time Emotion Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
