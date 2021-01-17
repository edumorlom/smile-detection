import cv2
import dlib
from tensorflow.keras.models import load_model
import numpy as np

GREEN_COLOR = (0, 255, 0)

def predict(model, frame: np.array):
    '''Given a face model (trained using 48x48 and grey)
    and a face frame/image, return the label index
    and the probability of that label.'''
    # The model was trained on 48x48 images, resize the image.
    data = cv2.resize(frame, (48, 48), interpolation = cv2.INTER_AREA)

    # Convert the numpy array to float32.
    data = np.array(data, dtype="float32")

    # Reshape the image.
    data = data.reshape(48, 48, 1)

    # The maximum number in grey images is 255.
    # Divide array by 255 to normalize the image with values from 0 to 1.
    data /= 255

    # Return a list of prediction probabilities.
    prediction = model.predict(np.array([data]))[0]

    # Get the index of the highest probability.
    idx = np.argmax(prediction)
    return np.argmax(prediction), prediction[idx]

# Load the smiling detector model.
model = load_model('./smile.model')

# Start the face detector. Used to identify faces.
detector = dlib.get_frontal_face_detector()

# Begin capturing video.
video = cv2.VideoCapture(0)

# While capturing video
while True:
    # Load each frame.
    check, frame = video.read()

    # Convert each frame to gray instead of RGB.
    # The model was trained using 1 channel (gray) instead of 3 channels (RGB).
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame.
    faces = detector(gray_frame)

    # For every face, predict whether they are smiling or neutral.
    for face in faces:
        # Draw a square on the face.
        x = face.left()
        y = face.top()
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN_COLOR, 3)

        # Crop the face.
        face_img = gray_frame[y:y+h, x:x+w]

        # Predict whether the face_img is smiling or neutral.
        label, probability = predict(model, face_img)

        # label 0 = "Smiling", label 1 = "Neutral"
        # Add label and probability next to square around face.
        text = "Smiling" if label == 0 else "Neutral"
        text += ", " + str(probability)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, GREEN_COLOR)

    # Display the user in RGB
    cv2.imshow("Analyzing", frame)

    # Display 1 frame per ms.
    key = cv2.waitKey(1)

    # If user presses 'q' key, quit.
    if key == ord('q'):
        break

# When the user has quitted, close video and exit cv2.
video.release()
cv2.destroyAllWindows()