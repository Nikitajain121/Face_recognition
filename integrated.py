import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from IPython.display import display, Image
import pickle

inv_map = None
with open("inverse_mapping.pkl", "rb") as file:
    inv_map = pickle.load(file)
    print(inv_map)
# Load the pre-trained model
model = load_model('model_weights3.h5')

# Open a video capture stream (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

image_count = 0

while image_count <= 10:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Webcam", frame)

        # Save the captured image to a file
        image_filename = f"image_{image_count}.jpg"
        cv2.imwrite('/Users/home/Downloads/proj_face_recog/images/' + image_filename, frame)

        # Preprocess the frame for prediction
        img = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, 0)

        # Make predictions
        predictions = model.predict(img_array)
        prediction_label = np.argmax(predictions)

        # Print the predicted label
        print(f"Predicted Label: {inv_map.get(prediction_label)}")


        image_count += 1

        # Display the captured image in the notebook
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame.")
            break
        display(Image(data=buffer.tobytes()))

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1000) & 0xFF == ord('k'):
            break
    else:
        print("Error: Could not read frame from camera.")
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
