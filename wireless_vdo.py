import cv2
from IPython.display import display, Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pickle
from deepface import DeepFace
import os
# Create an instance of the MTCNN detector
detector = MTCNN()

# Replace 'your_phone_ip_address' and 'your_port' with the actual IP address and port
camera_url = 'rtsp://admin:admin123@192.168.1.2:554/H264?ch=3&subtype=0'

# Open the video stream
cap = cv2.VideoCapture(0)

image_count = 0
model = load_model('model_weights3.h5')

def preprocess_image(img_array):
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Add a batch dimension
    return img_array
with open("inverse_mapping.pkl", "rb") as file:
    inv_map = pickle.load(file)
    print(inv_map)
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if frame is not None:

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)
        #print(frame)
        # Draw rectangles around the detected faces
        model_type = 'Deepface'
        if model_type!='Deepface':
            if faces:
                for face in faces:
                
                    x, y, width, height = face['box']
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    # Extract the face region for prediction
                    face_region = frame[y:y+height, x:x+width]
                    
                    # Resize the face region to the input size expected by the model
                    face_region = cv2.resize(face_region, (224, 224))

                    # Convert BGR to RGB (OpenCV uses BGR by default)
                    face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                    # Preprocess the image for model prediction
                    img_array = preprocess_image(face_region)

                    # Make predictions using the model
                    predictions = model.predict(img_array)
                    top_class_index = np.argmax(predictions)
                    top_class_label = inv_map[top_class_index]
                    confidence_score = predictions[0][top_class_index]
                    print(top_class_label, confidence_score)
                    # Adjust this part based on your model output

        curr_face = frame
        db_directory ="mixed"
        results = DeepFace.find(img_path = curr_face, db_path = db_directory, model_name='VGG-Face', detector_backend='mtcnn', distance_metric='euclidean', enforce_detection=False)
        #names = results[0]['identity'].str.split('/', n=1, expand=True)
        if  results:
            names = results[0]['identity'].apply(os.path.basename)#.str.split('/', n=1, expand=True)
            #names.columns = ['Folder', 'File_Name']  # Change column names as needed
            print("Split Names:")
            print(names)
        else:
            print("No faces or matches found.")
# Display the DataFrame with the new columns
       # print(names)

        cv2.imshow("IP Camera Feed", frame)

        # Save the captured image to a buffer
        ret, buffer = cv2.imencode('.jpg', frame)
        display(Image(data=buffer.tobytes()))

        # Save the captured image to a file
        image_filename = f"image_{image_count}.jpg"
        cv2.imwrite('images/' + image_filename, frame)

        image_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    else:
        print("Error: Could not read frame from the video.")
        break

# Release the video stream and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
