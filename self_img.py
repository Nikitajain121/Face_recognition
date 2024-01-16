import re
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from deepface import DeepFace
import pickle
from IPython.display import display, Image
from mtcnn import MTCNN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

# Specify the path to your dataset directory
dataset_directory = "images_dataset_rns/Training"

# Get the list of folders (classes) in the dataset directory
class_folders = [folder for folder in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, folder))]

# Create a mapping of class indices to class names
class_names = {index: class_name for index, class_name in enumerate(class_folders)}

print("Class Names:", class_names)
# Face detection using MTCNN
def detect_faces(frame):
    detector = MTCNN()
    return detector.detect_faces(frame)

# Preprocess the face image for prediction
def preprocess_image(img_array):
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Add a batch dimension
    return img_array

# Load the existing model and define new layers
# model = load_model('model_weights3.h5')
# new_model = Model(inputs=model.input, outputs=model.layers[-5].output)
# for layer in new_model.layers:
#     layer.trainable = False

# x = layers.Flatten()(new_model.output)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(256, activation='relu')(x)
# x = layers.Dense(4, activation='softmax')(x)

# # Compile the new model
# new_model2 = Model(new_model.input, x)
# new_model2.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model on the new dataset

# new_model2.fit(train_generator, epochs=20, steps_per_epoch=len(train_generator))
# new_model2.save('updated_model.h5')
#loaded_model = load_model('updated_model.h5')

#Video stream and face detection
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     if frame is not None:
#         faces = detect_faces(frame)

#         for face in faces:
#             x, y, width, height = face['box']
#             face_region = frame[y:(y+height), x:(x+width)]
#             face_region = cv2.resize(face_region, (224, 224))
#             face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

#             img_array = preprocess_image(face_region)
#             predictions = loaded_model.predict(img_array)
#             top_class_index = np.argmax(predictions)
#             # Adjust this part based on your model output
#             print(f"Predicted class: {class_names[top_class_index]}")

#         cv2.imshow("Video Feed", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         print("Error: Could not read frame from the viq.")
#         break

# cap.release()
# cv2.destroyAllWindows()


for i in range(num_batches_to_save):
    images, labels = train_generator.next()

    for j, image in enumerate(images):
        label_index = labels[j].argmax()
        label = class_names[label_index]

        # Convert the image array to PIL Image
        img_pil = array_to_img(image)

        # Save the image to the respective class subfolder
        img_pil.save(os.path.join(new_dataset, label, f"aug_{i * batch_size + j}.jpg"))


# Perform predictions on the test data
# model.evaluate(test_generator)
# predictions = loaded_model.evaluate(test_generator)
# print(predictions)

# Get predicted class indices
#predicted_class_indices = np.argmax(predictions, axis=1)

# Map predicted class indices to class names
#predicted_class_names = [class_names[idx] for idx in predicted_class_indices]

# Print the predicted class names for each test image
# for i, filename in enumerate(test_generator.filenames):
#     print(f"Image: {filename}, Predicted Class: {predicted_class_names[i]}")

#true_class_indices = test_generator.classes

# # Print the classification report
# print("Classification Report:")
# #print(classification_report(true_class_indices, predicted_class_indices, target_names=class_names.values()))

# # Print the confusion matrix
# print("Confusion Matrix:")
# print(confusion_matrix(true_class_indices, predicted_class_indices))
