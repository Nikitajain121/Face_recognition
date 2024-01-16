import cv2
from IPython.display import display, Image
from mtcnn import MTCNN
import numpy as np

def align_face(frame, landmarks):
    # Define the desired eye positions in the output image
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)

    # Calculate the angle between the eyes
    dX = landmarks[1][0] - landmarks[0][0]
    dY = landmarks[1][1] - landmarks[0][1]
    angle = np.degrees(np.arctan2(dY, dX))

    # Calculate the desired angle
    desired_angle = np.degrees(np.arctan2(
        desired_right_eye[1] - desired_left_eye[1],
        desired_right_eye[0] - desired_left_eye[0]
    ))

    # Calculate the scale factors for resizing
    dist = np.sqrt(dX**2 + dY**2)
    desired_dist = (desired_right_eye[0] - desired_left_eye[0])
    scale = desired_dist / dist

    # Get the rotation matrix and apply the transformation
    matrix = cv2.getRotationMatrix2D(tuple(landmarks[0]), angle - desired_angle, scale)
    aligned_face = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

    return aligned_face
# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

image_count = 0
start_time = cv2.getTickCount()

# Create an instance of the MTCNN detector
detector = MTCNN()


while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)
    print(faces)
    # Loop through detected faces
    for face in faces:
        x, y, width, height = face['box']

        # Extract face landmarks
        landmarks = face['keypoints'].values()
        print(landmarks)
        # Align the face
        aligned_face = align_face(frame, landmarks)

        # Display the aligned face using OpenCV's cv2.imshow
        cv2.imshow("Aligned Face", aligned_face)

        # Save the aligned face to a file
        image_filename = f"aligned_face_{image_count}.jpg"
        cv2.imwrite('aligned_faces/' + image_filename, aligned_face)

    # Draw rectangles around the detected faces on the original frame
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the original frame with rectangles
    cv2.imshow("Original Frame with Rectangles", frame)

    # Save the captured image to a file
    image_filename = f"image_{image_count}.jpg"
    cv2.imwrite('images/' + image_filename, frame)

    image_count += 1

    # Stop capturing after 3 seconds
    current_time = cv2.getTickCount()
    elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
    if elapsed_time > 3:
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

