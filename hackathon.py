# import cv2
# from IPython.display import display, Image
# import io
# from mtcnn import MTCNN  # Import the MTCNN module

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# image_count = 0

# # Create an instance of the MTCNN detector
# detector = MTCNN()

# while image_count <= 10:
#     ret, frame = cap.read()
#     if ret:
#         # Detect faces using MTCNN
#         faces = detector.detect_faces(frame)

#         # Draw rectangles around the detected faces
#         for face in faces:
#             x, y, width, height = face['box']
#             cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

#         cv2.imshow("Webcam", frame)

#         # Save the captured image to a buffer
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             print("Error: Could not encode frame.")
#             break

#         # Display the captured image in the notebook
#         display(Image(data=buffer.tobytes()))

#         # Save the captured image to a file
#         image_filename = f"image_{image_count}.jpg"
#         cv2.imwrite('images/' + image_filename, frame)

#         image_count += 1

#         # Exit the loop if the 'Ctrl + K' key combination is pressed
#         key = cv2.waitKey(1)
#         if key == ord('k') and cv2.waitKey(1) & 0xFF == 0x11:  # 'Ctrl' key has a ASCII code of 0x11
#             break
#     else:
#         print("Error: Could not read frame from the camera.")
#         break

# # Release the camera and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
from IPython.display import display, Image
import io
from mtcnn import MTCNN  # Import the MTCNN module

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

    # Draw rectangles around the detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Save the captured image to a buffer
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        print("Error: Could not encode frame.")
        break

    # Display the captured image in the notebook
    display(Image(data=buffer.tobytes()))

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
