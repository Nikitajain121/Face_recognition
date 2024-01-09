# import cv2
# from IPython.display import display, Image
# import io
# import requests
# import numpy as np
# from mtcnn import MTCNN

# # Replace 'your_mobile_ip_address' with the actual IP address of your mobile device
# camera_url = 'http://100.83.58.211:8080'

# # Initialize variables
# image_count = 0
# start_time = cv2.getTickCount()

# # Create an instance of the MTCNN detector
# detector = MTCNN()

# # Open the camera stream from the mobile device
# cap = cv2.VideoCapture(camera_url)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Could not read frame from the camera.")
#         break

#     # Detect faces using MTCNN
#     faces = detector.detect_faces(frame)

#     # Draw rectangles around the detected faces
#     for face in faces:
#         x, y, width, height = face['box']
#         cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

#     # Save the captured image to a buffer
#     ret, buffer = cv2.imencode('.jpg', frame)
#     if not ret:
#         print("Error: Could not encode frame.")
#         break

#     # Display the captured image in the notebook
#     display(Image(data=buffer.tobytes()))

#     # Save the captured image to a file
#     image_filename = f"image_{image_count}.jpg"
#     cv2.imwrite('images/' + image_filename, frame)

#     image_count += 1

#     # Stop capturing after 3 seconds
#     current_time = cv2.getTickCount()
#     elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
#     if elapsed_time > 3:
#         break

# # Release the camera and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import requests
# import numpy as np
# import time
# from IPython.display import display, Image
# from mtcnn import MTCNN 
# # URL of the camera stream
# camera_url = 'http://100.83.58.211:8080/'

# # Parameters for reconnecting when the camera goes down
# reconnect_interval = 10  # Time interval (in seconds) for reconnection attempts
# max_reconnect_attempts = 5  # Maximum number of reconnection attempts

# # Initialize variables for tracking reconnection attempts
# reconnect_attempts = 0
# last_reconnect_time = None

# # Function to calculate the frame timestamp, fps, and their latency
# def calculate_fps_and_latency(prev_frame_timestamp, frame_number):
#     current_time = time.time()
#     elapsed_time = current_time - prev_frame_timestamp
#     fps = frame_number / elapsed_time
#     latency = elapsed_time / frame_number
#     return current_time, fps, latency

# # Create an instance of the MTCNN detector
# detector = MTCNN()

# # Main loop for capturing frames
# cap = cv2.VideoCapture(camera_url)

# frame_number = 0
# prev_frame_timestamp = None

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         # Camera is down; attempt reconnection
#         if reconnect_attempts < max_reconnect_attempts:
#             if last_reconnect_time is None or (time.time() - last_reconnect_time) >= reconnect_interval:
#                 reconnect_attempts += 1
#                 last_reconnect_time = time.time()
#                 print(f"Reconnection attempt {reconnect_attempts}...")
#                 cap.release()
#                 cap = cv2.VideoCapture(camera_url)
#         else:
#             print("Maximum reconnection attempts reached. Exiting.")
#             break
#     else:
#         # Camera is online; reset reconnection attempts
#         reconnect_attempts = 0

#         # Calculate the frame timestamp and latency
#         prev_frame_timestamp = time.time()
#         frame_latency = time.time() - prev_frame_timestamp

#         # Detect faces using MTCNN
#         faces = detector.detect_faces(frame)

#         # Draw rectangles around the detected faces
#         for face in faces:
#             x, y, width, height = face['box']
#             cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

#         # Save the captured image to a buffer
#         _, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             print("Error: Could not encode frame.")
#             break

#         # Display the captured image in the notebook
#         display(Image(data=buffer.tobytes()))

#         # Save the captured image to a file
#         image_filename = f"image_{frame_number}.jpg"
#         cv2.imwrite('images/' + image_filename, frame)

#         frame_number += 1

#         # Check for the 'Esc' key to exit
#         key = cv2.waitKey(1)
#         if key == 27:  # 'Esc' key
#             break

# # Release the camera and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
