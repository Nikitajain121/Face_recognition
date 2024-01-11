from deepface import DeepFace
db_directory = "mixed"
curr_face = "mixed/Tabu_cropped_c0477cd2b17e1ba80979ddec65b6d028.jpg"
results = DeepFace.find(img_path = curr_face, db_path = db_directory, model_name='VGG-Face', detector_backend='mtcnn', distance_metric='euclidean', enforce_detection=False)
print(results[0]['identity'].str.rsplit('/', n=1, expand=True))