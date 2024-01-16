from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'images_dataset_rns/Testing',
        target_size=(150, 150),
        batch_size=32,
        save_to_dir = 'images_dataset_rns/aug',
        class_mode='binary')

# Specify the path to your dataset directory
dataset_directory = "images_dataset_rns/Training"

# Get the list of folders (classes) in the dataset directory
class_folders = [folder for folder in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, folder))]

# Create a mapping of class indices to class names
class_names = {index: class_name for index, class_name in enumerate(class_folders)}

print(class_folders)
print(class_names)
for class_folder in class_folders:
    folder_path = os.path.join('images_dataset_rns/Training', class_folder)
    
    if os.path.exists(folder_path):
        train_generator = train_datagen.flow_from_directory(
            'images_dataset_rns/Training',
            target_size=(150, 150),
            batch_size=1,
            save_to_dir=f'images_dataset_rns/Training/{class_folder}',
            class_mode='binary'
        )
        batch = train_generator.next()
        print(f"Saving images to directory: {class_folder}")
        current_directory = os.getcwd()
        print(f"Current Working Directory: {current_directory}")
        # Access batch data and labels
        images, labels = batch