# Face_recognition

## Overview
This repository contains Python code and models for face recognition tasks using advanced deep learning techniques. It integrates multiple tools and libraries like MTCNN for face detection and deep learning models for face recognition.

## Features
- Face detection using MTCNN
- Face recognition with pre-trained deep learning models
- Face alignment and augmentation scripts
- Support for real-time face recognition via webcam or video input
- Model weights included for quick setup
- Various utility scripts for data handling and processing

## Repository Structure
├── align.py # Face alignment utilities
├── augment.py # Data augmentation scripts
├── face_recog_jbk.py # Main face recognition script
├── hackathon.py # Script added with MTCNN integration
├── integrated.py # Integrated pipeline script
├── self_img.py # Self image capture/recognition script
├── wireless_vdo.py # Script for wireless video feed processing
├── model_weights3.h5 # Pre-trained model weights
├── updated_model.h5 # Updated model weights
├── inverse_mapping.pkl # Label mapping file
├── y_data.npy # Sample numpy data file
├── images_dataset_rns/ # Folder with training/testing images
├── mixed/ # Mixed dataset folder
├── .gitignore # Git ignore file
└── README.md # This file


## Installation

Clone the repository:


git clone https://github.com/Nikitajain121/Face_recognition.git
cd Face_recognition

## Create and activate a virtual environment (Linux/macOS)
python -m venv venv
source venv/bin/activate

## For Windows PowerShell, use:
## venv\Scripts\Activate.ps1

## For Windows CMD, use:
## venv\Scripts\activate.bat

## Install dependencies from requirements.txt
pip install -r requirements.txt

## If requirements.txt is missing, install main packages manually
pip install tensorflow mtcnn opencv-python numpy

## Run face recognition on an image or video
python face_recog_jbk.py --input path/to/image_or_video

## Run real-time face recognition using webcam
python wireless_vdo.py

## Align dataset images
python align.py

## Augment dataset images
python augment.py


