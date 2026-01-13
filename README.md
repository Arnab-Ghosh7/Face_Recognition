# Face Recognition System using VGG16

This project implements a comprehensive Face Recognition system leveraging Transfer Learning with the VGG16 architecture. It includes tools for data collection, model training, and real-time face recognition via webcam.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Collection](#1-data-collection)
  - [2. Model Training](#2-model-training)
  - [3. Real-time Recognition](#3-real-time-recognition)
- [Technologies Used](#technologies-used)
- [License](#license)

## 🔍 Overview

By freezing the initial layers of VGG16 and adding custom dense layers, we fine-tune the model to recognize specific faces with high accuracy, even with a limited number of training samples.

### Workflow:
1.  **Data Collection**: We capture facial data using a webcam and extract faces using Haar Cascade Classifiers.
2.  **Preprocessing**: Images are resized to 224x224 and normalized to match VGG16 input requirements.
3.  **Feature Extraction**: The pre-trained VGG16 layers process the image to identify complex patterns.
4.  **Classification**: A custom Softmax layer classifies the features into the respective person identities.

## ✨ Features
- **Automated Data Collection**: script to capture and save face images for training.
- **Transfer Learning**: Utilizes VGG16 weights (ImageNet) for effective feature extraction with limited data.
- **Real-time Detection**: Seamless integration with webcam for live recognition.
- **Data Augmentation**: Integrated image adjustments (shear, zoom, flip) to improve model robustness.

## 📂 Project Structure
```
Face_Recognition/
├── FaceSeamlessnetry.ipynb        # Data collection notebook
├── face_Recognition.py            # Model training script
├── facefrontend.py                # Real-time recognition application
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── Datasets/                      # Directory for Train/Test data (created by user)
│   ├── Train/
│   └── Test/
├── facefeatures_new_model.h5      # Trained model (generated after training)
├── README.md                      # Project documentation
└── LICENSE                        # MIT License
```

## ⚙️ Prerequisites
- Python 3.6+
- Webcam

## 🔧 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Arnab-Ghosh7/Face_Recognition
    cd Face_Recognition
    ```

2.  **Install dependencies**
    ```bash
    pip install tensorflow keras opencv-python pillow matplotlib numpy
    ```

## 🚀 Usage

### 1. Data Collection
Open `FaceSeamlessnetry.ipynb` in Jupyter Notebook or convert it to a script. This tool captures images from your webcam to create a custom dataset.
- Run the cells to initialize the webcam.
- It will automatically crop faces and save them to an `./Images/` directory (you may need to manually organize these into `Datasets/Train/<PersonName>` and `Datasets/Test/<PersonName>`).

### 2. Model Training
After organizing your collected images into `Datasets/Train` and `Datasets/Test` folders:
```bash
python face_Recognition.py
```
This script will:
- Load the VGG16 base model (excluding top layers).
- Train the model on your custom dataset.
- Save the trained model as `facefeatures_new_model.h5`.
- Save accuracy and loss plots.

### 3. Real-time Recognition
Run the frontend script to start the recognition system:
```bash
python facefrontend.py
```
- Ensure `facefeatures_new_model_final.h5` (or the model name you saved) is present.
- The webcam will open, detecting faces and displaying the predicted name with confidence.

## 🛠 Technologies Used
![Python](https://img.shields.io/badge/Python-3.6+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Author
 Arnab Ghosh
