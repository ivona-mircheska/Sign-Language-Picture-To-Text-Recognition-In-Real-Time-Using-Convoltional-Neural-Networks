Introduction

Sign languages are vital for communication among deaf and hard of hearing individuals, effectively bridging the gap between them and hearing people. Sign language interpreters facilitate this communication by translating between sign language and spoken language. Despite their importance, challenges persist due to the flexible nature of sign languages and the global shortage of expert interpreters. To address these challenges, technology-based systems are increasingly needed as complements to traditional interpretation.

The diversity of sign languages presents significant challenges:

Non-Manual Features: Includes facial expressions, body poses, and hand gestures.
Word-Level Sign Spelling: Each gesture represents an entire word.
Finger Vocabulary: Each gesture represents an individual letter or number.
Two methods were explored for sign language recognition:

Image-Based Prediction: Utilizing a model trained on existing images.
Real-Time Prediction: Recognizing signs using a camera.

Dataset

The dataset comprises 12,180 images (224x224 pixels) organized into 58 folders, each representing a different sign. The images, captured under various conditions (angles, lighting, hand positions), ensure data diversity. The 58 classes include 28 letters and several Macedonian words.

Data Preparation

A CSV file lists the image paths and target values. Labels are binarized using Scikit-Learn's LabelBinarizer, creating one-hot encoded vectors. For example, label ‘A’ is represented as a vector with '1' in the first position and '0's elsewhere. The dataset is divided into:

Training Set: 9,135 images (75%)

Validation Set: 3,045 images (25%)

Test Set: Initially 58 images, equally distributed across classes.

Data Augmentation is applied to training images to enhance model generalization and avoid overfitting. The augmentation parameters are:

Rotation Range	10

Width Shift Range	0.03

Height Shift Range	0.03

Shear Range	0.03

Zoom Range	0.03

Horizontal Flip	True

Model

A custom model was developed based on ResNet50, a deep CNN architecture known for its effective image classification capabilities.

Model Architecture:

Base Model: ResNet50 (pre-trained, top classifier removed)

Custom Classifier:
Flattening Layer
Dense Layers:
256 neurons (ReLU)
128 neurons (ReLU)
64 neurons (ReLU)
32 neurons (ReLU)
Output Layer: Dense with num_classes neurons (Softmax)

Initially, all ResNet50 layers were frozen. After training, some layers were unfrozen and fine-tuned to improve performance. This adjustment leveraged batch normalization to handle image variations effectively.

Experimental Results

Fine-tuning improved test accuracy from 96.55% to 100%. The model was trained for 5 epochs with a batch size of 64. Training and validation accuracies became stable after the third epoch, with validation accuracy generally higher. Figures 5 and 6 illustrate accuracy and loss trends for train and validation sets.

New test images demonstrated the model's generalization ability, though errors occurred due to hand positioning or lighting issues.

Testing the Data

The test.csv file includes paths and target values for test images. An additional 11 images were tested, with 9 correctly classified. The remaining 2 images were misclassified due to similar predicted and actual classes.

Real-Time Sign Language Recognition

Real-time recognition was implemented using OpenCV. The VideoCapture() function connects to the default camera, capturing 224x224 pixel frames. Each frame is saved as frame.jpg and used for prediction. Proper hand positioning within the frame is crucial for accurate results. The model generally performed well, though some errors were due to incorrect hand positioning or gesture execution.


This study introduced a ResNet50-based model for sign language recognition. Despite high accuracy, some errors were noted. To improve the model, it should be trained with more diverse data, including varied lighting conditions, backgrounds, and hand positions, to enhance overall performance.
