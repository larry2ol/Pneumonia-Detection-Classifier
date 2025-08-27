# Pneumonia-Detection-Classifier
This repository presents a deep learning solution for automated pneumonia detection from chest X-ray images using a custom Convolutional Neural Network (CNN) built with PyTorch. The model is trained on a custom, imbalanced dataset, utilising techniques to mitigate class imbalance and enhance diagnostic accuracy.

## Project Title
- Convolutional Neural Network Classification of an Imbalanced X-Ray Image Dataset using PyTorch.

## Overview
- The project utilises a custom Convolutional Neural Network (CNN) architecture, known for its effectiveness in image classification tasks due to its ability to learn and understand  spatial hierarchies in the data automatically.

## Why This Project Matters
- The project offers insightful strategies for effectively addressing class imbalance, a prevalent issue in many real-world medical datasets, thereby enhancing your understanding of this critical challenge in medical imaging.
- It also gives insight into efficient-size modelling compared to some pretrained models like ResNet or VGG.

## Design Approach
- The use of augmentation exposes the CNN model to variations in the data, especially for the minority class. Oversampling, weighted sampling, and focal loss effectively address the imbalance. 

- Modelling of the CNN custom architectures considers factors such as depth, kernel sizes, dropout, batch normalisation, and activation functions suited for image classification tasks. 

- Model training utilises a robust Stratified KFold, a method known for its effectiveness in addressing class imbalance, as well as optimisers, learning rate scheduling, and early stopping. These strategies ensure faster convergence and improved performance.. 

- Recall and Precision metrics, which are suited for imbalanced datasets, evaluate the model's performance.

## Deployment
- https://pneumonia-detection-classifier.streamlit.app/

## What You Can Do With It
- The project, deploy on Streamlit, enable users to upload X-ray medical images (in JPEG, JPG, or PNG format) online, allowing the application to detect whether the image is normal or infected with pneumonia within the margin of error 

## Acknowledgements
- This project, while inspired by SuperDataScience's work, is an independent undertaking completed with permission. 
