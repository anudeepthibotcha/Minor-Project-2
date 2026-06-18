The **Pneumonia Detection and Classification System** is a deep learning-based medical image analysis project developed to automatically identify and classify chest X-ray images into **Normal, Bacterial Pneumonia, and Viral Pneumonia** categories. 
The system uses a curated chest X-ray dataset that undergoes preprocessing techniques such as image resizing, normalization, and data augmentation to improve model performance and generalization. 
A pre-trained **MobileNetV2** Convolutional Neural Network (CNN) is employed as the feature extraction backbone, while additional dense layers perform the final classification. 
Transfer learning is utilized by freezing the MobileNetV2 layers, reducing training time and improving accuracy. 
The model is trained and validated using labeled X-ray images and evaluated through accuracy and loss metrics. 
Early stopping, model checkpointing, and learning rate reduction techniques are applied to prevent overfitting and optimize performance. 
After training, the system allows users to upload external chest X-ray images, which are processed by the trained model to generate predictions along with confidence scores. 
This project provides faster and more accurate pneumonia detection, reduces manual diagnostic effort, and supports healthcare professionals in clinical decision-making, demonstrating the effectiveness of artificial intelligence in medical image classification.
