# CIFAR10-Image-Classification
# CNN Image Classification Using CIFAR-10

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow & Keras** to classify images from the **CIFAR-10 dataset**. The model achieves **84% test accuracy** and **99% prediction accuracy**.

## 📂 Dataset Details
- **Dataset:** CIFAR-10 (Preloaded in TensorFlow)
- **Images:** 60,000 (50,000 for training, 10,000 for testing)
- **Classes:** 10 (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`)
- **Image Resolution:** 32x32 pixels, RGB

## ⚙️ Dependencies & Installation
Install the required dependencies using the following command:
```bash
pip install -r requare.txt
```
Or manually install:
```bash
pip install tensorflow keras numpy matplotlib seaborn pandas scikit-learn scipy Pillow
```

## 🚀 Training the Model
To train the CNN model on the CIFAR-10 dataset, run the following script:
```bash
python train_cifar10_cnn.py
```
### 📜 **train_cifar10_cnn.py (Key Features)**
✅ Uses **Data Augmentation** to prevent overfitting.  
✅ Consists of **3 Convolutional Layers** with **Batch Normalization & Max Pooling**.  
✅ Includes **Dropout Layer (0.5)** for better generalization.  
✅ Trains the model for **25 epochs** using the **Adam optimizer**.  
✅ Saves the model as `classify.keras`.  

## 🎯 Checking Model Accuracy
To check the test accuracy of the trained model, run:
```bash
python -c "import tensorflow as tf; from tensorflow.keras.datasets import cifar10; from tensorflow.keras.utils import to_categorical;
model = tf.keras.models.load_model('classify.keras');
(_, _), (x_test, y_test) = cifar10.load_data();
x_test = x_test.astype('float32') / 255.0;
y_test = to_categorical(y_test, 10);
loss, accuracy = model.evaluate(x_test, y_test, verbose=1);
print(f'Test Accuracy: {accuracy * 100:.2f}%')"
```
📌 **Expected Accuracy: 84%**

## 🖼️ Running Image Prediction
To classify a custom image, run:
```bash
python pro.py
```
It will prompt you to enter the image path. After processing, it will output the predicted class and confidence score.

### 📜 **pro.py (How It Works)**
✅ Loads the trained `classify.keras` model.  
✅ Accepts an image file as input.  
✅ Resizes the image to **32x32** and converts it to **RGB**.  
✅ Runs a prediction using the CNN model.  
✅ Outputs the **Predicted Class** & **Confidence Score**.  


## 📌 Conclusion
This CNN model successfully classifies CIFAR-10 images with **high accuracy**. Further improvements can include:
- Using **transfer learning** with pre-trained models (ResNet, VGG16, etc.)
- Fine-tuning hyperparameters to improve accuracy
- Deploying the model as a web application using Flask or Streamlit

🎯 **Project Completed! 🚀**


