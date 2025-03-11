from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image
from google.colab import files

model_path = "/content/drive/MyDrive/my project/Project-ImageClassificationUsingCIFAR-10-main/classify.keras"


model = load_model(model_path)
print("✅ Model Loaded Successfully!")


uploaded = files.upload()

# Image Preprocess Karo aur Predict Karo
for filename in uploaded.keys():
    image = Image.open(filename)


    image = image.convert("RGB")

    #  Automatic Resize (Koi bhi size ho, 32x32 me convert ho jayegi)
    image = image.resize((32, 32))

    # Convert to NumPy Array aur Normalize Karo
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 32, 32, 3)

    # Class Names (CIFAR-10 Dataset ke liye)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Prediction Lo
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    print(f"🎯 Predicted Class: {predicted_class}")
