from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image
import os

# ✅ Model Load Karo (Apne actual model path ka dhyan do)
model_path = os.path.join(os.getcwd(), "classify.keras")
model = load_model(model_path)
print("✅ Model Loaded Successfully!")

# ✅ Image Upload Karne Ka Option (Manually File Path Do)
image_path = input("📂 Enter image path: ")  # User se image ka path lo

# ✅ Image Preprocessing
image = Image.open(image_path)
image = image.convert("RGB")  # RGBA Fix
image = image.resize((32, 32))

img_array = img_to_array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ✅ Prediction Lo
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions[0])]
confidence_score = np.max(predictions[0]) * 100

# ✅ Prediction Output
print(f"🎯 Predicted Class: {predicted_class} ({confidence_score:.2f}
