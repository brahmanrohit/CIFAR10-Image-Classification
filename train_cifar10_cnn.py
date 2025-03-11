import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ CIFAR-10 dataset load karo
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# ✅ Data Normalize Karo
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ✅ Data Augmentation (Model Overfit Na Ho)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# ✅ Improved CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # ✅ Dropout Overfitting Avoid Karega
    layers.Dense(10, activation='softmax')
])

# ✅ Model Compile Karo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Model Train Karo (Data Augmentation Ke Sath)
history = model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=25, validation_data=(X_test, y_test))

# ✅ Model Save Karo
model.save("/content/drive/MyDrive/my project/Project-ImageClassificationUsingCIFAR-10-main/classify.keras")
print("✅ Model Training Complete & Saved Successfully!")
