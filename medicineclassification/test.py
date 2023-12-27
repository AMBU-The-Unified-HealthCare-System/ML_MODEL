import os
import tensorflow as tf
from keras import layers, models
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\collegeproject\ocr\tesseract.exe'
dataset_path = r"D:\collegeproject\medicinedataset\directory\train_data_directory"

# Define the model
num_classes = len(os.listdir(dataset_path))
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess the dataset
image_size = (128, 128)
batch_size = 21

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Assuming you have separate directories for train, validation, and test datasets
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, r"D:\collegeproject\medicinedataset\directory\train_data_directory"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, r"D:\collegeproject\medicinedataset\directory\train_data_directory"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path,r"D:\collegeproject\medicinedataset\directory\train_data_directory"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save("medicine_classifier_model.h5")

def perform_text_recognition(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Perform text recognition using Tesseract
    text = pytesseract.image_to_string(img)

    return text

# Function to predict medicine name from an input image
def predict_medicine(image_path):
    # Image recognition
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = np.reshape(img, (1, image_size[0], image_size[1], 3))
    img = img / 255.0  # Normalize the image

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_name = list(train_generator.class_indices.keys())[class_index]

    return class_name

# Example usage on test data
test_image_path = r"D:\collegeproject\medicinemodel\2.jpeg"
predicted_medicine = predict_medicine(test_image_path)
text_result = perform_text_recognition(test_image_path)

print("Predicted Medicine:", predicted_medicine)
print("Text Recognition Result:", text_result)