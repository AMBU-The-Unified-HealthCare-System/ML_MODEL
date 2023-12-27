import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
import tensorflow as tf
import pytesseract
from keras import layers, models
import cv2


app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = r'D:\collegeproject\medicinemodel\MedicineClassification\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd = r'D:\collegeproject\ocr\tesseract.exe'
dataset_path = r"D:\collegeproject\medicinedataset\directory\train_data_directory"

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
    epochs=40,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.reshape(img, (1, 128, 128, 3))
    img = img / 255.0  # Normalize the image
    return img

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return render_template('index1.html', image_path=None, result=None)

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform medicine prediction
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_name = list(train_generator.class_indices.keys())[class_index]

        return render_template('index1.html', result=class_name, image_path=file_path)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
