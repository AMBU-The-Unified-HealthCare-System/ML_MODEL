import os
import cv2
import pytesseract
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'D:\collegeproject\ocr\tesseract.exe'

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error loading image: {image_path}")
        return ""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh)
    return text.strip()


data_dir = r"D:\collegeproject\medicinedataset\directory\train_data_directory"

# Load data
X, y = [], []

for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                image_path = os.path.join(folder_path, filename)
                label = folder_name  # Folder name is the label
                
                try:
                    text = extract_text_from_image(image_path)
                    X.append(text)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing image: {image_path}")
                    print(e)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

for i in range(5):  # Display predictions for the first 5 test samples
    print(f"Image: {X_test[i]}\nPredicted Medicine: {y_pred[i]}\n")