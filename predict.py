from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = load_model('models/final_model.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(model, img_path, threshold=0.5):
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)
    return prediction > threshold

# Path to the new image
img_path = input("Enter the Image path: ")
prediction = predict_image(model, img_path, threshold=0.5)

if prediction[0][0]:
    print("Prediction: Negative (Clear Skin)")
    percentage = (1 - prediction[0][0]) * 100
    print(f"Expected chance of cancer: {percentage:.2f}%")
else:
    print("Prediction: Positive (Cancer Detected)")
    percentage = (1 - prediction[0][0]) * 100
    print(f"Expected chance of cancer: {percentage:.2f}%")

# Evaluate the model on a validation set
validation_images = [...]  # List of validation image paths
validation_labels = [...]  # List of validation labels (0 for non-cancer, 1 for cancer)

predictions = []
for img_path in validation_images:
    prediction = predict_image(model, img_path, threshold=0.5)
    predictions.append(prediction[0][0])

print(classification_report(validation_labels, predictions))
print(confusion_matrix(validation_labels, predictions))
