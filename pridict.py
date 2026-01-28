import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = 28
MODEL_PATH = "dnn_digits_1_to_9.h5"  


classes = list("123456789")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("❌ Image not found or invalid path")

   
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Invert image (white bg → black bg)
    img = 255 - img

   
    img = img / 255.0
   
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return img


def predict_digit(image_path):
    img = preprocess_image(image_path)

    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    predicted_digit = classes[class_index]

    print("Predicted Digit:", predicted_digit)
    print(f"Confidence: {confidence * 100:.2f}%")

    return predicted_digit, confidence

if __name__ == "__main__":
    image_path = "test_digit.png"   #  change path
    predict_digit(image_path)

