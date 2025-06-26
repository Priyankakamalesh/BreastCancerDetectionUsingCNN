import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the Breast Cancer Wisconsin dataset (you may need to download it)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["id", "diagnosis", "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se", "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"]
data = pd.read_csv(url, names=column_names)

# Data preprocessing
data = data.drop("id", axis=1)  # Remove the ID column
X = data.iloc[:, 1:]  # Features
y = (data["diagnosis"] == 'M').astype(int)  # Convert 'M' to 1 (malignant) and 'B' to 0 (benign)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for CNN
X_train = X_train.reshape(-1, 6, 5, 1)  # Adjust the shape according to your dataset
X_test = X_test.reshape(-1, 6, 5, 1)

# Build a CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6, 5, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Load and preprocess an input image
input_image_path = "/content/WhatsApp Image 2023-10-24 at 9.39.52 PM.jpeg"
input_image = Image.open(input_image_path).convert("L")
input_image = input_image.resize((5, 6))
input_image = np.array(input_image)
input_image = (input_image - input_image.mean()) / input_image.std()
input_image = input_image.reshape(1, 6, 5, 1)

# Make predictions on the input image
prediction = model.predict(input_image)

if prediction[0][0] > 0.5:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")  


  
