import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Disable oneDNN log warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Path to your UTKFace dataset folder
DATA_DIR = r"C:\Users\rishi\Downloads\artproj\UTKFace"
IMG_SIZE = 64

# Containers for data
images = []
ages = []
genders = []

# Load and preprocess data
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".jpg"):
        try:
            age, gender = map(int, filename.split("_")[:2])
            img_path = os.path.join(DATA_DIR, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            ages.append(age)
            genders.append(gender)
        except:
            continue

# Convert to NumPy arrays
images = np.array(images) / 255.0
ages = np.array(ages)
genders = to_categorical(genders, num_classes=2)

# Split dataset
X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
    images, ages, genders, test_size=0.2
)

# Define dual-output CNN model
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

gender_output = Dense(2, activation='softmax', name='gender')(x)
age_output = Dense(1, activation='linear', name='age')(x)

model = Model(inputs=input_layer, outputs=[gender_output, age_output])

# Compile model
model.compile(
    optimizer='adam',
    loss={'gender': 'categorical_crossentropy', 'age': 'mse'},
    metrics={'gender': 'accuracy', 'age': 'mae'}
)

# Train model
model.fit(
    X_train,
    {'gender': gender_train, 'age': age_train},
    validation_data=(X_test, {'gender': gender_test, 'age': age_test}),
    epochs=10,
    batch_size=32
)

# Save model
model.save("age_gender_model.h5")
print("Model saved as 'age_gender_model.h5'")

