import os
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.metrics import MeanAbsoluteError, Accuracy
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

# --- Settings ---
DATA_DIR = r"C:\Users\rishi\Downloads\artproj\UTKFace"  # <-- UPDATE this if needed
IMG_SIZE = 64
EPOCHS = 15
BATCH_SIZE = 64
MODEL_PATH = "age_gender_model.h5"

# --- Load and preprocess dataset ---
def load_utkface_dataset(data_dir, img_size=64):
    images = []
    ages = []
    genders = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            try:
                parts = filename.split('_')
                age = int(parts[0])
                gender = int(parts[1])
                img_path = os.path.join(data_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                ages.append(age)
                genders.append(gender)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

    images = np.array(images, dtype=np.float32) / 255.0
    ages = np.array(ages, dtype=np.float32)
    genders = to_categorical(genders, num_classes=2)

    return images, ages, genders

# --- Build model ---
def build_age_gender_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    gender_output = Dense(2, activation='softmax', name='gender_output')(x)
    age_output = Dense(1, activation='linear', name='age_output')(x)

    model = Model(inputs=inputs, outputs=[gender_output, age_output])
    
    model.compile(
        optimizer='adam',
        loss={'gender_output': 'categorical_crossentropy', 'age_output': MeanSquaredError()},
        metrics={'gender_output': 'accuracy', 'age_output': MeanAbsoluteError()}
    )



    return model

# --- Webcam inference ---
def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def run_webcam_inference(model):
    gender_labels = ['Male', 'Female']
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_img = preprocess_frame(frame)
        gender_pred, age_pred = model.predict(face_img)

        gender = gender_labels[np.argmax(gender_pred[0])]
        age = int(age_pred[0][0])

        cv2.putText(frame, f"Gender: {gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Age: {age}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Age and Gender Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Main ---
def main():
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model = load_model(MODEL_PATH)
    else:
        print("Loading dataset...")
        images, ages, genders = load_utkface_dataset(DATA_DIR, IMG_SIZE)
        X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
            images, ages, genders, test_size=0.2, random_state=42
        )

        print("Building model...")
        model = build_age_gender_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

        print("Training model...")
        model.fit(
            X_train,
            {'gender_output': y_gender_train, 'age_output': y_age_train},
            validation_data=(X_test, {'gender_output': y_gender_test, 'age_output': y_age_test}),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        print(f"Saving model to {MODEL_PATH}...")
        model.save(MODEL_PATH)

    run_webcam_inference(model)

if __name__ == "__main__":
    main()
