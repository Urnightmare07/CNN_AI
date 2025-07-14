import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64
MODEL_PATH = "age_gender_model.h5"

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def run_webcam_inference():
    model = load_model(MODEL_PATH)
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

        img = preprocess_frame(frame)
        gender_pred, age_pred = model.predict(img)
        gender = gender_labels[np.argmax(gender_pred)]
        age = int(age_pred[0][0])

        cv2.putText(frame, f"Gender: {gender}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Age: {age}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Age and Gender Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_inference()