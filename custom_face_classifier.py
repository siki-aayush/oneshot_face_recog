import cv2
import os
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model


model = load_model('./model.h5')


def normalize_data(img, by):
    return np.array(img)/by


def compare_imgs(cap_img, img_dir, rgb=True):
    persons = os.listdir(img_dir)
    confidences = []
    for person in persons:
        faces = os.listdir(os.path.join(img_dir, person))
        confidence = 0
        for i in range(5):
            choosen_face_path = os.path.join(
                img_dir, person, np.random.choice(faces))
            choosen_face_img = normalize_data(load_img(
                choosen_face_path,
                color_mode='grayscale' if not rgb else 'rgb',
                target_size=(64, 64)
            ), 255)
            reshape = (-1, 64, 64, 1) if not rgb else (-1, 64, 64, 3)
            confidence += model.predict([cap_img.reshape(reshape),
                                         choosen_face_img.reshape(reshape)])[0][0]
        confidence = np.round(confidence/5, 4)
        confidences.append(confidence)

    if np.max(confidences) > 0.5:
        print(f'testing\n{persons}\n{confidences}')
        return (persons[np.argmax(confidences)], str(np.max(confidences)))
    else:
        return ('unidentified', str(0))


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex+ew, ey+eh), (0, 255, 0), 2)

        processed_img = cv2.resize(roi_gray, (64, 64))/255
        output = compare_imgs(processed_img, './dataset/train', False)
        print('output', output)
        frame = cv2.putText(frame, ' '.join(output), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
