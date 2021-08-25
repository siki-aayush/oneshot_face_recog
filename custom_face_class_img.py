import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model


def predict(face1, face2):
    pass


def extract_faces(root_img_path, destination_img):
    face_cascade = cv2.CascadeClassifier(
        '/home/siki/.virtualenvs/datascience_env/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    os.mkdir(destination_img)
    for i in os.listdir(root_img_path):
        img = cv2.imread(os.path.join(root_img_path, i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0), 2)
            roi_color = img[y:y+h, x:x+w]
            cv2.imwrite(
                f'{os.path.join(destination_img, str(time.time()*1000))}.png', roi_color)


model = load_model('./model.h5')
#img_path1 = input('enter the path of the image1: ')
#img_path2 = input('enter the path of the image2: ')
#rgb = bool(input('process image in rgb?(True/False): '))

rgb_img1 = cv2.imread('./dataset/train/dad/1606054791120.png', 1)
rgb_img2 = cv2.imread('./dataset/train/siki/1606388010165.png', 1)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2RGB))

gray_img1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2GRAY)

plt.show()
face_cascade = cv2.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_default.xml')
face1 = face_cascade.detectMultiScale(gray_img1, 1.1, 5)
face2 = face_cascade.detectMultiScale(gray_img2, 1.1, 5)

# model.predict(image_pred)
